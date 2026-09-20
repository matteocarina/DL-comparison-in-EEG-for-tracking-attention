"""Epoch the SADT recordings and extract the per-epoch inputs both models read.
Applies a 0.5 Hz high-pass and a common-average reference over the 10 deployable electrodes, cuts
4 s epochs every 3.5 s, and writes one .pt per session holding the engineered features, the theta
and alpha wPLI matrices, and the decimated raw waveform."""

import os
import gc
import warnings
import numpy as np
from lib import sadt_io
import torch
import mne
from mne_connectivity import spectral_connectivity_time
from scipy.signal import freqz
import antropy as ant

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

FOLDER_PATH = os.environ.get('SADT_DATA', os.path.expanduser('~/PycharmProjects/7666055'))
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(HERE, 'eeg_data_local', 'extracted_features')
os.makedirs(FEATURE_DIR, exist_ok=True)

DEBUG_CROP = False # Set to True to process only the first 5 minutes (fast iteration / testing).
DEBUG_CROP_SECS = 5 * 60
EPOCH_LENGTH = 4.0
OVERLAP = 0.5 # MNE overlap in seconds -> 3.5 s spacing
HIGHPASS_HZ = 0.5
ARTIFACT_P2P_UV = 150.0 # epochs whose worst channel exceeds this peak-to-peak (uV) are flagged as
# artifacts (blink/movement).
CALIBRATION_SECS = 300.0 # normalise using only the first 5 minutes, then freeze (see compute_norm_stats)

SAVE_RAW_EPOCHS = True # store the downsampled raw waveform for DATA_MODE='raw'
RESAMPLE_HZ = 250 # Raw data stored at this rate. No anti-alias filter is applied before the
# decimation because the recordings arrive already band-limited from acquisition: measured on this
# dataset, power above 60 Hz is 3e-8 of the power below it, far under the 125 Hz Nyquist.
RAW_SAMPLES = int(EPOCH_LENGTH * RESAMPLE_HZ)
RAW_UNIT_SCALE = 1e6 # volts -> microvolts before the float16 cast. EEG in volts sits around 2.7e-6,
# so storing volts puts ~99% of samples in the subnormal range where the spacing is a fixed 5.96e-8
THETA_BAND, ALPHA_BAND, BETA_BAND, TOTAL_BAND = (4, 7), (8, 12), (12, 30), (1, 40)
FRONTAL_CH = ['f3', 'f4', 'fz', 'fcz']
PARIETAL_CH = ['pz', 'p3', 'p4']
OCCIPITAL_CH = ['o1', 'oz', 'o2']
ALL_INTEREST = FRONTAL_CH + PARIETAL_CH + OCCIPITAL_CH
AR_ORDER = 20 # order of the Yule-Walker autoregressive PSD used for band power

def find_channels(ch_names, targets): # Find indices of target channels in ch_names (case-insensitive).
    lower = [c.lower() for c in ch_names]
    return [lower.index(t.lower()) for t in targets if t.lower() in lower]

def ar_band_power(signal_1d, sfreq, band, order=AR_ORDER, nfft=512):
    # Band power via Yule-Walker AR PSD (numerically stable). Returns mean PSD in band per channel.
    try:
        x = signal_1d.astype(np.float64) - signal_1d.mean()
        n = len(x)
        r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(order + 1)])
        R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
        ar = np.linalg.solve(R, -r[1:order + 1])
        var = max(r[0] + np.dot(ar, r[1:order + 1]), 1e-10)
        a_poly = np.concatenate([[1.0], ar])
        w, h = freqz([1.0], a_poly, worN=nfft, whole=False)
        freqs = w * sfreq / (2.0 * np.pi)
        psd = var * (np.abs(h) ** 2)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return float(np.mean(psd[mask])) if mask.sum() > 0 else 0.0
    except Exception:
        return 0.0

def compute_wpli_per_epoch(epochs_data, sfreq, band):
    # Per-epoch within-epoch wPLI via Morlet time-frequency.
    n_ep, n_ch = epochs_data.shape[0], epochs_data.shape[1]
    fmin = band[0]
    fmax = band[1]
    if n_ch < 2 or fmin >= fmax:
        return np.zeros((n_ep, n_ch, n_ch), dtype=np.float32)
    freqs = np.arange(fmin, fmax + 1e-9, 1.0)
    if len(freqs) < 2:
        freqs = np.linspace(fmin, fmax, 3)
    n_cycles = freqs / 2.0
    rows, cols = np.tril_indices(n_ch, k=-1)
    con = spectral_connectivity_time(
        epochs_data, freqs=freqs, method='wpli', sfreq=sfreq,
        indices=(rows.tolist(), cols.tolist()), mode='cwt_morlet', n_cycles=n_cycles,
        fmin=fmin, fmax=fmax, faverage=True, average=False, verbose=False)
    vals = np.asarray(con.get_data())
    if vals.ndim == 3:
        vals = vals[:, :, 0]
    mats = np.zeros((n_ep, n_ch, n_ch), dtype=np.float32)
    mats[:, rows, cols] = vals
    mats[:, cols, rows] = vals
    return mats

def compute_entropy_measures(signal, sfreq):
    # Signal is z-scored first to be amplitude-independent. Then we extract all 4 entropy families
    # (sample, permutation, svd, spectral) here so the on-disk dataset stays flexible.
    try:
        s = signal.astype(np.float64)
        s = (s - s.mean()) / (s.std() + 1e-10)
        se = ant.sample_entropy(s, order=2)
        pe = ant.perm_entropy(s, order=3, normalize=True)
        sve = ant.svd_entropy(s, order=3, normalize=True)
        spe = ant.spectral_entropy(s, sf=sfreq, method='welch', normalize=True)
        result = np.array([se, pe, sve, spe])
        return np.where(np.isfinite(result), result, 0.0)
    except Exception:
        return np.zeros(4)

def compute_norm_stats(features, keep=None, calib_epochs=None):
    # Per-recording z-score statistics: Causal, and computed on clean epochs only. Artifacts:
    # including them standardises the recording partly to its own blinks. Causality: the statistics
    # come from a calibration window at the start of the recording and are then frozen and applied
    # forward, which is exactly what a real monitor would do.
    n = features.shape[0]
    end = n if calib_epochs is None else min(calib_epochs, n)
    X = features[:end] if keep is None else features[:end][keep[:end]]
    if X.shape[0] == 0: # every epoch in the window was flagged
        X = features[:end]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-10] = 1.0
    return mean, std

def extract_features_for_epoch(epoch_data, sfreq, ch_names):
    # Extract node features for one epoch.
    features, feature_names = [], []
    frontal_idx = find_channels(ch_names, FRONTAL_CH)
    parietal_idx = find_channels(ch_names, PARIETAL_CH)
    occipital_idx = find_channels(ch_names, OCCIPITAL_CH)
    # The raw EEG signal in this dataset is stored in volts (not microvolts), so instantaneous power
    # values are on the order of 1e-12 to 1e-7. These values are so small in absolute terms that their
    # epoch-to-epoch variance also falls in the range of 1e-14 to 1e-12, below the floor in
    # compute_norm_stats, which causes z-scoring to collapse all values to zero. log10 solves this by
    # transforming the scale: now epoch-to-epoch variation is on the order of 0.1 to 1.0 log units, which
    # z-scores cleanly. Does this destroy the features? No, for two reasons:
    #   1. Ratios are preserved. log10(a/b) = log10(a) - log10(b), so TBR and ATR computed as log
    #   differences are mathematically identical to the log of the original ratio.
    #   2. Power in EEG follows a 1/f (log-normal) distribution — taking log10 actually makes the
    #   distribution more Gaussian, which is the assumption underlying z-score normalisation.
    # A. TBR frontal
    for ch_i in frontal_idx:
        th = ar_band_power(epoch_data[ch_i], sfreq, THETA_BAND)
        be = ar_band_power(epoch_data[ch_i], sfreq, BETA_BAND)
        features.append(np.log10(th + 1e-30) - np.log10(be + 1e-30))
        feature_names.append(f'TBR_{ch_names[ch_i]}')
    # B. FMT (frontal-midline theta)
    fz_fcz = find_channels(ch_names, ['fz', 'fcz'])
    if fz_fcz:
        vals = [ar_band_power(epoch_data[i], sfreq, THETA_BAND) for i in fz_fcz]
        features.append(np.log10(np.mean(vals) + 1e-30))
        feature_names.append('FMT_mean')
    # C. ATR parietal
    for ch_i in parietal_idx:
        al = ar_band_power(epoch_data[ch_i], sfreq, ALPHA_BAND)
        th = ar_band_power(epoch_data[ch_i], sfreq, THETA_BAND)
        features.append(float(np.log10(al + 1e-30) - np.log10(th + 1e-30)))
        feature_names.append(f'ATR_{ch_names[ch_i]}')
    # D. Posterior theta
    for ch_i in occipital_idx:
        th = ar_band_power(epoch_data[ch_i], sfreq, THETA_BAND)
        features.append(np.log10(th + 1e-30))
        feature_names.append(f'theta_{ch_names[ch_i]}')
    # E. Entropy (all 4 families x 10 channels)
    for ch_i in (frontal_idx + parietal_idx + occipital_idx):
        ent = compute_entropy_measures(epoch_data[ch_i], sfreq)
        features.extend(ent.tolist())
        cn = ch_names[ch_i]
        feature_names += [f'samp_ent_{cn}', f'perm_ent_{cn}', f'svd_ent_{cn}', f'spec_ent_{cn}']
    return np.array(features, dtype=np.float32), feature_names


# Main loop
sessions = sadt_io.list_sessions(FOLDER_PATH)
print(f"Found {len(sessions)} sessions.  Raw epochs: {'ON' if SAVE_RAW_EPOCHS else 'OFF'} "
      f"({RESAMPLE_HZ} Hz, {RAW_SAMPLES} samples)")

n_done = 0
failed = []
for session_name in sessions:
    out_path = os.path.join(FEATURE_DIR, f'{session_name}.pt')
    if os.path.exists(out_path):
        print(f"  [skip] {session_name} (exists)")
        continue
    print(f"\n{'=' * 60}\nProcessing: {session_name}")
    try:
        # Unzips the session if it is not already extracted and removes it afterwards, so the
        # whole dataset never has to be expanded at once (see sadt_io).
        with sadt_io.session_bundle(session_name, FOLDER_PATH) as set_file:
            raw = mne.io.read_raw_eeglab(set_file, preload=True)
            sfreq = raw.info['sfreq']
            ch_names = raw.info['ch_names']
            if DEBUG_CROP:
                raw.crop(tmin=0, tmax=min(DEBUG_CROP_SECS, raw.times[-1]))

            # preprocessing: 0.5 Hz hp -> common average reference over the deployable electrodes so the 10
            # electrodes the target headset actually carries, not of all 30.
            raw.filter(l_freq=HIGHPASS_HZ, h_freq=None, fir_design='firwin', verbose=False)
            ref_channels = [raw.ch_names[i] for i in find_channels(raw.ch_names, ALL_INTEREST)]
            if len(ref_channels) < len(ALL_INTEREST):
                raise RuntimeError(f"only {len(ref_channels)}/{len(ALL_INTEREST)} reference electrodes "
                                   f"present in {session_name}: {ref_channels}")
            raw.set_eeg_reference(ref_channels=ref_channels, projection=False, verbose=False)

            # 4 s epochs every 3.5 s (overlap in seconds)
            stop = raw.times[-1] - EPOCH_LENGTH + 1.0 / sfreq
            events = mne.make_fixed_length_events(raw, start=0, duration=EPOCH_LENGTH, stop=stop, overlap=OVERLAP)
            epochs = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_LENGTH, preload=True, baseline=None, verbose=False)
            n_ep = len(epochs)
            print(f"  Epochs: {n_ep}")

            # Artifact flag: per-epoch worst-channel peak-to-peak over the 10 electrodes we use.
            # Only those 10 are pulled out: nothing downstream of here needs the other 20, and asking
            # for all 30 tripled the size of the single largest array in the pipeline.
            interest_idx = find_channels(ch_names, ALL_INTEREST)
            ep_data = epochs.get_data(picks=interest_idx)  # (n_ep, 10, n_times) volts
            ep_p2p = ep_data.max(axis=2) - ep_data.min(axis=2)
            artifact_mask = (ep_p2p.max(axis=1) > ARTIFACT_P2P_UV * 1e-6)
            print(f"  Artifact epochs: {int(artifact_mask.sum())}/{n_ep} "
                  f"({100 * artifact_mask.mean():.1f}%, scored on all "
                  f"{len(interest_idx)} deployable electrodes)")

            # raw epochs by decimation of the epochs we already have (memory fix). Striding is safe
            # here only because the recordings were low-passed at acquisition
            raw_epochs_f16, decim = None, max(1, int(round(sfreq / RESAMPLE_HZ)))
            if SAVE_RAW_EPOCHS:
                raw_epochs_f16 = (ep_data[:, :, ::decim][:, :, :RAW_SAMPLES]
                                  * RAW_UNIT_SCALE).astype(np.float16)
                print(f"  Raw epochs: {raw_epochs_f16.shape} fp16 in uV "
                      f"({len(interest_idx)} electrodes, decim={decim} -> {sfreq / decim:.0f} Hz)")

            # wPLI connectivity: one 10x10 matrix per epoch per band (45 unique pairs each).
            # Both models use it: the TCN can select individual pairs as flat features, the GNN-LSTM
            # consumes the full matrix as graph edge weights. 
            wpli_theta = compute_wpli_per_epoch(ep_data, sfreq, THETA_BAND)
            wpli_alpha = compute_wpli_per_epoch(ep_data, sfreq, ALPHA_BAND)

            # Free the large arrays before the feature loop: MNE's epochs object holds all 30
            # channels and the feature loop allocates on top of it, so nothing that is already
            # finished with should still be alive here.
            del ep_data, ep_p2p, raw
            gc.collect()

            # engineered features (on full 500 Hz)
            all_features, feature_names = [], None
            for epoch in epochs:
                feats, names = extract_features_for_epoch(epoch, sfreq, ch_names)
                all_features.append(feats)
                if feature_names is None:
                    feature_names = names
            all_features = np.array(all_features)
            calib_epochs = min(max(1, int(round(CALIBRATION_SECS / (EPOCH_LENGTH - OVERLAP)))),
                               all_features.shape[0])
            norm_mean, norm_std = compute_norm_stats(all_features, keep=~artifact_mask,
                                                     calib_epochs=calib_epochs)
            print(f"  Normalisation: first {CALIBRATION_SECS / 60:.0f} min "
                  f"({calib_epochs} epochs), frozen and applied forward")
            all_features_norm = (all_features - norm_mean) / norm_std
            print(f"  Feature matrix: {all_features.shape}")

            out = {
                'features_raw': torch.from_numpy(all_features).float(),
                'features_norm': torch.from_numpy(all_features_norm).float(),
                'feature_names': feature_names,
                'norm_mean': torch.from_numpy(norm_mean).float(),
                'norm_std': torch.from_numpy(norm_std).float(),
                'norm_calib_epochs': int(calib_epochs),
                'wpli_theta': torch.from_numpy(wpli_theta).float(),
                'wpli_alpha': torch.from_numpy(wpli_alpha).float(),
                'artifact_mask': torch.from_numpy(artifact_mask),
                'graph_ch_names': [ch_names[i] for i in interest_idx],
                'sfreq': sfreq,
                'ch_names': ch_names,
                'bands': {'theta': THETA_BAND, 'alpha': ALPHA_BAND, 'beta': BETA_BAND},
            }

            # raw epochs (already computed above by decimation, before ep_data was freed)
            if SAVE_RAW_EPOCHS and raw_epochs_f16 is not None:
                out['raw_epochs'] = torch.from_numpy(raw_epochs_f16)        # (n_ep, n_ch, RAW_SAMPLES)
                out['raw_ch_names'] = [ch_names[i] for i in interest_idx]   # the 10 headset electrodes
                out['raw_sfreq'] = float(sfreq / decim)
                out['raw_unit'] = 'uV'  # stated, not implied: features_raw is in volts, this is not

            torch.save(out, out_path)
            print(f"  Saved -> {out_path}")
            n_done += 1

    except Exception as e:
        # One unreadable recording must not end a 90-minute run; it is recorded and reported at
        # the end, and re-running the script retries only the sessions that have no .pt yet.
        print(f"  failed: {e}")
        failed.append((session_name, str(e)))

    finally:
        # Free every large object before the next session.
        for _name in ('raw', 'epochs', 'ep_data', 'ep_p2p', 'all_features', 'all_features_norm',
                      'wpli_theta', 'wpli_alpha', 'raw_epochs_f16', 'out'):
            globals().pop(_name, None)
        gc.collect()

print(f"\nDone. {n_done} sessions written to {FEATURE_DIR}.")
for s, e in failed:
    print(f"  failed: {s}: {e[:110]}")
