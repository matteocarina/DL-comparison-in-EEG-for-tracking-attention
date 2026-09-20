"""Session access for the SADT dataset.

Recordings ship as one .set.zip per session, about 15 GB in total. A session is unzipped on demand
and removed again once processed, so peak disk use stays at roughly one recording."""

import contextlib
import glob
import os
import shutil
import zipfile

def list_sessions(root):
    """Session bundle names, available either extracted or as a .set.zip archive.

    Names keep the '.set' suffix exactly as the bundle is named on disk (e.g. 's01_051017m.set'),
    which is the convention the feature files already use ('<bundle>.pt').
    """
    names = {os.path.basename(d) for d in glob.glob(os.path.join(root, '*.set'))
             if os.path.isdir(d)}
    names |= {os.path.basename(z)[:-4] for z in glob.glob(os.path.join(root, '*.set.zip'))}
    return sorted(names)

def _is_complete(bundle):
    """True if every file in the bundle has its bytes really on disk.

    Cloud-evicted files report their full logical size but zero allocated blocks; opening one
    fails with [Errno 60] or MNE's "Incorrect number of samples (0 != N)", so they must not be
    mistaken for a usable bundle.
    """
    files = [f for f in glob.glob(os.path.join(bundle, '*'))
             if not os.path.basename(f).startswith('.')]
    if not files:
        return False
    return all(os.stat(f).st_blocks * 512 >= os.stat(f).st_size * 0.9 for f in files)

@contextlib.contextmanager
def session_bundle(name, root, workdir=None):
    """Yield the path of the inner .set file MNE opens, for a bundle name like 's01_051017m.set'.

    Uses an already-extracted bundle untouched; otherwise expands the archive under `workdir`
    and deletes it on exit. MNE needs the large .fdt alongside the .set even with preload=False,
    so the whole archive is expanded, not just the header.
    """
    bundle = os.path.join(root, name)
    if _is_complete(bundle):
        yield os.path.join(bundle, name)
        return

    zip_path = os.path.join(root, name + '.zip')
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f'{name}: neither an extracted bundle nor {name}.zip')

    workdir = workdir or os.path.join(root, '_tmp_extract')
    dest = os.path.join(workdir, name)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        yield os.path.join(dest, name, name)
    finally:
        shutil.rmtree(dest, ignore_errors=True)
