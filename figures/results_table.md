| Model | Input | Labels | Protocol | Pooled r | Per-session r | MAE | Params |
|---|---|---|---|---|---|---|---|
| MLP-TCN | 15 selected feats | event-locked | within-subject | 0.393 | 0.269 | 0.190 | 116,801 |
| MLP-TCN | 15 selected feats | event-locked | cross-subject | 0.350 | 0.275 | 0.200 | 116,801 |
| MLP-TCN | all 51 feats | event-locked | within-subject | 0.280 | 0.244 | 0.210 | 121,409 |
| MLP-TCN | all 51 feats | event-locked | cross-subject | 0.305 | 0.247 | 0.213 | 121,409 |
| CNN-TCN | raw 10x1000 | event-locked | within-subject | 0.456 | 0.299 | 0.191 | 156,001 |
| CNN-TCN | raw 10x1000 | event-locked | cross-subject | 0.342 | 0.284 | 0.214 | 156,001 |
| GNN | wPLI graph, one epoch | event-locked | within-subject | 0.274 | 0.191 | 0.198 | — |
| GNN | wPLI graph, one epoch | event-locked | cross-subject | 0.265 | 0.208 | 0.198 | — |
| GNN-LSTM | feats + wPLI graph | event-locked | within-subject | 0.431 | 0.299 | 0.188 | 373,538 |
| GNN-LSTM | feats + wPLI graph | event-locked | cross-subject | 0.381 | 0.278 | 0.189 | 373,538 |
| TIME | elapsed time (no EEG) | event-locked | within-subject | 0.118 | 0.089 | 0.203 | 2 |
| TIME | elapsed time (no EEG) | event-locked | cross-subject | 0.084 | 0.058 | 0.207 | 2 |
| MLP-TCN | 15 selected feats | smoothed | within-subject | 0.506 | 0.459 | 0.150 | 116,801 |
| MLP-TCN | 15 selected feats | smoothed | cross-subject | 0.387 | 0.435 | 0.164 | 116,801 |
| MLP-TCN | all 51 feats | smoothed | within-subject | 0.418 | 0.384 | 0.170 | 121,409 |
| MLP-TCN | all 51 feats | smoothed | cross-subject | 0.359 | 0.414 | 0.171 | 121,409 |
| CNN-TCN | raw 10x1000 | smoothed | within-subject | 0.481 | 0.396 | 0.165 | 156,001 |
| CNN-TCN | raw 10x1000 | smoothed | cross-subject | 0.212 | 0.312 | 0.202 | 156,001 |
| GNN | wPLI graph, one epoch | smoothed | within-subject | 0.293 | 0.219 | 0.162 | — |
| GNN | wPLI graph, one epoch | smoothed | cross-subject | 0.277 | 0.240 | 0.161 | — |
| GNN-LSTM | feats + wPLI graph | smoothed | within-subject | 0.479 | 0.396 | 0.149 | 373,538 |
| GNN-LSTM | feats + wPLI graph | smoothed | cross-subject | 0.462 | 0.444 | 0.153 | 373,538 |
| TIME | elapsed time (no EEG) | smoothed | within-subject | -0.034 | -0.075 | 0.169 | 2 |
| TIME | elapsed time (no EEG) | smoothed | cross-subject | -0.036 | -0.071 | 0.170 | 2 |

> **GNN-LSTM selects its checkpoint on the test score** (peak-on-test), while
> both TCN rows select on a held-out validation set and touch test exactly once.
> The GNN numbers are an optimistic upper bound (~+0.026 per-session r), not a
> like-for-like comparison.
