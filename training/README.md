# Training and evaluation

The repository includes a temporal BiGRU checkpoint trained on generated feature sequences as a bootstrap demonstration. It is not a validated real-driver model. This workflow describes how to replace it with a model trained and evaluated on suitable real data.

## Data requirements

Use consented, appropriately licensed data with labels and participant identifiers. Keep recordings and private derived data out of Git. The training script expects a CSV schema; inspect its argument help and source for required columns before preparing a dataset.

## Train

```bash
python training/train_gru.py --help
python training/train_gru.py --csv datasets/real_features.csv --epochs 50
```

Do not commit private datasets or generated model files unless you have the rights and intend to distribute them. Update model metadata and documentation when replacing the bootstrap files.

## Evaluation

- Split by participant, not by adjacent frames from the same recording.
- Keep a held-out test set untouched until final evaluation.
- Report class-wise precision, recall, F1, confusion matrix, sample counts, and uncertainty where practical.
- Measure end-to-end latency and frame rate on the intended hardware.
- Describe the data source, label definitions, camera conditions, preprocessing, and limitations.
- Compare the model with the heuristic baseline; do not claim safety effectiveness from generated-data metrics.
