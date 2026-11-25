# Model files

This folder contains the optional temporal drowsiness model implementation, scaler, metadata, and a small bootstrap checkpoint.

The bundled BiGRU checkpoint was trained on generated feature sequences so the local inference path can be demonstrated. Its reported bootstrap accuracy is not a real-driver benchmark. Do not use it for safety decisions or claim real-world performance. Retrain and evaluate with appropriately licensed real labeled data and participant-independent splits before reporting model results.

See [`../training/README.md`](../training/README.md) and [`../README.md`](../README.md#model-and-evaluation-status).
