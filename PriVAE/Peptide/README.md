# Project Information

This folder contains the code, datasets, trained models, and visualizations for training and evaluating a Geometry Preserving model for Peptides sequence generation and analysis.
---

- **utils/**  
  Helper code used throughout the project:
  - `helpers.py`: Checks and sets up CUDA (GPU).
  - `model.py`: Base model class with helper tools.
  - `trainer.py`: Training class with reusable functions.
---

## Main Files 

- **optunaGenGridPep.py**  
  Main file to start training with Optuna (automatic hyperparameter tuning).

- **optuna_sampler_hyperopt.pkl**  
  Automatically created by Optuna. Stores the state of the TPE sampler so future searches can build on past trials.

- **sampling-parameters.json**  
  A settings file that defines:
  - What class to sample.
  - Dataset path.
  - Model.

- **sequenceDatasetPep.py**  
  Loads sequences in a format ready for training.

- **sequenceModelPep.py**  
  Defines the VAE model (encoder, decoder, latent space).

- **sequenceTrainer.py**  
  Runs training, loss, and evaluation for the model.

- **sampleSequences.py**  
  Uses the trained model to generate new DNA sequences.

- **LatentSpaceVis.py**  
  Creates 3D plots of the model’s latent space using PCA or UMAP.

- **plotRun.py**  
  Shows training progress like loss and accuracy.

- **embedder.py**
  Provides an `Embedder` class that leverages Facebook’s ESM-2 transformer model to encode biological sequences into high-dimensional embeddings.

---

## Requirements

The `requirements.txt` file includes all the Python packages you need to run this project.


## Peptide Results (PrIVAE – LSTM-based)

| Model    | Purity (Val) | Purity (Train) | Accuracy (Val) | Accuracy (Train) |
|----------|--------------|----------------|----------------|------------------|
| PrIVAE   | 0.38         | 0.39           | 0.93           | 0.96             |



## Reproduction Command

To reproduce the best-performing model above:

```bash
python optunaGenGridPep.py

