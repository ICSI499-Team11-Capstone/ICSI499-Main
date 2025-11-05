# Project Information

This folder contains the code, datasets, trained models, and visualizations for training and evaluating a Geometry Preserving model for DNA sequence generation and analysis.
---

- **utils/**  
  Helper code used throughout the project:
  - `helpers.py`: Checks and sets up CUDA (GPU).
  - `model.py`: Base model class with helper tools.
  - `trainer.py`: Training class with reusable functions.
---

## Main Files 

- **optunaGenGrid.py**  
  Main file to start training with Optuna (automatic hyperparameter tuning).

- **optuna_sampler_hyperopt.pkl**  
  Automatically created by Optuna. Stores the state of the TPE sampler so future searches can build on past trials.

- **sampling-parameters.json**  
  A settings file that defines:
  - What class to sample.
  - Dataset path.
  - Model.

- **sequenceDataset.py**  
  Loads DNA sequences in a format ready for training.

- **sequenceModel.py**  
  Defines the VAE model (encoder, decoder, latent space).

- **sequenceTrainer.py**  
  Runs training, loss, and evaluation for the model.

- **sampleSequences.py**  
  Uses the trained model to generate new DNA sequences.

- **LatentSpaceVis.py**  
  Creates 3D plots of the model’s latent space using PCA or UMAP.

- **plotRun.py**  
  Shows training progress like loss and accuracy.

- **kfoldrun.py**, **kfoldDataset.py**, **kfoldsequenceTrainer.py**  
  Used for evaluating the model using k-fold cross-validation.

---

## Requirements

The `requirements.txt` file includes all the Python packages you need to run this project.


## DNA nanoclusters Results (PrIVAE – LSTM-based)

| Model    | Purity (Val) | Purity (Train) | Accuracy (Val) | Accuracy (Train) |
|----------|--------------|----------------|----------------|------------------|
| PrIVAE   | 0.61         | 0.66           | 0.90           | 0.93             |


## Reproduction Command

To reproduce the best-performing model above:

```bash
python optunaGenGrid.py

