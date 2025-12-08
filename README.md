# DNA Nanomachine Designer (ICSI 499 Capstone)

A comprehensive tool for designing DNA sequences with specific fluorescence properties using VAE and PriVAE machine learning models.

## Quick Start

### Prerequisites
- **Python 3.12+**
- **Node.js 18+**
- **R (Required for VAE/PriVAE sampling)**
- **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone --recurse-submodules https://github.com/ICSI499-Team11-Capstone/ICSI499-Main.git
   cd ICSI499-Main
   ```

2. **Install Dependencies:**
   
   The provided startup scripts (`run_app.sh` and `run_app.bat`) will automatically check for and install most dependencies, including required R packages.

   *Manual Backend Setup:*
   ```bash
   cd DNA-Design-Web/backend
   pip install -r requirements.txt
   ```

   *Manual Frontend Setup:*
   ```bash
   cd ../  # Go to DNA-Design-Web/
   npm install
   cd ../  # Return to root
   ```

### Data Setup

> [!IMPORTANT]
> Some data files are too large to be hosted on GitHub. You must download them manually.

1.  **Download Data Files**:
    Go to the [Project Google Drive](https://drive.google.com/drive/folders/1b4egKmzdscT5vnMuIk53E3rzYnFzET5E) and download the following files:
    *   `CSdistance-250116-shuffled-final-labeled-data-log10.xlsx`
    *   `SORTED-250116-shuffled-final-labeled-data-log10.xlsx`

2.  **Place Files**:
    Move the downloaded files into the `PriVAE/DNA/data-and-cleaning/` directory.

### Configuration

> [!NOTE]
> This project includes pre-configured `.env` files in the repository for ease of setup.

- **Frontend**: The `.env` file in `DNA-Design-Web/` is already set up with default values.
- **Backend**: The `.env` file in `DNA-Design-Web/backend/` is already set up with default values.

You do not need to manually create these files unless you need to override specific settings.

1.  **Frontend Configuration** (`DNA-Design-Web/.env`):
    *   `PUBLIC_API_URL`: URL of the backend API (default: `http://localhost:8000`)
    *   `PORT`: Port for the frontend server (default: `4321`)

2.  **Backend Configuration** (`DNA-Design-Web/backend/.env`):
    *   `API_HOST`: Host for the backend server (default: `0.0.0.0`)
    *   `API_PORT`: Port for the backend server (default: `8000`)
    *   `FRONTEND_URL`: URL of the frontend (for CORS) (default: `http://localhost:4321`)

### Running the Application

- **Linux / macOS**:
  ```bash
  ./run_app.sh
  ```

- **Windows**:
  Double-click `run_app.bat` or run in CMD:
  ```cmd
  run_app.bat
  ```

The application will start:
- **Frontend**: [http://localhost:4321](http://localhost:4321)
- **Backend**: [http://localhost:8000](http://localhost:8000)

---

## Project Structure

- **`DNA-Design-Web/`**: The main web application (Astro + React frontend, FastAPI backend).
- **`VAE-Ag-DNA-design (VAE)/`**: Variational Autoencoder model for DNA sequence generation.
- **`PriVAE/`**: Advanced Property-Isometric VAE model.
- **`Ag-DNA-design (Classifier)/`**: DNA sequence color classifier.

## Components

### Web Interface (`DNA-Design-Web/`)
The user-facing interface for generating and classifying sequences.
- **Frontend**: Astro, React, TailwindCSS
- **Backend**: FastAPI, PyTorch

### VAE Model (`VAE-Ag-DNA-design (VAE)/`)
Generates DNA sequences based on wavelength and brightness targets.

### PriVAE Model (`PriVAE/`)
Generates sequences with specific property targets using an isometric latent space.

### Classifier (`Ag-DNA-design (Classifier)/`)
Predicts the fluorescence class of a given DNA sequence.

---

## Troubleshooting

> [!WARNING]
> If you encounter issues:
> 1. **"Module not found"**: Ensure you ran `pip install -r requirements.txt` in the `DNA-Design-Web/backend` folder.
> 2. **"npm not found"**: Ensure Node.js is installed and in your PATH.
> 3. **Ports in use**: The startup script attempts to free ports 8000 and 4321. If it fails, manually close applications using these ports.

For more detailed web-specific troubleshooting, see [DNA-Design-Web/TROUBLESHOOTING.md](DNA-Design-Web/TROUBLESHOOTING.md).
