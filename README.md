# ICSI 499 - DNA Design Project

## Project Overview

This project provides tools and a web interface for designing DNA sequences with specific fluorescence properties. It includes:
- A web interface for sequence generation and classification
- Command-line tools for advanced model training and sampling
- Multiple models: VAE, PriVAE, and a Classifier

## Project Structure

- `DNA-Design-Web/` - Web interface (Astro + React frontend, FastAPI backend)
- `VAE-Ag-DNA-design (VAE)/` - Command-line DNA generation and model training
- `PriVAE/` - Advanced property-isometric VAE (in development)
- `Ag-DNA-design (Classifier)/` - DNA sequence color classifier

## Requirements

- Python 3.8 or higher
- Node.js 18 or higher
- Git

---

## Components

### Web Interface
- Location: `DNA-Design-Web/`
- Tech: Astro + React frontend, FastAPI backend
- Features: Generate and classify DNA sequences, download results, user-friendly interface
- See: `DNA-Design-Web/QUICK_START.md` for setup and usage

### VAE Model
- Location: `VAE-Ag-DNA-design (VAE)/`
- Use: Command-line sequence generation and model training
- Key Files: `sampleSequences.py`, `kfoldrun.py`, `sequenceTrainer.py`

### PriVAE
- Location: `PriVAE/DNA/`
- Model implemented; web interface displays generated results
- Paper: [arXiv:2509.14287](https://arxiv.org/abs/2509.14287)

### Classifier
- Location: `Ag-DNA-design (Classifier)/`
- Model implemented; web interface displays classification results
- Paper: [ACS Nano 2022](https://doi.org/10.1021/acsnano.2c05390)

---

## Installation & Setup

1. Clone the main repository and enter the web directory:
	```bash
	git clone --recurse-submodules https://github.com/ICSI499-Team11-Capstone/ICSI499-Main.git
	cd ICSI499-Main
	cd DNA-Design-Web
	```

2. Environment configuration:
	- The `.env` file is included. Edit it if you need to change ports, model paths, etc.
	- If `.env` is missing, copy `.env.template` to `.env` and edit as needed.

3. Backend wadawdawsetup:
	```bash
	cd backend
	python3 -m venv venv
	source venv/bin/activate  # Windows: .\venv\Scripts\activate
	pip install -r requirements.txt
	uvicorn app.main:app --reload
	```

4. Frontend setup (in a new terminal, in DNA-Design-Web):
	```bash
	npm install
	npm run dev
	```

5. The frontend will be available at: http://localhost:3000
	The backend API will be at: http://localhost:8000

6. For command-line tools, see the README in each respective folder.

---

## Usage

### Web Interface
1. Start the backend:
	```bash
	cd DNA-Design-Web/backend
	source venv/bin/activate  # or .\venv\Scripts\activate on Windows
	uvicorn app.main:app --reload
	```
2. Start the frontend (in a new terminal):
	```bash
	cd DNA-Design-Web
	npm run dev
	```
3. Open your browser to: http://localhost:3000

### Command Line
See the README in `VAE-Ag-DNA-design (VAE)`, `PriVAE`, or `Ag-DNA-design (Classifier)` for details on running those tools.

---

## Documentation

- [Web Interface Guide](DNA-Design-Web/QUICK_START.md)
- [Troubleshooting](DNA-Design-Web/TROUBLESHOOTING.md)
- Individual component READMEs in each folder

---

## Links (When Running)

- Web Interface: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Research

- PriVAE: [arXiv:2509.14287](https://arxiv.org/abs/2509.14287)
- Classifier: [ACS Nano 2022](https://doi.org/10.1021/acsnano.2c05390)

---

**ICSI 499 - University at Albany**