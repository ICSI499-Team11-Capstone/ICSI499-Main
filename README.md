# ICSI 499 - DNA Design Project

## 🧬 Project Overview

We use AI to design DNA sequences that glow in specific colors when combined with silver. Generate thousands of custom DNA sequences for your target wavelength and brightness.

**Status:** 
- ✅ **Web Interface (VAE)** - Ready to use!
- 🚧 **PriVAE & Classifier** - In development

---

## 📁 Project Structure

- **DNA-Design-Web/** - ✅ Ready-to-use web interface 
- **VAE-Ag-DNA-design (VAE)/** - Command-line DNA generation
- **PriVAE/** - 🚧 Advanced model (in development)
- **Ag-DNA-design (Classifier)/** - 🚧 Color prediction (in development)

---

## 🚀 Quick Start

### Web Interface (Recommended)

```bash
cd DNA-Design-Web
npm install
```

**Setup Python backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

**Run the application:**
```bash
# Terminal 1 - Backend
cd backend && source venv/bin/activate
uvicorn app.main:app --reload

# Terminal 2 - Frontend  
npm run dev
```

**Open:** http://localhost:4321

### Command Line (Advanced)

```bash
cd "VAE-Ag-DNA-design (VAE)"
# Edit sampling-parameters.json
python sampleSequences.py
```

## 📋 Requirements

- Python 3.8+
- Node.js 18+
- Git

---

## � Components

### Web Interface (Ready ✅)
- **Location:** `DNA-Design-Web/`
- **Tech:** Astro + React frontend, FastAPI backend
- **Features:** Generate 10-10,000 sequences, choose target colors, download results
- **Docs:** See `DNA-Design-Web/QUICK_START.md`

### VAE Model 
- **Location:** `VAE-Ag-DNA-design (VAE)/`
- **Use:** Command-line sequence generation and model training
- **Key Files:** `sampleSequences.py`, `kfoldrun.py`, `sequenceTrainer.py`

### PriVAE (In Development 🚧)
- **Location:** `PriVAE/DNA/`
- **Improvement:** 16x better at generating rare sequences
- **Paper:** [arXiv:2509.14287](https://arxiv.org/abs/2509.14287)

### Classifier (In Development 🚧)
- **Location:** `Ag-DNA-design (Classifier)/`
- **Purpose:** Predict DNA sequence colors
- **Paper:** [ACS Nano 2022](https://doi.org/10.1021/acsnano.2c05390)

---

## 💻 Installation

```bash
git clone https://github.com/ICSI499-Team11-Capstone/ICSI499-Main.git --recursive
cd ICSI499
```

**For Web Interface:**
```bash
cd DNA-Design-Web
npm install

cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**For Command Line Tools:**
```bash
# VAE
cd "VAE-Ag-DNA-design (VAE)"
pip install torch numpy pandas matplotlib scikit-learn

# PriVAE  
cd PriVAE/DNA
pip install -r requirements.txt

# Classifier
cd "Ag-DNA-design (Classifier)"
pip install numpy pandas scikit-learn boruta shap
```

---

## � Usage Examples

### Web Interface (Easiest)
```bash
# Start backend
cd DNA-Design-Web/backend && source venv/bin/activate
uvicorn app.main:app --reload

# Start frontend (new terminal)
cd DNA-Design-Web && npm run dev
```
Visit: http://localhost:4321

### Command Line
```bash
cd "VAE-Ag-DNA-design (VAE)"
# Edit sampling-parameters.json
python sampleSequences.py
```

---

## 📚 Documentation

- [Web Interface Guide](DNA-Design-Web/QUICK_START.md)
- [Troubleshooting](DNA-Design-Web/TROUBLESHOOTING.md)
- Individual component READMEs in each folder

---

## 🔗 Links (When Running)

- Website: http://localhost:4321
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📄 Research

- **PriVAE:** [arXiv:2509.14287](https://arxiv.org/abs/2509.14287)
- **Classifier:** [ACS Nano 2022](https://doi.org/10.1021/acsnano.2c05390)

---

**ICSI 499 - University at Albany**