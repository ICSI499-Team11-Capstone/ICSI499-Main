# ICSI 499 - DNA Nanomachine Design Project

## 🧬 What is this project?

Imagine being able to design DNA sequences that glow in different colors! This project uses machine learning (AI) to create DNA sequences that, when combined with silver, produce fluorescent nanoparticles that emit specific colors of light.

**In simple terms:** We're using AI to design DNA that glows in the color you want (green, red, near-infrared, etc.).

---

## 🎯 Project Overview

**The Goal:** Design DNA sequences that produce specific colors and brightness levels when combined with silver nanoclusters.

**The Tools:** 
- **Web Interface** - Easy-to-use website where you can generate DNA sequences
- **VAE Model** - AI that creates DNA sequences with your target color/brightness
- **PriVAE Model** - Advanced AI that better preserves the properties you want
- **Classifier** - AI that predicts what color a DNA sequence will produce

**Real-World Application:** These fluorescent DNA nanomachines can be used in biosensors, medical imaging, and detecting diseases.

### Project Status

| Component | Status | What It Does |
|-----------|--------|--------------|
| **Web Interface (VAE)** | ✅ **Ready to Use!** | Website where you generate DNA sequences - works right now! |
| **PriVAE** | 🚧 Still Building | Better AI for creating DNA sequences - coming soon |
| **Classifier (SVM)** | 🚧 Still Building | AI that predicts what color your DNA will make |
| **PriVAE + Classifier** | 🔄 Future Update | Combining both tools for even better results |

---

## 📁 Repository Structure

```
ICSI499/
├── DNA-Design-Web/              # ✅ WORKING - Web interface for VAE model
│   ├── backend/                 # FastAPI backend
│   ├── src/                     # Astro + React frontend
│   ├── README.md               # Web interface documentation
│   ├── QUICK_START.md          # Quick setup guide
│   └── TROUBLESHOOTING.md      # Platform-specific troubleshooting
│
├── VAE-Ag-DNA-design (VAE)/    # VAE model training & sampling
│   ├── sequenceModel.py        # VAE model architecture
│   ├── sequenceTrainer.py      # Training pipeline
│   ├── sampleSequences.py      # Sequence generation
│   ├── kfoldrun.py            # K-fold cross-validation
│   ├── models/                 # Trained models
│   └── data-and-cleaning/      # Training data
│
├── PriVAE/                      # 🚧 Privacy-preserving VAE
│   ├── DNA/                     # DNA sequence implementation
│   │   ├── sequenceModel.py    # PriVAE model
│   │   ├── kfoldrun.py        # Training with k-fold
│   │   └── requirements.txt    # Dependencies
│   └── Peptide/                # Peptide sequence implementation
│
└── Ag-DNA-design (Classifier)/ # 🚧 SVM classifier
    ├── dna_featuregenerator.py # Feature extraction
    ├── SVM_accuracies.py       # Model training
    ├── predict_sequences.py    # Prediction pipeline
    ├── borutashap.py          # Feature selection
    └── training data/          # Training datasets
```

---

## 🚀 Quick Start

### Want to generate DNA sequences right now? Use the Web Interface!

The easiest way to get started - no biology knowledge needed:

```bash
# Navigate to the web interface folder
cd DNA-Design-Web

# Read the quick start guide
cat QUICK_START.md
```

**What you'll be able to do:**
- ✅ Generate 10-10,000 DNA sequences at once
- ✅ Choose your target color (wavelength)
- ✅ Choose your target brightness
- ✅ Download results as JSON, CSV, or Excel
- ✅ Preview sequences in your browser

**Website will run at:** http://localhost:4321 (on your computer)

---

### Advanced: Command Line (For the VAE Model)

If you prefer working in the terminal:

```bash
cd "VAE-Ag-DNA-design (VAE)"

# Edit the settings file
nano sampling-parameters.json

# Generate sequences
python sampleSequences.py
```

---

## � Prerequisites (What You Need Installed)

### Everyone Needs:
- **Python 3.8 or newer** - Programming language for the AI models
- **Node.js 18 or newer** - JavaScript runtime for the website
- **Git** - Version control (to download the code)
- **A terminal/command prompt** - To run commands

### Where to Download:
- Python: https://www.python.org/downloads/ (Get the latest version)
- Node.js: https://nodejs.org/ (Get the "LTS" version - Long Term Support)
- Git: https://git-scm.com/downloads

**Note:** If you have Python 3.8+ and Node.js 18+, you're good to go! Newer versions work fine.

---

## 📦 The Tools Explained

### 1. 🌐 DNA Design Web Interface ✅ YOU CAN USE THIS NOW!

**Location:** `DNA-Design-Web/`

**What it is:** A website (runs on your computer) where you can generate DNA sequences.

**What it does:**
- You pick a target color (wavelength) and brightness
- Click "Generate"
- The AI creates DNA sequences that should produce that color
- Download the results

**Technology Used:**
- **Frontend:** Astro + React (makes the website)
- **Backend:** FastAPI + Python (runs the AI)
- **AI Model:** PyTorch VAE (the machine learning part)

**Quick Setup:**
```bash
cd DNA-Design-Web

# Install website stuff
npm install

# Setup Python AI environment
cd backend
python3 -m venv venv
source venv/bin/activate  # Mac/Linux - OR - .\venv\Scripts\activate for Windows
pip install -r requirements.txt
```

**How to Run:**
```bash
# Terminal 1 - Start the AI backend
cd backend && source venv/bin/activate
uvicorn app.main:app --reload

# Terminal 2 - Start the website
npm run dev
```

**Then open in your browser:**
- Website: http://localhost:4321
- API Documentation: http://localhost:8000/docs

**Need Help?**
- [Full Guide](DNA-Design-Web/README.md)
- [5-Minute Quick Start](DNA-Design-Web/QUICK_START.md)
- [Fix Problems](DNA-Design-Web/TROUBLESHOOTING.md)

---

### 2. 🧬 VAE Model (The AI Brain - For Training)

**Location:** `VAE-Ag-DNA-design (VAE)/`

**What it is:** The AI model that learns from existing DNA sequences and can create new ones.

**When to use this:** Only if you want to train the AI from scratch or understand how it works. The web interface already uses a trained model!

**What's inside:**
- `sequenceModel.py` - The AI architecture (how the model is built)
- `sequenceTrainer.py` - Code to train the model
- `sampleSequences.py` - Generate sequences from command line
- `kfoldrun.py` - Train the model with cross-validation
- `models/` - Pre-trained models (already trained for you!)

**Key Concept - What is a VAE?**
- **Encoder:** Compresses DNA sequences into a simple "code"
- **Latent Space:** The "code" space where similar properties are close together
- **Decoder:** Turns the "code" back into a DNA sequence

**To Generate Sequences (Command Line):**
```bash
cd "VAE-Ag-DNA-design (VAE)"

# Edit settings
nano sampling-parameters.json

# Generate sequences
python sampleSequences.py
```

**To Train Your Own Model (Advanced):**
```bash
# Edit training settings
nano hyperparameters.json

# Train (takes a long time!)
python kfoldrun.py
```

**Sampling:**
```bash
# Edit sampling-parameters.json
{
  "Number of Samples": 100,
  "Wavelength Proxy Threshold": 90,
  "LII Proxy Threshold": 90,
  "Original Data Path": "./data-and-cleaning/cleandata.csv",
  "Model Path": "./models/kfold/a17lds17b0.007g2.0d1.0h15fold0.pt"
}

# Generate sequences
python sampleSequences.py
```

**Output:**
- Sequences saved to `data-for-sampling/past-samples-with-info/`
- Format: `.npz` files with sequences and properties

---

### 3. 🔒 PriVAE Model 🚧 COMING SOON!

**Location:** `PriVAE/DNA/`

**What it is:** An improved AI model that's better at creating DNA sequences with the exact properties you want.

**The Big Idea:** Regular VAE might "forget" what color you asked for. PriVAE uses a special trick (graph neural networks) to keep track of colors better.

**Why it's cool:**
- In real lab tests, it made **16.1 times more** rare-color DNA than expected!
- Organizes DNA sequences by their properties (like organizing songs by genre)
- Better at hitting your target color than the regular VAE

**Status:** 
- ✅ Code is done
- ✅ Tested in actual lab experiments (it works!)
- 🚧 Still hooking it up to the website

**Based on Research:**
- Published paper: [Property-Isometric Variational Autoencoders](https://arxiv.org/abs/2509.14287)
- By: Sadeghi et al., September 2025

**What's inside:**
- `sequenceModel.py` - The smarter AI model
- `kfoldrun.py` - Training code
- `sampleSequences.py` - Generate sequences
- `LatentSpaceVis.py` - See how the AI organizes sequences

**Color Groups You Can Generate:**
- Single colors: `clean-G` (Green), `clean-R` (Red), `clean-F` (Far-Red), `clean-N` (Near-Infrared)
- Mixed colors: `mixed-G-R`, `mixed-R-F`, `mixed-F-N`, etc.

**Setup:**
```bash
cd PriVAE/DNA
pip install -r requirements.txt
```

**How to use (once it's ready):**
1. Generate sequences with PriVAE
2. Check them with the Classifier (see below)
3. Use the best ones for your experiments

**Note:** Web interface integration coming soon. Currently available via command line only.

---

### 4. 🎯 Classifier (Color Predictor) 🚧 COMING SOON!

**Location:** `Ag-DNA-design (Classifier)/`

**What it is:** An AI that looks at a DNA sequence and predicts "This will probably glow GREEN" or "This will probably glow RED", etc.

**Why you need this:** After generating DNA sequences, you want to know which ones are most likely to give you the color you want!

**Based on Research:**
- Published paper: [Machine learning guided design of DNA-stabilized silver nanoclusters](https://doi.org/10.1021/acsnano.2c05390)
- By: Mastracco et al., ACS Nano 2022

**Status:** 
- ✅ The AI works
- ✅ Can predict colors
- 🚧 Still connecting it to PriVAE
- 🚧 Not on the website yet

**How it works:**
- Uses SVM (Support Vector Machine) - a type of AI
- Compares your sequence against known color pairs
- Gives you a probability score for each color (e.g., "75% chance of green")

**What's inside:**
- `dna_featuregenerator.py` - Converts DNA to numbers the AI can understand
- `SVM_accuracies.py` - Trains the AI model
- `predict_sequences.py` - **This is what you run** - Predicts colors
- Training data files in `training data/` folder

**⚠️ Important Format:** DNA sequences must look like this:
```
A C G T A C G T A C .
C G T A C G T A C A .
```
(Notice: spaces between letters, and a `.` at the end!)

**How to use it:**
```bash
cd "Ag-DNA-design (Classifier)"

# 1. Open the prediction script
nano predict_sequences.py

# 2. On line 27, change the file name to your sequences file

# 3. Run it!
python predict_sequences.py
```

**Output:**
- Creates a file called `out.csv`
- Shows probability for each color:
  - P(Green) - Probability it's green
  - P(Red) - Probability it's red  
  - P(Far-Red) - Probability it's far-red
  - P(NIR) - Probability it's near-infrared
  - P(Dark) - Probability it doesn't glow

**Best Practice Workflow:**
1. Generate DNA sequences with PriVAE
2. Format them properly (spaces + dot)
3. Run classifier to get probabilities
4. Pick sequences with high probability (>80%) for your target color
5. Use those in experiments!

---

## 💻 Installation (Setting Up Your Computer)

## 💻 Installation

### Clone Repository

```bash
git clone <repository-url>
cd ICSI499
```

### Web Interface Setup

See [DNA-Design-Web/QUICK_START.md](DNA-Design-Web/QUICK_START.md) for detailed instructions.

**Quick version:**
```bash
cd DNA-Design-Web

# Frontend
npm install

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### VAE Model Setup

```bash
cd "VAE-Ag-DNA-design (VAE)"
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### PriVAE Setup

```bash
cd PriVAE/DNA
pip install -r requirements.txt
```

### Classifier Setup

```bash
cd "Ag-DNA-design (Classifier)"
pip install numpy pandas scikit-learn boruta shap
```

---

## 📖 Step-by-Step Examples

### Example 1: Generate DNA Sequences (Easiest - Use the Website!)

**What you'll do:** Create 100 DNA sequences that should glow at a specific color and brightness.

**Steps:**
```bash
# 1. Start the backend (AI server)
cd DNA-Design-Web/backend
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
uvicorn app.main:app --reload

# 2. In another terminal, start the website
cd DNA-Design-Web
npm run dev
```

**Then in your browser:**
1. Go to: http://localhost:4321
2. Click "Open Designer"
3. Choose your settings:
   - Model: VAE
   - Wavelength (color) Threshold: 90%
   - Brightness Threshold: 90%
   - Number of Samples: 100
4. Click "Generate Sequences"
5. Download your sequences (JSON, CSV, or Excel format)

**Done!** You now have 100 DNA sequences to try in experiments.

---

### Example 2: Generate from Command Line (For Advanced Users)

**What you'll do:** Use Python directly to generate sequences.

```bash
cd "VAE-Ag-DNA-design (VAE)"

# Edit the settings file
nano sampling-parameters.json
# Change the values for:
# - "Number of Samples": 100
# - "Wavelength Proxy Threshold": 90
# - "LII Proxy Threshold": 90

# Generate sequences
python sampleSequences.py
```

**Output:** Check `data-for-sampling/past-samples-with-info/` for your results.

---

### Example 3: Train Your Own AI Model (Very Advanced!)

**Warning:** This takes a long time and requires understanding of machine learning!

```bash
cd "VAE-Ag-DNA-design (VAE)"

# Edit training settings
nano hyperparameters.json

# Train the model (this will take hours!)
python kfoldrun.py
```

**What this does:** Teaches the AI from scratch using your training data.

---

### Example 4: Using PriVAE + Classifier Together (Coming Soon!)

**When PriVAE and Classifier are integrated, you'll be able to:**

```bash
# Step 1: Generate sequences with PriVAE
cd PriVAE/DNA
python sampleSequences.py

# Step 2: Check quality with Classifier
cd "../../Ag-DNA-design (Classifier)"
# Edit predict_sequences.py to point to your PriVAE sequences
nano predict_sequences.py  # Update file path on line 27
python predict_sequences.py

# Step 3: Look at results
# Opens out.csv showing probability scores:
# - P(Green), P(Red), P(Far-Red), P(NIR), P(Dark)

# Step 4: Pick the best ones
# Use sequences with >80% probability for your target color
```

**Why this is useful:** You generate lots of sequences, then pick only the best ones!

---

## 🔬 How the AI Works (Optional - For Curious Minds)

### VAE (The AI Model) - Simple Explanation

Think of the VAE like this:
1. **Encoder:** Compresses DNA sequences into a simple "code" (like zipping a file)
2. **Latent Space:** The "code" where similar DNA sequences are grouped together
3. **Decoder:** Turns the "code" back into DNA sequences (like unzipping)

**Why it's cool:** You can pick a "code" for green DNA, and the decoder makes green DNA!

### Classifier - Simple Explanation

**What it does:** Looks at DNA sequences and says "I think this is 75% likely to be green"

**How it learns:** We showed it thousands of DNA sequences and told it what color each one makes. Now it can guess colors for new sequences!

---

## 📊 The Training Data

**Where:** `VAE-Ag-DNA-design (VAE)/data-and-cleaning/cleandata.csv`

**What's in it:**
- About 3000+ DNA sequences
- Each sequence has its color (wavelength: 500-900 nm)
- Each sequence has its brightness (0-100)

**Example:**
```
Sequence: ATCGTAGCTA...
Wavelength: 650 nm (red)
Brightness: 85
```

This is what the AI learned from!

---

## 🐛 Common Problems & Quick Fixes

### "Port 8000 already in use"
**Problem:** Something else is using port 8000.

**Fix:**
```bash
# Mac/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### "Module not found" or "Import Error"
**Problem:** Python can't find the packages it needs.

**Fix:**
```bash
cd DNA-Design-Web/backend
source venv/bin/activate  # Make sure you're in virtual environment!
pip install -r requirements.txt
```

### "npm install" is super slow or fails
**Problem:** Network issues or corrupted cache.

**Fix:**
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Website doesn't load
**Problem:** Servers aren't running.

**Check:**
1. Is the backend running? (Terminal should show "Uvicorn running...")
2. Is the frontend running? (Terminal should show "astro dev")
3. Try opening http://localhost:8000/docs - if this works, backend is up
4. Try opening http://localhost:4321 - if this works, frontend is up

**More Help:** Check [DNA-Design-Web/TROUBLESHOOTING.md](DNA-Design-Web/TROUBLESHOOTING.md) for detailed fixes.

---

## 🤝 Working as a Team

### When you make changes:

1. **Create your own branch:**
   ```bash
   git checkout -b your-name/what-you're-doing
   # Example: git checkout -b john/fix-button
   ```

2. **Make your changes and save:**
   ```bash
   git add .
   git commit -m "Simple description of what you did"
   # Example: git commit -m "Fixed the download button"
   ```

3. **Push to GitHub:**
   ```bash
   git push origin your-branch-name
   ```

4. **Create a Pull Request** on GitHub so others can review

### Before you start coding:
- Pull latest changes: `git pull origin main`
- Make sure everything works before changing things
- Ask teammates if you're not sure about something!

---

## 📚 Helpful Guides

**Start here if you're new:**
- [Web Interface Quick Start](DNA-Design-Web/QUICK_START.md) - Get the website running in 5 minutes
- [Full Web Documentation](DNA-Design-Web/README.md) - Everything about the website
- [Fix Problems](DNA-Design-Web/TROUBLESHOOTING.md) - Solutions for common issues

**For advanced users:**
- Each folder (VAE, PriVAE, Classifier) has its own README with technical details

---

## 🔗 Important Links (When Servers Are Running)

### Web Interface (When Running)
- **Frontend:** http://localhost:4321
- **Backend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
**Where to see things when running:**
- Website: http://localhost:4321
- Backend API: http://localhost:8000
- API documentation: http://localhost:8000/docs

**Helpful commands:**

```bash
# Start the website
cd DNA-Design-Web && npm run dev

# Start the backend
cd DNA-Design-Web/backend && source venv/bin/activate && uvicorn app.main:app --reload

# Make new DNA sequences (VAE)
cd "VAE-Ag-DNA-design (VAE)" && python sampleSequences.py

# Train the AI (VAE)
cd "VAE-Ag-DNA-design (VAE)" && python kfoldrun.py
```

---

## 📅 What's Done, What's Next

- **✅ Done:** VAE model works! Website for VAE is finished!
- **🚧 Working on now:** PriVAE model and Classifier
- **📋 Coming soon:** Connect everything together on the website

---

## 📖 Want to Learn More? (Optional Reading)

### About PriVAE
There's a research paper about PriVAE if you want to read more:
- **Title:** Property-Isometric Variational Autoencoders for Sequence Modeling and Design
- **Authors:** Elham Sadeghi, Xianqi Deng, I-Hsin Lin, Stacy M. Copp, Petko Bogdanov
- **Published:** September 2025 on arXiv
- **Link:** https://arxiv.org/abs/2509.14287
- **Cool fact:** PriVAE found 16 times more special DNA sequences than random searching!

### About the Classifier
There's also a paper about the classifier (the checker AI):
- **Title:** Machine learning guided design of highly fluorescent DNA-stabilized silver nanoclusters
- **Published:** ACS Nano 2022
- **Link:** https://doi.org/10.1021/acsnano.2c05390
- **What it does:** Uses machine learning to predict what colors DNA will glow

---

## 👥 Team

**University at Albany - ICSI 499 Project**