# DNA Nanomachine Designer

A web application for designing DNA sequences with specific optical properties using VAE and PriVAE machine learning models.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Windows Setup](#windows-setup)
  - [macOS Setup](#macos-setup)
  - [Linux/WSL Setup](#linuxwsl-setup)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### VAE Model
- **Wavelength Proxy Threshold**: Filter sequences by minimum wavelength threshold (0-100%)
- **LII (Brightness) Proxy Threshold**: Filter sequences by minimum brightness threshold (0-100%)
- **Number of Samples**: Generate 10 to 10,000 sequences
- **Export Formats**: JSON, CSV, Excel-compatible CSV

### PriVAE Model
- **Group Labels**: Select from wavelength groups
  - Clean: G (Green), R (Red), F (Far-Red), N (NIR)
  - Mixed: Various combinations (e.g., G-R, R-F, N-G, etc.)
- **Number of Sample**: Generate 10 to 10,000 sequences
- **Export Formats**: JSON, CSV, Excel-compatible CSV

---

## 📁 Project Structure

```
DNA-Design-Web/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # Main API application
│   │   └── routers/
│   │       └── vae.py         # VAE model endpoints
│   ├── venv/                  # Python virtual environment
│   └── requirements.txt       # Python dependencies
├── src/                       # Astro frontend
│   ├── components/
│   │   ├── DNADesigner.tsx    # Main designer component
│   │   └── Navigation.astro   # Navigation bar
│   ├── pages/
│   │   ├── index.astro        # Home page
│   │   └── designer.astro     # Designer page
│   └── layouts/
│       └── Layout.astro       # Base layout
├── node_modules/              # Node.js dependencies
├── package.json               # Node.js dependencies list
└── README.md                  # This file
```

---

## 🔧 Prerequisites

### All Platforms
- **Python 3.8+** - Backend API server
- **Node.js 18+** - Frontend development server
- **Git** - Version control
- **Terminal/Command Line** access

### Installation Links
- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/ (LTS version recommended)
- Git: https://git-scm.com/downloads

---

## 💻 Installation

### Windows Setup

#### 1. Install Prerequisites
```powershell
# Download and install Python from python.org
# Download and install Node.js from nodejs.org
# Download and install Git from git-scm.com

# Verify installations
python --version
node --version
npm --version
git --version
```

#### 2. Clone Repository
```powershell
git clone <repository-url>
cd DNA-Design-Web
```

#### 3. Setup Backend
```powershell
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

#### 4. Setup Frontend
```powershell
# Install Node.js dependencies
npm install
```

---

### macOS Setup

#### 1. Install Prerequisites
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python, Node.js, and Git
brew install python node git

# Verify installations
python3 --version
node --version
npm --version
git --version
```

#### 2. Clone Repository
```bash
git clone <repository-url>
cd DNA-Design-Web
```

#### 3. Setup Backend
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

#### 4. Setup Frontend
```bash
# Install Node.js dependencies
npm install
```

---

### Linux/WSL Setup

#### 1. Install Prerequisites (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python, Node.js, and Git
sudo apt install python3 python3-pip python3-venv nodejs npm git

# Verify installations
python3 --version
node --version
npm --version
git --version
```

**WSL Users:** Make sure you're running commands inside WSL, not from Windows PowerShell.

#### 2. Clone Repository
```bash
git clone <repository-url>
cd DNA-Design-Web
```

#### 3. Setup Backend
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

#### 4. Setup Frontend
```bash
# Install Node.js dependencies
npm install
```

## Running the Application

You need to run **both** the backend and frontend servers simultaneously.

### Two Terminal Setup (Recommended)

**Terminal 1 - Backend Server:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Terminal 2 - Frontend Server:**
```bash
npm run dev
```

The frontend will be available at:
- Web UI: http://localhost:4321

## Usage

1. Navigate to http://localhost:4321 in your browser
2. Click "Open Designer" to access the DNA sequence designer
3. Select a model (VAE or PriVAE)
4. Configure the parameters:
   - For VAE: Set wavelength and brightness thresholds
   - For PriVAE: Choose a group label
   - Select number of samples (preset or custom)
5. Choose an export format (JSON, CSV, or Excel)
6. Click "Generate & Download"
7. View the generated sequences and download if needed

## API Endpoints

### VAE Endpoints

- `POST /api/v1/vae/sample` - Generate sample DNA sequences
- `GET /api/v1/vae/transform/{sequence}` - Transform a sequence to one-hot encoding
- `GET /api/v1/vae/stats` - Get dataset statistics

Interactive API documentation: http://localhost:8000/docs

## Development

### Frontend Development
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

### Backend Development
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload  # Auto-reload on code changes
```

## Technology Stack

- **Frontend**: Astro, React, TypeScript, Tailwind CSS, TanStack Query
- **Backend**: FastAPI, Python, PyTorch, NumPy, Pandas
- **Models**: VAE, PriVAE (external model directories)

## Troubleshooting

### Port Already in Use
```bash
pkill -f uvicorn
```

### Module Import Errors
```bash
cd backend
source venv/bin/activate
```

### Frontend Build Errors
```bash
rm -rf node_modules package-lock.json
npm install
```

## License

UAlbany 2025
