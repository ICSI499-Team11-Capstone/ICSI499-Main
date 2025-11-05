# Quick Setup Guide for Teammates

> **TL;DR:** Clone the repo, install dependencies, run backend + frontend, open http://localhost:4321

## What This Project Does
We're using AI to design DNA sequences that glow in different colors! Right now, you can use the website to generate DNA sequences.

---

## Get Started in 5 Minutes

### Step 1: Clone the Project
```bash
git clone git@github.com:ICSI499-Team11-Capstone/DNA-Design-Web.git
cd DNA-Design-Web
```

### Step 2: Setup Backend (Python)
```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate     # On Windows

# Install requirements
pip install -r requirements.txt
```

### Step 3: Setup Frontend (Website)
```bash
# Go back to main folder
cd ..

# Install Node packages
npm install
```

### Step 4: Run Everything

> [!IMPORTANT]
> You need to run both backend and frontend in separate terminals

**Terminal 1 - Backend:**
```bash
cd DNA-Design-Web/backend
source venv/bin/activate
uvicorn app.main:app --reload
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 - Frontend:**
```bash
cd DNA-Design-Web
npm run dev
```
You should see: `Local: http://localhost:4321/`

### Step 5: Open Your Browser
Go to: **http://localhost:4321**

You should see the DNA Design website!

---

## What Works Right Now

- [x] Website is fully functional
- [x] Generate 10-10,000 DNA sequences
- [x] Choose target color and brightness
- [x] Download as JSON, CSV, or Excel

---

## Prerequisites

> [!NOTE]
> Check if you have these installed before starting

**Required software:**
- **Python 3.8+** - Check: `python3 --version`
- **Node.js 18+** - Check: `node --version`
- **npm** - Check: `npm --version`

**Don't have them?**
- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/ (includes npm)

---

## Common Problems

<details>
<summary><strong>Backend won't start?</strong></summary>

```bash
# Make sure you activated the virtual environment
source venv/bin/activate  # You should see (venv) in your terminal

# Try reinstalling
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Frontend won't start?</strong></summary>

```bash
# Delete node_modules and reinstall
rm -rf node_modules
npm install
```
</details>

<details>
<summary><strong>Port already in use?</strong></summary>

```bash
# Kill the process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Kill the process on port 4321 (frontend)
lsof -ti:4321 | xargs kill -9
```
</details>

<details>
<summary><strong>"Module not found" error?</strong></summary>

```bash
# Backend:
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Frontend:
cd DNA-Design-Web
npm install
```
</details>

---

## How to Use the Website

1. Open http://localhost:4321
2. Enter your settings:
   - Number of sequences (10-10,000)
   - Target wavelength (color)
   - Target brightness
3. Click "Generate Sequences"
4. Wait (might take 1-2 minutes for large batches)
5. Download your results

---

## Git Workflow

### Before you start coding

> [!WARNING]
> Always pull latest changes before starting work

```bash
# Always pull latest changes first
git pull origin main

# Check what branch you're on
git branch
```

### Making changes
```bash
# 1. Create a new branch
git checkout -b your-name/feature-description

# 2. Make your changes

# 3. Check what changed
git status

# 4. Add your changes
git add .

# 5. Commit with a clear message
git commit -m "Added feature X to do Y"

# 6. Push to GitHub
git push origin your-name/feature-description

# 7. Create a Pull Request on GitHub
```

### Commit message examples

| Good | Bad |
|------|-----|
| `Fixed backend API endpoint for VAE sampling` | `fixed stuff` |
| `Added validation for DNA sequence input` | `updates` |
| `Updated frontend to display error messages` | `changes` |

---

## Project Structure

```
DNA-Design-Web/
├── backend/              # Python FastAPI backend
│   ├── app/
│   │   ├── main.py      # Main API file
│   │   ├── routers/     # API endpoints
│   │   └── utils/       # Helper functions
│   ├── venv/            # Virtual environment (don't commit this!)
│   └── requirements.txt # Python packages
│
├── src/                 # Frontend code
│   ├── pages/           # Website pages
│   ├── components/      # React components
│   └── layouts/         # Page layouts
│
├── public/              # Static files (images, etc.)
└── package.json         # Node.js packages
```

---

## Need Help?

### Documentation
- Main project: `/home/aki/ICSI499/README.md`
- Quick start: `QUICK_START.md`
- Troubleshooting: `TROUBLESHOOTING.md`

### Ask the team
- Post in the group chat
- Create an issue on GitHub
- Ask during team meetings

### Still stuck?
- Google the error message
- Check Stack Overflow
- Ask ChatGPT or GitHub Copilot

---

## Tips for Success

> [!TIP]
> These practices will save you time and headaches

1. **Always pull before you code** - `git pull origin main`
2. **Use branches** - Don't commit directly to main
3. **Test before you commit** - Make sure it works
4. **Write clear commit messages** - Future you will thank you
5. **Ask questions** - No question is too small
6. **Read error messages** - They usually tell you what's wrong
7. **Save your work often** - Commit frequently

---

## You're All Set

If you can see the website at http://localhost:4321 and generate DNA sequences, you're good to go!

**Questions?** Ask the team! We're all learning together.
