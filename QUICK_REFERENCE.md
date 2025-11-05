# Quick Reference Card

> Copy-paste commands for quick setup and common tasks

## One-Line Setup (After Clone)

```bash
cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd .. && npm install
```

## Start Both Servers (Copy each to separate terminal)

**Terminal 1 (Backend):**
```bash
cd DNA-Design-Web/backend && source venv/bin/activate && uvicorn app.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd DNA-Design-Web && npm run dev
```

## Quick Fixes

### Kill stuck servers
```bash
# Kill backend
lsof -ti:8000 | xargs kill -9

# Kill frontend
lsof -ti:4321 | xargs kill -9

# Kill both
pkill -9 -f "uvicorn|astro"
```

### Reinstall dependencies
```bash
# Backend
cd backend && source venv/bin/activate && pip install -r requirements.txt

# Frontend
rm -rf node_modules && npm install
```

## Git Cheat Sheet

```bash
# Start work
git pull origin main
git checkout -b yourname/feature

# Save work
git add .
git commit -m "Clear description"
git push origin yourname/feature

# Update your branch
git pull origin main

# Check status
git status
git branch
```

## URLs When Running

| Service | URL |
|---------|-----|
| Website (Frontend) | http://localhost:4321 |
| API (Backend) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Python Virtual Environment

```bash
# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Check if active (you should see (venv) in prompt)
which python
```

## Common Commands

```bash
# Check versions
python3 --version
node --version
npm --version

# See what's running on ports
lsof -ti:8000  # Backend
lsof -ti:4321  # Frontend

# View logs
tail -f /tmp/backend.log  # If logging to file

# Test API
curl http://localhost:8000/
curl -X POST http://localhost:8000/vae/sample \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 5}'
```

## File Locations

```
/home/aki/ICSI499/
├── TEAM_SETUP.md           # Full setup guide
├── README.md               # Project overview
└── DNA-Design-Web/
    ├── backend/
    │   ├── app/main.py     # Backend entry point
    │   └── requirements.txt # Python packages
    └── package.json        # Node packages
```

## Emergency Reset

> [!CAUTION]
> This will delete your virtual environment and node_modules

```bash
# Full clean reinstall
cd DNA-Design-Web
rm -rf backend/venv node_modules
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
npm install
```

## Getting Help

1. Check error message carefully
2. Search error in project docs: `grep -r "error_text" .`
3. Check if servers are running: `lsof -ti:8000 -ti:4321`
4. Verify virtual environment: `which python` (should show venv path)
5. Ask team in group chat
6. Create GitHub issue with error details