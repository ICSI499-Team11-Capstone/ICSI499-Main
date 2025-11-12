# DNA-Design-Web Quick Start Guide

This guide will help you set up the project, including pulling the DNA-Design-Web submodule, and running the backend and frontend on both Mac and Windows.

---

## 1. Clone the Repository (with Submodule)


```
git clone --recurse-submodules https://github.com/ICSI499-Team11-Capstone/ICSI499-Main.git
cd ICSI499-Main
cd DNA-Design-Web
```

If you already cloned without submodules, run:
```
git submodule update --init --recursive
```

---

## 2. Update Submodule (if needed)

If the DNA-Design-Web submodule is updated:
```
git submodule update --remote --merge
```
The submodule URL is: https://github.com/ICSI499-Team11-Capstone/DNA-Design-Web

---

## 3. Environment Configuration

The `.env` file is already included in the repo for convenience. Edit it if you need to change ports, model paths, etc.

> **Warning:** The `.env` file may be removed from the repo in the future. If it is missing, copy `.env.template` to `.env` and edit as needed.

## 4. Backend Setup

### Mac/Linux
```
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Windows
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 5. Frontend Setup

### Mac/Linux/Windows
```
# (Make sure you are in DNA-Design-Web directory)
npm install
npm run dev
```

- The frontend will be available at: http://localhost:3000
- The backend API will be at: http://localhost:8000

---

## 5. Project Structure Note

Only the `DNA-Design-Web` folder is a git submodule. All other folders (such as `Ag-DNA-design (Classifier)`, `PriVAE`, `VAE-Ag-DNA-design (VAE)`, etc.) are part of the main project and do not require submodule commands.

---

## 6. Troubleshooting
- If you see `ModuleNotFoundError: No module named 'app'`, make sure you are in the `backend` directory when running Uvicorn, or use the correct import path from the project root.
- For any dependency issues, re-run `pip install -r requirements.txt` or `npm install` as needed.

---

## 7. Useful Commands
- Pull latest changes (including submodule):
  ```
  git pull --recurse-submodules
  git submodule update --remote --merge
  ```
- Check submodule status:
  ```
  git submodule status
  ```

---

For more details, see the `README.md` files in each directory.
