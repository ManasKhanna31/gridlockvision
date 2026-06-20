# SETUP.md — Run GridlockVision AI on Your Own Machine

This guide assumes you have a Windows, Mac, or Linux laptop and want to
go from "downloaded the zip" to "demo running in browser." Pick **Path
A (Docker)** if you have Docker installed — it's the least error-prone.
Pick **Path B (Plain Python)** if you don't want to install Docker.

---

## Before you start: what you need installed

| Tool | Check if installed | Get it |
|---|---|---|
| Python 3.10 or 3.11 | `python3 --version` | https://python.org/downloads |
| Docker Desktop (Path A only) | `docker --version` | https://docker.com/products/docker-desktop |
| Git (optional, to clone if you push this to GitHub) | `git --version` | https://git-scm.com |

> Windows users: run these commands in **PowerShell** or **Command
> Prompt**. Use `python` instead of `python3` if that's what your
> system recognizes.

---

## Path A — Docker (recommended, simplest)

1. Unzip the project folder anywhere, e.g. `Desktop/gridlockvision`.
2. Open a terminal **inside the `docker` folder**:
   ```bash
   cd path/to/gridlockvision/docker
   ```
3. Build and start both services:
   ```bash
   docker compose up --build
   ```
   First build takes a few minutes (downloading Python + OpenCV +
   YOLO + EasyOCR layers). Subsequent runs are fast.
4. Once you see logs from both `gridlockvision-backend` and
   `gridlockvision-dashboard` settle (no more "Building"/"Starting"
   spam), open your browser:
   - Dashboard: **http://localhost:8501**
   - API docs: **http://localhost:8000/docs**
5. To stop: press `Ctrl+C` in that terminal, then run
   `docker compose down` to clean up containers.

**If port 8000 or 8501 is already in use** on your machine, edit
`docker/docker-compose.yml` and change the left side of the port
mapping, e.g. `"8001:8000"`, then reopen at that new port.

---

## Path B — Plain Python (no Docker)

### 1. Create a virtual environment
```bash
cd path/to/gridlockvision
python3 -m venv venv
```

Activate it:
- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

You should see `(venv)` appear at the start of your terminal prompt.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
This installs FastAPI, OpenCV, Ultralytics (YOLOv8), EasyOCR, Streamlit,
and everything else listed in `requirements.txt`. It will take a few
minutes — EasyOCR and Ultralytics are the largest downloads.

> **If `pip install` fails on a specific package:** upgrade pip first
> (`pip install --upgrade pip`) and retry. EasyOCR pulls in PyTorch,
> which is the heaviest dependency — on a slow connection this step
> alone can take 5-10 minutes.

### 3. Start the backend API
Open **Terminal 1**:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Leave this running. The first time it runs, `ultralytics` will
auto-download the YOLOv8n weights (~6MB) — needs internet just this
once.

You should see something like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Visit **http://localhost:8000/docs** to confirm it's alive — you should
see the interactive API documentation page.

### 4. Start the dashboard
Open a **second terminal** (keep Terminal 1 running), activate the same
venv again, then:
```bash
cd frontend/dashboard
streamlit run app.py
```
It should auto-open your browser to **http://localhost:8501**. If not,
open that URL manually.

### 5. Try it
- Click **Judge Demo Mode** in the sidebar.
- Upload any traffic photo (a street scene with cars/bikes).
- Watch the pipeline steps render.
- Switch to **Analytics Dashboard** to see the result land in charts.

---

## Common problems

**"ModuleNotFoundError: No module named 'app'"**
You ran `uvicorn` from the wrong folder. Make sure you `cd backend`
first — the command needs to run from inside `backend/`, not the
project root.

**Streamlit says "Could not reach backend API"**
The FastAPI server (Terminal 1) isn't running, or crashed. Check
Terminal 1 for an error message. Also confirm you're not blocking
localhost traffic with a firewall/VPN.

**EasyOCR is very slow on first plate read**
It downloads its own OCR model weights (~50-100MB) the very first time
`recognize_plate()` runs. This only happens once and is cached after.

**"No module named cv2" or similar even after pip install**
You likely activated a different/no virtual environment in this
terminal than the one you installed packages into. Re-run the activate
command for this terminal (see Step 1 above) and try again.

**Detection looks empty / no boxes drawn**
Confirm the uploaded image actually contains vehicles/people the COCO
model recognizes (car, motorcycle, bus, truck, bicycle, person) and
that `VEHICLE_CONF_THRESHOLD` in `backend/app/core/config.py` (default
0.35) isn't filtering out a low-confidence detection — lower it
temporarily to test.

**Helmet/seatbelt/plate always show "UNKNOWN" or "STUB MODE"**
This is expected and correct — see the README's "Implemented vs Future
Enhancements" section. Those modules need a trained model file dropped
into `backend/app/models_store/` (see filenames in
`backend/app/core/config.py`) before they activate.

---

## Training your own helmet/seatbelt/plate models (optional, advanced)

If you want to move past stub mode before the hackathon:

1. Collect ~200-500 labelled images per task (helmet/no-helmet crops,
   seatbelt/no-seatbelt crops, or plate bounding boxes). Roboflow has
   several public starter datasets you can search and use as a base,
   or label your own with Roboflow/LabelImg.
2. Fine-tune a YOLOv8 model:
   ```bash
   pip install ultralytics
   yolo task=detect mode=train model=yolov8n.pt data=your_dataset.yaml epochs=50
   ```
3. Copy the resulting `runs/detect/train/weights/best.pt` to:
   - `backend/app/models_store/helmet_best.pt`
   - `backend/app/models_store/seatbelt_best.pt`
   - `backend/app/models_store/plate_best.pt`
   (matching whichever task you trained.)
4. Restart the backend — the console log will confirm the model loaded
   instead of printing the stub-mode notice.

---

## Pushing this to GitHub (optional)

```bash
cd gridlockvision
git init
git add .
git commit -m "Initial GridlockVision AI prototype"
git remote add origin <your-repo-url>
git push -u origin main
```

The included `.gitignore` already excludes generated data, model
weights, and virtual environments so your repo stays clean.
