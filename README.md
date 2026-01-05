# AlgoWarrior 🚀

**AlgoWarrior** is a small Flask-based project for traffic monitoring and counting that uses a YOLOv8 model (`yolov8n.pt`) for object detection. This repo contains Flask app code, a YOLO model checkpoint, sample uploads, and a small SQLite database used by the app.

---

## 🔎 Project overview

- **Main entry**: `01_FLASK_IMPLEMENTATION.py` (Flask application)
- **Model**: `yolov8n.pt` (YOLOv8 weights used by the app)
- **Templates**: `templates/frontend.html` (basic UI)
- **Uploads**: `uploads/videos/` (sample video files)
- **Database**: `database/traffic_system.db` (local SQLite DB)

---

## ✅ Features

- Real-time/near-real-time object detection using YOLOv8
- Simple Flask frontend for uploading and viewing video results
- Traffic counting and basic storage of counts in an SQLite database

---

## ⚙️ Requirements

- Python 3.8+ recommended
- Suggested packages (example):
  - Flask
  - opencv-python
  - ultralytics (or yolov8 dependencies)
  - numpy
  - sqlalchemy

Install dependencies (if you have a `requirements.txt`):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Or install common packages directly:

```bash
pip install Flask opencv-python numpy sqlalchemy
# Install yolov8/ultralytics according to their docs if needed
```

---

## 🚀 Quick start

1. Clone the repo:

```bash
git clone https://github.com/vky20006-beep/AlgoWarrior-
cd AlgoWarrior-
```

2. Create & activate a virtual environment and install dependencies (see above).

3. Run the Flask app (example):

```bash
python 01_FLASK_IMPLEMENTATION.py
# or, if the project uses flask command
# set FLASK_APP=01_FLASK_IMPLEMENTATION.py && flask run
```

4. Open the app in your browser at `http://127.0.0.1:5000/` or as configured.

---

## ⚠️ Important notes

- A `.env` file exists in the repo right now — it may contain sensitive configuration. It is recommended **not** to keep secrets committed. Consider removing it from the repo and adding it to `.gitignore`.

- Several video files and the model checkpoint are present. If you want to keep the repository slim, consider moving large assets to a release or using Git LFS for large model/media files.

### Quick commands to remove `.env` from future commits (keeps a local copy):

```bash
git rm --cached .env
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Remove .env from repo and add to .gitignore"
git push
```

To remove files from history entirely (use with care): use `git filter-repo` or `bfg` (I can help if you want).

---

## 📝 Development

- Add unit tests and a `requirements.txt` as needed.
- Consider adding a `Dockerfile` if you want reproducible deployments.

---

## 📄 License

Add a license to this repo (MIT/Apache/GPL/etc.). If you want, I can add an `MIT` license for you.

---

## 🙋 Contact

Maintainer: vky20006-beep — vky20006@gmail.com

---

If you'd like, I can also add a recommended `.gitignore`, remove `.env` from the repo, and/or move large files out of git history. Tell me which of these you'd like me to do next.