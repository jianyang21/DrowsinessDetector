# 👁️ Drowsiness Detection System using Python and OpenCV

This project detects driver drowsiness using eye aspect ratio (EAR) through real-time webcam video. If the user's eyes remain closed beyond a threshold number of frames, a visual alert appears on the screen indicating drowsiness.

---

## 🚀 Features

- Real-time drowsiness detection using a webcam
- Uses dlib facial landmark detection (68-point model)
- Visual alert with red bounding box and central alert text
- Detects both eyes and calculates EAR for accuracy
- Displays message when no face is detected

---

## 🧠 Tech Stack

- Python
- OpenCV
- Dlib
- imutils
- NumPy
- SciPy

---

## 📂 Clone the Repository

```bash
git clone https://github.com/jingyang21/drowsiness-detection.git
cd drowsiness-detection
