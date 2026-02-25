# 🚀 Face Space Shooter

**A space shooter game controlled entirely by your facial expressions!**

Smile to move down, raise your eyebrows to move up, open your mouth to shoot lasers, and turn your head to dodge enemies. No keyboard needed — just your face and a webcam!

## 🎮 Play Now

**[▶ Play in your browser](https://web-rana3112s-projects.vercel.app)**

## 🕹️ Controls

| Expression | Action |
|---|---|
| 😊 Smile | Move ship **DOWN** |
| 🤨 Raise eyebrows | Move ship **UP** |
| 😮 Open mouth | **SHOOT** laser |
| ↩ Turn head left | Move ship **LEFT** |
| ↪ Turn head right | Move ship **RIGHT** |
| ⏸ Press **P** key | Pause / Resume |

## 🎯 Features

- **Face-controlled gameplay** — MediaPipe FaceLandmarker detects 478 facial landmarks in real-time
- **Progressive difficulty** — Enemies get tougher as you level up
- **Obstacles** — Asteroids (level 2+), Space Mines (level 3+), Laser Gates (level 5+)
- **Ship speed scaling** — Your ship starts slow and gets faster as you level up
- **Live camera preview** — See your face in the bottom-right corner
- **Expression indicators** — Bottom bar shows which expressions are active
- **Debug metrics** — Tune thresholds with real-time face metric values

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Game Engine | HTML5 Canvas + Vanilla JavaScript |
| Face Detection | [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| Camera | WebRTC `getUserMedia` API |
| Hosting | [Vercel](https://vercel.com) (static site) |
| Desktop Version | Python + Pygame + OpenCV + MediaPipe |

## 📁 Project Structure

```
face-space-shooter/
├── web/                    # Web version (deployable)
│   ├── index.html          # Game page with canvas & UI
│   ├── style.css           # Dark space theme
│   └── game.js             # Full game logic + MediaPipe JS
├── game.py                 # Desktop version (Python/Pygame)
├── face_landmarker.task    # MediaPipe model (desktop only)
└── README.md
```

## 🚀 Run Locally

### Web Version
```bash
cd web
python -m http.server 8080
# Open http://localhost:8080 in Chrome
```

### Desktop Version (Python)
```bash
pip install mediapipe pygame opencv-python numpy
python game.py
```

## 💡 Tips for Best Performance

1. Sit in **good lighting** facing the camera
2. Keep your face at **arm's length** from the camera
3. Make **exaggerated expressions** at first while you calibrate
4. The debug numbers in the top-right show live metric values

## 📄 License

MIT License
