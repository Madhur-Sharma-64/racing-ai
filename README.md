# Racing AI 🏎️💨

[![CI](https://github.com/Madhur-Sharma-64/racing-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Madhur-Sharma-64/racing-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Engine](https://img.shields.io/badge/engine-NumPy%20%2B%20Pygame--CE-green)
![Tests](https://img.shields.io/badge/tests-37%20passed-success)
![Frame Rate](https://img.shields.io/badge/fps-60%20FPS%20smooth-brightgreen)
![License](https://img.shields.io/badge/license-MIT-purple)

A high-performance 2D racing simulation, autonomous driving environment, and neuroevolution championship. Built from scratch in Python and NumPy, featuring a kinematic vehicle model, 9-ray forward radar sensors, real-world & hybrid circuit splines, dynamic seasonal weather environments, authentic real-sample engine acoustics, and headless training environments capable of simulating **over 20,000 steps per second**.

Designed as a software engineering portfolio project showcasing clean simulation architecture, spatial algorithm optimization, real-time audio/VFX systems, and autonomous genetic algorithm controllers.

---

## 🌟 Highlights

* **Realistic 2D Vehicle Dynamics**: Kinematic bicycle model with longitudinal gas/brake curves, quadratic aerodynamic drag, rolling resistance, and a dynamic lateral grip threshold ($38\,\text{m/s}^2 \approx 3.9\,\text{G}$) enforcing high-downforce racing grip and power slides.
* **Liquid-Smooth 60 FPS Simulation & Precision Timing**: 60 FPS display loop paired with sub-frame finish-line transponder interpolation for true **3-decimal-place millisecond precision** (`mm:ss.mmm`), matching official FIA timing standards.
* **Dynamic Seasonal Weather & Scenery**:
  * 🌸 **Spring Sakura**: Lush green turf, blooming cherry blossom trees with drifting pink petals fluttering across the track.
  * 🍂 **Autumn Amber**: Golden grass, red-gold maple canopies shedding leaves caught in car aerodynamic wakes.
  * ❄️ **Cold Winter**: Frostbitten terrain, snowy pine trees, and floating snowflakes.
  * ☀️ **Summer GP**: Classic vibrant circuit with high-contrast racing tarmac.
* **Genuine Real-Sample Audio Engine**: SDL2 audio streaming with real recorded V8 combustion loops, dynamic 6-speed gearbox pitch sweeps, overrun coasting rumbles, and stereo-panned tire squeal during high-G drift slip.
* **Broadcast-Grade F1 Digital Dashboard**:
  * Curved 15-LED RPM shift light bar (Green $\rightarrow$ Red $\rightarrow$ Flashing Purple rev limiter).
  * Digital speedometer, 7-speed gear indicator, and dynamic G-Force accelerometer friction circle.
  * Persistent rubber skid marks, tire smoke, and titanium kerb sparks.
* **4 Autonomous AI Rival Champions**:
  * 🔷 **KEEPER** (Cyan): Defensive, ultra-clean consistency specialist.
  * 🔶 **SPEED DEMON** (Demon Orange): Aggressive late-braker maximizing corner entry velocity.
  * 🟡 **MADX** (Gold): High-downforce precision apex hunter.
  * 🟢 **ZORO** (Emerald Green): Universal champion trained across technical chicanes and hairpins.
* **Real-World & Giant Hybrid Circuits**:
  * Real-world circuits: **Monza**, **Silverstone**, **Spa-Francorchamps**, **Interlagos**, **Monaco GP**, and **Speedway Oval**.
  * Procedural & sketch circuits: **Grand Interlagos**, **Hairpin Haven**, **Chicane Mania**, **Sweeper Paradise**, and endless randomized tracks.

---

## ⚡ Performance & Benchmarks

Simulating thousands of car-track interactions during AI training requires extreme simulation throughput. Raycasting and centerline queries are accelerated using NumPy vectorization and spatial distance culling:

| Component | Baseline | Optimized | Speedup |
|---|---|---|---|
| **Raycasting (9 rays + 200m radar)** | `0.199 ms` | `0.039 ms` | **~5.1× faster** |
| **Track Centerline & Containment Query** | `0.012 ms` | `0.005 ms` | **~2.4× faster** |
| **Full Simulation Step (Physics + Rays + Lap)** | `0.550 ms` | `0.049 ms` | **~11.2× faster** |
| **Simulation Throughput (Single CPU Core)** | ~1,800 steps/s | **>20,000 steps/s** | **~11.2× faster** |
| **Seasonal Scenery & Particles (Culled)** | `18.5 ms` | **`2.2 ms`** | **~8.4× faster** |

> *A full 60-second lap (1,200 simulation steps at $dt = 0.05\,\text{s}$) runs in under **0.06 seconds** in headless training mode.*

---

## 🏗️ Project Architecture

```
racing-ai/
├── sim/                     # Core simulation & physics (pure NumPy)
│   ├── car.py               # Vehicle model (bicycle kinematics & grip limits)
│   ├── track.py             # Track representation, centerline, progress & containment
│   ├── tracks.py            # Real-world circuit definitions & spline interpolation
│   ├── generator.py         # Procedural track generator & circuit archetypes
│   ├── raycast.py           # Vectorized 2D ray-segment distance sensors & forward radar
│   ├── lap.py               # FIA millisecond lap timer with sub-frame interpolation
│   └── env.py               # Headless RL (RaceEnv) & batch GA (MultiCarEnv) environments
├── ui/                      # Presentation & audio layers
│   ├── camera.py            # Coordinate transformations (world meters -> screen pixels)
│   ├── render.py            # Pygame drawing (track polygons, F1 HUD dash, VFX particles)
│   ├── environment.py       # Seasonal scenery (trees, petals, leaves, snow & wake physics)
│   └── sound.py             # Real-sample streaming audio engine (V8 engine & tire acoustics)
├── ai/                      # AI controllers (neural networks & evolutionary training)
│   ├── network.py           # Neural network policy architecture
│   ├── evolution.py         # Genetic algorithm & population management
│   └── parallel_worker.py   # Multi-core parallel batch evaluation
├── models/                  # Trained neural network weights (.npz)
│   ├── KEEPER.npz           # Defensive safety champion
│   ├── SPEED_DEMON.npz      # High-speed aggressive champion
│   ├── MADX.npz             # Apex precision champion
│   └── ZORO.npz             # Universal master champion
├── assets/                  # Genuine recorded audio samples (.wav)
├── tests/                   # Automated test suite (37 tests)
│   ├── test_car.py          # Physics and grip limit tests
│   ├── test_camera.py       # Viewport projection and camera rotation tests
│   ├── test_lap.py          # Sub-frame millisecond lap timing tests
│   ├── test_raycast.py      # Wall intersection and sensor distance tests
│   ├── test_tracks.py       # Geometric circuit validation & length scaling
│   ├── test_env.py          # Headless environment reset and step tests
│   └── test_ai.py           # Neural network forward pass & genetic operators
├── drive.py                 # Playable 60 FPS simulator with seasons, rivals, and F1 HUD
├── train.py                 # Interactive neuroevolution visualizer
├── train_zoro.py            # Specialized championship training pipeline
├── eval.py                  # AI performance evaluation & benchmark suite
├── requirements.txt         # Project dependencies
└── .github/workflows/ci.yml # Automated CI test pipeline
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11, 3.12, 3.13, or 3.14.
* **`pygame-ce`** (Pygame Community Edition) is used for modern SDL2 audio and high-DPI graphics.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Madhur-Sharma-64/racing-ai.git
   cd racing-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Play & Run

### 1. Drive by Hand or Race Against AI Rivals (`drive.py`)
Launch the 60 FPS driving simulator. Race solo with the stopwatch or press **`G`** to race against 4 autonomous AI champions simultaneously:
```bash
python drive.py
```

| Key | Action |
|---|---|
| `UP` / `W` | Accelerate / Gas |
| `DOWN` / `S` | Brake (activates rear brake LEDs) |
| `LEFT` / `A` | Steer Left |
| `RIGHT` / `D` | Steer Right |
| `E` | **Cycle Season / Weather** (Spring 🌸, Autumn 🍂, Cold Winter ❄️, Summer ☀️) |
| `M` | **Toggle Audio Mute** (Engine roar & tire squeal) |
| `G` | **Toggle 4 AI Rivals** (KEEPER, SPEED DEMON, MADX, and ZORO) |
| `SPACE` / `ENTER` | Start run (from start screen) |
| `LEFT` / `RIGHT` | Choose circuit (from start screen) |
| `P` | Pause / Resume |
| `R` | Restart lap (teleports car to starting grid) |
| `N` | Next track |
| `C` | Toggle chase camera (Car-Up vs. North-Up) |
| `Mouse Wheel` or `+` / `-` | Zoom camera in / out |
| `ESC` | Return to menu / Quit |

---

### 2. Interactive AI Training Visualizer (`train.py`)
Launch the genetic algorithm visualizer to watch a swarm of 60 cars evolve around any circuit:
```bash
python train.py
```

| Key | Action |
|---|---|
| `SPACE` | Toggle between **Visual 60 FPS** and **Fast Turbo Mode** (>20,000 steps/s) |
| `M` | Toggle Multi-Track Auto-Rotation (cycles tracks every generation) |
| `N` | Cycle through tracks manually |
| `S` | Save current all-time best model to `models/` |
| `L` | Load model weights from disk |
| `R` | Reset population to Generation 1 |
| `C` | Toggle camera (Car-Up vs. North-Up) |
| `ESC` | Exit training |

---

### 3. Specialized Champion Training Scripts
Train specialized driving models using tailored fitness functions:
```bash
# Train ZORO universal champion across multiple challenging circuits:
python train_zoro.py

# Train high-speed demon specialist:
python train_speed.py

# Train safe, collision-free driver:
python train_safe.py
```

---

## 🧪 Testing

Run the automated test suite using `pytest`:
```bash
pytest -v
```

All **37 unit tests** validate:
* Vehicle kinematics, drag curves, and lateral acceleration limits.
* Raycasting distance accuracy and forward radar detection.
* Sub-frame finish-line interpolation and lap timer anti-cheat logic.
* Track geometry, spline interpolation, and containment calculations.
* Neural network architectures, weight mutation, and genetic operators.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
