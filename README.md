# Racing AI 🏎️💨

[![CI](https://github.com/Madhur-Sharma-64/racing-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Madhur-Sharma-64/racing-ai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Engine](https://img.shields.io/badge/engine-NumPy%20%2B%20Pygame--CE-green)
![Tests](https://img.shields.io/badge/tests-27%20passed-success)

A high-performance 2D racing physics simulation and autonomous driving environment. Built from scratch in Python and NumPy, featuring a kinematic vehicle model, vectorized raycast distance sensors, real-world track splines, and headless training environments capable of simulating **over 20,000 steps per second**.

Designed as a software engineering portfolio project showcasing clean simulation architecture, spatial algorithm optimization, and an AI training framework.

---

## 🌟 Highlights

* **Realistic 2D Vehicle Dynamics**: Kinematic bicycle model with longitudinal gas/brake curves, quadratic aerodynamic drag, rolling resistance, and a dynamic lateral grip threshold ($25\,\text{m/s}^2$) enforcing authentic cornering understeer.
* **Vectorized Raycasting Engine**: 5-direction obstacle/boundary distance sensors (`[-60°, -30°, 0°, +30°, +60°]`) optimized with spatial culling and simultaneous multi-ray intersection math.
* **Real-World & Procedural Tracks**: Simplified real-world circuit outlines (Monza, Silverstone, Spa-Francorchamps, Interlagos, Speedway Oval) generated via centripetal Catmull-Rom splines, plus procedural harmonic loop generation.
* **Gym-Compatible Headless Environments**:
  * `RaceEnv`: Single-agent environment implementing standard `reset()` and `step(action)` interfaces for Reinforcement Learning.
  * `MultiCarEnv`: Batched vector environment simulating populations of $N$ cars concurrently for Genetic Algorithms and Neuroevolution.
* **Decoupled Architecture**: Pure NumPy physics and logic separated from Pygame rendering routines (`ui/render.py`), enabling high-speed headless training without display overhead.
* **Playable Human Driving Simulator**: Interactive track carousel, chase camera (car-up and north-up modes), real-time HUD telemetry, and minimap.

---

## ⚡ Performance & Benchmarks

Simulating thousands of car-track interactions during AI training requires extreme simulation throughput. Raycasting and centerline queries are accelerated using NumPy vectorization and spatial distance culling:

| Component | Baseline | Optimized | Speedup |
|---|---|---|---|
| **Raycasting (5 rays, 1,450+ segments)** | `0.199 ms` | `0.039 ms` | **~5.1× faster** |
| **Track Centerline & Containment Query** | `0.012 ms` | `0.005 ms` | **~2.4× faster** |
| **Full Simulation Step (Physics + Rays + Lap)** | `0.550 ms` | `0.049 ms` | **~11.2× faster** |
| **Simulation Throughput (Single CPU Core)** | ~1,800 steps/s | **>20,000 steps/s** | **~11.2× faster** |

> *A full 60-second lap (1,200 simulation steps at $dt = 0.05\,\text{s}$) runs in under **0.06 seconds** in headless mode.*

---

## 🏗️ Project Architecture

```
racing-ai/
├── sim/                     # Core simulation & physics (pure NumPy)
│   ├── car.py               # Vehicle model (bicycle kinematics & grip limits)
│   ├── track.py             # Track representation, centerline, progress & containment
│   ├── tracks.py            # Real-world circuit definitions & spline interpolation
│   ├── raycast.py           # Vectorized 2D ray-segment distance sensors
│   ├── lap.py               # Anti-cheat lap timer and progress tracking
│   └── env.py               # Headless RL (RaceEnv) & batch GA (MultiCarEnv) environments
├── ui/                      # Presentation layer
│   ├── camera.py            # Coordinate transformations (world meters -> screen pixels)
│   └── render.py            # Pygame drawing routines (track, cars, rays, HUD, minimap)
├── ai/                      # AI controllers (neural networks & evolutionary training)
│   ├── network.py           # Neural network policy architecture
│   └── evolution.py         # Genetic algorithm & population management
├── tests/                   # Automated test suite
│   ├── test_car.py          # Physics and grip limit tests
│   ├── test_camera.py       # Viewport projection and camera rotation tests
│   ├── test_lap.py          # Lap counting and anti-cheat verification
│   ├── test_raycast.py      # Wall intersection and sensor distance tests
│   ├── test_tracks.py       # Geometric circuit validation & length scaling
│   └── test_env.py          # Headless environment reset and step tests
├── drive.py                 # Playable human driving simulator with menu and HUD
├── main.py                  # Track and raycast interactive debug visualizer
├── requirements.txt         # Project dependencies
└── .github/workflows/ci.yml # Automated CI test pipeline
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11, 3.12, 3.13, or 3.14.
* On Python 3.14+, **`pygame-ce`** (Pygame Community Edition) is used for official compatibility.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Madhur-Sharma-64/racing-ai.git
   cd racing-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Run

### 1. Multi-Track Championship Trainer (`train_all.py`)
Train a single generalized neural network champion across **all 5 circuits** (Speedway Oval, Monza, Silverstone, Spa-Francorchamps, and Interlagos) in headless high-speed mode:
```bash
# Default: 30 generations across all 5 circuits
python train_all.py

# Custom generations, population, and simulation steps
python train_all.py --generations 50 --pop-size 60 --steps 800
```
This automatically saves the all-around champion driver to `models/best_driver.npz`.

---

### 2. Interactive AI Training Visualizer (`train.py`)
Launch the neuroevolution visualizer to watch a swarm of 60 cars learn to navigate race tracks:
```bash
python train.py
```

| Key | Action |
|---|---|
| `SPACE` | Toggle between **Visual 60 FPS** and **Fast Turbo Mode** (20,000+ steps/s) |
| `M` | **Toggle Multi-Track Auto-Rotation** (cycles tracks every generation) |
| `N` | Manually cycle through tracks (Oval, Monza, Silverstone, Spa, Interlagos, Random) |
| `S` | Save current all-time best model to `models/best_driver.npz` |
| `L` | Load seed model from `models/best_driver.npz` |
| `R` | Reset population to Generation 1 |
| `C` | Toggle chase camera (Car-Up vs. North-Up) |
| `Mouse Wheel` or `+` / `-` | Zoom camera in / out |
| `ESC` | Exit training |

---

### 3. Drive by Hand or Race Against AI Rival (`drive.py`)
Launch the human simulator. If a trained model exists at `models/best_driver.npz`, press **`G`** to spawn an autonomous AI rival on the track:
```bash
python drive.py
```

| Key | Action |
|---|---|
| `UP` / `W` | Accelerate / Gas |
| `DOWN` / `S` | Brake |
| `LEFT` / `A` | Steer Left |
| `RIGHT` / `D` | Steer Right |
| `G` | **Toggle AI Rival / Ghost Car** (races alongside you in cyan) |
| `SPACE` / `ENTER` | Start run (from menu) |
| `LEFT` / `RIGHT` | Select track (from menu) |
| `P` | Pause / Resume |
| `R` | Restart current lap |
| `N` | Next track |
| `C` | Toggle camera (Car-Up vs. North-Up) |
| `Mouse Wheel` or `+` / `-` | Zoom camera in / out |
| `ESC` | Return to menu / Quit |

---

### 3. Interactive Raycaster Debugger (`main.py`)
Visualizes 5-direction raycast sensors and track containment in real-time:
```bash
python main.py
```
* **Mouse**: Move car position.
* **LEFT / RIGHT Arrows**: Rotate heading angle.
* **N**: Generate a new random harmonic track.

---

## 🧪 Testing

Run the automated test suite using `pytest`:
```bash
pytest -v
```

All 27 unit tests validate vehicle physics, camera projections, track geometry, raycasting accuracy, lap progression, and environment stepping.

---

## 🛣️ AI Roadmap & Milestones

- [x] **Milestone 1**: 2D vehicle physics engine with tire grip limits.
- [x] **Milestone 2**: Vectorized raycast distance sensors and real-world circuit generation.
- [x] **Milestone 3**: Simulation speed optimization (>20,000 steps/s) and headless environments.
- [x] **Milestone 4**: Feedforward neural network policy in `ai/network.py`.
- [x] **Milestone 5**: Neuroevolution / Genetic Algorithm training loop and visualizer (`train.py`).
- [ ] **Milestone 6**: Reinforcement Learning (PPO) integration using Stable-Baselines3.
- [ ] **Milestone 7**: Optimal racing line computation and telemetry comparison against AI lap times.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
