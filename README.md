# Adaptive Traffic Light Control with Multi-Agent RL

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Training a Multi‑Agent Policy](#training-a-multi‑agent-policy)
  - [Evaluating a Trained Policy](#evaluating-a-trained-policy)
  - [Visualizing Traffic Flow](#visualizing-traffic-flow)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This repository implements an adaptive traffic‑signal control system using **multi‑agent reinforcement learning (MARL)**. Each intersection in a simulated road network is modeled as an independent agent that learns to select phase timings based on local traffic conditions. The agents are trained jointly using **Ray RLlib** with a shared policy architecture built on **PyTorch**. The environment follows the **Gymnasium** API, enabling easy integration with other RL libraries.

The goal is to reduce average vehicle waiting time, improve throughput, and demonstrate scalability to larger networks.

## Key Features
- **Modular Gymnasium environment** (`traffic_env/`) that simulates realistic traffic dynamics using a cellular automaton model.
- **Multi‑agent RL** support via Ray RLlib with configurable algorithms (PPO, A2C, DQN, etc.).
- **Shared neural network policy** implemented in PyTorch for parameter efficiency across agents.
- **Comprehensive logging** (TensorBoard, CSV) and checkpointing.
- **Visualization tools** (Matplotlib) for real‑time traffic flow, queue lengths, and reward curves.
- **Configurable via YAML** – experiment parameters, network topology, and training hyper‑parameters are all externalized.
- **Unit and integration tests** using `pytest` to ensure reproducibility.

## Installation
> **Prerequisites**: Python 3.9 – 3.12, `pip`, and a C++ compiler for Ray.

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-traffic-rl.git
cd adaptive-traffic-rl

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install core dependencies
pip install -r requirements.txt

# Install the package in editable mode (includes the custom Gymnasium env)
pip install -e .
```

`requirements.txt` includes:
```
gymnasium>=0.28.0
torch>=2.0.0
ray[rllib]>=2.9.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
tensorboard>=2.13.0
pytest>=7.0.0
```

## Quick Start
The following commands train a PPO policy on a 4‑intersection grid for 1 M timesteps and then evaluate the learned policy.

```bash
# Train
python scripts/train.py --config configs/ppo_grid4.yaml

# Evaluate (uses the latest checkpoint automatically)
python scripts/evaluate.py --checkpoint logs/ppo_grid4/checkpoint_000500

# Plot training curves
python scripts/plot_metrics.py --logdir logs/ppo_grid4
```

All scripts accept `--help` for additional arguments.

## Usage Guide

### Training a Multi‑Agent Policy
```bash
python scripts/train.py \
    --config configs/ppo_grid4.yaml \
    --log-dir logs/ppo_grid4 \
    --stop-iters 1000 \
    --stop-timesteps 1_000_000 \
    --stop-reward 200.0
```

**Important CLI arguments**

| Argument | Description |
|----------|-------------|
| `--config` | Path to a YAML file that defines the environment, algorithm, and hyper‑parameters. |
| `--log-dir` | Directory where checkpoints, TensorBoard logs, and CSV metrics are saved. |
| `--stop-iters` | Maximum number of training iterations. |
| `--stop-timesteps` | Maximum number of environment timesteps. |
| `--stop-reward` | Early‑stop when the average episode reward exceeds this value. |

The YAML configuration (`configs/ppo_grid4.yaml`) contains sections such as:

```yaml
env:
  name: traffic_env:MultiIntersectionEnv
  config:
    grid_size: [2, 2]          # 2x2 grid = 4 intersections
    max_queue: 20
    episode_length: 3600       # 1 hour simulated time
    seed: 42

algorithm: PPO
algorithm_config:
  gamma: 0.99
  lr: 5e-4
  train_batch_size: 4000
  rollout_fragment_length: 200
  num_sgd_iter: 10
  vf_clip_param: 10.0
  entropy_coeff: 0.01
  clip_param: 0.2
  model:
    fcnet_hiddens: [256, 256]
    activation: relu
```

### Evaluating a Trained Policy
```bash
python scripts/evaluate.py \
    --checkpoint logs/ppo_grid4/checkpoint_000500 \
    --episodes 20 \
    --render
```

- `--render` launches a Matplotlib animation of traffic flow for each episode.
- Results (average waiting time, total throughput, etc.) are printed to the console and saved as `evaluation.json` in the checkpoint directory.

### Visualizing Traffic Flow
The repository provides two visualizers:

1. **Real‑time animation** (used with `--render` above).  
2. **Post‑hoc heatmaps** of queue lengths per intersection:

```bash
python scripts/plot_queues.py \
    --logdir logs/ppo_grid4 \
    --episode 10
```

Outputs a PNG file `queues_episode_10.png` showing the spatial distribution of queues at each timestep.

## Project Structure
```
adaptive-traffic-rl/
│
├─ traffic_env/                 # Custom Gymnasium multi‑agent environment
│   ├─ __init__.py
│   ├─ multi_intersection.py    # Core environment implementation
│   └─ utils.py                 # Helper functions (e.g., traffic generation)
│
├─ scripts/                     # CLI entry points
│   ├─ train.py                 # Training wrapper using Ray RLlib
│   ├─ evaluate.py              # Evaluation loop with optional rendering
│   ├─ plot_metrics.py          # TensorBoard CSV → Matplotlib plots
│   └─ plot_queues.py           # Queue‑heatmap visualizer
│
├─ configs/                     # YAML experiment specifications
│   ├─ ppo_grid4.yaml
│   └─ dqn_grid4.yaml
│
├─ tests/                       # Unit and integration tests
│   ├─ test_env.py
│   └─ test_training.py
│
├─ logs/                        # Auto‑generated training logs (git‑ignored)
│
├─ requirements.txt
├─ setup.cfg
├─ pyproject.toml
└─ README.md                    # ← This file
```

## Configuration
All hyper‑parameters are stored in YAML files under `configs/`. The schema is documented in `docs/config_schema.md`. Changing the road network is as simple as editing the `grid_size` or providing a custom adjacency matrix via `env.config.custom_topology`.

## Testing
Run the full test suite with:

```bash
pytest -v
```

Key tests include:

- **Environment sanity** (`test_env.py`): verifies reset/step signatures, observation shapes, and reward calculation.
- **Training loop stability** (`test_training.py`): launches a short PPO run (10 k timesteps) and asserts that the average reward improves monotonically.

All tests are designed to run in < 2 minutes on a standard laptop.

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome‑change`).
3. Ensure code style compliance with `ruff` (`ruff check .`).
4. Add or update tests as needed.
5. Submit a pull request with a clear description of the change.

See `CONTRIBUTING.md` for detailed guidelines.

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.