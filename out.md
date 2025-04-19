File: .dmypy.json
{"pid": 36032, "connection_name": "\\\\.\\pipe\\dmypy-dmR6cCdf.pipe"}


File: .python-version
3.10.13


File: LICENSE
MIT License

Copyright (c) 2025 Luis Guilherme P. M.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File: MANIFEST.in
# File: MANIFEST.in
include README.md
include LICENSE
include requirements.txt
graft muzerotriangle
graft tests
include .python-version
include pyproject.toml
global-exclude __pycache__
global-exclude *.py[co]

File: out.md


File: pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "muzerotriangle"
version = "0.4.0"
authors = [{ name="Luis Guilherme P. M.", email="lgpelin92@gmail.com" }]
description = "AlphaZero implementation for a triangle puzzle game."
readme = "README.md"
license = { file="LICENSE" }
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Games/Entertainment :: Puzzle Games",
    "Development Status :: 3 - Alpha",
]
dependencies = [
    "pygame>=2.1.0",
    "numpy>=1.20.0",
    "torch>=2.0.0",
    "torchvision>=0.11.0",
    "cloudpickle>=2.0.0",
    "numba>=0.55.0",
    "mlflow>=1.20.0",
    "matplotlib>=3.5.0",
    "ray>=2.8.0",
    "pydantic>=2.0.0",
    "typing_extensions>=4.0.0",
    "typer[all]>=0.9.0", # Added typer for CLI
]

[project.urls]
"Homepage" = "https://github.com/lguibr/muzerotriangle"
"Bug Tracker" = "https://github.com/lguibr/muzerotriangle/issues"

[project.scripts]
muzerotriangle = "muzerotriangle.cli:app"

[tool.setuptools.packages.find]
# No 'where' needed, find searches from the project root by default
# It will find the 'muzerotriangle' directory now.


[tool.setuptools.package-data]
"*" = ["*.txt", "*.md", "*.json"] # Include non-code files

# --- Tool Configurations (Optional but Recommended) ---

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "C4", "ARG", "SIM", "TCH", "PTH", "NPY"]
ignore = ["E501"] # Ignore line length errors if needed selectively

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true # Start with true, gradually reduce
# Add specific module ignores if necessary
# [[tool.mypy.overrides]]
# module = "some_missing_types_module.*"
# ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=muzerotriangle --cov-report=term-missing" # Point coverage to the new package dir
testpaths = [
    "tests",
]

[tool.coverage.run]
omit = [
    "muzerotriangle/cli.py", # Exclude CLI from coverage for now
    "muzerotriangle/visualization/*", # Exclude visualization for now
    "muzerotriangle/app.py",
    "run_*.py",
    "muzerotriangle/training/logging_utils.py", # Logging utils can be hard to cover fully
    "muzerotriangle/config/*", # Config models are mostly declarative
    "muzerotriangle/data/schemas.py",
    "muzerotriangle/structs/*",
    "muzerotriangle/utils/types.py",
    "muzerotriangle/mcts/core/types.py",
    "muzerotriangle/rl/types.py",
    "*/__init__.py",
    "*/README.md",
]

[tool.coverage.report]
fail_under = 28 # Set a reasonable initial coverage target
show_missing = true

File: README.md

[![CI/CD Status](https://github.com/lguibr/muzerotriangle/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/lguibr/muzerotriangle/actions/workflows/ci_cd.yml) - [![codecov](https://codecov.io/gh/lguibr/muzerotriangle/graph/badge.svg?token=YOUR_CODECOV_TOKEN_HERE)](https://codecov.io/gh/lguibr/muzerotriangle) - [![PyPI version](https://badge.fury.io/py/muzerotriangle.svg)](https://badge.fury.io/py/muzerotriangle)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) - [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# AlphaTriangle (MuZero Implementation)

<img src="bitmap.png" alt="AlphaTriangle Logo" width="300"/>

## Overview

AlphaTriangle is a project implementing an artificial intelligence agent based on **MuZero** principles to learn and play a custom puzzle game involving placing triangular shapes onto a grid. The agent learns through self-play reinforcement learning, guided by Monte Carlo Tree Search (MCTS) and a deep neural network (PyTorch).

The project includes:

*   A playable version of the triangle puzzle game using Pygame.
*   An implementation of the MCTS algorithm tailored for the game and MuZero's latent space search.
*   A deep neural network (representation, dynamics, and prediction functions) implemented in PyTorch, featuring convolutional layers and optional Transformer Encoder layers.
*   A reinforcement learning pipeline coordinating **parallel self-play (using Ray)**, data storage (trajectories), and network training (sequence-based), managed by the `muzerotriangle.training` module. **Supports N-step returns and Prioritized Experience Replay (PER).**
*   Visualization tools for interactive play, debugging, and monitoring training progress (**with near real-time plot updates**).
*   Experiment tracking using MLflow.
*   Unit tests for core components.
*   A command-line interface for easy execution.

## Core Technologies

*   **Python 3.10+**
*   **Pygame:** For game visualization and interactive modes.
*   **PyTorch:** For the deep learning model (CNNs, optional Transformers, Distributional Value/Reward Heads) and training, with CUDA/MPS support.
*   **NumPy:** For numerical operations, especially state representation and MCTS/buffer interactions.
*   **Ray:** For parallelizing self-play data generation and statistics collection across multiple CPU cores/processes.
*   **Numba:** (Optional, used in `features.grid_features`) For performance optimization of specific grid calculations.
*   **Cloudpickle:** For serializing the experience replay buffer (trajectories) and training checkpoints.
*   **MLflow:** For logging parameters, metrics, and artifacts (checkpoints, buffers) during training runs.
*   **Pydantic:** For configuration management and data validation.
*   **Typer:** For the command-line interface.
*   **Pytest:** For running unit tests.

## Project Structure

```markdown
.
├── .github/workflows/      # GitHub Actions CI/CD
│   └── ci_cd.yml
├── .muzerotriangle_data/    # Root directory for ALL persistent data (GITIGNORED)
│   ├── mlruns/             # MLflow tracking data
│   └── runs/               # Stores temporary/local artifacts per run
│       └── <run_name>/
│           ├── checkpoints/
│           ├── buffers/
│           ├── logs/
│           └── configs.json
├── muzerotriangle/          # Source code for the project package
│   ├── __init__.py
│   ├── app.py
│   ├── cli.py              # CLI logic
│   ├── config/             # Pydantic configuration models
│   │   └── README.md
│   ├── data/               # Data saving/loading logic (MuZero format)
│   │   └── README.md
│   ├── environment/        # Game rules, state, actions
│   │   └── README.md
│   ├── features/           # Feature extraction logic
│   │   └── README.md
│   ├── interaction/        # User input handling
│   │   └── README.md
│   ├── mcts/               # Monte Carlo Tree Search (MuZero adaptation)
│   │   └── README.md
│   ├── nn/                 # Neural network definition (MuZero) and wrapper
│   │   └── README.md
│   ├── rl/                 # RL components (Trainer, Buffer, Worker - MuZero)
│   │   └── README.md
│   ├── stats/              # Statistics collection and plotting
│   │   └── README.md
│   ├── structs/            # Core data structures (Triangle, Shape)
│   │   └── README.md
│   ├── training/           # Training orchestration (Loop, Setup, Runners)
│   │   └── README.md
│   ├── utils/              # Shared utilities and types (MuZero types)
│   │   └── README.md
│   └── visualization/      # Pygame rendering components
│       └── README.md
├── tests/                  # Unit tests
│   ├── ...
├── .gitignore
├── .python-version
├── LICENSE                 # License file (MIT)
├── MANIFEST.in             # Specifies files for source distribution
├── pyproject.toml          # Build system & package configuration
├── README.md               # This file
├── requirements.txt        # List of dependencies (also in pyproject.toml)
├── run_interactive.py      # Legacy script to run interactive modes
├── run_shape_editor.py     # Script to run the interactive shape definition tool
├── run_training_headless.py # Legacy script for headless training
└── run_training_visual.py  # Legacy script for visual training
```

## Key Modules (`muzerotriangle`)

*   **`cli`:** Defines the command-line interface using Typer. ([`muzerotriangle/cli.py`](muzerotriangle/cli.py))
*   **`config`:** Centralized Pydantic configuration classes. ([`muzerotriangle/config/README.md`](muzerotriangle/config/README.md))
*   **`structs`:** Defines core, low-level data structures (`Triangle`, `Shape`) and constants. ([`muzerotriangle/structs/README.md`](muzerotriangle/structs/README.md))
*   **`environment`:** Defines the game rules, `GameState`, action encoding/decoding, and grid/shape *logic*. ([`muzerotriangle/environment/README.md`](muzerotriangle/environment/README.md))
*   **`features`:** Contains logic to convert `GameState` objects into numerical features (`StateType`). ([`muzerotriangle/features/README.md`](muzerotriangle/features/README.md))
*   **`nn`:** Contains the PyTorch `nn.Module` definition (`MuZeroNet`) and a wrapper class (`NeuralNetwork`). ([`muzerotriangle/nn/README.md`](muzerotriangle/nn/README.md))
*   **`mcts`:** Implements the Monte Carlo Tree Search algorithm (`Node`, `run_mcts_simulations`), adapted for MuZero. ([`muzerotriangle/mcts/README.md`](muzerotriangle/mcts/README.md))
*   **`rl`:** Contains RL components: `Trainer` (network updates), `ExperienceBuffer` (trajectory storage, **supports PER**), and `SelfPlayWorker` (Ray actor for parallel self-play, **calculates N-step returns**). ([`muzerotriangle/rl/README.md`](muzerotriangle/rl/README.md))
*   **`training`:** Orchestrates the training process using `TrainingLoop`, managing workers, data flow, logging, and checkpoints. Includes `runners.py` for callable training functions. ([`muzerotriangle/training/README.md`](muzerotriangle/training/README.md))
*   **`stats`:** Contains the `StatsCollectorActor` (Ray actor) for asynchronous statistics collection and the `Plotter` class for rendering plots. ([`muzerotriangle/stats/README.md`](muzerotriangle/stats/README.md))
*   **`visualization`:** Uses Pygame to render the game state, previews, HUD, plots, etc. `DashboardRenderer` handles the training visualization layout. ([`muzerotriangle/visualization/README.md`](muzerotriangle/visualization/README.md))
*   **`interaction`:** Handles keyboard/mouse input for interactive modes via `InputHandler`. ([`muzerotriangle/interaction/README.md`](muzerotriangle/interaction/README.md))
*   **`data`:** Manages saving and loading of training artifacts (`DataManager`) using Pydantic schemas and `cloudpickle`. ([`muzerotriangle/data/README.md`](muzerotriangle/data/README.md))
*   **`utils`:** Provides common helper functions, shared type definitions (including MuZero types), and geometry helpers. ([`muzerotriangle/utils/README.md`](muzerotriangle/utils/README.md))
*   **`app`:** Integrates components for interactive modes (`run_interactive.py`). ([`muzerotriangle/app.py`](muzerotriangle/app.py))

## Setup

1.  **Clone the repository (for development):**
    ```bash
    git clone https://github.com/lguibr/muzerotriangle.git
    cd muzerotriangle
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the package:**
    *   **For users:**
        ```bash
        pip install muzerotriangle # Or pip install git+https://github.com/lguibr/muzerotriangle.git
        ```
    *   **For developers (editable install):**
        ```bash
        pip install -e .
        # Install dev dependencies (optional, for running tests/linting)
        pip install pytest pytest-cov pytest-mock ruff mypy codecov twine build
        ```
    *Note: Ensure you have the correct PyTorch version installed for your system (CPU/CUDA/MPS). See [pytorch.org](https://pytorch.org/). Ray may have specific system requirements.*
4.  **(Optional) Add data directory to `.gitignore`:**
    Create or edit the `.gitignore` file in your project root and add the line:
    ```
    .muzerotriangle_data/
    ```

## Running the Code (CLI)

Use the `muzerotriangle` command:

*   **Show Help:**
    ```bash
    muzerotriangle --help
    ```
*   **Interactive Play Mode:**
    ```bash
    muzerotriangle play [--seed 42] [--log-level INFO]
    ```
*   **Interactive Debug Mode:**
    ```bash
    muzerotriangle debug [--seed 42] [--log-level DEBUG]
    ```
*   **Run Training (Visual Mode):**
    ```bash
    muzerotriangle train [--seed 42] [--log-level INFO]
    ```
*   **Run Training (Headless Mode):**
    ```bash
    muzerotriangle train --headless [--seed 42] [--log-level INFO]
    # or
    muzerotriangle train -H [--seed 42] [--log-level INFO]
    ```
*   **Shape Editor (Run directly):**
    ```bash
    python run_shape_editor.py
    ```
*   **Monitoring Training (MLflow UI):**
    While training (headless or visual), or after runs have completed, open a separate terminal in the project root and run:
    ```bash
    mlflow ui --backend-store-uri file:./.muzerotriangle_data/mlruns
    ```
    Then navigate to `http://localhost:5000` (or the specified port) in your browser.
*   **Running Unit Tests (Development):**
    ```bash
    pytest tests/
    ```

## Configuration

All major parameters are defined in the Pydantic classes within the `muzerotriangle/config/` directory. Modify these files to experiment with different settings. The `muzerotriangle/config/validation.py` script performs basic checks on startup.

## Data Storage

All persistent data, including MLflow tracking data and run-specific artifacts, is stored within the `.muzerotriangle_data/` directory in the project root, managed by the `DataManager` and MLflow.

## Maintainability

This project includes README files within each major `muzerotriangle` submodule. **Please keep these READMEs updated** when making changes to the code's structure, interfaces, or core logic.

File: requirements.txt
pygame>=2.1.0
numpy>=1.20.0
torch>=2.0.0
torchvision>=0.11.0
cloudpickle>=2.0.0
numba>=0.55.0
mlflow>=1.20.0
matplotlib>=3.5.0
ray>=2.8.0
pydantic>=2.0.0
typing_extensions>=4.0.0
pytest>=7.0.0
pytest-mock>=3.0.0
typer[all]>=0.9.0
pytest-cov>=3.0.0

File: .muzerotriangle_data\mlruns\753884796909271804\meta.yaml
artifact_location: file:///C:/Users/lgui_/lab/muzerotriangle/.muzerotriangle_data/mlruns/753884796909271804
creation_time: 1745058303933
experiment_id: '753884796909271804'
last_update_time: 1745058303933
lifecycle_stage: active
name: AlphaTriangle


File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\meta.yaml
artifact_uri: file:///C:/Users/lgui_/lab/muzerotriangle/.muzerotriangle_data/mlruns/753884796909271804/cd3104731f064853bd3c04cbb570febc/artifacts
end_time: 1745058623396
entry_point_name: ''
experiment_id: '753884796909271804'
lifecycle_stage: active
run_id: cd3104731f064853bd3c04cbb570febc
run_name: train_muzero_20250419_072454
run_uuid: cd3104731f064853bd3c04cbb570febc
source_name: ''
source_type: 4
source_version: ''
start_time: 1745058304126
status: 3
tags: []
user_id: lgui_


File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\ACTION_DIM
360

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\ACTION_ENCODING_DIM
16

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\ACTIVATION_FUNCTION
ReLU

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\AUTO_RESUME_LATEST
True

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\BATCH_SIZE
64

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\BEST_CHECKPOINT_FILENAME
best.pkl

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\BUFFER_CAPACITY
100000

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\BUFFER_FILENAME
buffer.pkl

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\BUFFER_SAVE_DIR_NAME
buffers

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\BUFFER_SAVE_FREQ_STEPS
10

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CHECKPOINT_SAVE_DIR_NAME
checkpoints

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CHECKPOINT_SAVE_FREQ_STEPS
2500

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\COLS
15

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\COLS_PER_ROW
[9, 11, 13, 15, 15, 13, 11, 9]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\COMPILE_MODEL
True

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CONFIG_FILENAME
configs.json

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CONV_FILTERS
[32, 64, 128]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CONV_KERNEL_SIZES
[3, 3, 3]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CONV_PADDING
[1, 1, 1]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\CONV_STRIDES
[1, 1, 1]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\DEVICE
auto

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\dirichlet_alpha
0.3

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\dirichlet_epsilon
0.25

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\DISCOUNT
0.99

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\GAME_STATE_SAVE_DIR_NAME
game_states

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\GRADIENT_CLIP_VALUE
5.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\HIDDEN_STATE_DIM
128

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LATEST_CHECKPOINT_FILENAME
latest.pkl

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LEARNING_RATE
0.0001

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LOAD_BUFFER_PATH
None

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LOAD_CHECKPOINT_PATH
None

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LOG_DIR_NAME
logs

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LR_SCHEDULER_ETA_MIN
1e-06

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LR_SCHEDULER_TYPE
CosineAnnealingLR

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\LR_SCHEDULER_T_MAX
100000

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\max_search_depth
10

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\MAX_TRAINING_STEPS
100000

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\MIN_BUFFER_SIZE_TO_TRAIN
10000

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\MLFLOW_DIR_NAME
mlruns

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\MLFLOW_TRACKING_URI
file:///C:/Users/lgui_/lab/muzerotriangle/.muzerotriangle_data/mlruns

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\model_total_params
2780512

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\model_trainable_params
2780512

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\NUM_SELF_PLAY_WORKERS
18

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\num_simulations
50

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\NUM_VALUE_ATOMS
51

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\N_STEP_RETURNS
10

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\OPTIMIZER_TYPE
AdamW

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\OTHER_NN_INPUT_FEATURES_DIM
30

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\PENALTY_GAME_OVER
-10.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\PER_ALPHA
0.6

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\PER_BETA_ANNEAL_STEPS
100000

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\PER_BETA_FINAL
1.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\PER_BETA_INITIAL
0.4

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\PER_EPSILON
1e-05

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\POLICY_HEAD_DIMS
[64]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\POLICY_LOSS_WEIGHT
1.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\puct_coefficient
1.5

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\RANDOM_SEED
42

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REP_FC_DIMS_AFTER_ENCODER
[]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REP_TRANSFORMER_FC_DIM
256

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\RESIDUAL_BLOCK_FILTERS
128

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REWARD_HEAD_DIMS
[64]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REWARD_LOSS_WEIGHT
1.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REWARD_PER_CLEARED_TRIANGLE
0.5

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REWARD_PER_PLACED_TRIANGLE
0.01

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REWARD_PER_STEP_ALIVE
0.005

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\REWARD_SUPPORT_SIZE
21

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\ROOT_DATA_DIR
.muzerotriangle_data

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\RUNS_DIR_NAME
runs

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\RUN_NAME
train_muzero_20250419_072454

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\SAVE_BUFFER
True

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\SAVE_GAME_STATES
False

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\temperature_anneal_steps
100

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\temperature_final
0.1

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\temperature_initial
1.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\training_status
INTERRUPTED

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\USE_BATCH_NORM
True

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\USE_PER
True

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\USE_TRANSFORMER_IN_REP
False

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\VALUE_HEAD_DIMS
[64]

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\VALUE_LOSS_WEIGHT
0.25

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\VALUE_MAX
10.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\VALUE_MIN
-10.0

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\WEIGHT_DECAY
0.0001

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\WORKER_DEVICE
cpu

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\params\WORKER_UPDATE_FREQ_STEPS
500

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\tags\mlflow.runName
train_muzero_20250419_072454

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\tags\mlflow.source.name
C:\Users\lgui_\AppData\Local\Programs\Python\Python310\Scripts\muzerotriangle

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\tags\mlflow.source.type
LOCAL

File: .muzerotriangle_data\mlruns\753884796909271804\cd3104731f064853bd3c04cbb570febc\tags\mlflow.user
lgui_

File: .ruff_cache\CACHEDIR.TAG
Signature: 8a477f597d28d172789f06886806bc55

File: muzerotriangle\app.py
import logging

import pygame

from . import (
    config,
    environment,
    interaction,
    visualization,
)

logger = logging.getLogger(__name__)


class Application:
    """Main application integrating visualization and interaction."""

    def __init__(self, mode: str = "play"):
        self.vis_config = config.VisConfig()
        self.env_config = config.EnvConfig()
        self.mode = mode

        pygame.init()
        pygame.font.init()
        self.screen = self._setup_screen()
        self.clock = pygame.time.Clock()
        self.fonts = visualization.load_fonts()

        if self.mode in ["play", "debug"]:
            # Create GameState first
            self.game_state = environment.GameState(self.env_config)
            # Create Visualizer
            self.visualizer = visualization.Visualizer(
                self.screen, self.vis_config, self.env_config, self.fonts
            )
            # Create InputHandler, passing GameState and Visualizer
            self.input_handler = interaction.InputHandler(
                self.game_state, self.visualizer, self.mode, self.env_config
            )
        else:
            # Handle other modes or raise error if necessary
            logger.error(f"Unsupported application mode: {self.mode}")
            raise ValueError(f"Unsupported application mode: {self.mode}")

        self.running = True

    def _setup_screen(self) -> pygame.Surface:
        """Initializes the Pygame screen."""
        screen = pygame.display.set_mode(
            (self.vis_config.SCREEN_WIDTH, self.vis_config.SCREEN_HEIGHT),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(f"{config.APP_NAME} - {self.mode.capitalize()} Mode")
        return screen

    def run(self):
        """Main application loop."""
        logger.info(f"Starting application in {self.mode} mode.")
        while self.running:
            # dt = ( # Unused variable
            #     self.clock.tick(self.vis_config.FPS) / 1000.0
            # )  # Delta time (unused currently)
            self.clock.tick(self.vis_config.FPS)  # Still tick the clock

            # Handle Input using InputHandler
            if self.input_handler:
                self.running = self.input_handler.handle_input()
                if not self.running:
                    break  # Exit loop if handle_input returns False
            else:
                # Fallback event handling if input_handler is not initialized (should not happen in play/debug)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Basic resize handling needed even without input handler
                    # Combine nested if statements
                    if event.type == pygame.VIDEORESIZE and self.visualizer:
                        try:
                            w, h = max(320, event.w), max(240, event.h)
                            # Update visualizer's screen reference
                            self.visualizer.screen = pygame.display.set_mode(
                                (w, h), pygame.RESIZABLE
                            )
                            # Invalidate visualizer's layout cache
                            self.visualizer.layout_rects = None
                        except pygame.error as e:
                            logger.error(f"Error resizing window: {e}")
                if not self.running:
                    break

            # Render using Visualizer
            if (
                self.mode in ["play", "debug"]
                and self.visualizer
                and self.game_state
                and self.input_handler
            ):
                # Get interaction state needed for rendering from InputHandler
                interaction_render_state = (
                    self.input_handler.get_render_interaction_state()
                )
                # Pass game state, mode, and interaction state to visualizer
                self.visualizer.render(
                    self.game_state,
                    self.mode,
                    **interaction_render_state,  # Unpack the dict as keyword arguments
                )
                pygame.display.flip()  # Update the full display

        logger.info("Application loop finished.")
        pygame.quit()


File: muzerotriangle\cli.py
# File: muzerotriangle/cli.py
import logging
import sys
from typing import Annotated

import typer

from muzerotriangle import config, utils
from muzerotriangle.app import Application
from muzerotriangle.config import (
    MCTSConfig,
    PersistenceConfig,
    TrainConfig,
)

# --- REVERTED: Import from the re-exporting runners.py ---
from muzerotriangle.training.runners import (
    run_training_headless_mode,
    run_training_visual_mode,
)

# --- END REVERTED ---

app = typer.Typer(
    name="muzerotriangle",
    help="AlphaZero implementation for a triangle puzzle game.",
    add_completion=False,
)

LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level",
        "-l",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        case_sensitive=False,
    ),
]

SeedOption = Annotated[
    int,
    typer.Option(
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
]


def setup_logging(log_level_str: str):
    """Configures root logger based on string level."""
    log_level_str = log_level_str.upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override existing config
    )
    logging.getLogger("ray").setLevel(logging.WARNING)  # Keep Ray less verbose
    logging.getLogger("matplotlib").setLevel(
        logging.WARNING
    )  # Keep Matplotlib less verbose
    logging.info(f"Root logger level set to {logging.getLevelName(log_level)}")


def run_interactive_mode(mode: str, seed: int, log_level: str):
    """Runs the interactive application."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)  # Get logger after setup
    logger.info(f"Running in {mode.capitalize()} mode...")
    utils.set_random_seeds(seed)

    mcts_config = MCTSConfig()
    config.print_config_info_and_validate(mcts_config)

    try:
        app_instance = Application(mode=mode)
        app_instance.run()
    except ImportError as e:
        logger.error(f"Runtime ImportError: {e}")
        logger.error("Please ensure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Exiting.")
    sys.exit(0)


@app.command()
def play(
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
):
    """Run the game in interactive Play mode."""
    run_interactive_mode(mode="play", seed=seed, log_level=log_level)


@app.command()
def debug(
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
):
    """Run the game in interactive Debug mode."""
    run_interactive_mode(mode="debug", seed=seed, log_level=log_level)


@app.command()
def train(
    headless: Annotated[
        bool,
        typer.Option("--headless", "-H", help="Run training without visualization."),
    ] = False,
    log_level: LogLevelOption = "INFO",
    seed: SeedOption = 42,
):
    """Run the AlphaTriangle training pipeline."""
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    train_config_override = TrainConfig()
    persist_config_override = PersistenceConfig()
    train_config_override.RANDOM_SEED = seed

    if headless:
        logger.info("Starting training in Headless mode...")
        exit_code = run_training_headless_mode(
            log_level_str=log_level,
            train_config_override=train_config_override,
            persist_config_override=persist_config_override,
        )
    else:
        logger.info("Starting training in Visual mode...")
        exit_code = run_training_visual_mode(
            log_level_str=log_level,
            train_config_override=train_config_override,
            persist_config_override=persist_config_override,
        )

    logger.info(f"Training finished with exit code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    app()


File: muzerotriangle\__init__.py


File: muzerotriangle\config\app_config.py
APP_NAME: str = "AlphaTriangle"


File: muzerotriangle\config\env_config.py
# File: muzerotriangle/config/env_config.py
# No changes needed for this refactoring step. Keep the existing content.
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class EnvConfig(BaseModel):
    """Configuration for the game environment (Pydantic model)."""

    ROWS: int = Field(default=8, gt=0)
    # Provide a default that matches the default ROWS
    COLS_PER_ROW: list[int] = Field(default=[9, 11, 13, 15, 15, 13, 11, 9])
    COLS: int = Field(default=15, gt=0)
    NUM_SHAPE_SLOTS: int = Field(default=3, gt=0)
    MIN_LINE_LENGTH: int = Field(default=3, gt=0)

    # --- Reward System Constants (v3) ---
    REWARD_PER_PLACED_TRIANGLE: float = Field(default=0.01)
    REWARD_PER_CLEARED_TRIANGLE: float = Field(default=0.5)
    REWARD_PER_STEP_ALIVE: float = Field(default=0.005)
    PENALTY_GAME_OVER: float = Field(default=-10.0)
    # --- End Reward System Constants ---

    @field_validator("COLS_PER_ROW")
    @classmethod
    def check_cols_per_row_length(cls, v: list[int], info) -> list[int]:
        data = info.data if info.data else info.values
        rows = data.get("ROWS")
        if rows is None:
            return v
        if len(v) != rows:
            raise ValueError(f"COLS_PER_ROW length ({len(v)}) must equal ROWS ({rows})")
        if any(width <= 0 for width in v):
            raise ValueError("All values in COLS_PER_ROW must be positive.")
        return v

    @model_validator(mode="after")
    def check_cols_match_max_cols_per_row(self) -> "EnvConfig":
        """Ensure COLS is at least the maximum width required by any row."""
        if hasattr(self, "COLS_PER_ROW") and self.COLS_PER_ROW:
            max_row_width = max(self.COLS_PER_ROW, default=0)
            if max_row_width > self.COLS:
                raise ValueError(
                    f"COLS ({self.COLS}) must be >= the maximum value in COLS_PER_ROW ({max_row_width})"
                )
        elif not hasattr(self, "COLS_PER_ROW"):
            pass
        return self

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def ACTION_DIM(self) -> int:
        """Total number of possible actions (shape_slot * row * col)."""
        if (
            hasattr(self, "NUM_SHAPE_SLOTS")
            and hasattr(self, "ROWS")
            and hasattr(self, "COLS")
        ):
            return self.NUM_SHAPE_SLOTS * self.ROWS * self.COLS
        return 0


EnvConfig.model_rebuild(force=True)


File: muzerotriangle\config\mcts_config.py
# File: muzerotriangle/config/mcts_config.py
from pydantic import BaseModel, Field, field_validator


class MCTSConfig(BaseModel):
    """
    Configuration for Monte Carlo Tree Search (Pydantic model).
    --- TUNED FOR INCREASED EXPLORATION & DEPTH ---
    """

    num_simulations: int = Field(default=50, ge=1)  # Reduced for faster debugging
    puct_coefficient: float = Field(default=1.5, gt=0)  # Adjusted c1/c2 balance
    temperature_initial: float = Field(default=1.0, ge=0)
    temperature_final: float = Field(default=0.1, ge=0)
    temperature_anneal_steps: int = Field(default=100, ge=0)  # Reduced anneal
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1.0)
    max_search_depth: int = Field(default=10, ge=1)  # Reduced depth
    discount: float = Field(
        default=0.99,
        gt=0,
        le=1.0,
        description="Discount factor (gamma) used in MCTS backpropagation and value targets.",
    )  # ADDED

    @field_validator("temperature_final")
    @classmethod
    def check_temp_final_le_initial(cls, v: float, info) -> float:
        data = info.data if info.data else info.values
        initial_temp = data.get("temperature_initial")
        if initial_temp is not None and v > initial_temp:
            raise ValueError(
                "temperature_final cannot be greater than temperature_initial"
            )
        return v


MCTSConfig.model_rebuild(force=True)


File: muzerotriangle\config\model_config.py
# File: muzerotriangle/config/model_config.py
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the MuZero Neural Network model (Pydantic model).
    Defines parameters for representation, dynamics, and prediction functions.
    """

    # --- Input Representation ---
    GRID_INPUT_CHANNELS: int = Field(default=1, gt=0)
    OTHER_NN_INPUT_FEATURES_DIM: int = Field(
        default=30, ge=0
    )  # Dimension of non-grid features

    # --- Shared Components ---
    HIDDEN_STATE_DIM: int = Field(
        default=128, gt=0, description="Dimension of the MuZero hidden state (s_k)."
    )
    ACTION_ENCODING_DIM: int = Field(
        default=16,
        gt=0,
        description="Dimension for embedding actions before dynamics function.",
    )
    ACTIVATION_FUNCTION: Literal["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"] = Field(
        default="ReLU"
    )
    USE_BATCH_NORM: bool = Field(default=True)

    # --- Representation Function (h) ---
    # (CNN/ResNet/Transformer part, outputs initial hidden state)
    CONV_FILTERS: list[int] = Field(default=[32, 64, 128])
    CONV_KERNEL_SIZES: list[int | tuple[int, int]] = Field(default=[3, 3, 3])
    CONV_STRIDES: list[int | tuple[int, int]] = Field(default=[1, 1, 1])
    CONV_PADDING: list[int | tuple[int, int] | str] = Field(default=[1, 1, 1])

    NUM_RESIDUAL_BLOCKS: int = Field(default=2, ge=0)
    RESIDUAL_BLOCK_FILTERS: int = Field(default=128, gt=0)  # Match last conv filter

    # Transformer for Representation Encoder (Optional)
    USE_TRANSFORMER_IN_REP: bool = Field(
        default=False, description="Use Transformer in the representation function."
    )
    REP_TRANSFORMER_HEADS: int = Field(default=4, gt=0)
    REP_TRANSFORMER_LAYERS: int = Field(default=2, ge=0)
    REP_TRANSFORMER_FC_DIM: int = Field(default=256, gt=0)

    # Final projection to hidden state dim in representation function
    REP_FC_DIMS_AFTER_ENCODER: list[int] = Field(default=[])
    # If empty, a single Linear layer projects directly to HIDDEN_STATE_DIM

    # --- Dynamics Function (g) ---
    # (Takes s_k, a_{k+1} -> s_{k+1}, r_{k+1})
    DYNAMICS_NUM_RESIDUAL_BLOCKS: int = Field(
        default=2, ge=0, description="Number of ResBlocks in the dynamics function."
    )
    # Dynamics function combines hidden_state + encoded_action

    # Reward Prediction Head (part of Dynamics)
    REWARD_HEAD_DIMS: list[int] = Field(default=[64])
    # Assuming categorical reward prediction (like MuZero paper)
    REWARD_SUPPORT_SIZE: int = Field(
        default=21,
        gt=1,
        description="Number of atoms for categorical reward prediction (e.g., -10 to +10). Must be odd.",
    )

    # --- Prediction Function (f) ---
    # (Takes s_k -> p_k, v_k)
    PREDICTION_NUM_RESIDUAL_BLOCKS: int = Field(
        default=1, ge=0, description="Number of ResBlocks in the prediction function."
    )
    POLICY_HEAD_DIMS: list[int] = Field(default=[64])
    VALUE_HEAD_DIMS: list[int] = Field(default=[64])
    NUM_VALUE_ATOMS: int = Field(
        default=51, gt=1, description="Number of atoms for distributional value head."
    )
    VALUE_MIN: float = Field(default=-10.0)
    VALUE_MAX: float = Field(default=10.0)

    # --- Validation ---
    @model_validator(mode="after")
    def check_conv_layers_consistency(self) -> "ModelConfig":
        n_filters = len(self.CONV_FILTERS)
        if not (
            len(self.CONV_KERNEL_SIZES) == n_filters
            and len(self.CONV_STRIDES) == n_filters
            and len(self.CONV_PADDING) == n_filters
        ):
            raise ValueError(
                "Lengths of CONV_FILTERS, CONV_KERNEL_SIZES, CONV_STRIDES, and CONV_PADDING must match."
            )
        return self

    @model_validator(mode="after")
    def check_residual_filter_match(self) -> "ModelConfig":
        if (
            self.NUM_RESIDUAL_BLOCKS > 0
            and self.CONV_FILTERS
            and self.CONV_FILTERS[-1] != self.RESIDUAL_BLOCK_FILTERS
        ):
            # Representation function will handle projection if needed
            pass
        return self

    @model_validator(mode="after")
    def check_transformer_config(self) -> "ModelConfig":
        if self.USE_TRANSFORMER_IN_REP:
            if self.REP_TRANSFORMER_LAYERS < 0:
                raise ValueError("REP_TRANSFORMER_LAYERS cannot be negative.")
            if self.REP_TRANSFORMER_LAYERS > 0:
                if self.HIDDEN_STATE_DIM <= 0:
                    raise ValueError(
                        "HIDDEN_STATE_DIM must be positive if REP_TRANSFORMER_LAYERS > 0."
                    )
                if self.REP_TRANSFORMER_HEADS <= 0:
                    raise ValueError(
                        "REP_TRANSFORMER_HEADS must be positive if REP_TRANSFORMER_LAYERS > 0."
                    )
                if self.HIDDEN_STATE_DIM % self.REP_TRANSFORMER_HEADS != 0:
                    raise ValueError(
                        f"HIDDEN_STATE_DIM ({self.HIDDEN_STATE_DIM}) must be divisible by REP_TRANSFORMER_HEADS ({self.REP_TRANSFORMER_HEADS})."
                    )
                if self.REP_TRANSFORMER_FC_DIM <= 0:
                    raise ValueError(
                        "REP_TRANSFORMER_FC_DIM must be positive if REP_TRANSFORMER_LAYERS > 0."
                    )
        return self

    @model_validator(mode="after")
    def check_value_distribution_params(self) -> "ModelConfig":
        if self.VALUE_MIN >= self.VALUE_MAX:
            raise ValueError("VALUE_MIN must be strictly less than VALUE_MAX.")
        return self

    @model_validator(mode="after")
    def check_reward_support_size(self) -> "ModelConfig":
        # Often assumed to be odd for symmetry around 0, but not strictly required by algo
        if self.REWARD_SUPPORT_SIZE % 2 == 0:
            # pass # Allow even for now
            raise ValueError("REWARD_SUPPORT_SIZE must be odd.")
        return self


ModelConfig.model_rebuild(force=True)


File: muzerotriangle\config\persistence_config.py
from pathlib import Path

from pydantic import BaseModel, Field, computed_field


class PersistenceConfig(BaseModel):
    """Configuration for saving/loading artifacts (Pydantic model)."""

    ROOT_DATA_DIR: str = Field(default=".muzerotriangle_data")
    RUNS_DIR_NAME: str = Field(default="runs")
    MLFLOW_DIR_NAME: str = Field(default="mlruns")

    CHECKPOINT_SAVE_DIR_NAME: str = Field(default="checkpoints")
    BUFFER_SAVE_DIR_NAME: str = Field(default="buffers")
    GAME_STATE_SAVE_DIR_NAME: str = Field(default="game_states")
    LOG_DIR_NAME: str = Field(default="logs")

    LATEST_CHECKPOINT_FILENAME: str = Field(default="latest.pkl")
    BEST_CHECKPOINT_FILENAME: str = Field(default="best.pkl")
    BUFFER_FILENAME: str = Field(default="buffer.pkl")
    CONFIG_FILENAME: str = Field(default="configs.json")

    RUN_NAME: str = Field(default="default_run")

    SAVE_GAME_STATES: bool = Field(default=False)
    GAME_STATE_SAVE_FREQ_EPISODES: int = Field(default=5, ge=1)

    SAVE_BUFFER: bool = Field(default=True)
    BUFFER_SAVE_FREQ_STEPS: int = Field(default=10, ge=1)

    @computed_field  # type: ignore[misc] # Decorator requires Pydantic v2
    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """Constructs the file URI for MLflow tracking using pathlib."""
        # Ensure attributes exist before calculating
        if hasattr(self, "ROOT_DATA_DIR") and hasattr(self, "MLFLOW_DIR_NAME"):
            abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
            return abs_path.as_uri()
        return ""

    def get_run_base_dir(self, run_name: str | None = None) -> str:
        """Gets the base directory for a specific run."""
        # Ensure attributes exist before calculating
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "RUNS_DIR_NAME"):
            return ""  # Fallback
        name = run_name if run_name else self.RUN_NAME
        return str(Path(self.ROOT_DATA_DIR).joinpath(self.RUNS_DIR_NAME, name))

    def get_mlflow_abs_path(self) -> str:
        """Gets the absolute OS path to the MLflow directory as a string."""
        # Ensure attributes exist before calculating
        if not hasattr(self, "ROOT_DATA_DIR") or not hasattr(self, "MLFLOW_DIR_NAME"):
            return ""  # Fallback
        abs_path = Path(self.ROOT_DATA_DIR).joinpath(self.MLFLOW_DIR_NAME).resolve()
        return str(abs_path)


# Ensure model is rebuilt after changes
PersistenceConfig.model_rebuild(force=True)


File: muzerotriangle\config\README.md
# File: muzerotriangle/config/README.md
# Configuration Module (`muzerotriangle.config`)

## Purpose and Architecture

This module centralizes all configuration parameters for the MuZeroTriangle project. It uses separate **Pydantic models** for different aspects of the application (environment, model, training, visualization, persistence) to promote modularity, clarity, and automatic validation.

-   **Modularity:** Separating configurations makes it easier to manage parameters for different components.
-   **Type Safety & Validation:** Using Pydantic models (`BaseModel`) provides strong type hinting, automatic parsing, and validation of configuration values based on defined types and constraints (e.g., `Field(gt=0)`).
-   **Validation Script:** The [`validation.py`](validation.py) script instantiates all configuration models, triggering Pydantic's validation, and prints a summary.
-   **Dynamic Defaults:** Some configurations, like `RUN_NAME` in `TrainConfig`, use `default_factory` for dynamic defaults (e.g., timestamp).
-   **Computed Fields:** Properties like `ACTION_DIM` in `EnvConfig` or `MLFLOW_TRACKING_URI` in `PersistenceConfig` are defined using `@computed_field` for clarity.
-   **Tuned Defaults:** The default values in `TrainConfig` and `ModelConfig` are now tuned for **more substantial learning runs** compared to the previous quick-testing defaults.

## Exposed Interfaces

-   **Pydantic Models:**
    -   [`EnvConfig`](env_config.py): Environment parameters (grid size, shapes).
    -   [`ModelConfig`](model_config.py): Defines the MuZero **network architecture** parameters (representation, dynamics, prediction functions, hidden state dimension, reward/value support sizes, etc.).
    -   [`TrainConfig`](train_config.py): Training loop hyperparameters (batch size, learning rate, workers, **MuZero unroll steps**, **loss weights**, etc.). **PER disabled by default.**
    -   [`VisConfig`](vis_config.py): Visualization parameters (screen size, FPS, layout).
    -   [`PersistenceConfig`](persistence_config.py): Data saving/loading parameters (directories, filenames).
    -   [`MCTSConfig`](mcts_config.py): MCTS parameters (simulations, exploration constants, temperature).
-   **Constants:**
    -   [`APP_NAME`](app_config.py): The name of the application.
-   **Functions:**
    -   `print_config_info_and_validate(mcts_config_instance: MCTSConfig | None)`: Validates and prints a summary of all configurations by instantiating the Pydantic models.

## Dependencies

This module primarily defines configurations and relies heavily on **Pydantic**.

-   **`pydantic`**: The core library used for defining models and validation.
-   **Standard Libraries:** `typing`, `time`, `os`, `logging`, `pathlib`.

---

**Note:** Please keep this README updated when adding, removing, or significantly modifying configuration parameters or the structure of the Pydantic models. Accurate documentation is crucial for maintainability.

File: muzerotriangle\config\train_config.py
# File: muzerotriangle/config/train_config.py
# File: muzerotriangle/config/train_config.py
import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class TrainConfig(BaseModel):
    """
    Configuration for the MuZero training process (Pydantic model).
    """

    RUN_NAME: str = Field(
        default_factory=lambda: f"train_muzero_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    LOAD_CHECKPOINT_PATH: str | None = Field(default=None)
    LOAD_BUFFER_PATH: str | None = Field(default=None)
    AUTO_RESUME_LATEST: bool = Field(default=True)
    DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="auto")
    RANDOM_SEED: int = Field(default=42)

    # --- Training Loop ---
    MAX_TRAINING_STEPS: int | None = Field(default=100_000, ge=1)

    # --- Workers & Batching ---
    NUM_SELF_PLAY_WORKERS: int = Field(default=8, ge=1)
    WORKER_DEVICE: Literal["auto", "cuda", "cpu", "mps"] = Field(default="cpu")
    BATCH_SIZE: int = Field(default=64, ge=1)  # Batches of sequences
    BUFFER_CAPACITY: int = Field(
        default=100_000, ge=1
    )  # Total steps across trajectories
    MIN_BUFFER_SIZE_TO_TRAIN: int = Field(
        default=10_000,
        ge=1,  # Minimum total steps
    )
    WORKER_UPDATE_FREQ_STEPS: int = Field(default=500, ge=1)

    # --- MuZero Specific ---
    MUZERO_UNROLL_STEPS: int = Field(
        default=5,
        ge=0,
        description="Number of steps to unroll the dynamics model during training.",
    )
    N_STEP_RETURNS: int = Field(
        default=10,
        ge=1,
        description="Number of steps for calculating N-step reward targets.",
    )
    POLICY_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    VALUE_LOSS_WEIGHT: float = Field(default=0.25, ge=0)  # Often lower than policy
    REWARD_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    DISCOUNT: float = Field(
        default=0.99,
        gt=0,
        le=1.0,
        description="Discount factor (gamma) used for N-step returns and MCTS.",
    )

    # --- Optimizer ---
    OPTIMIZER_TYPE: Literal["Adam", "AdamW", "SGD"] = Field(default="AdamW")
    LEARNING_RATE: float = Field(default=1e-4, gt=0)  # MuZero often uses smaller LR
    WEIGHT_DECAY: float = Field(default=1e-4, ge=0)
    GRADIENT_CLIP_VALUE: float | None = Field(default=5.0)  # Clip grads

    # --- LR Scheduler ---
    LR_SCHEDULER_TYPE: Literal["StepLR", "CosineAnnealingLR"] | None = Field(
        default="CosineAnnealingLR"
    )
    LR_SCHEDULER_T_MAX: int | None = Field(
        default=None
    )  # Auto-set from MAX_TRAINING_STEPS
    LR_SCHEDULER_ETA_MIN: float = Field(default=1e-6, ge=0)

    # --- Checkpointing ---
    CHECKPOINT_SAVE_FREQ_STEPS: int = Field(default=2500, ge=1)

    # --- Prioritized Experience Replay (PER) ---
    # --- RE-ENABLED ---
    USE_PER: bool = Field(default=True)
    PER_ALPHA: float = Field(default=0.6, ge=0)
    PER_BETA_INITIAL: float = Field(default=0.4, ge=0, le=1.0)
    PER_BETA_FINAL: float = Field(default=1.0, ge=0, le=1.0)
    PER_BETA_ANNEAL_STEPS: int | None = Field(default=None)  # Auto-set
    PER_EPSILON: float = Field(default=1e-5, gt=0)

    # --- Model Compilation ---
    COMPILE_MODEL: bool = Field(default=True)

    @model_validator(mode="after")
    def check_buffer_sizes(self) -> "TrainConfig":
        if self.MIN_BUFFER_SIZE_TO_TRAIN > self.BUFFER_CAPACITY:
            raise ValueError(
                "MIN_BUFFER_SIZE_TO_TRAIN cannot be greater than BUFFER_CAPACITY."
            )
        return self

    @model_validator(mode="after")
    def set_scheduler_t_max(self) -> "TrainConfig":
        if (
            self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
            and self.LR_SCHEDULER_T_MAX is None
            and self.MAX_TRAINING_STEPS is not None
            and self.MAX_TRAINING_STEPS >= 1
        ):
            self.LR_SCHEDULER_T_MAX = self.MAX_TRAINING_STEPS
            logger.info(
                f"Set LR_SCHEDULER_T_MAX to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
            )
        elif (
            self.LR_SCHEDULER_T_MAX is None
            and self.LR_SCHEDULER_TYPE == "CosineAnnealingLR"
        ):
            self.LR_SCHEDULER_T_MAX = 100_000  # Fallback if MAX_TRAINING_STEPS is None
            logger.warning(
                f"MAX_TRAINING_STEPS is None, using fallback T_max {self.LR_SCHEDULER_T_MAX}"
            )

        if self.LR_SCHEDULER_T_MAX is not None and self.LR_SCHEDULER_T_MAX <= 0:
            raise ValueError("LR_SCHEDULER_T_MAX must be positive if set.")
        return self

    @model_validator(mode="after")
    def set_per_beta_anneal_steps(self) -> "TrainConfig":
        if self.USE_PER and self.PER_BETA_ANNEAL_STEPS is None:
            if self.MAX_TRAINING_STEPS is not None and self.MAX_TRAINING_STEPS >= 1:
                self.PER_BETA_ANNEAL_STEPS = self.MAX_TRAINING_STEPS
                logger.info(
                    f"Set PER_BETA_ANNEAL_STEPS to MAX_TRAINING_STEPS ({self.MAX_TRAINING_STEPS})"
                )
            else:
                self.PER_BETA_ANNEAL_STEPS = 100_000  # Fallback
                logger.warning(
                    f"MAX_TRAINING_STEPS invalid, using fallback PER_BETA_ANNEAL_STEPS {self.PER_BETA_ANNEAL_STEPS}"
                )

        if (
            self.USE_PER
            and self.PER_BETA_ANNEAL_STEPS is not None
            and self.PER_BETA_ANNEAL_STEPS <= 0
        ):
            raise ValueError("PER_BETA_ANNEAL_STEPS must be positive if set.")
        return self

    @field_validator("GRADIENT_CLIP_VALUE")
    @classmethod
    def check_gradient_clip(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("GRADIENT_CLIP_VALUE must be positive if set.")
        return v

    @field_validator("PER_BETA_FINAL")
    @classmethod
    def check_per_beta_final(cls, v: float, info) -> float:
        data = info.data if info.data else info.values
        initial_beta = data.get("PER_BETA_INITIAL")
        if initial_beta is not None and v < initial_beta:
            raise ValueError("PER_BETA_FINAL cannot be less than PER_BETA_INITIAL")
        return v


TrainConfig.model_rebuild(force=True)


File: muzerotriangle\config\validation.py
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from .env_config import EnvConfig
from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig
from .vis_config import VisConfig

logger = logging.getLogger(__name__)


def print_config_info_and_validate(mcts_config_instance: MCTSConfig | None):
    """Prints configuration summary and performs validation using Pydantic."""
    print("-" * 40)
    print("Configuration Validation & Summary")
    print("-" * 40)
    all_valid = True
    configs_validated: dict[str, Any] = {}

    config_classes: dict[str, type[BaseModel]] = {
        "Environment": EnvConfig,
        "Model": ModelConfig,
        "Training": TrainConfig,
        "Visualization": VisConfig,
        "Persistence": PersistenceConfig,
        "MCTS": MCTSConfig,
    }

    for name, ConfigClass in config_classes.items():
        instance: BaseModel | None = None
        try:
            if name == "MCTS":
                if mcts_config_instance is not None:
                    instance = MCTSConfig.model_validate(
                        mcts_config_instance.model_dump()
                    )
                    print(f"[{name}] - Instance provided & validated OK")
                else:
                    instance = ConfigClass()
                    print(f"[{name}] - Validated OK (Instantiated Default)")
            else:
                instance = ConfigClass()
                print(f"[{name}] - Validated OK")
            configs_validated[name] = instance
        except ValidationError as e:
            logger.error(f"Validation failed for {name} Config:")
            logger.error(e)
            all_valid = False
            configs_validated[name] = None
        except Exception as e:
            logger.error(
                f"Unexpected error instantiating/validating {name} Config: {e}"
            )
            all_valid = False
            configs_validated[name] = None

    print("-" * 40)
    print("Configuration Values:")
    print("-" * 40)

    for name, instance in configs_validated.items():
        print(f"--- {name} Config ---")
        if instance:
            dump_data = instance.model_dump()
            for field_name, value in dump_data.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {field_name}: [List with {len(value)} items]")
                elif isinstance(value, dict) and len(value) > 5:
                    print(f"  {field_name}: {{Dict with {len(value)} keys}}")
                else:
                    print(f"  {field_name}: {value}")
        else:
            print("  <Validation Failed>")
        print("-" * 20)

    print("-" * 40)
    if not all_valid:
        logger.critical("Configuration validation failed. Please check errors above.")
        raise ValueError("Invalid configuration settings.")
    else:
        logger.info("All configurations validated successfully.")
    print("-" * 40)


File: muzerotriangle\config\vis_config.py
from pydantic import BaseModel, Field


class VisConfig(BaseModel):
    """Configuration for visualization (Pydantic model)."""

    FPS: int = Field(default=30, gt=0)
    SCREEN_WIDTH: int = Field(default=1000, gt=0)
    SCREEN_HEIGHT: int = Field(default=800, gt=0)

    # Layout
    GRID_AREA_RATIO: float = Field(default=0.7, gt=0, le=1.0)
    PREVIEW_AREA_WIDTH: int = Field(default=150, gt=0)
    PADDING: int = Field(default=10, ge=0)
    HUD_HEIGHT: int = Field(default=40, ge=0)

    # Fonts (sizes)
    FONT_UI_SIZE: int = Field(default=24, gt=0)
    FONT_SCORE_SIZE: int = Field(default=30, gt=0)
    FONT_HELP_SIZE: int = Field(default=18, gt=0)

    # Preview Area
    PREVIEW_PADDING: int = Field(default=5, ge=0)
    PREVIEW_BORDER_WIDTH: int = Field(default=1, ge=0)
    PREVIEW_SELECTED_BORDER_WIDTH: int = Field(default=3, ge=0)
    PREVIEW_INNER_PADDING: int = Field(default=2, ge=0)


VisConfig.model_rebuild(force=True)


File: muzerotriangle\config\__init__.py
from .app_config import APP_NAME
from .env_config import EnvConfig
from .mcts_config import MCTSConfig
from .model_config import ModelConfig
from .persistence_config import PersistenceConfig
from .train_config import TrainConfig
from .validation import print_config_info_and_validate
from .vis_config import VisConfig

__all__ = [
    "APP_NAME",
    "EnvConfig",
    "ModelConfig",
    "PersistenceConfig",
    "TrainConfig",
    "VisConfig",
    "MCTSConfig",
    "print_config_info_and_validate",
]


File: muzerotriangle\data\data_manager.py
# File: muzerotriangle/data/data_manager.py
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import ray
from pydantic import ValidationError

from .path_manager import PathManager
from .schemas import (  # Import BufferData
    BufferData,
    CheckpointData,
    LoadedTrainingState,
)
from .serializer import Serializer

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..config import PersistenceConfig, TrainConfig
    from ..nn import NeuralNetwork
    from ..rl.core.buffer import ExperienceBuffer

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates loading and saving of training artifacts using PathManager and Serializer.
    Handles MLflow artifact logging. Adapted for MuZero buffer format.
    """

    def __init__(
        self, persist_config: "PersistenceConfig", train_config: "TrainConfig"
    ):
        self.persist_config = persist_config
        self.train_config = train_config
        self.persist_config.RUN_NAME = self.train_config.RUN_NAME

        self.path_manager = PathManager(self.persist_config)
        self.serializer = Serializer()

        self.path_manager.create_run_directories()
        logger.info(
            f"DataManager initialized for run '{self.persist_config.RUN_NAME}'."
        )

    def load_initial_state(self) -> LoadedTrainingState:
        """
        Loads the initial training state (checkpoint and MuZero buffer).
        Handles AUTO_RESUME_LATEST logic.
        """
        loaded_state = LoadedTrainingState()
        checkpoint_path = self.path_manager.determine_checkpoint_to_load(
            self.train_config.LOAD_CHECKPOINT_PATH,
            self.train_config.AUTO_RESUME_LATEST,
        )
        checkpoint_run_name: str | None = None

        # --- Load Checkpoint ---
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                loaded_checkpoint_model = self.serializer.load_checkpoint(
                    checkpoint_path
                )
                if loaded_checkpoint_model:
                    loaded_state.checkpoint_data = loaded_checkpoint_model
                    checkpoint_run_name = loaded_state.checkpoint_data.run_name
                    logger.info(
                        f"Checkpoint loaded (Run: {cpd.run_name}, Step: {cpd.global_step})"
                        if (cpd := loaded_state.checkpoint_data)
                        else "Checkpoint data invalid"
                    )  # Use walrus
                else:
                    logger.error(f"Loading checkpoint failed: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}", exc_info=True)

        # --- Load MuZero Buffer ---
        # Buffer saving logic uses config, no specific flag check needed here
        buffer_path = self.path_manager.determine_buffer_to_load(
            self.train_config.LOAD_BUFFER_PATH,
            self.train_config.AUTO_RESUME_LATEST,
            checkpoint_run_name,  # Pass run name from loaded checkpoint
        )
        if buffer_path:
            logger.info(f"Loading MuZero buffer: {buffer_path}")
            try:
                # Use the updated serializer method
                loaded_buffer_model: BufferData | None = self.serializer.load_buffer(
                    buffer_path
                )
                if loaded_buffer_model:
                    loaded_state.buffer_data = loaded_buffer_model
                    logger.info(
                        f"MuZero Buffer loaded. Trajectories: {len(loaded_buffer_model.trajectories)}, Total Steps: {loaded_buffer_model.total_steps}"
                    )
                else:
                    logger.error(f"Loading buffer failed: {buffer_path}")
            except Exception as e:
                logger.error(f"Failed to load MuZero buffer: {e}", exc_info=True)

        if not loaded_state.checkpoint_data and not loaded_state.buffer_data:
            logger.info("No checkpoint or buffer loaded. Starting fresh.")

        return loaded_state

    def save_training_state(
        self,
        nn: "NeuralNetwork",
        optimizer: "Optimizer",
        stats_collector_actor: ray.actor.ActorHandle | None,
        buffer: "ExperienceBuffer",  # Type hint remains the same
        global_step: int,
        episodes_played: int,
        total_simulations_run: int,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Saves the training state (checkpoint and MuZero buffer)."""
        run_name = self.persist_config.RUN_NAME
        logger.info(
            f"Saving training state for run '{run_name}' at step {global_step}. Final={is_final}, Best={is_best}"
        )

        stats_collector_state = {}
        if stats_collector_actor is not None:
            try:
                stats_state_ref = stats_collector_actor.get_state.remote()
                stats_collector_state = ray.get(stats_state_ref, timeout=5.0)
            except Exception as e:
                logger.error(
                    f"Error fetching StatsCollectorActor state: {e}", exc_info=True
                )

        # --- Save Checkpoint ---
        saved_checkpoint_path: Path | None = None
        try:
            checkpoint_data = CheckpointData(
                run_name=run_name,
                global_step=global_step,
                episodes_played=episodes_played,
                total_simulations_run=total_simulations_run,
                model_config_dict=nn.model_config.model_dump(),
                env_config_dict=nn.env_config.model_dump(),
                model_state_dict=nn.get_weights(),
                optimizer_state_dict=self.serializer.prepare_optimizer_state(optimizer),
                stats_collector_state=stats_collector_state,
            )
            step_checkpoint_path = self.path_manager.get_checkpoint_path(
                step=global_step, is_final=is_final
            )
            self.serializer.save_checkpoint(checkpoint_data, step_checkpoint_path)
            saved_checkpoint_path = step_checkpoint_path
            self.path_manager.update_checkpoint_links(
                step_checkpoint_path, is_best=is_best
            )
        except ValidationError as e:
            logger.error(f"CheckpointData validation failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

        # --- Save MuZero Buffer ---
        saved_buffer_path: Path | None = None
        # Use config flag if explicit control is desired, otherwise always save if buffer exists
        # if self.persist_config.SAVE_BUFFER:
        try:
            # prepare_buffer_data handles MuZero format now
            buffer_data_model: BufferData | None = self.serializer.prepare_buffer_data(
                buffer
            )
            if buffer_data_model:
                buffer_path = self.path_manager.get_buffer_path(
                    step=global_step, is_final=is_final
                )
                self.serializer.save_buffer(
                    buffer_data_model, buffer_path
                )  # save_buffer handles MuZero format
                saved_buffer_path = buffer_path
                self.path_manager.update_buffer_link(buffer_path)
            else:
                logger.warning("Buffer data preparation failed, buffer not saved.")
        except Exception as e:
            logger.error(f"Failed to save MuZero buffer: {e}", exc_info=True)

        # --- Log Artifacts ---
        self._log_artifacts(saved_checkpoint_path, saved_buffer_path, is_best)

    def _log_artifacts(
        self, checkpoint_path: Path | None, buffer_path: Path | None, is_best: bool
    ):
        """Logs saved checkpoint and buffer files to MLflow."""
        # Logic remains the same, paths point to saved files
        try:
            if checkpoint_path and checkpoint_path.exists():
                ckpt_artifact_path = self.persist_config.CHECKPOINT_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(checkpoint_path), artifact_path=ckpt_artifact_path
                )
                latest_path = self.path_manager.get_checkpoint_path(is_latest=True)
                if latest_path.exists():
                    mlflow.log_artifact(
                        str(latest_path), artifact_path=ckpt_artifact_path
                    )
                if is_best:
                    best_path = self.path_manager.get_checkpoint_path(is_best=True)
                    if best_path.exists():
                        mlflow.log_artifact(
                            str(best_path), artifact_path=ckpt_artifact_path
                        )
                logger.info(
                    f"Logged checkpoint artifacts to MLflow path: {ckpt_artifact_path}"
                )

            if buffer_path and buffer_path.exists():
                buffer_artifact_path = self.persist_config.BUFFER_SAVE_DIR_NAME
                mlflow.log_artifact(
                    str(buffer_path), artifact_path=buffer_artifact_path
                )
                default_buffer_path = self.path_manager.get_buffer_path()
                if default_buffer_path.exists():
                    mlflow.log_artifact(
                        str(default_buffer_path), artifact_path=buffer_artifact_path
                    )
                logger.info(
                    f"Logged MuZero buffer artifacts to MLflow path: {buffer_artifact_path}"
                )
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}", exc_info=True)

    def save_run_config(self, configs: dict[str, Any]):
        """Saves the combined configuration dictionary as a JSON artifact."""
        try:
            config_path = self.path_manager.get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            self.serializer.save_config_json(configs, config_path)
            mlflow.log_artifact(str(config_path), artifact_path="config")
        except Exception as e:
            logger.error(f"Failed to save/log run config JSON: {e}", exc_info=True)

    # PathManager getters remain the same
    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        return self.path_manager.get_checkpoint_path(
            run_name, step, is_latest, is_best, is_final
        )

    def get_buffer_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_final: bool = False,
    ) -> Path:
        return self.path_manager.get_buffer_path(run_name, step, is_final)


File: muzerotriangle\data\path_manager.py
# File: muzerotriangle/data/path_manager.py
import datetime
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import PersistenceConfig

logger = logging.getLogger(__name__)


class PathManager:
    """Manages file paths, directory creation, and discovery for training runs."""

    def __init__(self, persist_config: "PersistenceConfig"):
        self.persist_config = persist_config
        self.root_data_dir = Path(self.persist_config.ROOT_DATA_DIR)
        self._update_paths()  # Initialize paths based on config

    def _update_paths(self):
        """Updates paths based on the current RUN_NAME in persist_config."""
        self.run_base_dir = Path(self.persist_config.get_run_base_dir())
        self.checkpoint_dir = (
            self.run_base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        )
        self.buffer_dir = self.run_base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        self.log_dir = self.run_base_dir / self.persist_config.LOG_DIR_NAME
        self.config_path = self.run_base_dir / self.persist_config.CONFIG_FILENAME

    def create_run_directories(self):
        """Creates necessary directories for the current run."""
        self.root_data_dir.mkdir(parents=True, exist_ok=True)
        self.run_base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.persist_config.SAVE_BUFFER:
            self.buffer_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_latest: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """Constructs the path for a checkpoint file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        checkpoint_dir = base_dir / self.persist_config.CHECKPOINT_SAVE_DIR_NAME
        if is_latest:
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        elif is_best:
            filename = self.persist_config.BEST_CHECKPOINT_FILENAME
        elif is_final and step is not None:
            filename = f"checkpoint_final_step_{step}.pkl"
        elif step is not None:
            filename = f"checkpoint_step_{step}.pkl"
        else:
            # Default to latest if no specific type is given
            filename = self.persist_config.LATEST_CHECKPOINT_FILENAME
        # Ensure filename ends with .pkl
        filename_pkl = Path(filename).with_suffix(".pkl")
        return checkpoint_dir / filename_pkl

    def get_buffer_path(
        self,
        run_name: str | None = None,
        step: int | None = None,
        is_final: bool = False,
    ) -> Path:
        """Constructs the path for the replay buffer file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        buffer_dir = base_dir / self.persist_config.BUFFER_SAVE_DIR_NAME
        if is_final and step is not None:
            filename = f"buffer_final_step_{step}.pkl"
        elif step is not None and self.persist_config.BUFFER_SAVE_FREQ_STEPS > 0:
            # Use default name for intermediate saves if frequency is set
            filename = self.persist_config.BUFFER_FILENAME
        else:
            # Default name for initial load or if frequency is not set
            filename = self.persist_config.BUFFER_FILENAME
        return buffer_dir / Path(filename).with_suffix(".pkl")

    def get_config_path(self, run_name: str | None = None) -> Path:
        """Constructs the path for the config JSON file."""
        target_run_name = run_name if run_name else self.persist_config.RUN_NAME
        base_dir = Path(self.persist_config.get_run_base_dir(target_run_name))
        return base_dir / self.persist_config.CONFIG_FILENAME

    def find_latest_run_dir(self, current_run_name: str) -> str | None:
        """
        Finds the most recent *previous* run directory based on timestamp parsing.
        Assumes run names follow a pattern like 'prefix_YYYYMMDD_HHMMSS'.
        """
        runs_root_dir = self.root_data_dir / self.persist_config.RUNS_DIR_NAME
        potential_runs: list[tuple[datetime.datetime, str]] = []
        run_name_pattern = re.compile(r"^(?:test_run|train)_(\d{8}_\d{6})$")

        try:
            if not runs_root_dir.exists():
                return None

            for d in runs_root_dir.iterdir():
                if d.is_dir() and d.name != current_run_name:
                    match = run_name_pattern.match(d.name)
                    if match:
                        timestamp_str = match.group(1)
                        try:
                            run_time = datetime.datetime.strptime(
                                timestamp_str, "%Y%m%d_%H%M%S"
                            )
                            potential_runs.append((run_time, d.name))
                        except ValueError:
                            logger.warning(
                                f"Could not parse timestamp from directory name: {d.name}"
                            )
                    else:
                        logger.debug(
                            f"Directory name {d.name} does not match expected pattern."
                        )

            if not potential_runs:
                logger.info("No previous run directories found matching the pattern.")
                return None

            potential_runs.sort(key=lambda item: item[0], reverse=True)
            latest_run_name = potential_runs[0][1]
            logger.debug(
                f"Found potential previous runs (sorted): {[name for _, name in potential_runs]}. Latest: {latest_run_name}"
            )
            return latest_run_name

        except Exception as e:
            logger.error(f"Error finding latest run directory: {e}", exc_info=True)
            return None

    def determine_checkpoint_to_load(
        self, load_path_config: str | None, auto_resume: bool
    ) -> Path | None:
        """Determines the absolute path of the checkpoint file to load."""
        current_run_name = self.persist_config.RUN_NAME
        checkpoint_to_load: Path | None = None

        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                checkpoint_to_load = load_path.resolve()
                logger.info(f"Using specified checkpoint path: {checkpoint_to_load}")
            else:
                logger.warning(
                    f"Specified checkpoint path not found: {load_path_config}"
                )

        if not checkpoint_to_load and auto_resume:
            latest_run_name = self.find_latest_run_dir(current_run_name)
            if latest_run_name:
                potential_latest_path = self.get_checkpoint_path(
                    run_name=latest_run_name, is_latest=True
                )
                if potential_latest_path.exists():
                    checkpoint_to_load = potential_latest_path.resolve()
                    logger.info(
                        f"Auto-resuming from latest checkpoint in previous run '{latest_run_name}': {checkpoint_to_load}"
                    )
                else:
                    logger.info(
                        f"Latest checkpoint file not found in latest run directory '{latest_run_name}'."
                    )
            else:
                logger.info("Auto-resume enabled, but no previous run directory found.")

        if not checkpoint_to_load:
            logger.info("No checkpoint found to load. Starting training from scratch.")

        return checkpoint_to_load

    def determine_buffer_to_load(
        self,
        load_path_config: str | None,
        auto_resume: bool,
        checkpoint_run_name: str | None,
    ) -> Path | None:
        """Determines the buffer file path to load."""
        if load_path_config:
            load_path = Path(load_path_config)
            if load_path.exists():
                logger.info(f"Using specified buffer path: {load_path_config}")
                return load_path.resolve()
            else:
                logger.warning(f"Specified buffer path not found: {load_path_config}")

        if checkpoint_run_name:
            potential_buffer_path = self.get_buffer_path(run_name=checkpoint_run_name)
            if potential_buffer_path.exists():
                logger.info(
                    f"Loading buffer from checkpoint run '{checkpoint_run_name}': {potential_buffer_path}"
                )
                return potential_buffer_path.resolve()
            else:
                logger.info(
                    f"Default buffer file not found in checkpoint run directory '{checkpoint_run_name}'."
                )

        if auto_resume and not checkpoint_run_name:
            latest_previous_run_name = self.find_latest_run_dir(
                self.persist_config.RUN_NAME
            )
            if latest_previous_run_name:
                potential_buffer_path = self.get_buffer_path(
                    run_name=latest_previous_run_name
                )
                if potential_buffer_path.exists():
                    logger.info(
                        f"Auto-resuming buffer from latest previous run '{latest_previous_run_name}' (no checkpoint loaded): {potential_buffer_path}"
                    )
                    return potential_buffer_path.resolve()
                else:
                    logger.info(
                        f"Default buffer file not found in latest run directory '{latest_previous_run_name}'."
                    )

        logger.info("No suitable buffer file found to load.")
        return None

    def update_checkpoint_links(self, step_checkpoint_path: Path, is_best: bool):
        """Updates the 'latest' and optionally 'best' checkpoint links."""
        if not step_checkpoint_path.exists():
            logger.error(
                f"Source checkpoint path does not exist: {step_checkpoint_path}"
            )
            return

        latest_path = self.get_checkpoint_path(is_latest=True)
        best_path = self.get_checkpoint_path(is_best=True)
        try:
            shutil.copy2(step_checkpoint_path, latest_path)
            logger.debug(f"Updated latest checkpoint link to {step_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to update latest checkpoint link: {e}")
        if is_best:
            try:
                shutil.copy2(step_checkpoint_path, best_path)
                logger.info(
                    f"Updated best checkpoint link to step {step_checkpoint_path.stem}"
                )
            except Exception as e:
                logger.error(f"Failed to update best checkpoint link: {e}")

    def update_buffer_link(self, step_buffer_path: Path):
        """Updates the default buffer link."""
        if not step_buffer_path.exists():
            logger.error(f"Source buffer path does not exist: {step_buffer_path}")
            return

        default_buffer_path = self.get_buffer_path()
        try:
            shutil.copy2(step_buffer_path, default_buffer_path)
            logger.debug(f"Updated default buffer file: {default_buffer_path}")
        except Exception as e_default:
            logger.error(
                f"Error updating default buffer file {default_buffer_path}: {e_default}"
            )


File: muzerotriangle\data\README.md
# File: muzerotriangle/data/README.md
# Data Management Module (`muzerotriangle.data`)

## Purpose and Architecture

This module is responsible for handling the persistence of training artifacts using structured data schemas defined with Pydantic. It manages:

-   Neural network checkpoints (model weights, optimizer state).
-   **MuZero Experience Replay Buffer data (list of trajectories).**
-   Statistics collector state.
-   Run configuration files.

The core component is the [`DataManager`](data_manager.py) class, which centralizes file path management and saving/loading logic based on the [`PersistenceConfig`](../config/persistence_config.py) and [`TrainConfig`](../config/train_config.py). It uses `cloudpickle` for robust serialization of complex Python objects, including Pydantic models containing trajectories and tensors.

-   **Schemas ([`schemas.py`](schemas.py)):** Defines Pydantic models (`CheckpointData`, `BufferData`, `LoadedTrainingState`) to structure the data being saved and loaded. **`BufferData` now stores a list of `Trajectory` objects and the `total_steps` across all trajectories.**
-   **Path Management ([`path_manager.py`](path_manager.py)):** Manages file paths, directory creation, and discovery.
-   **Serialization ([`serializer.py`](serializer.py)):** Handles reading/writing files using `cloudpickle` and JSON. **`prepare_buffer_data` extracts trajectories and total steps from the `ExperienceBuffer`. `load_buffer` loads and validates the trajectory list.**
-   **Centralization:** `DataManager` provides a single point of control for saving/loading.
-   **Configuration-Driven:** Uses `PersistenceConfig` and `TrainConfig`.
-   **Run Management:** Organizes artifacts into subdirectories based on `RUN_NAME`.
-   **State Loading:** `DataManager.load_initial_state` determines files, deserializes, validates, and returns `LoadedTrainingState`. **Buffer loading reconstructs the buffer state from the trajectory list and total steps.**
-   **State Saving:** `DataManager.save_training_state` assembles data, serializes, and saves.
-   **MLflow Integration:** Logs artifacts to MLflow.

## Exposed Interfaces

-   **Classes:**
    -   `DataManager`: Orchestrates saving and loading.
        -   `__init__(...)`
        -   `load_initial_state() -> LoadedTrainingState`: Loads state.
        -   `save_training_state(...)`: Saves state.
        -   `save_run_config(...)`: Saves config JSON.
        -   `get_checkpoint_path(...)`, `get_buffer_path(...)`
    -   `PathManager`: Manages file paths.
    -   `Serializer`: Handles serialization/deserialization.
    -   `CheckpointData` (from `schemas.py`).
    -   `BufferData` (from `schemas.py`, **MuZero format**).
    -   `LoadedTrainingState` (from `schemas.py`).

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: `PersistenceConfig`, `TrainConfig`.
-   **[`muzerotriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`muzerotriangle.rl`](../rl/README.md)**: `ExperienceBuffer`.
-   **[`muzerotriangle.stats`](../stats/README.md)**: `StatsCollectorActor`.
-   **[`muzerotriangle.utils`](../utils/README.md)**: `types` (**including `Trajectory`**).
-   **`torch.optim`**: `Optimizer`.
-   **Standard Libraries:** `os`, `shutil`, `logging`, `glob`, `re`, `json`, `collections.deque`, `pathlib`, `datetime`.
-   **Third-Party:** `pydantic`, `cloudpickle`, `torch`, `ray`, `mlflow`, `numpy`.

---

**Note:** Keep this README updated when changing schemas, artifact types, or saving/loading mechanisms, especially concerning the MuZero buffer format.

File: muzerotriangle\data\schemas.py
# File: muzerotriangle/data/schemas.py
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Use relative import for Trajectory
from ..utils.types import Trajectory  # Import Trajectory type

arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class CheckpointData(BaseModel):
    """Pydantic model defining the structure of saved checkpoint data."""

    model_config = arbitrary_types_config

    run_name: str
    global_step: int = Field(..., ge=0)
    episodes_played: int = Field(..., ge=0)
    total_simulations_run: int = Field(..., ge=0)
    model_config_dict: dict[str, Any]
    env_config_dict: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    stats_collector_state: dict[str, Any]


class BufferData(BaseModel):
    """Pydantic model defining the structure of saved MuZero buffer data."""

    model_config = arbitrary_types_config

    # --- CHANGED: Store list of Trajectories ---
    trajectories: list[Trajectory]
    total_steps: int = Field(..., ge=0)  # Store total steps for quicker loading
    # --- END CHANGED ---


class LoadedTrainingState(BaseModel):
    """Pydantic model representing the fully loaded state."""

    model_config = arbitrary_types_config

    checkpoint_data: CheckpointData | None = None
    buffer_data: BufferData | None = None


# Rebuild models after changes
CheckpointData.model_rebuild(force=True)
BufferData.model_rebuild(force=True)
LoadedTrainingState.model_rebuild(force=True)


File: muzerotriangle\data\serializer.py
# File: muzerotriangle/data/serializer.py
# File: muzerotriangle/data/serializer.py
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
from pydantic import ValidationError

# --- ADDED: Import Trajectory ---
# --- END ADDED ---
from .schemas import BufferData, CheckpointData

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..rl.core.buffer import ExperienceBuffer
    from ..utils.types import Trajectory

logger = logging.getLogger(__name__)


class Serializer:
    """Handles serialization and deserialization of training data."""

    def load_checkpoint(self, path: Path) -> CheckpointData | None:
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, CheckpointData):
                return loaded_data
            else:
                logger.error(
                    f"Loaded checkpoint {path} type mismatch: {type(loaded_data)}."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed for checkpoint {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading checkpoint {path}: {e}", exc_info=True)
            return None

    def save_checkpoint(self, data: CheckpointData, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
                logger.info(f"Checkpoint data saved to {path}")
        except Exception as e:
            logger.error(f"Failed save checkpoint {path}: {e}", exc_info=True)
            raise

    def load_buffer(self, path: Path) -> BufferData | None:
        """Loads and validates MuZero buffer data (trajectories) from a file."""
        try:
            with path.open("rb") as f:
                loaded_data = cloudpickle.load(f)
            if isinstance(loaded_data, BufferData):
                # --- FIXED: Add type hint ---
                valid_trajectories: list[Trajectory] = []
                # --- END FIXED ---
                total_valid_steps = 0
                invalid_traj_count = 0
                invalid_step_count = 0
                for i, traj in enumerate(loaded_data.trajectories):
                    if not isinstance(traj, list):
                        invalid_traj_count += 1
                        continue
                    valid_steps_in_traj = []
                    for j, step in enumerate(traj):
                        is_valid_step = False
                        try:
                            if (
                                isinstance(step, dict)
                                and all(
                                    k in step
                                    for k in [
                                        "observation",
                                        "action",
                                        "reward",
                                        "policy_target",
                                        "value_target",
                                    ]
                                )
                                and isinstance(step["observation"], dict)
                                and "grid" in step["observation"]
                                and "other_features" in step["observation"]
                                and isinstance(step["observation"]["grid"], np.ndarray)
                                and isinstance(
                                    step["observation"]["other_features"], np.ndarray
                                )
                                and np.all(np.isfinite(step["observation"]["grid"]))
                                and np.all(
                                    np.isfinite(step["observation"]["other_features"])
                                )
                                and isinstance(step["action"], int)
                                and isinstance(step["reward"], float | int)
                                and isinstance(step["policy_target"], dict)
                                and isinstance(step["value_target"], float | int)
                            ):
                                is_valid_step = True
                        except Exception as val_err:
                            logger.warning(
                                f"Validation error step {j} traj {i}: {val_err}"
                            )
                        if is_valid_step:
                            valid_steps_in_traj.append(step)
                        else:
                            invalid_step_count += 1
                    if valid_steps_in_traj:
                        valid_trajectories.append(valid_steps_in_traj)
                        total_valid_steps += len(valid_steps_in_traj)
                    else:
                        invalid_traj_count += 1
                if invalid_traj_count > 0 or invalid_step_count > 0:
                    logger.warning(
                        f"Loaded buffer: Skipped {invalid_traj_count} invalid trajs, {invalid_step_count} invalid steps."
                    )
                loaded_data.trajectories = valid_trajectories
                loaded_data.total_steps = total_valid_steps
                return loaded_data
            else:
                logger.error(
                    f"Loaded buffer {path} type mismatch: {type(loaded_data)}."
                )
                return None
        except ValidationError as e:
            logger.error(
                f"Pydantic validation failed buffer {path}: {e}", exc_info=True
            )
            return None
        except FileNotFoundError:
            logger.warning(f"Buffer file not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Failed load MuZero buffer {path}: {e}", exc_info=True)
            return None

    def save_buffer(self, data: BufferData, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                cloudpickle.dump(data, f)
                logger.info(
                    f"MuZero Buffer saved to {path} ({len(data.trajectories)} trajs, {data.total_steps} steps)"
                )
        except Exception as e:
            logger.error(f"Error saving MuZero buffer {path}: {e}", exc_info=True)
            raise

    def prepare_optimizer_state(self, optimizer: "Optimizer") -> dict[str, Any]:
        optimizer_state_cpu = {}
        try:
            optimizer_state_dict = optimizer.state_dict()

            def move_to_cpu(item):
                if isinstance(item, torch.Tensor):
                    return item.cpu()
                elif isinstance(item, dict):
                    return {k: move_to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [move_to_cpu(elem) for elem in item]
                else:
                    return item

            optimizer_state_cpu = move_to_cpu(optimizer_state_dict)
        except Exception as e:
            logger.error(f"Could not prepare optimizer state: {e}")
        return optimizer_state_cpu

    def prepare_buffer_data(self, buffer: "ExperienceBuffer") -> BufferData | None:
        try:
            # --- CHANGED: Get trajectories from tuples ---
            if not hasattr(buffer, "buffer") or not isinstance(buffer.buffer, deque):
                logger.error("Buffer missing 'buffer' deque.")
                return None
            trajectories_list: list[Trajectory] = [
                traj for _, traj in buffer.buffer
            ]  # Extract only trajectories
            # --- END CHANGED ---
            total_steps = buffer.total_steps
            valid_trajectories = []
            actual_steps = 0
            for traj in trajectories_list:
                if isinstance(traj, list) and traj:
                    valid_trajectories.append(traj)
                    actual_steps += len(traj)
                else:
                    logger.warning(
                        "Skipping invalid/empty trajectory during save prep."
                    )
            if actual_steps != total_steps:
                logger.warning(
                    f"Buffer total_steps mismatch: Stored={total_steps}, Calc={actual_steps}. Saving calc value."
                )
                total_steps = actual_steps
            return BufferData(trajectories=valid_trajectories, total_steps=total_steps)
        except Exception as e:
            logger.error(f"Error preparing MuZero buffer data: {e}")
            return None

    def save_config_json(self, configs: dict[str, Any], path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:

                def default_serializer(obj):
                    if isinstance(obj, torch.Tensor | np.ndarray):
                        return "<tensor/array>"
                    if isinstance(obj, deque):
                        return list(obj)
                    try:
                        if hasattr(obj, "__dict__"):
                            return obj.__dict__
                        else:
                            return str(obj)
                    except TypeError:
                        return f"<object type {type(obj).__name__}>"

                json.dump(configs, f, indent=4, default=default_serializer)
                logger.info(f"Run config saved to {path}")
        except Exception as e:
            logger.error(f"Failed save run config JSON {path}: {e}", exc_info=True)
            raise


File: muzerotriangle\data\__init__.py
# File: muzerotriangle/data/__init__.py
"""
Data management module for handling checkpoints, buffers, and potentially logs.
Uses Pydantic schemas for data structure definition.
"""

from .data_manager import DataManager
from .path_manager import PathManager
from .schemas import BufferData, CheckpointData, LoadedTrainingState
from .serializer import Serializer

__all__ = [
    "DataManager",
    "PathManager",
    "Serializer",
    "CheckpointData",
    "BufferData",
    "LoadedTrainingState",
]


File: muzerotriangle\environment\README.md
# File: muzerotriangle/environment/README.md
# Environment Module (`muzerotriangle.environment`)

## Purpose and Architecture

This module defines the game world for AlphaTriangle. It encapsulates the rules, state representation, actions, and core game logic. **Crucially, this module is now independent of any feature extraction logic specific to the neural network.** Its sole focus is the simulation of the game itself.

-   **State Representation:** [`GameState`](core/game_state.py) holds the current board ([`GridData`](grid/grid_data.py)), available shapes (`List[Shape]`), score, and game status. It represents the canonical state of the game. It uses core structures like `Shape` and `Triangle` defined in [`muzerotriangle.structs`](../structs/README.md).
-   **Core Logic:** Submodules ([`grid`](grid/README.md), [`shapes`](shapes/README.md), [`logic`](logic/README.md)) handle specific aspects like checking valid placements, clearing lines, managing shape generation, and calculating rewards. These logic modules operate on `GridData`, `Shape`, and `Triangle`. **Shape refilling now happens in batches: all slots are refilled only when all slots become empty.**
-   **Action Handling:** [`action_codec`](core/action_codec.py) provides functions to convert between a structured action (shape index, row, column) and a single integer representation used by the RL agent and MCTS.
-   **Modularity:** Separating grid logic, shape logic, and core state makes the code easier to understand and modify.

**Note:** Feature extraction (converting `GameState` to NN input tensors) is handled by the separate [`muzerotriangle.features`](../features/README.md) module. Core data structures (`Triangle`, `Shape`) are defined in [`muzerotriangle.structs`](../structs/README.md).

## Exposed Interfaces

-   **Core ([`core/README.md`](core/README.md)):**
    -   `GameState`: The main class representing the environment state.
        -   `reset()`
        -   `step(action_index: ActionType) -> Tuple[float, bool]`
        -   `valid_actions() -> List[ActionType]`
        -   `is_over() -> bool`
        -   `get_outcome() -> float`
        -   `copy() -> GameState`
        -   Public attributes like `grid_data`, `shapes`, `game_score`, `current_step`, etc.
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`
-   **Grid ([`grid/README.md`](grid/README.md)):**
    -   `GridData`: Class holding grid triangle data and line information.
    -   `GridLogic`: Namespace containing functions like `link_neighbors`, `initialize_lines_and_index`, `can_place`, `check_and_clear_lines`.
-   **Shapes ([`shapes/README.md`](shapes/README.md)):**
    -   `ShapeLogic`: Namespace containing functions like `refill_shape_slots`, `generate_random_shape`. **Includes `PREDEFINED_SHAPE_TEMPLATES` constant.**
-   **Logic ([`logic/README.md`](logic/README.md)):**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float` **(Triggers batch refill)**
    -   `calculate_reward(...) -> float` (Used internally by `execute_placement`)
-   **Config:**
    -   `EnvConfig`: Configuration class (re-exported for convenience).

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**:
    -   Uses `EnvConfig` extensively to define grid dimensions, shape slots, etc.
-   **[`muzerotriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `SHAPE_COLORS`, `NO_COLOR_ID`, `DEBUG_COLOR_ID`, `COLOR_TO_ID_MAP`.
-   **[`muzerotriangle.utils`](../utils/README.md)**:
    -   Uses `ActionType`.
-   **`numpy`**:
    -   Used for grid representation (`GridData`).
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`, `copy`.

---

**Note:** Please keep this README updated when changing game rules, state representation, action space, or the module's internal structure (especially the shape refill logic). Accurate documentation is crucial for maintainability.

File: muzerotriangle\environment\__init__.py
"""
Environment module defining the game rules, state, actions, and logic.
This module is now independent of feature extraction for the NN.
"""

from muzerotriangle.config import EnvConfig

from .core.action_codec import decode_action, encode_action
from .core.game_state import GameState
from .grid import logic as GridLogic
from .grid.grid_data import GridData
from .logic.actions import get_valid_actions
from .logic.step import calculate_reward, execute_placement
from .shapes import logic as ShapeLogic

__all__ = [
    # Core
    "GameState",
    "encode_action",
    "decode_action",
    # Grid
    "GridData",
    "GridLogic",
    # Shapes
    "ShapeLogic",
    # Logic
    "get_valid_actions",
    "execute_placement",
    "calculate_reward",
    # Config
    "EnvConfig",
]


File: muzerotriangle\environment\core\action_codec.py
from ...config import EnvConfig
from ...utils.types import ActionType


def encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType:
    """Encodes a (shape_idx, r, c) action into a single integer."""
    if not (0 <= shape_idx < config.NUM_SHAPE_SLOTS):
        raise ValueError(
            f"Invalid shape index: {shape_idx}, must be < {config.NUM_SHAPE_SLOTS}"
        )
    if not (0 <= r < config.ROWS):
        raise ValueError(f"Invalid row index: {r}, must be < {config.ROWS}")
    if not (0 <= c < config.COLS):
        raise ValueError(f"Invalid column index: {c}, must be < {config.COLS}")

    action_index = shape_idx * (config.ROWS * config.COLS) + r * config.COLS + c
    return action_index


def decode_action(action_index: ActionType, config: EnvConfig) -> tuple[int, int, int]:
    """Decodes an integer action into (shape_idx, r, c)."""
    # Cast ACTION_DIM to int for comparison
    action_dim_int = int(config.ACTION_DIM)  # type: ignore[call-overload]
    if not (0 <= action_index < action_dim_int):
        raise ValueError(
            f"Invalid action index: {action_index}, must be < {action_dim_int}"
        )

    grid_size = config.ROWS * config.COLS
    shape_idx = action_index // grid_size
    remainder = action_index % grid_size
    r = remainder // config.COLS
    c = remainder % config.COLS

    return shape_idx, r, c


File: muzerotriangle\environment\core\game_state.py
import logging
import random
from typing import TYPE_CHECKING

from ...config import EnvConfig
from ...utils.types import ActionType
from .. import shapes
from ..grid.grid_data import GridData
from ..logic.actions import get_valid_actions
from ..logic.step import execute_placement
from .action_codec import decode_action

if TYPE_CHECKING:
    from ...structs import Shape


logger = logging.getLogger(__name__)


class GameState:
    """
    Represents the mutable state of the game. Does not handle NN feature extraction
    or visualization/interaction-specific state.
    """

    def __init__(
        self, config: EnvConfig | None = None, initial_seed: int | None = None
    ):
        self.env_config = config if config else EnvConfig()  # type: ignore[call-arg]
        self._rng = (
            random.Random(initial_seed) if initial_seed is not None else random.Random()
        )

        self.grid_data: GridData = None  # type: ignore
        self.shapes: list[Shape | None] = []
        self.game_score: float = 0.0
        self.game_over: bool = False
        self.triangles_cleared_this_episode: int = 0
        self.pieces_placed_this_episode: int = 0
        self.current_step: int = 0

        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.grid_data = GridData(self.env_config)
        self.shapes = [None] * self.env_config.NUM_SHAPE_SLOTS
        self.game_score = 0.0
        self.triangles_cleared_this_episode = 0
        self.pieces_placed_this_episode = 0
        self.game_over = False
        self.current_step = 0

        # Call refill_shape_slots with the updated signature (no index)
        shapes.refill_shape_slots(self, self._rng)

        if not self.valid_actions():
            logger.warning(
                "Game is over immediately after reset (no valid initial moves)."
            )
            self.game_over = True

    def step(self, action_index: ActionType) -> tuple[float, bool]:
        """
        Performs one game step.
        Returns:
            Tuple[float, bool]: (reward, done)
        """
        if self.is_over():
            logger.warning("Attempted to step in a game that is already over.")
            return 0.0, True

        shape_idx, r, c = decode_action(action_index, self.env_config)
        reward = execute_placement(self, shape_idx, r, c, self._rng)
        self.current_step += 1

        if not self.game_over and not self.valid_actions():
            self.game_over = True
            logger.info(f"Game over detected after step {self.current_step}.")

        return reward, self.game_over

    def valid_actions(self) -> list[ActionType]:
        """Returns a list of valid encoded action indices."""
        return get_valid_actions(self)

    def is_over(self) -> bool:
        """Checks if the game is over."""
        return self.game_over

    def get_outcome(self) -> float:
        """Returns the terminal outcome value (e.g., final score). Used by MCTS."""
        if not self.is_over():
            logger.warning("get_outcome() called on a non-terminal state.")
            # Consider returning a default value or raising an error?
            # Returning current score might be misleading for MCTS if not terminal.
            # Let's return 0.0 as a neutral value if not over.
            return 0.0
        return self.game_score

    def copy(self) -> "GameState":
        """Creates a deep copy for simulations (e.g., MCTS)."""
        new_state = GameState.__new__(GameState)
        new_state.env_config = self.env_config
        new_state._rng = random.Random()
        new_state._rng.setstate(self._rng.getstate())
        new_state.grid_data = self.grid_data.deepcopy()
        new_state.shapes = [s.copy() if s else None for s in self.shapes]
        new_state.game_score = self.game_score
        new_state.game_over = self.game_over
        new_state.triangles_cleared_this_episode = self.triangles_cleared_this_episode
        new_state.pieces_placed_this_episode = self.pieces_placed_this_episode
        new_state.current_step = self.current_step
        return new_state

    def __str__(self) -> str:
        shape_strs = [str(s) if s else "None" for s in self.shapes]
        return f"GameState(Step:{self.current_step}, Score:{self.game_score:.1f}, Over:{self.is_over()}, Shapes:[{', '.join(shape_strs)}])"


File: muzerotriangle\environment\core\README.md
# File: muzerotriangle/environment/core/README.md
# Environment Core Submodule (`muzerotriangle.environment.core`)

## Purpose and Architecture

This submodule contains the most fundamental components of the game environment: the [`GameState`](game_state.py) class and the [`action_codec`](action_codec.py).

-   **`GameState`:** This class acts as the central hub for the environment's state. It holds references to the [`GridData`](../grid/grid_data.py), the current shapes, score, game status, and other relevant information. It provides the primary interface (`reset`, `step`, `valid_actions`, `is_over`, `get_outcome`, `copy`) for agents (like MCTS or self-play workers) to interact with the game. It delegates specific logic (like placement validation, line clearing, shape generation) to other submodules ([`grid`](../grid/README.md), [`shapes`](../shapes/README.md), [`logic`](../logic/README.md)).
-   **`action_codec`:** Provides simple, stateless functions (`encode_action`, `decode_action`) to translate between the agent's integer action representation and the game's internal representation (shape index, row, column). This decouples the agent's action space from the internal game logic.

## Exposed Interfaces

-   **Classes:**
    -   `GameState`: The main state class (see [`muzerotriangle/environment/README.md`](../README.md) for methods).
-   **Functions:**
    -   `encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType`
    -   `decode_action(action_index: ActionType, config: EnvConfig) -> Tuple[int, int, int]`

## Dependencies

-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used by `GameState` and `action_codec`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**:
    -   `ActionType`: Used for method signatures and return types.
-   **[`muzerotriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used internally by `GameState`.
-   **[`muzerotriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used internally by `GameState`.
-   **[`muzerotriangle.environment.logic`](../logic/README.md)**:
    -   `get_valid_actions`, `execute_placement`: Used internally by `GameState`.
-   **Standard Libraries:** `typing`, `numpy`, `logging`, `random`.

---

**Note:** Please keep this README updated when modifying the core `GameState` interface or the action encoding/decoding scheme. Accurate documentation is crucial for maintainability.

File: muzerotriangle\environment\core\__init__.py


File: muzerotriangle\environment\grid\grid_data.py
# File: muzerotriangle/environment/grid/grid_data.py
import copy
import logging

import numpy as np

from ...config import EnvConfig
from ...structs import NO_COLOR_ID

logger = logging.getLogger(__name__)


def _precompute_lines(config: EnvConfig) -> list[list[tuple[int, int]]]:
    """
    Generates all potential horizontal and diagonal lines based on grid geometry.
    Returns a list of lines, where each line is a list of (row, col) tuples.
    This function no longer needs actual Triangle objects.
    """
    lines = []
    rows, cols = config.ROWS, config.COLS
    min_len = config.MIN_LINE_LENGTH

    # --- Determine playable cells based on config ---
    playable_mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        playable_width = config.COLS_PER_ROW[r]
        padding = cols - playable_width
        pad_left = padding // 2
        playable_start_col = pad_left
        playable_end_col = pad_left + playable_width
        for c in range(cols):
            if playable_start_col <= c < playable_end_col:
                playable_mask[r, c] = True
    # --- End Playable Mask ---

    # Helper to check validity and playability
    def is_valid_playable(r, c):
        return 0 <= r < rows and 0 <= c < cols and playable_mask[r, c]

    # --- Trace Lines using Coordinates ---
    visited_in_line: set[tuple[int, int, str]] = set()  # (r, c, direction)

    for r_start in range(rows):
        for c_start in range(cols):
            if not is_valid_playable(r_start, c_start):
                continue

            # --- Trace Horizontal ---
            if (r_start, c_start, "h") not in visited_in_line:
                current_line_h = []
                # Trace left
                cr, cc = r_start, c_start
                while is_valid_playable(cr, cc - 1):
                    cc -= 1
                # Trace right from the start
                while is_valid_playable(cr, cc):
                    if (cr, cc, "h") not in visited_in_line:
                        current_line_h.append((cr, cc))
                        visited_in_line.add((cr, cc, "h"))
                    else:
                        # If we hit a visited cell, the rest of the line was already processed
                        break
                    cc += 1
                if len(current_line_h) >= min_len:
                    lines.append(current_line_h)

            # --- Trace Diagonal TL-BR (Down-Right) ---
            if (r_start, c_start, "d1") not in visited_in_line:
                current_line_d1 = []
                # Trace backwards (Up-Left)
                cr, cc = r_start, c_start
                while True:
                    is_up = (cr + cc) % 2 != 0
                    prev_r, prev_c = (cr, cc - 1) if is_up else (cr - 1, cc)
                    if is_valid_playable(prev_r, prev_c):
                        cr, cc = prev_r, prev_c
                    else:
                        break
                # Trace forwards
                while is_valid_playable(cr, cc):
                    if (cr, cc, "d1") not in visited_in_line:
                        current_line_d1.append((cr, cc))
                        visited_in_line.add((cr, cc, "d1"))
                    else:
                        break
                    is_up = (cr + cc) % 2 != 0
                    next_r, next_c = (cr + 1, cc) if is_up else (cr, cc + 1)
                    cr, cc = next_r, next_c
                if len(current_line_d1) >= min_len:
                    lines.append(current_line_d1)

            # --- Trace Diagonal BL-TR (Up-Right) ---
            if (r_start, c_start, "d2") not in visited_in_line:
                current_line_d2 = []
                # Trace backwards (Down-Left)
                cr, cc = r_start, c_start
                while True:
                    is_up = (cr + cc) % 2 != 0
                    prev_r, prev_c = (cr + 1, cc) if is_up else (cr, cc - 1)
                    if is_valid_playable(prev_r, prev_c):
                        cr, cc = prev_r, prev_c
                    else:
                        break
                # Trace forwards
                while is_valid_playable(cr, cc):
                    if (cr, cc, "d2") not in visited_in_line:
                        current_line_d2.append((cr, cc))
                        visited_in_line.add((cr, cc, "d2"))
                    else:
                        break
                    is_up = (cr + cc) % 2 != 0
                    next_r, next_c = (cr, cc + 1) if is_up else (cr - 1, cc)
                    cr, cc = next_r, next_c
                if len(current_line_d2) >= min_len:
                    lines.append(current_line_d2)
    # --- End Line Tracing ---

    # Remove duplicates (lines traced from different start points)
    unique_lines_tuples = {tuple(sorted(line)) for line in lines}
    unique_lines = [list(line_tuple) for line_tuple in unique_lines_tuples]

    # Final filter by length (should be redundant but safe)
    final_lines = [line for line in unique_lines if len(line) >= min_len]

    return final_lines


class GridData:
    """
    Holds the grid state using NumPy arrays for occupancy, death zones, and color IDs.
    Manages precomputed line information based on coordinates.
    """

    def __init__(self, config: EnvConfig):
        self.rows = config.ROWS
        self.cols = config.COLS
        self.config = config

        # --- NumPy Array State ---
        self._occupied_np: np.ndarray = np.zeros((self.rows, self.cols), dtype=bool)
        self._death_np: np.ndarray = np.zeros((self.rows, self.cols), dtype=bool)
        # Stores color ID, NO_COLOR_ID (-1) means empty/no color
        self._color_id_np: np.ndarray = np.full(
            (self.rows, self.cols), NO_COLOR_ID, dtype=np.int8
        )
        # --- End NumPy Array State ---

        self._initialize_death_zone(config)
        self._occupied_np[self._death_np] = True  # Death cells are considered occupied

        # --- Line Information (Coordinate Based) ---
        # Stores frozensets of (r, c) tuples
        self.potential_lines: set[frozenset[tuple[int, int]]] = set()
        # Maps (r, c) tuple to a set of line frozensets it belongs to
        self._coord_to_lines_map: dict[
            tuple[int, int], set[frozenset[tuple[int, int]]]
        ] = {}
        # --- End Line Information ---

        self._initialize_lines_and_index()
        logger.debug(
            f"GridData initialized ({self.rows}x{self.cols}) using NumPy arrays. Found {len(self.potential_lines)} potential lines."
        )

    def _initialize_death_zone(self, config: EnvConfig):
        """Initializes the death zone numpy array."""
        cols_per_row = config.COLS_PER_ROW
        if len(cols_per_row) != self.rows:
            raise ValueError(
                f"COLS_PER_ROW length mismatch: {len(cols_per_row)} vs {self.rows}"
            )

        for r in range(self.rows):
            playable_width = cols_per_row[r]
            padding = self.cols - playable_width
            pad_left = padding // 2
            playable_start_col = pad_left
            playable_end_col = pad_left + playable_width
            for c in range(self.cols):
                if not (playable_start_col <= c < playable_end_col):
                    self._death_np[r, c] = True

    def _initialize_lines_and_index(self) -> None:
        """
        Precomputes potential lines (as coordinate sets) and creates a map
        from coordinates to the lines they belong to.
        """
        self.potential_lines = set()
        self._coord_to_lines_map = {}

        potential_lines_coords = _precompute_lines(self.config)

        for line_coords in potential_lines_coords:
            # Filter out lines containing death cells
            valid_line = True
            line_coord_set: set[tuple[int, int]] = set()
            for r, c in line_coords:
                # Use self.valid() and self._death_np directly
                if self.valid(r, c) and not self._death_np[r, c]:
                    line_coord_set.add((r, c))
                else:
                    valid_line = False
                    break  # Skip this line if any part is invalid/death

            if valid_line and len(line_coord_set) >= self.config.MIN_LINE_LENGTH:
                frozen_line = frozenset(line_coord_set)
                self.potential_lines.add(frozen_line)
                # Add to the reverse map
                for coord in line_coord_set:
                    if coord not in self._coord_to_lines_map:
                        self._coord_to_lines_map[coord] = set()
                    self._coord_to_lines_map[coord].add(frozen_line)

        logger.debug(
            f"Initialized {len(self.potential_lines)} potential lines and mapping for {len(self._coord_to_lines_map)} coordinates."
        )

    def valid(self, r: int, c: int) -> bool:
        """Checks if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_death(self, r: int, c: int) -> bool:
        """Checks if a cell is a death cell."""
        if not self.valid(r, c):
            return True  # Out of bounds is considered death
        # Cast NumPy bool_ to Python bool for type consistency
        return bool(self._death_np[r, c])

    def is_occupied(self, r: int, c: int) -> bool:
        """Checks if a cell is occupied (includes death cells)."""
        if not self.valid(r, c):
            return True  # Out of bounds is considered occupied
        # Cast NumPy bool_ to Python bool for type consistency
        return bool(self._occupied_np[r, c])

    def get_color_id(self, r: int, c: int) -> int:
        """Gets the color ID of a cell."""
        if not self.valid(r, c):
            return NO_COLOR_ID
        # Cast NumPy int8 to Python int for type consistency
        return int(self._color_id_np[r, c])

    def get_occupied_state(self) -> np.ndarray:
        """Returns a copy of the occupancy numpy array."""
        return self._occupied_np.copy()

    def get_death_state(self) -> np.ndarray:
        """Returns a copy of the death zone numpy array."""
        return self._death_np.copy()

    def get_color_id_state(self) -> np.ndarray:
        """Returns a copy of the color ID numpy array."""
        return self._color_id_np.copy()

    def deepcopy(self) -> "GridData":
        """
        Creates a deep copy of the grid data using NumPy array copying
        and standard dictionary/set copying for line data.
        """
        new_grid = GridData.__new__(
            GridData
        )  # Create new instance without calling __init__
        new_grid.rows = self.rows
        new_grid.cols = self.cols
        new_grid.config = self.config  # Config is likely immutable, shallow copy ok

        # 1. Copy NumPy arrays
        new_grid._occupied_np = self._occupied_np.copy()
        new_grid._death_np = self._death_np.copy()
        new_grid._color_id_np = self._color_id_np.copy()

        # 2. Copy Line Data (Set of frozensets and Dict[Tuple, Set[frozenset]])
        # potential_lines contains immutable frozensets, shallow copy is fine
        new_grid.potential_lines = self.potential_lines.copy()
        # _coord_to_lines_map values are sets, need deepcopy
        new_grid._coord_to_lines_map = copy.deepcopy(self._coord_to_lines_map)

        # No Triangle objects or neighbors to handle anymore

        return new_grid

    def __str__(self) -> str:
        # Basic representation, could be enhanced to show grid visually
        occupied_count = np.sum(self._occupied_np & ~self._death_np)
        return f"GridData({self.rows}x{self.cols}, Occupied: {occupied_count})"


File: muzerotriangle\environment\grid\logic.py
# File: muzerotriangle/environment/grid/logic.py
import logging
from typing import TYPE_CHECKING

# Import NO_COLOR_ID from the structs package directly
from ...structs import NO_COLOR_ID

if TYPE_CHECKING:
    from ...structs import Shape
    from .grid_data import GridData

logger = logging.getLogger(__name__)


# Removed link_neighbors function as it's no longer needed


def can_place(grid_data: "GridData", shape: "Shape", r: int, c: int) -> bool:
    """
    Checks if a shape can be placed at the specified (r, c) top-left position
    on the grid, considering occupancy, death zones, and triangle orientation.
    Reads state from GridData's NumPy arrays.
    """
    if not shape or not shape.triangles:
        return False

    for dr, dc, is_up_shape in shape.triangles:
        tri_r, tri_c = r + dr, c + dc

        # Check bounds and death zone first
        if not grid_data.valid(tri_r, tri_c) or grid_data._death_np[tri_r, tri_c]:
            return False

        # Check occupancy
        if grid_data._occupied_np[tri_r, tri_c]:
            return False

        # Check orientation match
        is_up_grid = (tri_r + tri_c) % 2 != 0
        if is_up_grid != is_up_shape:
            # Log the mismatch for debugging the test failure
            logger.debug(
                f"Orientation mismatch at ({tri_r},{tri_c}): Grid is {'Up' if is_up_grid else 'Down'}, Shape requires {'Up' if is_up_shape else 'Down'}"
            )
            return False

    return True


def check_and_clear_lines(
    grid_data: "GridData", newly_occupied_coords: set[tuple[int, int]]
) -> tuple[int, set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    """
    Checks for completed lines involving the newly occupied coordinates and clears them.
    Operates on GridData's NumPy arrays.

    Args:
        grid_data: The GridData object (will be modified).
        newly_occupied_coords: A set of (r, c) tuples that were just occupied.

    Returns:
        Tuple containing:
            - int: Number of lines cleared.
            - set[tuple[int, int]]: Set of unique (r, c) coordinates cleared.
            - set[frozenset[tuple[int, int]]]: Set containing the frozenset representations
                                                of the actual lines that were cleared.
    """
    lines_to_check: set[frozenset[tuple[int, int]]] = set()
    for coord in newly_occupied_coords:
        if coord in grid_data._coord_to_lines_map:
            lines_to_check.update(grid_data._coord_to_lines_map[coord])

    cleared_lines_set: set[frozenset[tuple[int, int]]] = set()
    unique_coords_cleared: set[tuple[int, int]] = set()

    if not lines_to_check:
        return 0, unique_coords_cleared, cleared_lines_set

    logger.debug(f"Checking {len(lines_to_check)} potential lines for completion.")

    for line_coords_fs in lines_to_check:
        is_complete = True
        for r_line, c_line in line_coords_fs:
            # Check occupancy directly from the NumPy array
            if not grid_data._occupied_np[r_line, c_line]:
                is_complete = False
                break

        if is_complete:
            logger.debug(f"Line completed: {line_coords_fs}")
            cleared_lines_set.add(line_coords_fs)
            # Add coordinates from this cleared line to the set of unique cleared coordinates
            unique_coords_cleared.update(line_coords_fs)

    if unique_coords_cleared:
        logger.info(
            f"Clearing {len(cleared_lines_set)} lines involving {len(unique_coords_cleared)} unique coordinates."
        )
        # Update NumPy arrays for cleared coordinates
        # Convert set to tuple of arrays for advanced indexing
        if unique_coords_cleared:  # Ensure set is not empty
            rows_idx, cols_idx = zip(*unique_coords_cleared, strict=False)
            grid_data._occupied_np[rows_idx, cols_idx] = False
            grid_data._color_id_np[rows_idx, cols_idx] = NO_COLOR_ID

    return len(cleared_lines_set), unique_coords_cleared, cleared_lines_set


File: muzerotriangle\environment\grid\README.md
# File: muzerotriangle/environment/grid/README.md
# Environment Grid Submodule (`muzerotriangle.environment.grid`)

## Purpose and Architecture

This submodule manages the game's grid structure and related logic. It defines the triangular cells, their properties, relationships, and operations like placement validation and line clearing.

-   **Cell Representation:** The `Triangle` class (defined in [`muzerotriangle.structs`](../../structs/README.md)) represents a single cell, storing its position and orientation (`is_up`). The actual state (occupied, death, color) is managed within `GridData`.
-   **Grid Data Structure:** The [`GridData`](grid_data.py) class holds the grid state using efficient `numpy` arrays (`_occupied_np`, `_death_np`, `_color_id_np`). It also manages precomputed information about potential lines (sets of coordinates) for efficient clearing checks.
-   **Grid Logic:** The [`logic.py`](logic.py) module (exposed as `GridLogic`) contains functions operating on `GridData`. This includes:
    -   Initializing the grid based on `EnvConfig` (defining death zones).
    -   Precomputing potential lines (`_precompute_lines`) and indexing them (`initialize_lines_and_index`) for efficient checking.
    -   Checking if a shape can be placed (`can_place`), **including matching triangle orientations**.
    -   Checking for and clearing completed lines (`check_and_clear_lines`). **This function does NOT implement gravity.**
-   **Grid Features:** Note: The `grid_features.py` module, which previously provided functions to calculate scalar metrics (heights, holes, bumpiness), has been **moved** to the top-level [`muzerotriangle.features`](../../features/README.md) module (`muzerotriangle/features/grid_features.py`) as part of decoupling feature extraction from the core environment.

## Exposed Interfaces

-   **Classes:**
    -   `GridData`: Holds the grid state using NumPy arrays.
        -   `__init__(config: EnvConfig)`
        -   `valid(r: int, c: int) -> bool`
        -   `is_death(r: int, c: int) -> bool`
        -   `is_occupied(r: int, c: int) -> bool`
        -   `get_color_id(r: int, c: int) -> int`
        -   `get_occupied_state() -> np.ndarray`
        -   `get_death_state() -> np.ndarray`
        -   `get_color_id_state() -> np.ndarray`
        -   `deepcopy() -> GridData`
-   **Modules/Namespaces:**
    -   `logic` (often imported as `GridLogic`):
        -   `initialize_lines_and_index(grid_data: GridData)`
        -   `can_place(grid_data: GridData, shape: Shape, r: int, c: int) -> bool`
        -   `check_and_clear_lines(grid_data: GridData, newly_occupied_coords: Set[Tuple[int, int]]) -> Tuple[int, Set[Tuple[int, int]], Set[frozenset[Tuple[int, int]]]]` **(Returns: lines_cleared_count, unique_coords_cleared_set, set_of_cleared_lines_coord_sets)**

## Dependencies

-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used by `GridData` initialization and logic functions.
-   **[`muzerotriangle.structs`](../../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `NO_COLOR_ID`.
-   **`numpy`**:
    -   Used extensively in `GridData`.
-   **Standard Libraries:** `typing`, `logging`, `numpy`, `copy`.

---

**Note:** Please keep this README updated when changing the grid structure, cell properties, placement rules, or line clearing logic. Accurate documentation is crucial for maintainability.

File: muzerotriangle\environment\grid\triangle.py
class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up
        self.is_death = is_death
        self.is_occupied = is_death
        self.color: tuple[int, int, int] | None = None

        self.neighbor_left: Triangle | None = None
        self.neighbor_right: Triangle | None = None
        self.neighbor_vert: Triangle | None = None

    def get_points(
        self, ox: float, oy: float, cw: float, ch: float
    ) -> list[tuple[float, float]]:
        """Calculates vertex points for drawing, relative to origin (ox, oy)."""
        x = ox + self.col * (cw * 0.75)
        y = oy + self.row * ch
        if self.is_up:
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def copy(self) -> "Triangle":
        """Creates a copy of the Triangle object's state (neighbors are not copied)."""
        new_tri = Triangle(self.row, self.col, self.is_up, self.is_death)
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color
        return new_tri

    def __repr__(self) -> str:
        state = "D" if self.is_death else ("O" if self.is_occupied else ".")
        orient = "^" if self.is_up else "v"
        return f"T({self.row},{self.col} {orient}{state})"

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.row == other.row and self.col == other.col


File: muzerotriangle\environment\grid\__init__.py
# File: muzerotriangle/environment/grid/__init__.py
"""
Grid submodule handling the triangular grid structure, data, and logic.
"""

# Removed: from .triangle import Triangle
from . import logic
from .grid_data import GridData

# DO NOT import grid_features here. It has been moved up one level
# to muzerotriangle/environment/grid_features.py to break circular dependencies.

__all__ = [
    "GridData",
    "logic",
]


File: muzerotriangle\environment\logic\actions.py
import logging
from typing import TYPE_CHECKING

from ..core.action_codec import encode_action
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ...utils.types import ActionType
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def get_valid_actions(state: "GameState") -> list["ActionType"]:
    """
    Calculates and returns a list of all valid encoded action indices
    for the current game state.
    """
    valid_actions: list[ActionType] = []
    for shape_idx, shape in enumerate(state.shapes):
        if shape is None:
            continue

        for r in range(state.env_config.ROWS):
            for c in range(state.env_config.COLS):
                if GridLogic.can_place(state.grid_data, shape, r, c):
                    action_index = encode_action(shape_idx, r, c, state.env_config)
                    valid_actions.append(action_index)

    return valid_actions


File: muzerotriangle\environment\logic\README.md
# File: muzerotriangle/environment/logic/README.md
# Environment Logic Submodule (`muzerotriangle.environment.logic`)

## Purpose and Architecture

This submodule contains higher-level game logic that operates on the `GameState` and its components (`GridData`, `Shape`). It bridges the gap between basic actions/rules and the overall game flow.

-   **`actions.py`:**
    -   `get_valid_actions`: Determines all possible valid moves (shape placements) from the current `GameState` by iterating through available shapes and grid positions, checking placement validity using [`GridLogic.can_place`](../grid/logic.py). Returns a list of encoded `ActionType` integers.
-   **`step.py`:**
    -   `execute_placement`: Performs the core logic when a shape is placed. It updates the `GridData` (occupancy and color), checks for and clears completed lines using [`GridLogic.check_and_clear_lines`](../grid/logic.py), calculates the reward for the step using `calculate_reward`, updates the game score and step counters, and **triggers a batch refill of shape slots using [`ShapeLogic.refill_shape_slots`](../shapes/logic.py) only if all slots become empty after the placement.**
    -   `calculate_reward`: Calculates the reward based on the number of triangles placed, triangles cleared, and whether the game ended.

## Exposed Interfaces

-   **Functions:**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float`
    -   `calculate_reward(placed_count: int, unique_coords_cleared: Set[Tuple[int, int]], is_game_over: bool, config: EnvConfig) -> float`

## Dependencies

-   **[`muzerotriangle.environment.core`](../core/README.md)**:
    -   `GameState`: The primary object operated upon.
    -   `ActionCodec`: Used by `get_valid_actions`.
-   **[`muzerotriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used for placement checks and line clearing.
-   **[`muzerotriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used for shape refilling.
-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used for reward calculation and action encoding.
-   **[`muzerotriangle.structs`](../../structs/README.md)**:
    -   `Shape`, `Triangle`, `COLOR_TO_ID_MAP`, `NO_COLOR_ID`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**:
    -   `ActionType`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the logic for determining valid actions, executing placements (including reward calculation and shape refilling), or modifying dependencies.

File: muzerotriangle\environment\logic\step.py
# File: muzerotriangle/environment/logic/step.py
import logging
import random
from typing import TYPE_CHECKING

# Correct import path for constants
from ...structs.constants import COLOR_TO_ID_MAP, NO_COLOR_ID
from .. import shapes as ShapeLogic

# Import the logic submodule correctly
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ...config import EnvConfig
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def calculate_reward(
    placed_count: int,
    unique_coords_cleared: set[tuple[int, int]],
    is_game_over: bool,
    config: "EnvConfig",
) -> float:
    """
    Calculates the step reward based on the new specification (v3).

    Args:
        placed_count: Number of triangles successfully placed.
        unique_coords_cleared: Set of unique (r, c) coordinates cleared this step.
        is_game_over: Boolean indicating if the game ended *after* this step.
        config: Environment configuration containing reward constants.

    Returns:
        The calculated step reward.
    """
    reward = 0.0

    # 1. Placement Reward
    reward += placed_count * config.REWARD_PER_PLACED_TRIANGLE

    # 2. Line Clear Reward
    reward += len(unique_coords_cleared) * config.REWARD_PER_CLEARED_TRIANGLE

    # 3. Survival Reward OR Game Over Penalty
    if is_game_over:
        reward += config.PENALTY_GAME_OVER
    else:
        reward += config.REWARD_PER_STEP_ALIVE

    logger.debug(
        f"Calculated Reward: Placement({placed_count * config.REWARD_PER_PLACED_TRIANGLE:.3f}) "
        f"+ LineClear({len(unique_coords_cleared) * config.REWARD_PER_CLEARED_TRIANGLE:.3f}) "
        f"+ {'GameOver' if is_game_over else 'Survival'}({config.PENALTY_GAME_OVER if is_game_over else config.REWARD_PER_STEP_ALIVE:.3f}) "
        f"= {reward:.3f}"
    )
    return reward


def execute_placement(
    game_state: "GameState", shape_idx: int, r: int, c: int, rng: random.Random
) -> float:
    """
    Places a shape, clears lines, updates game state (NumPy arrays), and calculates reward.
    Handles batch refilling of shapes.

    Args:
        game_state: The current game state (will be modified).
        shape_idx: Index of the shape to place.
        r: Target row for placement.
        c: Target column for placement.
        rng: Random number generator for shape refilling.

    Returns:
        The reward obtained for this step.
    """
    shape = game_state.shapes[shape_idx]
    if not shape:
        logger.error(f"Attempted to place an empty shape slot: {shape_idx}")
        return 0.0

    # Use the NumPy-based can_place from GridLogic
    if not GridLogic.can_place(game_state.grid_data, shape, r, c):
        logger.error(f"Invalid placement attempted: Shape {shape_idx} at ({r},{c})")
        # It's possible this check fails even if valid_actions included it,
        # especially if the state changed unexpectedly (e.g., in multi-threaded envs, though not the case here).
        # Returning 0 reward is reasonable.
        return 0.0

    # --- Place the shape ---
    placed_coords: set[tuple[int, int]] = set()
    placed_count = 0
    # Get color ID from the shape's color
    color_id = COLOR_TO_ID_MAP.get(shape.color, NO_COLOR_ID)
    if color_id == NO_COLOR_ID:
        logger.warning(f"Shape color {shape.color} not found in COLOR_TO_ID_MAP!")
        # Assign a default color ID? Or handle as error? Let's use 0 for now.
        color_id = 0

    for dr, dc, _ in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        # Check validity using GridData method (which checks bounds)
        if game_state.grid_data.valid(tri_r, tri_c):
            # Check death and occupancy using NumPy arrays
            if (
                not game_state.grid_data._death_np[tri_r, tri_c]
                and not game_state.grid_data._occupied_np[tri_r, tri_c]
            ):
                # Update NumPy arrays
                game_state.grid_data._occupied_np[tri_r, tri_c] = True
                game_state.grid_data._color_id_np[tri_r, tri_c] = color_id
                placed_coords.add((tri_r, tri_c))
                placed_count += 1
            else:
                # This case should ideally not be reached if can_place passed. Log if it does.
                logger.error(
                    f"Placement conflict at ({tri_r},{tri_c}) during execution, though can_place was true."
                )
        else:
            # This case should ideally not be reached if can_place passed. Log if it does.
            logger.error(
                f"Invalid coordinates ({tri_r},{tri_c}) encountered during placement execution."
            )

    game_state.shapes[shape_idx] = None  # Remove shape from slot
    game_state.pieces_placed_this_episode += 1

    # --- Check and clear lines ---
    # Use check_and_clear_lines from GridLogic
    lines_cleared_count, unique_coords_cleared, _ = GridLogic.check_and_clear_lines(
        game_state.grid_data, placed_coords
    )
    game_state.triangles_cleared_this_episode += len(unique_coords_cleared)

    # --- Update Score (Optional tracking) ---
    game_state.game_score += placed_count + len(unique_coords_cleared) * 2

    # --- Refill shapes if all slots are empty ---
    if all(s is None for s in game_state.shapes):
        logger.debug("All shape slots empty, triggering batch refill.")
        ShapeLogic.refill_shape_slots(game_state, rng)

    # --- Check for game over AFTER placement and refill ---
    # Game is over if no valid moves remain for the *new* state
    if not game_state.valid_actions():
        game_state.game_over = True
        logger.info(
            f"Game over detected after placing shape {shape_idx} and potential refill."
        )

    # --- Calculate Reward based on the outcome of this step ---
    step_reward = calculate_reward(
        placed_count=placed_count,
        unique_coords_cleared=unique_coords_cleared,  # Pass the set of cleared coords
        is_game_over=game_state.game_over,
        config=game_state.env_config,
    )

    return step_reward


File: muzerotriangle\environment\logic\__init__.py


File: muzerotriangle\environment\shapes\logic.py
# File: muzerotriangle/environment/shapes/logic.py
import logging
import random
from typing import TYPE_CHECKING

from ...structs import SHAPE_COLORS, Shape
from .templates import PREDEFINED_SHAPE_TEMPLATES

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def generate_random_shape(rng: random.Random) -> Shape:
    """Generates a random shape from predefined templates and colors."""
    template = rng.choice(PREDEFINED_SHAPE_TEMPLATES)
    color = rng.choice(SHAPE_COLORS)
    return Shape(template, color)


def refill_shape_slots(game_state: "GameState", rng: random.Random) -> None:
    """
    Refills ALL empty shape slots in the GameState, but ONLY if ALL slots are currently empty.
    This implements batch refilling.
    """
    # --- CHANGED: Check if ALL slots are None ---
    if all(shape is None for shape in game_state.shapes):
        logger.debug("All shape slots are empty. Refilling all slots.")
        for i in range(game_state.env_config.NUM_SHAPE_SLOTS):
            game_state.shapes[i] = generate_random_shape(rng)
            logger.debug(f"Refilled slot {i} with {game_state.shapes[i]}")
    else:
        logger.debug("Not all shape slots are empty. Skipping refill.")
    # --- END CHANGED ---


def get_neighbors(r: int, c: int, is_up: bool) -> list[tuple[int, int]]:
    """Gets potential neighbor coordinates for connectivity check."""
    if is_up:
        # Up triangle neighbors: (r, c-1), (r, c+1), (r+1, c)
        return [(r, c - 1), (r, c + 1), (r + 1, c)]
    else:
        # Down triangle neighbors: (r, c-1), (r, c+1), (r-1, c)
        return [(r, c - 1), (r, c + 1), (r - 1, c)]


def is_shape_connected(shape: Shape) -> bool:
    """Checks if all triangles in a shape are connected."""
    if not shape.triangles or len(shape.triangles) <= 1:
        return True

    coords_set = {(r, c) for r, c, _ in shape.triangles}
    start_node = shape.triangles[0][:2]  # (r, c) of the first triangle
    visited: set[tuple[int, int]] = set()
    queue = [start_node]
    visited.add(start_node)

    while queue:
        current_r, current_c = queue.pop(0)
        # Find the orientation of the current triangle in the shape list
        current_is_up = False
        for r, c, is_up in shape.triangles:
            if r == current_r and c == current_c:
                current_is_up = is_up
                break

        for nr, nc in get_neighbors(current_r, current_c, current_is_up):
            neighbor_coord = (nr, nc)
            if neighbor_coord in coords_set and neighbor_coord not in visited:
                visited.add(neighbor_coord)
                queue.append(neighbor_coord)

    return len(visited) == len(coords_set)


File: muzerotriangle\environment\shapes\README.md
# File: muzerotriangle/environment/shapes/README.md
# Environment Shapes Submodule (`muzerotriangle.environment.shapes`)

## Purpose and Architecture

This submodule defines the logic for managing placeable shapes within the game environment.

-   **Shape Representation:** The `Shape` class (defined in [`muzerotriangle.structs`](../../structs/README.md)) stores the geometry of a shape as a list of relative triangle coordinates (`(dr, dc, is_up)`) and its color.
-   **Shape Templates:** The [`templates.py`](templates.py) file contains the `PREDEFINED_SHAPE_TEMPLATES` list, which defines the geometry of all possible shapes used in the game. **This list should not be modified.**
-   **Shape Logic:** The [`logic.py`](logic.py) module (exposed as `ShapeLogic`) contains functions related to shapes:
    -   `generate_random_shape`: Creates a new `Shape` instance by randomly selecting a template from `PREDEFINED_SHAPE_TEMPLATES` and assigning a random color (using `SHAPE_COLORS` from [`muzerotriangle.structs`](../../structs/README.md)).
    -   `refill_shape_slots`: **Refills ALL empty shape slots** in the `GameState`, but **only if ALL slots are currently empty**. This implements batch refilling.

## Exposed Interfaces

-   **Modules/Namespaces:**
    -   `logic` (often imported as `ShapeLogic`):
        -   `generate_random_shape(rng: random.Random) -> Shape`
        -   `refill_shape_slots(game_state: GameState, rng: random.Random)` **(Refills all slots only if all are empty)**
-   **Constants:**
    -   `PREDEFINED_SHAPE_TEMPLATES` (from `templates.py`): The list of shape geometries.

## Dependencies

-   **[`muzerotriangle.environment.core`](../core/README.md)**:
    -   `GameState`: Used by `ShapeLogic.refill_shape_slots` to access and modify the list of available shapes.
-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Accessed via `GameState` (e.g., for `NUM_SHAPE_SLOTS`).
-   **[`muzerotriangle.structs`](../../structs/README.md)**:
    -   Uses `Shape`, `Triangle`, `SHAPE_COLORS`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the shape generation algorithm or the logic for managing shape slots in the game state (especially the batch refill mechanism). Accurate documentation is crucial for maintainability. **Do not modify `templates.py`.**

File: muzerotriangle\environment\shapes\shape.py
class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(
        self, triangles: list[tuple[int, int, bool]], color: tuple[int, int, int]
    ):
        self.triangles: list[tuple[int, int, bool]] = triangles
        self.color: tuple[int, int, int] = color

    def bbox(self) -> tuple[int, int, int, int]:
        """Calculates bounding box (min_r, min_c, max_r, max_c) in relative coords."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))

    def copy(self) -> "Shape":
        """Creates a shallow copy (triangle list is copied, color is shared)."""
        new_shape = Shape.__new__(Shape)
        new_shape.triangles = list(self.triangles)
        new_shape.color = self.color
        return new_shape

    def __str__(self) -> str:
        return f"Shape(Color:{self.color}, Tris:{len(self.triangles)})"


File: muzerotriangle\environment\shapes\templates.py
# ==============================================================================
# ==                    PREDEFINED SHAPE TEMPLATES                          ==
# ==                                                                        ==
# ==    DO NOT MODIFY THIS LIST MANUALLY unless you are absolutely sure!    ==
# == These shapes are fundamental to the game's design and balance.         ==
# == Modifying them can have unintended consequences on gameplay and agent  ==
# == training.                                                              ==
# ==============================================================================

# List of predefined shape templates. Each template is a list of relative triangle coordinates (dr, dc, is_up).
# Coordinates are relative to the shape's origin (typically the top-leftmost triangle).
# is_up = True for upward-pointing triangle, False for downward-pointing.
PREDEFINED_SHAPE_TEMPLATES: list[list[tuple[int, int, bool]]] = [
    [  # Shape 1
        (
            0,
            0,
            True,
        )
    ],
    [  # Shape 1
        (
            0,
            0,
            True,
        )
    ],
    [  # Shape 2
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 2
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 3
        (
            0,
            0,
            False,
        )
    ],
    [  # Shape 4
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
    ],
    [  # Shape 4
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
    ],
    [  # Shape 5
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
    ],
    [  # Shape 5
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
    ],
    [  # Shape 6
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
    ],
    [  # Shape 7
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            0,
            2,
            False,
        ),
    ],
    [  # Shape 8
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 9
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 10
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            0,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 11
        (
            0,
            0,
            True,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 12
        (
            0,
            0,
            True,
        ),
        (
            1,
            -2,
            False,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 13
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 14
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 15
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 16
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 17
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 18
        (
            0,
            0,
            True,
        ),
        (
            0,
            2,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 19
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
        (
            1,
            2,
            False,
        ),
    ],
    [  # Shape 20
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 21
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 22
        (
            0,
            0,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 23
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
        (
            1,
            1,
            True,
        ),
    ],
    [  # Shape 24
        (
            0,
            0,
            True,
        ),
        (
            1,
            -1,
            True,
        ),
        (
            1,
            0,
            False,
        ),
    ],
    [  # Shape 25
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            0,
            2,
            False,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 26
        (
            0,
            0,
            False,
        ),
        (
            0,
            1,
            True,
        ),
        (
            1,
            1,
            False,
        ),
    ],
    [  # Shape 27
        (
            0,
            0,
            True,
        ),
        (
            0,
            1,
            False,
        ),
        (
            1,
            0,
            False,
        ),
    ],
]


File: muzerotriangle\environment\shapes\__init__.py
"""
Shapes submodule handling shape generation and management.
"""

from .logic import (
    generate_random_shape,
    get_neighbors,
    is_shape_connected,
    refill_shape_slots,
)
from .templates import PREDEFINED_SHAPE_TEMPLATES

__all__ = [
    "generate_random_shape",
    "refill_shape_slots",
    "is_shape_connected",
    "get_neighbors",
    "PREDEFINED_SHAPE_TEMPLATES",
]


File: muzerotriangle\features\extractor.py
# File: muzerotriangle/features/extractor.py
import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from ..config import ModelConfig
from ..utils.types import StateType
from . import grid_features  # Keep this import

if TYPE_CHECKING:
    from ..environment import GameState


logger = logging.getLogger(__name__)


class GameStateFeatures:
    """Extracts features from GameState for NN input. Reads from GridData NumPy arrays."""

    def __init__(self, game_state: "GameState", model_config: ModelConfig):
        self.gs = game_state
        self.env_config = game_state.env_config
        self.model_config = model_config
        # Get direct references to NumPy arrays for efficiency
        self.occupied_np = game_state.grid_data._occupied_np
        self.death_np = game_state.grid_data._death_np
        # self.color_id_np = game_state.grid_data._color_id_np # Not used in current features

    def _get_grid_state(self) -> np.ndarray:
        """
        Returns grid state as a single channel numpy array based on NumPy arrays.
        Values: 1.0 (occupied playable), 0.0 (empty playable), -1.0 (death cell).
        Shape: (C, H, W) where C is GRID_INPUT_CHANNELS
        """
        rows, cols = self.env_config.ROWS, self.env_config.COLS
        # Initialize with 0.0 (empty playable)
        grid_state: np.ndarray = np.zeros(
            (self.model_config.GRID_INPUT_CHANNELS, rows, cols), dtype=np.float32
        )

        # Mark occupied playable cells as 1.0
        playable_occupied_mask = self.occupied_np & ~self.death_np
        grid_state[0, playable_occupied_mask] = 1.0

        # Mark death cells as -1.0
        grid_state[0, self.death_np] = -1.0

        # No need for the loop or isfinite check here if input arrays are guaranteed finite

        return grid_state

    def _get_shape_features(self) -> np.ndarray:
        """Extracts features for each shape slot. (No change needed here)"""
        num_slots = self.env_config.NUM_SHAPE_SLOTS

        FEATURES_PER_SHAPE_HERE = 7
        shape_feature_matrix = np.zeros(
            (num_slots, FEATURES_PER_SHAPE_HERE), dtype=np.float32
        )

        for i, shape in enumerate(self.gs.shapes):
            if shape and shape.triangles:
                n_tris = len(shape.triangles)
                ups = sum(1 for _, _, is_up in shape.triangles if is_up)
                downs = n_tris - ups
                min_r, min_c, max_r, max_c = shape.bbox()
                height = max_r - min_r + 1
                width_eff = (max_c - min_c + 1) * 0.75 + 0.25 if n_tris > 0 else 0

                # Populate features
                shape_feature_matrix[i, 0] = np.clip(n_tris / 5.0, 0, 1)
                shape_feature_matrix[i, 1] = ups / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 2] = downs / n_tris if n_tris > 0 else 0
                shape_feature_matrix[i, 3] = np.clip(
                    height / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 4] = np.clip(
                    width_eff / self.env_config.COLS, 0, 1
                )
                shape_feature_matrix[i, 5] = np.clip(
                    ((min_r + max_r) / 2.0) / self.env_config.ROWS, 0, 1
                )
                shape_feature_matrix[i, 6] = np.clip(
                    ((min_c + max_c) / 2.0) / self.env_config.COLS, 0, 1
                )
        # Flatten the matrix to get a 1D array
        return shape_feature_matrix.flatten()

    def _get_shape_availability(self) -> np.ndarray:
        """Returns a binary vector indicating which shape slots are filled. (No change needed)"""
        return np.array([1.0 if s else 0.0 for s in self.gs.shapes], dtype=np.float32)

    def _get_explicit_features(self) -> np.ndarray:
        """
        Extracts scalar features like score, heights, holes, etc.
        Uses GridData NumPy arrays directly.
        """
        EXPLICIT_FEATURES_DIM_HERE = 6
        features = np.zeros(EXPLICIT_FEATURES_DIM_HERE, dtype=np.float32)
        # Use the direct references stored in self
        occupied = self.occupied_np
        death = self.death_np
        rows, cols = self.env_config.ROWS, self.env_config.COLS

        # Pass NumPy arrays directly to grid_features functions
        heights = grid_features.get_column_heights(occupied, death, rows, cols)
        holes = grid_features.count_holes(occupied, death, heights, rows, cols)
        bump = grid_features.get_bumpiness(heights)
        total_playable_cells = np.sum(~death)

        # Populate features
        features[0] = np.clip(self.gs.game_score / 100.0, -5.0, 5.0)
        features[1] = np.mean(heights) / rows if rows > 0 else 0
        features[2] = np.max(heights) / rows if rows > 0 else 0
        features[3] = holes / total_playable_cells if total_playable_cells > 0 else 0
        features[4] = (bump / (cols - 1)) / rows if cols > 1 and rows > 0 else 0
        features[5] = np.clip(self.gs.pieces_placed_this_episode / 100.0, 0, 1)

        return cast(
            "np.ndarray", np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        )

    def get_combined_other_features(self) -> np.ndarray:
        """Combines all non-grid features into a single flat vector."""
        shape_feats = self._get_shape_features()
        avail_feats = self._get_shape_availability()
        explicit_feats = self._get_explicit_features()
        combined = np.concatenate([shape_feats, avail_feats, explicit_feats])

        expected_dim = self.model_config.OTHER_NN_INPUT_FEATURES_DIM
        if combined.shape[0] != expected_dim:
            # Log error instead of raising ValueError immediately during feature extraction
            logger.error(
                f"Combined other_features dimension mismatch! Extracted {combined.shape[0]}, but ModelConfig expects {expected_dim}. Padding/truncating."
            )
            # Pad or truncate to match expected dimension
            if combined.shape[0] < expected_dim:
                padding = np.zeros(
                    expected_dim - combined.shape[0], dtype=combined.dtype
                )
                combined = np.concatenate([combined, padding])
            else:
                combined = combined[:expected_dim]

        if not np.all(np.isfinite(combined)):
            logger.error(
                f"Non-finite values detected in combined other_features! Min: {np.nanmin(combined)}, Max: {np.nanmax(combined)}, Mean: {np.nanmean(combined)}"
            )
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        return cast("np.ndarray", combined.astype(np.float32))


def extract_state_features(
    game_state: "GameState", model_config: ModelConfig
) -> StateType:
    """
    Extracts and returns the state dictionary {grid, other_features} for NN input.
    Requires ModelConfig to ensure dimensions match the network's expectations.
    Includes validation for non-finite values. Now reads from GridData NumPy arrays.
    """
    extractor = GameStateFeatures(game_state, model_config)
    state_dict: StateType = {
        "grid": extractor._get_grid_state(),
        "other_features": extractor.get_combined_other_features(),
    }
    grid_feat = state_dict["grid"]
    other_feat = state_dict["other_features"]
    logger.debug(
        f"Extracted Features (State {game_state.current_step}): Grid(shape={grid_feat.shape}, min={grid_feat.min():.2f}, max={grid_feat.max():.2f}, mean={grid_feat.mean():.2f}), Other(shape={other_feat.shape}, min={other_feat.min():.2f}, max={other_feat.max():.2f}, mean={other_feat.mean():.2f})"
    )
    return state_dict


File: muzerotriangle\features\grid_features.py
import numpy as np
from numba import njit, prange


@njit(cache=True)
def get_column_heights(
    occupied: np.ndarray, death: np.ndarray, rows: int, cols: int
) -> np.ndarray:
    """Calculates the height of each column (highest occupied non-death cell)."""
    heights = np.zeros(cols, dtype=np.int32)
    for c in prange(cols):
        max_r = -1
        for r in range(rows):
            if occupied[r, c] and not death[r, c]:
                max_r = r
        heights[c] = max_r + 1
    return heights


@njit(cache=True)
def count_holes(
    occupied: np.ndarray, death: np.ndarray, heights: np.ndarray, _rows: int, cols: int
) -> int:
    """Counts the number of empty, non-death cells below the column height."""
    holes = 0
    for c in prange(cols):
        col_height = heights[c]
        for r in range(col_height):
            if not occupied[r, c] and not death[r, c]:
                holes += 1
    return holes


@njit(cache=True)
def get_bumpiness(heights: np.ndarray) -> float:
    """Calculates the total absolute difference between adjacent column heights."""
    bumpiness = 0.0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness


File: muzerotriangle\features\README.md
# File: muzerotriangle/features/README.md
# Feature Extraction Module (`muzerotriangle.features`)

## Purpose and Architecture

This module is solely responsible for converting raw [`GameState`](../environment/core/game_state.py) objects from the [`muzerotriangle.environment`](../environment/README.md) module into numerical representations (features) suitable for input into the neural network ([`muzerotriangle.nn`](../nn/README.md)). It acts as a bridge between the game's internal state and the requirements of the machine learning model.

-   **Decoupling:** This module completely decouples feature engineering from the core game environment logic. The `environment` module focuses only on game rules and state transitions, while this module handles the transformation for the NN.
-   **Feature Engineering:**
    -   [`extractor.py`](extractor.py): Contains the `GameStateFeatures` class and the main `extract_state_features` function. This orchestrates the extraction process, calling helper functions to generate different feature types. It uses `Triangle` and `Shape` from [`muzerotriangle.structs`](../structs/README.md).
    -   [`grid_features.py`](grid_features.py): Contains low-level, potentially performance-optimized (e.g., using Numba) functions for calculating specific scalar metrics derived from the grid state (like column heights, holes, bumpiness). **This module now operates directly on NumPy arrays passed from `GameStateFeatures`.**
-   **Output Format:** The `extract_state_features` function returns a `StateType` (a `TypedDict` defined in [`muzerotriangle.utils.types`](../utils/types.py) containing `grid` and `other_features` numpy arrays), which is the standard input format expected by the `NeuralNetwork` interface.
-   **Configuration Dependency:** The extractor requires [`ModelConfig`](../config/model_config.py) to ensure the dimensions of the extracted features match the expectations of the neural network architecture.

## Exposed Interfaces

-   **Functions:**
    -   `extract_state_features(game_state: GameState, model_config: ModelConfig) -> StateType`: The main function to perform feature extraction.
    -   Low-level grid feature functions from `grid_features` (e.g., `get_column_heights`, `count_holes`, `get_bumpiness`).
-   **Classes:**
    -   `GameStateFeatures`: Class containing the feature extraction logic (primarily used internally by `extract_state_features`).

## Dependencies

-   **[`muzerotriangle.environment`](../environment/README.md)**:
    -   `GameState`: The input object for feature extraction.
    -   `GridData`: Accessed via `GameState` to get grid information (NumPy arrays).
-   **[`muzerotriangle.config`](../config/README.md)**:
    -   `EnvConfig`: Accessed via `GameState` for environment dimensions.
    -   `ModelConfig`: Required by `extract_state_features` to ensure output dimensions match the NN input layer.
-   **[`muzerotriangle.structs`](../structs/README.md)**:
    -   Uses `Triangle`, `Shape`.
-   **[`muzerotriangle.utils`](../utils/README.md)**:
    -   `StateType`: The return type dictionary format.
-   **`numpy`**:
    -   Used extensively for creating and manipulating the numerical feature arrays.
-   **`numba`**:
    -   Used in `grid_features` for performance optimization.
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when changing the feature extraction logic, the set of extracted features, or the output format (`StateType`). Accurate documentation is crucial for maintainability.

File: muzerotriangle\features\__init__.py
"""
Feature extraction module.
Converts raw GameState objects into numerical representations suitable for NN input.
"""

from . import grid_features
from .extractor import GameStateFeatures, extract_state_features

__all__ = [
    "extract_state_features",
    "GameStateFeatures",
    "grid_features",
]


File: muzerotriangle\interaction\debug_mode_handler.py
# File: muzerotriangle/interaction/debug_mode_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from ..environment import grid as env_grid

# Import constants from the structs package directly
from ..structs import DEBUG_COLOR_ID, NO_COLOR_ID
from ..visualization import core as vis_core

if TYPE_CHECKING:
    # Keep Triangle for type hinting if GridLogic still uses it temporarily
    from .input_handler import InputHandler

logger = logging.getLogger(__name__)


def handle_debug_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in debug mode (toggle triangle state using NumPy arrays)."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect:
        logger.error("Grid layout rectangle not available for debug click.")
        return

    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )
    if not grid_coords:
        return

    r, c = grid_coords
    if game_state.grid_data.valid(r, c):
        # Check death zone first
        if not game_state.grid_data._death_np[r, c]:
            # Toggle occupancy state in NumPy array
            current_occupied_state = game_state.grid_data._occupied_np[r, c]
            new_occupied_state = not current_occupied_state
            game_state.grid_data._occupied_np[r, c] = new_occupied_state

            # Update color ID based on new state
            new_color_id = DEBUG_COLOR_ID if new_occupied_state else NO_COLOR_ID
            game_state.grid_data._color_id_np[r, c] = new_color_id

            logger.debug(
                f": Toggled triangle ({r},{c}) -> {'Occupied' if new_occupied_state else 'Empty'}"
            )

            # Check for line clears if the cell became occupied
            if new_occupied_state:
                # Pass the coordinate tuple in a set
                lines_cleared, unique_tris_coords, _ = (
                    env_grid.logic.check_and_clear_lines(
                        game_state.grid_data, newly_occupied_coords={(r, c)}
                    )
                )
                if lines_cleared > 0:
                    logger.debug(
                        f"Cleared {lines_cleared} lines ({len(unique_tris_coords)} coords) after toggle."
                    )
        else:
            logger.info(f"Clicked on death cell ({r},{c}). No action.")


def update_debug_hover(handler: "InputHandler") -> None:
    """Updates the debug highlight position within the InputHandler."""
    handler.debug_highlight_coord = None  # Reset hover state

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect or not grid_rect.collidepoint(mouse_pos):
        return  # Not hovering over grid

    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    if grid_coords:
        r, c = grid_coords
        # Highlight only valid, non-death cells
        if game_state.grid_data.valid(r, c) and not game_state.grid_data.is_death(r, c):
            handler.debug_highlight_coord = grid_coords


File: muzerotriangle\interaction\event_processor.py
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pygame

if TYPE_CHECKING:
    from ..visualization.core.visualizer import Visualizer

logger = logging.getLogger(__name__)


def process_pygame_events(
    visualizer: "Visualizer",
) -> Generator[pygame.event.Event, Any, bool]:
    """
    Processes basic Pygame events like QUIT, ESCAPE, VIDEORESIZE.
    Yields other events for mode-specific handlers.
    Returns False via StopIteration value if the application should quit, True otherwise.
    """
    should_quit = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            logger.info("Received QUIT event.")
            should_quit = True
            break
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            logger.info("Received ESCAPE key press.")
            should_quit = True
            break
        if event.type == pygame.VIDEORESIZE:
            try:
                w, h = max(320, event.w), max(240, event.h)
                visualizer.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                visualizer.layout_rects = None
                logger.info(f"Window resized to {w}x{h}")
            except pygame.error as e:
                logger.error(f"Error resizing window: {e}")
            yield event
        else:
            yield event
    return not should_quit


File: muzerotriangle\interaction\input_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from .. import environment, visualization
from . import debug_mode_handler, event_processor, play_mode_handler

if TYPE_CHECKING:
    from ..structs import Shape


logger = logging.getLogger(__name__)


class InputHandler:
    """
    Handles user input, manages interaction state (selection, hover),
    and delegates actions to mode-specific handlers.
    """

    def __init__(
        self,
        game_state: environment.GameState,
        visualizer: visualization.Visualizer,
        mode: str,
        env_config: environment.EnvConfig,
    ):
        self.game_state = game_state
        self.visualizer = visualizer
        self.mode = mode
        self.env_config = env_config

        # Interaction state managed here
        self.selected_shape_idx: int = -1
        self.hover_grid_coord: tuple[int, int] | None = None
        self.hover_is_valid: bool = False
        self.hover_shape: Shape | None = None
        self.debug_highlight_coord: tuple[int, int] | None = None
        self.mouse_pos: tuple[int, int] = (0, 0)

    def handle_input(self) -> bool:
        """Processes Pygame events and updates state based on mode. Returns False to quit."""
        self.mouse_pos = pygame.mouse.get_pos()

        # Reset hover/highlight state each frame before processing events/updates
        self.hover_grid_coord = None
        self.hover_is_valid = False
        self.hover_shape = None
        self.debug_highlight_coord = None

        running = True
        event_generator = event_processor.process_pygame_events(self.visualizer)
        try:
            while True:
                event = next(event_generator)
                # Pass self to handlers so they can modify interaction state
                if self.mode == "play":
                    play_mode_handler.handle_play_click(event, self)
                elif self.mode == "debug":
                    debug_mode_handler.handle_debug_click(event, self)
        except StopIteration as e:
            running = e.value  # False if quit requested

        # Update hover state after processing events
        if running:
            if self.mode == "play":
                play_mode_handler.update_play_hover(self)
            elif self.mode == "debug":
                debug_mode_handler.update_debug_hover(self)

        return running

    def get_render_interaction_state(self) -> dict:
        """Returns interaction state needed by Visualizer.render"""
        return {
            "selected_shape_idx": self.selected_shape_idx,
            "hover_shape": self.hover_shape,
            "hover_grid_coord": self.hover_grid_coord,
            "hover_is_valid": self.hover_is_valid,
            "hover_screen_pos": self.mouse_pos,  # Pass current mouse pos
            "debug_highlight_coord": self.debug_highlight_coord,
        }


File: muzerotriangle\interaction\play_mode_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from ..environment import core as env_core
from ..environment import grid as env_grid
from ..visualization import core as vis_core

if TYPE_CHECKING:
    from ..structs import Shape
    from .input_handler import InputHandler

logger = logging.getLogger(__name__)


def handle_play_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in play mode (select preview, place shape). Modifies handler state."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    if game_state.is_over():
        logger.info("Game is over, ignoring click.")
        return

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    # Get preview rects from visualizer cache
    preview_rects = visualizer.preview_rects

    # 1. Check for clicks on shape previews
    preview_idx = vis_core.coord_mapper.get_preview_index_from_screen(
        mouse_pos, preview_rects
    )
    if preview_idx is not None:
        if handler.selected_shape_idx == preview_idx:
            # Clicked selected shape again: deselect
            handler.selected_shape_idx = -1
            handler.hover_grid_coord = None  # Clear hover state on deselect
            handler.hover_shape = None
            logger.info("Deselected shape.")
        elif (
            0 <= preview_idx < len(game_state.shapes) and game_state.shapes[preview_idx]
        ):
            # Clicked a valid, available shape: select it
            handler.selected_shape_idx = preview_idx
            logger.info(f"Selected shape index: {preview_idx}")
            # Immediately update hover based on current mouse pos after selection
            update_play_hover(handler)  # Update hover state within handler
        else:
            # Clicked an empty or invalid slot
            logger.info(f"Clicked empty/invalid preview slot: {preview_idx}")
            # Deselect if clicking an empty slot while another is selected
            if handler.selected_shape_idx != -1:
                handler.selected_shape_idx = -1
                handler.hover_grid_coord = None
                handler.hover_shape = None
        return  # Handled preview click

    # 2. Check for clicks on the grid (if a shape is selected)
    selected_idx = handler.selected_shape_idx
    if selected_idx != -1 and grid_rect and grid_rect.collidepoint(mouse_pos):
        # A shape is selected, and the click is within the grid area.
        grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
            mouse_pos, grid_rect, game_state.env_config
        )
        # Use TYPE_CHECKING import for Shape type hint
        shape_to_place: Shape | None = game_state.shapes[selected_idx]

        # Check if the placement is valid *at the clicked location*
        if (
            grid_coords
            and shape_to_place
            and env_grid.logic.can_place(
                game_state.grid_data, shape_to_place, grid_coords[0], grid_coords[1]
            )
        ):
            # Valid placement click!
            r, c = grid_coords
            action = env_core.action_codec.encode_action(
                selected_idx, r, c, game_state.env_config
            )
            # Execute the step using the game state's method
            reward, done = game_state.step(action)  # Now returns (reward, done)
            logger.info(
                f"Placed shape {selected_idx} at {grid_coords}. R={reward:.1f}, Done={done}"
            )
            # Deselect shape after successful placement
            handler.selected_shape_idx = -1
            handler.hover_grid_coord = None  # Clear hover state
            handler.hover_shape = None
        else:
            # Clicked grid, shape selected, but not a valid placement spot for the click
            logger.info(f"Clicked grid at {grid_coords}, but placement invalid.")


def update_play_hover(handler: "InputHandler") -> None:
    """Updates the hover state within the InputHandler."""
    # Reset hover state first
    handler.hover_grid_coord = None
    handler.hover_is_valid = False
    handler.hover_shape = None

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    if game_state.is_over() or handler.selected_shape_idx == -1:
        return  # No hover if game over or no shape selected

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect or not grid_rect.collidepoint(mouse_pos):
        return  # Not hovering over grid

    shape_idx = handler.selected_shape_idx
    if not (0 <= shape_idx < len(game_state.shapes)):
        return
    shape: Shape | None = game_state.shapes[shape_idx]
    if not shape:
        return

    # Get grid coordinates under mouse
    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    if grid_coords:
        # Check if placement is valid at these coordinates
        is_valid = env_grid.logic.can_place(
            game_state.grid_data, shape, grid_coords[0], grid_coords[1]
        )
        # Update handler's hover state
        handler.hover_grid_coord = grid_coords
        handler.hover_is_valid = is_valid
        handler.hover_shape = shape  # Store the shape being hovered
    else:
        handler.hover_shape = shape  # Store shape for floating preview


File: muzerotriangle\interaction\README.md
# File: muzerotriangle/interaction/README.md
# Interaction Module (`muzerotriangle.interaction`)

## Purpose and Architecture

This module handles user input (keyboard and mouse) for interactive modes of the application, such as "play" and "debug". It bridges the gap between raw Pygame events and actions within the game simulation ([`GameState`](../environment/core/game_state.py)).

-   **Event Processing:** [`event_processor.py`](event_processor.py) handles common Pygame events like quitting (QUIT, ESC) and window resizing. It acts as a generator, yielding other events for mode-specific processing.
-   **Input Handler:** The [`InputHandler`](input_handler.py) class is the main entry point.
    -   It receives Pygame events (via the `event_processor`).
    -   It **manages interaction-specific state** internally (e.g., `selected_shape_idx`, `hover_grid_coord`, `debug_highlight_coord`).
    -   It determines the current interaction mode ("play" or "debug") and delegates event handling and hover updates to specific handler functions ([`play_mode_handler`](play_mode_handler.py), [`debug_mode_handler`](debug_mode_handler.py)).
    -   It provides the necessary interaction state to the [`Visualizer`](../visualization/core/visualizer.py) for rendering feedback (hover previews, selection highlights).
-   **Mode-Specific Handlers:** `play_mode_handler.py` and `debug_mode_handler.py` contain the logic specific to each mode, operating on the `InputHandler`'s state and the `GameState`.
    -   `play`: Handles selecting shapes, checking placement validity, and triggering `GameState.step` on valid clicks. Updates hover state in the `InputHandler`.
    -   `debug`: Handles toggling the state of individual triangles directly on the `GameState.grid_data`. Updates hover state in the `InputHandler`.
-   **Decoupling:** It separates input handling logic from the core game simulation ([`environment`](../environment/README.md)) and rendering ([`visualization`](../visualization/README.md)), although it needs references to both to function. The `Visualizer` is now only responsible for drawing based on the state provided by the `GameState` and the `InputHandler`.

## Exposed Interfaces

-   **Classes:**
    -   `InputHandler`:
        -   `__init__(game_state: GameState, visualizer: Visualizer, mode: str, env_config: EnvConfig)`
        -   `handle_input() -> bool`: Processes events for one frame, returns `False` if quitting.
        -   `get_render_interaction_state() -> dict`: Returns interaction state needed by `Visualizer.render`.
-   **Functions:**
    -   `process_pygame_events(visualizer: Visualizer) -> Generator[pygame.event.Event, Any, bool]`: Processes common events, yields others.
    -   `handle_play_click(event: pygame.event.Event, handler: InputHandler)`: Handles clicks in play mode.
    -   `update_play_hover(handler: InputHandler)`: Updates hover state in play mode.
    -   `handle_debug_click(event: pygame.event.Event, handler: InputHandler)`: Handles clicks in debug mode.
    -   `update_debug_hover(handler: InputHandler)`: Updates hover state in debug mode.

## Dependencies

-   **[`muzerotriangle.environment`](../environment/README.md)**:
    -   `GameState`: Modifies the game state based on user actions (placing shapes, toggling debug cells).
    -   `EnvConfig`: Used for coordinate mapping and action encoding.
    -   `GridLogic`, `ActionCodec`: Used by mode-specific handlers.
-   **[`muzerotriangle.visualization`](../visualization/README.md)**:
    -   `Visualizer`: Used to get layout information (`grid_rect`, `preview_rects`) and for coordinate mapping (`get_grid_coords_from_screen`, `get_preview_index_from_screen`). Also updated directly during resize events.
    -   `VisConfig`: Accessed via `Visualizer`.
-   **[`muzerotriangle.structs`](../structs/README.md)**:
    -   `Shape`, `Triangle`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **`pygame`**:
    -   Relies heavily on Pygame for event handling (`pygame.event`, `pygame.mouse`) and constants (`MOUSEBUTTONDOWN`, `KEYDOWN`, etc.).
-   **Standard Libraries:** `typing`, `logging`.

---

**Note:** Please keep this README updated when adding new interaction modes, changing input handling logic, or modifying the interfaces between interaction, environment, and visualization. Accurate documentation is crucial for maintainability.

File: muzerotriangle\interaction\__init__.py
from .debug_mode_handler import handle_debug_click, update_debug_hover
from .event_processor import process_pygame_events
from .input_handler import InputHandler
from .play_mode_handler import handle_play_click, update_play_hover

__all__ = [
    "InputHandler",
    "process_pygame_events",
    "handle_play_click",
    "update_play_hover",
    "handle_debug_click",
    "update_debug_hover",
]


File: muzerotriangle\mcts\README.md
# File: muzerotriangle/mcts/README.md
# Monte Carlo Tree Search Module (`muzerotriangle.mcts`)

## Purpose and Architecture

This module implements the Monte Carlo Tree Search algorithm, adapted for the MuZero framework. MCTS is used during self-play to explore the game tree, generate improved policies, and estimate state values, providing training targets for the neural network.

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Node`: Represents a state in the search tree. In MuZero, nodes store the *hidden state* (`s_k`) predicted by the dynamics function, the predicted *reward* (`r_k`) to reach that state, and MCTS statistics (visit counts, value sum, prior probability). The root node holds the initial `GameState` and its corresponding initial hidden state (`s_0`) after the first inference.
    -   `search`: Contains the main `run_mcts_simulations` function orchestrating the selection, expansion, and backpropagation phases. It uses the `NeuralNetwork` interface for initial inference (`h+f`) and recurrent inference (`g+f`). **It handles potential gradient issues by detaching tensors before converting to NumPy.**
    -   `config`: Defines the `MCTSConfig` class holding hyperparameters like the number of simulations, PUCT coefficient, temperature settings, Dirichlet noise parameters, and the discount factor (`gamma`).
    -   `types`: Defines necessary type hints and protocols, notably `ActionPolicyValueEvaluator` (though the `NeuralNetwork` interface is now used directly) and `ActionPolicyMapping`.
-   **Strategy Components ([`strategy/README.md`](strategy/README.md)):**
    -   `selection`: Implements the tree traversal logic (PUCT calculation, Dirichlet noise addition, leaf selection).
    -   `expansion`: Handles expanding leaf nodes using policy predictions from the network's prediction function (`f`).
    -   `backpropagation`: Implements the process of updating node statistics back up the tree, incorporating predicted rewards and the discount factor.
    -   `policy`: Provides functions to select the final action based on visit counts (`select_action_based_on_visits`) and to generate the policy target vector for training (`get_policy_target`).

## Exposed Interfaces

-   **Core:**
    -   `Node`: The tree node class (MuZero version).
    -   `MCTSConfig`: Configuration class (defined in [`muzerotriangle.config`](../config/README.md)).
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network: NeuralNetwork, valid_actions_from_state: List[ActionType]) -> int`: The main function to run MCTS.
    -   `ActionPolicyMapping`: Type alias for the policy dictionary.
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Strategy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.

## Dependencies

-   **[`muzerotriangle.environment`](../environment/README.md)**:
    -   `GameState`: Represents the state within the root `Node`. MCTS interacts with `GameState` methods like `is_over()`, `valid_actions()`, `copy()`.
    -   `EnvConfig`: Accessed via `GameState`.
-   **[`muzerotriangle.nn`](../nn/README.md)**:
    -   `NeuralNetwork`: The interface used by `run_mcts_simulations` and `expansion` to perform initial inference (`h+f`) and recurrent inference (`g+f`).
-   **[`muzerotriangle.config`](../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters.
-   **[`muzerotriangle.features`](../features/README.md)**:
    -   `extract_state_features`: Used by `run_mcts_simulations` for the initial root inference.
-   **[`muzerotriangle.utils`](../utils/README.md)**:
    -   `ActionType`, `StateType`, `PolicyTargetMapping`: Used for actions, state representation, and policy targets.
-   **`numpy`**:
    -   Used for Dirichlet noise generation and policy calculations. **Requires careful handling (e.g., `.detach()`) when converting from `torch.Tensor`.**
-   **`torch`**:
    -   Used for hidden states within `Node`.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `time`.

---

**Note:** Please keep this README updated when changing the MCTS algorithm phases (selection, expansion, backpropagation), the node structure, configuration options, or the interaction with the environment or neural network. Accurate documentation is crucial for maintainability.

File: muzerotriangle\mcts\__init__.py
"""
Monte Carlo Tree Search (MCTS) module.
Provides the core algorithm and components for game tree search.
"""

from muzerotriangle.config import MCTSConfig

from .core.node import Node
from .core.search import (
    MCTSExecutionError,
    run_mcts_simulations,
)
from .core.types import ActionPolicyMapping, ActionPolicyValueEvaluator
from .strategy.policy import get_policy_target, select_action_based_on_visits

__all__ = [
    # Core
    "Node",
    "run_mcts_simulations",
    "MCTSConfig",
    "ActionPolicyValueEvaluator",
    "ActionPolicyMapping",
    "MCTSExecutionError",
    # Strategy
    "select_action_based_on_visits",
    "get_policy_target",
]


File: muzerotriangle\mcts\core\node.py
# File: muzerotriangle/mcts/core/node.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from muzerotriangle.environment import GameState
    from muzerotriangle.utils.types import ActionType

logger = logging.getLogger(__name__)


class Node:
    """
    Represents a node in the Monte Carlo Search Tree for MuZero.
    Stores hidden state, predicted reward, and MCTS statistics.
    The root node holds the actual GameState.
    """

    def __init__(
        self,
        prior: float = 0.0,
        # --- MuZero Specific ---
        hidden_state: torch.Tensor | None = None,  # Stores s_k
        reward: float = 0.0,  # Stores r_k (predicted reward to reach this state)
        initial_game_state: GameState | None = None,  # Only for root node
        # --- Common MCTS ---
        parent: Node | None = None,
        action_taken: ActionType | None = None,  # Action a_k that led to this state s_k
    ):
        self.parent = parent
        self.action_taken = action_taken
        self.prior_probability = prior

        # State Representation
        self.hidden_state = hidden_state  # Tensor representing the state s_k
        self.initial_game_state = initial_game_state  # For root node observation
        self.reward = reward  # Predicted reward r_k from g() to reach this state

        # MCTS Statistics
        self.visit_count: int = 0
        self.value_sum: float = (
            0.0  # Sum of backed-up values (G_i) from simulations passing through here
        )
        self.children: dict[ActionType, Node] = {}

        # --- Cached values from prediction function f(s_k) ---
        # These are calculated when the node is expanded or selected
        self.predicted_value: float | None = None  # Cached v_k

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_expanded(self) -> bool:
        """Checks if the node has children."""
        return bool(self.children)

    @property
    def value_estimate(self) -> float:
        """
        Calculates the Q-value estimate Q(s,a) for the *action* leading to this node.
        Average of values G backpropagated through this node.
        Returns 0 if unvisited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_state(self) -> torch.Tensor | GameState:
        """Returns the representation of the state (hidden_state or GameState for root)."""
        if self.is_root and self.initial_game_state is not None:
            return self.initial_game_state
        elif self.hidden_state is not None:
            return self.hidden_state
        else:
            # This should ideally not happen for a non-root node after expansion
            raise ValueError(
                "Node state is missing (neither initial_game_state nor hidden_state is set)."
            )

    def __repr__(self) -> str:
        state_desc = (
            f"Root(Step={self.initial_game_state.current_step})"
            if self.is_root and self.initial_game_state
            else f"HiddenState(shape={self.hidden_state.shape if self.hidden_state is not None else 'None'})"
        )
        return (
            f"Node(Action={self.action_taken}, State={state_desc}, "
            f"Reward={self.reward:.2f}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )


File: muzerotriangle\mcts\core\README.md
# File: muzerotriangle/mcts/core/README.md
# MCTS Core Submodule (`muzerotriangle.mcts.core`)

## Purpose and Architecture

This submodule defines the fundamental building blocks and interfaces for the MuZero Monte Carlo Tree Search implementation.

-   **[`Node`](node.py):** The `Node` class is the cornerstone, representing a single state within the search tree. It stores the associated *hidden state* (`torch.Tensor`), the predicted *reward* to reach this state, parent/child relationships, the action that led to it, and crucial MCTS statistics (visit count, value sum, prior probability). The root node additionally holds the initial `GameState`. It provides properties like `value_estimate` (Q-value) and `is_expanded`.
-   **[`search`](search.py):** The `search.py` module contains the high-level `run_mcts_simulations` function. This function orchestrates the core MCTS loop for a given root node: performing initial inference, then repeatedly selecting leaves, expanding them using the network's dynamics and prediction functions, and backpropagating the results. It uses helper functions from the [`muzerotriangle.mcts.strategy`](../strategy/README.md) submodule. It handles potential gradient issues by detaching tensors before converting to NumPy.
-   **[`types`](types.py):** The `types.py` module defines essential type hints and protocols for the MCTS module, such as `ActionPolicyMapping`. The `ActionPolicyValueEvaluator` protocol is less relevant now as the `NeuralNetwork` interface is used directly.

## Exposed Interfaces

-   **Classes:**
    -   `Node`: The tree node class (MuZero version).
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Functions:**
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network: NeuralNetwork, valid_actions_from_state: List[ActionType]) -> int`: Orchestrates the MCTS process.
-   **Types:**
    -   `ActionPolicyMapping`: Type alias for the policy dictionary (mapping action index to probability).
    -   `ActionPolicyValueEvaluator`: Protocol defining the evaluation interface (though `NeuralNetwork` is used directly).

## Dependencies

-   **[`muzerotriangle.environment`](../../environment/README.md)**:
    -   `GameState`: Represents the state within the root `Node`. MCTS interacts with `GameState` methods like `is_over`, `valid_actions`, `copy`.
-   **[`muzerotriangle.mcts.strategy`](../strategy/README.md)**:
    -   `selection`, `expansion`, `backpropagation`: The `run_mcts_simulations` function delegates the core algorithm phases to functions within this submodule.
-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters.
-   **[`muzerotriangle.nn`](../../nn/README.md)**:
    -   `NeuralNetwork`: Used by `run_mcts_simulations` and `expansion`.
-   **[`muzerotriangle.features`](../../features/README.md)**:
    -   `extract_state_features`: Used by `run_mcts_simulations`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**:
    -   `ActionType`, `StateType`, `PolicyTargetMapping`.
-   **`torch`**:
    -   Used for hidden states.
-   **`numpy`**:
    -   Used for Dirichlet noise and policy calculations. Requires careful handling (e.g., `.detach()`) when converting from `torch.Tensor`.
-   **Standard Libraries:** `typing`, `math`, `logging`.

---

**Note:** Please keep this README updated when modifying the `Node` structure, the `run_mcts_simulations` logic, or the interfaces used. Accurate documentation is crucial for maintainability.

File: muzerotriangle\mcts\core\search.py
# File: muzerotriangle/mcts/core/search.py
import logging
from typing import TYPE_CHECKING

from ...config import MCTSConfig
from ...features import extract_state_features  # Import feature extractor
from ...utils.types import ActionType, StateType  # Import StateType
from ..strategy import backpropagation, expansion, selection
from .node import Node

if TYPE_CHECKING:
    from ...nn import NeuralNetwork

logger = logging.getLogger(__name__)


class MCTSExecutionError(Exception):
    pass


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network: "NeuralNetwork",
    valid_actions_from_state: list[ActionType],
) -> int:
    """Runs MuZero MCTS simulations."""
    if root_node.initial_game_state is None:
        raise MCTSExecutionError("Root node needs initial_game_state.")
    if root_node.initial_game_state.is_over():
        return 0

    max_depth_overall = 0
    sim_success_count = 0
    sim_error_count = 0

    # Initial Root Inference and Expansion
    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node initial inference...")
        try:
            # Extract features first, pass StateType dict
            state_dict: StateType = extract_state_features(
                root_node.initial_game_state, network.model_config
            )
            # Ensure tensors are on the correct device
            # initial_inference expects StateType dict with numpy arrays
            # The conversion to tensor happens inside initial_inference
            (
                policy_logits,
                value_logits,
                _,
                initial_hidden_state,
            ) = network.initial_inference(
                state_dict  # type: ignore[arg-type] # MyPy seems confused here
            )

            root_node.hidden_state = initial_hidden_state.squeeze(
                0
            ).detach()  # Detach here
            policy_probs = (
                network._logits_to_probs(policy_logits)
                .squeeze(0)
                .detach()  # Detach here
                .cpu()
                .numpy()
            )
            root_node.predicted_value = network._logits_to_scalar(
                value_logits, network.support
            ).item()

            # Filter policy map by valid actions for root
            full_policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
            filtered_policy_map = {
                action: full_policy_map.get(action, 0.0)
                for action in valid_actions_from_state
            }

            # Pass valid actions AND the filtered policy map for the initial root expansion
            expansion.expand_node(
                root_node,
                filtered_policy_map,  # Pass filtered map
                network,
                valid_actions_from_state,  # Pass valid actions explicitly
            )

            if root_node.predicted_value is not None:
                depth_bp = backpropagation.backpropagate_value(
                    root_node, root_node.predicted_value, config.discount
                )
                max_depth_overall = max(max_depth_overall, depth_bp)
            # Apply noise only if children were actually created
            if root_node.children:
                selection.add_dirichlet_noise(root_node, config)
        except Exception as e:
            raise MCTSExecutionError(f"Initial root inference failed: {e}") from e
    elif (
        root_node.visit_count == 0 and root_node.children
    ):  # Apply noise only if children exist
        # Apply noise only if root hasn't been visited yet in this MCTS run
        selection.add_dirichlet_noise(root_node, config)

    # Simulation Loop
    for sim in range(config.num_simulations):
        current_node = root_node
        search_path = [current_node]
        depth = 0
        try:
            # Selection
            while current_node.is_expanded:
                if (
                    config.max_search_depth is not None
                    and depth >= config.max_search_depth
                ):
                    break
                current_node = selection.select_child_node(current_node, config)
                search_path.append(current_node)
                depth += 1
            leaf_node = current_node
            max_depth_overall = max(max_depth_overall, depth)

            # --- ADDED CHECK: Don't expand if leaf represents a terminal state ---
            # We can only reliably check this for the root node in MuZero MCTS
            is_terminal_leaf = False
            if (
                leaf_node.is_root
                and leaf_node.initial_game_state is not None
                and not leaf_node.initial_game_state.valid_actions()
            ):
                is_terminal_leaf = True
                logger.debug(
                    "Leaf node (root) has no valid actions. Treating as terminal."
                )
            # --- END ADDED CHECK ---

            # Expansion & Prediction
            value_for_backprop = 0.0
            # --- MODIFIED: Only expand/predict if not terminal ---
            if not is_terminal_leaf and leaf_node.hidden_state is not None:
                # --- END MODIFIED ---
                # Ensure hidden_state is on the correct device and has batch dim
                hidden_state_batch = leaf_node.hidden_state.to(
                    network.device
                ).unsqueeze(0)
                policy_logits, value_logits = network.model.predict(hidden_state_batch)

                leaf_node.predicted_value = network._logits_to_scalar(
                    value_logits, network.support
                ).item()
                value_for_backprop = leaf_node.predicted_value
                policy_probs = (
                    network._logits_to_probs(policy_logits)
                    .squeeze(0)
                    .detach()  # Detach here
                    .cpu()
                    .numpy()
                )
                policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
                # Pass valid_actions_from_state here as well, as the leaf node might be the root
                # or we might need to re-evaluate valid actions if the game state was available
                # For MuZero, we rely on the policy mask from the network if available,
                # otherwise, we might need the game state if the node represents a real state.
                # Since we only expand based on the policy prediction, we don't strictly need
                # valid_actions here unless we want to mask the policy further.
                # Let's assume the policy prediction already accounts for valid actions.
                expansion.expand_node(
                    leaf_node,
                    policy_map,
                    network,
                    valid_actions=None,  # Pass None here as we use the network's policy
                )
            elif (
                leaf_node.is_root and leaf_node.predicted_value is not None
            ):  # Handle root being leaf
                value_for_backprop = leaf_node.predicted_value
            # --- ADDED: Handle terminal leaf value ---
            elif is_terminal_leaf:
                # If it's terminal, the value is known (e.g., 0 or from game outcome if available)
                # For simplicity, let's use 0, assuming no outcome is readily available here.
                value_for_backprop = 0.0
                logger.debug("Using 0.0 as value for terminal leaf node.")
            # --- END ADDED ---
            else:
                # This case should ideally not happen if the game hasn't ended before expansion
                # If it's a terminal state reached during simulation, the value should be the actual outcome
                # For MuZero's pure MCTS (without explicit game state simulation),
                # we rely on the network's value prediction. If hidden_state is None here, it's an error.
                logger.error(
                    f"Leaf node state invalid: is_root={leaf_node.is_root}, hidden_state is None, is_terminal={is_terminal_leaf}"
                )
                value_for_backprop = 0.0  # Fallback, but indicates an issue

            # Backpropagation
            _ = backpropagation.backpropagate_value(
                leaf_node, value_for_backprop, config.discount
            )
            sim_success_count += 1
        except Exception as e:
            sim_error_count += 1
            logger.error(f"Error in simulation {sim + 1}: {e}", exc_info=True)
            # Optionally break or continue based on error tolerance
            # break

    if sim_error_count > config.num_simulations * 0.1:  # Allow up to 10% errors
        logger.warning(
            f"MCTS completed with {sim_error_count} errors out of {config.num_simulations} simulations."
        )
        # Decide if this should be a fatal error
        # raise MCTSExecutionError(f"MCTS failed: High error rate ({sim_error_count} errors).")

    return max_depth_overall


File: muzerotriangle\mcts\core\types.py
# File: muzerotriangle/mcts/core/types.py
# No changes needed for this refactoring step. Keep the existing content.
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from ...utils.types import PolicyValueOutput

if TYPE_CHECKING:
    from ...environment import GameState
    from ...utils.types import ActionType

ActionPolicyMapping = Mapping["ActionType", float]


class ActionPolicyValueEvaluator(Protocol):
    """Defines the interface for evaluating a game state using a neural network."""

    def evaluate(self, state: "GameState") -> PolicyValueOutput:
        """
        Evaluates a single game state using the neural network.

        Args:
            state: The GameState object to evaluate.

        Returns:
            A tuple containing:
                - ActionPolicyMapping: A mapping from valid action indices
                    to their prior probabilities (output by the policy head).
                - float: The estimated value of the state (output by the value head).
        """
        ...

    def evaluate_batch(self, states: list["GameState"]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of game states using the neural network.
        (Optional but recommended for performance if MCTS supports batch evaluation).

        Args:
            states: A list of GameState objects to evaluate.

        Returns:
            A list of tuples, where each tuple corresponds to an input state and contains:
                - ActionPolicyMapping: Action probabilities for that state.
                - float: The estimated value of that state.
        """
        ...


File: muzerotriangle\mcts\core\__init__.py


File: muzerotriangle\mcts\strategy\backpropagation.py
# File: muzerotriangle/mcts/strategy/backpropagation.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(
    leaf_node: "Node",
    value_from_leaf: float,  # Value estimate v_L from f(s_L) or terminal reward
    discount: float,  # Gamma discount factor
) -> int:
    """
    Propagates the simulation value back up the tree from the leaf node.
    In MuZero, the value incorporates predicted rewards along the path.
    G = r_{k+1} + gamma * r_{k+2} + ... + gamma^N * v_L
    Returns the depth of the backpropagation path.
    """
    current_node: Node | None = leaf_node
    path_str = []
    depth = 0
    value_to_propagate: float = value_from_leaf  # Start with v_L or terminal reward

    logger.debug(
        f"[Backprop] Starting backprop from leaf node {leaf_node.action_taken} "
        f"with initial value_from_leaf={value_from_leaf:.4f} and discount={discount}"
    )

    while current_node is not None:
        # --- MuZero Modification ---
        # The value estimate Q(s,a) stored at the parent relates to the value G
        # derived *from* the current_node.
        # The value_to_propagate represents the value starting from *this* node's state.
        # We add this value to the node's statistics.
        # When moving to the parent, we discount the current value_to_propagate
        # and add the predicted reward 'r' that led *to* the current_node.
        # ---

        q_before = current_node.value_estimate
        total_val_before = current_node.value_sum
        visits_before = current_node.visit_count

        current_node.visit_count += 1
        current_node.value_sum += value_to_propagate  # Add the calculated G

        q_after = current_node.value_estimate
        total_val_after = current_node.value_sum
        visits_after = current_node.visit_count

        action_str = (
            f"Act={current_node.action_taken}"
            if current_node.action_taken is not None
            else "Root"
        )
        path_str.append(
            f"N({action_str},R={current_node.reward:.2f},V={visits_after},Q={q_after:.3f})"
        )

        logger.debug(
            f"  [Backprop] Depth {depth}: Node({action_str}), "
            f"Visits: {visits_before} -> {visits_after}, "
            f"PropagatedG={value_to_propagate:.4f}, "
            f"ValueSum: {total_val_before:.3f} -> {total_val_after:.3f}, "
            f"Q: {q_before:.3f} -> {q_after:.3f}"
        )

        # --- MuZero: Calculate value for the parent ---
        # G_{parent} = r_{current} + gamma * G_{current}
        # where r_{current} is the reward predicted by g() for reaching current_node
        if current_node.parent is not None:  # Don't update G beyond the root
            value_to_propagate = current_node.reward + discount * value_to_propagate
            logger.debug(
                f"    [Backprop] PrevG={value_to_propagate / discount - current_node.reward:.4f} -> "
                f"NextG (for parent) = r_k({current_node.reward:.3f}) + gamma({discount}) * PrevG = {value_to_propagate:.4f}"
            )
        # ---

        current_node = current_node.parent
        depth += 1

    logger.debug(f"[Backprop] Finished. Path: {' <- '.join(reversed(path_str))}")
    return depth


File: muzerotriangle\mcts\strategy\expansion.py
# File: muzerotriangle/mcts/strategy/expansion.py
import logging
from typing import TYPE_CHECKING

import torch  # Import torch for tensor check

from ..core.node import Node
from ..core.types import ActionPolicyMapping

if TYPE_CHECKING:
    # Use NeuralNetwork interface type hint
    from muzerotriangle.nn import NeuralNetwork
    from muzerotriangle.utils.types import ActionType


logger = logging.getLogger(__name__)


def expand_node(
    node_to_expand: "Node",  # Node containing s_k
    policy_prediction: ActionPolicyMapping,  # Policy p_k from f(s_k)
    network: "NeuralNetwork",  # Network interface to call dynamics (g)
    valid_actions: list["ActionType"] | None = None,  # Pass valid actions explicitly
):
    """
    Expands a leaf node in the MuZero search tree.
    If valid_actions is provided, only those actions are considered for expansion.
    Otherwise, actions with non-zero prior probability in policy_prediction are considered.
    """
    if node_to_expand.is_expanded:
        logger.debug(f"Node {node_to_expand.action_taken} already expanded. Skipping.")
        return
    hidden_state_k = node_to_expand.hidden_state
    if hidden_state_k is None:
        logger.error(
            f"[Expand] Node {node_to_expand.action_taken} has no hidden state."
        )
        return

    logger.debug(f"[Expand] Expanding Node via action: {node_to_expand.action_taken}")

    # --- MODIFIED: Determine actions to expand based on valid_actions first ---
    if valid_actions is not None:
        # If valid_actions are provided (e.g., for root), only expand these
        actions_to_expand_set = set(valid_actions)
        if not actions_to_expand_set:
            logger.warning(
                f"[Expand] No valid actions provided for node {node_to_expand.action_taken}. Node will remain unexpanded."
            )
            return  # Do not create children if valid_actions is empty
    else:
        # If valid_actions not provided, use policy prediction keys (for internal nodes)
        # Filter by non-zero prior? Optional, but can reduce unnecessary dynamics calls.
        # Let's keep expanding all actions predicted by the policy for now.
        actions_to_expand_set = set(policy_prediction.keys())
        if not actions_to_expand_set:
            logger.warning(
                f"[Expand] No actions found in policy prediction for node {node_to_expand.action_taken}. Node will remain unexpanded."
            )
            return
    # --- END MODIFIED ---

    children_created = 0
    for action in actions_to_expand_set:
        # Get prior probability for this action from the prediction
        prior = policy_prediction.get(action, 0.0)
        if prior < 0:
            logger.warning(f"Negative prior {prior} for action {action}. Clamping.")
            prior = 0.0
        # Optional: Skip expansion if prior is zero?
        # if prior <= 1e-6:
        #     continue

        try:
            if not isinstance(hidden_state_k, torch.Tensor):
                logger.error(f"Hidden state is not a tensor: {type(hidden_state_k)}")
                continue
            # Ensure hidden state is batched for dynamics call
            hidden_state_k_batch = (
                hidden_state_k.unsqueeze(0)
                if hidden_state_k.dim() == 1
                else hidden_state_k
            )

            # Call dynamics on the underlying model
            # Ensure action is a tensor for the model
            action_tensor = torch.tensor(
                [action], dtype=torch.long, device=network.device
            )
            hidden_state_k_plus_1, reward_logits = network.model.dynamics(
                hidden_state_k_batch, action_tensor
            )

            # Remove batch dimension and calculate scalar reward
            hidden_state_k_plus_1 = hidden_state_k_plus_1.squeeze(
                0
            ).detach()  # Detach here
            reward_logits = reward_logits.squeeze(0)
            reward_k_plus_1 = network._logits_to_scalar(
                reward_logits.unsqueeze(0), network.reward_support
            ).item()

        except Exception as e:
            logger.error(
                f"[Expand] Error calling dynamics for action {action}: {e}",
                exc_info=True,
            )
            continue

        # Create the child node
        child = Node(
            prior=prior,
            hidden_state=hidden_state_k_plus_1,
            reward=reward_k_plus_1,
            parent=node_to_expand,
            action_taken=action,
        )
        node_to_expand.children[action] = child
        logger.debug(
            f"  [Expand] Created child for action {action}: Prior={prior:.4f}, Reward={reward_k_plus_1:.3f}"
        )
        children_created += 1

    logger.debug(f"[Expand] Node expanded with {children_created} children.")


File: muzerotriangle\mcts\strategy\policy.py
# File: muzerotriangle/mcts/strategy/policy.py
# No changes needed for this refactoring step. Keep the existing content.
import logging
import random

import numpy as np

from ...utils.types import ActionType
from ..core.node import Node
from ..core.types import ActionPolicyMapping

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class PolicyGenerationError(Exception):
    """Custom exception for errors during policy generation or action selection."""

    pass


def select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType:
    """
    Selects an action from the root node based on visit counts and temperature.
    Raises PolicyGenerationError if selection is not possible.
    """
    if not root_node.children:
        # Ensure root_node.state access is safe (check if it's initial_game_state)
        state_info = (
            f"Step {root_node.initial_game_state.current_step}"
            if root_node.is_root and root_node.initial_game_state
            else f"Action {root_node.action_taken}"
        )
        raise PolicyGenerationError(
            f"Cannot select action: Root node ({state_info}) has no children."
        )

    actions = list(root_node.children.keys())
    visit_counts = np.array(
        [root_node.children[action].visit_count for action in actions],
        dtype=np.float64,
    )

    if len(actions) == 0:
        state_info = (
            f"Step {root_node.initial_game_state.current_step}"
            if root_node.is_root and root_node.initial_game_state
            else f"Action {root_node.action_taken}"
        )
        raise PolicyGenerationError(
            f"Cannot select action: No actions available in children for root node ({state_info})."
        )

    total_visits = np.sum(visit_counts)
    state_info = (
        f"Step {root_node.initial_game_state.current_step}"
        if root_node.is_root and root_node.initial_game_state
        else f"Action {root_node.action_taken}"
    )
    logger.debug(
        f"[PolicySelect] Selecting action for node {state_info}. Total child visits: {total_visits}. Num children: {len(actions)}"
    )

    if total_visits == 0:
        logger.warning(
            f"[PolicySelect] Total visit count for children is zero at root node ({state_info}). MCTS might have failed. Selecting uniformly."
        )
        selected_action = random.choice(actions)
        logger.debug(
            f"[PolicySelect] Uniform random action selected: {selected_action}"
        )
        return selected_action

    if temperature == 0.0:
        max_visits = np.max(visit_counts)
        logger.debug(
            f"[PolicySelect] Greedy selection (temp=0). Max visits: {max_visits}"
        )
        best_action_indices = np.where(visit_counts == max_visits)[0]
        logger.debug(
            f"[PolicySelect] Greedy selection. Best action indices: {best_action_indices}"
        )
        chosen_index = random.choice(best_action_indices)
        selected_action = actions[chosen_index]
        logger.debug(f"[PolicySelect] Greedy action selected: {selected_action}")
        return selected_action

    else:
        logger.debug(f"[PolicySelect] Probabilistic selection: Temp={temperature:.4f}")
        logger.debug(f"  Visit Counts: {visit_counts}")
        # Add small epsilon to prevent log(0)
        log_visits = np.log(np.maximum(visit_counts, 1e-9))
        scaled_log_visits = log_visits / temperature
        # Subtract max for numerical stability before exponentiation
        scaled_log_visits -= np.max(scaled_log_visits)
        probabilities = np.exp(scaled_log_visits)
        sum_probs = np.sum(probabilities)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            raise PolicyGenerationError(
                f"Could not normalize visit probabilities (sum={sum_probs}). Visits: {visit_counts}"
            )
        else:
            probabilities /= sum_probs

        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PolicyGenerationError(
                f"Invalid probabilities generated after normalization: {probabilities}"
            )
        if abs(np.sum(probabilities) - 1.0) > 1e-5:
            logger.warning(
                f"[PolicySelect] Probabilities sum to {np.sum(probabilities):.6f} after normalization. Attempting re-normalization."
            )
            probabilities /= np.sum(probabilities)
            if abs(np.sum(probabilities) - 1.0) > 1e-5:
                raise PolicyGenerationError(
                    f"Probabilities still do not sum to 1 after re-normalization: {probabilities}, Sum: {np.sum(probabilities)}"
                )

        logger.debug(f"  Final Probabilities (normalized): {probabilities}")
        logger.debug(f"  Final Probabilities Sum: {np.sum(probabilities):.6f}")

        try:
            selected_action = rng.choice(actions, p=probabilities)
            logger.debug(
                f"[PolicySelect] Sampled action (temp={temperature:.2f}): {selected_action}"
            )
            return int(selected_action)  # Ensure return type is ActionType (int)
        except ValueError as e:
            raise PolicyGenerationError(
                f"Error during np.random.choice: {e}. Probs: {probabilities}, Sum: {np.sum(probabilities)}"
            ) from e


def get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping:
    """
    Calculates the policy target distribution based on MCTS visit counts.
    Raises PolicyGenerationError if target cannot be generated.
    """
    # Need action_dim from the environment config associated with the root state
    if not root_node.is_root or root_node.initial_game_state is None:
        raise PolicyGenerationError(
            "Cannot get policy target from non-root node without initial game state."
        )
    # Cast ACTION_DIM to int
    action_dim = int(root_node.initial_game_state.env_config.ACTION_DIM)
    full_target = dict.fromkeys(range(action_dim), 0.0)

    if not root_node.children or root_node.visit_count == 0:
        state_info = (
            f"Step {root_node.initial_game_state.current_step}"
            if root_node.is_root and root_node.initial_game_state
            else f"Action {root_node.action_taken}"
        )
        logger.warning(
            f"[PolicyTarget] Cannot compute policy target: Root node ({state_info}) has no children or zero visits. Returning zero target."
        )
        return full_target

    child_visits = {
        action: child.visit_count for action, child in root_node.children.items()
    }
    actions = list(child_visits.keys())
    visits = np.array(list(child_visits.values()), dtype=np.float64)
    total_visits = np.sum(visits)

    if not actions:
        logger.warning(
            "[PolicyTarget] Cannot compute policy target: No actions found in children."
        )
        return full_target

    if total_visits == 0:
        logger.warning(
            "[PolicyTarget] Total visits is zero, returning uniform target over valid actions."
        )
        num_valid = len(actions)
        if num_valid > 0:
            prob = 1.0 / num_valid
            for a in actions:
                if 0 <= a < action_dim:
                    full_target[a] = prob
        return full_target

    if temperature == 0.0:
        max_visits = np.max(visits)
        if max_visits == 0:
            logger.warning(
                "[PolicyTarget] Temperature is 0 but max visits is 0. Returning zero target."
            )
            return full_target

        best_actions = [actions[i] for i, v in enumerate(visits) if v == max_visits]
        prob = 1.0 / len(best_actions)
        for a in best_actions:
            if 0 <= a < action_dim:
                full_target[a] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Best action {a} is out of bounds ({action_dim}). Skipping."
                )

    else:
        visit_probs = visits ** (1.0 / temperature)
        sum_probs = np.sum(visit_probs)

        if sum_probs < 1e-9 or not np.isfinite(sum_probs):
            raise PolicyGenerationError(
                f"Could not normalize policy target probabilities (sum={sum_probs}). Visits: {visits}"
            )

        probabilities = visit_probs / sum_probs
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise PolicyGenerationError(
                f"Invalid probabilities generated for policy target: {probabilities}"
            )
        if abs(np.sum(probabilities) - 1.0) > 1e-5:
            logger.warning(
                f"[PolicyTarget] Target probabilities sum to {np.sum(probabilities):.6f}. Re-normalizing."
            )
            probabilities /= np.sum(probabilities)
            if abs(np.sum(probabilities) - 1.0) > 1e-5:
                raise PolicyGenerationError(
                    f"Target probabilities still do not sum to 1 after re-normalization: {probabilities}, Sum: {np.sum(probabilities)}"
                )

        raw_policy = {action: probabilities[i] for i, action in enumerate(actions)}
        for action, prob in raw_policy.items():
            if 0 <= action < action_dim:
                full_target[action] = prob
            else:
                logger.warning(
                    f"[PolicyTarget] Action {action} from MCTS children is out of bounds ({action_dim}). Skipping."
                )

    final_sum = sum(full_target.values())
    if abs(final_sum - 1.0) > 1e-5 and total_visits > 0:
        logger.error(
            f"[PolicyTarget] Final policy target does not sum to 1 ({final_sum:.6f}). Target: {full_target}"
        )

    return full_target


File: muzerotriangle\mcts\strategy\README.md
# File: muzerotriangle/mcts/strategy/README.md
# MCTS Strategy Submodule (`muzerotriangle.mcts.strategy`)

## Purpose and Architecture

This submodule implements the specific algorithms and heuristics used within the different phases of the Monte Carlo Tree Search, as orchestrated by [`muzerotriangle.mcts.core.search.run_mcts_simulations`](../core/search.py).

-   **[`selection`](selection.py):** Contains the logic for traversing the tree from the root to a leaf node.
    -   `calculate_puct_score`: Implements the PUCT (Polynomial Upper Confidence Trees) formula, balancing exploitation (node value) and exploration (prior probability and visit counts).
    -   `add_dirichlet_noise`: Adds noise to the root node's prior probabilities to encourage exploration early in the search, as done in AlphaZero.
    -   `select_child_node`: Chooses the child with the highest PUCT score.
    -   `traverse_to_leaf`: Repeatedly applies `select_child_node` to navigate down the tree.
-   **[`expansion`](expansion.py):** Handles the expansion of a selected leaf node.
    -   `expand_node`: Takes a node, a policy prediction (from the network's `f` function), the network itself (for the `g` function), and optionally a list of valid actions. It creates child `Node` objects by applying the dynamics function (`g`) for each action, storing the resulting hidden state (`s_{k+1}`) and predicted reward (`r_{k+1}`). **It handles potential gradient issues by detaching tensors before storing them in the node.**
-   **[`backpropagation`](backpropagation.py):** Implements the update step after a simulation.
    -   `backpropagate_value`: Traverses from the expanded leaf node back up to the root, incrementing the `visit_count` and adding the simulation's resulting `value` (calculated using the network's value prediction and discounted rewards) to the `value_sum` of each node along the path.
-   **[`policy`](policy.py):** Provides functions related to action selection and policy target generation after MCTS has run.
    -   `select_action_based_on_visits`: Selects the final action to be played in the game based on the visit counts of the root's children, using a temperature parameter to control exploration vs. exploitation.
    -   `get_policy_target`: Generates the policy target vector (a probability distribution over actions) based on the visit counts, which is used as a training target for the neural network's policy head.

## Exposed Interfaces

-   **Selection:**
    -   `traverse_to_leaf(root_node: Node, config: MCTSConfig) -> Tuple[Node, int]`: Returns leaf node and depth.
    -   `add_dirichlet_noise(node: Node, config: MCTSConfig)`
    -   `select_child_node(node: Node, config: MCTSConfig) -> Node` (Primarily internal use)
    -   `calculate_puct_score(...) -> Tuple[float, float, float]` (Primarily internal use)
    -   `SelectionError`: Custom exception.
-   **Expansion:**
    -   `expand_node(node_to_expand: Node, policy_prediction: ActionPolicyMapping, network: NeuralNetwork, valid_actions: Optional[List[ActionType]] = None)`
-   **Backpropagation:**
    -   `backpropagate_value(leaf_node: Node, value: float, discount: float) -> int`: Returns depth.
-   **Policy:**
    -   `select_action_based_on_visits(root_node: Node, temperature: float) -> ActionType`: Selects the final move.
    -   `get_policy_target(root_node: Node, temperature: float = 1.0) -> ActionPolicyMapping`: Generates the training policy target.
    -   `PolicyGenerationError`: Custom exception.

## Dependencies

-   **[`muzerotriangle.mcts.core`](../core/README.md)**:
    -   `Node`: The primary data structure operated upon.
    -   `ActionPolicyMapping`: Used in `expansion` and `policy`.
-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `MCTSConfig`: Provides hyperparameters (PUCT coeff, noise params, etc.).
-   **[`muzerotriangle.nn`](../../nn/README.md)**:
    -   `NeuralNetwork`: Used by `expansion`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**:
    -   `ActionType`: Used for representing actions.
-   **`numpy`**:
    -   Used for Dirichlet noise and policy/selection calculations.
-   **`torch`**:
    -   Used for hidden states.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `random`.

---

**Note:** Please keep this README updated when modifying the algorithms within selection, expansion, backpropagation, or policy generation, or changing how they interact with the `Node` structure or `MCTSConfig`. Accurate documentation is crucial for maintainability.

File: muzerotriangle\mcts\strategy\selection.py
# File: muzerotriangle/mcts/strategy/selection.py
import logging
import math
import random

import numpy as np

from ...config import MCTSConfig
from ..core.node import Node

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class SelectionError(Exception):
    """Custom exception for errors during node selection."""

    pass


def calculate_puct_score(
    parent_node: "Node",  # The node *from* which we are selecting a child
    child_node: "Node",  # The child node being evaluated
    config: MCTSConfig,
) -> tuple[float, float, float]:
    """
    Calculates the PUCT score for a child node, used for selection from the parent.
    Score = Q(parent, action_to_child) + U(parent, action_to_child)
    Q value is the average value derived from simulations passing through the child.
    U value is the exploration bonus based on the child's prior and visit counts.
    """
    # Q(s, a) is the value estimate of the child node itself
    # It represents the expected return *after* taking 'action_taken' from the parent.
    q_value = child_node.value_estimate

    # P(a|s) is the prior probability stored in the child node
    prior = child_node.prior_probability
    parent_visits = parent_node.visit_count
    child_visits = child_node.visit_count

    # Exploration bonus U(s, a)
    # Use max(1, parent_visits) to avoid math domain error if parent_visits is 0 (though it shouldn't be if we're selecting a child)
    exploration_term = (
        config.puct_coefficient
        * prior
        * (math.sqrt(max(1, parent_visits)) / (1 + child_visits))
    )
    score = q_value + exploration_term

    # Ensure score is finite
    if not np.isfinite(score):
        logger.warning(
            f"Non-finite PUCT score calculated (Q={q_value}, P={prior}, ChildN={child_visits}, ParentN={parent_visits}, Exp={exploration_term}). Defaulting to Q-value."
        )
        score = q_value
        exploration_term = 0.0

    return score, q_value, exploration_term


def add_dirichlet_noise(node: Node, config: MCTSConfig):
    """Adds Dirichlet noise to the prior probabilities of the children of this node."""
    if (
        config.dirichlet_alpha <= 0.0
        or config.dirichlet_epsilon <= 0.0
        or not node.children
        or len(node.children) <= 1
    ):
        return

    actions = list(node.children.keys())
    noise = rng.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon

    noisy_priors_sum = 0.0
    for i, action in enumerate(actions):
        child = node.children[action]
        original_prior = child.prior_probability
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]
        noisy_priors_sum += child.prior_probability
        logger.debug(
            f"  [Noise] Action {action}: OrigP={original_prior:.4f}, Noise={noise[i]:.4f} -> NewP={child.prior_probability:.4f}"
        )

    # Re-normalize priors
    if abs(noisy_priors_sum - 1.0) > 1e-6:
        logger.debug(
            f"Re-normalizing priors after Dirichlet noise (Sum={noisy_priors_sum:.6f})"
        )
        norm_factor = noisy_priors_sum if noisy_priors_sum > 1e-9 else 1.0
        if norm_factor > 1e-9:
            for action in actions:
                node.children[action].prior_probability /= norm_factor
        else:
            # If sum is zero (e.g., all priors were zero), distribute uniformly
            logger.warning(
                "Sum of priors after noise is near zero. Resetting to uniform."
            )
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                node.children[action].prior_probability = uniform_prob

    logger.debug(
        f"[Noise] Added Dirichlet noise (alpha={config.dirichlet_alpha}, eps={eps}) to {len(actions)} root node priors."
    )


def select_child_node(node: Node, config: MCTSConfig) -> Node:
    """
    Selects the child node with the highest PUCT score. Assumes noise already added if root.
    Raises SelectionError if no valid child can be selected.
    """
    if not node.children:
        raise SelectionError(f"Cannot select child from node {node} with no children.")

    best_score = -float("inf")
    best_children: list[Node] = []  # Use a list for tie-breaking

    if logger.isEnabledFor(logging.DEBUG):
        # Check if root node to display step correctly
        state_info = (
            f"Step={node.initial_game_state.current_step}"
            if node.is_root and node.initial_game_state
            else f"Action={node.action_taken}"
        )
        logger.debug(
            f"  [Select] Selecting child for Node (Visits={node.visit_count}, Children={len(node.children)}, {state_info}):"
        )

    for action, child in node.children.items():
        # Pass the parent (current node) and the child being evaluated
        score, q, exp_term = calculate_puct_score(node, child, config)

        if logger.isEnabledFor(logging.DEBUG):
            log_entry = (
                f"    Act={action}, Score={score:.4f} "
                f"(Q={q:.3f}, P={child.prior_probability:.4f}, N={child.visit_count}, Exp={exp_term:.4f})"
            )
            logger.debug(log_entry)  # Log each child's score

        if not np.isfinite(score):
            logger.warning(
                f"    [Select] Non-finite PUCT score ({score}) calculated for child action {action}. Skipping."
            )
            continue

        if score > best_score:
            best_score = score
            best_children = [child]  # Start a new list with this best child
        # Tie-breaking: if scores are equal, add to the list
        elif abs(score - best_score) < 1e-9:  # Use tolerance for float comparison
            best_children.append(child)

    if not best_children:  # Changed from checking best_child is None
        child_details = [
            f"Act={a}, N={c.visit_count}, P={c.prior_probability:.4f}, Q={c.value_estimate:.3f}"
            for a, c in node.children.items()
        ]
        state_info = (
            f"Root Step {node.initial_game_state.current_step}"
            if node.is_root and node.initial_game_state
            else f"Node Action {node.action_taken}"
        )
        logger.error(
            f"Could not select best child for {state_info}. Child details: {child_details}"
        )
        # Fallback: if all scores are non-finite or no children found, pick a random child
        if not node.children:
            raise SelectionError(
                f"Cannot select child from node {node} with no children (should have been caught earlier)."
            )
        logger.warning(
            f"All child scores were non-finite or no children found. Selecting a random child for node {state_info}."
        )
        # Ensure we actually have children before choosing randomly
        if not node.children:
            raise SelectionError(
                f"Node {state_info} has no children to select from, even randomly."
            )
        selected_child = random.choice(list(node.children.values()))
    else:
        # If there are ties, select randomly among the best
        selected_child = random.choice(best_children)

    logger.debug(
        f"  [Select] --> Selected Child: Action {selected_child.action_taken}, Score {best_score:.4f}, Q-value {selected_child.value_estimate:.3f}"
    )
    return selected_child


def traverse_to_leaf(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """
    Traverses the tree from root to a leaf node using PUCT selection.
    A leaf is defined as a node that has not been expanded.
    Stops also if the maximum search depth has been reached.
    Note: Terminal state check now happens during expansion/prediction.
    Raises SelectionError if child selection fails during traversal.
    Returns the leaf node and the depth reached.
    """
    current_node = root_node
    depth = 0
    state_info = (
        f"Root Step {root_node.initial_game_state.current_step}"
        if root_node.is_root and root_node.initial_game_state
        else f"Node Action {root_node.action_taken}"
    )
    logger.debug(f"[Traverse] --- Start Traverse (Start Node: {state_info}) ---")
    stop_reason = "Unknown"

    while current_node.is_expanded:  # Traverse while node has children
        state_info = (
            f"Root Step {current_node.initial_game_state.current_step}"
            if current_node.is_root and current_node.initial_game_state
            else f"Node Action {current_node.action_taken}"
        )
        logger.debug(
            f"  [Traverse] Depth {depth}: Considering Node {state_info} (Expanded={current_node.is_expanded})"
        )

        if config.max_search_depth is not None and depth >= config.max_search_depth:
            stop_reason = "Max Depth Reached"
            logger.debug(
                f"  [Traverse] Depth {depth}: Hit MAX DEPTH ({config.max_search_depth}). Stopping traverse."
            )
            break

        # Node is expanded and below max depth - select child
        try:
            selected_child = select_child_node(current_node, config)
            logger.debug(
                f"  [Traverse] Depth {depth}: Selected child with action {selected_child.action_taken}"
            )
            current_node = selected_child
            depth += 1
        except SelectionError as e:
            stop_reason = f"Child Selection Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Error during child selection: {e}. Breaking traverse.",
                exc_info=False,
            )
            logger.warning(
                f"  [Traverse] Returning current node {current_node.action_taken} due to SelectionError."
            )
            break
        except Exception as e:
            stop_reason = f"Unexpected Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Unexpected error during child selection: {e}. Breaking traverse.",
                exc_info=True,
            )
            logger.warning(
                f"  [Traverse] Returning current node {current_node.action_taken} due to Unexpected Error."
            )
            break
    else:
        # Loop finished because node is not expanded (it's a leaf)
        stop_reason = "Unexpanded Leaf"
        state_info = (
            f"Root Step {current_node.initial_game_state.current_step}"
            if current_node.is_root and current_node.initial_game_state
            else f"Node Action {current_node.action_taken}"
        )
        logger.debug(
            f"  [Traverse] Depth {depth}: Node {state_info} is LEAF (not expanded). Stopping traverse."
        )

    state_info_final = (
        f"Root Step {current_node.initial_game_state.current_step}"
        if current_node.is_root and current_node.initial_game_state
        else f"Node Action {current_node.action_taken}"
    )
    logger.debug(
        f"[Traverse] --- End Traverse: Reached Node {state_info_final} at Depth {depth}. Reason: {stop_reason} ---"
    )
    return current_node, depth


File: muzerotriangle\mcts\strategy\__init__.py


File: muzerotriangle\nn\model.py
# File: muzerotriangle/nn/model.py
import logging
import math
from typing import cast  # Keep cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EnvConfig, ModelConfig

logger = logging.getLogger(__name__)


# --- conv_block, ResidualBlock, PositionalEncoding remain the same ---
def conv_block(
    in_channels, out_channels, kernel_size, stride, padding, use_batch_norm, activation
):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=not use_batch_norm,
        )
    ]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_batch_norm, activation):
        super().__init__()
        self.conv1 = conv_block(channels, channels, 3, 1, 1, use_batch_norm, activation)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = x
        out: torch.Tensor = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out_sum: torch.Tensor = out + residual
        out_activated: torch.Tensor = self.activation(out_sum)
        return out_activated


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe_buffer = self.pe
        assert isinstance(pe_buffer, torch.Tensor)
        if x.shape[0] > pe_buffer.shape[0]:
            raise ValueError(f"Seq len {x.shape[0]} > max_len {pe_buffer.shape[0]}")
        if x.shape[2] != pe_buffer.shape[2]:
            raise ValueError(f"Dim {x.shape[2]} != PE dim {pe_buffer.shape[2]}")
        x = x + pe_buffer[: x.size(0)]
        return cast("torch.Tensor", self.dropout(x))


# --- MuZeroNet Implementation ---
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ConcatFeatures(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, features_tuple):
        return torch.cat(features_tuple, dim=self.dim)


class RepresentationEncoderWrapper(nn.Module):
    """Wraps CNN/TF layers for representation fn. Returns flattened tensor."""

    def __init__(self, cnn_tf_layers: nn.Module):
        super().__init__()
        self.cnn_tf_layers = cnn_tf_layers
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, grid_state: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.cnn_tf_layers(grid_state)
        encoded_flat: torch.Tensor = (
            self.flatten(encoded) if len(encoded.shape) > 2 else encoded
        )
        return encoded_flat


class MuZeroNet(nn.Module):
    """MuZero Network Implementation."""

    def __init__(self, model_config: ModelConfig, env_config: EnvConfig):
        super().__init__()
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(env_config.ACTION_DIM)
        self.hidden_dim = model_config.HIDDEN_STATE_DIM
        self.activation_cls: type[nn.Module] = getattr(
            nn, model_config.ACTIVATION_FUNCTION
        )
        dummy_input_grid = torch.zeros(
            1, model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS
        )
        self.representation_encoder = self._build_representation_cnn_tf_encoder()
        with torch.no_grad():
            encoded_output = self.representation_encoder(dummy_input_grid)
            self.encoded_flat_size = encoded_output.shape[1]
        rep_projector_input_dim = (
            self.encoded_flat_size + model_config.OTHER_NN_INPUT_FEATURES_DIM
        )
        rep_fc_layers: list[nn.Module] = []
        in_features = rep_projector_input_dim
        for hidden_dim_fc in model_config.REP_FC_DIMS_AFTER_ENCODER:
            rep_fc_layers.append(nn.Linear(in_features, hidden_dim_fc))
            in_features = hidden_dim_fc
        rep_fc_layers.append(nn.Linear(in_features, self.hidden_dim))
        self.representation_projector = nn.Sequential(*rep_fc_layers)
        self.action_encoder = nn.Linear(
            self.action_dim, model_config.ACTION_ENCODING_DIM
        )
        dynamics_input_dim = self.hidden_dim + model_config.ACTION_ENCODING_DIM
        dynamics_layers: list[nn.Module] = [
            nn.Linear(dynamics_input_dim, self.hidden_dim),
            self.activation_cls(),
        ]
        for _ in range(model_config.DYNAMICS_NUM_RESIDUAL_BLOCKS):
            dynamics_layers.extend(
                [nn.Linear(self.hidden_dim, self.hidden_dim), self.activation_cls()]
            )
        self.dynamics_core = nn.Sequential(*dynamics_layers)
        reward_head_layers: list[nn.Module] = []
        reward_in = self.hidden_dim
        for hidden_dim_fc in model_config.REWARD_HEAD_DIMS:
            reward_head_layers.append(nn.Linear(reward_in, hidden_dim_fc))
            reward_in = hidden_dim_fc
        reward_head_layers.append(
            nn.Linear(reward_in, model_config.REWARD_SUPPORT_SIZE)
        )
        self.reward_head = nn.Sequential(*reward_head_layers)
        prediction_layers: list[nn.Module] = []
        pred_in = self.hidden_dim
        for _ in range(model_config.PREDICTION_NUM_RESIDUAL_BLOCKS):
            prediction_layers.extend(
                [nn.Linear(pred_in, self.hidden_dim), self.activation_cls()]
            )
            pred_in = self.hidden_dim
        self.prediction_core = nn.Sequential(*prediction_layers)
        policy_head_layers: list[nn.Module] = []
        policy_in = self.hidden_dim
        for hidden_dim_fc in model_config.POLICY_HEAD_DIMS:
            policy_head_layers.append(nn.Linear(policy_in, hidden_dim_fc))
            policy_in = hidden_dim_fc
        policy_head_layers.append(nn.Linear(policy_in, self.action_dim))
        self.policy_head = nn.Sequential(*policy_head_layers)
        value_head_layers: list[nn.Module] = []
        value_in = self.hidden_dim
        for hidden_dim_fc in model_config.VALUE_HEAD_DIMS:
            value_head_layers.append(nn.Linear(value_in, hidden_dim_fc))
            value_in = hidden_dim_fc
        value_head_layers.append(nn.Linear(value_in, model_config.NUM_VALUE_ATOMS))
        self.value_head = nn.Sequential(*value_head_layers)

    def _build_representation_cnn_tf_encoder(self) -> RepresentationEncoderWrapper:
        layers: list[nn.Module] = []
        in_channels = self.model_config.GRID_INPUT_CHANNELS
        for i, out_channels in enumerate(self.model_config.CONV_FILTERS):
            layers.append(
                conv_block(
                    in_channels,
                    out_channels,
                    self.model_config.CONV_KERNEL_SIZES[i],
                    self.model_config.CONV_STRIDES[i],
                    self.model_config.CONV_PADDING[i],
                    self.model_config.USE_BATCH_NORM,
                    self.activation_cls,
                )
            )
            in_channels = out_channels
        if self.model_config.NUM_RESIDUAL_BLOCKS > 0:
            res_channels = self.model_config.RESIDUAL_BLOCK_FILTERS
            if in_channels != res_channels:
                layers.append(
                    conv_block(
                        in_channels,
                        res_channels,
                        1,
                        1,
                        0,
                        self.model_config.USE_BATCH_NORM,
                        self.activation_cls,
                    )
                )
                in_channels = res_channels
            for _ in range(self.model_config.NUM_RESIDUAL_BLOCKS):
                layers.append(
                    ResidualBlock(
                        in_channels,
                        self.model_config.USE_BATCH_NORM,
                        self.activation_cls,
                    )
                )
        if (
            self.model_config.USE_TRANSFORMER_IN_REP
            and self.model_config.REP_TRANSFORMER_LAYERS > 0
        ):
            transformer_input_dim = self.hidden_dim
            if in_channels != transformer_input_dim:
                layers.append(
                    nn.Conv2d(in_channels, transformer_input_dim, kernel_size=1)
                )
                in_channels = transformer_input_dim
            pos_encoder = PositionalEncoding(transformer_input_dim, dropout=0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_input_dim,
                nhead=self.model_config.REP_TRANSFORMER_HEADS,
                dim_feedforward=self.model_config.REP_TRANSFORMER_FC_DIM,
                activation=self.model_config.ACTIVATION_FUNCTION.lower(),
                batch_first=False,
                norm_first=True,
            )
            transformer_norm = nn.LayerNorm(transformer_input_dim)
            transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.model_config.REP_TRANSFORMER_LAYERS,
                norm=transformer_norm,
            )
            layers.append(nn.Flatten(start_dim=2))
            layers.append(Permute(2, 0, 1))
            layers.append(pos_encoder)
            layers.append(transformer_encoder)
            layers.append(Permute(1, 0, 2))
            layers.append(nn.Flatten(start_dim=1))
        else:
            layers.append(nn.Flatten(start_dim=1))
        return RepresentationEncoderWrapper(nn.Sequential(*layers))

    def represent(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> torch.Tensor:
        encoded_grid_flat = self.representation_encoder(grid_state)
        combined_features = torch.cat([encoded_grid_flat, other_features], dim=1)
        hidden_state = self.representation_projector(combined_features)
        # Explicitly cast the output to satisfy MyPy
        return cast("torch.Tensor", hidden_state)

    def dynamics(self, hidden_state, action) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(action, int) or (
            isinstance(action, torch.Tensor) and action.numel() == 1
        ):
            action_tensor = cast(
                "torch.Tensor", torch.tensor([action], device=hidden_state.device)
            )
            action_one_hot = F.one_hot(
                action_tensor, num_classes=self.action_dim
            ).float()
        elif isinstance(action, torch.Tensor) and action.dim() == 1:
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        elif (
            isinstance(action, torch.Tensor)
            and action.dim() == 2
            and action.shape[1] == self.action_dim
        ):
            action_one_hot = action
        else:
            raise TypeError(
                f"Unsupported action type/shape: {type(action), action.shape if isinstance(action, torch.Tensor) else ''}"
            )
        action_embedding = self.action_encoder(action_one_hot)
        dynamics_input = torch.cat([hidden_state, action_embedding], dim=1)
        next_hidden_state = self.dynamics_core(dynamics_input)
        reward_logits = self.reward_head(next_hidden_state)
        return next_hidden_state, reward_logits

    def predict(self, hidden_state) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_features = self.prediction_core(hidden_state)
        policy_logits = self.policy_head(prediction_features)
        value_logits = self.value_head(prediction_features)
        return policy_logits, value_logits

    def forward(
        self, grid_state: torch.Tensor, other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial_hidden_state = self.represent(grid_state, other_features)
        policy_logits, value_logits = self.predict(initial_hidden_state)
        return policy_logits, value_logits, initial_hidden_state


File: muzerotriangle\nn\network.py
# File: muzerotriangle/nn/network.py
import logging
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..config import EnvConfig, ModelConfig, TrainConfig
from ..environment import GameState
from ..features import extract_state_features
from ..utils.types import ActionType, PolicyValueOutput, StateType
from .model import MuZeroNet  # Import MuZeroNet

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class NetworkEvaluationError(Exception):
    """Custom exception for errors during network evaluation."""

    pass


class NeuralNetwork:
    """
    Wrapper for the MuZeroNet model providing methods for representation,
    dynamics, and prediction, as well as initial inference.
    Handles distributional value/reward heads.
    Optionally compiles the model using torch.compile().
    """

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,
        train_config: TrainConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.env_config = env_config
        self.train_config = train_config
        self.device = device
        self.model = MuZeroNet(model_config, env_config).to(device)
        self.action_dim = env_config.ACTION_DIM
        self.model.eval()

        # Distributional Value Head parameters
        self.num_value_atoms = model_config.NUM_VALUE_ATOMS
        self.v_min = model_config.VALUE_MIN
        self.v_max = model_config.VALUE_MAX
        if self.num_value_atoms <= 1:
            raise ValueError("NUM_VALUE_ATOMS must be greater than 1")
        self.delta_z = (self.v_max - self.v_min) / (self.num_value_atoms - 1)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_value_atoms, device=self.device
        )

        # Distributional Reward Head parameters (assuming symmetric support around 0)
        self.num_reward_atoms = model_config.REWARD_SUPPORT_SIZE
        if self.num_reward_atoms <= 1:
            raise ValueError("REWARD_SUPPORT_SIZE must be greater than 1")
        # Calculate reward min/max based on support size (e.g., size 21 -> -10 to 10)
        self.r_max = float((self.num_reward_atoms - 1) // 2)
        self.r_min = -self.r_max
        self.delta_r = 1.0  # Assuming integer steps for reward support
        self.reward_support = torch.linspace(
            self.r_min, self.r_max, self.num_reward_atoms, device=self.device
        )

        # Compile model if requested and compatible
        self._try_compile_model()

    def _try_compile_model(self):
        """Attempts to compile the model if configured and compatible."""
        if not self.train_config.COMPILE_MODEL:
            logger.info("Model compilation skipped (COMPILE_MODEL=False).")
            return

        if sys.platform == "win32":
            logger.warning("Model compilation skipped on Windows (Triton dependency).")
            return
        if self.device.type == "mps":
            logger.warning("Model compilation skipped on MPS (compatibility issues).")
            return
        if not hasattr(torch, "compile"):
            logger.warning("Model compilation skipped (torch.compile not available).")
            return

        try:
            logger.info(
                f"Attempting to compile model with torch.compile() on device '{self.device}'..."
            )
            # Compile the underlying MuZeroNet instance
            self.model = torch.compile(self.model)  # type: ignore
            logger.info(f"Model compiled successfully on device '{self.device}'.")
        except Exception as e:
            logger.warning(
                f"torch.compile() failed on device '{self.device}': {e}. Proceeding without compilation.",
                exc_info=False,
            )

    def _state_to_tensors(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from GameState and converts them to tensors."""
        state_dict: StateType = extract_state_features(state, self.model_config)
        grid_tensor = torch.from_numpy(state_dict["grid"]).unsqueeze(0).to(self.device)
        other_features_tensor = (
            torch.from_numpy(state_dict["other_features"]).unsqueeze(0).to(self.device)
        )
        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError("Non-finite values in input grid_tensor")
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values in input other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    def _batch_states_to_tensors(
        self, states: list[GameState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from a batch of GameStates and converts to batched tensors."""
        if not states:
            grid_shape = (
                0,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            )
            other_shape = (0, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
            return torch.empty(grid_shape, device=self.device), torch.empty(
                other_shape, device=self.device
            )

        batch_grid = []
        batch_other = []
        for state in states:
            state_dict: StateType = extract_state_features(state, self.model_config)
            batch_grid.append(state_dict["grid"])
            batch_other.append(state_dict["other_features"])

        grid_tensor = torch.from_numpy(np.stack(batch_grid)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(batch_other)).to(self.device)

        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError("Non-finite values in batched grid_tensor")
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values in batched other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to probabilities using softmax."""
        return F.softmax(logits, dim=-1)

    def _logits_to_scalar(
        self, logits: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the expected scalar value from distribution logits."""
        probs = self._logits_to_probs(logits)
        # Expand support to match batch size if needed
        support_expanded = support.expand_as(probs)
        scalar = torch.sum(probs * support_expanded, dim=-1)
        return scalar

    @torch.inference_mode()
    def initial_inference(
        self, observation: StateType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the initial inference h(o) -> s_0 and f(s_0) -> p_0, v_0.
        Args:
            observation: The StateType dictionary from feature extraction.
        Returns:
            Tuple: (policy_logits, value_logits, reward_logits (dummy), initial_hidden_state)
                   Reward logits are dummy here as they come from dynamics.
        """
        self.model.eval()
        grid_tensor = torch.as_tensor(
            observation["grid"], dtype=torch.float32, device=self.device
        )
        other_features_tensor = torch.as_tensor(
            observation["other_features"], dtype=torch.float32, device=self.device
        )

        # Add batch dimension if necessary
        if grid_tensor.dim() == 3:
            grid_tensor = grid_tensor.unsqueeze(0)
        if other_features_tensor.dim() == 1:
            other_features_tensor = other_features_tensor.unsqueeze(0)

        policy_logits, value_logits, initial_hidden_state = self.model(
            grid_tensor, other_features_tensor
        )

        # Create dummy reward logits (batch_size, num_reward_atoms)
        # Initial state doesn't have a predicted reward from dynamics
        dummy_reward_logits = torch.zeros(
            (1, self.num_reward_atoms), device=self.device
        )

        return policy_logits, value_logits, dummy_reward_logits, initial_hidden_state

    @torch.inference_mode()
    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: ActionType | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one step of recurrent inference:
        g(s_{k-1}, a_k) -> s_k, r_k
        f(s_k) -> p_k, v_k
        Args:
            hidden_state: The previous hidden state (s_{k-1}). Shape [B, H] or [H].
            action: The action taken (a_k). Can be int or Tensor.
        Returns:
            Tuple: (policy_logits, value_logits, reward_logits, next_hidden_state)
                   All tensors will have a batch dimension.
        """
        self.model.eval()
        # Ensure hidden_state has a batch dimension
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # Ensure action is a tensor with a batch dimension
        if isinstance(action, int):
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        elif isinstance(action, torch.Tensor):
            if action.dim() == 0:  # Scalar tensor
                action_tensor = action.unsqueeze(0).to(self.device)
            elif action.dim() == 1:  # Already a batch of actions
                action_tensor = action.to(self.device)
            else:
                raise ValueError(f"Unsupported action tensor shape: {action.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Ensure action_tensor has the same batch size as hidden_state
        if action_tensor.shape[0] != hidden_state.shape[0]:
            if hidden_state.shape[0] == 1 and action_tensor.shape[0] > 1:
                # Repeat hidden state if it's a single state for a batch of actions
                hidden_state = hidden_state.expand(action_tensor.shape[0], -1)
            elif action_tensor.shape[0] == 1 and hidden_state.shape[0] > 1:
                # Repeat action if it's a single action for a batch of states
                action_tensor = action_tensor.expand(hidden_state.shape[0])
            else:
                raise ValueError(
                    f"Batch size mismatch between hidden_state ({hidden_state.shape[0]}) and action ({action_tensor.shape[0]})"
                )

        next_hidden_state, reward_logits = self.model.dynamics(
            hidden_state, action_tensor
        )
        policy_logits, value_logits = self.model.predict(next_hidden_state)
        return policy_logits, value_logits, reward_logits, next_hidden_state

    # --- Compatibility methods for MCTS/Workers expecting PolicyValueOutput ---
    # These now perform initial inference.

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """
        Evaluates a single state using initial inference (h + f).
        Returns policy mapping and EXPECTED scalar value from the distribution.
        """
        self.model.eval()
        try:
            # 1. Feature Extraction
            state_dict: StateType = extract_state_features(state, self.model_config)
            # 2. Initial Inference
            policy_logits, value_logits, _, _ = self.initial_inference(state_dict)

            # 3. Process Outputs
            policy_probs_tensor = self._logits_to_probs(policy_logits)
            expected_value_tensor = self._logits_to_scalar(value_logits, self.support)

            # Validate and normalize policy probabilities
            policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
            if not np.all(np.isfinite(policy_probs)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax for state {state.current_step}."
                )
            policy_probs = np.maximum(policy_probs, 0)
            prob_sum = np.sum(policy_probs)
            if abs(prob_sum - 1.0) > 1e-5:
                logger.warning(
                    f"Evaluate: Policy probabilities sum to {prob_sum:.6f}. Re-normalizing."
                )
                if prob_sum <= 1e-9:
                    policy_probs.fill(1.0 / len(policy_probs))
                else:
                    policy_probs /= prob_sum

            # Convert to expected output format
            action_policy: Mapping[ActionType, float] = {
                i: float(p) for i, p in enumerate(policy_probs)
            }
            expected_value_scalar = expected_value_tensor.item()

            return action_policy, expected_value_scalar

        except Exception as e:
            logger.error(f"Exception during single evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(
                f"Evaluation failed for state {state}: {e}"
            ) from e

    @torch.inference_mode()
    def evaluate_batch(self, states: list[GameState]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of states using initial inference (h + f).
        Returns a list of (policy mapping, EXPECTED scalar value).
        """
        if not states:
            return []
        self.model.eval()
        try:
            # 1. Batch Feature Extraction
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)

            # 2. Batch Initial Inference (using model's forward)
            policy_logits, value_logits, _ = self.model(
                grid_tensor, other_features_tensor
            )

            # 3. Batch Process Outputs
            policy_probs_tensor = self._logits_to_probs(policy_logits)
            expected_values_tensor = self._logits_to_scalar(value_logits, self.support)

            # Validate and normalize policies
            policy_probs = policy_probs_tensor.cpu().numpy()
            expected_values = expected_values_tensor.cpu().numpy()

            results: list[PolicyValueOutput] = []
            for batch_idx in range(len(states)):
                probs_i = policy_probs[batch_idx]
                if not np.all(np.isfinite(probs_i)):
                    raise NetworkEvaluationError(
                        f"Non-finite policy probabilities AFTER softmax for batch item {batch_idx}."
                    )
                probs_i = np.maximum(probs_i, 0)
                prob_sum_i = np.sum(probs_i)
                if abs(prob_sum_i - 1.0) > 1e-5:
                    logger.warning(
                        f"EvaluateBatch: Policy probs sum to {prob_sum_i:.6f} for item {batch_idx}. Re-normalizing."
                    )
                    if prob_sum_i <= 1e-9:
                        probs_i.fill(1.0 / len(probs_i))
                    else:
                        probs_i /= prob_sum_i

                policy_i: Mapping[ActionType, float] = {
                    i: float(p) for i, p in enumerate(probs_i)
                }
                value_i = float(expected_values[batch_idx])
                results.append((policy_i, value_i))

            return results

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(f"Batch evaluation failed: {e}") from e

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        model_to_save = getattr(self.model, "_orig_mod", self.model)
        return {k: v.cpu() for k, v in model_to_save.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]):
        """Loads the model's state dictionary from the provided weights."""
        try:
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            model_to_load = getattr(self.model, "_orig_mod", self.model)
            model_to_load.load_state_dict(weights_on_device)
            self.model.eval()  # Ensure model is in eval mode after loading weights
            logger.debug("NN weights set successfully.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise


File: muzerotriangle\nn\README.md
# File: muzerotriangle/nn/README.md
# Neural Network Module (`muzerotriangle.nn`)

## Purpose and Architecture

This module defines and manages the neural network used by the MuZeroTriangle agent. It implements the core MuZero architecture consisting of representation, dynamics, and prediction functions.

-   **Model Definition ([`model.py`](model.py)):**
    -   The `MuZeroNet` class (inheriting from `torch.nn.Module`) defines the network architecture.
    -   It implements three core functions:
        -   **Representation Function (h):** Takes an observation (from `features.extract_state_features`) as input and outputs an initial *hidden state* (`s_0`). This typically involves CNNs, potentially ResNets or Transformers, followed by a projection.
        -   **Dynamics Function (g):** Takes a previous hidden state (`s_{k-1}`) and an action (`a_k`) as input. It outputs the *next hidden state* (`s_k`) and a *predicted reward distribution* (`r_k`). This function models the environment's transition and reward dynamics in the latent space. It often involves MLPs or RNNs combined with action embeddings.
        -   **Prediction Function (f):** Takes a hidden state (`s_k`) as input and outputs a *predicted policy distribution* (`p_k`) and a *predicted value distribution* (`v_k`). This function predicts the outcome from a given latent state.
    -   The architecture details (e.g., hidden dimensions, block types, heads) are configurable via [`ModelConfig`](../config/model_config.py).
    -   Both reward and value heads output **logits for categorical distributions**, supporting distributional RL.
-   **Network Interface ([`network.py`](network.py)):**
    -   The `NeuralNetwork` class acts as a wrapper around the `MuZeroNet` PyTorch model.
    -   It provides a clean interface for the rest of the system (MCTS, Trainer) to interact with the network's functions (`h`, `g`, `f`).
    -   It **internally uses [`muzerotriangle.features.extract_state_features`](../features/extractor.py)** to convert input `GameState` objects into observations for the representation function.
    -   It handles device placement (`torch.device`).
    -   It **optionally compiles** the underlying model using `torch.compile()` based on `TrainConfig.COMPILE_MODEL`.
    -   Key methods:
        -   `initial_inference(observation: StateType)`: Runs `h` and `f` to get `s_0`, `p_0`, `v_0`. Returns logits and hidden state. **Ensures tensors are detached before returning numpy arrays.**
        -   `recurrent_inference(hidden_state: Tensor, action: ActionType | Tensor)`: Runs `g` and `f` to get `s_k`, `r_k`, `p_k`, `v_k`. Returns logits and next hidden state. **Ensures tensors are detached before returning numpy arrays.**
        -   `evaluate(state: GameState)` / `evaluate_batch(states: List[GameState])`: Convenience methods performing initial inference and returning processed policy/value outputs (expected scalar value) for potential use by components still expecting the old interface (like initial MCTS root evaluation).
        -   `get_weights()` / `set_weights()`: Standard weight management.
    -   Provides access to value and reward support tensors (`support`, `reward_support`).

## Exposed Interfaces

-   **Classes:**
    -   `MuZeroNet(model_config: ModelConfig, env_config: EnvConfig)`: The PyTorch `nn.Module`.
    -   `NeuralNetwork(model_config: ModelConfig, env_config: EnvConfig, train_config: TrainConfig, device: torch.device)`: The wrapper class.
        -   `initial_inference(...) -> Tuple[Tensor, Tensor, Tensor, Tensor]`
        -   `recurrent_inference(...) -> Tuple[Tensor, Tensor, Tensor, Tensor]`
        -   `evaluate(...) -> PolicyValueOutput`
        -   `evaluate_batch(...) -> List[PolicyValueOutput]`
        -   `get_weights() -> Dict[str, torch.Tensor]`
        -   `set_weights(weights: Dict[str, torch.Tensor])`
        -   `model`: Public attribute (underlying `MuZeroNet`).
        -   `device`, `support`, `reward_support`: Public attributes.

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: `ModelConfig`, `EnvConfig`, `TrainConfig`.
-   **[`muzerotriangle.environment`](../environment/README.md)**: `GameState` (for `evaluate`/`evaluate_batch`).
-   **[`muzerotriangle.features`](../features/README.md)**: `extract_state_features`.
-   **[`muzerotriangle.utils`](../utils/README.md)**: `types`.
-   **`torch`**: Core deep learning framework.
-   **`numpy`**: Used for feature processing.
-   **Standard Libraries:** `typing`, `logging`, `math`, `sys`.

---

**Note:** Please keep this README updated when changing the MuZero network architecture (`MuZeroNet`), the `NeuralNetwork` interface methods, or its interaction with configuration or other modules. Accurate documentation is crucial for maintainability.

File: muzerotriangle\nn\__init__.py
"""
Neural Network module for the MuZeroTriangle agent.
Contains the MuZero model definition (h, g, f) and a wrapper interface.
"""

from .model import MuZeroNet  # Changed from AlphaTriangleNet
from .network import NeuralNetwork

__all__ = [
    "MuZeroNet",  # Changed from AlphaTriangleNet
    "NeuralNetwork",
]


File: muzerotriangle\rl\README.md
# File: muzerotriangle/rl/README.md
# Reinforcement Learning Module (`muzerotriangle.rl`)

## Purpose and Architecture

This module contains core components related to the MuZero reinforcement learning algorithm, specifically the `Trainer` for network updates, the `ExperienceBuffer` for storing data, and the `SelfPlayWorker` actor for generating data. The overall orchestration of the training process resides in the [`muzerotriangle.training`](../training/README.md) module.

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Trainer`: Responsible for performing the neural network update steps. It takes batches of sequences from the buffer, calculates losses (policy cross-entropy, distributional value/reward cross-entropy), applies importance sampling weights if using PER, updates the network weights, and calculates TD errors (based on the initial value prediction) for PER priority updates. Uses N-step reward targets for reward loss and N-step value targets for value loss.
    -   `ExperienceBuffer`: A replay buffer storing complete game `Trajectory` objects. Supports both uniform sampling and Prioritized Experience Replay (PER) based on initial TD error of sequences. Manages trajectory storage, sampling logic (including sequence selection within trajectories), and priority updates.
-   **Self-Play Components ([`self_play/README.md`](self_play/README.md)):**
    -   `worker`: Defines the `SelfPlayWorker` Ray actor. Each actor runs game episodes independently using MCTS and its local copy of the neural network. It collects `Trajectory` data, calculates N-step reward targets after the episode, and returns results via a `SelfPlayResult` object. It also logs stats and game state asynchronously.
-   **Types ([`types.py`](types.py)):**
    -   Defines Pydantic models like `SelfPlayResult` and TypedDicts like `SampledBatchPER` for structured data transfer and buffer sampling.

## Exposed Interfaces

-   **Core:**
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(batch_sample: SampledBatchPER | SampledBatch) -> Optional[Tuple[Dict[str, float], np.ndarray]]`: Takes PER or uniform sample, returns loss info and TD errors (for PER).
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(trajectory: Trajectory)`
        -   `add_batch(trajectories: List[Trajectory])`
        -   `sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[SampledBatchPER | SampledBatch]`: Samples batch, requires step for PER beta.
        -   `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray)`: Updates priorities for PER.
        -   `is_ready() -> bool`
        -   `__len__() -> int` (Returns total steps, not number of trajectories)
-   **Self-Play:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `run_episode() -> SelfPlayResult`
        -   `set_weights(weights: Dict)`
        -   `set_current_trainer_step(global_step: int)`
-   **Types:**
    -   `SelfPlayResult`: Pydantic model for self-play results.
    -   `SampledBatchPER`: TypedDict for PER samples.
    -   `SampledBatch`: Type alias for uniform samples.

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: `TrainConfig`, `EnvConfig`, `ModelConfig`, `MCTSConfig`.
-   **[`muzerotriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`muzerotriangle.features`](../features/README.md)**: `extract_state_features`.
-   **[`muzerotriangle.mcts`](../mcts/README.md)**: Core MCTS components.
-   **[`muzerotriangle.environment`](../environment/README.md)**: `GameState`.
-   **[`muzerotriangle.stats`](../stats/README.md)**: `StatsCollectorActor` (used indirectly via `muzerotriangle.training`).
-   **[`muzerotriangle.utils`](../utils/README.md)**: Types (`Trajectory`, `TrajectoryStep`, `SampledBatchPER`, `SampledBatch`, `StepInfo`) and helpers (`SumTree`).
-   **[`muzerotriangle.structs`](../structs/README.md)**: Implicitly used via `GameState`.
-   **`torch`**: Used by `Trainer` and `NeuralNetwork`.
-   **`ray`**: Used by `SelfPlayWorker`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `numpy`, `random`, `time`.

---

**Note:** Keep this README updated when changing the responsibilities of the Trainer, Buffer, or SelfPlayWorker, especially regarding N-step returns and PER.

File: muzerotriangle\rl\types.py
# File: muzerotriangle/rl/types.py
import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..utils.types import Trajectory

logger = logging.getLogger(__name__)
arbitrary_types_config = ConfigDict(arbitrary_types_allowed=True)


class SelfPlayResult(BaseModel):
    """Pydantic model for structuring results from a self-play worker (MuZero)."""

    model_config = arbitrary_types_config
    trajectory: Trajectory
    final_score: float
    episode_steps: int
    total_simulations: int = Field(..., ge=0)
    avg_root_visits: float = Field(..., ge=0)
    avg_tree_depth: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_trajectory_structure(self) -> "SelfPlayResult":
        """Basic structural validation for trajectory steps."""
        invalid_count = 0
        valid_steps = []
        for i, step in enumerate(self.trajectory):
            is_valid = False
            try:
                # --- Apply SIM102: Combine nested ifs ---
                if (
                    isinstance(step, dict)
                    and all(
                        k in step
                        for k in [
                            "observation",
                            "action",
                            "reward",
                            "policy_target",
                            "value_target",
                        ]
                    )
                    and isinstance(step["observation"], dict)
                    and "grid" in step["observation"]
                    and "other_features" in step["observation"]
                    and isinstance(step["observation"]["grid"], np.ndarray)
                    and isinstance(step["observation"]["other_features"], np.ndarray)
                    and np.all(np.isfinite(step["observation"]["grid"]))
                    and np.all(np.isfinite(step["observation"]["other_features"]))
                    and isinstance(step["action"], int)
                    and isinstance(step["reward"], float | int)
                    and isinstance(step["policy_target"], dict)
                    and isinstance(step["value_target"], float | int)
                ):
                    is_valid = True
                # --- End Combined If ---
            except Exception as e:
                logger.warning(f"Error validating step {i}: {e}")
            if is_valid:
                valid_steps.append(step)
            else:
                invalid_count += 1
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid steps. Keeping valid.")
            object.__setattr__(self, "trajectory", valid_steps)
        return self


SelfPlayResult.model_rebuild(force=True)


File: muzerotriangle\rl\__init__.py
"""
Reinforcement Learning (RL) module.
Contains the core components for training an agent using self-play and MCTS.
"""

from .core.buffer import ExperienceBuffer
from .core.trainer import Trainer
from .self_play.worker import SelfPlayWorker
from .types import SelfPlayResult

__all__ = [
    # Core components used by the training pipeline
    "Trainer",
    "ExperienceBuffer",
    # Self-play components
    "SelfPlayWorker",
    "SelfPlayResult",
]


File: muzerotriangle\rl\core\buffer.py
# File: muzerotriangle/rl/core/buffer.py
import logging
import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt  # Import numpy typing

from ...utils.sumtree import SumTree
from ...utils.types import (
    SampledBatch,
    SampledBatchPER,  # Import PER batch type
    Trajectory,
)

if TYPE_CHECKING:
    from ...config import TrainConfig

    pass

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience Replay Buffer for MuZero. Stores complete game trajectories.
    Samples sequences of fixed length for training.
    Supports Prioritized Experience Replay (PER).
    """

    def __init__(self, config: "TrainConfig"):
        self.config = config
        self.capacity = (
            config.BUFFER_CAPACITY
        )  # Capacity in terms of total steps/transitions
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN  # Min total steps
        self.unroll_steps = config.MUZERO_UNROLL_STEPS
        self.sequence_length = self.unroll_steps + 1  # K unroll steps + 1 initial state

        # --- Data Storage ---
        # Stores tuples of (unique_buffer_index, trajectory)
        self.buffer: deque[tuple[int, Trajectory]] = deque()
        self.tree_idx_to_buffer_idx: dict[
            int, int
        ] = {}  # Maps SumTree leaf index to unique buffer_idx
        self.buffer_idx_to_tree_idx: dict[
            int, int
        ] = {}  # Maps unique buffer_idx to SumTree leaf index
        self.next_buffer_idx = 0  # Monotonically increasing index for unique ID
        self.total_steps = 0

        # --- PER Attributes ---
        self.use_per = config.USE_PER
        self.sum_tree: SumTree | None = None  # Initialize as None
        if self.use_per:
            # Estimate SumTree capacity based on trajectories, not steps
            # A better estimate might be needed, but this is a starting point
            estimated_avg_traj_len = 50  # Heuristic, adjust as needed
            estimated_num_trajectories = max(
                1, config.BUFFER_CAPACITY // estimated_avg_traj_len
            )
            # Give SumTree more capacity than just estimated trajectories
            sumtree_capacity = int(estimated_num_trajectories * 1.5)
            # Ensure it's large enough for batch size and some minimum
            sumtree_capacity = max(
                sumtree_capacity, config.BATCH_SIZE * 10
            )  # Increased buffer
            sumtree_capacity = max(sumtree_capacity, 1000)  # Absolute minimum
            self.sum_tree = SumTree(sumtree_capacity)
            self.per_alpha = config.PER_ALPHA
            self.per_beta_initial = config.PER_BETA_INITIAL
            self.per_beta_final = config.PER_BETA_FINAL
            self.per_beta_anneal_steps = config.PER_BETA_ANNEAL_STEPS
            self.per_epsilon = config.PER_EPSILON
            logger.info(
                f"MuZero Experience buffer initialized with PER. "
                f"Capacity (total steps): {self.capacity}, Sequence Length: {self.sequence_length}, "
                f"SumTree Capacity (trajectories): {sumtree_capacity}"
            )
        else:
            logger.info(
                f"MuZero Experience buffer initialized with uniform sampling. "
                f"Capacity (total steps): {self.capacity}, Sequence Length: {self.sequence_length}"
            )

    def _get_priority(self, error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates priority from TD error array using PER parameters."""
        if not self.use_per:
            # Should not be called if PER is disabled, but return default if it is
            return np.ones_like(error)
        # Use np.abs for potential array input
        error_abs = np.abs(error)
        return (error_abs + self.per_epsilon) ** self.per_alpha

    def add(self, trajectory: Trajectory):
        """Adds a complete trajectory to the buffer and SumTree (if PER enabled)."""
        if not trajectory:
            logger.warning("Attempted to add an empty trajectory.")
            return

        traj_len = len(trajectory)
        if traj_len < self.sequence_length:
            logger.debug(
                f"Skipping short trajectory (length {traj_len} < {self.sequence_length})"
            )
            return  # Don't add trajectories shorter than the required sequence length

        buffer_idx = self.next_buffer_idx  # Use unique ID for this trajectory
        self.next_buffer_idx += 1

        # --- Eviction Logic ---
        while self.total_steps + traj_len > self.capacity and len(self.buffer) > 0:
            removed_buffer_idx, removed_traj = self.buffer.popleft()
            self.total_steps -= len(removed_traj)
            if self.use_per and self.sum_tree:
                removed_tree_idx = self.buffer_idx_to_tree_idx.pop(
                    removed_buffer_idx, None
                )
                if removed_tree_idx is not None:
                    # Update priority to 0 in the tree, effectively removing it from sampling
                    # We don't actually remove the node, just nullify its priority
                    self.sum_tree.update(removed_tree_idx, 0.0)
                    # Remove the mapping from tree_idx back to buffer_idx
                    if removed_tree_idx in self.tree_idx_to_buffer_idx:
                        del self.tree_idx_to_buffer_idx[removed_tree_idx]
                else:
                    # This might happen if buffer_idx wasn't added correctly or already evicted
                    logger.warning(
                        f"Could not find tree index for evicted buffer index {removed_buffer_idx}"
                    )
            logger.debug(
                f"Buffer capacity reached. Removed oldest trajectory (BufferIdx: {removed_buffer_idx}, Len: {len(removed_traj)}). Current total steps: {self.total_steps}"
            )
        # ---

        # Add new trajectory
        self.buffer.append((buffer_idx, trajectory))
        self.total_steps += traj_len

        if self.use_per and self.sum_tree:
            # Calculate initial priority (e.g., max priority or based on initial TD error if available)
            # Using max priority ensures new samples are likely to be picked soon
            priority = self.sum_tree.max_priority  # Use the property here
            if (
                priority == 0
            ):  # Handle the case where the tree is empty or only has 0-priority items
                priority = 1.0  # Assign a default high priority for new samples

            # Add the buffer_idx to the SumTree with the calculated priority
            tree_idx = self.sum_tree.add(priority, buffer_idx)
            if tree_idx is not None:
                # Store the mapping between tree_idx and buffer_idx
                self.tree_idx_to_buffer_idx[tree_idx] = buffer_idx
                self.buffer_idx_to_tree_idx[buffer_idx] = tree_idx
            else:
                logger.error(
                    f"SumTree add returned None index for buffer_idx {buffer_idx}. PER might be inconsistent."
                )

        logger.debug(
            f"Added trajectory (BufferIdx: {buffer_idx}, Len: {traj_len}). Buffer trajectories: {len(self.buffer)}, Total steps: {self.total_steps}"
        )

    def add_batch(self, trajectories: list[Trajectory]):
        """Adds a batch of trajectories."""
        for traj in trajectories:
            self.add(traj)

    def _anneal_beta(self, current_train_step: int) -> float:
        """Linearly anneals PER beta."""
        if (
            not self.use_per
            or self.per_beta_anneal_steps is None
            or self.per_beta_anneal_steps <= 0
            or self.config.MAX_TRAINING_STEPS  # Use self.config here
            is None  # Avoid division by zero if MAX_TRAINING_STEPS is None
        ):
            return self.per_beta_initial

        # Ensure anneal_steps doesn't exceed total steps
        anneal_steps = min(
            self.per_beta_anneal_steps,
            self.config.MAX_TRAINING_STEPS,  # Use self.config
        )
        fraction = min(1.0, current_train_step / anneal_steps)
        beta = self.per_beta_initial + fraction * (
            self.per_beta_final - self.per_beta_initial
        )
        return beta

    def sample(
        self,
        batch_size: int,
        current_train_step: int | None = None,
    ) -> SampledBatchPER | SampledBatch | None:
        """
        Samples a batch of sequences. Uses PER if enabled, otherwise uniform.
        Returns SampledBatchPER if PER is used, SampledBatch otherwise, or None if not ready or sampling fails.
        """
        if not self.is_ready():
            logger.debug(
                f"Buffer not ready for sampling. Steps: {self.total_steps}/{self.min_size_to_train}"
            )
            return None

        if self.use_per:
            return self._sample_per(batch_size, current_train_step)
        else:
            return self._sample_uniform(batch_size)

    def _sample_uniform(self, batch_size: int) -> SampledBatch | None:
        """Uniformly samples sequences."""
        sampled_sequences: SampledBatch = []
        attempts = 0
        max_attempts = batch_size * 20  # Increased attempts

        if len(self.buffer) == 0:
            logger.warning("Uniform sample called on empty buffer.")
            return None

        # Create a list of (deque_index, trajectory_length) for trajectories long enough
        eligible_trajectories = [
            (idx, len(traj))
            for idx, (_, traj) in enumerate(self.buffer)
            if len(traj) >= self.sequence_length
        ]

        if not eligible_trajectories:
            logger.warning(
                f"No trajectories long enough ({self.sequence_length}) for uniform sampling."
            )
            return None

        while len(sampled_sequences) < batch_size and attempts < max_attempts:
            attempts += 1
            # Sample a trajectory index uniformly from eligible ones
            traj_deque_idx, traj_len = random.choice(eligible_trajectories)
            _, trajectory = self.buffer[traj_deque_idx]  # Access deque by index

            # Sample a valid start index for the sequence
            start_index = random.randrange(traj_len - self.sequence_length + 1)
            sequence = trajectory[start_index : start_index + self.sequence_length]

            if len(sequence) == self.sequence_length:
                sampled_sequences.append(sequence)
            else:
                # This should not happen if start_index logic is correct
                logger.error(
                    f"Uniform Sample: Sequence incorrect length {len(sequence)} (expected {self.sequence_length}). Traj len: {traj_len}, Start: {start_index}"
                )

        if len(sampled_sequences) < batch_size:
            logger.warning(
                f"Uniform Sample: Could only sample {len(sampled_sequences)} sequences after {attempts} attempts."
            )

        return sampled_sequences if sampled_sequences else None

    def _sample_per(
        self, batch_size: int, current_train_step: int | None
    ) -> SampledBatchPER | None:
        """Samples sequences using Prioritized Experience Replay."""
        if (
            self.sum_tree is None
            or self.sum_tree.n_entries == 0
            or self.sum_tree.total() <= 0  # Use total() method
        ):
            logger.warning(
                f"PER sample called but SumTree empty or total priority zero. "
                f"n_entries: {self.sum_tree.n_entries if self.sum_tree else 'None'}, "
                f"total_priority: {self.sum_tree.total() if self.sum_tree else 'None'}. Cannot sample."
            )
            return None

        if current_train_step is None:
            logger.warning(
                "PER sample requires current_train_step for beta annealing. Using initial beta."
            )
            beta = self.per_beta_initial
        else:
            beta = self._anneal_beta(current_train_step)

        sampled_sequences_list: SampledBatch = []
        # Initialize as lists to append easily
        tree_indices_list: list[int] = []
        priorities_list: list[float] = []
        buffer_indices_sampled_list: list[int] = []  # Store the unique buffer_idx

        segment = self.sum_tree.total() / batch_size  # Use total() method
        attempts = 0
        max_attempts_per_sample = (
            20  # Limit attempts per sample to avoid infinite loops
        )
        sampled_count = 0

        # Create a temporary mapping from buffer_idx to deque index for quick lookup
        # This avoids iterating the deque repeatedly inside the loop
        buffer_idx_to_deque_idx = {
            buf_idx: i for i, (buf_idx, _) in enumerate(self.buffer)
        }

        while (
            sampled_count < batch_size
            and attempts < max_attempts_per_sample * batch_size
        ):
            attempts += 1
            a = segment * sampled_count
            b = segment * (sampled_count + 1)
            # Ensure b doesn't exceed total priority due to floating point issues
            b = min(b, self.sum_tree.total())  # Use total() method
            # Ensure a < b even with floating point issues
            if a >= b:
                if (
                    self.sum_tree.total() > 1e-9
                ):  # Avoid division by zero if total_priority is tiny
                    a = b - (self.sum_tree.total() * 1e-6)  # Sample very close to b
                else:
                    a = 0.0  # Sample from the beginning if total priority is ~0

            value = random.uniform(a, b)

            try:
                # get_leaf returns (tree_idx, priority, buffer_idx)
                tree_idx, priority, buffer_idx = self.sum_tree.get_leaf(value)
            except (IndexError, ValueError, RuntimeError) as e:
                logger.warning(
                    f"PER sample: SumTree get_leaf failed for value {value}. Error: {e}. Retrying."
                )
                continue

            if not isinstance(buffer_idx, int):
                logger.warning(
                    f"PER sample: SumTree returned invalid buffer_idx {buffer_idx} (type: {type(buffer_idx)}). Retrying."
                )
                continue

            # Check if we already sampled this trajectory in this batch (optional, but can improve diversity)
            # if buffer_idx in buffer_indices_sampled_list:
            #     continue

            # Find the trajectory in the deque using the buffer_idx
            deque_idx = buffer_idx_to_deque_idx.get(buffer_idx)
            if deque_idx is None:
                logger.error(
                    f"PER sample: Trajectory for buffer_idx {buffer_idx} (TreeIdx: {tree_idx}) not found in deque map. SumTree/Deque inconsistent! Setting priority to 0."
                )
                # Attempt to recover by removing the bad index from the tree
                self.sum_tree.update(tree_idx, 0.0)
                if tree_idx in self.tree_idx_to_buffer_idx:
                    del self.tree_idx_to_buffer_idx[tree_idx]
                if buffer_idx in self.buffer_idx_to_tree_idx:
                    del self.buffer_idx_to_tree_idx[buffer_idx]
                continue

            _, trajectory = self.buffer[deque_idx]

            if len(trajectory) < self.sequence_length:
                logger.debug(
                    f"PER Sample: Trajectory {buffer_idx} too short ({len(trajectory)} < {self.sequence_length}). Skipping."
                )
                # Optionally reduce priority of short trajectories?
                # self.sum_tree.update(tree_idx, self.per_epsilon ** self.per_alpha)
                continue

            # Sample a valid start index for the sequence
            start_index = random.randrange(len(trajectory) - self.sequence_length + 1)
            sequence = trajectory[start_index : start_index + self.sequence_length]

            if len(sequence) == self.sequence_length:
                sampled_sequences_list.append(sequence)
                tree_indices_list.append(tree_idx)
                priorities_list.append(priority)
                buffer_indices_sampled_list.append(buffer_idx)  # Store the unique ID
                sampled_count += 1
            else:
                logger.error(
                    f"PER Sample: Sequence incorrect length {len(sequence)} (expected {self.sequence_length}). BufferIdx: {buffer_idx}, TreeIdx: {tree_idx}"
                )

        if sampled_count == 0:
            logger.warning(
                f"PER Sample: Could not sample any valid sequences after {attempts} attempts."
            )
            return None
        if sampled_count < batch_size:
            logger.warning(
                f"PER Sample: Could only sample {sampled_count} sequences out of {batch_size} requested after {attempts} attempts."
            )
            # Trim lists to the actual number sampled
            sampled_sequences_list = sampled_sequences_list[:sampled_count]
            tree_indices_list = tree_indices_list[:sampled_count]
            priorities_list = priorities_list[:sampled_count]
            # buffer_indices_sampled_list = buffer_indices_sampled_list[:sampled_count] # Not used later, but good practice

        # Convert lists to numpy arrays
        # --- FIXED: Use typed arrays directly ---
        tree_indices: npt.NDArray[np.int32] = np.array(
            tree_indices_list, dtype=np.int32
        )
        priorities_np: npt.NDArray[np.float32] = np.array(
            priorities_list, dtype=np.float32
        )
        # --- END FIXED ---

        # Calculate IS weights
        sampling_probabilities = priorities_np / max(
            self.sum_tree.total(),
            1e-9,  # Use total() method
        )  # Avoid division by zero
        weights = np.power(
            max(self.sum_tree.n_entries, 1) * sampling_probabilities + 1e-9, -beta
        )
        max_weight = np.max(weights) if len(weights) > 0 else 1.0
        weights /= max(max_weight, 1e-9)  # Avoid division by zero

        return SampledBatchPER(
            sequences=sampled_sequences_list,
            indices=tree_indices,  # Use typed array
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Updates priorities of sampled experiences based on TD errors."""
        if not self.use_per or self.sum_tree is None:
            return
        if len(tree_indices) != len(td_errors):
            logger.error(
                f"PER update failed: Mismatch length indices ({len(tree_indices)}) vs errors ({len(td_errors)})"
            )
            return

        # Ensure td_errors is a numpy array for vectorized operations
        td_errors_np = np.asarray(td_errors)
        # Calculate priorities using the internal method
        priorities_array: npt.NDArray[np.float64] = self._get_priority(td_errors_np)

        if len(priorities_array) != len(tree_indices):
            logger.error(
                f"PER update failed: Mismatch length indices ({len(tree_indices)}) vs calculated priorities ({len(priorities_array)})"
            )
            return

        for i in range(len(tree_indices)):
            idx = int(tree_indices[i])  # Cast numpy int to python int
            p = float(priorities_array[i])  # Cast numpy float to python float
            # Check index validity before updating
            if not (0 <= idx < len(self.sum_tree.tree)):
                logger.warning(
                    f"PER update: Invalid tree index {idx} provided. Skipping update for this index."
                )
                continue
            try:
                self.sum_tree.update(idx, p)
            except IndexError:
                # This might happen if the tree structure is somehow corrupted
                logger.error(
                    f"PER update: Error updating tree index {idx} with priority {p}. Skipping.",
                    exc_info=True,
                )

    def __len__(self) -> int:
        """Returns the total number of steps (transitions) stored in the buffer."""
        return self.total_steps

    def is_ready(self) -> bool:
        """Checks if the buffer has enough total steps and trajectories to start training."""
        sufficient_steps = self.total_steps >= self.min_size_to_train
        sufficient_trajectories = True
        if self.use_per:
            if self.sum_tree is None:
                sufficient_trajectories = False  # Cannot sample if tree is missing
            else:
                # Need enough trajectories in the tree to form at least one batch
                sufficient_trajectories = (
                    self.sum_tree.n_entries >= self.config.BATCH_SIZE
                )
        return sufficient_steps and sufficient_trajectories


File: muzerotriangle\rl\core\README.md
# File: muzerotriangle/rl/core/README.md
# RL Core Submodule (`muzerotriangle.rl.core`)

## Purpose and Architecture

This submodule contains core classes directly involved in the MuZero reinforcement learning update process and data storage.

-   **[`Trainer`](trainer.py):** This class encapsulates the logic for updating the neural network's weights.
    -   It holds the main `NeuralNetwork` interface, optimizer, and scheduler.
    -   Its `train_step` method takes a batch of sequences (potentially with PER indices and weights), performs forward/backward passes through the unrolled model, calculates losses (policy cross-entropy, distributional value cross-entropy, **distributional N-step reward cross-entropy**), applies importance sampling weights if using PER, updates weights, and returns calculated TD errors (based on initial value prediction) for PER priority updates.
-   **[`ExperienceBuffer`](buffer.py):** This class implements a replay buffer storing complete game `Trajectory` objects. It supports Prioritized Experience Replay (PER) via a SumTree, including prioritized sampling and priority updates, based on configuration. **It calculates priorities from TD errors internally using PER parameters (`alpha`, `epsilon`)** before storing/updating them in the SumTree. It samples fixed-length sequences for training. Its `is_ready` method correctly checks both total steps and the number of available trajectories in the SumTree when PER is enabled. **The underlying `SumTree` now correctly tracks the number of entries.**

## Exposed Interfaces

-   **Classes:**
    -   `Trainer`:
        -   `__init__(nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig)`
        -   `train_step(batch_sample: SampledBatchPER | SampledBatch) -> Optional[Tuple[Dict[str, float], np.ndarray]]`: Takes PER or uniform sample, returns loss info and TD errors (for PER).
        -   `load_optimizer_state(state_dict: dict)`
        -   `get_current_lr() -> float`
    -   `ExperienceBuffer`:
        -   `__init__(config: TrainConfig)`
        -   `add(trajectory: Trajectory)`
        -   `add_batch(trajectories: List[Trajectory])`
        -   `sample(batch_size: int, current_train_step: Optional[int] = None) -> Optional[SampledBatchPER | SampledBatch]`: Samples batch, requires step for PER beta.
        -   `update_priorities(tree_indices: np.ndarray, td_errors: np.ndarray)`: Updates priorities for PER.
        -   `is_ready() -> bool`
        -   `__len__() -> int`

## Dependencies

-   **[`muzerotriangle.config`](../../config/README.md)**: `TrainConfig`, `EnvConfig`, `ModelConfig`.
-   **[`muzerotriangle.nn`](../../nn/README.md)**: `NeuralNetwork`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**: Types (`Trajectory`, `TrajectoryStep`, `SampledBatchPER`, `StateType`, etc.) and helpers (`SumTree`).
-   **`torch`**: Used heavily by `Trainer`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `numpy`, `random`.

---

**Note:** Please keep this README updated when changing the responsibilities or interfaces of the Trainer or Buffer, especially regarding PER and N-step return handling.

File: muzerotriangle\rl\core\trainer.py
# File: muzerotriangle/rl/core/trainer.py
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ...utils.types import (
    SampledBatch,
    SampledBatchPER,  # Import PER batch type
)

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler

    from ...config import EnvConfig, TrainConfig
    from ...nn import NeuralNetwork

logger = logging.getLogger(__name__)


class Trainer:
    """MuZero Trainer."""

    def __init__(
        self,
        nn_interface: "NeuralNetwork",
        train_config: "TrainConfig",
        env_config: "EnvConfig",
    ):
        self.nn = nn_interface
        self.model = nn_interface.model
        self.train_config = train_config
        self.env_config = env_config
        self.model_config = nn_interface.model_config
        self.device = nn_interface.device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)
        self.num_value_atoms = self.nn.num_value_atoms
        self.value_support = self.nn.support.to(self.device)
        self.num_reward_atoms = self.nn.num_reward_atoms
        self.reward_support = self.nn.reward_support.to(self.device)
        self.unroll_steps = self.train_config.MUZERO_UNROLL_STEPS

    def _create_optimizer(self):
        lr = self.train_config.LEARNING_RATE
        wd = self.train_config.WEIGHT_DECAY
        params = self.model.parameters()
        opt_type = self.train_config.OPTIMIZER_TYPE.lower()
        logger.info(f"Creating optimizer: {opt_type}, LR: {lr}, WD: {wd}")
        if opt_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.train_config.OPTIMIZER_TYPE}"
            )

    def _create_scheduler(self, optimizer):
        scheduler_type_config = self.train_config.LR_SCHEDULER_TYPE
        scheduler_type = None
        if scheduler_type_config:
            scheduler_type = scheduler_type_config.lower()
        if not scheduler_type or scheduler_type == "none":
            return None
        logger.info(f"Creating LR scheduler: {scheduler_type}")
        if scheduler_type == "steplr":
            step_size = getattr(self.train_config, "LR_SCHEDULER_STEP_SIZE", 100000)
            gamma = getattr(self.train_config, "LR_SCHEDULER_GAMMA", 0.1)
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            )
        elif scheduler_type == "cosineannealinglr":
            t_max = self.train_config.LR_SCHEDULER_T_MAX
            eta_min = self.train_config.LR_SCHEDULER_ETA_MIN
            if t_max is None:
                t_max = self.train_config.MAX_TRAINING_STEPS or 1_000_000
            return cast(
                "_LRScheduler",
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                ),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type_config}")

    def _prepare_batch(self, batch_sequences: SampledBatch) -> dict[str, torch.Tensor]:
        """Prepares batch tensors from sampled sequences."""
        batch_size = len(batch_sequences)
        seq_len = self.unroll_steps + 1
        action_dim = int(self.env_config.ACTION_DIM)
        batch_grids = torch.zeros(
            (
                batch_size,
                seq_len,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        batch_others = torch.zeros(
            (batch_size, seq_len, self.model_config.OTHER_NN_INPUT_FEATURES_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        batch_actions = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        batch_n_step_rewards = torch.zeros(
            (batch_size, seq_len), dtype=torch.float32, device=self.device
        )
        batch_policy_targets = torch.zeros(
            (batch_size, seq_len, action_dim), dtype=torch.float32, device=self.device
        )
        batch_value_targets = torch.zeros(
            (batch_size, seq_len), dtype=torch.float32, device=self.device
        )
        for b_idx, sequence in enumerate(batch_sequences):
            if len(sequence) != seq_len:
                raise ValueError(f"Sequence {b_idx} len {len(sequence)} != {seq_len}")
            for s_idx, step_data in enumerate(sequence):
                obs = step_data["observation"]
                batch_grids[b_idx, s_idx] = torch.from_numpy(obs["grid"])
                batch_others[b_idx, s_idx] = torch.from_numpy(obs["other_features"])
                batch_actions[b_idx, s_idx] = step_data["action"]
                batch_n_step_rewards[b_idx, s_idx] = step_data["n_step_reward_target"]
                policy_map = step_data["policy_target"]
                for action, prob in policy_map.items():
                    if 0 <= action < action_dim:
                        batch_policy_targets[b_idx, s_idx, action] = prob
                policy_sum = batch_policy_targets[b_idx, s_idx].sum()
                if abs(policy_sum - 1.0) > 1e-5 and policy_sum > 1e-9:
                    batch_policy_targets[b_idx, s_idx] /= policy_sum
                elif policy_sum <= 1e-9 and action_dim > 0:
                    batch_policy_targets[b_idx, s_idx].fill_(1.0 / action_dim)
                batch_value_targets[b_idx, s_idx] = step_data["value_target"]
        return {
            "grids": batch_grids,
            "others": batch_others,
            "actions": batch_actions,
            "n_step_rewards": batch_n_step_rewards,
            "policy_targets": batch_policy_targets,
            "value_targets": batch_value_targets,
        }

    def _calculate_target_distribution(
        self, target_scalars: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Projects scalar targets onto the fixed support atoms (z or r)."""
        target_shape = target_scalars.shape
        num_atoms = support.size(0)
        v_min = support[0]
        v_max = support[-1]
        delta = (v_max - v_min) / (num_atoms - 1) if num_atoms > 1 else 0.0

        target_scalars_flat = target_scalars.flatten()
        target_scalars_flat = target_scalars_flat.clamp(v_min, v_max)
        b: torch.Tensor = (
            (target_scalars_flat - v_min) / delta
            if delta > 0
            else torch.zeros_like(target_scalars_flat)
        )
        lower_idx: torch.Tensor = b.floor().long()
        upper_idx: torch.Tensor = b.ceil().long()

        lower_idx = torch.max(
            torch.tensor(0, device=self.device, dtype=torch.long), lower_idx
        )
        upper_idx = torch.min(
            torch.tensor(num_atoms - 1, device=self.device, dtype=torch.long), upper_idx
        )
        lower_eq_upper = lower_idx == upper_idx
        lower_idx[lower_eq_upper & (lower_idx > 0)] -= 1
        upper_idx[lower_eq_upper & (upper_idx < num_atoms - 1)] += 1

        m_lower: torch.Tensor = (upper_idx.float() - b).clamp(min=0.0, max=1.0)
        m_upper: torch.Tensor = (b - lower_idx.float()).clamp(min=0.0, max=1.0)

        m = torch.zeros(target_scalars_flat.size(0), num_atoms, device=self.device)
        # Create index tensor explicitly
        batch_indices = torch.arange(
            target_scalars_flat.size(0), device=self.device, dtype=torch.long
        )

        # Use index_put_ for sparse updates (more robust than index_add_)
        m.index_put_((batch_indices, lower_idx), m_lower, accumulate=True)
        m.index_put_((batch_indices, upper_idx), m_upper, accumulate=True)

        m = m.view(*target_shape, num_atoms)
        return m

    def _calculate_loss(
        self,
        policy_logits,
        value_logits,
        reward_logits,
        target_data,
        is_weights,  # Add IS weights
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates MuZero losses, applying IS weights."""
        pi_target = target_data["policy_targets"]
        z_target = target_data["value_targets"]
        r_target_n_step = target_data["n_step_rewards"]
        batch_size, seq_len, action_dim = pi_target.shape

        # --- Expand IS weights ---
        # is_weights has shape [batch_size, 1]
        # Expand to match the shape of per-step losses [batch_size, seq_len]
        is_weights_expanded = is_weights.expand(-1, seq_len).reshape(-1)
        # For reward loss, we only need weights for steps 1 to K (unroll_steps)
        is_weights_reward = is_weights.expand(-1, self.unroll_steps).reshape(-1)
        # --- END Expand IS weights ---

        # --- Policy Loss ---
        policy_logits_flat = policy_logits.view(-1, action_dim)
        pi_target_flat = pi_target.view(-1, action_dim)
        log_pred_p = F.log_softmax(policy_logits_flat, dim=1)
        policy_loss_per_sample = -torch.sum(pi_target_flat * log_pred_p, dim=1)
        policy_loss = (is_weights_expanded * policy_loss_per_sample).mean()

        # --- Value Loss ---
        value_target_dist = self._calculate_target_distribution(
            z_target, self.value_support
        )
        value_logits_flat = value_logits.view(-1, self.num_value_atoms)
        value_target_dist_flat = value_target_dist.view(-1, self.num_value_atoms)
        log_pred_v = F.log_softmax(value_logits_flat, dim=1)
        value_loss_per_sample = -torch.sum(value_target_dist_flat * log_pred_v, dim=1)
        value_loss = (is_weights_expanded * value_loss_per_sample).mean()

        # --- Reward Loss ---
        # Target rewards are for steps k=1 to K (n_step_reward_target[t] is target for r_{t+1})
        r_target_k = r_target_n_step[:, 1 : self.unroll_steps + 1]
        reward_target_dist = self._calculate_target_distribution(
            r_target_k, self.reward_support
        )
        # reward_logits are for steps k=1 to K (output of dynamics for action a_k)
        reward_logits_flat = reward_logits.reshape(-1, self.num_reward_atoms)
        reward_target_dist_flat = reward_target_dist.reshape(-1, self.num_reward_atoms)
        log_pred_r = F.log_softmax(reward_logits_flat, dim=1)
        reward_loss_per_sample = -torch.sum(reward_target_dist_flat * log_pred_r, dim=1)
        reward_loss = (is_weights_reward * reward_loss_per_sample).mean()

        # --- Total Loss ---
        total_loss = (
            self.train_config.POLICY_LOSS_WEIGHT * policy_loss
            + self.train_config.VALUE_LOSS_WEIGHT * value_loss
            + self.train_config.REWARD_LOSS_WEIGHT * reward_loss
        )

        # --- Calculate TD Errors for PER Update ---
        # Use the value prediction for the initial state (k=0) vs its target
        value_pred_scalar_0 = self.nn._logits_to_scalar(
            value_logits[:, 0, :], self.value_support
        )
        value_target_scalar_0 = z_target[:, 0]
        td_errors_tensor = (value_target_scalar_0 - value_pred_scalar_0).detach()

        return total_loss, policy_loss, value_loss, reward_loss, td_errors_tensor

    def train_step(
        self, batch_sample: SampledBatchPER | SampledBatch
    ) -> tuple[dict[str, float], np.ndarray] | None:
        """Performs one training step, handling PER if enabled."""
        if not batch_sample:
            return None

        self.model.train()

        # --- Unpack batch sample ---
        if isinstance(batch_sample, dict) and "sequences" in batch_sample:
            batch_sequences = batch_sample["sequences"]
            is_weights_np = batch_sample["weights"]
            is_weights = torch.from_numpy(is_weights_np).to(self.device).unsqueeze(-1)
        else:
            batch_sequences = batch_sample
            batch_size = len(batch_sequences)
            is_weights = torch.ones((batch_size, 1), device=self.device)

        if not batch_sequences:
            return None

        try:
            target_data = self._prepare_batch(batch_sequences)
        except Exception as e:
            logger.error(f"Error preparing batch: {e}", exc_info=True)
            return None

        self.optimizer.zero_grad()
        obs_grid_0 = target_data["grids"][:, 0].contiguous()
        obs_other_0 = target_data["others"][:, 0].contiguous()

        # Initial inference (h + f)
        policy_logits_0, value_logits_0, initial_hidden_state = self.model(
            obs_grid_0, obs_other_0
        )

        policy_logits_list = [policy_logits_0]
        value_logits_list = [value_logits_0]
        reward_logits_list = []
        hidden_state = initial_hidden_state

        # Unroll dynamics and prediction (g + f)
        for k in range(self.unroll_steps):
            action_k = target_data["actions"][:, k + 1]  # Action a_{k+1}
            hidden_state, reward_logits_k_plus_1 = self.model.dynamics(
                hidden_state, action_k
            )
            policy_logits_k_plus_1, value_logits_k_plus_1 = self.model.predict(
                hidden_state
            )

            policy_logits_list.append(policy_logits_k_plus_1)
            value_logits_list.append(value_logits_k_plus_1)
            reward_logits_list.append(reward_logits_k_plus_1)

        # Stack predictions
        policy_logits_all = torch.stack(policy_logits_list, dim=1)
        value_logits_all = torch.stack(value_logits_list, dim=1)
        reward_logits_k = torch.stack(reward_logits_list, dim=1)

        # Calculate loss
        total_loss, policy_loss, value_loss, reward_loss, td_errors_tensor = (
            self._calculate_loss(
                policy_logits_all,
                value_logits_all,
                reward_logits_k,
                target_data,
                is_weights,
            )
        )

        # Backpropagate
        total_loss.backward()
        if self.train_config.GRADIENT_CLIP_VALUE is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_config.GRADIENT_CLIP_VALUE
            )
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # Calculate entropy (optional logging)
        with torch.no_grad():
            policy_probs = F.softmax(policy_logits_all, dim=-1)
            entropy = (
                -torch.sum(policy_probs * torch.log(policy_probs + 1e-9), dim=-1)
                .mean()
                .item()
            )

        loss_info = {
            "total_loss": float(total_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "entropy": float(entropy),
        }

        td_errors_np = td_errors_tensor.cpu().numpy()

        return loss_info, td_errors_np

    def get_current_lr(self) -> float:
        try:
            return float(self.optimizer.param_groups[0]["lr"])
        except Exception:
            logger.warning("Could not retrieve LR.")
            return 0.0

    def load_optimizer_state(self, state_dict: dict):
        try:
            self.optimizer.load_state_dict(state_dict)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            logger.info("Optimizer state loaded.")
        except Exception as e:
            logger.error(f"Failed load optimizer state: {e}", exc_info=True)


File: muzerotriangle\rl\core\visual_state_actor.py
# File: muzerotriangle/rl/core/visual_state_actor.py
import logging
import time
from typing import Any

import ray

from ...environment import GameState

logger = logging.getLogger(__name__)


@ray.remote
class VisualStateActor:
    """A simple Ray actor to hold the latest game states from workers for visualization."""

    def __init__(self) -> None:
        self.worker_states: dict[int, GameState] = {}
        self.global_stats: dict[str, Any] = {}
        self.last_update_times: dict[int, float] = {}

    def update_state(self, worker_id: int, game_state: GameState):
        """Workers call this to update their latest state."""
        self.worker_states[worker_id] = game_state
        self.last_update_times[worker_id] = time.time()

    def update_global_stats(self, stats: dict[str, Any]):
        """Orchestrator calls this to update global stats."""
        # Ensure stats is a dictionary
        if isinstance(stats, dict):
            # Use update to merge instead of direct assignment
            self.global_stats.update(stats)
        else:
            # Handle error or log warning if stats is not a dict
            logger.error(
                f"VisualStateActor received non-dict type for global stats: {type(stats)}"
            )
            # Don't reset, just ignore the update
            # self.global_stats = {}

    def get_all_states(self) -> dict[int, Any]:
        """
        Called by the orchestrator to get states for the visual queue.
        Key -1 holds the global_stats dictionary.
        Other keys hold GameState objects.
        """
        # Use dict() constructor instead of comprehension for ruff C416
        # Cast worker_states to dict[int, Any] before combining
        combined_states: dict[int, Any] = dict(self.worker_states)
        combined_states[-1] = self.global_stats.copy()
        return combined_states

    def get_state(self, worker_id: int) -> GameState | None:
        """Get state for a specific worker (unused currently)."""
        return self.worker_states.get(worker_id)


File: muzerotriangle\rl\core\__init__.py
"""
Core RL components: Trainer, Buffer.
The Orchestrator logic has been moved to the muzerotriangle.training module.
"""

from .buffer import ExperienceBuffer
from .trainer import Trainer

__all__ = [
    "Trainer",
    "ExperienceBuffer",
]


File: muzerotriangle\rl\self_play\README.md
# File: muzerotriangle/rl/self_play/README.md
# RL Self-Play Submodule (`muzerotriangle.rl.self_play`)

## Purpose and Architecture

This submodule focuses specifically on generating game episodes through self-play, driven by the current neural network and MCTS. It is designed to run in parallel using Ray actors managed by the [`muzerotriangle.training.worker_manager`](../../training/worker_manager.py).

-   **[`worker.py`](worker.py):** Defines the `SelfPlayWorker` class, decorated with `@ray.remote`.
    -   Each `SelfPlayWorker` actor runs independently, typically on a separate CPU core.
    -   It initializes its own `GameState` environment and `NeuralNetwork` instance (usually on the CPU).
    -   It receives configuration objects (`EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`) during initialization.
    -   It has a `set_weights` method allowing the `TrainingLoop` to periodically update its local neural network with the latest trained weights from the central model. It also has `set_current_trainer_step` to store the global step associated with the current weights.
    -   Its main method, `run_episode`, simulates a complete game episode:
        -   Uses its local `NeuralNetwork` evaluator and `MCTSConfig` to run MCTS ([`muzerotriangle.mcts.run_mcts_simulations`](../../mcts/core/search.py)).
        -   Selects actions based on MCTS results ([`muzerotriangle.mcts.strategy.policy.select_action_based_on_visits`](../../mcts/strategy/policy.py)).
        -   Generates policy targets ([`muzerotriangle.mcts.strategy.policy.get_policy_target`](../../mcts/strategy/policy.py)).
        -   Stores `TrajectoryStep` dictionaries containing `observation`, `action`, `reward`, `policy_target`, `value_target`, and optionally the `hidden_state`.
        -   Steps its local game environment (`GameState.step`).
        -   **After the episode concludes, it iterates backwards through the collected steps to calculate and store the N-step discounted reward target (`n_step_reward_target`) for each step.**
        -   Returns the completed `Trajectory` list, final score, episode length, and MCTS statistics via a `SelfPlayResult` object.
        -   Asynchronously logs per-step statistics and reports its current `GameState` to the `StatsCollectorActor`.

## Exposed Interfaces

-   **Classes:**
    -   `SelfPlayWorker`: Ray actor class.
        -   `__init__(...)`
        -   `run_episode() -> SelfPlayResult`: Runs one episode and returns results.
        -   `set_weights(weights: Dict)`: Updates the actor's local network weights.
        -   `set_current_trainer_step(global_step: int)`: Updates the stored trainer step.
-   **Types:**
    -   `SelfPlayResult`: Pydantic model defined in [`muzerotriangle.rl.types`](../types.py).

## Dependencies

-   **[`muzerotriangle.config`](../../config/README.md)**: `EnvConfig`, `MCTSConfig`, `ModelConfig`, `TrainConfig`.
-   **[`muzerotriangle.nn`](../../nn/README.md)**: `NeuralNetwork`.
-   **[`muzerotriangle.mcts`](../../mcts/README.md)**: Core MCTS functions and types.
-   **[`muzerotriangle.environment`](../../environment/README.md)**: `GameState`.
-   **[`muzerotriangle.features`](../../features/README.md)**: `extract_state_features`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**: `types` (including `Trajectory`, `TrajectoryStep`, `n_step_reward_target`).
-   **[`muzerotriangle.rl.types`](../types.py)**: `SelfPlayResult`.
-   **[`muzerotriangle.stats`](../../stats/README.md)**: `StatsCollectorActor`.
-   **`numpy`**: Used by MCTS strategies and for storing hidden states.
-   **`ray`**: The `@ray.remote` decorator makes this a Ray actor.
-   **`torch`**: Used by the local `NeuralNetwork` and for hidden states.
-   **Standard Libraries:** `typing`, `logging`, `random`, `time`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the self-play episode generation logic, the data collected (especially `TrajectoryStep` and N-step targets), or the asynchronous logging behavior.

File: muzerotriangle\rl\self_play\worker.py
# File: muzerotriangle/rl/self_play/worker.py
# File: muzerotriangle/rl/self_play/worker.py
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import ray

from ...environment import GameState
from ...features import extract_state_features
from ...mcts import (
    MCTSExecutionError,
    Node,
    get_policy_target,
    run_mcts_simulations,
    select_action_based_on_visits,
)
from ...nn import NeuralNetwork
from ...utils import get_device, set_random_seeds
from ..types import SelfPlayResult

if TYPE_CHECKING:
    # import torch # Already imported above

    import torch

    from ...utils.types import (
        ActionType,  # Import ActionType
        PolicyTargetMapping,  # Import PolicyTargetMapping
        StateType,  # Import StateType
        StepInfo,
        Trajectory,
        TrajectoryStep,
    )

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """MuZero self-play worker."""

    def __init__(
        self,
        actor_id,
        env_config,
        mcts_config,
        model_config,
        train_config,
        stats_collector_actor,
        initial_weights,
        seed,
        worker_device_str,
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.stats_collector_actor = stats_collector_actor
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str
        self.current_trainer_step = 0
        worker_log_level = logging.INFO
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        global logger
        logger = logging.getLogger(__name__)
        logging.getLogger("muzerotriangle.mcts").setLevel(logging.WARNING)
        logging.getLogger("muzerotriangle.nn").setLevel(logging.WARNING)
        set_random_seeds(self.seed)
        self.device = get_device(self.worker_device_str)
        self.nn_evaluator = NeuralNetwork(
            model_config, env_config, train_config, self.device
        )
        if initial_weights:
            self.set_weights(initial_weights)
        else:
            self.nn_evaluator.model.eval()
        logger.info(
            f"Worker {actor_id} initialized on device {self.device}. Seed: {self.seed}."
        )

    def set_weights(self, weights):
        try:
            self.nn_evaluator.set_weights(weights)
            logger.debug(f"W{self.actor_id}: Weights updated.")
        except Exception as e:
            logger.error(f"W{self.actor_id}: Failed set weights: {e}", exc_info=True)

    def set_current_trainer_step(self, global_step):
        self.current_trainer_step = global_step
        logger.debug(f"W{self.actor_id}: Trainer step set {global_step}")

    def _report_current_state(self, game_state):
        if self.stats_collector_actor:
            try:
                state_copy = game_state.copy()
                self.stats_collector_actor.update_worker_game_state.remote(
                    self.actor_id, state_copy
                )
            except Exception as e:
                logger.error(f"W{self.actor_id}: Failed report state: {e}")

    def _log_step_stats_async(self, game_step, mcts_visits, mcts_depth, step_reward):
        if self.stats_collector_actor:
            try:
                step_info: StepInfo = {
                    "game_step_index": game_step,
                    "global_step": self.current_trainer_step,
                }
                step_stats: dict[str, tuple[float, StepInfo]] = {
                    "MCTS/Step_Visits": (float(mcts_visits), step_info),
                    "MCTS/Step_Depth": (float(mcts_depth), step_info),
                    "RL/Step_Reward": (step_reward, step_info),
                }
                self.stats_collector_actor.log_batch.remote(step_stats)
            except Exception as e:
                logger.error(f"W{self.actor_id}: Failed log step stats: {e}")

    def _calculate_n_step_targets(self, trajectory_raw: list[dict]):
        """Calculates N-step reward targets and returns a completed Trajectory."""
        n_steps = self.train_config.N_STEP_RETURNS
        discount = self.train_config.DISCOUNT
        traj_len = len(trajectory_raw)
        completed_trajectory: Trajectory = []

        for t in range(traj_len):
            n_step_reward_target = 0.0
            for i in range(n_steps):
                step_index = t + i
                if step_index < traj_len:
                    n_step_reward_target += (
                        discount**i * trajectory_raw[step_index]["reward"]
                    )
                else:
                    if traj_len > 0:
                        last_step_value = trajectory_raw[-1]["value_target"]
                        n_step_reward_target += discount**i * last_step_value
                    break

            bootstrap_index = t + n_steps
            if bootstrap_index < traj_len:
                n_step_reward_target += (
                    discount**n_steps * trajectory_raw[bootstrap_index]["value_target"]
                )
            elif bootstrap_index == traj_len and traj_len > 0:
                last_step_value = trajectory_raw[-1]["value_target"]
                n_step_reward_target += discount**n_steps * last_step_value

            # Create the final TrajectoryStep dict
            step_data: TrajectoryStep = {
                "observation": trajectory_raw[t]["observation"],
                "action": trajectory_raw[t]["action"],
                "reward": trajectory_raw[t]["reward"],
                "policy_target": trajectory_raw[t]["policy_target"],
                "value_target": trajectory_raw[t]["value_target"],
                "n_step_reward_target": n_step_reward_target,  # Add the calculated target
                "hidden_state": trajectory_raw[t]["hidden_state"],
            }
            completed_trajectory.append(step_data)

        return completed_trajectory

    def run_episode(self) -> SelfPlayResult:
        """Runs a single MuZero self-play episode."""
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)
        trajectory_raw: list[dict] = []  # Store raw data before adding n-step target
        step_root_visits: list[int] = []
        step_tree_depths: list[int] = []
        step_simulations: list[int] = []
        current_hidden_state: torch.Tensor | None = None

        logger.info(f"W{self.actor_id}: Starting episode seed {episode_seed}")
        self._report_current_state(game)

        while not game.is_over():
            game_step = game.current_step
            try:
                observation: StateType = extract_state_features(game, self.model_config)
                valid_actions = game.valid_actions()
                if not valid_actions:
                    logger.warning(
                        f"W{self.actor_id}: No valid actions step {game_step}"
                    )
                    break
            except Exception as e:
                logger.error(
                    f"W{self.actor_id}: Feat/Action error step {game_step}: {e}",
                    exc_info=True,
                )
                break

            if current_hidden_state is None:
                _, _, _, hidden_state_tensor = self.nn_evaluator.initial_inference(
                    observation
                )
                current_hidden_state = hidden_state_tensor.squeeze(0)
                root_node = Node(
                    hidden_state=current_hidden_state, initial_game_state=game.copy()
                )
            else:
                root_node = Node(
                    hidden_state=current_hidden_state, initial_game_state=game.copy()
                )

            mcts_max_depth = 0
            try:
                mcts_max_depth = run_mcts_simulations(
                    root_node, self.mcts_config, self.nn_evaluator, valid_actions
                )
                step_root_visits.append(root_node.visit_count)
                step_tree_depths.append(mcts_max_depth)
                step_simulations.append(self.mcts_config.num_simulations)
            except MCTSExecutionError as mcts_err:
                logger.error(
                    f"W{self.actor_id} Step {game_step}: MCTS failed: {mcts_err}"
                )
                break

            try:
                temp = (
                    self.mcts_config.temperature_initial
                    if game_step < self.mcts_config.temperature_anneal_steps
                    else self.mcts_config.temperature_final
                )
                action: ActionType = select_action_based_on_visits(
                    root_node, temperature=temp
                )
                policy_target: PolicyTargetMapping = get_policy_target(
                    root_node, temperature=1.0
                )
                value_target: float = root_node.value_estimate
            except Exception as policy_err:
                logger.error(
                    f"W{self.actor_id} Step {game_step}: Policy/Action failed: {policy_err}",
                    exc_info=True,
                )
                break

            real_reward, done = game.step(action)

            # Store raw step data (n_step_reward_target will be added later)
            step_data_raw: dict = {
                "observation": observation,
                "action": action,
                "reward": real_reward,
                "policy_target": policy_target,
                "value_target": value_target,
                "hidden_state": (
                    current_hidden_state.detach().cpu().numpy()
                    if current_hidden_state is not None
                    else None
                ),
            }
            trajectory_raw.append(step_data_raw)

            if not done:
                try:
                    if current_hidden_state is not None:
                        hs_batch = current_hidden_state.to(self.device).unsqueeze(0)
                        next_hidden_state_tensor, _ = self.nn_evaluator.model.dynamics(
                            hs_batch, action
                        )
                        current_hidden_state = next_hidden_state_tensor.squeeze(0)
                    else:
                        logger.error(
                            f"W{self.actor_id} Step {game_step}: hidden_state is None before dynamics call"
                        )
                        break
                except Exception as dyn_err:
                    logger.error(
                        f"W{self.actor_id} Step {game_step}: Dynamics error: {dyn_err}",
                        exc_info=True,
                    )
                    break
            else:
                current_hidden_state = None

            self._report_current_state(game)
            self._log_step_stats_async(
                game_step, root_node.visit_count, mcts_max_depth, real_reward
            )
            if done:
                break

        # --- Episode End ---
        final_score = game.game_score
        total_steps_episode = game.current_step
        logger.info(
            f"W{self.actor_id}: Episode finished. Score: {final_score:.2f}, Steps: {total_steps_episode}"
        )

        # --- Calculate N-step targets and create final Trajectory ---
        trajectory: Trajectory = self._calculate_n_step_targets(trajectory_raw)
        # ---

        total_sims = sum(step_simulations)
        avg_visits = np.mean(step_root_visits) if step_root_visits else 0.0
        avg_depth = np.mean(step_tree_depths) if step_tree_depths else 0.0
        if not trajectory:
            logger.warning(f"W{self.actor_id}: Episode finished empty trajectory.")

        return SelfPlayResult(
            trajectory=trajectory,
            final_score=final_score,
            episode_steps=total_steps_episode,
            total_simulations=total_sims,
            avg_root_visits=float(avg_visits),
            avg_tree_depth=float(avg_depth),
        )


File: muzerotriangle\rl\self_play\__init__.py


File: muzerotriangle\stats\collector.py
# File: muzerotriangle/stats/collector.py
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any, cast  # Added cast

import numpy as np
import ray

from ..utils.types import StatsCollectorData, StepInfo

if TYPE_CHECKING:
    from ..environment import GameState

logger = logging.getLogger(__name__)


@ray.remote
class StatsCollectorActor:
    """
    Ray actor for collecting time-series statistics and latest worker game states.
    Stores metrics as (StepInfo, value) tuples.
    """

    def __init__(self, max_history: int | None = 1000):
        self.max_history = max_history
        self._data: StatsCollectorData = {}
        # Store the latest GameState reported by each worker
        self._latest_worker_states: dict[int, GameState] = {}
        self._last_state_update_time: dict[int, float] = {}

        # Ensure logger is configured for the actor process
        log_level = logging.INFO
        # Check if runtime_context is available before using it
        actor_id_str = "UnknownActor"
        try:
            if ray.is_initialized():
                actor_id_str = ray.get_runtime_context().get_actor_id()
        except Exception:
            pass  # Ignore if context cannot be retrieved
        log_format = f"%(asctime)s [%(levelname)s] [StatsCollectorActor pid={actor_id_str}] %(name)s: %(message)s"
        logging.basicConfig(level=log_level, format=log_format, force=True)
        global logger  # Re-assign logger after config
        logger = logging.getLogger(__name__)

        logger.info(f"Initialized with max_history={max_history}.")

    # --- Metric Logging ---

    def log(self, metric_name: str, value: float, step_info: StepInfo):
        """Logs a single metric value with its associated step information."""
        logger.debug(
            f"Attempting to log metric='{metric_name}', value={value}, step_info={step_info}"
        )
        if not isinstance(metric_name, str):
            logger.error(f"Invalid metric_name type: {type(metric_name)}")
            return
        if not isinstance(step_info, dict):
            logger.error(f"Invalid step_info type: {type(step_info)}")
            return
        if not np.isfinite(value):
            logger.warning(
                f"Received non-finite value for metric '{metric_name}': {value}. Skipping log."
            )
            return

        try:
            if metric_name not in self._data:
                logger.debug(f"Creating new deque for metric: '{metric_name}'")
                self._data[metric_name] = deque(maxlen=self.max_history)

            # Ensure value is float for consistency
            value_float = float(value)
            # Store the StepInfo dict directly
            self._data[metric_name].append((step_info, value_float))
            logger.debug(
                f"Successfully logged metric='{metric_name}', value={value_float}, step_info={step_info}. Deque size: {len(self._data[metric_name])}"
            )
        except (ValueError, TypeError) as e:
            logger.error(
                f"Could not log metric '{metric_name}'. Invalid value conversion: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error logging metric '{metric_name}' (value={value}, step_info={step_info}): {e}",
                exc_info=True,
            )

    def log_batch(self, metrics: dict[str, tuple[float, StepInfo]]):
        """Logs a batch of metrics, each with its StepInfo."""
        received_keys = list(metrics.keys())
        logger.debug(
            f"Log batch received with {len(metrics)} metrics. Keys: {received_keys}"
        )
        for name, (value, step_info) in metrics.items():
            self.log(name, value, step_info)  # Delegate to single log method

    # --- Game State Handling (No change needed) ---

    def update_worker_game_state(self, worker_id: int, game_state: "GameState"):
        """Stores the latest game state received from a worker."""
        if not isinstance(worker_id, int):
            logger.error(f"Invalid worker_id type: {type(worker_id)}")
            return
        # Basic check if it looks like a GameState object (can add more checks if needed)
        if not hasattr(game_state, "grid_data") or not hasattr(game_state, "shapes"):
            logger.error(
                f"Invalid game_state object received from worker {worker_id}: type={type(game_state)}"
            )
            return
        # Store the received state (it should be a copy from the worker)
        self._latest_worker_states[worker_id] = game_state
        self._last_state_update_time[worker_id] = time.time()
        logger.debug(
            f"Updated game state for worker {worker_id} (Step: {game_state.current_step})"
        )

    def get_latest_worker_states(self) -> dict[int, "GameState"]:
        """Returns a shallow copy of the latest worker states dictionary."""
        logger.debug(
            f"get_latest_worker_states called. Returning states for workers: {list(self._latest_worker_states.keys())}"
        )
        return self._latest_worker_states.copy()

    # --- Data Retrieval & Management ---

    def get_data(self) -> StatsCollectorData:
        """Returns a copy of the collected statistics data."""
        logger.debug(f"get_data called. Returning {len(self._data)} metrics.")
        # Return copies of deques to prevent external modification
        return {k: dq.copy() for k, dq in self._data.items()}

    def get_metric_data(self, metric_name: str) -> deque[tuple[StepInfo, float]] | None:
        """Returns a copy of the data deque for a specific metric."""
        dq = self._data.get(metric_name)
        return dq.copy() if dq else None

    def clear(self):
        """Clears all collected statistics and worker states."""
        self._data = {}
        self._latest_worker_states = {}
        self._last_state_update_time = {}
        logger.info("Data and worker states cleared.")

    def get_state(self) -> dict[str, Any]:
        """Returns the internal state for saving."""
        # Convert deques to lists for serialization compatibility with cloudpickle/json
        # The items in the list are now (StepInfo, float) tuples
        serializable_metrics = {key: list(dq) for key, dq in self._data.items()}

        state = {
            "max_history": self.max_history,
            "_metrics_data_list": serializable_metrics,  # Use the list version
        }
        logger.info(
            f"get_state called. Returning state for {len(serializable_metrics)} metrics. Worker states NOT included."
        )
        return state

    def set_state(self, state: dict[str, Any]):
        """Restores the internal state from saved data."""
        self.max_history = state.get("max_history", self.max_history)
        loaded_metrics_list = state.get("_metrics_data_list", {})
        self._data = {}
        restored_metrics_count = 0
        for key, items_list in loaded_metrics_list.items():
            if isinstance(items_list, list) and all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict)
                for item in items_list
            ):
                # Ensure items are (StepInfo, float)
                valid_items: list[tuple[StepInfo, float]] = []
                for item in items_list:
                    try:
                        # Basic check for StepInfo structure (can be enhanced)
                        if not isinstance(item[0], dict):
                            raise TypeError("StepInfo is not a dict")
                        # Ensure value is float
                        value = float(item[1])
                        # Cast the dict to StepInfo for type safety
                        step_info = cast("StepInfo", item[0])
                        valid_items.append((step_info, value))
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid item {item} in metric '{key}' during state restore: {e}"
                        )
                # Convert list back to deque with maxlen
                # Cast valid_items to the expected type for deque
                self._data[key] = deque(
                    cast("list[tuple[StepInfo, float]]", valid_items),
                    maxlen=self.max_history,
                )
                restored_metrics_count += 1
            else:
                logger.warning(
                    f"Skipping restore for metric '{key}'. Invalid data format: {type(items_list)}"
                )
        # Clear worker states on restore, as they are transient
        self._latest_worker_states = {}
        self._last_state_update_time = {}
        logger.info(
            f"State restored. Restored {restored_metrics_count} metrics. Max history: {self.max_history}. Worker states cleared."
        )


File: muzerotriangle\stats\plotter.py
# File: muzerotriangle/stats/plotter.py
import contextlib
import logging
import time
from collections import deque
from io import BytesIO
from typing import TYPE_CHECKING, Any

import matplotlib

if TYPE_CHECKING:
    import numpy as np

    # --- MOVED: Import vis_colors only for type checking ---

import pygame

# Use Agg backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- MOVED: Import normalize_color_for_matplotlib here ---
from ..utils.helpers import normalize_color_for_matplotlib  # noqa: E402

# --- CHANGED: Import StepInfo ---
from ..utils.types import StatsCollectorData  # noqa: E402

# --- END CHANGED ---
from .plot_definitions import (  # noqa: E402
    WEIGHT_UPDATE_METRIC_KEY,  # Import key
    PlotDefinitions,
)
from .plot_rendering import render_subplot  # Import subplot rendering logic

logger = logging.getLogger(__name__)


class Plotter:
    """
    Handles creation and caching of the multi-plot Matplotlib surface.
    Uses PlotDefinitions for layout and plot_rendering for drawing subplots.
    """

    def __init__(self, plot_update_interval: float = 0.75):  # Increased interval
        self.plot_surface_cache: pygame.Surface | None = None
        self.last_plot_update_time: float = 0.0
        self.plot_update_interval: float = plot_update_interval
        self.colors = self._init_colors()
        self.plot_definitions = PlotDefinitions(self.colors)  # Instantiate definitions

        self.rolling_window_sizes: list[int] = [
            10,
            50,
            100,
            500,
            1000,
            5000,
        ]

        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None  # type: ignore # numpy is type-checked \only
        self.last_target_size: tuple[int, int] = (0, 0)
        self.last_data_hash: int | None = None

        logger.info(
            f"[Plotter] Initialized with update interval: {self.plot_update_interval}s"
        )

    def _init_colors(self) -> dict[str, tuple[float, float, float]]:
        """Initializes plot colors using hardcoded values to avoid vis import."""
        # This breaks the circular import. Ensure these match vis_colors.py
        colors_rgb = {
            "YELLOW": (230, 230, 40),
            "WHITE": (255, 255, 255),
            "LIGHT_GRAY": (180, 180, 180),
            "LIGHTG": (144, 238, 144),
            "RED": (220, 40, 40),
            "BLUE": (60, 60, 220),
            "GREEN": (40, 200, 40),
            "CYAN": (40, 200, 200),
            "PURPLE": (140, 40, 140),
            "BLACK": (0, 0, 0),
            "GRAY": (100, 100, 100),
            "ORANGE": (240, 150, 20),
            "HOTPINK": (255, 105, 180),
        }

        return {
            "RL/Current_Score": normalize_color_for_matplotlib(colors_rgb["YELLOW"]),
            "RL/Step_Reward": normalize_color_for_matplotlib(colors_rgb["WHITE"]),
            "MCTS/Step_Visits": normalize_color_for_matplotlib(
                colors_rgb["LIGHT_GRAY"]
            ),
            "MCTS/Step_Depth": normalize_color_for_matplotlib(colors_rgb["LIGHTG"]),
            "Loss/Total": normalize_color_for_matplotlib(colors_rgb["RED"]),
            "Loss/Value": normalize_color_for_matplotlib(colors_rgb["BLUE"]),
            "Loss/Policy": normalize_color_for_matplotlib(colors_rgb["GREEN"]),
            "LearningRate": normalize_color_for_matplotlib(colors_rgb["CYAN"]),
            "Buffer/Size": normalize_color_for_matplotlib(colors_rgb["PURPLE"]),
            WEIGHT_UPDATE_METRIC_KEY: normalize_color_for_matplotlib(
                colors_rgb["BLACK"]
            ),
            "placeholder": normalize_color_for_matplotlib(colors_rgb["GRAY"]),
            "Rate/Steps_Per_Sec": normalize_color_for_matplotlib(colors_rgb["ORANGE"]),
            "Rate/Episodes_Per_Sec": normalize_color_for_matplotlib(
                colors_rgb["HOTPINK"]
            ),
            "Rate/Simulations_Per_Sec": normalize_color_for_matplotlib(
                colors_rgb["LIGHTG"]
            ),
            "PER/Beta": normalize_color_for_matplotlib(colors_rgb["ORANGE"]),
            "Loss/Entropy": normalize_color_for_matplotlib(colors_rgb["PURPLE"]),
            "Loss/Mean_TD_Error": normalize_color_for_matplotlib(colors_rgb["RED"]),
            "Progress/Train_Step_Percent": normalize_color_for_matplotlib(
                colors_rgb["GREEN"]
            ),
            "Progress/Total_Simulations": normalize_color_for_matplotlib(
                colors_rgb["CYAN"]
            ),
        }

    def _init_figure(self, target_width: int, target_height: int):
        """Initializes the Matplotlib figure and axes based on plot definitions."""
        logger.info(
            f"[Plotter] Initializing Matplotlib figure for size {target_width}x{target_height}"
        )
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                logger.warning(f"Error closing previous figure: {e}")

        dpi = 96
        fig_width_in = max(1, target_width / dpi)
        fig_height_in = max(1, target_height / dpi)

        try:
            nrows = self.plot_definitions.nrows
            ncols = self.plot_definitions.ncols
            self.fig, self.axes = plt.subplots(
                nrows,
                ncols,
                figsize=(fig_width_in, fig_height_in),
                dpi=dpi,
                sharex=False,
            )
            if self.axes is None:
                raise RuntimeError("Failed to create Matplotlib subplots.")

            self.fig.patch.set_facecolor((0.1, 0.1, 0.1))
            self.fig.subplots_adjust(
                hspace=0.40,
                wspace=0.08,
                left=0.03,
                right=0.99,
                bottom=0.05,
                top=0.98,
            )
            self.last_target_size = (target_width, target_height)
            logger.info(
                f"[Plotter] Matplotlib figure initialized ({nrows}x{ncols} grid)."
            )
        except Exception as e:
            logger.error(f"Error creating Matplotlib figure: {e}", exc_info=True)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

    def _get_data_hash(self, plot_data: StatsCollectorData) -> int:
        """Generates a hash based on data lengths and recent values."""
        hash_val = 0
        sample_size = 5
        for key in sorted(plot_data.keys()):
            dq = plot_data[key]
            hash_val ^= hash(key) ^ len(dq)
            if not dq:
                continue
            try:
                num_to_sample = min(len(dq), sample_size)
                for i in range(-1, -num_to_sample - 1, -1):
                    # Hash StepInfo dict and value
                    step_info, val = dq[i]
                    # Simple hash for dict: hash tuple of sorted items
                    step_info_hash = hash(tuple(sorted(step_info.items())))
                    hash_val ^= step_info_hash ^ hash(f"{val:.6f}")
            except IndexError:
                pass
        return hash_val

    def _update_plot_data(self, plot_data: StatsCollectorData) -> bool:
        """Updates the data on the existing Matplotlib axes using render_subplot."""
        if self.fig is None or self.axes is None:
            logger.warning("[Plotter] Cannot update plot data, figure not initialized.")
            return False

        plot_update_start = time.monotonic()
        try:
            axes_flat = self.axes.flatten()
            plot_defs = self.plot_definitions.get_definitions()
            num_plots = len(plot_defs)

            # Extract weight update steps (global_step values)
            weight_update_steps: list[int] = []
            if WEIGHT_UPDATE_METRIC_KEY in plot_data:
                dq = plot_data[WEIGHT_UPDATE_METRIC_KEY]
                if dq:
                    # Extract global_step from StepInfo
                    weight_update_steps = [
                        step_info["global_step"]
                        for step_info, _ in dq
                        if "global_step" in step_info
                    ]

            for i, plot_def in enumerate(plot_defs):
                if i >= len(axes_flat):
                    break
                ax = axes_flat[i]
                # Pass weight_update_steps
                render_subplot(
                    ax=ax,
                    plot_data=plot_data,
                    plot_def=plot_def,
                    colors=self.colors,
                    rolling_window_sizes=self.rolling_window_sizes,
                    weight_update_steps=weight_update_steps,  # Pass the list
                )

            for i in range(num_plots, len(axes_flat)):
                ax = axes_flat[i]
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_color("gray")
                ax.spines["left"].set_color("gray")

            self._apply_final_axis_formatting(axes_flat)

            plot_update_duration = time.monotonic() - plot_update_start
            logger.debug(f"[Plotter] Plot data updated in {plot_update_duration:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating plot data: {e}", exc_info=True)
            try:
                if self.axes is not None:
                    for ax in self.axes.flatten():
                        ax.clear()
            except Exception:
                pass
            return False

    def _apply_final_axis_formatting(self, axes_flat: Any):
        """Hides x-axis labels for plots not in the bottom row."""
        if not hasattr(axes_flat, "__iter__"):
            logger.error("axes_flat is not iterable in _apply_final_axis_formatting")
            return

        nrows = self.plot_definitions.nrows
        ncols = self.plot_definitions.ncols
        num_plots = len(self.plot_definitions.get_definitions())

        for i, ax in enumerate(axes_flat):
            if i >= num_plots:
                continue

            if i < (nrows - 1) * ncols:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0)

    def _render_figure_to_surface(
        self, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Renders the current Matplotlib figure to a Pygame surface."""
        if self.fig is None:
            logger.warning("[Plotter] Cannot render figure, not initialized.")
            return None

        render_start = time.monotonic()
        try:
            self.fig.canvas.draw_idle()
            buf = BytesIO()
            self.fig.savefig(
                buf, format="png", transparent=False, facecolor=self.fig.get_facecolor()
            )
            buf.seek(0)
            plot_img_surface = pygame.image.load(buf, "png").convert_alpha()
            buf.close()

            current_size = plot_img_surface.get_size()
            if current_size != (target_width, target_height):
                plot_img_surface = pygame.transform.smoothscale(
                    plot_img_surface, (target_width, target_height)
                )
            render_duration = time.monotonic() - render_start
            logger.debug(
                f"[Plotter] Figure rendered to surface in {render_duration:.4f}s"
            )
            return plot_img_surface

        except Exception as e:
            logger.error(f"Error rendering Matplotlib figure: {e}", exc_info=True)
            return None

    def get_plot_surface(
        self, plot_data: StatsCollectorData, target_width: int, target_height: int
    ) -> pygame.Surface | None:
        """Returns the cached plot surface or creates/updates one if needed."""
        current_time = time.time()
        has_data = any(
            isinstance(dq, deque) and dq
            for key, dq in plot_data.items()
            if not key.startswith("Internal/")
        )
        target_size = (target_width, target_height)

        needs_reinit = (
            self.fig is None
            or self.axes is None
            or self.last_target_size != target_size
        )
        current_data_hash = self._get_data_hash(plot_data)
        data_changed = self.last_data_hash != current_data_hash
        time_elapsed = (
            current_time - self.last_plot_update_time
        ) > self.plot_update_interval
        needs_update = data_changed or time_elapsed
        can_create_plot = target_width > 50 and target_height > 50

        if not can_create_plot:
            if self.plot_surface_cache is not None:
                logger.info("[Plotter] Target size too small, clearing cache/figure.")
                self.plot_surface_cache = None
                if self.fig:
                    plt.close(self.fig)
                self.fig, self.axes, self.last_target_size = None, None, (0, 0)
            return None

        if not has_data:
            logger.debug("[Plotter] No plot data available, returning cache (if any).")
            return self.plot_surface_cache

        try:
            if needs_reinit:
                self._init_figure(target_width, target_height)
                needs_update = True

            if needs_update and self.fig:
                if self._update_plot_data(plot_data):
                    self.plot_surface_cache = self._render_figure_to_surface(
                        target_width, target_height
                    )
                    self.last_plot_update_time = current_time
                    self.last_data_hash = current_data_hash
                else:
                    logger.warning(
                        "[Plotter] Plot update failed, returning stale cache if available."
                    )
            elif (
                self.plot_surface_cache is None
                and self.fig
                and self._update_plot_data(plot_data)
            ):
                self.plot_surface_cache = self._render_figure_to_surface(
                    target_width, target_height
                )
                self.last_plot_update_time = current_time
                self.last_data_hash = current_data_hash

        except Exception as e:
            logger.error(f"[Plotter] Error in get_plot_surface: {e}", exc_info=True)
            self.plot_surface_cache = None
            if self.fig:
                with contextlib.suppress(Exception):
                    plt.close(self.fig)
            self.fig, self.axes, self.last_target_size = None, None, (0, 0)

        return self.plot_surface_cache

    def __del__(self):
        """Ensure Matplotlib figure is closed."""
        if self.fig:
            try:
                plt.close(self.fig)
            except Exception as e:
                print(f"[Plotter] Error closing figure in destructor: {e}")


File: muzerotriangle\stats\plot_definitions.py
# File: muzerotriangle/stats/plot_definitions.py
from typing import Literal, NamedTuple

# Define type for x-axis data source
PlotXAxisType = Literal["index", "global_step", "buffer_size", "game_step_index"]

# Define metric key constant for weight updates
WEIGHT_UPDATE_METRIC_KEY = "Internal/Weight_Update_Step"


class PlotDefinition(NamedTuple):
    """Configuration for a single subplot."""

    metric_key: str  # Key in the StatsCollectorData dictionary
    label: str  # Title displayed on the plot
    y_log_scale: bool  # Use logarithmic scale for y-axis
    x_axis_type: PlotXAxisType  # What the x-axis represents


class PlotDefinitions:
    """Holds the definitions for all plots in the dashboard."""

    def __init__(self, colors: dict[str, tuple[float, float, float]]):
        self.colors = colors  # Store colors if needed for default lookups
        self.nrows: int = 4
        self.ncols: int = 3
        # Key used to get weight update steps for vertical lines
        self.weight_update_key = WEIGHT_UPDATE_METRIC_KEY  # Use the constant

        # Define the layout and properties of each plot
        self._definitions: list[PlotDefinition] = [
            # Row 1
            # --- CHANGED: x_axis_type to "game_step_index" for Score ---
            PlotDefinition("RL/Current_Score", "Score", False, "game_step_index"),
            # --- END CHANGED ---
            PlotDefinition(
                "Rate/Episodes_Per_Sec", "Episodes/sec", False, "buffer_size"
            ),
            PlotDefinition("Loss/Total", "Total Loss", True, "global_step"),
            # Row 2
            # --- CHANGED: x_axis_type to "game_step_index" for Step Reward ---
            PlotDefinition("RL/Step_Reward", "Step Reward", False, "game_step_index"),
            # --- END CHANGED ---
            PlotDefinition(
                "Rate/Simulations_Per_Sec", "Sims/sec", False, "buffer_size"
            ),
            PlotDefinition("Loss/Policy", "Policy Loss", True, "global_step"),
            # Row 3
            # --- CHANGED: x_axis_type to "game_step_index" for MCTS Visits ---
            PlotDefinition("MCTS/Step_Visits", "MCTS Visits", False, "game_step_index"),
            # --- END CHANGED ---
            PlotDefinition("Buffer/Size", "Buffer Size", False, "buffer_size"),
            PlotDefinition("Loss/Value", "Value Loss", True, "global_step"),
            # Row 4
            # --- CHANGED: x_axis_type to "game_step_index" for MCTS Depth ---
            PlotDefinition("MCTS/Step_Depth", "MCTS Depth", False, "game_step_index"),
            # --- END CHANGED ---
            PlotDefinition("Rate/Steps_Per_Sec", "Steps/sec", False, "global_step"),
            PlotDefinition("LearningRate", "Learn Rate", True, "global_step"),
        ]

        # Validate grid size
        if len(self._definitions) > self.nrows * self.ncols:
            raise ValueError(
                f"Number of plot definitions ({len(self._definitions)}) exceeds grid size ({self.nrows}x{self.ncols})"
            )

    def get_definitions(self) -> list[PlotDefinition]:
        """Returns the list of plot definitions."""
        return self._definitions


# Define PlotType for potential external use, though PlotDefinition is more specific
PlotType = PlotDefinition


File: muzerotriangle\stats\plot_rendering.py
# File: muzerotriangle/stats/plot_rendering.py
import logging
from collections import deque
from typing import TYPE_CHECKING  # Added cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..utils.types import StepInfo
from .plot_definitions import PlotDefinition
from .plot_utils import calculate_rolling_average, format_value

if TYPE_CHECKING:
    from .collector import StatsCollectorData

logger = logging.getLogger(__name__)


def _find_closest_index_for_global_step(
    target_global_step: int,
    step_info_list: list[StepInfo],
) -> int | None:
    """
    Finds the index in the step_info_list where the stored 'global_step'
    is closest to the target_global_step.
    Returns None if no suitable point is found or list is empty.
    """
    if not step_info_list:
        return None

    best_match_idx = None
    min_step_diff = float("inf")

    for i, step_info in enumerate(step_info_list):
        global_step_in_info = step_info.get("global_step")

        if global_step_in_info is not None:
            step_diff = abs(global_step_in_info - target_global_step)
            if step_diff < min_step_diff:
                min_step_diff = step_diff
                best_match_idx = i
            # Optimization: If we found an exact match, we can stop early
            # Also, if the steps start increasing again, we passed the best point
            if step_diff == 0 or (
                best_match_idx is not None and global_step_in_info > target_global_step
            ):
                break

    # Optional: Add a tolerance? If min_step_diff is too large, maybe don't return a match?
    # For now, return the index of the closest found value.
    return best_match_idx


def render_subplot(
    ax: plt.Axes,
    plot_data: "StatsCollectorData",
    plot_def: PlotDefinition,
    colors: dict[str, tuple[float, float, float]],
    rolling_window_sizes: list[int],
    weight_update_steps: list[int] | None = None,  # Global steps where updates happened
) -> bool:
    """
    Renders data for a single metric onto the given Matplotlib Axes object.
    Scatter point size/alpha decrease linearly as more data/longer averages are shown.
    Draws semi-transparent black, solid vertical lines for weight updates on all plots.
    Returns True if data was plotted, False otherwise.
    """
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    metric_key = plot_def.metric_key
    label = plot_def.label
    log_scale = plot_def.y_log_scale
    x_axis_type = plot_def.x_axis_type  # e.g., "global_step", "buffer_size", "index"

    x_data: list[int] = []
    y_data: list[float] = []
    x_label_text = "Index"  # Default label
    step_info_list: list[StepInfo] = []  # Store step info for mapping

    dq = plot_data.get(metric_key, deque())
    if dq:
        # Extract x-axis value and store StepInfo
        temp_x = []
        temp_y = []
        for i, (step_info, value) in enumerate(dq):
            x_val: int | None = None
            if x_axis_type == "global_step":
                x_val = step_info.get("global_step")
                x_label_text = "Train Step"
            elif x_axis_type == "buffer_size":
                x_val = step_info.get("buffer_size")
                x_label_text = "Buffer Size"
            else:  # index
                x_val = i  # Use the simple index 'i' as the x-value
                if (
                    "Score" in metric_key
                    or "Reward" in metric_key
                    or "MCTS" in metric_key
                ):
                    x_label_text = "Game Step Index"  # Label remains descriptive
                else:
                    x_label_text = "Data Point Index"

            if x_val is not None:
                temp_x.append(x_val)
                temp_y.append(value)
                step_info_list.append(
                    step_info
                )  # Keep StepInfo aligned with data points
            else:
                logger.warning(
                    f"Missing x-axis key '{x_axis_type}' in step_info for metric '{metric_key}'. Skipping point."
                )

        x_data = temp_x
        y_data = temp_y

    color_mpl = colors.get(metric_key, (0.5, 0.5, 0.5))
    placeholder_color_mpl = colors.get("placeholder", (0.5, 0.5, 0.5))

    data_plotted = False
    if not x_data or not y_data:
        ax.text(
            0.5,
            0.5,
            f"{label}\n(No Data)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=placeholder_color_mpl,
            fontsize=9,
        )
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        data_plotted = True

        # Determine best rolling average window
        num_points = len(y_data)
        best_window = 0
        for window in sorted(rolling_window_sizes, reverse=True):
            if num_points >= window:
                best_window = window
                break

        # Determine scatter size/alpha based on best_window
        # Initialize as float
        scatter_size: float = 0.0
        scatter_alpha: float = 0.0
        max_scatter_size = 10.0  # Use float
        min_scatter_size = 1.0  # Use float
        max_scatter_alpha = 0.3
        min_scatter_alpha = 0.03
        max_window_for_interp = float(max(rolling_window_sizes))

        if best_window == 0:
            scatter_size = max_scatter_size
            scatter_alpha = max_scatter_alpha
        elif best_window >= max_window_for_interp:
            scatter_size = min_scatter_size
            scatter_alpha = min_scatter_alpha
        else:
            interp_fraction = best_window / max_window_for_interp
            # Cast result of np.interp to float
            scatter_size = float(
                np.interp(interp_fraction, [0, 1], [max_scatter_size, min_scatter_size])
            )
            scatter_alpha = float(
                np.interp(
                    interp_fraction, [0, 1], [max_scatter_alpha, min_scatter_alpha]
                )
            )

        # Plot raw data with dynamic size/alpha
        ax.scatter(
            x_data,
            y_data,
            color=color_mpl,
            alpha=scatter_alpha,
            s=scatter_size,  # Pass float size
            label="_nolegend_",
            zorder=2,
        )

        # Plot best rolling average
        if best_window > 0:
            rolling_avg = calculate_rolling_average(y_data, best_window)
            if len(rolling_avg) == len(x_data):
                ax.plot(
                    x_data,
                    rolling_avg,
                    color=color_mpl,
                    alpha=0.9,
                    linewidth=1.5,
                    label=f"Avg {best_window}",
                    zorder=3,
                )
                ax.legend(
                    fontsize=6, loc="upper right", frameon=False, labelcolor="lightgray"
                )
            else:
                logger.warning(
                    f"Length mismatch for rolling avg ({len(rolling_avg)}) vs x_data ({len(x_data)}) for {label}. Skipping avg plot."
                )

        # Draw vertical lines by mapping global_step to current x-axis value
        if weight_update_steps and step_info_list:
            plotted_line_x_values: set[float] = set()  # Store plotted x-values as float
            for update_global_step in weight_update_steps:
                x_index_for_line = _find_closest_index_for_global_step(
                    update_global_step, step_info_list
                )

                if x_index_for_line is not None and x_index_for_line < len(x_data):
                    actual_x_value: int | float
                    if x_axis_type == "index":
                        actual_x_value = x_index_for_line  # int
                    else:
                        # Explicitly cast list access to int to satisfy MyPy
                        actual_x_value = int(x_data[x_index_for_line])  # int

                    # Cast to float for axvline and check if already plotted
                    actual_x_float = float(actual_x_value)
                    if actual_x_float not in plotted_line_x_values:
                        ax.axvline(
                            x=actual_x_float,  # Pass float
                            color="black",
                            linestyle="-",
                            linewidth=0.7,
                            alpha=0.5,
                            zorder=10,
                        )
                        plotted_line_x_values.add(actual_x_float)
                else:
                    logger.debug(
                        f"Could not map global_step {update_global_step} to an index for plot '{label}'"
                    )

        # Formatting
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.tick_params(axis="both", which="major", labelsize=7, colors="lightgray")
        ax.grid(
            True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4), zorder=1
        )  # Ensure grid is behind lines

        # Set y-axis scale
        if log_scale:
            ax.set_yscale("log")
            min_val = min((v for v in y_data if v > 0), default=1e-6)
            max_val = max(y_data, default=1.0)
            ylim_bottom = max(1e-9, min_val * 0.1)
            ylim_top = max_val * 10
            if ylim_bottom < ylim_top:
                ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
            else:
                ax.set_ylim(bottom=1e-9, top=1.0)
        else:
            ax.set_yscale("linear")
            min_val = min(y_data) if y_data else 0.0
            max_val = max(y_data) if y_data else 0.0
            val_range = max_val - min_val
            if abs(val_range) < 1e-6:
                center_val = (min_val + max_val) / 2.0
                buffer = max(abs(center_val * 0.1), 0.5)
                ylim_bottom, ylim_top = center_val - buffer, center_val + buffer
            else:
                buffer = val_range * 0.1
                ylim_bottom, ylim_top = min_val - buffer, max_val + buffer
            if all(v >= 0 for v in y_data) and ylim_bottom < 0:
                ylim_bottom = 0.0
            if ylim_bottom >= ylim_top:
                ylim_bottom, ylim_top = min_val - 0.5, max_val + 0.5
                if ylim_bottom >= ylim_top:
                    ylim_bottom, ylim_top = 0.0, max(1.0, max_val)
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

        # Format x-axis
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"
            )
        )
        ax.set_xlabel(x_label_text, fontsize=8, color="gray")

        # Format y-axis
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_value(y)))

        # Add info text (min/max/current)
        current_val_str = format_value(y_data[-1])
        min_val_overall = min(y_data)
        max_val_overall = max(y_data)
        min_str = format_value(min_val_overall)
        max_str = format_value(max_val_overall)
        info_text = f"Min:{min_str} | Max:{max_str} | Cur:{current_val_str}"
        ax.text(
            1.0,
            1.01,
            info_text,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=6,
            color="white",
        )

    # Common Axis Styling (applied regardless of data presence)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")

    return data_plotted


File: muzerotriangle\stats\plot_utils.py
# File: muzerotriangle/stats/plot_utils.py
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

logger = logging.getLogger(__name__)


def calculate_rolling_average(data: list[float], window: int) -> list[float]:
    """Calculates the rolling average with handling for edges."""
    if not data or window <= 0:
        return []
    if window > len(data):
        # If window is larger than data, return average of all data for all points
        avg = np.mean(data)
        # Cast to float explicitly
        return [float(avg)] * len(data)
    # Use convolution for efficient rolling average
    weights = np.ones(window) / window
    rolling_avg = np.convolve(data, weights, mode="valid")
    # Pad the beginning to match the original length
    padding = [float(np.mean(data[:i])) for i in range(1, window)]
    # Cast result to list of floats
    return padding + [float(x) for x in rolling_avg]


def calculate_trend_line(
    steps: list[int], values: list[float]
) -> tuple[list[int], list[float]]:
    """Calculates a simple linear trend line."""
    if len(steps) < 2:
        return [], []
    try:
        coeffs = np.polyfit(steps, values, 1)
        poly = np.poly1d(coeffs)
        trend_values = poly(steps)
        return steps, list(trend_values)
    except Exception as e:
        logger.warning(f"Could not calculate trend line: {e}")
        return [], []


def format_value(value: float) -> str:
    """Formats a float value for display on the plot."""
    if abs(value) < 1e-6:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}k"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def render_single_plot(
    ax: plt.Axes,
    steps: list[int],
    values: list[float],
    label: str,
    color: tuple[float, float, float],
    placeholder_color: tuple[float, float, float],
    rolling_window_sizes: list[int],
    show_placeholder: bool = False,
    placeholder_text: str | None = None,
    y_log_scale: bool = False,
):
    """
    Renders a single metric plot onto a Matplotlib Axes object.
    Plots raw data and the single best rolling average line.
    """
    ax.clear()
    ax.set_facecolor((0.15, 0.15, 0.15))  # Dark background for axes

    if show_placeholder or not steps or not values:
        text_to_display = placeholder_text if placeholder_text else "(No Data)"
        ax.text(
            0.5,
            0.5,
            text_to_display,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=placeholder_color,
            fontsize=9,
        )
        ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")
        return

    # Plot raw data (thin, semi-transparent)
    ax.plot(steps, values, color=color, alpha=0.3, linewidth=0.7, label="_nolegend_")

    # --- CHANGED: Plot only the single best rolling average ---
    num_points = len(steps)
    best_window = 0
    # Iterate through window sizes in descending order
    for window in sorted(rolling_window_sizes, reverse=True):
        if num_points >= window:
            best_window = window
            break  # Found the largest applicable window

    if best_window > 0:
        rolling_avg = calculate_rolling_average(values, best_window)
        ax.plot(
            steps,
            rolling_avg,
            color=color,
            alpha=0.9,  # Make it more prominent
            linewidth=1.5,
            label=f"Avg {best_window}",  # Label this single line
        )
        # Add legend only if rolling average was plotted
        ax.legend(fontsize=6, loc="upper right", frameon=False, labelcolor="lightgray")
    # --- END CHANGED ---

    # Formatting
    ax.set_title(label, loc="left", fontsize=9, color="white", pad=2)
    ax.tick_params(axis="both", which="major", labelsize=7, colors="lightgray")
    ax.grid(True, linestyle=":", linewidth=0.5, color=(0.4, 0.4, 0.4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")

    # Set y-axis scale
    if y_log_scale:
        ax.set_yscale("log")
        # Ensure positive values for log scale, adjust limits if needed
        min_val = (
            min(v for v in values if v > 0) if any(v > 0 for v in values) else 1e-6
        )
        max_val = max(values) if values else 1.0
        # Add buffer for log scale
        ylim_bottom = max(1e-9, min_val * 0.1)
        ylim_top = max_val * 10
        # Prevent potential errors if limits are invalid
        if ylim_bottom < ylim_top:
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
        else:
            ax.set_ylim(bottom=1e-9, top=1.0)  # Fallback limits
    else:
        ax.set_yscale("linear")

    # Format x-axis (steps)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(
        # Remove unused 'p' argument
        FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}")
    )
    ax.set_xlabel("Step", fontsize=8, color="gray")

    # Format y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # Remove unused 'p' argument
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: format_value(y)))

    # Add current value text
    current_val_str = format_value(values[-1])
    ax.text(
        1.0,
        1.01,
        f"Cur: {current_val_str}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7,
        color="white",
    )


File: muzerotriangle\stats\README.md
# File: muzerotriangle/stats/README.md
# Statistics Module (`muzerotriangle.stats`)

## Purpose and Architecture

This module provides utilities for collecting, storing, and visualizing time-series statistics generated during the reinforcement learning training process using Matplotlib rendered onto Pygame surfaces.

-   **[`collector.py`](collector.py):** Defines the `StatsCollectorActor` class, a **Ray actor**. This actor uses dictionaries of `deque`s to store metric values (like losses, rewards, learning rate) associated with **step context information** ([`StepInfo`](../utils/types.py) dictionary containing `global_step`, `buffer_size`, `game_step_index`, etc.). It provides **remote methods** (`log`, `log_batch`) for asynchronous logging from multiple sources and methods (`get_data`, `get_metric_data`) for fetching the stored data. It supports limiting the history size and includes `get_state` and `set_state` methods for checkpointing.
-   **[`plot_definitions.py`](plot_definitions.py):** Defines the structure and properties of each plot in the dashboard (`PlotDefinition`, `PlotDefinitions`), including which step information (`x_axis_type`) should be used for the x-axis (e.g., `global_step`, `buffer_size`, `game_step_index`). **Also defines the `WEIGHT_UPDATE_METRIC_KEY` constant.**
-   **[`plot_rendering.py`](plot_rendering.py):** Contains the `render_subplot` function, responsible for drawing a single metric onto a Matplotlib `Axes` object based on a `PlotDefinition`. **It now accepts a list of `global_step` values where weight updates occurred and draws semi-transparent black, solid vertical lines on all plots by mapping the `global_step` to the corresponding value on the plot's specific x-axis. The raw data scatter points are now rendered with dynamically adjusted size and opacity, starting larger and fading as more data accumulates.**
-   **[`plot_utils.py`](plot_utils.py):** Contains helper functions for Matplotlib plotting, including calculating rolling averages and formatting values.
-   **[`plotter.py`](plotter.py):** Defines the `Plotter` class which manages the overall Matplotlib figure and axes.
    -   It orchestrates the plotting of multiple metrics onto a grid within the figure using `render_subplot`.
    -   It handles rendering the Matplotlib figure to an in-memory buffer and then converting it to a `pygame.Surface`.
    -   It implements caching logic.
    -   **It now extracts the weight update steps (`global_step` values) from the collected data and passes them to `render_subplot`.**

## Exposed Interfaces

-   **Classes:**
    -   `StatsCollectorActor`: Ray actor for collecting stats.
        -   `log.remote(metric_name: str, value: float, step_info: StepInfo)`
        -   `log_batch.remote(metrics: Dict[str, Tuple[float, StepInfo]])`
        -   `get_data.remote() -> StatsCollectorData`
        -   `get_metric_data.remote(metric_name: str) -> Optional[Deque[Tuple[StepInfo, float]]]`
        -   (Other methods: `clear`, `get_state`, `set_state`)
    -   `Plotter`:
        -   `get_plot_surface(plot_data: StatsCollectorData, target_width: int, target_height: int) -> Optional[pygame.Surface]`
    -   `PlotDefinitions`: Holds the layout and properties of all plots.
    -   `PlotDefinition`: NamedTuple defining a single plot.
-   **Types:**
    -   `StatsCollectorData`: Type alias `Dict[str, Deque[Tuple[StepInfo, float]]]` representing the stored data.
    -   `StepInfo`: TypedDict holding step context.
    -   `PlotType`: Alias for `PlotDefinition`.
-   **Functions:**
    -   `render_subplot`: Renders a single subplot, including mapped weight update lines and dynamic scatter points.
-   **Modules:**
    -   `plot_utils`: Contains helper functions.
-   **Constants:**
    -   `WEIGHT_UPDATE_METRIC_KEY`: Key used for logging/retrieving weight update events.

## Dependencies

-   **[`muzerotriangle.visualization`](../visualization/README.md)**: `colors` (used indirectly via `Plotter`).
-   **[`muzerotriangle.utils`](../utils/README.md)**: `helpers`, `types` (including `StepInfo`).
-   **`pygame`**: Used by `plotter.py` to create the final surface.
-   **`matplotlib`**: Used by `plotter.py`, `plot_rendering.py`, and `plot_utils.py` for generating plots.
-   **`numpy`**: Used by `plot_utils.py` and `plot_rendering.py` for calculations.
-   **`ray`**: Used by `StatsCollectorActor`.
-   **Standard Libraries:** `typing`, `logging`, `collections.deque`, `math`, `time`, `io`, `contextlib`.

## Integration

-   The `TrainingLoop` ([`muzerotriangle.training.loop`](../training/loop.py)) instantiates `StatsCollectorActor` and calls its remote `log` or `log_batch` methods, **passing `StepInfo` dictionaries**. It logs the `WEIGHT_UPDATE_METRIC_KEY` when worker weights are updated.
-   The `SelfPlayWorker` ([`muzerotriangle.rl.self_play.worker`](../rl/self_play/worker.py)) calls `log_batch` **passing `StepInfo` dictionaries containing `game_step_index` and `global_step` (of its current weights). It now logs `RL/Current_Score` in addition to `RL/Step_Reward`.**
-   The `DashboardRenderer` ([`muzerotriangle.visualization.core.dashboard_renderer`](../visualization/core/dashboard_renderer.py)) holds a handle to the `StatsCollectorActor` and calls `get_data.remote()` periodically to fetch data for plotting.
-   The `DashboardRenderer` instantiates `Plotter` and calls `get_plot_surface` using the fetched stats data and the target plot area dimensions. It then blits the returned surface.
-   The `DataManager` ([`muzerotriangle.data.data_manager`](../data/data_manager.py)) interacts with the `StatsCollectorActor` via `get_state.remote()` and `set_state.remote()` during checkpoint saving and loading.

---

**Note:** Please keep this README updated when changing the data collection methods (especially the `StepInfo` structure), the plotting functions (especially x-axis definitions), or the way statistics are managed and displayed. Accurate documentation is crucial for maintainability.

File: muzerotriangle\stats\__init__.py
# File: muzerotriangle/stats/__init__.py
"""
Statistics collection and plotting module.
"""

from muzerotriangle.utils.types import StatsCollectorData

from . import plot_utils
from .collector import StatsCollectorActor
from .plot_definitions import PlotDefinitions, PlotType  # Import new definitions
from .plot_rendering import render_subplot  # Import new rendering function
from .plotter import Plotter

__all__ = [
    # Core Collector
    "StatsCollectorActor",
    "StatsCollectorData",
    # Plotting Orchestrator
    "Plotter",
    # Plotting Definitions & Rendering Logic
    "PlotDefinitions",
    "PlotType",
    "render_subplot",
    # Plotting Utilities
    "plot_utils",
]


File: muzerotriangle\structs\constants.py
# File: muzerotriangle/structs/constants.py

# Define standard colors used for shapes
# Ensure these colors are distinct and visually clear
# Also ensure BLACK (0,0,0) is NOT used here if it represents empty in color_np
SHAPE_COLORS: list[tuple[int, int, int]] = [
    (220, 40, 40),  # 0: Red
    (60, 60, 220),  # 1: Blue
    (40, 200, 40),  # 2: Green
    (230, 230, 40),  # 3: Yellow
    (240, 150, 20),  # 4: Orange
    (140, 40, 140),  # 5: Purple
    (40, 200, 200),  # 6: Cyan
    (200, 100, 180),  # 7: Pink (Example addition)
    (100, 180, 200),  # 8: Light Blue (Example addition)
]

# --- NumPy GridData Color Representation ---
# ID for empty cells in the _color_id_np array
NO_COLOR_ID: int = -1
# ID for debug-toggled cells
DEBUG_COLOR_ID: int = -2

# Mapping from Color ID (int >= 0) to RGB tuple.
# Index 0 corresponds to SHAPE_COLORS[0], etc.
# This list is used by visualization to get the RGB from the ID.
COLOR_ID_MAP: list[tuple[int, int, int]] = SHAPE_COLORS

# Reverse mapping for efficient lookup during placement (Color Tuple -> ID)
# Note: Ensure SHAPE_COLORS have unique tuples.
COLOR_TO_ID_MAP: dict[tuple[int, int, int], int] = {
    color: i for i, color in enumerate(COLOR_ID_MAP)
}

# Add special colors to the map if needed for rendering debug/other states
# These IDs won't be stored during normal shape placement.
# Example: If you want to render the debug color:
# DEBUG_RGB_COLOR = (255, 255, 0) # Example Yellow
# COLOR_ID_MAP.append(DEBUG_RGB_COLOR) # Append if needed elsewhere, but generally lookup handled separately.

# --- End NumPy GridData Color Representation ---


File: muzerotriangle\structs\README.md
# File: muzerotriangle/structs/README.md
# Core Structures Module (`muzerotriangle.structs`)

## Purpose and Architecture

This module defines fundamental data structures and constants that are shared across multiple major components of the application (like [`environment`](../environment/README.md), [`visualization`](../visualization/README.md), [`features`](../features/README.md)). Its primary purpose is to break potential circular dependencies that arise when these components need to know about the same basic building blocks.

-   **[`triangle.py`](triangle.py):** Defines the `Triangle` class, representing a single cell on the game grid.
-   **[`shape.py`](shape.py):** Defines the `Shape` class, representing a placeable piece composed of triangles.
-   **[`constants.py`](constants.py):** Defines shared constants, such as the list of possible `SHAPE_COLORS` and color IDs used in `GridData`.

By placing these core definitions in a low-level module with minimal dependencies, other modules can import them without creating import cycles.

## Exposed Interfaces

-   **Classes:**
    -   `Triangle`: Represents a grid cell.
    -   `Shape`: Represents a placeable piece.
-   **Constants:**
    -   `SHAPE_COLORS`: A list of RGB tuples for shape generation.
    -   `NO_COLOR_ID`: Integer ID for empty cells in `GridData`.
    -   `DEBUG_COLOR_ID`: Integer ID for debug-toggled cells in `GridData`.
    -   `COLOR_ID_MAP`: List mapping color ID index to RGB tuple.
    -   `COLOR_TO_ID_MAP`: Dictionary mapping RGB tuple to color ID index.

## Dependencies

This module has minimal dependencies, primarily relying on standard Python libraries (`typing`). It should **not** import from higher-level modules like `environment`, `visualization`, `nn`, `rl`, etc.

---

**Note:** This module should only contain widely shared, fundamental data structures and constants. More complex logic or structures specific to a particular domain (like game rules or rendering details) should remain in their respective modules.
```

**22. File:** `muzerotriangle/training/README.md`
**Explanation:** Review content and add relative links.

```markdown
# File: muzerotriangle/training/README.md
# Training Module (`muzerotriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **[`setup.py`](setup.py):** Contains `setup_training_components` which initializes Ray, detects resources, adjusts worker counts, loads configurations, and creates the core components bundle (`TrainingComponents`).
-   **[`components.py`](components.py):** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **[`loop.py`](loop.py):** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, **passing the current `global_step` to the workers**, and logging a special event (`Internal/Weight_Update_Step`) with the `global_step` to the `StatsCollectorActor` when updates occur.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **[`worker_manager.py`](worker_manager.py):** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. **It now passes the `global_step` to workers when updating weights.**
-   **[`loop_helpers.py`](loop_helpers.py):** Contains helper functions used by `TrainingLoop` for tasks like logging rates, updating the visual queue, validating experiences, and logging results. **It constructs the `StepInfo` dictionary containing relevant step counters (`global_step`, `buffer_size`) for logging.** It also includes logic to log the weight update event.
-   **[`runners.py`](runners.py):** Re-exports the main entry point functions (`run_training_headless_mode`, `run_training_visual_mode`) from their respective modules.
-   **[`headless_runner.py`](headless_runner.py) / [`visual_runner.py`](visual_runner.py):** Contain the top-level logic for running training in either headless or visual mode. They handle argument parsing (via CLI), setup logging, call `setup_training_components`, load initial state, run the `TrainingLoop` (potentially in a separate thread for visual mode), handle visualization setup (visual mode), and manage overall cleanup (MLflow, Ray shutdown).
-   **[`logging_utils.py`](logging_utils.py):** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`headless_runner`/`visual_runner`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingLoop`: Contains the core async loop logic.
    -   `TrainingComponents`: Dataclass holding initialized components.
    -   `WorkerManager`: Manages worker actors.
    -   `LoopHelpers`: Provides helper functions for the loop.
-   **Functions (from `runners.py`):**
    -   `run_training_headless_mode(...) -> int`
    -   `run_training_visual_mode(...) -> int`
-   **Functions (from `setup.py`):**
    -   `setup_training_components(...) -> Tuple[Optional[TrainingComponents], bool]`
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: All configuration classes.
-   **[`muzerotriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`muzerotriangle.rl`](../rl/README.md)**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **[`muzerotriangle.data`](../data/README.md)**: `DataManager`, `LoadedTrainingState`.
-   **[`muzerotriangle.stats`](../stats/README.md)**: `StatsCollectorActor`, `PlotDefinitions`.
-   **[`muzerotriangle.environment`](../environment/README.md)**: `GameState`.
-   **[`muzerotriangle.utils`](../utils/README.md)**: Helper functions and types (including `StepInfo`).
-   **[`muzerotriangle.visualization`](../visualization/README.md)**: `ProgressBar`, `DashboardRenderer`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`, `sys`, `traceback`, `pathlib`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the logging of step context information (`StepInfo`) and worker weight updates.

File: muzerotriangle\structs\shape.py
from __future__ import annotations


class Shape:
    """Represents a polyomino-like shape made of triangles."""

    def __init__(
        self, triangles: list[tuple[int, int, bool]], color: tuple[int, int, int]
    ):
        self.triangles: list[tuple[int, int, bool]] = sorted(triangles)
        self.color: tuple[int, int, int] = color

    def bbox(self) -> tuple[int, int, int, int]:
        """Calculates bounding box (min_r, min_c, max_r, max_c) in relative coords."""
        if not self.triangles:
            return (0, 0, 0, 0)
        rows = [t[0] for t in self.triangles]
        cols = [t[1] for t in self.triangles]
        return (min(rows), min(cols), max(rows), max(cols))

    def copy(self) -> Shape:
        """Creates a shallow copy (triangle list is copied, color is shared)."""
        new_shape = Shape.__new__(Shape)
        new_shape.triangles = list(self.triangles)
        new_shape.color = self.color
        return new_shape

    def __str__(self) -> str:
        return f"Shape(Color:{self.color}, Tris:{len(self.triangles)})"

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on triangles and color."""
        if not isinstance(other, Shape):
            return NotImplemented
        return self.triangles == other.triangles and self.color == other.color

    def __hash__(self) -> int:
        """Allows shapes to be used in sets/dicts if needed."""
        return hash((tuple(self.triangles), self.color))


File: muzerotriangle\structs\triangle.py
from __future__ import annotations


class Triangle:
    """Represents a single triangular cell on the grid."""

    def __init__(self, row: int, col: int, is_up: bool, is_death: bool = False):
        self.row = row
        self.col = col
        self.is_up = is_up
        self.is_death = is_death
        self.is_occupied = is_death
        self.color: tuple[int, int, int] | None = None

        self.neighbor_left: Triangle | None = None
        self.neighbor_right: Triangle | None = None
        self.neighbor_vert: Triangle | None = None

    def get_points(
        self, ox: float, oy: float, cw: float, ch: float
    ) -> list[tuple[float, float]]:
        """Calculates vertex points for drawing, relative to origin (ox, oy)."""
        x = ox + self.col * (cw * 0.75)
        y = oy + self.row * ch
        if self.is_up:
            return [(x, y + ch), (x + cw, y + ch), (x + cw / 2, y)]
        else:
            return [(x, y), (x + cw, y), (x + cw / 2, y + ch)]

    def copy(self) -> Triangle:
        """Creates a copy of the Triangle object's state (neighbors are not copied)."""
        new_tri = Triangle(self.row, self.col, self.is_up, self.is_death)
        new_tri.is_occupied = self.is_occupied
        new_tri.color = self.color
        return new_tri

    def __repr__(self) -> str:
        state = "D" if self.is_death else ("O" if self.is_occupied else ".")
        orient = "^" if self.is_up else "v"
        return f"T({self.row},{self.col} {orient}{state})"

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return NotImplemented
        return self.row == other.row and self.col == other.col


File: muzerotriangle\structs\__init__.py
# File: muzerotriangle/structs/__init__.py
"""
Module for core data structures used across different parts of the application,
like environment, visualization, and features. Helps avoid circular dependencies.
"""

# Correctly export constants from the constants submodule
from .constants import (
    COLOR_ID_MAP,
    COLOR_TO_ID_MAP,
    DEBUG_COLOR_ID,
    NO_COLOR_ID,
    SHAPE_COLORS,
)
from .shape import Shape
from .triangle import Triangle

__all__ = [
    "Triangle",
    "Shape",
    # Exported Constants
    "SHAPE_COLORS",
    "NO_COLOR_ID",
    "DEBUG_COLOR_ID",
    "COLOR_ID_MAP",
    "COLOR_TO_ID_MAP",
]


File: muzerotriangle\training\components.py
# File: muzerotriangle/training/components.py
from dataclasses import dataclass
from typing import TYPE_CHECKING

# --- ADDED: Import ActorHandle ---
import ray

# --- END ADDED ---

if TYPE_CHECKING:
    # Keep ray import here as well for consistency if needed elsewhere
    # import ray

    from muzerotriangle.config import (
        EnvConfig,
        MCTSConfig,
        ModelConfig,
        PersistenceConfig,
        TrainConfig,
    )
    from muzerotriangle.data import DataManager
    from muzerotriangle.nn import NeuralNetwork
    from muzerotriangle.rl import ExperienceBuffer, Trainer

    # --- REMOVED: Import StatsCollectorActor class ---
    # from muzerotriangle.stats import StatsCollectorActor
    # --- END REMOVED ---


@dataclass
class TrainingComponents:
    """Holds the initialized core components needed for training."""

    nn: "NeuralNetwork"
    buffer: "ExperienceBuffer"
    trainer: "Trainer"
    data_manager: "DataManager"
    # --- CORRECTED: Use ActorHandle type hint ---
    stats_collector_actor: ray.actor.ActorHandle | None
    # --- END CORRECTED ---
    train_config: "TrainConfig"
    env_config: "EnvConfig"
    model_config: "ModelConfig"
    mcts_config: "MCTSConfig"
    persist_config: "PersistenceConfig"


File: muzerotriangle\training\headless_runner.py
# File: muzerotriangle/training/headless_runner.py
import logging
import sys
import traceback
from pathlib import Path

import mlflow
import ray

from ..config import APP_NAME, PersistenceConfig, TrainConfig

# Import Trajectory type
from .components import TrainingComponents
from .logging_utils import (
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    # (No changes needed here)
    try:
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
        logger.info(f"Set MLflow experiment to: {APP_NAME}")
        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
            return True
        else:
            logger.error("MLflow run failed to start.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        return False


def _load_and_apply_initial_state(components: TrainingComponents) -> TrainingLoop:
    """Loads initial state (checkpoint, MuZero buffer) and applies it."""
    loaded_state = components.data_manager.load_initial_state()
    # Pass None for visual queue in headless mode
    training_loop = TrainingLoop(components, visual_state_queue=None)

    # --- Apply Checkpoint Data (No change needed here) ---
    if loaded_state.checkpoint_data:
        cp_data = loaded_state.checkpoint_data
        logger.info(
            f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
        )
        if cp_data.model_state_dict:
            components.nn.set_weights(cp_data.model_state_dict)
        if cp_data.optimizer_state_dict:
            try:
                components.trainer.load_optimizer_state(
                    cp_data.optimizer_state_dict
                )  # Use helper
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            try:
                set_state_ref = components.stats_collector_actor.set_state.remote(
                    cp_data.stats_collector_state
                )
                ray.get(set_state_ref, timeout=5.0)
                logger.info("StatsCollectorActor state restored.")
            except Exception as e:
                logger.error(
                    f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                )
        training_loop.set_initial_state(
            cp_data.global_step, cp_data.episodes_played, cp_data.total_simulations_run
        )
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    # --- Apply MuZero Buffer Data ---
    if loaded_state.buffer_data:
        logger.info("Loading MuZero buffer data...")
        # Ensure buffer_idx mapping is rebuilt correctly
        components.buffer.buffer.clear()
        components.buffer.tree_idx_to_buffer_idx.clear()
        components.buffer.buffer_idx_to_tree_idx.clear()
        components.buffer.total_steps = 0
        components.buffer.next_buffer_idx = 0
        if components.buffer.use_per and components.buffer.sum_tree:
            components.buffer.sum_tree.reset()  # Reset sumtree

        for _i, traj in enumerate(loaded_state.buffer_data.trajectories):
            # Re-add trajectories to ensure buffer and sumtree are consistent
            components.buffer.add(traj)

        # Verify counts after loading
        logger.info(
            f"MuZero Buffer loaded. Trajectories in deque: {len(components.buffer.buffer)}, "
            f"Total Steps: {components.buffer.total_steps}, "
            f"SumTree Entries: {components.buffer.sum_tree.n_entries if components.buffer.sum_tree else 'N/A'}"
        )
        if training_loop.buffer_fill_progress:
            training_loop.buffer_fill_progress.set_current_steps(len(components.buffer))
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    return training_loop


# --- ADDED: Define _save_final_state ---
def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    if not training_loop:
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_final=True,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


# --- END ADDED ---


def run_training_headless_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline in headless mode."""
    # (Rest of the function remains largely the same)
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    log_file_path = None
    file_handler = None
    ray_initialized_by_setup = False
    mlflow_run_active = False

    try:
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "headless"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
        )

        components, ray_initialized_by_setup = setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        training_loop = _load_and_apply_initial_state(components)
        training_loop.initialize_workers()
        training_loop.run()

        if training_loop.training_complete:
            exit_code = 0
        elif training_loop.training_exception:
            exit_code = 1
        else:
            exit_code = 1  # Consider interrupted as non-zero exit

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during headless training setup or execution: {e}"
        )
        traceback.print_exc()
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            _save_final_state(training_loop)  # Call the defined function
            training_loop.cleanup_actors()
            if training_loop.training_exception:
                final_status = "FAILED"
                error_msg = str(training_loop.training_exception)
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = (
                    "INTERRUPTED"  # Or maybe CANCELLED if stop_requested was set
                )

        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by headless runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        root_logger = get_root_logger()
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Headless training finished with exit code {exit_code}.")
    return exit_code


File: muzerotriangle\training\logging_utils.py
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np

if TYPE_CHECKING:
    from muzerotriangle.config import PersistenceConfig

    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class Tee:
    """Helper class to redirect stdout/stderr to both console and a file."""

    def __init__(self, stream1, stream2, main_stream_for_fileno):
        self.stream1 = stream1
        self.stream2 = stream2
        self._main_stream_for_fileno = main_stream_for_fileno

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

    def fileno(self):
        return self._main_stream_for_fileno.fileno()

    def isatty(self):
        return self._main_stream_for_fileno.isatty()


def get_root_logger() -> logging.Logger:
    """Gets the root logger instance."""
    return logging.getLogger()


def setup_file_logging(
    persist_config: "PersistenceConfig", run_name: str, mode_suffix: str
) -> str:
    """Sets up file logging for the current run."""
    run_base_dir = Path(persist_config.get_run_base_dir(run_name))
    log_dir = run_base_dir / persist_config.LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{run_name}_{mode_suffix}.log"

    file_handler = logging.FileHandler(log_file_path, mode="w")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    root_logger = get_root_logger()
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)
        logger.info(f"Added file handler logging to: {log_file_path}")
    else:
        logger.warning("File handler already exists for root logger.")

    return str(log_file_path)


def log_configs_to_mlflow(components: "TrainingComponents"):
    """Logs configuration parameters to MLflow."""
    if not mlflow.active_run():
        logger.warning("No active MLflow run found. Cannot log configs.")
        return

    logger.info("Logging configuration parameters to MLflow...")
    try:
        mlflow.log_params(components.env_config.model_dump())
        mlflow.log_params(components.model_config.model_dump())
        mlflow.log_params(components.train_config.model_dump())
        mlflow.log_params(components.mcts_config.model_dump())
        mlflow.log_params(components.persist_config.model_dump())
        logger.info("Configuration parameters logged to MLflow.")
    except Exception as e:
        logger.error(f"Failed to log parameters to MLflow: {e}", exc_info=True)


def log_metrics_to_mlflow(metrics: dict[str, Any], step: int):
    """Logs metrics to MLflow."""
    if not mlflow.active_run():
        logger.warning("No active MLflow run found. Cannot log metrics.")
        return

    try:
        # Filter only numeric, finite metrics
        numeric_metrics = {}
        for k, v in metrics.items():
            # Use isinstance with | for multiple types
            if isinstance(v, int | float | np.number) and np.isfinite(v):
                numeric_metrics[k] = float(v)
            else:
                logger.debug(
                    f"Skipping non-numeric or non-finite metric for MLflow: {k}={v} (type: {type(v)})"
                )
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step=step)
            logger.debug(
                f"Logged {len(numeric_metrics)} metrics to MLflow at step {step}."
            )
        else:
            logger.debug(f"No valid numeric metrics to log at step {step}.")
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {e}", exc_info=True)


File: muzerotriangle\training\loop.py
# File: muzerotriangle/training/loop.py
# File: muzerotriangle/training/loop.py
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any  # Import Optional

# Use MuZero types
from ..rl.types import SelfPlayResult
from .loop_helpers import LoopHelpers
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    from ..visualization.ui import ProgressBar
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Manages the core asynchronous MuZero training loop logic.
    Coordinates worker tasks (trajectory generation), buffer (trajectory storage),
    trainer (sequence-based training), and visualization.
    Handles PER sampling and priority updates.
    """

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: (
            queue.Queue[dict[int, Any] | None] | None
        ) = None,  # Make optional
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.train_config = components.train_config

        self.buffer = components.buffer
        self.trainer = components.trainer
        self.data_manager = components.data_manager  # Add data manager

        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.worker_weight_updates_count = 0
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.training_exception: Exception | None = None  # Make optional

        self.train_step_progress: ProgressBar | None = None  # Make optional
        self.buffer_fill_progress: ProgressBar | None = None  # Make optional

        self.worker_manager = WorkerManager(components)
        self.loop_helpers = LoopHelpers(
            components, self.visual_state_queue, self._get_loop_state
        )

        logger.info("MuZero TrainingLoop initialized.")

    def _get_loop_state(self) -> dict[str, Any]:
        """Provides current loop state to helpers."""
        return {
            "global_step": self.global_step,
            "episodes_played": self.episodes_played,
            "total_simulations_run": self.total_simulations_run,
            "worker_weight_updates": self.worker_weight_updates_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "num_active_workers": self.worker_manager.get_num_active_workers(),
            "num_pending_tasks": self.worker_manager.get_num_pending_tasks(),
            "train_progress": self.train_step_progress,
            "buffer_progress": self.buffer_fill_progress,
            "start_time": self.start_time,
            "num_workers": self.train_config.NUM_SELF_PLAY_WORKERS,
        }

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        self.worker_weight_updates_count = (
            global_step // self.train_config.WORKER_UPDATE_FREQ_STEPS
        )
        self.train_step_progress, self.buffer_fill_progress = (
            self.loop_helpers.initialize_progress_bars(
                global_step,
                len(self.buffer),
                self.start_time,
            )
        )
        self.loop_helpers.reset_rate_counters(
            global_step, episodes_played, total_simulations
        )
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}, "
            f"WeightUpdates={self.worker_weight_updates_count}, BufferSteps={len(self.buffer)}"
        )

    def initialize_workers(self):
        """Initializes self-play workers using WorkerManager."""
        self.worker_manager.initialize_workers()

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _process_self_play_result(self, result: SelfPlayResult, worker_id: int):
        """Processes a validated MuZero result (trajectory) from a worker."""
        logger.debug(
            f"Processing result from worker {worker_id} "
            f"(Ep Steps: {result.episode_steps}, Score: {result.final_score:.2f}, "
            f"Trajectory Len: {len(result.trajectory)})"
        )

        if not result.trajectory:
            logger.warning(
                f"Worker {worker_id} returned an empty trajectory. Skipping."
            )
            return

        try:
            self.buffer.add(result.trajectory)
            logger.debug(
                f"Added trajectory (len {len(result.trajectory)}) from worker {worker_id} to buffer. "
                f"Buffer total steps: {len(self.buffer)}"
            )
        except Exception as e:
            logger.error(
                f"Error adding trajectory to buffer from worker {worker_id}: {e}",
                exc_info=True,
            )
            return

        if self.buffer_fill_progress:
            self.buffer_fill_progress.set_current_steps(len(self.buffer))
        self.episodes_played += 1
        self.total_simulations_run += result.total_simulations

    def _run_training_step(self) -> bool:
        """Runs one MuZero training step using sampled sequences."""
        if not self.buffer.is_ready():
            return False

        # --- Sample Sequences (Handles PER internally) ---
        sampled_batch_data = self.buffer.sample(
            self.train_config.BATCH_SIZE,
            current_train_step=self.global_step,  # Pass step for PER beta
        )
        if not sampled_batch_data:
            return False

        # --- Train on Sequences ---
        # Trainer handles PER weights internally if batch_sample is SampledBatchPER
        train_result = self.trainer.train_step(sampled_batch_data)

        if train_result:
            loss_info, td_errors = train_result
            self.global_step += 1
            if self.train_step_progress:
                self.train_step_progress.set_current_steps(self.global_step)

            # --- Update PER Priorities ---
            if (
                self.train_config.USE_PER
                and isinstance(sampled_batch_data, dict)
                and "indices" in sampled_batch_data
            ):
                tree_indices = sampled_batch_data["indices"]
                if len(tree_indices) == len(td_errors):
                    self.buffer.update_priorities(tree_indices, td_errors)
                else:
                    logger.error(
                        f"PER Update Error: Mismatch between tree indices ({len(tree_indices)}) and TD errors ({len(td_errors)})"
                    )

            # Log results
            self.loop_helpers.log_training_results_async(
                loss_info, self.global_step, self.total_simulations_run
            )

            # Check for worker weight update
            if self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS == 0:
                try:
                    self.worker_manager.update_worker_networks(self.global_step)
                    self.worker_weight_updates_count += 1
                    self.loop_helpers.log_weight_update_event(self.global_step)
                except Exception as update_err:
                    logger.error(f"Failed to update worker networks: {update_err}")

            if self.global_step % 100 == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"Loss(T={loss_info['total_loss']:.3f} P={loss_info['policy_loss']:.3f} "
                    f"V={loss_info['value_loss']:.3f} R={loss_info['reward_loss']:.3f}) "
                    f"Ent={loss_info['entropy']:.3f}"
                )
            return True
        else:
            logger.warning(
                f"Training step {self.global_step + 1} failed (Trainer returned None)."
            )
            return False

    def run(self):
        """Main MuZero training loop."""
        max_steps_info = (
            f"Target steps: {self.train_config.MAX_TRAINING_STEPS}"
            if self.train_config.MAX_TRAINING_STEPS is not None
            else "Running indefinitely"
        )
        logger.info(f"Starting MuZero TrainingLoop run... {max_steps_info}")
        self.start_time = time.time()

        try:
            self.worker_manager.submit_initial_tasks()
            last_save_time = time.time()

            while not self.stop_requested.is_set():
                # Check max steps
                if (
                    self.train_config.MAX_TRAINING_STEPS is not None
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached MAX_TRAINING_STEPS ({self.train_config.MAX_TRAINING_STEPS}). Stopping."
                    )
                    self.training_complete = True
                    self.request_stop()
                    break

                # Training Step
                training_happened = False
                if self.buffer.is_ready():
                    training_happened = self._run_training_step()
                else:
                    time.sleep(0.1)

                if self.stop_requested.is_set():
                    break

                # Handle Completed Worker Tasks
                wait_timeout = 0.01 if training_happened else 0.2
                completed_tasks = self.worker_manager.get_completed_tasks(wait_timeout)

                for worker_id, result_or_error in completed_tasks:
                    if isinstance(result_or_error, SelfPlayResult):
                        try:
                            self._process_self_play_result(result_or_error, worker_id)
                        except Exception as proc_err:
                            logger.error(
                                f"Error processing result: {proc_err}", exc_info=True
                            )
                    elif isinstance(result_or_error, Exception):
                        logger.error(
                            f"Worker {worker_id} task failed: {result_or_error}"
                        )
                    else:
                        logger.error(
                            f"Unexpected item from worker {worker_id}: {type(result_or_error)}"
                        )
                    self.worker_manager.submit_task(worker_id)

                if self.stop_requested.is_set():
                    break

                # Periodic Tasks
                self.loop_helpers.update_visual_queue()
                self.loop_helpers.log_progress_eta()
                self.loop_helpers.calculate_and_log_rates()

                # Checkpointing
                current_time = time.time()
                if (
                    self.global_step > 0
                    and self.global_step % self.train_config.CHECKPOINT_SAVE_FREQ_STEPS
                    == 0
                    and current_time - last_save_time > 60
                ):
                    logger.info(f"Saving checkpoint at step {self.global_step}...")
                    self.data_manager.save_training_state(
                        nn=self.components.nn,
                        optimizer=self.trainer.optimizer,
                        stats_collector_actor=self.components.stats_collector_actor,
                        buffer=self.buffer,
                        global_step=self.global_step,
                        episodes_played=self.episodes_played,
                        total_simulations_run=self.total_simulations_run,
                    )
                    last_save_time = current_time

                if not completed_tasks and not training_happened:
                    time.sleep(0.02)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Stopping.")
        except Exception as e:
            logger.critical(f"Unhandled exception in TrainingLoop: {e}", exc_info=True)
            self.training_exception = e
        finally:
            self.request_stop()
            self.training_complete = (
                self.training_complete and self.training_exception is None
            )
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Cleans up worker actors."""
        self.worker_manager.cleanup_actors()


File: muzerotriangle\training\loop_helpers.py
# File: muzerotriangle/training/loop_helpers.py
import logging
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any  # Import Optional

import ray

from ..environment import GameState
from ..stats.plot_definitions import WEIGHT_UPDATE_METRIC_KEY
from ..utils import format_eta

# Import MuZero types
from ..visualization.core import colors  # Keep colors import
from ..visualization.ui import ProgressBar

if TYPE_CHECKING:
    from ..utils.types import StatsCollectorData, StepInfo
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

VISUAL_UPDATE_INTERVAL = 0.2
STATS_FETCH_INTERVAL = 0.5
VIS_STATE_FETCH_TIMEOUT = 0.1
RATE_CALCULATION_INTERVAL = 5.0


class LoopHelpers:
    """Provides helper functions for the MuZero TrainingLoop."""

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None,  # Make optional
        get_loop_state_func: Callable[[], dict[str, Any]],
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.get_loop_state = get_loop_state_func

        self.stats_collector_actor = components.stats_collector_actor
        self.train_config = components.train_config
        self.trainer = components.trainer

        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = 0
        self.last_rate_calc_episodes = 0
        self.last_rate_calc_sims = 0

    def reset_rate_counters(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Resets counters used for rate calculation."""
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = global_step
        self.last_rate_calc_episodes = episodes_played
        self.last_rate_calc_sims = total_simulations

    def initialize_progress_bars(
        self, global_step: int, buffer_total_steps: int, start_time: float
    ) -> tuple[ProgressBar, ProgressBar]:
        """Initializes and returns progress bars."""
        train_total = self.train_config.MAX_TRAINING_STEPS or 1
        # Buffer progress now tracks total steps vs capacity
        buffer_total = self.train_config.BUFFER_CAPACITY

        train_pb = ProgressBar(
            "Training Steps", train_total, start_time, global_step, colors.GREEN
        )
        # Use buffer_total_steps for initial buffer progress
        buffer_pb = ProgressBar(
            "Buffer Steps", buffer_total, start_time, buffer_total_steps, colors.ORANGE
        )
        return train_pb, buffer_pb

    def _fetch_latest_stats(self):
        """Fetches the latest stats data from the actor."""
        current_time = time.time()
        if current_time - self.last_stats_fetch_time < STATS_FETCH_INTERVAL:
            return
        self.last_stats_fetch_time = current_time
        if self.stats_collector_actor:
            try:
                data_ref = self.stats_collector_actor.get_data.remote()  # type: ignore
                self.latest_stats_data = ray.get(data_ref, timeout=1.0)
            except Exception as e:
                logger.warning(f"Failed to fetch latest stats: {e}")

    def calculate_and_log_rates(self):
        """Calculates and logs rates. Buffer size now represents total steps."""
        current_time = time.time()
        time_delta = current_time - self.last_rate_calc_time
        if time_delta < RATE_CALCULATION_INTERVAL:
            return

        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        episodes_played = loop_state["episodes_played"]
        total_simulations = loop_state["total_simulations_run"]
        current_buffer_total_steps = int(
            loop_state["buffer_size"]
        )  # This is total steps

        steps_delta = global_step - self.last_rate_calc_step
        episodes_delta = episodes_played - self.last_rate_calc_episodes
        sims_delta = total_simulations - self.last_rate_calc_sims

        steps_per_sec = steps_delta / time_delta if time_delta > 0 else 0.0
        episodes_per_sec = episodes_delta / time_delta if time_delta > 0 else 0.0
        sims_per_sec = sims_delta / time_delta if time_delta > 0 else 0.0

        if self.stats_collector_actor:
            # Buffer size context uses total steps
            step_info_buffer: StepInfo = {
                "global_step": global_step,
                "buffer_size": current_buffer_total_steps,
            }
            step_info_global: StepInfo = {"global_step": global_step}

            rate_stats: dict[str, tuple[float, StepInfo]] = {
                "Rate/Episodes_Per_Sec": (episodes_per_sec, step_info_buffer),
                "Rate/Simulations_Per_Sec": (sims_per_sec, step_info_buffer),
                "Buffer/Size": (
                    float(current_buffer_total_steps),
                    step_info_buffer,
                ),  # Log total steps
            }
            log_msg_steps = "Steps/s=N/A"
            if steps_delta > 0:
                rate_stats["Rate/Steps_Per_Sec"] = (steps_per_sec, step_info_global)
                log_msg_steps = f"Steps/s={steps_per_sec:.2f}"

            try:
                self.stats_collector_actor.log_batch.remote(rate_stats)  # type: ignore
                logger.debug(
                    f"Logged rates/buffer at step {global_step} / buffer_steps {current_buffer_total_steps}: "
                    f"{log_msg_steps}, Eps/s={episodes_per_sec:.2f}, Sims/s={sims_per_sec:.1f}, "
                    f"BufferSteps={current_buffer_total_steps}"
                )
            except Exception as e:
                logger.error(f"Failed to log rate/buffer stats: {e}")

        self.reset_rate_counters(global_step, episodes_played, total_simulations)

    def log_progress_eta(self):
        """Logs progress and ETA based on training steps."""
        # (No change needed here, uses training steps)
        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        train_progress = loop_state["train_progress"]
        if global_step == 0 or global_step % 100 != 0 or not train_progress:
            return

        elapsed_time = time.time() - loop_state["start_time"]
        steps_since_load = global_step - train_progress.initial_steps
        steps_per_sec = 0.0
        self._fetch_latest_stats()
        rate_dq = self.latest_stats_data.get("Rate/Steps_Per_Sec")
        if rate_dq:
            steps_per_sec = rate_dq[-1][1]
        elif elapsed_time > 1 and steps_since_load > 0:
            steps_per_sec = steps_since_load / elapsed_time

        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = f"{target_steps:,}" if target_steps else "Infinite"
        progress_str = f"Step {global_step:,}/{target_steps_str}"
        eta_str = format_eta(train_progress.get_eta_seconds())
        buffer_total_steps = loop_state["buffer_size"]
        buffer_capacity = loop_state["buffer_capacity"]
        buffer_fill_perc = (
            (buffer_total_steps / buffer_capacity) * 100 if buffer_capacity > 0 else 0.0
        )
        total_sims = loop_state["total_simulations_run"]
        total_sims_str = (
            f"{total_sims / 1e6:.2f}M"
            if total_sims >= 1e6
            else (f"{total_sims / 1e3:.1f}k" if total_sims >= 1000 else str(total_sims))
        )
        num_pending = loop_state["num_pending_tasks"]

        logger.info(
            f"Progress: {progress_str}, Episodes: {loop_state['episodes_played']:,}, Total Sims: {total_sims_str}, "
            f"Buffer Steps: {buffer_total_steps:,}/{buffer_capacity:,} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending}, Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def update_visual_queue(self):
        """Fetches latest states/stats and puts them onto the visual queue."""
        # (No change needed here, passes loop state and stats data)
        if not self.visual_state_queue or not self.stats_collector_actor:
            return
        current_time = time.time()
        if current_time - self.last_visual_update_time < VISUAL_UPDATE_INTERVAL:
            return
        self.last_visual_update_time = current_time

        latest_worker_states: dict[int, GameState] = {}
        try:
            states_ref = self.stats_collector_actor.get_latest_worker_states.remote()  # type: ignore
            latest_worker_states = ray.get(states_ref, timeout=VIS_STATE_FETCH_TIMEOUT)
            if not isinstance(latest_worker_states, dict):
                latest_worker_states = {}
        except Exception as e:
            logger.warning(f"Failed to fetch worker states: {e}")
            latest_worker_states = {}

        self._fetch_latest_stats()

        visual_data: dict[int, Any] = {}
        for worker_id, state in latest_worker_states.items():
            if isinstance(state, GameState):
                visual_data[worker_id] = state

        visual_data[-1] = {
            **self.get_loop_state(),
            "stats_data": self.latest_stats_data,
        }

        if not visual_data or len(visual_data) == 1:
            return  # Only global data

        try:
            while self.visual_state_queue.qsize() > 2:
                self.visual_state_queue.get_nowait()
            self.visual_state_queue.put_nowait(visual_data)
        except queue.Full:
            logger.warning("Visual queue full.")
        except Exception as qe:
            logger.error(f"Error putting in visual queue: {qe}")

    # --- REMOVED: validate_experiences (handled by SelfPlayResult) ---

    def log_training_results_async(
        self, loss_info: dict[str, float], global_step: int, total_simulations: int
    ) -> None:
        """Logs MuZero training results asynchronously."""
        current_lr = self.trainer.get_current_lr()
        loop_state = self.get_loop_state()
        buffer_total_steps = int(loop_state["buffer_size"])  # Use total steps

        if self.stats_collector_actor:
            step_info: StepInfo = {
                "global_step": global_step,
                "buffer_size": buffer_total_steps,
            }
            stats_batch: dict[str, tuple[float, StepInfo]] = {
                # --- CHANGED: Log MuZero Losses ---
                "Loss/Total": (loss_info["total_loss"], step_info),
                "Loss/Policy": (loss_info["policy_loss"], step_info),
                "Loss/Value": (loss_info["value_loss"], step_info),
                "Loss/Reward": (loss_info["reward_loss"], step_info),  # Add reward loss
                # --- END CHANGED ---
                "Loss/Entropy": (
                    loss_info.get("entropy", 0.0),
                    step_info,
                ),  # Keep entropy if calculated
                "LearningRate": (current_lr, step_info),
                "Progress/Total_Simulations": (float(total_simulations), step_info),
            }
            # --- REMOVED: PER Beta ---
            # if per_beta is not None: stats_batch["PER/Beta"] = (per_beta, step_info)
            try:
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

    def log_weight_update_event(self, global_step: int) -> None:
        """Logs the event of a worker weight update."""
        # (No change needed here)
        if self.stats_collector_actor:
            try:
                step_info: StepInfo = {"global_step": global_step}
                update_metric = {WEIGHT_UPDATE_METRIC_KEY: (1.0, step_info)}
                self.stats_collector_actor.log_batch.remote(update_metric)  # type: ignore
                logger.info(f"Logged worker weight update event at step {global_step}.")
            except Exception as e:
                logger.error(f"Failed to log weight update event: {e}")


File: muzerotriangle\training\README.md
# File: muzerotriangle/training/README.md
# Training Module (`muzerotriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the MuZero reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **[`setup.py`](setup.py):** Contains `setup_training_components` which initializes Ray, detects resources, adjusts worker counts, loads configurations, and creates the core components bundle (`TrainingComponents`).
-   **[`components.py`](components.py):** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **[`loop.py`](loop.py):** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting and collecting results from self-play tasks.
    -   Adding experiences to the `ExperienceBuffer`.
    -   Triggering training steps on the `Trainer`.
    -   Updating worker network weights periodically, passing the current `global_step` to the workers, and logging a special event (`Internal/Weight_Update_Step`) with the `global_step` to the `StatsCollectorActor` when updates occur.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **[`worker_manager.py`](worker_manager.py):** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results from the `SelfPlayWorker` actors. Passes `global_step` to workers during weight updates.
-   **[`loop_helpers.py`](loop_helpers.py):** Contains helper functions used by `TrainingLoop` for tasks like logging rates, updating the visual queue, validating experiences, and logging results. Constructs `StepInfo` dictionary containing relevant step counters (`global_step`, `buffer_size`) for logging. It also includes logic to log the weight update event.
-   **[`runners.py`](runners.py):** Re-exports the main entry point functions (`run_training_headless_mode`, `run_training_visual_mode`) from their respective modules.
-   **[`headless_runner.py`](headless_runner.py) / [`visual_runner.py`](visual_runner.py):** Contain the top-level logic for running training in either headless or visual mode. They handle argument parsing (via CLI), setup logging, call `setup_training_components`, load initial state, run the `TrainingLoop` (potentially in a separate thread for visual mode), handle visualization setup (visual mode), and manage overall cleanup (MLflow, Ray shutdown).
-   **[`logging_utils.py`](logging_utils.py):** Contains helper functions for setting up file logging, redirecting output (`Tee` class), and logging configurations/metrics to MLflow.

This structure separates the high-level setup/teardown (`headless_runner`/`visual_runner`) from the core iterative logic (`loop`), making the system more modular and potentially easier to test or modify.

## Exposed Interfaces

-   **Classes:**
    -   `TrainingLoop`: Contains the core async loop logic.
    -   `TrainingComponents`: Dataclass holding initialized components.
    -   `WorkerManager`: Manages worker actors.
    -   `LoopHelpers`: Provides helper functions for the loop.
-   **Functions (from `runners.py`):**
    -   `run_training_headless_mode(...) -> int`
    -   `run_training_visual_mode(...) -> int`
-   **Functions (from `setup.py`):**
    -   `setup_training_components(...) -> Tuple[Optional[TrainingComponents], bool]`
-   **Functions (from `logging_utils.py`):**
    -   `setup_file_logging(...) -> str`
    -   `get_root_logger() -> logging.Logger`
    -   `Tee` class
    -   `log_configs_to_mlflow(...)`

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: All configuration classes.
-   **[`muzerotriangle.nn`](../nn/README.md)**: `NeuralNetwork`.
-   **[`muzerotriangle.rl`](../rl/README.md)**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult`.
-   **[`muzerotriangle.data`](../data/README.md)**: `DataManager`, `LoadedTrainingState`.
-   **[`muzerotriangle.stats`](../stats/README.md)**: `StatsCollectorActor`, `PlotDefinitions`.
-   **[`muzerotriangle.environment`](../environment/README.md)**: `GameState`.
-   **[`muzerotriangle.utils`](../utils/README.md)**: Helper functions and types (including `StepInfo`).
-   **[`muzerotriangle.visualization`](../visualization/README.md)**: `ProgressBar`, `DashboardRenderer`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `typing`, `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`, `sys`, `traceback`, `pathlib`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding the logging of step context information (`StepInfo`) and worker weight updates.

File: muzerotriangle\training\runners.py
# File: muzerotriangle/training/runners.py
"""
Entry points for running training modes.
Imports functions from specific runner modules.
"""

from .headless_runner import run_training_headless_mode
from .visual_runner import run_training_visual_mode

__all__ = [
    "run_training_headless_mode",
    "run_training_visual_mode",
]


File: muzerotriangle\training\setup.py
# File: muzerotriangle/training/setup.py
import logging
from typing import TYPE_CHECKING

import ray
import torch

from .. import config, utils
from ..data import DataManager
from ..nn import NeuralNetwork
from ..rl import ExperienceBuffer, Trainer
from ..stats import StatsCollectorActor
from .components import TrainingComponents

if TYPE_CHECKING:
    from ..config import PersistenceConfig, TrainConfig

logger = logging.getLogger(__name__)


def setup_training_components(
    train_config_override: "TrainConfig",
    persist_config_override: "PersistenceConfig",
) -> tuple[TrainingComponents | None, bool]:
    """
    Initializes Ray (if not already initialized), detects cores, updates config,
    and returns the TrainingComponents bundle and a flag indicating if Ray was initialized here.
    Adjusts worker count based on detected cores.
    """
    ray_initialized_here = False
    detected_cpu_cores: int | None = None

    try:
        # --- Initialize Ray (if needed) and Detect Cores ---
        if not ray.is_initialized():
            try:
                # Attempt initialization
                ray.init(logging_level=logging.WARNING, log_to_driver=True)
                ray_initialized_here = True
                logger.info("Ray initialized by setup_training_components.")
            except Exception as e:
                # Log critical error and re-raise to stop setup
                logger.critical(f"Failed to initialize Ray: {e}", exc_info=True)
                raise RuntimeError("Ray initialization failed") from e
        else:
            logger.info("Ray already initialized.")
            # Ensure flag is False if Ray was already running
            ray_initialized_here = False

        # --- Detect Cores (proceed even if Ray was already initialized) ---
        try:
            resources = ray.cluster_resources()
            detected_cpu_cores = int(resources.get("CPU", 0)) - 2
            logger.info(f"Ray detected {detected_cpu_cores} CPU cores.")
        except Exception as e:
            logger.error(f"Could not get Ray cluster resources: {e}")
            # Continue without detected cores, will use configured value

        # --- Initialize Configurations ---
        train_config = train_config_override
        persist_config = persist_config_override
        env_config = config.EnvConfig()
        model_config = config.ModelConfig()
        mcts_config = config.MCTSConfig()

        # --- Adjust Worker Count based on Detected Cores ---
        requested_workers = train_config.NUM_SELF_PLAY_WORKERS
        actual_workers = requested_workers  # Start with configured value

        if detected_cpu_cores is not None and detected_cpu_cores > 0:
            # --- CHANGED: Prioritize detected cores ---
            actual_workers = detected_cpu_cores  # Use detected cores
            if actual_workers != requested_workers:
                logger.info(
                    f"Overriding configured workers ({requested_workers}) with detected CPU cores ({actual_workers})."
                )
            else:
                logger.info(
                    f"Using {actual_workers} self-play workers (matches detected cores)."
                )
            # --- END CHANGED ---
        else:
            logger.warning(
                f"Could not detect valid CPU cores ({detected_cpu_cores}). Using configured NUM_SELF_PLAY_WORKERS: {requested_workers}"
            )
            actual_workers = requested_workers  # Fallback to configured value

        # Update the config object with the final determined number
        train_config.NUM_SELF_PLAY_WORKERS = actual_workers
        logger.info(f"Final worker count set to: {train_config.NUM_SELF_PLAY_WORKERS}")

        # --- Validate Configs ---
        config.print_config_info_and_validate(mcts_config)

        # --- Setup ---
        utils.set_random_seeds(train_config.RANDOM_SEED)
        device = utils.get_device(train_config.DEVICE)
        worker_device = utils.get_device(train_config.WORKER_DEVICE)
        logger.info(f"Determined Training Device: {device}")
        logger.info(f"Determined Worker Device: {worker_device}")
        logger.info(f"Model Compilation Enabled: {train_config.COMPILE_MODEL}")

        # --- Initialize Core Components ---
        stats_collector_actor = StatsCollectorActor.remote(max_history=500_000)  # type: ignore
        logger.info("Initialized StatsCollectorActor with max_history=500k.")
        neural_net = NeuralNetwork(model_config, env_config, train_config, device)
        buffer = ExperienceBuffer(train_config)
        trainer = Trainer(neural_net, train_config, env_config)
        data_manager = DataManager(persist_config, train_config)

        # --- Create Components Bundle ---
        components = TrainingComponents(
            nn=neural_net,
            buffer=buffer,
            trainer=trainer,
            data_manager=data_manager,
            stats_collector_actor=stats_collector_actor,
            train_config=train_config,
            env_config=env_config,
            model_config=model_config,
            mcts_config=mcts_config,
            persist_config=persist_config,
        )
        # Return components and the flag indicating if Ray was initialized *by this function*
        return components, ray_initialized_here
    except Exception as e:
        logger.critical(f"Error setting up training components: {e}", exc_info=True)
        # Return None and the Ray init flag (which might be True if init succeeded before error)
        return None, ray_initialized_here


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Counts total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


File: muzerotriangle\training\visual_runner.py
# File: muzerotriangle/training/visual_runner.py
import logging
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import mlflow
import pygame
import ray

from .. import config, environment, visualization
from ..config import APP_NAME, PersistenceConfig, TrainConfig

# Import Trajectory type
from .components import TrainingComponents
from .logging_utils import (
    Tee,
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)

# Queue for communication between training thread and visualization thread
visual_state_queue: queue.Queue[dict[int, Any] | None] = queue.Queue(maxsize=5)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    # (No changes needed here)
    try:
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
        logger.info(f"Set MLflow experiment to: {APP_NAME}")
        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
            return True
        else:
            logger.error("MLflow run failed to start.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        return False


def _load_and_apply_initial_state(components: TrainingComponents) -> TrainingLoop:
    """Loads initial state (checkpoint, MuZero buffer) and applies it."""
    loaded_state = components.data_manager.load_initial_state()
    # Pass the visual queue to the TrainingLoop
    training_loop = TrainingLoop(components, visual_state_queue=visual_state_queue)

    # --- Apply Checkpoint Data (No change needed here) ---
    if loaded_state.checkpoint_data:
        cp_data = loaded_state.checkpoint_data
        logger.info(
            f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
        )
        if cp_data.model_state_dict:
            components.nn.set_weights(cp_data.model_state_dict)
        if cp_data.optimizer_state_dict:
            try:
                components.trainer.load_optimizer_state(
                    cp_data.optimizer_state_dict
                )  # Use helper
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            try:
                set_state_ref = components.stats_collector_actor.set_state.remote(
                    cp_data.stats_collector_state
                )
                ray.get(set_state_ref, timeout=5.0)
                logger.info("StatsCollectorActor state restored.")
            except Exception as e:
                logger.error(
                    f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                )
        training_loop.set_initial_state(
            cp_data.global_step, cp_data.episodes_played, cp_data.total_simulations_run
        )
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    # --- Apply MuZero Buffer Data ---
    if loaded_state.buffer_data:
        logger.info("Loading MuZero buffer data...")
        # Ensure buffer_idx mapping is rebuilt correctly
        components.buffer.buffer.clear()
        components.buffer.tree_idx_to_buffer_idx.clear()
        components.buffer.buffer_idx_to_tree_idx.clear()
        components.buffer.total_steps = 0
        components.buffer.next_buffer_idx = 0
        if components.buffer.use_per and components.buffer.sum_tree:
            components.buffer.sum_tree.reset()  # Reset sumtree

        for _i, traj in enumerate(loaded_state.buffer_data.trajectories):
            # Re-add trajectories to ensure buffer and sumtree are consistent
            components.buffer.add(traj)

        # Verify counts after loading
        logger.info(
            f"MuZero Buffer loaded. Trajectories in deque: {len(components.buffer.buffer)}, "
            f"Total Steps: {components.buffer.total_steps}, "
            f"SumTree Entries: {components.buffer.sum_tree.n_entries if components.buffer.sum_tree else 'N/A'}"
        )
        if training_loop.buffer_fill_progress:
            training_loop.buffer_fill_progress.set_current_steps(len(components.buffer))
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    return training_loop


def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    # (No changes needed here, uses DataManager)
    if not training_loop:
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_final=True,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


# --- ADDED: Define _training_loop_thread_func ---
def _training_loop_thread_func(training_loop: TrainingLoop):
    """Target function for the training loop thread."""
    try:
        training_loop.initialize_workers()
        training_loop.run()
    except Exception as e:
        logger.critical(f"Exception in training loop thread: {e}", exc_info=True)
        training_loop.training_exception = e  # Store exception
    finally:
        # Signal visualization thread to exit by putting None in the queue
        if training_loop.visual_state_queue:
            try:
                training_loop.visual_state_queue.put_nowait(None)
            except queue.Full:
                logger.warning("Visual queue full when trying to send exit signal.")
        logger.info("Training loop thread finished.")


# --- END ADDED ---


def run_training_visual_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline with visualization."""
    main_thread_exception = None
    train_thread = None
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    file_handler = None
    tee_stdout = None
    tee_stderr = None
    ray_initialized_by_setup = False
    mlflow_run_active = False
    total_params: int | None = None
    trainable_params: int | None = None

    try:
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "visual"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
        )

        # Redirect stdout/stderr to also go to the log file
        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if file_handler and hasattr(file_handler, "stream") and file_handler.stream:
            tee_stdout = Tee(
                original_stdout,
                file_handler.stream,
                main_stream_for_fileno=original_stdout,
            )
            tee_stderr = Tee(
                original_stderr,
                file_handler.stream,
                main_stream_for_fileno=original_stderr,
            )
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            print("--- Stdout/Stderr redirected to console and log file ---")
            logger.info("Stdout/Stderr redirected to console and log file.")
        else:
            logger.error(
                "Could not redirect stdout/stderr: File handler stream not available."
            )

        components, ray_initialized_by_setup = setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        training_loop = _load_and_apply_initial_state(components)

        # Start the training loop in a separate thread
        train_thread = threading.Thread(
            target=_training_loop_thread_func, args=(training_loop,), daemon=True
        )
        train_thread.start()
        logger.info("Training loop thread launched.")

        # Initialize Pygame and the DashboardRenderer
        vis_config = config.VisConfig()
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode(
            (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"{config.APP_NAME} - Training ({components.train_config.RUN_NAME})"
        )
        clock = pygame.time.Clock()
        fonts = visualization.load_fonts()
        dashboard_renderer = visualization.DashboardRenderer(
            screen,
            vis_config,
            components.env_config,
            fonts,
            components.stats_collector_actor,
            components.model_config,
            total_params=total_params,
            trainable_params=trainable_params,
        )

        # Main visualization loop
        running = True
        # --- ADDED: Initialize loop variables ---
        current_worker_states: dict[int, environment.GameState] = {}
        current_global_stats: dict[str, Any] = {}
        # --- END ADDED ---
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.VIDEORESIZE:
                    try:
                        w, h = max(640, event.w), max(240, event.h)
                        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        dashboard_renderer.screen = screen
                        dashboard_renderer.layout_rects = None  # Force recalculation
                    except pygame.error as e:
                        logger.error(f"Error resizing window: {e}")

            # Check if the training thread is still alive
            if train_thread and not train_thread.is_alive():
                logger.warning("Training loop thread terminated.")
                running = False
                if training_loop and training_loop.training_exception:
                    logger.error(
                        f"Training thread terminated due to exception: {training_loop.training_exception}"
                    )
                    main_thread_exception = training_loop.training_exception

            # Get data from the queue (non-blocking)
            try:
                visual_data = visual_state_queue.get_nowait()
                if visual_data is None:  # Signal to exit
                    running = False
                    logger.info("Received exit signal from training thread.")
                    continue
                # Update local copies for rendering
                current_global_stats = visual_data.pop(-1, {})
                current_worker_states = visual_data
            except queue.Empty:
                # No new data, just redraw with existing data
                pass
            except Exception as e:
                logger.error(f"Error getting data from visual queue: {e}")
                time.sleep(0.1)  # Avoid busy-waiting on error
                continue

            # Render the dashboard
            screen.fill(visualization.colors.DARK_GRAY)
            dashboard_renderer.render(current_worker_states, current_global_stats)
            pygame.display.flip()
            clock.tick(vis_config.FPS)

        # Signal the training loop to stop if it hasn't already
        if training_loop and not training_loop.stop_requested.is_set():
            training_loop.request_stop()

        # Wait for the training thread to finish
        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training thread to finish...")
            train_thread.join(timeout=10)  # Wait up to 10 seconds
            if train_thread.is_alive():
                logger.warning("Training thread did not terminate gracefully.")

        # Determine exit code based on exceptions
        if main_thread_exception or training_loop and training_loop.training_exception:
            exit_code = 1
        elif training_loop and training_loop.training_complete:
            exit_code = 0
        else:
            exit_code = 1  # Interrupted or other issue

    except Exception as e:
        logger.critical(f"An unhandled error occurred in visual training script: {e}")
        traceback.print_exc()
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            # Ensure final state is saved even if visualization loop crashed
            if (
                not training_loop.training_complete
                and not training_loop.training_exception
            ):
                _save_final_state(training_loop)

            training_loop.cleanup_actors()  # Ensure actors are cleaned up

            if main_thread_exception:
                final_status = "VIS_FAILED"
                error_msg = str(main_thread_exception)
            elif training_loop.training_exception:
                final_status = "TRAIN_FAILED"
                error_msg = str(training_loop.training_exception)
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"

        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by visual runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        pygame.quit()

        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                get_root_logger().removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Visual training finished with exit code {exit_code}.")
    return exit_code


File: muzerotriangle\training\worker_manager.py
# File: muzerotriangle/training/worker_manager.py
import contextlib  # Import contextlib
import logging
from typing import TYPE_CHECKING

import ray
from pydantic import ValidationError

from ..rl.self_play.worker import SelfPlayWorker
from ..rl.types import SelfPlayResult

if TYPE_CHECKING:
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages the pool of SelfPlayWorker actors, task submission, and results."""

    def __init__(self, components: "TrainingComponents"):
        self.components = components
        self.train_config = components.train_config
        self.nn = components.nn
        self.stats_collector_actor = components.stats_collector_actor
        self.workers: list[ray.actor.ActorHandle | None] = []
        self.worker_tasks: dict[ray.ObjectRef, int] = {}
        self.active_worker_indices: set[int] = set()

    def initialize_workers(self):
        num_workers = self.train_config.NUM_SELF_PLAY_WORKERS
        logger.info(f"Initializing {num_workers} workers...")
        initial_weights = self.nn.get_weights()
        weights_ref = ray.put(initial_weights)
        self.workers = [None] * num_workers
        for i in range(num_workers):
            try:
                worker = SelfPlayWorker.options(num_cpus=1).remote(
                    actor_id=i,
                    env_config=self.components.env_config,
                    mcts_config=self.components.mcts_config,
                    model_config=self.components.model_config,
                    train_config=self.train_config,
                    stats_collector_actor=self.stats_collector_actor,
                    initial_weights=weights_ref,
                    seed=self.train_config.RANDOM_SEED + i,
                    worker_device_str=self.train_config.WORKER_DEVICE,
                )
                self.workers[i] = worker
                self.active_worker_indices.add(i)
            except Exception as e:
                logger.error(f"Failed init worker {i}: {e}", exc_info=True)
        logger.info(f"Initialized {len(self.active_worker_indices)} active workers.")
        del weights_ref

    def submit_initial_tasks(self):
        logger.info("Submitting initial tasks...")
        for i in self.active_worker_indices:
            self.submit_task(i)

    def submit_task(self, worker_idx: int):
        if worker_idx not in self.active_worker_indices:
            return
        worker = self.workers[worker_idx]
        if worker:
            try:
                task_ref = worker.run_episode.remote()
                self.worker_tasks[task_ref] = worker_idx
                logger.debug(f"Submitted task to worker {worker_idx}")
            except Exception as e:
                logger.error(
                    f"Failed submit task worker {worker_idx}: {e}", exc_info=True
                )
                self.active_worker_indices.discard(worker_idx)
                self.workers[worker_idx] = None
        else:
            logger.error(f"Worker {worker_idx} None during submit.")
            self.active_worker_indices.discard(worker_idx)

    def get_completed_tasks(
        self, timeout: float = 0.1
    ) -> list[tuple[int, SelfPlayResult | Exception]]:
        completed_results: list[tuple[int, SelfPlayResult | Exception]] = []
        if not self.worker_tasks:
            return completed_results
        ready_refs, _ = ray.wait(
            list(self.worker_tasks.keys()), num_returns=1, timeout=timeout
        )
        if not ready_refs:
            return completed_results
        for ref in ready_refs:
            worker_idx = self.worker_tasks.pop(ref, -1)
            if worker_idx == -1 or worker_idx not in self.active_worker_indices:
                continue
            try:
                result_raw = ray.get(ref)
                result_validated = SelfPlayResult.model_validate(result_raw)
                completed_results.append((worker_idx, result_validated))
            except ValidationError as e_val:
                error_msg = f"Pydantic validation failed worker {worker_idx}: {e_val}"
                logger.error(error_msg, exc_info=False)
                completed_results.append((worker_idx, ValueError(error_msg)))
            except ray.exceptions.RayActorError as e_actor:
                logger.error(
                    f"Worker {worker_idx} actor failed: {e_actor}", exc_info=True
                )
                completed_results.append((worker_idx, e_actor))
                self.workers[worker_idx] = None
                self.active_worker_indices.discard(worker_idx)
            except Exception as e_get:
                logger.error(
                    f"Error getting result worker {worker_idx}: {e_get}", exc_info=True
                )
                completed_results.append((worker_idx, e_get))
        return completed_results

    def update_worker_networks(self, global_step: int):
        active_workers = [
            w
            for i, w in enumerate(self.workers)
            if i in self.active_worker_indices and w is not None
        ]
        if not active_workers:
            return
        logger.debug(f"Updating worker networks for step {global_step}...")
        current_weights = self.nn.get_weights()
        weights_ref = ray.put(current_weights)
        set_weights_tasks = [
            worker.set_weights.remote(weights_ref) for worker in active_workers
        ]
        set_step_tasks = [
            worker.set_current_trainer_step.remote(global_step)
            for worker in active_workers
        ]
        all_tasks = set_weights_tasks + set_step_tasks
        if not all_tasks:
            del weights_ref
            return
        try:
            ray.get(all_tasks, timeout=120.0)
            logger.debug(
                f"Worker networks updated for {len(active_workers)} workers to step {global_step}."
            )
        except ray.exceptions.RayActorError as e:
            logger.error(f"Worker failed during update: {e}", exc_info=True)
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout updating workers.")
        except Exception as e:
            logger.error(f"Error updating workers: {e}", exc_info=True)
        finally:
            del weights_ref

    def get_num_active_workers(self) -> int:
        return len(self.active_worker_indices)

    def get_num_pending_tasks(self) -> int:
        return len(self.worker_tasks)

    def cleanup_actors(self):
        """Kills Ray actors associated with this manager."""
        logger.info("Cleaning up WorkerManager actors...")
        for task_ref in list(self.worker_tasks.keys()):
            with contextlib.suppress(Exception):
                ray.cancel(task_ref, force=True)
        self.worker_tasks = {}
        for i, worker in enumerate(self.workers):
            if worker:
                with contextlib.suppress(Exception):
                    ray.kill(worker, no_restart=True)
                    logger.debug(f"Killed worker {i}.")
        self.workers = []
        self.active_worker_indices = set()
        logger.info("WorkerManager actors cleaned up.")


File: muzerotriangle\training\__init__.py
# File: muzerotriangle/training/__init__.py
"""
Training module containing the pipeline, loop, components, and utilities
for orchestrating the reinforcement learning training process.
"""

# Core components & classes
from .components import TrainingComponents

# Utilities
from .logging_utils import Tee, get_root_logger, setup_file_logging
from .loop import TrainingLoop
from .loop_helpers import LoopHelpers

# Re-export runner functions
from .runners import (
    run_training_headless_mode,
    run_training_visual_mode,
)
from .setup import setup_training_components

# from .pipeline import TrainingPipeline # REMOVED
from .worker_manager import WorkerManager

# Explicitly define __all__
__all__ = [
    # Core Components
    "TrainingComponents",
    "TrainingLoop",
    # "TrainingPipeline", # REMOVED
    # Helpers & Managers
    "WorkerManager",
    "LoopHelpers",
    "setup_training_components",
    # Runners (re-exported)
    "run_training_headless_mode",
    "run_training_visual_mode",
    # Logging Utilities
    "setup_file_logging",
    "get_root_logger",
    "Tee",
]


File: muzerotriangle\utils\geometry.py
def is_point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    """
    Checks if a point is inside a polygon using the ray casting algorithm.

    Args:
        point: Tuple (x, y) representing the point coordinates.
        polygon: List of tuples [(x1, y1), (x2, y2), ...] representing polygon vertices in order.

    Returns:
        True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        # Combine nested if statements
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            # Use ternary operator for xinters calculation
            xinters = ((y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x) if p1y != p2y else x

            # Check if point is on the segment boundary or crosses the ray
            if abs(p1x - p2x) < 1e-9:  # Vertical line segment
                if abs(x - p1x) < 1e-9:
                    return True  # Point is on the vertical segment
            elif abs(x - xinters) < 1e-9:  # Point is exactly on the intersection
                return True  # Point is on the boundary
            elif (
                p1x == p2x or x <= xinters
            ):  # Point is to the left or on a non-horizontal segment
                inside = not inside

        p1x, p1y = p2x, p2y

    # Check if the point is exactly one of the vertices
    for px, py in polygon:
        if abs(x - px) < 1e-9 and abs(y - py) < 1e-9:
            return True

    return inside


File: muzerotriangle\utils\helpers.py
# File: muzerotriangle/utils/helpers.py
import logging
import random
from typing import cast

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Gets the appropriate torch device based on preference and availability.
    Prioritizes MPS on Mac if 'auto' is selected.
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device.")
        return torch.device("cuda")
    # --- CHANGED: Prioritize MPS if available and preferred/auto ---
    if (
        device_preference in ["auto", "mps"]
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()  # Check if MPS is built
    ):
        logger.info(f"Using MPS device (Preference: {device_preference}).")
        return torch.device("mps")
    # --- END CHANGED ---
    if device_preference == "cpu":
        logger.info("Using CPU device.")
        return torch.device("cpu")

    # Auto-detection fallback (after MPS check)
    if torch.cuda.is_available():
        logger.info("Auto-selected CUDA device.")
        return torch.device("cuda")
    # Check MPS again in fallback (should have been caught above if available)
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        logger.info("Auto-selected MPS device.")
        return torch.device("mps")

    logger.info("Auto-selected CPU device.")
    return torch.device("cpu")


def set_random_seeds(seed: int = 42):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    # Use NumPy's recommended way to seed the global RNG state if needed,
    # or preferably use a Generator instance. For simplicity here, we seed global.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional: Set deterministic algorithms for CuDNN
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    # --- ADDED: Seed MPS if available ---
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Use torch.mps.manual_seed if available (newer PyTorch versions)
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)  # type: ignore
            else:
                # Fallback for older versions if needed, though less common
                # torch.manual_seed(seed) might cover MPS indirectly in some versions
                pass
        except Exception as e:
            logger.warning(f"Could not set MPS seed: {e}")
    # --- END ADDED ---
    logger.info(f"Set random seeds to {seed}")


def format_eta(seconds: float | None) -> str:
    """Formats seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    if seconds > 3600 * 24 * 30:
        return ">1 month"

    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def normalize_color_for_matplotlib(
    color_tuple_0_255: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        valid_color = tuple(max(0, min(255, c)) for c in color_tuple_0_255)
        return cast("tuple[float, float, float]", tuple(c / 255.0 for c in valid_color))
    logger.warning(
        f"Invalid color format for normalization: {color_tuple_0_255}, returning black."
    )
    return (0.0, 0.0, 0.0)


File: muzerotriangle\utils\README.md
# File: muzerotriangle/utils/README.md
# Utilities Module (`muzerotriangle.utils`)

## Purpose and Architecture

This module provides common utility functions and type definitions used across various parts of the AlphaTriangle project. Its goal is to avoid code duplication and provide central definitions for shared concepts.

-   **Helper Functions ([`helpers.py`](helpers.py)):** Contains miscellaneous helper functions like `get_device`, `set_random_seeds`, `format_eta`, `normalize_color_for_matplotlib`.
-   **Type Definitions ([`types.py`](types.py)):** Defines common type aliases and `TypedDict`s used throughout the codebase. Key types include:
    -   `StateType`: Structure of NN input features.
    -   `ActionType`: Integer representation of actions.
    -   `PolicyTargetMapping`: MCTS policy target.
    -   `PolicyValueOutput`: Output of NN evaluation.
    -   `TrajectoryStep`: Data stored for each step in a trajectory, **including `n_step_reward_target`**.
    -   `Trajectory`: A list of `TrajectoryStep` dicts.
    -   `SampledSequence`: A fixed-length sequence sampled from a `Trajectory`.
    -   `SampledBatch`: A list of `SampledSequence`s (for uniform sampling).
    -   `SampledBatchPER`: A `TypedDict` including sequences, SumTree indices, and IS weights (for PER sampling).
    -   `StatsCollectorData`: Structure for storing collected statistics.
    -   `StepInfo`: Contextual information for statistics logging.
-   **Geometry Utilities ([`geometry.py`](geometry.py)):** Contains geometric helper functions like `is_point_in_polygon`.
-   **Data Structures ([`sumtree.py`](sumtree.py)):**
    -   `SumTree`: A SumTree implementation used for Prioritized Experience Replay. Stores pre-calculated priorities and associated data. **Correctly tracks the number of entries (`n_entries`), updates max priority, and handles leaf retrieval proportionally to stored priorities, including edge cases.**

## Exposed Interfaces

-   **Functions:** `get_device`, `set_random_seeds`, `format_eta`, `normalize_color_for_matplotlib`, `is_point_in_polygon`.
-   **Classes:** `SumTree`.
-   **Types:** `StateType`, `ActionType`, `PolicyTargetMapping`, `PolicyValueOutput`, `TrajectoryStep`, `Trajectory`, `SampledSequence`, `SampledBatch`, `SampledBatchPER`, `StatsCollectorData`, `StepInfo`.

## Dependencies

-   **`torch`**: Used by `get_device` and `set_random_seeds`.
-   **`numpy`**: Used by `set_random_seeds`, `SumTree`, and in type definitions.
-   **Standard Libraries:** `typing`, `random`, `os`, `math`, `logging`, `collections.deque`.

---

**Note:** Please keep this README updated when adding or modifying utility functions or type definitions, especially those related to MuZero data structures (like `TrajectoryStep`) or PER.

File: muzerotriangle\utils\sumtree.py
# File: muzerotriangle/utils/sumtree.py
import logging
import sys

import numpy as np

# Use a dedicated logger for SumTree internal debugging
sumtree_logger = logging.getLogger("muzerotriangle.utils.sumtree_internal")
sumtree_logger.setLevel(logging.WARNING)  # Default level
if not sumtree_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    sumtree_logger.addHandler(handler)


class SumTree:
    """
    Simple SumTree implementation for efficient prioritized sampling.
    Stores priorities and allows sampling proportional to priority.
    Handles circular buffer logic for data storage using a Python list.
    Uses internal capacity padding to the next power of 2 for simplified tree structure.
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

        # User-facing capacity
        self.capacity = capacity
        # Internal capacity (power of 2) for the tree structure
        self._internal_capacity = 1
        while self._internal_capacity < capacity:
            self._internal_capacity *= 2

        # Tree size based on internal capacity
        self.tree = np.zeros(2 * self._internal_capacity - 1)
        # Data storage size based on internal capacity (though only user capacity is used)
        self.data: list[object | None] = [None] * self._internal_capacity
        # data_pointer points to the next index to write to in self.data (wraps around user capacity)
        self.data_pointer = 0
        # n_entries tracks the number of valid entries (up to user capacity)
        self.n_entries = 0
        # _max_priority tracks the maximum priority ever added/updated
        self._max_priority = 0.0
        sumtree_logger.debug(
            f"SumTree initialized with user_capacity={capacity}, internal_capacity={self._internal_capacity}"
        )

    def reset(self):
        """Resets the tree and data."""
        self.tree.fill(0.0)
        # Recreate the data list based on internal capacity
        self.data = [None] * self._internal_capacity
        self.data_pointer = 0
        self.n_entries = 0
        self._max_priority = 0.0
        sumtree_logger.debug("SumTree reset.")

    def _propagate(self, tree_idx: int, change: float):
        """Propagates priority change up the tree."""
        parent = (tree_idx - 1) // 2
        if parent < 0:
            return
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _update_leaf(self, tree_idx: int, priority: float):
        """Updates a leaf node and propagates the change."""
        if not (
            self._internal_capacity - 1 <= tree_idx < 2 * self._internal_capacity - 1
        ):
            msg = f"Invalid tree_idx {tree_idx} for leaf update. InternalCapacity={self._internal_capacity}"
            sumtree_logger.error(msg)
            raise IndexError(msg)

        if priority < 0 or not np.isfinite(priority):
            priority = 0.0

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self._max_priority = max(self._max_priority, priority)

    def add(self, priority: float, data: object) -> int:
        """Adds data with a given priority. Returns the tree index."""
        if self.capacity == 0:  # Check user capacity
            raise ValueError("Cannot add to a SumTree with zero capacity.")

        # Calculate tree index based on data_pointer and internal capacity
        tree_idx = self.data_pointer + self._internal_capacity - 1

        # Store data at data_pointer index
        self.data[self.data_pointer] = data
        # Update the corresponding leaf in the tree
        self.update(tree_idx, priority)

        # Update data_pointer, wrapping around the *user* capacity
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        # Increment n_entries up to the *user* capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

        return tree_idx

    def update(self, tree_idx: int, priority: float):
        """Public method to update priority at a given tree index."""
        self._update_leaf(tree_idx, priority)

    # --- CORRECTED Iterative Retrieve with Strict Inequality ---
    def _retrieve(self, tree_idx: int, sample_value: float) -> int:
        """Finds the leaf index for a given sample value using binary search on the tree."""
        current_idx = tree_idx  # Start search from the provided index (usually 0)
        sumtree_logger.debug(
            f"Retrieve START: initial_idx={current_idx}, sample_value={sample_value:.6f}"
        )

        while True:
            left_child_idx = 2 * current_idx + 1
            right_child_idx = left_child_idx + 1
            sumtree_logger.debug(
                f"  Loop: current_idx={current_idx}, sample_value={sample_value:.6f}"
            )

            # If left child index is out of bounds, current_idx is a leaf
            if left_child_idx >= len(self.tree):
                sumtree_logger.debug(
                    f"  Leaf condition met: left_child_idx={left_child_idx} >= tree_len={len(self.tree)}. Returning leaf_idx={current_idx}"
                )
                break

            left_sum = self.tree[left_child_idx]
            sumtree_logger.debug(
                f"    left_child_idx={left_child_idx}, left_sum={left_sum:.6f}"
            )

            # --- Use strict less than comparison ---
            if sample_value < left_sum:
                # --- End change ---
                sumtree_logger.debug(
                    f"    Condition TRUE: {sample_value:.6f} < {left_sum:.6f}. Going LEFT."
                )
                current_idx = left_child_idx
            else:
                sumtree_logger.debug(
                    f"    Condition FALSE: {sample_value:.6f} >= {left_sum:.6f}. Going RIGHT."
                )
                sample_value -= left_sum
                sumtree_logger.debug(f"      Adjusted sample_value={sample_value:.6f}")
                # Ensure right child exists before assigning
                if right_child_idx >= len(self.tree):
                    sumtree_logger.warning(
                        f"      Right child index {right_child_idx} out of bounds! Tree len={len(self.tree)}. Breaking loop at idx={current_idx}."
                    )
                    break
                current_idx = right_child_idx
                sumtree_logger.debug(f"      New current_idx={current_idx}")

        sumtree_logger.debug(f"Retrieve END: Returning leaf_idx={current_idx}")
        return current_idx

    # --- End CORRECTED Iterative Retrieve ---

    def get_leaf(self, value: float) -> tuple[int, float, object]:
        """
        Finds the leaf node index, priority, and associated data for a given sample value.
        """
        total_p = self.total()
        if total_p <= 0:
            raise ValueError(
                f"Cannot sample from SumTree with zero or negative total priority ({total_p}). n_entries: {self.n_entries}"
            )

        # Clamp value to be within [0, total_p) using epsilon
        value = np.clip(value, 0, total_p - 1e-9)

        # Start retrieval from the root (index 0)
        leaf_tree_idx = self._retrieve(0, value)

        # Ensure returned index is actually a leaf index based on internal capacity
        if not (
            self._internal_capacity - 1
            <= leaf_tree_idx
            < 2 * self._internal_capacity - 1
        ):
            sumtree_logger.error(
                f"GetLeaf: _retrieve returned non-leaf index {leaf_tree_idx}. "
                f"InternalCapacity={self._internal_capacity}, Sampled value: {value:.4f}, Total P: {total_p:.4f}."
            )
            # Fallback: Find the leftmost leaf based on internal capacity
            leaf_tree_idx = self._internal_capacity - 1

        data_idx = leaf_tree_idx - (self._internal_capacity - 1)

        # Check if the data index corresponds to a valid *entry* (within user capacity and n_entries)
        if not (0 <= data_idx < self.n_entries):
            tree_dump = self.tree[
                self._internal_capacity - 1 : self._internal_capacity
                - 1
                + self.n_entries
            ]
            sumtree_logger.error(
                f"GetLeaf: Invalid data_idx {data_idx} (from leaf_tree_idx {leaf_tree_idx}) retrieved. "
                f"n_entries={self.n_entries}, user_capacity={self.capacity}. "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}. "
                f"Leaf priorities (first {self.n_entries}): {tree_dump}"
            )
            raise IndexError(
                f"Retrieved data_idx {data_idx} is out of bounds for n_entries {self.n_entries}."
            )

        priority = self.tree[leaf_tree_idx]
        # Retrieve data using the calculated data_idx (which is within [0, user_capacity))
        data = self.data[data_idx]

        if data is None:
            # This should ideally not happen if data_idx < n_entries check passed
            tree_dump = self.tree[
                self._internal_capacity - 1 : self._internal_capacity
                - 1
                + self.n_entries
            ]
            sumtree_logger.error(
                f"Sampled None data at data_idx {data_idx} (tree_idx {leaf_tree_idx}). "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}, "
                f"n_entries: {self.n_entries}, data_pointer: {self.data_pointer}. "
                f"Leaf priorities (first {self.n_entries}): {tree_dump}"
            )
            raise RuntimeError(
                f"Sampled None data at data_idx {data_idx} (tree_idx {leaf_tree_idx}). "
            )

        return leaf_tree_idx, priority, data

    def total(self) -> float:
        """Returns the total priority (root node value)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns the maximum priority seen so far, or 1.0 if empty."""
        return float(self._max_priority) if self.n_entries > 0 else 1.0


File: muzerotriangle\utils\types.py
# File: muzerotriangle/utils/types.py
# File: muzerotriangle/utils/types.py
from collections import deque
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict

# --- Core State & Action ---


class StateType(TypedDict):
    grid: np.ndarray  # (C, H, W) float32
    other_features: np.ndarray  # (OtherFeatDim,) float32


ActionType = int

# --- MCTS & Policy ---

PolicyTargetMapping = Mapping[ActionType, float]
PolicyValueOutput = tuple[
    Mapping[ActionType, float], float
]  # (Policy Map, Expected Scalar Value)

# --- MuZero Trajectory Data ---


class TrajectoryStep(TypedDict):
    """Data stored for a single step in a game trajectory."""

    observation: StateType  # Observation o_t from the environment
    action: ActionType  # Action a_{t+1} taken after observation
    reward: float  # Actual reward r_{t+1} received from environment
    policy_target: PolicyTargetMapping  # MCTS policy target pi_t at step t
    value_target: float  # MCTS value target z_t (e.g., root value) at step t
    n_step_reward_target: float  # N-step discounted reward target R_t^{(N)}
    hidden_state: (
        np.ndarray | None
    )  # Optional: Store hidden state s_t from NN for debugging/analysis


# A complete game trajectory
Trajectory = list[TrajectoryStep]

# --- Training Data ---

# A sequence sampled from a trajectory for training
# Contains K unroll steps + 1 initial step = K+1 steps total
SampledSequence = list[TrajectoryStep]
SampledBatch = list[SampledSequence]  # Batch of sequences


# --- Prioritized Experience Replay (PER) ---
class SampledBatchPER(TypedDict):
    """Data structure for samples from PER buffer."""

    sequences: SampledBatch  # The batch of sampled sequences
    indices: np.ndarray  # Indices in the SumTree for priority updates
    weights: np.ndarray  # Importance sampling weights


# --- Statistics ---


class StepInfo(TypedDict, total=False):
    """Dictionary to hold various step counters associated with a metric."""

    global_step: int
    buffer_size: int  # Can now represent total steps or trajectories in buffer
    game_step_index: int
    # Add other relevant step types if needed


StatsCollectorData = dict[str, deque[tuple[StepInfo, float]]]


File: muzerotriangle\utils\__init__.py
# File: muzerotriangle/utils/__init__.py
from .geometry import is_point_in_polygon
from .helpers import (
    format_eta,
    get_device,
    normalize_color_for_matplotlib,
    set_random_seeds,
)
from .sumtree import SumTree

# Import MuZero-specific types
from .types import (
    ActionType,
    PolicyTargetMapping,  # Keep
    PolicyValueOutput,  # Keep
    SampledBatch,  # Keep
    SampledSequence,  # Keep
    StateType,  # Keep
    StatsCollectorData,  # Keep
    StepInfo,  # Keep
    Trajectory,  # Keep
    TrajectoryStep,  # Keep
)

# REMOVED: Experience, ExperienceBatch, PERBatchSample

__all__ = [
    # helpers
    "get_device",
    "set_random_seeds",
    "format_eta",
    "normalize_color_for_matplotlib",
    # types (MuZero relevant)
    "StateType",
    "ActionType",
    "PolicyTargetMapping",
    "PolicyValueOutput",
    "Trajectory",
    "TrajectoryStep",
    "SampledSequence",
    "SampledBatch",
    "StatsCollectorData",
    "StepInfo",
    # geometry
    "is_point_in_polygon",
    # structures
    "SumTree",  # Keep SumTree even if PER disabled, might be used later
]


File: muzerotriangle\visualization\README.md
# File: muzerotriangle/visualization/README.md
# Visualization Module (`muzerotriangle.visualization`)

## Purpose and Architecture

This module handles all visual aspects of the AlphaTriangle project, primarily using Pygame for rendering the game board, pieces, and training progress.

-   **Core ([`core/README.md`](core/README.md)):** Contains the main rendering classes (`Visualizer`, `GameRenderer`, `DashboardRenderer`), layout logic (`layout.py`), color definitions (`colors.py`), font loading (`fonts.py`), and coordinate mapping utilities (`coord_mapper.py`).
    -   `Visualizer`: Manages the display for interactive modes (play, debug).
    -   `GameRenderer`: Renders a single game instance, used by `DashboardRenderer`.
    -   `DashboardRenderer`: Manages the complex layout for training visualization, displaying multiple worker games, plots, and progress bars. **Progress bars now show specific information (model/params on train bar, global stats on buffer bar).**
-   **Drawing ([`drawing/README.md`](drawing/README.md)):** Contains specific functions for drawing individual elements like the grid, shapes, previews, HUD, and highlights. These are used by the core renderers.
-   **UI ([`ui/README.md`](ui/README.md)):** Contains reusable UI elements like `ProgressBar`.

## Exposed Interfaces

-   **Classes:**
    -   `Visualizer`: Main renderer for interactive modes.
    -   `GameRenderer`: Renderer for a single worker's game state.
    -   `DashboardRenderer`: Renderer for training visualization.
    -   `ProgressBar`: UI element for progress display.
-   **Functions:**
    -   `load_fonts`: Loads Pygame fonts.
    -   Various drawing functions (see `drawing/README.md`).
    -   Layout functions (see `core/README.md`).
    -   Coordinate mapping functions (see `core/README.md`).
-   **Modules:**
    -   `colors`: Provides color constants.

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: `VisConfig`, `EnvConfig`, `ModelConfig`.
-   **[`muzerotriangle.environment`](../environment/README.md)**: `GameState`, `GridData`.
-   **[`muzerotriangle.structs`](../structs/README.md)**: `Triangle`, `Shape`, color constants.
-   **[`muzerotriangle.stats`](../stats/README.md)**: `Plotter`, `StatsCollectorActor`.
-   **[`muzerotriangle.utils`](../utils/README.md)**: `geometry`, `helpers`, `types`.
-   **`pygame`**: Core library for graphics rendering.
-   **`matplotlib`**: Used by `Plotter` for generating plots.
-   **`numpy`**: Used for plot data handling.
-   **Standard Libraries:** `typing`, `logging`, `math`, `time`, `queue`, `io`.

---

**Note:** Keep this README updated with any changes to the visualization structure, components, or dependencies.

File: muzerotriangle\visualization\utils.py
import logging
from typing import cast

logger = logging.getLogger(__name__)


def normalize_color_for_matplotlib(
    color_tuple_0_255: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Converts RGB tuple (0-255) to Matplotlib format (0.0-1.0)."""
    if isinstance(color_tuple_0_255, tuple) and len(color_tuple_0_255) == 3:
        # Ensure values are within 0-255 before dividing
        valid_color = tuple(max(0, min(255, c)) for c in color_tuple_0_255)
        # Cast the result to the expected precise tuple type
        return cast("tuple[float, float, float]", tuple(c / 255.0 for c in valid_color))
    logger.warning(
        f"Invalid color format for normalization: {color_tuple_0_255}, returning black."
    )
    return (0.0, 0.0, 0.0)


File: muzerotriangle\visualization\__init__.py
"""
Visualization module for rendering the game state using Pygame.
"""

from ..config import VisConfig
from .core import colors
from .core.coord_mapper import (
    get_grid_coords_from_screen,
    get_preview_index_from_screen,
)
from .core.dashboard_renderer import DashboardRenderer
from .core.fonts import load_fonts
from .core.game_renderer import GameRenderer
from .core.layout import (
    calculate_interactive_layout,
    calculate_training_layout,
)
from .core.visualizer import Visualizer
from .drawing.grid import (
    draw_grid_background,
    draw_grid_indices,
    draw_grid_triangles,
)
from .drawing.highlight import draw_debug_highlight
from .drawing.hud import render_hud
from .drawing.previews import (
    draw_floating_preview,
    draw_placement_preview,
    render_previews,
)
from .drawing.shapes import draw_shape
from .ui.progress_bar import ProgressBar

__all__ = [
    # Core Renderers & Layout
    "Visualizer",
    "GameRenderer",
    "DashboardRenderer",
    "calculate_interactive_layout",
    "calculate_training_layout",
    "load_fonts",
    "colors",  # Export colors module
    "get_grid_coords_from_screen",
    "get_preview_index_from_screen",
    # Drawing Functions
    "draw_grid_background",
    "draw_grid_triangles",
    "draw_grid_indices",
    "draw_shape",
    "render_previews",
    "draw_placement_preview",
    "draw_floating_preview",
    "render_hud",
    "draw_debug_highlight",
    # UI Components
    "ProgressBar",
    # Config
    "VisConfig",
]


File: muzerotriangle\visualization\core\colors.py
# File: muzerotriangle/visualization/core/colors.py
"""Centralized color definitions (RGB tuples 0-255)."""

WHITE: tuple[int, int, int] = (255, 255, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
LIGHT_GRAY: tuple[int, int, int] = (180, 180, 180)
GRAY: tuple[int, int, int] = (100, 100, 100)
DARK_GRAY: tuple[int, int, int] = (40, 40, 40)
RED: tuple[int, int, int] = (220, 40, 40)
DARK_RED: tuple[int, int, int] = (100, 10, 10)
BLUE: tuple[int, int, int] = (60, 60, 220)
YELLOW: tuple[int, int, int] = (230, 230, 40)
GREEN: tuple[int, int, int] = (40, 200, 40)
DARK_GREEN: tuple[int, int, int] = (10, 80, 10)
ORANGE: tuple[int, int, int] = (240, 150, 20)
PURPLE: tuple[int, int, int] = (140, 40, 140)
CYAN: tuple[int, int, int] = (40, 200, 200)
LIGHTG: tuple[int, int, int] = (144, 238, 144)
HOTPINK: tuple[int, int, int] = (255, 105, 180)  # Added for plots

GOOGLE_COLORS: list[tuple[int, int, int]] = [
    (15, 157, 88),  # Green
    (244, 180, 0),  # Yellow
    (66, 133, 244),  # Blue
    (219, 68, 55),  # Red
]

# Game Specific Visuals
GRID_BG_DEFAULT: tuple[int, int, int] = (20, 20, 30)
GRID_BG_GAME_OVER: tuple[int, int, int] = DARK_RED
GRID_LINE_COLOR: tuple[int, int, int] = GRAY
TRIANGLE_EMPTY_COLOR: tuple[int, int, int] = (60, 60, 70)
PREVIEW_BG: tuple[int, int, int] = (30, 30, 40)
PREVIEW_BORDER: tuple[int, int, int] = GRAY
PREVIEW_SELECTED_BORDER: tuple[int, int, int] = BLUE
PLACEMENT_VALID_COLOR: tuple[int, int, int, int] = (*GREEN, 150)  # RGBA
PLACEMENT_INVALID_COLOR: tuple[int, int, int, int] = (*RED, 100)  # RGBA
DEBUG_TOGGLE_COLOR: tuple[int, int, int] = YELLOW

# --- ADDED: Colors for Progress Bar Cycling ---
PROGRESS_BAR_CYCLE_COLORS: list[tuple[int, int, int]] = [
    GREEN,
    BLUE,
    YELLOW,
    ORANGE,
    PURPLE,
    CYAN,
    HOTPINK,
    RED,  # Add red towards the end
]
# --- END ADDED ---


File: muzerotriangle\visualization\core\coord_mapper.py
import pygame

from ...config import EnvConfig
from ...structs import Triangle
from ...utils.geometry import is_point_in_polygon


def _calculate_render_params(
    width: int, height: int, config: EnvConfig
) -> tuple[float, float, float, float]:
    """Calculates scale (cw, ch) and offset (ox, oy) for rendering the grid."""
    rows, cols = config.ROWS, config.COLS
    cols_eff = cols * 0.75 + 0.25 if cols > 0 else 1
    scale_w = width / cols_eff if cols_eff > 0 else 1
    scale_h = height / rows if rows > 0 else 1
    scale = max(1.0, min(scale_w, scale_h))
    cell_size = scale
    grid_w_px = cols_eff * cell_size
    grid_h_px = rows * cell_size
    offset_x = (width - grid_w_px) / 2
    offset_y = (height - grid_h_px) / 2
    return cell_size, cell_size, offset_x, offset_y


def get_grid_coords_from_screen(
    screen_pos: tuple[int, int], grid_area_rect: pygame.Rect, config: EnvConfig
) -> tuple[int, int] | None:
    """Maps screen coordinates (relative to screen) to grid row/column."""
    if not grid_area_rect or not grid_area_rect.collidepoint(screen_pos):
        return None

    local_x = screen_pos[0] - grid_area_rect.left
    local_y = screen_pos[1] - grid_area_rect.top
    cw, ch, ox, oy = _calculate_render_params(
        grid_area_rect.width, grid_area_rect.height, config
    )
    if cw <= 0 or ch <= 0:
        return None

    row = int((local_y - oy) / ch) if ch > 0 else -1
    approx_col_center_index = (local_x - ox - cw / 4) / (cw * 0.75) if cw > 0 else -1
    col = int(round(approx_col_center_index))

    for r_check in [row, row - 1, row + 1]:
        if not (0 <= r_check < config.ROWS):
            continue
        for c_check in [col, col - 1, col + 1]:
            if not (0 <= c_check < config.COLS):
                continue
            # Use corrected orientation check
            is_up = (r_check + c_check) % 2 != 0
            temp_tri = Triangle(r_check, c_check, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            if is_point_in_polygon((local_x, local_y), pts):
                return r_check, c_check

    if 0 <= row < config.ROWS and 0 <= col < config.COLS:
        return row, col
    return None


def get_preview_index_from_screen(
    screen_pos: tuple[int, int], preview_rects: dict[int, pygame.Rect]
) -> int | None:
    """Maps screen coordinates to a shape preview index."""
    if not preview_rects:
        return None
    for idx, rect in preview_rects.items():
        if rect and rect.collidepoint(screen_pos):
            return idx
    return None


File: muzerotriangle\visualization\core\dashboard_renderer.py
# File: muzerotriangle/visualization/core/dashboard_renderer.py
import logging
import math
from collections import deque  # Import deque
from typing import TYPE_CHECKING, Any

import pygame
import ray  # Import ray

from ...environment import GameState
from ...stats import Plotter
from ..drawing import hud as hud_drawing
from ..ui import ProgressBar  # Import ProgressBar
from . import colors, layout
from .game_renderer import GameRenderer

if TYPE_CHECKING:
    from ...config import EnvConfig, ModelConfig, VisConfig
    from ...utils.types import StatsCollectorData

logger = logging.getLogger(__name__)


class DashboardRenderer:
    """Renders the training dashboard, including multiple game states and plots."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
        stats_collector_actor: ray.actor.ActorHandle | None,
        model_config: "ModelConfig",
        total_params: int | None = None,
        trainable_params: int | None = None,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.stats_collector_actor = stats_collector_actor
        self.model_config = model_config
        self.total_params = total_params
        self.trainable_params = trainable_params

        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.worker_sub_rects: dict[int, pygame.Rect] = {}
        self.last_worker_grid_size = (0, 0)
        self.last_num_workers = 0

        self.game_renderer = GameRenderer(vis_config, env_config, fonts)
        self.plotter = Plotter()
        self.progress_bar_height_per_bar = 25
        self.num_progress_bars = 2
        self.progress_bar_spacing = 2
        self.progress_bars_total_height = (
            (
                (self.progress_bar_height_per_bar * self.num_progress_bars)
                + (self.progress_bar_spacing * (self.num_progress_bars - 1))
            )
            if self.num_progress_bars > 0
            else 0
        )
        self._layout_calculated_for_size: tuple[int, int] = (0, 0)

    def ensure_layout(self):
        """Recalculates layout if screen size changes."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)
        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            self.layout_rects = layout.calculate_training_layout(
                current_w,
                current_h,
                self.vis_config,
                progress_bars_total_height=self.progress_bars_total_height,
            )
            self._layout_calculated_for_size = current_size
            logger.info(f"Recalculated dashboard layout for {current_size}")
            # Reset worker layout cache as well
            self.last_worker_grid_size = (0, 0)
            self.worker_sub_rects = {}
        return self.layout_rects if self.layout_rects is not None else {}

    def _calculate_worker_sub_layout(self, worker_grid_area, worker_ids):
        """Calculates the positions and sizes for each worker's game view."""
        area_w, area_h = worker_grid_area.size
        num_workers = len(worker_ids)

        # Only recalculate if size or number of workers changes
        if (
            area_w,
            area_h,
        ) == self.last_worker_grid_size and num_workers == self.last_num_workers:
            return

        self.last_worker_grid_size = (area_w, area_h)
        self.last_num_workers = num_workers
        self.worker_sub_rects = {}

        if area_h <= 10 or area_w <= 10 or num_workers <= 0:
            logger.warning(
                f"Worker grid area too small or zero workers: {area_w}x{area_h}, {num_workers} workers"
            )
            return

        # Simple grid layout calculation
        cols = int(math.ceil(math.sqrt(num_workers)))
        rows = math.ceil(num_workers / cols)
        cell_w = area_w // cols
        cell_h = area_h // rows

        logger.info(
            f"Calculated worker sub-layout: {rows}x{cols}. Cell: {cell_w}x{cell_h}"
        )

        sorted_worker_ids = sorted(worker_ids)
        for i, worker_id in enumerate(sorted_worker_ids):
            row = i // cols
            col = i % cols
            worker_area_x = worker_grid_area.left + col * cell_w
            worker_area_y = worker_grid_area.top + row * cell_h
            worker_rect = pygame.Rect(worker_area_x, worker_area_y, cell_w, cell_h)
            self.worker_sub_rects[worker_id] = worker_rect.clip(worker_grid_area)

    def render(
        self,
        worker_states: dict[int, GameState],
        global_stats: dict[str, Any] | None = None,
    ):
        """Renders the entire training dashboard."""
        self.screen.fill(colors.DARK_GRAY)
        layout_rects = self.ensure_layout()
        if not layout_rects:
            return

        worker_grid_area = layout_rects.get("worker_grid")
        plots_rect = layout_rects.get("plots")
        progress_bar_area_rect = layout_rects.get("progress_bar_area")
        layout_rects.get("hud")

        # --- Render Worker Grids ---
        if (
            worker_grid_area
            and worker_grid_area.width > 0
            and worker_grid_area.height > 0
        ):
            worker_ids = list(worker_states.keys())
            if not worker_ids and global_stats and "num_workers" in global_stats:
                # If no states yet, create placeholders based on expected worker count
                worker_ids = list(range(global_stats["num_workers"]))

            self._calculate_worker_sub_layout(worker_grid_area, worker_ids)

            for worker_id, worker_area_rect in self.worker_sub_rects.items():
                game_state = worker_states.get(worker_id)
                # Extract step-specific stats for this worker if available
                worker_step_stats = None
                if global_stats and "latest_worker_stats" in global_stats:
                    worker_step_stats = global_stats["latest_worker_stats"].get(
                        worker_id
                    )

                # --- CORRECTED TYPO ---
                self.game_renderer.render_worker_state(
                    self.screen,
                    worker_area_rect,
                    worker_id,
                    game_state,
                    worker_step_stats=worker_step_stats,
                )
                # --- END CORRECTION ---
                pygame.draw.rect(self.screen, colors.GRAY, worker_area_rect, 1)

        # --- Render Plots ---
        if plots_rect and plots_rect.width > 0 and plots_rect.height > 0:
            stats_data_for_plot: StatsCollectorData | None = (
                global_stats.get("stats_data") if global_stats else None
            )
            plot_surface = None
            if stats_data_for_plot is not None:
                has_any_metric_data = any(
                    isinstance(dq, deque) and dq  # Use imported deque
                    for key, dq in stats_data_for_plot.items()
                    if not key.startswith("Internal/")
                )
                if has_any_metric_data:
                    plot_surface = self.plotter.get_plot_surface(
                        stats_data_for_plot,
                        int(plots_rect.width),
                        int(plots_rect.height),
                    )

            if plot_surface:
                self.screen.blit(plot_surface, plots_rect.topleft)
            else:  # Draw placeholder if no data or plotter failed
                pygame.draw.rect(self.screen, colors.DARK_GRAY, plots_rect)
                plot_font = self.fonts.get("help")
                if plot_font:
                    wait_text = (
                        "Plot Area (Waiting for data...)"
                        if stats_data_for_plot is None
                        else "Plot Area (No data yet)"
                    )
                    wait_surf = plot_font.render(wait_text, True, colors.LIGHT_GRAY)
                    wait_rect = wait_surf.get_rect(center=plots_rect.center)
                    self.screen.blit(wait_surf, wait_rect)
                pygame.draw.rect(self.screen, colors.GRAY, plots_rect, 1)

        # --- Render Progress Bars ---
        if progress_bar_area_rect and global_stats:
            current_y = progress_bar_area_rect.top
            progress_bar_font = self.fonts.get("help")
            if progress_bar_font:
                bar_width = progress_bar_area_rect.width
                bar_x = progress_bar_area_rect.left
                bar_height = self.progress_bar_height_per_bar

                # Training Progress Bar
                train_progress = global_stats.get("train_progress")
                if isinstance(train_progress, ProgressBar):  # Use imported ProgressBar
                    # Construct info string for training bar
                    train_info_parts = []
                    if self.model_config:
                        model_str = f"CNN:{len(self.model_config.CONV_FILTERS)}L"
                        if self.model_config.NUM_RESIDUAL_BLOCKS > 0:
                            model_str += (
                                f"/Res:{self.model_config.NUM_RESIDUAL_BLOCKS}L"
                            )
                        if (
                            self.model_config.USE_TRANSFORMER_IN_REP
                            and self.model_config.REP_TRANSFORMER_LAYERS > 0
                        ):
                            model_str += (
                                f"/TF:{self.model_config.REP_TRANSFORMER_LAYERS}L"
                            )
                        train_info_parts.append(model_str)
                    if self.total_params is not None:
                        train_info_parts.append(
                            f"Params:{self.total_params / 1e6:.1f}M"
                        )
                    train_bar_info_str = " | ".join(train_info_parts)

                    train_progress.render(
                        self.screen,
                        (bar_x, current_y),
                        int(bar_width),
                        bar_height,
                        progress_bar_font,
                        border_radius=3,
                        info_line=train_bar_info_str,
                    )
                    current_y += bar_height + self.progress_bar_spacing

                # Buffer Progress Bar
                buffer_progress = global_stats.get("buffer_progress")
                if isinstance(buffer_progress, ProgressBar):  # Use imported ProgressBar
                    # Construct info string for buffer bar
                    buffer_info_parts = []
                    updates = global_stats.get("worker_weight_updates", "?")
                    episodes = global_stats.get("episodes_played", "?")
                    sims = global_stats.get("total_simulations_run", "?")
                    num_workers = global_stats.get("num_workers", "?")
                    pending_tasks = global_stats.get("num_pending_tasks", "?")

                    buffer_info_parts.append(f"Weight Updates: {updates}")
                    buffer_info_parts.append(f"Episodes: {episodes}")
                    if isinstance(sims, int | float):
                        sims_str = (
                            f"{sims / 1e6:.1f}M"
                            if sims >= 1e6
                            else (
                                f"{sims / 1e3:.0f}k" if sims >= 1000 else str(int(sims))
                            )
                        )
                        buffer_info_parts.append(f"Simulations: {sims_str}")
                    else:
                        buffer_info_parts.append(f"Simulations: {sims}")

                    if isinstance(pending_tasks, int) and isinstance(num_workers, int):
                        buffer_info_parts.append(
                            f"Workers: {num_workers - pending_tasks}/{num_workers} Active"
                        )
                    else:
                        buffer_info_parts.append("Workers: ?/?")

                    buffer_bar_info_str = " | ".join(buffer_info_parts)

                    buffer_progress.render(
                        self.screen,
                        (bar_x, current_y),
                        int(bar_width),
                        bar_height,
                        progress_bar_font,
                        border_radius=3,
                        info_line=buffer_bar_info_str,
                    )

        # Render HUD (always last)
        hud_drawing.render_hud(
            surface=self.screen,
            mode="training_visual",
            fonts=self.fonts,
            display_stats=None,  # HUD doesn't need detailed stats anymore
        )


File: muzerotriangle\visualization\core\fonts.py
# File: muzerotriangle/visualization/core/fonts.py
import logging

import pygame

logger = logging.getLogger(__name__)

DEFAULT_FONT_NAME: str | None = None  # Add type hint
FALLBACK_FONT_NAME = "arial,freesans"


def load_single_font(name: str | None, size: int) -> pygame.font.Font | None:
    """Loads a single font, handling potential errors."""
    try:
        font = pygame.font.SysFont(name, size)
        return font
    except Exception as e:
        logger.error(f"Error loading font '{name}' size {size}: {e}")
        if name != FALLBACK_FONT_NAME:
            logger.warning(f"Attempting fallback font: {FALLBACK_FONT_NAME}")
            try:
                font = pygame.font.SysFont(FALLBACK_FONT_NAME, size)
                logger.info(f"Loaded fallback font: {FALLBACK_FONT_NAME} size {size}")
                return font
            except Exception as e_fallback:
                logger.error(f"Fallback font failed: {e_fallback}")
                return None
        return None


def load_fonts(
    font_sizes: dict[str, int] | None = None,
) -> dict[str, pygame.font.Font | None]:
    """Loads standard game fonts."""
    if font_sizes is None:
        font_sizes = {
            "ui": 24,
            "score": 30,
            "help": 18,
            "title": 48,
        }

    fonts: dict[str, pygame.font.Font | None] = {}
    required_fonts = ["score", "help"]

    logger.info("Loading fonts...")
    for name, size in font_sizes.items():
        fonts[name] = load_single_font(DEFAULT_FONT_NAME, size)

    for name in required_fonts:
        if fonts.get(name) is None:
            logger.critical(
                f"Essential font '{name}' failed to load. Text rendering will be affected."
            )

    return fonts


File: muzerotriangle\visualization\core\game_renderer.py
import logging
from typing import TYPE_CHECKING, Any

import pygame

from ...environment import GameState
from ..drawing import grid as grid_drawing
from ..drawing import previews as preview_drawing
from . import colors

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig

logger = logging.getLogger(__name__)


class GameRenderer:
    """
    Renders a single GameState (grid and previews) within a specified area.
    Used by DashboardRenderer for displaying worker states.
    Also displays latest per-step stats for the worker.
    """

    def __init__(
        self,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
    ):
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.preview_width_ratio = 0.2
        self.min_preview_width = 30
        self.max_preview_width = 60
        self.padding = 5

    def render_worker_state(
        self,
        target_surface: pygame.Surface,
        area_rect: pygame.Rect,
        worker_id: int,
        game_state: GameState | None,
        # Add worker_step_stats parameter
        worker_step_stats: dict[str, Any] | None = None,
    ):
        """
        Renders the game state of a single worker into the specified area_rect
        on the target_surface. Includes per-step stats display.
        """
        if not game_state:
            # Optionally draw a placeholder if state is None
            pygame.draw.rect(target_surface, colors.DARK_GRAY, area_rect)
            pygame.draw.rect(target_surface, colors.GRAY, area_rect, 1)
            id_font = self.fonts.get("help")
            if id_font:
                id_surf = id_font.render(
                    f"W{worker_id} (No State)", True, colors.LIGHT_GRAY
                )
                id_rect = id_surf.get_rect(center=area_rect.center)
                target_surface.blit(id_surf, id_rect)
            return

        # Calculate layout within the worker's area_rect
        preview_w = max(
            self.min_preview_width,
            min(area_rect.width * self.preview_width_ratio, self.max_preview_width),
        )
        grid_w = max(0, area_rect.width - preview_w - self.padding)
        grid_h = area_rect.height
        preview_h = area_rect.height

        grid_rect_local = pygame.Rect(0, 0, grid_w, grid_h)
        preview_rect_local = pygame.Rect(grid_w + self.padding, 0, preview_w, preview_h)

        # Create subsurfaces relative to the target_surface
        try:
            worker_surface = target_surface.subsurface(area_rect)
            worker_surface.fill(colors.DARK_GRAY)  # Background for the worker area
            pygame.draw.rect(
                target_surface, colors.GRAY, area_rect, 1
            )  # Border around worker area

            # Render Grid
            if grid_rect_local.width > 0 and grid_rect_local.height > 0:
                grid_surf = worker_surface.subsurface(grid_rect_local)
                bg_color = (
                    colors.GRID_BG_GAME_OVER
                    if game_state.is_over()
                    else colors.GRID_BG_DEFAULT
                )
                grid_drawing.draw_grid_background(grid_surf, bg_color)
                grid_drawing.draw_grid_triangles(
                    grid_surf, game_state.grid_data, self.env_config
                )

                # --- Render Worker Info Text ---
                id_font = self.fonts.get("help")
                if id_font:
                    line_y = 3
                    line_height = id_font.get_height() + 1

                    # Worker ID and Game Step
                    id_text = f"W{worker_id} (Step {game_state.current_step})"
                    id_surf = id_font.render(id_text, True, colors.LIGHT_GRAY)
                    grid_surf.blit(id_surf, (3, line_y))
                    line_y += line_height

                    # Current Score
                    score_text = f"Score: {game_state.game_score:.0f}"
                    score_surf = id_font.render(score_text, True, colors.YELLOW)
                    grid_surf.blit(score_surf, (3, line_y))
                    line_y += line_height

                    # MCTS Stats (if available)
                    if worker_step_stats:
                        visits = worker_step_stats.get("mcts_visits", "?")
                        depth = worker_step_stats.get("mcts_depth", "?")
                        mcts_text = f"MCTS: V={visits} D={depth}"
                        mcts_surf = id_font.render(mcts_text, True, colors.CYAN)
                        grid_surf.blit(mcts_surf, (3, line_y))
                        line_y += line_height

            # Render Previews
            if preview_rect_local.width > 0 and preview_rect_local.height > 0:
                preview_surf = worker_surface.subsurface(preview_rect_local)
                # Pass 0,0 as topleft because preview_surf is already positioned
                _ = preview_drawing.render_previews(
                    preview_surf,
                    game_state,
                    (0, 0),  # Relative to preview_surf
                    "training_visual",  # Mode context
                    self.env_config,
                    self.vis_config,
                    selected_shape_idx=-1,  # No selection in training view
                )

        except ValueError as e:
            # Handle cases where subsurface creation fails (e.g., zero dimensions)
            if "subsurface rectangle is invalid" not in str(e):
                logger.error(
                    f"Error creating subsurface for W{worker_id} in area {area_rect}: {e}"
                )
            # Draw error indicator if subsurface fails
            pygame.draw.rect(target_surface, colors.RED, area_rect, 2)
            id_font = self.fonts.get("help")
            if id_font:
                id_surf = id_font.render(f"W{worker_id} (Render Err)", True, colors.RED)
                id_rect = id_surf.get_rect(center=area_rect.center)
                target_surface.blit(id_surf, id_rect)


File: muzerotriangle\visualization\core\layout.py
# File: muzerotriangle/visualization/core/layout.py
import logging

import pygame

from ...config import VisConfig

logger = logging.getLogger(__name__)


def calculate_interactive_layout(
    screen_width: int, screen_height: int, vis_config: VisConfig
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for interactive modes (play/debug).
    Places grid on the left and preview on the right.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    hud_h = vis_config.HUD_HEIGHT
    preview_w = vis_config.PREVIEW_AREA_WIDTH

    available_h = max(0, sh - hud_h - 2 * pad)
    available_w = max(0, sw - 3 * pad)  # Account for padding around grid and preview

    grid_w = max(0, available_w - preview_w)
    grid_h = available_h

    grid_rect = pygame.Rect(pad, pad, grid_w, grid_h)
    preview_rect = pygame.Rect(grid_rect.right + pad, pad, preview_w, grid_h)
    hud_rect = pygame.Rect(pad, grid_rect.bottom + pad, sw - 2 * pad, hud_h)

    # Clip rects to screen bounds to prevent errors with small windows
    screen_rect = pygame.Rect(0, 0, sw, sh)
    grid_rect = grid_rect.clip(screen_rect)
    preview_rect = preview_rect.clip(screen_rect)
    hud_rect = hud_rect.clip(screen_rect)

    logger.debug(
        f"Interactive Layout calculated: Grid={grid_rect}, Preview={preview_rect}, HUD={hud_rect}"
    )

    return {
        "grid": grid_rect,
        "preview": preview_rect,
        "hud": hud_rect,
    }


def calculate_training_layout(
    screen_width: int,
    screen_height: int,
    vis_config: VisConfig,
    progress_bars_total_height: int,
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for training visualization mode.
    Worker grid top, progress bars bottom (above HUD), plots fill middle.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    hud_h = vis_config.HUD_HEIGHT

    # Calculate total available height excluding top/bottom padding and HUD
    total_content_height = sh - hud_h - 2 * pad

    # Allocate space for worker grid (e.g., 30% of available height)
    worker_grid_h = int(total_content_height * 0.3)
    worker_grid_w = sw - 2 * pad
    worker_grid_rect = pygame.Rect(pad, pad, worker_grid_w, worker_grid_h)

    # Allocate space for progress bars at the bottom, just above the HUD
    pb_area_y = sh - hud_h - pad - progress_bars_total_height
    pb_area_w = sw - 2 * pad
    progress_bar_area_rect = pygame.Rect(
        pad, pb_area_y, pb_area_w, progress_bars_total_height
    )

    # Plot area fills the remaining space between worker grid and progress bars
    plot_area_y = worker_grid_rect.bottom + pad
    plot_area_h = max(0, progress_bar_area_rect.top - plot_area_y - pad)
    plot_area_w = sw - 2 * pad
    plot_rect = pygame.Rect(pad, plot_area_y, plot_area_w, plot_area_h)

    # HUD area at the very bottom
    hud_rect = pygame.Rect(pad, sh - hud_h - pad, sw - 2 * pad, hud_h)

    # Clip all rects to screen bounds
    screen_rect = pygame.Rect(0, 0, sw, sh)
    worker_grid_rect = worker_grid_rect.clip(screen_rect)
    plot_rect = plot_rect.clip(screen_rect)
    progress_bar_area_rect = progress_bar_area_rect.clip(screen_rect)
    hud_rect = hud_rect.clip(screen_rect)

    logger.debug(
        f"Training Layout calculated: WorkerGrid={worker_grid_rect}, PlotRect={plot_rect}, ProgressBarArea={progress_bar_area_rect}, HUD={hud_rect}"
    )

    return {
        "worker_grid": worker_grid_rect,
        "plots": plot_rect,
        "progress_bar_area": progress_bar_area_rect,
        "hud": hud_rect,
    }


File: muzerotriangle\visualization\core\README.md
# File: muzerotriangle/visualization/core/README.md
# Visualization Core Submodule (`muzerotriangle.visualization.core`)

## Purpose and Architecture

This submodule contains the central classes and foundational elements for the visualization system. It orchestrates rendering, manages layout and coordinate systems, and defines core visual properties like colors and fonts.

-   **Render Orchestration:**
    -   [`Visualizer`](visualizer.py): The main class for rendering in **interactive modes** ("play", "debug"). It maintains the Pygame screen, calculates layout using `layout.py`, manages cached preview area rectangles, and calls appropriate drawing functions from [`muzerotriangle.visualization.drawing`](../drawing/README.md). **It receives interaction state (hover position, selected index) via its `render` method to display visual feedback.**
    -   [`GameRenderer`](game_renderer.py): **Simplified renderer** responsible for drawing a **single** worker's `GameState` (grid and previews) within a specified sub-rectangle. Used by the `DashboardRenderer`.
    -   [`DashboardRenderer`](dashboard_renderer.py): Renderer specifically for the **training visualization mode**. It uses `layout.py` to divide the screen into a worker game grid area and a statistics area. It renders multiple worker `GameState` objects (using `GameRenderer` instances) in the top grid and displays statistics plots (using [`muzerotriangle.stats.Plotter`](../../stats/plotter.py)) and progress bars in the bottom area. **The training progress bar shows model/parameter info, while the buffer progress bar shows global training stats (worker weight updates, episodes, sims, worker status). Plots now include black, solid vertical lines (drawn on top) indicating the `global_step` when worker weights were updated, mapped to the appropriate position on each plot's x-axis.** It takes a dictionary mapping worker IDs to `GameState` objects and a dictionary of global statistics.
-   **Layout Management:**
    -   [`layout.py`](layout.py): Contains functions (`calculate_interactive_layout`, `calculate_training_layout`) to determine the size and position of the main UI areas based on the screen dimensions, mode, and `VisConfig`.
-   **Coordinate System:**
    -   [`coord_mapper.py`](coord_mapper.py): Provides essential mapping functions:
        -   `_calculate_render_params`: Internal helper to get scaling and offset for grid rendering.
        -   `get_grid_coords_from_screen`: Converts mouse/screen coordinates into logical grid (row, column) coordinates.
        -   `get_preview_index_from_screen`: Converts mouse/screen coordinates into the index of the shape preview slot being pointed at.
-   **Visual Properties:**
    -   [`colors.py`](colors.py): Defines a centralized palette of named color constants (RGB tuples).
    -   [`fonts.py`](fonts.py): Contains the `load_fonts` function to load and manage Pygame font objects.

## Exposed Interfaces

-   **Classes:**
    -   `Visualizer`: Renderer for interactive modes.
        -   `__init__(...)`
        -   `render(game_state: GameState, mode: str, **interaction_state)`: Renders based on game state and interaction hints.
        -   `ensure_layout() -> Dict[str, pygame.Rect]`
        -   `screen`: Public attribute (Pygame Surface).
        -   `preview_rects`: Public attribute (cached preview area rects).
    -   `GameRenderer`: Renderer for a single worker's game state.
        -   `__init__(...)`
        -   `render_worker_state(target_surface: pygame.Surface, area_rect: pygame.Rect, worker_id: int, game_state: Optional[GameState], worker_step_stats: Optional[Dict[str, Any]])`
    -   `DashboardRenderer`: Renderer for combined multi-game/stats training visualization.
        -   `__init__(...)`
        -   `render(worker_states: Dict[int, GameState], global_stats: Optional[Dict[str, Any]])`
        -   `screen`: Public attribute (Pygame Surface).
-   **Functions:**
    -   `calculate_interactive_layout(...) -> Dict[str, pygame.Rect]`
    -   `calculate_training_layout(...) -> Dict[str, pygame.Rect]`
    -   `load_fonts() -> Dict[str, Optional[pygame.font.Font]]`
    -   `get_grid_coords_from_screen(...) -> Optional[Tuple[int, int]]`
    -   `get_preview_index_from_screen(...) -> Optional[int]`
-   **Modules:**
    -   `colors`: Provides color constants (e.g., `colors.RED`).

## Dependencies

-   **[`muzerotriangle.config`](../../config/README.md)**: `VisConfig`, `EnvConfig`, `ModelConfig`.
-   **[`muzerotriangle.environment`](../../environment/README.md)**: `GameState`, `GridData`.
-   **[`muzerotriangle.stats`](../../stats/README.md)**: `Plotter`, `StatsCollectorActor`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**: `types`, `helpers`.
-   **[`muzerotriangle.visualization.drawing`](../drawing/README.md)**: Drawing functions are called by renderers.
-   **[`muzerotriangle.visualization.ui`](../ui/README.md)**: `ProgressBar` (used by `DashboardRenderer`).
-   **`pygame`**: Used for surfaces, rectangles, fonts, display management.
-   **`ray`**: Used by `DashboardRenderer` (for actor handle type hint).
-   **Standard Libraries:** `typing`, `logging`, `math`, `collections.deque`.

---

**Note:** Please keep this README updated when changing the core rendering logic, layout calculations, coordinate mapping, or the interfaces of the renderers. Accurate documentation is crucial for maintainability.

File: muzerotriangle\visualization\core\visualizer.py
import logging
from typing import TYPE_CHECKING

import pygame

from ...structs import Shape
from ..drawing import grid as grid_drawing
from ..drawing import highlight as highlight_drawing
from ..drawing import hud as hud_drawing
from ..drawing import previews as preview_drawing
from ..drawing.previews import (
    draw_floating_preview,
    draw_placement_preview,
)
from . import colors, layout

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig
    from ...environment.core.game_state import GameState

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Orchestrates rendering of a single game state for interactive modes.
    Receives interaction state (hover, selection) via render parameters.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.preview_rects: dict[int, pygame.Rect] = {}  # Cache preview rects
        self._layout_calculated_for_size: tuple[int, int] = (0, 0)
        self.ensure_layout()  # Initial layout calculation

    def ensure_layout(self) -> dict[str, pygame.Rect]:
        """Returns cached layout or calculates it if needed."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)

        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            # Use the interactive layout calculation
            self.layout_rects = layout.calculate_interactive_layout(
                current_w, current_h, self.vis_config
            )
            self._layout_calculated_for_size = current_size
            logger.info(
                f"Recalculated interactive layout for size {current_size}: {self.layout_rects}"
            )
            # Clear preview rect cache when layout changes
            self.preview_rects = {}

        return self.layout_rects if self.layout_rects is not None else {}

    def render(
        self,
        game_state: "GameState",
        mode: str,
        # Interaction state passed in:
        selected_shape_idx: int = -1,
        hover_shape: Shape | None = None,
        hover_grid_coord: tuple[int, int] | None = None,
        hover_is_valid: bool = False,
        hover_screen_pos: tuple[int, int] | None = None,
        debug_highlight_coord: tuple[int, int] | None = None,
    ):
        """
        Renders the entire game visualization for interactive modes.
        Uses interaction state passed as parameters for visual feedback.
        """
        self.screen.fill(colors.GRID_BG_DEFAULT)  # Clear screen
        layout_rects = self.ensure_layout()
        grid_rect = layout_rects.get("grid")
        preview_rect = layout_rects.get("preview")

        # Render Grid Area
        if grid_rect and grid_rect.width > 0 and grid_rect.height > 0:
            try:
                grid_surf = self.screen.subsurface(grid_rect)
                self._render_grid_area(
                    grid_surf,
                    game_state,
                    mode,
                    grid_rect,
                    hover_shape,
                    hover_grid_coord,
                    hover_is_valid,
                    hover_screen_pos,
                    debug_highlight_coord,
                )
            except ValueError as e:
                logger.error(f"Error creating grid subsurface ({grid_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, grid_rect, 1)

        # Render Preview Area
        if preview_rect and preview_rect.width > 0 and preview_rect.height > 0:
            try:
                preview_surf = self.screen.subsurface(preview_rect)
                # Pass selected_shape_idx for highlighting
                self._render_preview_area(
                    preview_surf, game_state, mode, preview_rect, selected_shape_idx
                )
            except ValueError as e:
                logger.error(f"Error creating preview subsurface ({preview_rect}): {e}")
                pygame.draw.rect(self.screen, colors.RED, preview_rect, 1)

        # Render HUD
        # --- CORRECTED CALL ---
        hud_drawing.render_hud(
            surface=self.screen,
            mode=mode,
            fonts=self.fonts,
            display_stats=None,  # Pass None for display_stats in interactive modes
        )
        # --- END CORRECTION ---

    def _render_grid_area(
        self,
        grid_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        grid_rect: pygame.Rect,  # Pass grid_rect for hover calculations
        hover_shape: Shape | None,
        hover_grid_coord: tuple[int, int] | None,
        hover_is_valid: bool,
        hover_screen_pos: tuple[int, int] | None,
        debug_highlight_coord: tuple[int, int] | None,
    ):
        """Renders the main game grid and overlays onto the provided grid_surf."""
        # Background
        bg_color = (
            colors.GRID_BG_GAME_OVER if game_state.is_over() else colors.GRID_BG_DEFAULT
        )
        grid_drawing.draw_grid_background(grid_surf, bg_color)

        # Grid Triangles
        grid_drawing.draw_grid_triangles(
            grid_surf, game_state.grid_data, self.env_config
        )

        # Debug Indices
        if mode == "debug":
            grid_drawing.draw_grid_indices(
                grid_surf, game_state.grid_data, self.env_config, self.fonts
            )

        # Play Mode Hover Previews
        if mode == "play" and hover_shape:
            if hover_grid_coord:  # Snapped preview
                draw_placement_preview(
                    grid_surf,
                    hover_shape,
                    hover_grid_coord[0],
                    hover_grid_coord[1],
                    is_valid=hover_is_valid,  # Use validity passed in
                    config=self.env_config,
                )
            elif hover_screen_pos:  # Floating preview (relative to grid_surf)
                # Adjust screen pos to be relative to grid_surf
                local_hover_pos = (
                    hover_screen_pos[0] - grid_rect.left,
                    hover_screen_pos[1] - grid_rect.top,
                )
                if grid_surf.get_rect().collidepoint(local_hover_pos):
                    draw_floating_preview(
                        grid_surf,
                        hover_shape,
                        local_hover_pos,
                        self.env_config,
                    )

        # Debug Mode Highlight
        if mode == "debug" and debug_highlight_coord:
            r, c = debug_highlight_coord
            highlight_drawing.draw_debug_highlight(grid_surf, r, c, self.env_config)

        # --- ADDED: Display Score in Grid Area for Interactive Modes ---
        score_font = self.fonts.get("score")
        if score_font:
            score_text = f"Score: {game_state.game_score:.0f}"
            score_surf = score_font.render(score_text, True, colors.YELLOW)
            # Position score at top-left of grid area
            score_rect = score_surf.get_rect(topleft=(5, 5))
            grid_surf.blit(score_surf, score_rect)
        # --- END ADDED ---

    def _render_preview_area(
        self,
        preview_surf: pygame.Surface,
        game_state: "GameState",
        mode: str,
        preview_rect: pygame.Rect,
        selected_shape_idx: int,  # Pass selected index
    ):
        """Renders the shape preview slots onto preview_surf and caches rects."""
        # Pass selected_shape_idx to render_previews for highlighting
        current_preview_rects = preview_drawing.render_previews(
            preview_surf,
            game_state,
            preview_rect.topleft,  # Pass absolute top-left
            mode,
            self.env_config,
            self.vis_config,
            selected_shape_idx=selected_shape_idx,  # Pass selection state
        )
        # Update cache only if it changed (or first time)
        if not self.preview_rects or self.preview_rects != current_preview_rects:
            self.preview_rects = current_preview_rects


File: muzerotriangle\visualization\core\__init__.py


File: muzerotriangle\visualization\drawing\grid.py
# File: muzerotriangle/visualization/drawing/grid.py
from typing import TYPE_CHECKING

import pygame

# Import constants from the structs package directly
from ...structs import COLOR_ID_MAP, DEBUG_COLOR_ID, NO_COLOR_ID, Triangle
from ..core import colors, coord_mapper

if TYPE_CHECKING:
    from ...config import EnvConfig
    from ...environment import GridData


def draw_grid_background(surface: pygame.Surface, bg_color: tuple) -> None:
    """Fills the grid area surface with a background color."""
    surface.fill(bg_color)


def draw_grid_triangles(
    surface: pygame.Surface, grid_data: "GridData", config: "EnvConfig"
) -> None:
    """Draws all triangles (empty, occupied, death) on the grid surface using NumPy state."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Get direct references to NumPy arrays
    occupied_np = grid_data._occupied_np
    death_np = grid_data._death_np
    color_id_np = grid_data._color_id_np

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            is_death = death_np[r, c]
            is_occupied = occupied_np[r, c]
            color_id = color_id_np[r, c]
            is_up = (r + c) % 2 != 0  # Calculate orientation

            color: tuple[int, int, int] | None = None
            border_color = colors.GRID_LINE_COLOR
            border_width = 1

            if is_death:
                color = colors.DARK_GRAY
                border_color = colors.RED
            elif is_occupied:
                if color_id == DEBUG_COLOR_ID:
                    color = colors.DEBUG_TOGGLE_COLOR  # Special debug color
                elif color_id != NO_COLOR_ID and 0 <= color_id < len(COLOR_ID_MAP):
                    color = COLOR_ID_MAP[color_id]
                else:
                    # Fallback if occupied but no valid color ID (shouldn't happen)
                    color = colors.PURPLE  # Error color
            else:  # Empty playable cell
                color = colors.TRIANGLE_EMPTY_COLOR

            # Create temporary Triangle only for geometry calculation
            temp_tri = Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)

            if color:  # Should always be true unless error
                pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, border_color, pts, border_width)


def draw_grid_indices(
    surface: pygame.Surface,
    grid_data: "GridData",
    config: "EnvConfig",
    fonts: dict[str, pygame.font.Font | None],
) -> None:
    """Draws the index number inside each triangle, including death cells."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    font = fonts.get("help")
    if not font:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Get direct references to NumPy arrays
    occupied_np = grid_data._occupied_np
    death_np = grid_data._death_np
    color_id_np = grid_data._color_id_np

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            is_death = death_np[r, c]
            is_occupied = occupied_np[r, c]
            color_id = color_id_np[r, c]
            is_up = (r + c) % 2 != 0  # Calculate orientation

            # Create temporary Triangle only for geometry calculation
            temp_tri = Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            center_x = sum(p[0] for p in pts) / 3
            center_y = sum(p[1] for p in pts) / 3

            text_color = colors.WHITE  # Default

            if is_death:
                text_color = colors.LIGHT_GRAY
            elif is_occupied:
                bg_color: tuple[int, int, int] | None = None
                if color_id == DEBUG_COLOR_ID:
                    bg_color = colors.DEBUG_TOGGLE_COLOR
                elif color_id != NO_COLOR_ID and 0 <= color_id < len(COLOR_ID_MAP):
                    bg_color = COLOR_ID_MAP[color_id]

                if bg_color:
                    brightness = sum(bg_color) / 3
                    text_color = colors.WHITE if brightness < 128 else colors.BLACK
                else:  # Fallback if color missing
                    text_color = colors.RED
            else:  # Empty playable
                bg_color = colors.TRIANGLE_EMPTY_COLOR
                brightness = sum(bg_color) / 3
                text_color = colors.WHITE if brightness < 128 else colors.BLACK

            index = r * config.COLS + c
            text_surf = font.render(str(index), True, text_color)
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            surface.blit(text_surf, text_rect)


File: muzerotriangle\visualization\drawing\highlight.py
from typing import TYPE_CHECKING

import pygame

from ...structs import Triangle
from ..core import colors, coord_mapper

if TYPE_CHECKING:
    from ...config import EnvConfig


def draw_debug_highlight(
    surface: pygame.Surface, r: int, c: int, config: "EnvConfig"
) -> None:
    """Highlights a specific triangle border for debugging."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    is_up = (r + c) % 2 != 0
    temp_tri = Triangle(r, c, is_up)
    pts = temp_tri.get_points(ox, oy, cw, ch)

    pygame.draw.polygon(surface, colors.DEBUG_TOGGLE_COLOR, pts, 3)


File: muzerotriangle\visualization\drawing\hud.py
# File: muzerotriangle/visualization/drawing/hud.py
from typing import Any

import pygame

from ..core import colors


def render_hud(
    surface: pygame.Surface,
    mode: str,
    fonts: dict[str, pygame.font.Font | None],
    display_stats: dict[str, Any] | None = None,
) -> None:
    """
    Renders the Heads-Up Display (HUD) at the bottom of the screen.

    Args:
        surface: The Pygame surface to draw on.
        mode: The current mode ('play', 'debug', 'training_visual').
        fonts: A dictionary containing loaded Pygame fonts.
        display_stats: Optional dictionary containing stats to display (used in training mode).
    """
    screen_width, screen_height = surface.get_size()
    hud_height = 40  # Define HUD height or get from config if available
    hud_rect = pygame.Rect(0, screen_height - hud_height, screen_width, hud_height)
    pygame.draw.rect(surface, colors.GRAY, hud_rect)  # Draw HUD background

    font = fonts.get("help")
    if not font:
        return  # Cannot render text without font

    # Common text for all modes
    common_text = "[ESC] Quit"

    # Mode-specific text
    if mode == "play":
        mode_text = " | [Click] Select/Place Shape"
    elif mode == "debug":
        mode_text = " | [Click] Toggle Cell"
    elif mode == "training_visual":
        mode_text = " | Training Mode"  # Keep it simple for training view
    else:
        mode_text = ""

    full_text = common_text + mode_text

    # Render and blit the text
    text_surface = font.render(full_text, True, colors.WHITE)
    text_rect = text_surface.get_rect(center=hud_rect.center)
    surface.blit(text_surface, text_rect)

    # Display additional stats in training mode if provided
    if mode == "training_visual" and display_stats:
        stats_font = fonts.get("help") or font  # Use help font or fallback
        stats_text_parts = []

        # Example stats to display (customize as needed)
        if "global_step" in display_stats:
            stats_text_parts.append(f"Step: {display_stats['global_step']:,}")
        if "episodes_played" in display_stats:
            stats_text_parts.append(f"Eps: {display_stats['episodes_played']:,}")
        if "total_simulations_run" in display_stats:
            sims = display_stats["total_simulations_run"]
            sims_str = (
                f"{sims / 1e6:.1f}M"
                if sims >= 1e6
                else (f"{sims / 1e3:.0f}k" if sims >= 1000 else str(int(sims)))
            )
            stats_text_parts.append(f"Sims: {sims_str}")
        if "buffer_size" in display_stats and "buffer_capacity" in display_stats:
            stats_text_parts.append(
                f"Buffer: {display_stats['buffer_size']:,}/{display_stats['buffer_capacity']:,}"
            )
        if "num_active_workers" in display_stats and "num_workers" in display_stats:
            stats_text_parts.append(
                f"Workers: {display_stats['num_active_workers']}/{display_stats['num_workers']}"
            )

        stats_text = " | ".join(stats_text_parts)
        if stats_text:
            stats_surf = stats_font.render(stats_text, True, colors.YELLOW)
            # Position stats text to the left of the help text if space allows, otherwise below
            stats_rect = stats_surf.get_rect(
                midleft=(hud_rect.left + 10, hud_rect.centery)
            )
            if stats_rect.right > text_rect.left - 10:  # Check for overlap
                stats_rect.topleft = (
                    hud_rect.left + 10,
                    hud_rect.top + 2,
                )  # Position above if overlapping
                text_rect.topleft = (hud_rect.left + 10, stats_rect.bottom + 2)
            surface.blit(stats_surf, stats_rect)


File: muzerotriangle\visualization\drawing\previews.py
import logging
from typing import TYPE_CHECKING

import pygame

from ...structs import Shape, Triangle
from ..core import colors, coord_mapper
from .shapes import draw_shape

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig
    from ...environment import GameState

logger = logging.getLogger(__name__)


def render_previews(
    surface: pygame.Surface,
    game_state: "GameState",
    area_topleft: tuple[int, int],
    _mode: str,
    env_config: "EnvConfig",
    vis_config: "VisConfig",
    selected_shape_idx: int = -1,
) -> dict[int, pygame.Rect]:
    """Renders shape previews in their area. Returns dict {index: screen_rect}."""
    surface.fill(colors.PREVIEW_BG)
    preview_rects_screen: dict[int, pygame.Rect] = {}
    num_slots = env_config.NUM_SHAPE_SLOTS
    pad = vis_config.PREVIEW_PADDING
    inner_pad = vis_config.PREVIEW_INNER_PADDING
    border = vis_config.PREVIEW_BORDER_WIDTH
    selected_border = vis_config.PREVIEW_SELECTED_BORDER_WIDTH

    if num_slots <= 0:
        return {}

    # Calculate dimensions for each slot
    total_pad_h = (num_slots + 1) * pad
    available_h = surface.get_height() - total_pad_h
    slot_h = available_h / num_slots if num_slots > 0 else 0
    slot_w = surface.get_width() - 2 * pad

    current_y = float(pad)  # Start y position as float

    for i in range(num_slots):
        # Calculate local rectangle for the slot within the preview surface
        slot_rect_local = pygame.Rect(pad, int(current_y), int(slot_w), int(slot_h))
        # Calculate screen rectangle by offsetting local rect
        slot_rect_screen = slot_rect_local.move(area_topleft)
        preview_rects_screen[i] = (
            slot_rect_screen  # Store screen rect for interaction mapping
        )

        shape: Shape | None = game_state.shapes[i]
        # Use the passed selected_shape_idx for highlighting
        is_selected = selected_shape_idx == i

        # Determine border style based on selection
        border_width = selected_border if is_selected else border
        border_color = (
            colors.PREVIEW_SELECTED_BORDER if is_selected else colors.PREVIEW_BORDER
        )
        # Draw the border rectangle onto the local preview surface
        pygame.draw.rect(surface, border_color, slot_rect_local, border_width)

        # Draw the shape if it exists
        if shape:
            # Calculate drawing area inside the border and padding
            draw_area_w = slot_w - 2 * (border_width + inner_pad)
            draw_area_h = slot_h - 2 * (border_width + inner_pad)

            if draw_area_w > 0 and draw_area_h > 0:
                # Calculate shape bounding box and required cell size
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_rows = max_r - min_r + 1
                # Effective width considering triangle geometry (0.75 factor)
                shape_cols_eff = (
                    (max_c - min_c + 1) * 0.75 + 0.25 if shape.triangles else 1
                )

                # Determine cell size based on available space and shape dimensions
                scale_w = (
                    draw_area_w / shape_cols_eff if shape_cols_eff > 0 else draw_area_w
                )
                scale_h = draw_area_h / shape_rows if shape_rows > 0 else draw_area_h
                cell_size = max(1.0, min(scale_w, scale_h))  # Use the smaller scale

                # Calculate centered top-left position for drawing the shape
                shape_render_w = shape_cols_eff * cell_size
                shape_render_h = shape_rows * cell_size
                draw_topleft_x = (
                    slot_rect_local.left
                    + border_width
                    + inner_pad
                    + (draw_area_w - shape_render_w) / 2
                )
                draw_topleft_y = (
                    slot_rect_local.top
                    + border_width
                    + inner_pad
                    + (draw_area_h - shape_render_h) / 2
                )

                # Draw the shape onto the local preview surface
                # Cast float coordinates to int for draw_shape
                # Use _is_selected to match the function signature
                draw_shape(
                    surface,
                    shape,
                    (int(draw_topleft_x), int(draw_topleft_y)),
                    cell_size,
                    _is_selected=is_selected,
                    origin_offset=(
                        -min_r,
                        -min_c,
                    ),  # Adjust drawing origin based on bbox
                )

        # Move to the next slot position
        current_y += slot_h + pad

    return preview_rects_screen


def draw_placement_preview(
    surface: pygame.Surface,
    shape: "Shape",
    r: int,
    c: int,
    is_valid: bool,
    config: "EnvConfig",
) -> None:
    """Draws a semi-transparent shape snapped to the grid."""
    if not shape or not shape.triangles:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Use valid/invalid colors (could be passed in or defined here)
    base_color = (
        colors.PLACEMENT_VALID_COLOR[:3]
        if is_valid
        else colors.PLACEMENT_INVALID_COLOR[:3]
    )
    alpha = (
        colors.PLACEMENT_VALID_COLOR[3]
        if is_valid
        else colors.PLACEMENT_INVALID_COLOR[3]
    )
    color = list(base_color) + [alpha]  # Combine RGB and Alpha

    # Use a temporary surface for transparency
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))  # Fully transparent background

    for dr, dc, is_up in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        # Create a temporary Triangle to get points easily
        temp_tri = Triangle(tri_r, tri_c, is_up)
        pts = temp_tri.get_points(ox, oy, cw, ch)
        pygame.draw.polygon(temp_surface, color, pts)

    # Blit the transparent preview onto the main grid surface
    surface.blit(temp_surface, (0, 0))


def draw_floating_preview(
    surface: pygame.Surface,
    shape: "Shape",
    screen_pos: tuple[int, int],  # Position relative to the surface being drawn on
    _config: "EnvConfig",  # Mark config as unused
) -> None:
    """Draws a semi-transparent shape floating at the screen position."""
    if not shape or not shape.triangles:
        return

    cell_size = 20.0  # Fixed size for floating preview? Or scale based on config?
    color = list(shape.color) + [100]  # Base color with fixed alpha

    # Use a temporary surface for transparency
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))

    # Center the shape around the screen_pos
    min_r, min_c, max_r, max_c = shape.bbox()
    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0

    for dr, dc, is_up in shape.triangles:
        # Calculate position relative to shape center and screen_pos
        pt_x = screen_pos[0] + (dc - center_c) * (cell_size * 0.75)
        pt_y = screen_pos[1] + (dr - center_r) * cell_size

        # Create a temporary Triangle at origin to get relative points
        temp_tri = Triangle(0, 0, is_up)
        # Get points relative to 0,0 and scale
        rel_pts = temp_tri.get_points(0, 0, cell_size, cell_size)
        # Translate points to the calculated screen position
        pts = [(px + pt_x, py + pt_y) for px, py in rel_pts]
        pygame.draw.polygon(temp_surface, color, pts)

    # Blit the transparent preview onto the target surface
    surface.blit(temp_surface, (0, 0))


File: muzerotriangle\visualization\drawing\README.md
# File: muzerotriangle/visualization/drawing/README.md
# Visualization Drawing Submodule (`muzerotriangle.visualization.drawing`)

## Purpose and Architecture

This submodule contains specialized functions responsible for drawing specific visual elements of the game onto Pygame surfaces. These functions are typically called by the core renderers (`Visualizer`, `GameRenderer`) in [`muzerotriangle.visualization.core`](../core/README.md). Separating drawing logic makes the core renderers cleaner and promotes reusability of drawing code.

-   **[`grid.py`](grid.py):** Functions for drawing the grid background (`draw_grid_background`), the individual triangles within it colored based on occupancy/emptiness (`draw_grid_triangles`), and optional indices (`draw_grid_indices`). Uses `Triangle` from [`muzerotriangle.structs`](../../structs/README.md) and `GridData` from [`muzerotriangle.environment`](../../environment/README.md).
-   **[`shapes.py`](shapes.py):** Contains `draw_shape`, a function to render a given `Shape` object at a specific location on a surface (used primarily for previews). Uses `Shape` and `Triangle` from [`muzerotriangle.structs`](../../structs/README.md).
-   **[`previews.py`](previews.py):** Handles rendering related to shape previews:
    -   `render_previews`: Draws the dedicated preview area, including borders and the shapes within their slots, handling selection highlights. Uses `Shape` from [`muzerotriangle.structs`](../../structs/README.md).
    -   `draw_placement_preview`: Draws a semi-transparent version of a shape snapped to the grid, indicating a potential placement location (used in play mode hover). Uses `Shape` and `Triangle` from [`muzerotriangle.structs`](../../structs/README.md).
    -   `draw_floating_preview`: Draws a semi-transparent shape directly under the mouse cursor when hovering over the grid but not snapped (used in play mode hover). Uses `Shape` and `Triangle` from [`muzerotriangle.structs`](../../structs/README.md).
-   **[`hud.py`](hud.py):** `render_hud` draws Heads-Up Display elements like the game score, help text, and optional training statistics onto the main screen surface.
-   **[`highlight.py`](highlight.py):** `draw_debug_highlight` draws a distinct border around a specific triangle, used for visual feedback in debug mode. Uses `Triangle` from [`muzerotriangle.structs`](../../structs/README.md).
-   **[`utils.py`](utils.py):** Contains general drawing utility functions (currently empty).

## Exposed Interfaces

-   **Grid Drawing:**
    -   `draw_grid_background(surface: pygame.Surface, bg_color: tuple)`
    -   `draw_grid_triangles(surface: pygame.Surface, grid_data: GridData, config: EnvConfig)`
    -   `draw_grid_indices(surface: pygame.Surface, grid_data: GridData, config: EnvConfig, fonts: Dict[str, Optional[pygame.font.Font]])`
-   **Shape Drawing:**
    -   `draw_shape(surface: pygame.Surface, shape: Shape, topleft: Tuple[int, int], cell_size: float, is_selected: bool = False, origin_offset: Tuple[int, int] = (0, 0))`
-   **Preview Drawing:**
    -   `render_previews(surface: pygame.Surface, game_state: GameState, area_topleft: Tuple[int, int], mode: str, env_config: EnvConfig, vis_config: VisConfig, selected_shape_idx: int = -1) -> Dict[int, pygame.Rect]`
    -   `draw_placement_preview(surface: pygame.Surface, shape: Shape, r: int, c: int, is_valid: bool, config: EnvConfig)`
    -   `draw_floating_preview(surface: pygame.Surface, shape: Shape, screen_pos: Tuple[int, int], config: EnvConfig)`
-   **HUD Drawing:**
    -   `render_hud(surface: pygame.Surface, mode: str, fonts: Dict[str, Optional[pygame.font.Font]], display_stats: Optional[Dict[str, Any]] = None)`
-   **Highlight Drawing:**
    -   `draw_debug_highlight(surface: pygame.Surface, r: int, c: int, config: EnvConfig)`
-   **Utility Functions:**
    -   (Currently empty or contains other drawing-specific utils)

## Dependencies

-   **[`muzerotriangle.visualization.core`](../core/README.md)**:
    -   `colors`: Used extensively for drawing colors.
    -   `coord_mapper`: Used internally (e.g., by `draw_placement_preview`) or relies on its calculations passed in.
-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `EnvConfig`, `VisConfig`: Provide dimensions, padding, etc., needed for drawing calculations.
-   **[`muzerotriangle.environment`](../../environment/README.md)**:
    -   `GameState`, `GridData`: Provide the data to be drawn.
-   **[`muzerotriangle.structs`](../../structs/README.md)**:
    -   Uses `Triangle`, `Shape`, `COLOR_ID_MAP`, `DEBUG_COLOR_ID`, `NO_COLOR_ID`.
-   **[`muzerotriangle.visualization.ui`](../ui/README.md)**:
    -   `ProgressBar` (used by `hud.py`).
-   **`pygame`**:
    -   The core library used for all drawing operations (`pygame.draw.polygon`, `surface.fill`, `surface.blit`, etc.) and font rendering.
-   **Standard Libraries:** `typing`, `logging`, `math`.

---

**Note:** Please keep this README updated when adding new drawing functions, modifying existing ones, or changing their dependencies on configuration or environment data structures. Accurate documentation is crucial for maintainability.

File: muzerotriangle\visualization\drawing\shapes.py
import pygame

from ...structs import Shape, Triangle
from ..core import colors


def draw_shape(
    surface: pygame.Surface,
    shape: Shape,
    topleft: tuple[int, int],
    cell_size: float,
    _is_selected: bool = False,
    origin_offset: tuple[int, int] = (0, 0),
) -> None:
    """Draws a single shape onto a surface."""
    if not shape or not shape.triangles or cell_size <= 0:
        return

    shape_color = shape.color
    border_color = colors.GRAY

    cw = cell_size
    ch = cell_size

    for dr, dc, is_up in shape.triangles:
        adj_r, adj_c = dr + origin_offset[0], dc + origin_offset[1]

        tri_x = topleft[0] + adj_c * (cw * 0.75)
        tri_y = topleft[1] + adj_r * ch

        temp_tri = Triangle(0, 0, is_up)
        pts = [(px + tri_x, py + tri_y) for px, py in temp_tri.get_points(0, 0, cw, ch)]

        pygame.draw.polygon(surface, shape_color, pts)
        pygame.draw.polygon(surface, border_color, pts, 1)


File: muzerotriangle\visualization\drawing\utils.py


File: muzerotriangle\visualization\drawing\__init__.py


File: muzerotriangle\visualization\ui\progress_bar.py
# File: muzerotriangle/visualization/ui/progress_bar.py
# Changes:
# - Modify render text logic: If info_line is provided, prepend default progress info.
# - Cast return type of _get_pulsing_color to satisfy mypy.

import math
import random
import time
from typing import Any, cast  # Added cast

import pygame

from ...utils import format_eta
from ..core import colors


class ProgressBar:
    """
    A reusable progress bar component for visualization.
    Handles overflow by cycling colors and displaying actual progress count.
    Includes rounded corners and subtle pulsing effect.
    Can display a custom info line alongside default progress text.
    """

    def __init__(
        self,
        entity_title: str,
        total_steps: int,
        start_time: float | None = None,
        initial_steps: int = 0,
        initial_color: tuple[int, int, int] = colors.BLUE,
    ):
        self.entity_title = entity_title
        self._original_total_steps = max(
            1, total_steps if total_steps is not None else 1
        )
        self.initial_steps = max(0, initial_steps)
        self.current_steps = self.initial_steps
        self.start_time = start_time if start_time is not None else time.time()
        self._last_step_time = self.start_time
        self._step_times: list[float] = []
        self.extra_data: dict[str, Any] = {}
        self._current_bar_color = initial_color
        self._last_cycle = -1
        self._rng = random.Random()
        self._pulse_phase = random.uniform(0, 2 * math.pi)

    def add_steps(self, steps_added: int):
        """Adds steps to the progress bar's current count."""
        if steps_added <= 0:
            return
        self.current_steps += steps_added
        self._check_color_cycle()

    def set_current_steps(self, steps: int):
        """Directly sets the current step count."""
        self.current_steps = max(0, steps)
        self._check_color_cycle()

    def _check_color_cycle(self):
        """Updates the bar color if a new cycle is reached."""
        current_cycle = self.current_steps // self._original_total_steps
        if current_cycle > self._last_cycle:
            self._last_cycle = current_cycle
            if current_cycle > 0:
                available_colors = [
                    c
                    for c in colors.PROGRESS_BAR_CYCLE_COLORS
                    if c != self._current_bar_color
                ]
                if not available_colors:
                    available_colors = colors.PROGRESS_BAR_CYCLE_COLORS
                if available_colors:
                    self._current_bar_color = self._rng.choice(available_colors)

    def update_extra_data(self, data: dict[str, Any]):
        """Updates or adds key-value pairs to display."""
        self.extra_data.update(data)

    def reset_time(self):
        """Resets the start time to now, keeping current steps."""
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.initial_steps = self.current_steps

    def reset_all(self, new_total_steps: int | None = None):
        """Resets steps to 0 and start time to now. Optionally updates total steps."""
        self.current_steps = 0
        self.initial_steps = 0
        if new_total_steps is not None:
            self._original_total_steps = max(1, new_total_steps)
        self.start_time = time.time()
        self._last_step_time = self.start_time
        self._step_times = []
        self.extra_data = {}
        self._last_cycle = -1
        self._current_bar_color = (
            colors.PROGRESS_BAR_CYCLE_COLORS[0]
            if colors.PROGRESS_BAR_CYCLE_COLORS
            else colors.BLUE
        )

    def get_progress_fraction(self) -> float:
        """Returns progress within the current cycle as a fraction (0.0 to 1.0)."""
        if self._original_total_steps <= 1:
            return 1.0
        if self.current_steps == 0:
            return 0.0
        progress_in_cycle = self.current_steps % self._original_total_steps
        if progress_in_cycle == 0 and self.current_steps > 0:
            return 1.0
        else:
            return progress_in_cycle / self._original_total_steps

    def get_elapsed_time(self) -> float:
        """Returns the time elapsed since the start time."""
        return time.time() - self.start_time

    def get_eta_seconds(self) -> float | None:
        """Calculates the estimated time remaining in seconds."""
        if (
            self._original_total_steps <= 1
            or self.current_steps >= self._original_total_steps
        ):
            return None
        steps_processed = self.current_steps - self.initial_steps
        if steps_processed <= 0:
            return None
        elapsed = self.get_elapsed_time()
        if elapsed < 1.0:
            return None
        speed = steps_processed / elapsed
        if speed < 1e-6:
            return None
        remaining_steps = self._original_total_steps - self.current_steps
        if remaining_steps <= 0:
            return 0.0
        eta = remaining_steps / speed
        return eta

    def _get_pulsing_color(self) -> tuple[int, int, int]:
        """Applies a subtle brightness pulse to the current bar color."""
        base_color = self._current_bar_color
        pulse_amplitude = 15
        brightness_offset = int(
            pulse_amplitude * math.sin(time.time() * 4 + self._pulse_phase)
        )
        # --- CHANGED: Explicitly cast to tuple[int, int, int] ---
        pulsed_color = cast(
            "tuple[int, int, int]",
            tuple(max(0, min(255, c + brightness_offset)) for c in base_color),
        )
        # --- END CHANGED ---
        return pulsed_color

    def render(
        self,
        surface: pygame.Surface,
        position: tuple[int, int],
        width: int,
        height: int,
        font: pygame.font.Font,
        bg_color: tuple[int, int, int] = colors.DARK_GRAY,
        text_color: tuple[int, int, int] = colors.WHITE,
        border_width: int = 1,
        border_color: tuple[int, int, int] = colors.GRAY,
        border_radius: int = 3,
        info_line: str | None = None,  # Keep optional info_line
    ):
        """Draws the progress bar onto the given surface."""
        x, y = position
        progress_fraction = self.get_progress_fraction()

        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, bg_color, bg_rect, border_radius=border_radius)

        # Pulsing Bar Fill
        fill_width = int(width * progress_fraction)
        if fill_width > 0:
            fill_width = min(width, fill_width)
            fill_rect = pygame.Rect(x, y, fill_width, height)
            pulsing_bar_color = self._get_pulsing_color()
            pygame.draw.rect(
                surface, pulsing_bar_color, fill_rect, border_radius=border_radius
            )

        # Border
        if border_width > 0:
            pygame.draw.rect(
                surface,
                border_color,
                bg_rect,
                border_width,
                border_radius=border_radius,
            )

        # --- Text Rendering (Modified) ---
        line_height = font.get_height()
        if height >= line_height + 4:
            # Always generate default progress text parts
            elapsed_time_str = format_eta(self.get_elapsed_time())
            eta_seconds = self.get_eta_seconds()
            eta_str = format_eta(eta_seconds) if eta_seconds is not None else "--"
            processed_steps = self.current_steps
            expected_steps = self._original_total_steps
            processed_str = (
                f"{processed_steps / 1e6:.1f}M"
                if processed_steps >= 1e6
                else (
                    f"{processed_steps / 1e3:.0f}k"
                    if processed_steps >= 1000
                    else f"{processed_steps:,}"
                )
            )
            expected_str = (
                f"{expected_steps / 1e6:.1f}M"
                if expected_steps >= 1e6
                else (
                    f"{expected_steps / 1e3:.0f}k"
                    if expected_steps >= 1000
                    else f"{expected_steps:,}"
                )
            )
            progress_text = f"{processed_str}/{expected_str}"
            if self._original_total_steps <= 1:
                progress_text = f"{processed_str}"
            extra_text = ""
            if self.extra_data:
                extra_items = [f"{k}:{v}" for k, v in self.extra_data.items()]
                extra_text = " | " + " ".join(extra_items)

            # Construct the display text
            default_progress_info = f"{self.entity_title}: {progress_text} (T:{elapsed_time_str} ETA:{eta_str}){extra_text}"

            # --- CHANGED: Prepend default info if info_line is given ---
            if info_line is not None:
                display_text = (
                    f"{default_progress_info} || {info_line}"  # Combine with separator
                )
            else:
                display_text = default_progress_info  # Use only default
            # --- END CHANGED ---

            # Truncate if necessary
            max_text_width = width - 10
            if font.size(display_text)[0] > max_text_width:
                while (
                    font.size(display_text + "...")[0] > max_text_width
                    and len(display_text) > 20
                ):
                    display_text = display_text[:-1]
                display_text += "..."

            # Render and blit centered vertically
            text_surf = font.render(display_text, True, text_color)
            text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
            surface.blit(text_surf, text_rect)
        # --- End Text Rendering ---


File: muzerotriangle\visualization\ui\__init__.py
"""
UI Components subpackage for visualization.
Contains reusable UI elements like progress bars, buttons, etc.
"""

from .progress_bar import ProgressBar

__all__ = [
    "ProgressBar",
]


File: tests\conftest.py
# File: tests/conftest.py
# File: tests/conftest.py
import random
from typing import cast

import numpy as np
import pytest
import torch
import torch.optim as optim

from muzerotriangle.config import EnvConfig, MCTSConfig, ModelConfig, TrainConfig
from muzerotriangle.nn import NeuralNetwork
from muzerotriangle.rl import ExperienceBuffer, Trainer

# Import MuZero Types
from muzerotriangle.utils.types import StateType, Trajectory, TrajectoryStep

# REMOVED: Experience

rng = np.random.default_rng()


# --- Fixtures --- (mock_env_config, mock_model_config, mock_train_config, mock_mcts_config remain the same) ---
@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    rows = 3
    cols = 3
    cols_per_row = [cols] * rows
    return EnvConfig(
        ROWS=rows,
        COLS=cols,
        COLS_PER_ROW=cols_per_row,
        NUM_SHAPE_SLOTS=1,
        MIN_LINE_LENGTH=3,
    )


@pytest.fixture(scope="session")
def mock_model_config(mock_env_config: EnvConfig) -> ModelConfig:
    int(mock_env_config.ACTION_DIM)  # type: ignore[call-overload]
    return ModelConfig(
        GRID_INPUT_CHANNELS=1,
        OTHER_NN_INPUT_FEATURES_DIM=10,
        HIDDEN_STATE_DIM=32,
        ACTION_ENCODING_DIM=8,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=False,
        CONV_FILTERS=[4],
        CONV_KERNEL_SIZES=[3],
        CONV_STRIDES=[1],
        CONV_PADDING=[1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=4,
        USE_TRANSFORMER_IN_REP=False,
        REP_FC_DIMS_AFTER_ENCODER=[],
        DYNAMICS_NUM_RESIDUAL_BLOCKS=1,
        REWARD_HEAD_DIMS=[16],
        REWARD_SUPPORT_SIZE=5,
        PREDICTION_NUM_RESIDUAL_BLOCKS=1,
        POLICY_HEAD_DIMS=[16],
        VALUE_HEAD_DIMS=[16],
        NUM_VALUE_ATOMS=11,
        VALUE_MIN=-5.0,
        VALUE_MAX=5.0,
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    # Use MuZero defaults for the base mock config
    return TrainConfig(
        BATCH_SIZE=4,
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        USE_PER=True,  # Enable PER for testing
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        MAX_TRAINING_STEPS=200,
        MUZERO_UNROLL_STEPS=3,
        N_STEP_RETURNS=5,
        DISCOUNT=0.99,
    )


@pytest.fixture(scope="session")
def mock_mcts_config() -> MCTSConfig:
    # Add discount here
    return MCTSConfig(
        num_simulations=10,
        puct_coefficient=1.5,
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_anneal_steps=5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_search_depth=10,
        discount=0.99,
    )


@pytest.fixture(scope="session")
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape, dtype=np.float32),
        "other_features": rng.random(other_shape, dtype=np.float32),
    }


# --- ADD MuZero Data Fixtures ---
@pytest.fixture(scope="session")
def mock_trajectory_step_global(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> TrajectoryStep:
    """Creates a single mock TrajectoryStep (session scoped)."""
    action_dim = int(mock_env_config.ACTION_DIM)
    return {
        "observation": mock_state_type,
        "action": random.randint(0, action_dim - 1) if action_dim > 0 else 0,
        "reward": random.uniform(-1, 1),
        "policy_target": (
            dict.fromkeys(range(action_dim), 1.0 / action_dim) if action_dim > 0 else {}
        ),
        "value_target": random.uniform(-1, 1),
        "n_step_reward_target": random.uniform(-1, 1),  # Add n-step target
        "hidden_state": None,
    }


@pytest.fixture(scope="session")
def mock_trajectory_global(mock_trajectory_step_global: TrajectoryStep) -> Trajectory:
    """Creates a mock Trajectory (session scoped)."""
    return [mock_trajectory_step_global.copy() for _ in range(10)]


# --- END ADD MuZero Data Fixtures ---


@pytest.fixture(scope="session")
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    device = torch.device("cpu")
    nn = NeuralNetwork(mock_model_config, mock_env_config, mock_train_config, device)
    return nn


@pytest.fixture(scope="session")
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture(scope="session")
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    return cast("optim.Optimizer", mock_trainer.optimizer)


@pytest.fixture
def mock_experience_buffer(mock_train_config: TrainConfig) -> ExperienceBuffer:
    return ExperienceBuffer(mock_train_config)


@pytest.fixture
def filled_mock_buffer(
    mock_experience_buffer: ExperienceBuffer, mock_trajectory_global: Trajectory
) -> ExperienceBuffer:
    """Provides a buffer filled with mock trajectories."""
    while mock_experience_buffer.total_steps < mock_experience_buffer.min_size_to_train:
        mock_experience_buffer.add(mock_trajectory_global[:])
    return mock_experience_buffer


File: tests\__init__.py


File: tests\environment\conftest.py
import pytest

# Use relative imports for muzerotriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from muzerotriangle.config import EnvConfig
    from muzerotriangle.environment import GameState
    from muzerotriangle.structs import Shape
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from muzerotriangle.config import EnvConfig
    from muzerotriangle.environment import GameState
    from muzerotriangle.structs import Shape


# Use session-scoped config from top-level conftest
@pytest.fixture(scope="session")
def default_env_config() -> EnvConfig:
    """Provides the default EnvConfig used in the specification (session-scoped)."""
    # Pydantic models with defaults can be instantiated without args
    return EnvConfig()


@pytest.fixture
def game_state(default_env_config: EnvConfig) -> GameState:
    """Provides a fresh GameState instance for testing."""
    # Use a fixed seed for reproducibility within tests if needed
    return GameState(config=default_env_config, initial_seed=123)


@pytest.fixture
def game_state_with_fixed_shapes(default_env_config: EnvConfig) -> GameState:
    """
    Provides a game state with predictable initial shapes.
    Uses a modified EnvConfig with NUM_SHAPE_SLOTS=3 for this specific fixture.
    """
    # Create a specific config for this fixture
    config_3_slots = default_env_config.model_copy(update={"NUM_SHAPE_SLOTS": 3})
    gs = GameState(config=config_3_slots, initial_seed=456)

    # Override the random shapes with fixed ones for testing placement/refill
    fixed_shapes = [
        Shape([(0, 0, False)], (255, 0, 0)),  # Single down (matches grid at 0,0)
        Shape([(0, 0, True)], (0, 255, 0)),  # Single up (matches grid at 0,1)
        Shape(
            [(0, 0, False), (0, 1, True)], (0, 0, 255)
        ),  # Domino (matches grid at 0,0 and 0,1)
    ]
    # This fixture now guarantees NUM_SHAPE_SLOTS is 3
    assert len(fixed_shapes) == gs.env_config.NUM_SHAPE_SLOTS

    for i in range(len(fixed_shapes)):
        gs.shapes[i] = fixed_shapes[i]
    return gs


@pytest.fixture
def simple_shape() -> Shape:
    """Provides a simple 3-triangle shape (Down, Up, Down)."""
    # Example: L-shape (Down at 0,0; Up at 1,0; Down at 1,1 relative)
    # Grid at (r,c) is Down if r+c is even, Up if odd.
    # (0,0) is Down. (1,0) is Up. (1,1) is Down. This shape matches grid orientation.
    triangles = [(0, 0, False), (1, 0, True), (1, 1, False)]
    color = (255, 0, 0)
    return Shape(triangles, color)


File: tests\environment\test_actions.py
# File: tests/environment/test_actions.py
import pytest

from muzerotriangle.config import EnvConfig
from muzerotriangle.environment.core.action_codec import decode_action
from muzerotriangle.environment.core.game_state import GameState

# Import GridLogic correctly
from muzerotriangle.environment.grid import logic as GridLogic
from muzerotriangle.environment.logic import actions as ActionLogic
from muzerotriangle.structs import Shape

# Fixtures are now implicitly injected from tests/environment/conftest.py


@pytest.fixture
def game_state_almost_full(default_env_config: EnvConfig) -> GameState:
    """
    Provides a game state where only a few placements are possible.
    Grid is filled completely, then specific spots are made empty.
    """
    # Use a fresh GameState to avoid side effects from other tests using the shared 'game_state' fixture
    gs = GameState(config=default_env_config, initial_seed=987)
    # Fill the entire playable grid first using NumPy arrays
    playable_mask = ~gs.grid_data._death_np
    gs.grid_data._occupied_np[playable_mask] = True

    # Explicitly make specific spots empty: (0,4) [Down] and (0,5) [Up]
    empty_spots = [(0, 4), (0, 5)]
    for r_empty, c_empty in empty_spots:
        if gs.grid_data.valid(r_empty, c_empty):
            gs.grid_data._occupied_np[r_empty, c_empty] = False
            # Reset color ID as well
            gs.grid_data._color_id_np[
                r_empty, c_empty
            ] = -1  # Assuming -1 is NO_COLOR_ID
    return gs


# --- Test Action Logic ---
def test_get_valid_actions_initial(game_state: GameState):
    """Test valid actions in an initial empty state."""
    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert isinstance(valid_actions, list)
    assert len(valid_actions) > 0  # Should be many valid actions initially

    # Check if decoded actions are valid placements
    for action_index in valid_actions[:10]:  # Check a sample
        shape_idx, r, c = decode_action(action_index, game_state.env_config)
        shape = game_state.shapes[shape_idx]
        assert shape is not None
        assert GridLogic.can_place(game_state.grid_data, shape, r, c)


def test_get_valid_actions_almost_full(game_state_almost_full: GameState):
    """Test valid actions in a nearly full state with only (0,4) and (0,5) free."""
    gs = game_state_almost_full
    # Ensure cells (0,4) and (0,5) are indeed empty using NumPy array
    assert not gs.grid_data._occupied_np[0, 4], "Cell (0,4) should be empty"
    assert not gs.grid_data._occupied_np[0, 5], "Cell (0,5) should be empty"
    # Check orientation (calculated) - Apply Ruff suggestion
    assert (0 + 4) % 2 == 0, "Cell (0,4) should be Down"  # Changed from not (... != 0)
    assert (0 + 5) % 2 != 0, "Cell (0,5) should be Up"

    # Single down triangle fits at (0,4) [which is Down]
    gs.shapes[0] = Shape([(0, 0, False)], (255, 0, 0))
    # Single up triangle fits at (0,5) [which is Up]
    gs.shapes[1] = Shape([(0, 0, True)], (0, 255, 0))
    # Make other slots empty or contain unfittable shapes
    if len(gs.shapes) > 2:
        gs.shapes[2] = Shape([(0, 0, False), (1, 0, False)], (0, 0, 255))  # Unfittable
    if len(gs.shapes) > 3:
        gs.shapes[3] = None

    valid_actions = ActionLogic.get_valid_actions(gs)

    # Expect fewer valid actions
    assert isinstance(valid_actions, list)
    # The setup should allow placing shape 0 at (0,4) and shape 1 at (0,5)
    assert len(valid_actions) == 2, (
        f"Expected 2 valid actions, found {len(valid_actions)}. Actions: {valid_actions}"
    )

    expected_placements = {(0, 0, 4), (1, 0, 5)}  # (shape_idx, r, c)
    found_placements = set()

    # Check if decoded actions are valid placements in the few remaining spots
    for action_index in valid_actions:
        shape_idx, r, c = decode_action(action_index, gs.env_config)
        shape = gs.shapes[shape_idx]
        assert shape is not None, f"Shape at index {shape_idx} is None"
        assert GridLogic.can_place(gs.grid_data, shape, r, c), (
            f"can_place returned False for action {action_index} -> shape_idx={shape_idx}, r={r}, c={c}"
        )
        # Check if placement is in the expected empty area
        is_expected_placement = (r == 0 and c == 4 and shape_idx == 0) or (
            r == 0 and c == 5 and shape_idx == 1
        )
        assert is_expected_placement, (
            f"Action {action_index} -> {(shape_idx, r, c)} is not one of the expected placements (0,0,4) or (1,0,5)"
        )
        found_placements.add((shape_idx, r, c))

    assert found_placements == expected_placements


def test_get_valid_actions_no_shapes(game_state: GameState):
    """Test valid actions when no shapes are available."""
    game_state.shapes = [None] * game_state.env_config.NUM_SHAPE_SLOTS
    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert valid_actions == []


def test_get_valid_actions_no_space(game_state: GameState):
    """Test valid actions when the grid is completely full (or no space for any shape)."""
    # Fill the entire playable grid using NumPy arrays
    playable_mask = ~game_state.grid_data._death_np
    game_state.grid_data._occupied_np[playable_mask] = True

    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert valid_actions == []


File: tests\environment\test_grid_logic.py
# File: tests/environment/test_grid_logic.py
import pytest

# Use relative imports for muzerotriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from muzerotriangle.config import EnvConfig
    from muzerotriangle.environment.grid import GridData
    from muzerotriangle.environment.grid import logic as GridLogic
    from muzerotriangle.structs import Shape
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from muzerotriangle.config import EnvConfig
    from muzerotriangle.environment.grid import GridData
    from muzerotriangle.environment.grid import logic as GridLogic
    from muzerotriangle.structs import Shape

# Use shared fixtures implicitly via pytest injection
# from .conftest import game_state, simple_shape # Import fixtures if needed


@pytest.fixture
def grid_data(default_env_config: EnvConfig) -> GridData:
    """Provides a fresh GridData instance."""
    return GridData(config=default_env_config)


# --- Test can_place with NumPy GridData ---
def test_can_place_empty_grid(grid_data: GridData, simple_shape: Shape):
    """Test placement on an empty grid."""
    # Place at (2,2). Grid(2,2) is Down (2+2=4, even). Shape(0,0) is Down. OK.
    # Grid(3,2) is Up (3+2=5, odd). Shape(1,0) is Up. OK.
    # Grid(3,3) is Down (3+3=6, even). Shape(1,1) is Down. OK.
    assert GridLogic.can_place(grid_data, simple_shape, 2, 2)


def test_can_place_occupied(grid_data: GridData, simple_shape: Shape):
    """Test placement fails if any target cell is occupied."""
    # Occupy one cell where the shape would go
    target_r, target_c = 3, 2
    grid_data._occupied_np[target_r, target_c] = True
    assert not GridLogic.can_place(grid_data, simple_shape, 2, 2)


# Remove unused simple_shape argument
def test_can_place_death_zone(grid_data: GridData):
    """Test placement fails if any target cell is in a death zone."""
    # Find a death zone cell (e.g., top-left corner in default config)
    death_r, death_c = 0, 0
    assert grid_data._death_np[death_r, death_c]
    # Try placing a single triangle shape there
    single_down_shape = Shape([(0, 0, False)], (255, 0, 0))
    assert not GridLogic.can_place(grid_data, single_down_shape, death_r, death_c)


def test_can_place_orientation_mismatch(grid_data: GridData):
    """Test placement fails if triangle orientations don't match."""
    # Shape: Single UP triangle at its origin (0,0)
    shape_up = Shape([(0, 0, True)], (0, 255, 0))
    # Target grid cell: (0,4), which is DOWN in default config (0+4=4, even)
    target_r_down, target_c_down = 0, 4
    assert grid_data.valid(target_r_down, target_c_down) and not grid_data.is_death(
        target_r_down, target_c_down
    )
    assert not GridLogic.can_place(grid_data, shape_up, target_r_down, target_c_down)

    # Shape: Single DOWN triangle at its origin (0,0)
    shape_down = Shape([(0, 0, False)], (255, 0, 0))
    # Target grid cell: (0,3), which is UP in default config (0+3=3, odd)
    target_r_up, target_c_up = 0, 3
    assert grid_data.valid(target_r_up, target_c_up) and not grid_data.is_death(
        target_r_up, target_c_up
    )
    assert not GridLogic.can_place(grid_data, shape_down, target_r_up, target_c_up)

    # Test valid placement using playable coordinates
    assert GridLogic.can_place(grid_data, shape_down, 0, 4)  # Down on Down at (0,4)
    assert GridLogic.can_place(grid_data, shape_up, 0, 3)  # Up on Up at (0,3)


# --- Test check_and_clear_lines with NumPy GridData ---


def occupy_coords(grid_data: GridData, coords: set[tuple[int, int]]):
    """Helper to occupy specific coordinates."""
    for r, c in coords:
        if grid_data.valid(r, c) and not grid_data._death_np[r, c]:
            grid_data._occupied_np[r, c] = True


def test_check_and_clear_lines_no_clear(grid_data: GridData):
    """Test when newly occupied cells don't complete any lines."""
    newly_occupied = {(2, 2), (3, 2), (3, 3)}  # Coords from simple_shape placement
    occupy_coords(grid_data, newly_occupied)
    lines_cleared, unique_cleared, cleared_lines_set = GridLogic.check_and_clear_lines(
        grid_data, newly_occupied
    )
    assert lines_cleared == 0
    assert not unique_cleared
    assert not cleared_lines_set
    # Check grid state unchanged (except for initial occupation)
    assert grid_data._occupied_np[2, 2]
    assert grid_data._occupied_np[3, 2]
    assert grid_data._occupied_np[3, 3]


def test_check_and_clear_lines_single_line(grid_data: GridData):
    """Test clearing a single horizontal line."""
    # Find a valid horizontal line from the precomputed set
    # Example: Look for a line in row 1 (often has long lines)
    expected_line_coords = None
    for line_fs in grid_data.potential_lines:
        coords = list(line_fs)
        # Check if it's horizontal and in row 1
        if len(coords) >= grid_data.config.MIN_LINE_LENGTH and all(
            r == 1 for r, c in coords
        ):
            expected_line_coords = frozenset(coords)
            break

    assert expected_line_coords is not None, (
        "Could not find a suitable horizontal line in row 1 for testing"
    )
    # line_len = len(expected_line_coords) # Removed unused variable
    coords_list = list(expected_line_coords)

    # Occupy all but one cell in the line
    occupy_coords(grid_data, set(coords_list[:-1]))
    # Occupy the last cell
    last_coord = coords_list[-1]
    newly_occupied = {last_coord}
    occupy_coords(grid_data, newly_occupied)

    lines_cleared, unique_cleared, cleared_lines_set = GridLogic.check_and_clear_lines(
        grid_data, newly_occupied
    )

    assert lines_cleared == 1
    assert unique_cleared == set(expected_line_coords)  # Expect set of coords
    assert cleared_lines_set == {
        expected_line_coords
    }  # Expect set of frozensets of coords

    # Verify the line is now empty in the NumPy array
    for r, c in expected_line_coords:
        assert not grid_data._occupied_np[r, c]


File: tests\environment\test_shape_logic.py
# File: tests/environment/test_shape_logic.py
import random

import pytest

from muzerotriangle.environment import GameState
from muzerotriangle.environment.shapes import logic as ShapeLogic
from muzerotriangle.structs import Shape

# Fixtures are now implicitly injected from tests/environment/conftest.py


@pytest.fixture
def fixed_rng() -> random.Random:
    """Provides a Random instance with a fixed seed."""
    return random.Random(12345)


def test_generate_random_shape(fixed_rng: random.Random):
    """Test generating a single random shape."""
    shape = ShapeLogic.generate_random_shape(fixed_rng)
    assert isinstance(shape, Shape)
    assert shape.triangles is not None
    assert shape.color is not None
    assert len(shape.triangles) > 0
    # Check connectivity (optional but good)
    assert ShapeLogic.is_shape_connected(shape)


def test_generate_multiple_shapes(fixed_rng: random.Random):
    """Test generating multiple shapes to ensure variety (or lack thereof with fixed seed)."""
    shape1 = ShapeLogic.generate_random_shape(fixed_rng)
    # Re-seed or use different rng instance if true randomness is needed per call
    # For this test, using the same fixed_rng will likely produce the same shape again
    shape2 = ShapeLogic.generate_random_shape(fixed_rng)
    # --- REMOVED INCORRECT ASSERTION ---
    # assert shape1 == shape2  # Expect same shape due to fixed seed - THIS IS INCORRECT
    # --- END REMOVED ---
    # Check that subsequent calls produce different results with the same RNG instance
    assert shape1 != shape2, (
        "Two consecutive calls with the same RNG produced the exact same shape (template and color), which is highly unlikely."
    )

    # Use a different seed for variation
    rng2 = random.Random(54321)
    shape3 = ShapeLogic.generate_random_shape(rng2)
    # Check that different RNGs produce different results (highly likely)
    assert shape1 != shape3 or shape1.color != shape3.color


def test_refill_shape_slots_empty(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when all slots are initially empty."""
    game_state.shapes = [None] * game_state.env_config.NUM_SHAPE_SLOTS
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)
    assert len(game_state.shapes) == game_state.env_config.NUM_SHAPE_SLOTS


def test_refill_shape_slots_partial(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when some slots are empty - SHOULD NOT REFILL."""
    num_slots = game_state.env_config.NUM_SHAPE_SLOTS
    if num_slots < 2:
        pytest.skip("Test requires at least 2 shape slots")

    # Start with full slots
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)

    # Empty one slot
    game_state.shapes[0] = None
    # Store original state (important: copy shapes if they are mutable)
    original_shapes = [s.copy() if s else None for s in game_state.shapes]

    # Attempt refill - it should do nothing
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)

    # Check that shapes remain unchanged
    assert game_state.shapes == original_shapes, "Refill happened unexpectedly"


def test_refill_shape_slots_full(game_state: GameState, fixed_rng: random.Random):
    """Test refilling when all slots are already full - SHOULD NOT REFILL."""
    # Start with full slots
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)
    assert all(s is not None for s in game_state.shapes)
    original_shapes = [s.copy() if s else None for s in game_state.shapes]

    # Attempt refill - should do nothing
    ShapeLogic.refill_shape_slots(game_state, fixed_rng)

    # Check shapes are unchanged
    assert game_state.shapes == original_shapes, "Refill happened when slots were full"


def test_refill_shape_slots_batch_trigger(game_state: GameState) -> None:
    """Test that refill only happens when ALL slots are empty."""
    num_slots = game_state.env_config.NUM_SHAPE_SLOTS
    if num_slots < 2:
        pytest.skip("Test requires at least 2 shape slots")

    # Fill all slots initially
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    initial_shapes = [s.copy() if s else None for s in game_state.shapes]
    assert all(s is not None for s in initial_shapes)

    # Empty one slot - refill should NOT happen
    game_state.shapes[0] = None
    shapes_after_one_empty = [s.copy() if s else None for s in game_state.shapes]
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    assert game_state.shapes == shapes_after_one_empty, (
        "Refill happened when only one slot was empty"
    )

    # Empty all slots - refill SHOULD happen
    game_state.shapes = [None] * num_slots
    ShapeLogic.refill_shape_slots(game_state, game_state._rng)
    assert all(s is not None for s in game_state.shapes), (
        "Refill did not happen when all slots were empty"
    )
    # Check that the shapes are different from the initial ones (probabilistically)
    assert game_state.shapes != initial_shapes, (
        "Shapes after refill are identical to initial shapes (unlikely)"
    )


File: tests\environment\test_step.py
# File: tests/environment/test_step.py
import random
from time import sleep

import pytest

# Import mocker fixture from pytest-mock
from pytest_mock import MockerFixture

from muzerotriangle.config import EnvConfig
from muzerotriangle.environment.core.game_state import GameState
from muzerotriangle.environment.grid import (
    logic as GridLogic,
)
from muzerotriangle.environment.grid.grid_data import GridData
from muzerotriangle.environment.logic.step import calculate_reward, execute_placement
from muzerotriangle.structs import Shape, Triangle

# Fixtures are now implicitly injected from tests/environment/conftest.py


def occupy_line(
    grid_data: GridData, line_indices: list[int], config: EnvConfig
) -> set[Triangle]:
    """Helper to occupy triangles for a given line index list."""
    # occupied_tris: set[Triangle] = set() # Removed unused variable
    for idx in line_indices:
        r, c = divmod(idx, config.COLS)
        # Combine nested if using 'and'
        if grid_data.valid(r, c) and not grid_data.is_death(r, c):
            grid_data._occupied_np[r, c] = True
            # Cannot easily return Triangle objects anymore
    # Return empty set as Triangle objects are not the primary state
    return set()


def occupy_coords(grid_data: GridData, coords: set[tuple[int, int]]):
    """Helper to occupy specific coordinates."""
    for r, c in coords:
        if grid_data.valid(r, c) and not grid_data.is_death(r, c):
            grid_data._occupied_np[r, c] = True


# --- New Reward Calculation Tests (v3) ---


def test_calculate_reward_v3_placement_only(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: only placement, game not over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = set()
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_single_line_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + line clear, game not over."""
    placed_count = len(simple_shape.triangles)
    # Simulate a cleared line of 9 unique coordinates
    unique_coords_cleared: set[tuple[int, int]] = {(0, i) for i in range(9)}
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_multi_line_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + multi-line clear (overlapping coords), game not over."""
    placed_count = len(simple_shape.triangles)
    # Simulate two lines sharing coordinate (0,0)
    line1_coords = {(0, i) for i in range(9)}
    line2_coords = {(i, 0) for i in range(5)}
    unique_coords_cleared = line1_coords.union(line2_coords)  # Union handles uniqueness
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_game_over(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement, no line clear, game IS over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = set()
    is_game_over = True
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + default_env_config.PENALTY_GAME_OVER
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_game_over_with_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + line clear, game IS over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = {(0, i) for i in range(9)}
    is_game_over = True
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.PENALTY_GAME_OVER
    )
    assert reward == pytest.approx(expected_reward)


# --- Test execute_placement with new reward ---


def test_execute_placement_simple_no_refill_v3(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape without clearing lines, verify reward and NO immediate refill."""
    gs = game_state_with_fixed_shapes  # Uses 3 slots, initially filled
    config = gs.env_config
    shape_idx = 0
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]
    shape_to_place = gs.shapes[shape_idx]
    assert shape_to_place is not None
    placed_count = len(shape_to_place.triangles)

    r, c = 2, 2
    assert GridLogic.can_place(gs.grid_data, shape_to_place, r, c)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + survival)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE + config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)
    # Score is still tracked separately
    assert gs.game_score == placed_count

    # Verify grid state using NumPy arrays
    for dr, dc, _ in shape_to_place.triangles:
        tri_r, tri_c = r + dr, c + dc
        assert gs.grid_data._occupied_np[tri_r, tri_c]
        # Cannot easily check color ID without map here, trust placement logic

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1
    assert gs.shapes[2] is original_shape_in_slot_2

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == 0
    assert not gs.is_over()


def test_execute_placement_clear_line_no_refill_v3(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape that clears a line, verify reward and NO immediate refill."""
    gs = game_state_with_fixed_shapes
    config = gs.env_config
    shape_idx = 0
    shape_single_down = gs.shapes[shape_idx]
    assert (
        shape_single_down is not None
        and len(shape_single_down.triangles) == 1
        and not shape_single_down.triangles[0][2]
    )
    placed_count = len(shape_single_down.triangles)
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]

    # Pre-occupy line using coordinates
    # Line indices [3..11] correspond to r=0, c=3 to c=11
    line_coords_to_occupy = {(0, i) for i in range(3, 12) if i != 4}
    occupy_coords(gs.grid_data, line_coords_to_occupy)
    cleared_line_coords = {(0, i) for i in range(3, 12)}  # Coords (0,3) to (0,11)

    r, c = 0, 4  # Placement position
    assert GridLogic.can_place(gs.grid_data, shape_single_down, r, c)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + line clear + survival)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE
        + len(cleared_line_coords) * config.REWARD_PER_CLEARED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)
    # Score is still tracked separately
    assert gs.game_score == placed_count + len(cleared_line_coords) * 2

    # Verify line is cleared using NumPy array
    for row, col in cleared_line_coords:
        assert not gs.grid_data._occupied_np[row, col]

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1
    assert gs.shapes[2] is original_shape_in_slot_2

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == len(cleared_line_coords)
    assert not gs.is_over()


def test_execute_placement_batch_refill_v3(game_state_with_fixed_shapes: GameState):
    """Test that placing the last shape triggers a refill and correct reward."""
    gs = game_state_with_fixed_shapes
    config = gs.env_config
    mock_rng = random.Random(123)

    # Place first shape
    shape_1_coords = (0, 4)
    assert gs.shapes[0] is not None
    placed_count_1 = len(gs.shapes[0].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[0], *shape_1_coords)
    reward1 = execute_placement(gs, 0, 0, 4, mock_rng)
    expected_reward1 = (
        placed_count_1 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward1 == pytest.approx(expected_reward1)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is not None
    assert gs.shapes[2] is not None

    # Place second shape
    shape_2_coords = (0, 3)
    assert gs.shapes[1] is not None
    placed_count_2 = len(gs.shapes[1].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[1], *shape_2_coords)
    reward2 = execute_placement(gs, 1, 0, 3, mock_rng)
    expected_reward2 = (
        placed_count_2 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward2 == pytest.approx(expected_reward2)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is None
    assert gs.shapes[2] is not None

    # Place third shape (triggers refill)
    shape_3_coords = (2, 2)
    assert gs.shapes[2] is not None
    placed_count_3 = len(gs.shapes[2].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[2], *shape_3_coords)
    reward3 = execute_placement(gs, 2, 2, 2, mock_rng)
    expected_reward3 = (
        placed_count_3 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )  # Game not over yet
    assert reward3 == pytest.approx(expected_reward3)
    sleep(0.01)  # Allow time for refill to happen (though it should be synchronous)

    # --- Verify REFILL happened ---
    assert all(s is not None for s in gs.shapes), "Not all slots were refilled"
    assert gs.shapes[0] != Shape([(0, 0, False)], (255, 0, 0))
    assert gs.shapes[1] != Shape([(0, 0, True)], (0, 255, 0))
    assert gs.shapes[2] != Shape([(0, 0, False), (0, 1, True)], (0, 0, 255))

    assert gs.pieces_placed_this_episode == 3
    assert not gs.is_over()


# Add mocker fixture to the test signature
def test_execute_placement_game_over_v3(game_state: GameState, mocker: MockerFixture):
    """Test reward when placement leads to game over, mocking line clears."""
    config = game_state.env_config
    # Fill grid almost completely using NumPy arrays
    playable_mask = ~game_state.grid_data._death_np
    game_state.grid_data._occupied_np[playable_mask] = True

    # Make one spot empty
    empty_r, empty_c = 0, 4
    if not game_state.grid_data.is_death(empty_r, empty_c):  # Ensure it's playable
        game_state.grid_data._occupied_np[empty_r, empty_c] = False

    # Provide a shape that fits the empty spot
    shape_to_place = Shape([(0, 0, False)], (255, 0, 0))  # Single down triangle
    placed_count = len(shape_to_place.triangles)

    # --- Modify setup to prevent refill ---
    unplaceable_shape = Shape([(0, 0, False), (1, 0, False), (2, 0, False)], (1, 1, 1))
    game_state.shapes = [None] * config.NUM_SHAPE_SLOTS
    game_state.shapes[0] = shape_to_place
    if config.NUM_SHAPE_SLOTS > 1:
        game_state.shapes[1] = unplaceable_shape
    # --- End modification ---

    assert GridLogic.can_place(game_state.grid_data, shape_to_place, empty_r, empty_c)
    mock_rng = random.Random(999)

    # --- Mock check_and_clear_lines ---
    # Patch the function within the logic module where execute_placement imports it from
    mock_clear = mocker.patch(
        "muzerotriangle.environment.grid.logic.check_and_clear_lines",
        return_value=(0, set(), set()),  # Simulate no lines cleared
    )
    # --- End Mock ---

    # Execute placement - this should fill the last spot and trigger game over
    reward = execute_placement(game_state, 0, empty_r, empty_c, mock_rng)

    # Verify the mock was called (optional but good practice)
    mock_clear.assert_called_once()

    # Verify game is over
    assert game_state.is_over(), (
        "Game should be over after placing the final piece with no other valid moves"
    )

    # Verify reward (placement + game over penalty)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE + config.PENALTY_GAME_OVER
    )
    # Use a slightly larger tolerance if needed, but approx should work
    assert reward == pytest.approx(expected_reward)


File: tests\environment\__init__.py


File: tests\mcts\conftest.py
# File: tests/mcts/conftest.py
import pytest
import torch

from muzerotriangle.config import EnvConfig, MCTSConfig, ModelConfig
from muzerotriangle.environment import GameState
from muzerotriangle.mcts.core.node import Node
from muzerotriangle.structs import Shape
from muzerotriangle.utils.types import StateType


@pytest.fixture
def real_game_state(mock_env_config: EnvConfig) -> GameState:
    return GameState(config=mock_env_config, initial_seed=123)


class MockMuZeroNetwork:
    def __init__(self, model_config, env_config):
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(env_config.ACTION_DIM)
        self.hidden_dim = model_config.HIDDEN_STATE_DIM
        self.device = torch.device("cpu")
        self.support = torch.linspace(
            model_config.VALUE_MIN,
            model_config.VALUE_MAX,
            model_config.NUM_VALUE_ATOMS,
            device=self.device,
        )
        r_max = float((model_config.REWARD_SUPPORT_SIZE - 1) // 2)
        r_min = -r_max
        self.reward_support = torch.linspace(
            r_min, r_max, model_config.REWARD_SUPPORT_SIZE, device=self.device
        )
        self.default_value = (model_config.VALUE_MAX + model_config.VALUE_MIN) / 2.0
        self.default_reward = 0.0
        self.model = self

    def _state_to_tensors(self, _state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        grid_shape = (
            1,
            self.model_config.GRID_INPUT_CHANNELS,
            self.env_config.ROWS,
            self.env_config.COLS,
        )
        other_shape = (1, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
        return torch.randn(grid_shape), torch.randn(other_shape)

    def _get_mock_logits(self, batch_size, num_classes):
        # Return uniform logits for mock
        return torch.zeros((batch_size, num_classes), device=self.device)

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def _logits_to_scalar(
        self, logits: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        probs = self._logits_to_probs(logits)
        support_expanded = support.expand_as(probs)
        scalar = torch.sum(probs * support_expanded, dim=-1)
        return scalar

    def initial_inference(
        self, observation: StateType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(observation, GameState):
            raise TypeError("Mock initial_inference expects StateType dict")
        batch_size = 1
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        reward_logits = torch.zeros(
            (batch_size, self.model_config.REWARD_SUPPORT_SIZE), device=self.device
        )
        hidden_state = torch.randn((batch_size, self.hidden_dim), device=self.device)
        return policy_logits, value_logits, reward_logits, hidden_state

    def dynamics(self, hidden_state, action):
        if hidden_state is None:
            raise ValueError("Dynamics received None hidden_state")
        batch_size = hidden_state.shape[0]
        next_hidden_state = torch.randn(
            (batch_size, self.hidden_dim), device=self.device
        )
        reward_logits = self._get_mock_logits(
            batch_size, self.model_config.REWARD_SUPPORT_SIZE
        )
        if isinstance(action, int):
            action_val = action
        elif isinstance(action, torch.Tensor) and action.numel() == 1:
            action_val = int(action.item())
        else:
            action_val = 0
        # Add slight deterministic change based on action for testing
        next_hidden_state[:, 0] += (action_val / self.action_dim) * 0.1
        return next_hidden_state, reward_logits

    def predict(self, hidden_state):
        batch_size = hidden_state.shape[0]
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        # --- REVERTED: Return uniform value logits ---
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        # --- END REVERTED ---
        return policy_logits, value_logits

    def forward(
        self, grid_state: torch.Tensor, _other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = grid_state.shape[0]
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        hidden_state = torch.randn((batch_size, self.hidden_dim), device=self.device)
        return policy_logits, value_logits, hidden_state

    def evaluate(self, _state):
        policy_logits = self._get_mock_logits(1, self.action_dim)
        value_logits = self._get_mock_logits(1, self.model_config.NUM_VALUE_ATOMS)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        value = self._logits_to_scalar(value_logits, self.support).item()
        policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
        return policy_map, value

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]


@pytest.fixture
def mock_muzero_network(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> MockMuZeroNetwork:
    return MockMuZeroNetwork(mock_model_config, mock_env_config)


@pytest.fixture
def root_node_real_state(real_game_state: GameState) -> Node:
    return Node(initial_game_state=real_game_state)


@pytest.fixture
def node_with_hidden_state(mock_model_config: ModelConfig) -> Node:
    hidden_state = torch.randn((mock_model_config.HIDDEN_STATE_DIM,))
    return Node(prior=0.2, hidden_state=hidden_state, reward=0.1, action_taken=1)


@pytest.fixture
def expanded_root_node(
    root_node_real_state: Node, mock_muzero_network: MockMuZeroNetwork
) -> Node:
    root = root_node_real_state
    game_state = root.initial_game_state
    assert game_state is not None

    mock_state: StateType = {
        "grid": torch.randn(
            1,
            mock_muzero_network.model_config.GRID_INPUT_CHANNELS,
            mock_muzero_network.env_config.ROWS,
            mock_muzero_network.env_config.COLS,
        )
        .squeeze(0)
        .numpy(),
        "other_features": torch.randn(
            1, mock_muzero_network.model_config.OTHER_NN_INPUT_FEATURES_DIM
        )
        .squeeze(0)
        .numpy(),
    }
    policy_logits, value_logits_init, _, initial_hidden_state = (
        mock_muzero_network.initial_inference(mock_state)
    )

    root.hidden_state = initial_hidden_state.squeeze(0)
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
    valid_actions = game_state.valid_actions()

    for action in valid_actions:
        prior = policy_map.get(action, 0.0)
        hs_batch = (
            root.hidden_state.unsqueeze(0) if root.hidden_state is not None else None
        )
        if hs_batch is None:
            continue
        next_hidden_state_tensor, reward_logits = mock_muzero_network.dynamics(
            hs_batch, action
        )
        reward = mock_muzero_network._logits_to_scalar(
            reward_logits, mock_muzero_network.reward_support
        ).item()
        child = Node(
            prior=prior,
            hidden_state=next_hidden_state_tensor.squeeze(0),
            reward=reward,
            parent=root,
            action_taken=action,
        )
        root.children[action] = child

    root.visit_count = 1
    root.value_sum = mock_muzero_network._logits_to_scalar(
        value_logits_init, mock_muzero_network.support
    ).item()
    return root


@pytest.fixture
def deep_expanded_node_mock_state(
    expanded_root_node: Node,
    mock_muzero_network: MockMuZeroNetwork,
    mock_mcts_config: MCTSConfig,
) -> Node:
    """Creates a tree of depth 2 for testing traversal."""
    root = expanded_root_node
    if not root.children:
        pytest.skip("Cannot create deep tree, root has no children.")

    # --- Make selection deterministic: Boost one child's Q-value ---
    child_to_expand = None
    boost = 10000.0  # Significantly increased boost
    first_child_action = next(iter(root.children.keys()), None)
    if first_child_action is None:
        pytest.skip("Cannot create deep tree, root has no children keys.")

    for action, child in root.children.items():
        # Give a large boost to the first child found
        current_boost = boost if action == first_child_action else 0.0
        child.value_sum += current_boost
        child.visit_count += (
            1  # Add a visit to avoid infinite exploration bonus initially
        )
        if current_boost > 0:
            child_to_expand = child

    if child_to_expand is None or child_to_expand.hidden_state is None:
        pytest.skip("Cannot create deep tree, a valid child to expand is needed.")

    # Update root visit count to reflect added visits
    root.visit_count += len(root.children)
    # --- End deterministic selection setup ---

    # Predict for the child
    policy_logits_child, value_logits_child = mock_muzero_network.predict(
        child_to_expand.hidden_state.unsqueeze(0)
    )
    policy_probs_child = (
        torch.softmax(policy_logits_child, dim=-1).squeeze(0).cpu().numpy()
    )
    policy_map_child = {i: float(p) for i, p in enumerate(policy_probs_child)}

    valid_actions_child = [1, 2]  # Mock valid actions for grandchild level

    for action in valid_actions_child:
        prior = policy_map_child.get(action, 0.0)
        hs_batch = child_to_expand.hidden_state.unsqueeze(0)
        next_hidden_state_tensor, reward_logits = mock_muzero_network.dynamics(
            hs_batch, action
        )
        reward = mock_muzero_network._logits_to_scalar(
            reward_logits, mock_muzero_network.reward_support
        ).item()
        grandchild = Node(
            prior=prior,
            hidden_state=next_hidden_state_tensor.squeeze(0),
            reward=reward,
            parent=child_to_expand,
            action_taken=action,
        )
        child_to_expand.children[action] = grandchild

    # Update stats for the child node as if it was visited during expansion
    # child_to_expand.visit_count = 1 # Already incremented above
    child_to_expand.value_sum += mock_muzero_network._logits_to_scalar(
        value_logits_child, mock_muzero_network.support
    ).item()  # Add predicted value

    # Update root stats to reflect the visit down this path (simplified backprop)
    # root.visit_count += 1 # Already incremented above
    root.value_sum += (
        child_to_expand.reward
        + mock_mcts_config.discount * child_to_expand.value_estimate  # Use estimate now
    )

    return root


@pytest.fixture
def root_node_no_valid_actions(mock_env_config: EnvConfig) -> Node:
    """Creates a GameState where no valid actions should exist."""
    gs = GameState(config=mock_env_config, initial_seed=789)

    # Fill all UP cells
    for r in range(gs.env_config.ROWS):
        for c in range(gs.env_config.COLS):
            is_up = (r + c) % 2 != 0
            if is_up and not gs.grid_data.is_death(r, c):
                gs.grid_data._occupied_np[r, c] = True

    # Provide only UP shapes
    up_shape_1 = Shape([(0, 0, True)], (0, 255, 0))
    up_shape_2_adj = Shape([(0, 1, True)], (0, 0, 255))  # Example different UP shape

    gs.shapes = [None] * gs.env_config.NUM_SHAPE_SLOTS
    gs.shapes[0] = up_shape_1
    if gs.env_config.NUM_SHAPE_SLOTS > 1:
        gs.shapes[1] = up_shape_2_adj
    # Fill remaining slots with copies or other UP shapes
    for i in range(2, gs.env_config.NUM_SHAPE_SLOTS):
        gs.shapes[i] = up_shape_1.copy()

    assert not gs.valid_actions(), "Fixture setup failed: Valid actions still exist."

    return Node(initial_game_state=gs)


File: tests\mcts\fixtures.py
# File: tests/mcts/fixtures.py
# This file might be deprecated if all fixtures moved to conftest.py
# Assuming it's still used, make the necessary changes:

from typing import Any

import pytest
import torch  # Import torch

# Use absolute imports for consistency
from muzerotriangle.config import EnvConfig, MCTSConfig
from muzerotriangle.mcts.core.node import Node


# --- Mock GameState --- (Keep as is)
class MockGameState:
    def __init__(
        self,
        current_step=0,
        is_terminal=False,
        outcome=0.0,
        valid_actions=None,
        env_config=None,
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        self.env_config = env_config if env_config else EnvConfig()
        action_dim_int = int(self.env_config.ACTION_DIM)  # type: ignore[call-overload]
        self._valid_actions = (
            valid_actions if valid_actions is not None else list(range(action_dim_int))
        )

    def is_over(self):
        return self._is_over

    def get_outcome(self):
        if not self._is_over:
            raise ValueError("Cannot get outcome of non-terminal state.")
        return self._outcome

    def valid_actions(self):
        return self._valid_actions

    def copy(self):
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,
        )

    def step(self, action):
        if action not in self._valid_actions:
            raise ValueError(f"Invalid action {action}")
        self.current_step += 1
        self._is_over = self.current_step >= 5
        self._outcome = 1.0 if self._is_over else 0.0
        return 0.0, self._is_over  # Return reward, done

    def __hash__(self):
        return hash((self.current_step, self._is_over, tuple(self._valid_actions)))

    def __eq__(self, other):
        return (
            isinstance(other, MockGameState)
            and self.current_step == other.current_step
            and self._is_over == other._is_over
            and self._valid_actions == other._valid_actions
        )


# --- Mock Network Evaluator --- (Keep as is)
class MockNetworkEvaluator:
    def __init__(self, default_policy=None, default_value=0.5, action_dim=3):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim
        self.evaluation_history = []
        self.batch_evaluation_history = []

    def _get_policy(self, state):
        if self._default_policy:
            return self._default_policy
        valid_actions = state.valid_actions()
        prob = 1.0 / len(valid_actions) if valid_actions else 0
        return dict.fromkeys(valid_actions, prob) if valid_actions else {}

    def evaluate(self, state):
        self.evaluation_history.append(state)
        policy = self._get_policy(state)
        full_policy = dict.fromkeys(range(self._action_dim), 0.0)
        full_policy.update(policy)
        return full_policy, self._default_value

    def evaluate_batch(self, states):
        self.batch_evaluation_history.append(states)
        return [self.evaluate(s) for s in states]


# --- Pytest Fixtures --- (Adapt Node creation)
@pytest.fixture
def mock_env_config_local() -> EnvConfig:  # Renamed to avoid clash if imported
    return EnvConfig()


@pytest.fixture
def mock_mcts_config_local() -> MCTSConfig:  # Renamed
    return MCTSConfig()


@pytest.fixture
def mock_evaluator_local(
    mock_env_config_local: EnvConfig,
) -> MockNetworkEvaluator:  # Renamed
    action_dim_int = int(mock_env_config_local.ACTION_DIM)  # type: ignore[call-overload]
    return MockNetworkEvaluator(action_dim=action_dim_int)


@pytest.fixture
def root_node_mock_state_local(mock_env_config_local: EnvConfig) -> Node:  # Renamed
    state = MockGameState(env_config=mock_env_config_local)
    # --- Use initial_game_state ---
    return Node(initial_game_state=state)  # type: ignore[arg-type]


@pytest.fixture
def expanded_node_mock_state_local(
    root_node_mock_state_local: Node, mock_evaluator_local: MockNetworkEvaluator
) -> Node:  # Renamed
    root = root_node_mock_state_local
    mock_state: Any = root.initial_game_state  # Root holds GameState
    assert mock_state is not None  # Ensure game state exists
    policy, value = mock_evaluator_local.evaluate(mock_state)
    # --- Add dummy hidden state ---
    root.hidden_state = torch.randn(
        (32,)
    )  # Add dummy state after initial eval if needed

    for action in mock_state.valid_actions():
        prior = policy.get(action, 0.0)
        # --- Create child with hidden_state, reward, prior ---
        MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=mock_state.env_config
        )
        # In reality, hidden_state and reward come from dynamics
        child_hidden_state = torch.randn((32,))
        child_reward = 0.05
        child = Node(
            # Use correct keywords
            prior=prior,
            hidden_state=child_hidden_state,
            reward=child_reward,
            parent=root,
            action_taken=action,
        )
        root.children[action] = child
    root.visit_count = 1
    # --- Use value_sum ---
    root.value_sum = value
    return root


File: tests\mcts\test_expansion.py
# File: tests/mcts/test_expansion.py
from typing import Any  # Import Any

import pytest
import torch

from muzerotriangle.mcts.core.node import Node
from muzerotriangle.mcts.strategy import expansion

# Use fixtures from conftest implicitly or explicitly if needed
# from .conftest import MockMuZeroNetwork, Node, mock_muzero_network, node_with_hidden_state


# Use Any for mock network type hint
def test_expand_node_basic(node_with_hidden_state: Node, mock_muzero_network: Any):
    node = node_with_hidden_state
    hidden_state = node.hidden_state
    assert hidden_state is not None
    policy_logits, _ = mock_muzero_network.predict(hidden_state.unsqueeze(0))
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
    valid_actions = list(range(mock_muzero_network.action_dim))
    assert not node.is_expanded
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    for action in valid_actions:
        assert action in node.children
        child = node.children[action]
        assert child.parent is node
        assert child.action_taken == action
        assert child.prior_probability == pytest.approx(policy_map[action])
        assert child.hidden_state is not None
        assert child.hidden_state.shape == hidden_state.shape
        assert isinstance(child.reward, float)
        assert not child.is_expanded
        assert child.visit_count == 0
        assert child.value_sum == 0.0
        if action > 0:
            assert not torch.equal(child.hidden_state, hidden_state)


def test_expand_node_no_valid_actions(
    node_with_hidden_state: Node, mock_muzero_network: Any
):
    node = node_with_hidden_state
    policy_map = {0: 1.0}
    valid_actions: list[int] = []
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert not node.is_expanded


def test_expand_node_already_expanded(
    node_with_hidden_state: Node, mock_muzero_network: Any
):
    node = node_with_hidden_state
    policy_map = {0: 1.0}
    valid_actions = [0]
    child_state = (
        torch.randn_like(node.hidden_state) if node.hidden_state is not None else None
    )
    node.children[0] = Node(hidden_state=child_state, parent=node, action_taken=0)
    assert node.is_expanded
    original_children = node.children.copy()
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert node.children == original_children


def test_expand_node_missing_hidden_state(mock_muzero_network: Any):
    node = Node(parent=None, action_taken=None)
    policy_map = {0: 1.0}
    valid_actions = [0]
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert not node.is_expanded


File: tests\mcts\test_search.py
# File: tests/mcts/test_search.py
from typing import Any

import pytest

from muzerotriangle.config import MCTSConfig
from muzerotriangle.mcts import run_mcts_simulations
from muzerotriangle.mcts.core.node import Node

# Use fixtures from top-level conftest implicitly


def test_run_mcts_simulations_basic(
    root_node_real_state: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    root = root_node_real_state
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    valid_actions = root.initial_game_state.valid_actions()
    if not valid_actions:
        pytest.skip("Initial state has no valid actions.")
    test_config = mock_mcts_config.model_copy(update={"num_simulations": 5})
    max_depth = run_mcts_simulations(
        root_node=root,
        config=test_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )
    assert root.visit_count >= test_config.num_simulations
    assert root.is_expanded
    assert len(root.children) > 0
    assert max_depth >= 0


def test_run_mcts_simulations_on_terminal_state(
    root_node_real_state: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    root = root_node_real_state
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    root.initial_game_state.game_over = True
    valid_actions = root.initial_game_state.valid_actions()
    max_depth = run_mcts_simulations(
        root_node=root,
        config=mock_mcts_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )
    assert max_depth == 0
    assert root.visit_count == 0
    assert not root.is_expanded


def test_run_mcts_simulations_no_valid_actions(
    root_node_no_valid_actions: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    """Test running MCTS when the root state has no valid actions."""
    root = root_node_no_valid_actions
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")

    valid_actions = root.initial_game_state.valid_actions()
    assert not valid_actions

    # Run simulations - it should perform initial inference but fail expansion
    max_depth = run_mcts_simulations(
        root_node=root,
        config=mock_mcts_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )

    # Check state after running simulations
    # Initial inference happens, value is backpropagated once.
    # Simulation loop runs, but expansion fails each time. Backprop happens each time.
    expected_visits = 1 + mock_mcts_config.num_simulations
    assert root.visit_count == expected_visits, (
        f"Root visit count should be 1 + num_simulations ({expected_visits})"
    )
    # --- ADJUSTED ASSERTION ---
    # The root node's hidden_state and predicted_value are set during initial inference.
    # expand_node is called, but should return early without adding children if valid_actions is empty.
    assert not root.children, "Root should have no children when no valid actions exist"
    # is_expanded checks if self.children is non-empty.
    assert not root.is_expanded, "Root should not be expanded (no children added)"
    # --- END ADJUSTED ASSERTION ---
    assert max_depth >= 0  # Depth reflects initial inference/backprop


File: tests\mcts\test_selection.py
# File: tests/mcts/test_selection.py
# File: tests/mcts/test_selection.py
import math

import pytest
import torch

# Import from top-level conftest implicitly, or specific fixtures if needed
from muzerotriangle.config import MCTSConfig, ModelConfig
from muzerotriangle.mcts.core.node import Node
from muzerotriangle.mcts.strategy import selection


# --- Test PUCT Calculation ---
@pytest.mark.usefixtures("mock_mcts_config")
def test_puct_calculation_basic(
    mock_mcts_config: MCTSConfig, node_with_hidden_state: Node
):
    """Test basic PUCT score calculation."""
    if node_with_hidden_state.hidden_state is None:
        pytest.skip("Node needs hidden state")
    parent = Node(hidden_state=torch.randn_like(node_with_hidden_state.hidden_state))
    parent.visit_count = 25
    child = node_with_hidden_state
    child.parent = parent
    child.visit_count = 5
    child.value_sum = 3.0
    child.prior_probability = 0.2
    score, q_value, exploration = selection.calculate_puct_score(
        parent, child, mock_mcts_config
    )
    assert q_value == pytest.approx(0.6)
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.2 * (math.sqrt(25) / (1 + 5))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


@pytest.mark.usefixtures("mock_mcts_config")
def test_puct_calculation_unvisited_child(
    mock_mcts_config: MCTSConfig, node_with_hidden_state: Node
):
    """Test PUCT score for an unvisited child node."""
    if node_with_hidden_state.hidden_state is None:
        pytest.skip("Node needs hidden state")
    parent = Node(hidden_state=torch.randn_like(node_with_hidden_state.hidden_state))
    parent.visit_count = 10
    child = node_with_hidden_state
    child.parent = parent
    child.visit_count = 0
    child.value_sum = 0.0
    child.prior_probability = 0.5
    score, q_value, exploration = selection.calculate_puct_score(
        parent, child, mock_mcts_config
    )
    assert q_value == 0.0
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.5 * (math.sqrt(10) / (1 + 0))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


# --- Test Child Selection ---
def test_select_child_node_basic(
    expanded_root_node: Node, mock_mcts_config: MCTSConfig
):
    """Test selecting the best child based on PUCT."""
    parent = expanded_root_node
    parent.visit_count = 10  # Set parent visits for calculation

    # Ensure there are at least two children to compare
    if len(parent.children) < 2:
        pytest.skip("Requires at least two children for meaningful selection test.")

    # Assign different stats to make selection deterministic for the test
    child_list = list(parent.children.values())
    child0 = child_list[0]
    child1 = child_list[1]

    # Make child1 clearly better according to PUCT
    child0.visit_count = 5
    child0.value_sum = 0.8 * child0.visit_count  # Q = 0.8
    child0.prior_probability = 0.1

    child1.visit_count = 1
    child1.value_sum = 0.5 * child1.visit_count  # Q = 0.5
    child1.prior_probability = 0.6  # Higher prior, lower visits -> higher exploration

    # Calculate scores manually to verify selection logic
    score0, _, _ = selection.calculate_puct_score(parent, child0, mock_mcts_config)
    score1, _, _ = selection.calculate_puct_score(parent, child1, mock_mcts_config)

    selected_child = selection.select_child_node(parent, mock_mcts_config)

    # Assert that the child with the higher score was selected
    if score1 > score0:
        assert selected_child is child1
    else:
        assert selected_child is child0


def test_select_child_node_no_children(
    root_node_real_state: Node, mock_mcts_config: MCTSConfig
):
    parent = root_node_real_state
    assert not parent.children
    with pytest.raises(selection.SelectionError):
        selection.select_child_node(parent, mock_mcts_config)


# --- Test Dirichlet Noise ---
def test_add_dirichlet_noise(expanded_root_node: Node, mock_mcts_config: MCTSConfig):
    node = expanded_root_node
    if not node.children:
        pytest.skip("Node needs children to test noise.")

    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.dirichlet_alpha = 0.5
    config_copy.dirichlet_epsilon = 0.25
    n_children = len(node.children)
    original_priors = {a: c.prior_probability for a, c in node.children.items()}

    selection.add_dirichlet_noise(node, config_copy)

    new_priors = {a: c.prior_probability for a, c in node.children.items()}
    mixed_sum = sum(new_priors.values())
    assert len(new_priors) == n_children
    priors_changed = False
    for action, new_p in new_priors.items():
        assert 0.0 <= new_p <= 1.0
        if abs(new_p - original_priors[action]) > 1e-9:
            priors_changed = True
    assert priors_changed, "Priors did not change after adding noise"
    assert mixed_sum == pytest.approx(1.0, abs=1e-6)


# --- Test Traversal ---
def test_traverse_to_leaf_unexpanded(
    root_node_real_state: Node, mock_mcts_config: MCTSConfig
):
    leaf, depth = selection.traverse_to_leaf(root_node_real_state, mock_mcts_config)
    assert leaf is root_node_real_state
    assert depth == 0


def test_traverse_to_leaf_expanded(
    expanded_root_node: Node, mock_mcts_config: MCTSConfig
):
    root = expanded_root_node
    if not root.children:
        pytest.skip("Root node fixture did not expand.")
    for child in root.children.values():
        assert not child.is_expanded  # Ensure children are leaves initially
    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)
    assert leaf in root.children.values()
    assert depth == 1


def test_traverse_to_leaf_max_depth(
    expanded_root_node: Node,
    mock_mcts_config: MCTSConfig,
    mock_model_config: ModelConfig,
):
    root = expanded_root_node
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 0
    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    assert leaf is root
    assert depth == 0

    config_copy.max_search_depth = 1
    if not root.children:
        pytest.skip("Root has no children.")
    child0 = next(iter(root.children.values()))
    hidden_dim = mock_model_config.HIDDEN_STATE_DIM
    if child0.hidden_state is None:
        child0.hidden_state = torch.randn((hidden_dim,))
    gc_state = (
        torch.randn_like(child0.hidden_state)
        if child0.hidden_state is not None
        else None
    )
    # Ensure grandchild action is different from child action if possible
    gc_action = 0 if child0.action_taken != 0 else 1
    child0.children[gc_action] = Node(
        hidden_state=gc_state, parent=child0, action_taken=gc_action
    )

    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    assert leaf in root.children.values()  # Should stop at depth 1
    assert depth == 1


# Use the corrected fixture name
def test_traverse_to_leaf_deeper_muzero(
    deep_expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    root = deep_expanded_node_mock_state
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 10  # Allow deep traversal

    # Find the child that was expanded in the fixture
    expanded_child = None
    for child in root.children.values():
        if child.children:
            expanded_child = child
            break
    assert expanded_child is not None, "Fixture error: No expanded child found"
    assert expanded_child.children, "Fixture error: Expanded child has no children"

    # Find an expected leaf (grandchild) - Removed as selection isn't guaranteed
    # expected_leaf = next(iter(expanded_child.children.values()), None)
    # assert expected_leaf is not None, "Fixture error: No grandchild found"

    # Traverse and check
    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    # --- FIXED ASSERTION ---
    assert leaf in expanded_child.children.values(), (
        "Returned leaf is not one of the expected grandchildren"
    )
    # --- END FIXED ASSERTION ---
    assert depth == 2


File: tests\mcts\__init__.py


File: tests\nn\test_model.py
# File: tests/nn/test_model.py
import pytest
import torch

# Use absolute imports for consistency
from muzerotriangle.config import EnvConfig, ModelConfig
from muzerotriangle.nn.model import MuZeroNet  # Import MuZeroNet


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def model(model_config: ModelConfig, env_config: EnvConfig) -> MuZeroNet:
    """Provides an instance of the MuZeroNet model."""
    return MuZeroNet(model_config, env_config)


def test_muzero_model_initialization(
    model: MuZeroNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test if the MuZeroNet model initializes without errors."""
    assert model is not None
    assert model.action_dim == int(env_config.ACTION_DIM)  # type: ignore[call-overload]
    assert model.hidden_dim == model_config.HIDDEN_STATE_DIM
    # Add more checks for internal components if needed
    assert hasattr(model, "representation_encoder")
    assert hasattr(model, "representation_projector")
    assert hasattr(model, "action_encoder")
    assert hasattr(model, "dynamics_core")
    assert hasattr(model, "reward_head")
    assert hasattr(model, "prediction_core")
    assert hasattr(model, "policy_head")
    assert hasattr(model, "value_head")


def test_representation_function(
    model: MuZeroNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test the representation function (h)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)
    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        hidden_state = model.represent(dummy_grid, dummy_other)

    assert hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert hidden_state.dtype == torch.float32


def test_dynamics_function(model: MuZeroNet, model_config: ModelConfig):
    """Test the dynamics function (g)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=device
    )
    # Test with batch of actions
    dummy_actions = torch.randint(0, model.action_dim, (batch_size,), device=device)

    with torch.no_grad():
        next_hidden_state, reward_logits = model.dynamics(
            dummy_hidden_state, dummy_actions
        )

    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)
    assert next_hidden_state.dtype == torch.float32
    assert reward_logits.dtype == torch.float32


def test_dynamics_function_single_action(model: MuZeroNet, model_config: ModelConfig):
    """Test the dynamics function (g) with a single action."""
    batch_size = 1  # Test with batch size 1
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=device
    )
    # Test with single integer action
    dummy_action_int = model.action_dim // 2

    with torch.no_grad():
        next_hidden_state, reward_logits = model.dynamics(
            dummy_hidden_state, dummy_action_int
        )

    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)


def test_prediction_function(model: MuZeroNet, model_config: ModelConfig):
    """Test the prediction function (f)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=device
    )

    with torch.no_grad():
        policy_logits, value_logits = model.predict(dummy_hidden_state)

    assert policy_logits.shape == (batch_size, model.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32


def test_forward_initial_inference(
    model: MuZeroNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test the main forward pass for initial inference (h + f)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)
    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        policy_logits, value_logits, initial_hidden_state = model(
            dummy_grid, dummy_other
        )

    assert policy_logits.shape == (batch_size, model.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert initial_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)


File: tests\nn\test_network.py
# File: tests/nn/test_network.py
from typing import cast  # Import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from muzerotriangle.config import (  # Import TrainConfig
    EnvConfig,
    ModelConfig,
    TrainConfig,
)
from muzerotriangle.environment import GameState
from muzerotriangle.nn import MuZeroNet, NeuralNetwork
from muzerotriangle.utils.types import StateType
from tests.conftest import rng


# --- Fixtures ---
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.COMPILE_MODEL = True
    # Explicitly cast the copied and modified config
    return cast("TrainConfig", cfg)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> NeuralNetwork:
    nn = NeuralNetwork(model_config, env_config, train_config, device)
    nn.model.to(device)
    nn.model.eval()
    return nn


@pytest.fixture
def mock_game_state(env_config: EnvConfig) -> GameState:
    return GameState(config=env_config, initial_seed=123)


@pytest.fixture
def mock_game_state_batch(mock_game_state: GameState) -> list[GameState]:
    return [mock_game_state.copy() for _ in range(3)]


@pytest.fixture
def mock_state_type_nn(model_config: ModelConfig, env_config: EnvConfig) -> StateType:
    grid_shape = (model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS)
    other_shape = (model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape).astype(np.float32),
        "other_features": rng.random(other_shape).astype(np.float32),
    }


# --- Tests ---
def test_nn_initialization_muzero(nn_interface: NeuralNetwork, device: torch.device):
    assert nn_interface is not None
    assert nn_interface.device == device
    model_to_check = getattr(nn_interface.model, "_orig_mod", nn_interface.model)
    assert isinstance(model_to_check, MuZeroNet)
    assert not nn_interface.model.training


@patch("muzerotriangle.nn.network.extract_state_features")
def test_initial_inference(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_state_type_nn: StateType,
    model_config: ModelConfig,
):
    mock_extract.return_value = mock_state_type_nn
    batch_size = 1
    policy_logits, value_logits, reward_logits, hidden_state = (
        nn_interface.initial_inference(mock_state_type_nn)
    )
    assert policy_logits.shape == (batch_size, nn_interface.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)
    assert hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32
    assert reward_logits.dtype == torch.float32
    assert hidden_state.dtype == torch.float32
    assert policy_logits.device == nn_interface.device


def test_recurrent_inference(nn_interface: NeuralNetwork, model_config: ModelConfig):
    batch_size = 4
    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=nn_interface.device
    )
    dummy_actions = torch.randint(
        0, nn_interface.action_dim, (batch_size,), device=nn_interface.device
    )
    policy_logits, value_logits, reward_logits, next_hidden_state = (
        nn_interface.recurrent_inference(dummy_hidden_state, dummy_actions)
    )
    assert policy_logits.shape == (batch_size, nn_interface.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)
    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32
    assert reward_logits.dtype == torch.float32
    assert next_hidden_state.dtype == torch.float32
    assert policy_logits.device == nn_interface.device


@patch("muzerotriangle.nn.network.extract_state_features")
def test_evaluate_single_muzero(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    mock_extract.return_value = mock_state_type_nn
    action_dim_int = int(env_config.ACTION_DIM)
    policy_map, value = nn_interface.evaluate(mock_game_state)
    mock_extract.assert_called_once_with(mock_game_state, nn_interface.model_config)
    assert isinstance(policy_map, dict)
    assert isinstance(value, float)
    assert len(policy_map) == action_dim_int
    assert abs(sum(policy_map.values()) - 1.0) < 1e-5


@patch("muzerotriangle.nn.network.extract_state_features")
def test_evaluate_batch_muzero(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state_batch: list[GameState],
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    mock_states = mock_game_state_batch
    batch_size = len(mock_states)
    mock_extract.side_effect = [
        {
            k: (v.copy() + i * 0.1 if isinstance(v, np.ndarray) else v)
            for k, v in mock_state_type_nn.items()
        }
        for i in range(batch_size)
    ]
    action_dim_int = int(env_config.ACTION_DIM)
    results = nn_interface.evaluate_batch(mock_states)
    assert mock_extract.call_count == batch_size
    assert isinstance(results, list)
    assert len(results) == batch_size
    for policy_map, value in results:
        assert isinstance(policy_map, dict)
        assert isinstance(value, float)
        assert len(policy_map) == action_dim_int
        assert abs(sum(policy_map.values()) - 1.0) < 1e-5


def test_get_set_weights_muzero(nn_interface: NeuralNetwork):
    initial_weights = nn_interface.get_weights()
    assert isinstance(initial_weights, dict)
    modified_weights = {}
    for k, v in initial_weights.items():
        modified_weights[k] = v + 0.1 if v.dtype.is_floating_point else v
    nn_interface.set_weights(modified_weights)
    new_weights = nn_interface.get_weights()
    for key in initial_weights:
        if initial_weights[key].dtype.is_floating_point:
            assert torch.allclose(modified_weights[key], new_weights[key], atol=1e-6), (
                f"Mismatch key {key}"
            )
        else:
            assert torch.equal(initial_weights[key], new_weights[key]), (
                f"Non-float mismatch key {key}"
            )


File: tests\nn\__init__.py


File: tests\rl\test_buffer.py
# File: tests/rl/test_buffer.py
# No changes needed here, ensure atol=1e-5 is still present from previous step.
import random

import numpy as np
import pytest

from muzerotriangle.config import TrainConfig
from muzerotriangle.rl import ExperienceBuffer
from muzerotriangle.utils.sumtree import SumTree  # Import SumTree for type check
from muzerotriangle.utils.types import (
    StateType,
    Trajectory,
    TrajectoryStep,
)

# Import the helper function
from tests.utils.test_sumtree import dump_sumtree_state

# Use default_rng for modern numpy random generation
rng = np.random.default_rng(seed=42)


# --- Fixtures ---
@pytest.fixture
def muzero_train_config() -> TrainConfig:
    # Enable PER for buffer tests
    return TrainConfig(
        BUFFER_CAPACITY=1000,
        MIN_BUFFER_SIZE_TO_TRAIN=50,
        BATCH_SIZE=4,
        MUZERO_UNROLL_STEPS=3,
        N_STEP_RETURNS=5,
        USE_PER=True,  # Enable PER
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        REWARD_LOSS_WEIGHT=1.0,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        MAX_TRAINING_STEPS=200,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
    )


@pytest.fixture
def muzero_buffer(muzero_train_config: TrainConfig) -> ExperienceBuffer:
    # The ExperienceBuffer init should handle PER setup correctly based on config
    return ExperienceBuffer(muzero_train_config)


@pytest.fixture
def mock_state_type() -> StateType:
    return {
        "grid": rng.random((1, 3, 3)).astype(np.float32),
        "other_features": rng.random((10,)).astype(np.float32),
    }


@pytest.fixture
def mock_trajectory_step(mock_state_type: StateType) -> TrajectoryStep:
    action_dim = 9
    return {
        "observation": mock_state_type,
        "action": random.randint(0, action_dim - 1) if action_dim > 0 else 0,
        "reward": random.uniform(-1, 1),
        "policy_target": (
            dict.fromkeys(range(action_dim), 1.0 / action_dim) if action_dim > 0 else {}
        ),
        "value_target": random.uniform(-1, 1),
        "n_step_reward_target": random.uniform(-1, 1),  # Add n-step target
        "hidden_state": None,
    }


@pytest.fixture
def short_trajectory(mock_trajectory_step: TrajectoryStep) -> Trajectory:
    return [mock_trajectory_step.copy() for _ in range(3)]  # Length 3


@pytest.fixture
def long_trajectory(mock_trajectory_step: TrajectoryStep) -> Trajectory:
    traj = []
    for i in range(10):
        step = mock_trajectory_step.copy()
        step["reward"] += i * 0.1
        step["value_target"] += i * 0.1
        step["action"] = i % 9
        traj.append(step)
    return traj


# --- PER Buffer Tests ---
def test_muzero_buffer_init_per(muzero_buffer: ExperienceBuffer):
    assert muzero_buffer.use_per
    assert muzero_buffer.sum_tree is not None
    assert isinstance(muzero_buffer.sum_tree, SumTree)  # Check specific type
    assert len(muzero_buffer) == 0
    assert muzero_buffer.sum_tree.n_entries == 0


def test_muzero_buffer_add_per(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert muzero_buffer.sum_tree is not None
    initial_total_priority = muzero_buffer.sum_tree.total()  # Use total() method
    initial_n_entries = muzero_buffer.sum_tree.n_entries
    assert initial_n_entries == 0  # Should start at 0

    print("\n[test_muzero_buffer_add_per] Before add:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_add_per_before")
    muzero_buffer.add(long_trajectory)
    print("[test_muzero_buffer_add_per] After add:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_add_per_after")

    assert len(muzero_buffer.buffer) == 1
    assert len(muzero_buffer) == len(long_trajectory)
    # Check n_entries incremented
    assert muzero_buffer.sum_tree.n_entries == initial_n_entries + 1, (
        f"n_entries did not increment. Before: {initial_n_entries}, After: {muzero_buffer.sum_tree.n_entries}"
    )
    # New entry added with max priority
    assert muzero_buffer.sum_tree.total() > initial_total_priority  # Use total() method
    # Check if the stored item is a tuple
    assert isinstance(muzero_buffer.buffer[0], tuple)
    assert isinstance(muzero_buffer.buffer[0][0], int)  # buffer_idx
    assert isinstance(muzero_buffer.buffer[0][1], list)  # trajectory


def test_muzero_buffer_sample_per(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert muzero_buffer.sum_tree is not None
    # Ensure buffer is ready
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for i in range(num_needed):
        traj_copy = [step.copy() for step in long_trajectory]
        for step in traj_copy:
            step["reward"] += i * 0.01  # Add slight variation
        muzero_buffer.add(traj_copy)

    print("\n[test_muzero_buffer_sample_per] Before is_ready check:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_sample_per")
    # This assertion should now pass because SumTree.n_entries is correct
    assert muzero_buffer.is_ready(), (
        f"Buffer not ready. Steps: {len(muzero_buffer)}, Min: {muzero_buffer.min_size_to_train}, "
        f"SumTree Entries: {muzero_buffer.sum_tree.n_entries}, BatchSize: {muzero_buffer.config.BATCH_SIZE}"
    )
    assert muzero_buffer.sum_tree.n_entries > 0, (
        "SumTree has no entries after adding trajectories"
    )
    assert (
        muzero_buffer.sum_tree.total() > 1e-9  # Use total() method
    ), "SumTree total priority is near zero"

    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(
        batch_size, current_train_step=1
    )  # Pass step for beta

    assert sample is not None, "PER sampling returned None unexpectedly"
    assert isinstance(sample, dict), (
        f"Expected dict (PER sample), got {type(sample)}"
    )  # Check it's a dict

    # Check keys instead of isinstance for TypedDict
    assert "sequences" in sample
    assert "indices" in sample
    assert "weights" in sample
    assert isinstance(sample["sequences"], list)
    assert len(sample["sequences"]) == batch_size
    assert isinstance(sample["indices"], np.ndarray)
    assert len(sample["indices"]) == batch_size
    assert isinstance(sample["weights"], np.ndarray)
    assert len(sample["weights"]) == batch_size
    assert sample["weights"].dtype == np.float32
    assert np.all(sample["weights"] > 0)
    assert np.all(sample["weights"] <= 1.0 + 1e-6)  # Check normalization

    for sequence in sample["sequences"]:
        assert isinstance(sequence, list)
        assert len(sequence) == muzero_buffer.sequence_length


def test_muzero_buffer_update_priorities(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert muzero_buffer.sum_tree is not None
    # Ensure buffer is ready
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        muzero_buffer.add(long_trajectory.copy())

    print("\n[test_muzero_buffer_update_priorities] Before is_ready check:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_update_priorities")
    # This assertion should now pass
    assert muzero_buffer.is_ready(), (
        f"Buffer not ready. Steps: {len(muzero_buffer)}, Min: {muzero_buffer.min_size_to_train}, "
        f"SumTree Entries: {muzero_buffer.sum_tree.n_entries}, BatchSize: {muzero_buffer.config.BATCH_SIZE}"
    )

    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(batch_size, current_train_step=1)

    assert sample is not None, "PER sampling returned None unexpectedly"
    assert isinstance(sample, dict), f"Expected dict (PER sample), got {type(sample)}"
    assert "indices" in sample

    tree_indices = sample["indices"]
    # Get initial priorities correctly
    initial_priorities_list = []
    for idx_val in tree_indices:
        idx = int(idx_val)
        if 0 <= idx < len(muzero_buffer.sum_tree.tree):
            initial_priorities_list.append(float(muzero_buffer.sum_tree.tree[idx]))
        else:
            initial_priorities_list.append(0.0)
    initial_priorities = np.array(initial_priorities_list)

    # Use rng.random for modern numpy
    td_errors = rng.random(len(tree_indices)) * 0.5  # Match length of indices
    muzero_buffer.update_priorities(tree_indices, td_errors)

    # Check if priorities changed
    new_priorities_list = []
    for idx_val in tree_indices:
        idx = int(idx_val)
        if 0 <= idx < len(muzero_buffer.sum_tree.tree):
            new_priorities_list.append(float(muzero_buffer.sum_tree.tree[idx]))
        else:
            new_priorities_list.append(0.0)
    new_priorities = np.array(new_priorities_list)

    assert not np.allclose(initial_priorities, new_priorities)
    # Check if priorities reflect errors (higher error -> higher priority)
    expected_priorities = (
        np.abs(td_errors) + muzero_buffer.per_epsilon
    ) ** muzero_buffer.per_alpha
    assert new_priorities.shape == expected_priorities.shape
    # --- Reverted Tolerance ---
    assert np.allclose(new_priorities, expected_priorities, atol=1e-5), (
        f"Priorities mismatch.\nNew: {new_priorities}\nExpected: {expected_priorities}"
    )
    # --- End Reverted Tolerance ---


# --- Uniform Fallback Tests (if PER fails or disabled) ---
def test_muzero_buffer_sample_uniform_fallback(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    """Test fallback to uniform sampling if PER is enabled but SumTree is empty."""
    assert muzero_buffer.use_per
    assert muzero_buffer.sum_tree is not None
    # Add just enough steps to be ready
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        muzero_buffer.add(long_trajectory.copy())

    # Check if buffer is ready based on steps (should be)
    assert len(muzero_buffer) >= muzero_buffer.min_size_to_train
    print("\n[test_muzero_buffer_sample_uniform_fallback] Before n_entries check:")
    dump_sumtree_state(
        muzero_buffer.sum_tree, "test_muzero_buffer_sample_uniform_fallback"
    )
    # Check if buffer is ready based on SumTree entries (should be)
    assert muzero_buffer.sum_tree.n_entries >= muzero_buffer.config.BATCH_SIZE
    assert muzero_buffer.is_ready()  # This should now pass

    # Force total priority to 0 AND n_entries to 0 to trigger fallback
    muzero_buffer.sum_tree.tree.fill(0.0)  # Zero out entire tree
    muzero_buffer.sum_tree.n_entries = 0  # Crucial for the check in _sample_per
    assert muzero_buffer.sum_tree.total() < 1e-9  # Use total() method
    assert muzero_buffer.sum_tree.n_entries == 0

    # is_ready should now fail because n_entries < batch_size
    assert not muzero_buffer.is_ready()

    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(batch_size, current_train_step=1)

    # Expect None because is_ready() failed
    assert sample is None, "Expected None when PER sampling fails, but got a sample."


def test_muzero_buffer_sample_uniform_when_per_disabled(long_trajectory: Trajectory):
    """Test uniform sampling when PER is explicitly disabled."""
    config_no_per = TrainConfig(
        BUFFER_CAPACITY=1000,
        MIN_BUFFER_SIZE_TO_TRAIN=50,
        BATCH_SIZE=4,
        MUZERO_UNROLL_STEPS=3,
        N_STEP_RETURNS=5,
        USE_PER=False,  # PER Disabled
    )
    buffer_no_per = ExperienceBuffer(config_no_per)
    assert not buffer_no_per.use_per
    assert buffer_no_per.sum_tree is None

    num_needed = (buffer_no_per.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        buffer_no_per.add(long_trajectory.copy())

    batch_size = config_no_per.BATCH_SIZE
    sample = buffer_no_per.sample(batch_size)  # No step needed for uniform

    assert sample is not None
    assert isinstance(sample, list)  # Uniform returns list
    assert not isinstance(sample, dict)
    assert len(sample) == batch_size


File: tests\rl\test_trainer.py
# File: tests/rl/test_trainer.py
# File: tests/rl/test_trainer.py
import random
from typing import cast  # Import cast

import numpy as np
import pytest
import torch

from muzerotriangle.config import (  # Import TrainConfig
    EnvConfig,
    ModelConfig,
    TrainConfig,
)
from muzerotriangle.nn import NeuralNetwork
from muzerotriangle.rl import Trainer
from muzerotriangle.utils.types import (
    SampledBatch,
    SampledBatchPER,  # Import PER type
    SampledSequence,
    StateType,
    TrajectoryStep,
)
from tests.conftest import rng


# --- Fixtures ---
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    """Use MuZero defaults: PER enabled, Unroll steps."""
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = True  # Enable PER
    cfg.MUZERO_UNROLL_STEPS = 3
    cfg.N_STEP_RETURNS = 5
    cfg.POLICY_LOSS_WEIGHT = 1.0
    cfg.VALUE_LOSS_WEIGHT = 0.25
    cfg.REWARD_LOSS_WEIGHT = 1.0
    return cast("TrainConfig", cfg)


@pytest.fixture
def nn_interface(
    model_config: ModelConfig, env_config: EnvConfig, train_config: TrainConfig
) -> NeuralNetwork:
    device = torch.device("cpu")
    nn = NeuralNetwork(model_config, env_config, train_config, device)
    nn.model.to(device)
    nn.model.eval()
    return nn


@pytest.fixture
def trainer(
    nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig
) -> Trainer:
    return Trainer(nn_interface, train_config, env_config)


@pytest.fixture
def mock_state_type() -> StateType:
    return {
        "grid": rng.random((1, 3, 3)).astype(np.float32),
        "other_features": rng.random((10,)).astype(np.float32),
    }


@pytest.fixture
def mock_trajectory_step(
    mock_state_type: StateType, env_config: EnvConfig
) -> TrajectoryStep:
    action_dim = int(env_config.ACTION_DIM)
    return {
        "observation": mock_state_type,
        "action": random.randint(0, action_dim - 1) if action_dim > 0 else 0,
        "reward": random.uniform(-1, 1),
        "policy_target": (
            dict.fromkeys(range(action_dim), 1.0 / action_dim) if action_dim > 0 else {}
        ),
        "value_target": random.uniform(-1, 1),
        "n_step_reward_target": random.uniform(-1, 1),  # Add n-step target
        "hidden_state": None,
    }


@pytest.fixture
def mock_sequence(
    mock_trajectory_step: TrajectoryStep, train_config: TrainConfig
) -> SampledSequence:
    seq_len = train_config.MUZERO_UNROLL_STEPS + 1
    return [mock_trajectory_step.copy() for _ in range(seq_len)]


@pytest.fixture
def mock_batch(
    mock_sequence: SampledSequence, train_config: TrainConfig
) -> SampledBatch:
    return [mock_sequence[:] for _ in range(train_config.BATCH_SIZE)]


@pytest.fixture
def mock_per_batch(
    mock_batch: SampledBatch, train_config: TrainConfig
) -> SampledBatchPER:
    batch_size = train_config.BATCH_SIZE
    return SampledBatchPER(
        sequences=mock_batch,
        indices=np.arange(batch_size, dtype=np.int32),
        weights=np.ones(batch_size, dtype=np.float32) * 0.5,  # Example weights
    )


# --- Tests ---
def test_trainer_initialization_muzero(trainer: Trainer):
    assert trainer.nn is not None
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert hasattr(trainer, "scheduler")
    assert trainer.unroll_steps == trainer.train_config.MUZERO_UNROLL_STEPS


def test_prepare_batch_muzero(trainer: Trainer, mock_batch: SampledBatch):
    batch_size = trainer.train_config.BATCH_SIZE
    seq_len = trainer.unroll_steps + 1
    action_dim = int(trainer.env_config.ACTION_DIM)
    prepared_data = trainer._prepare_batch(mock_batch)
    assert isinstance(prepared_data, dict)
    expected_keys = {
        "grids",
        "others",
        "actions",
        "n_step_rewards",  # Check for n-step key
        "policy_targets",
        "value_targets",
    }
    assert set(prepared_data.keys()) == expected_keys
    assert prepared_data["grids"].shape == (
        batch_size,
        seq_len,
        trainer.model_config.GRID_INPUT_CHANNELS,
        trainer.env_config.ROWS,
        trainer.env_config.COLS,
    )
    assert prepared_data["others"].shape == (
        batch_size,
        seq_len,
        trainer.model_config.OTHER_NN_INPUT_FEATURES_DIM,
    )
    assert prepared_data["actions"].shape == (batch_size, seq_len)
    assert prepared_data["n_step_rewards"].shape == (batch_size, seq_len)  # Check shape
    assert prepared_data["policy_targets"].shape == (batch_size, seq_len, action_dim)
    assert prepared_data["value_targets"].shape == (batch_size, seq_len)
    for key in expected_keys:
        assert prepared_data[key].device == trainer.device


def test_train_step_muzero_uniform(trainer: Trainer, mock_batch: SampledBatch):
    """Test train step with uniform batch."""
    trainer.model.to(trainer.device)
    initial_params = [p.clone() for p in trainer.model.parameters()]
    train_result = trainer.train_step(mock_batch)
    assert train_result is not None
    loss_info, td_errors = train_result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert "reward_loss" in loss_info
    assert loss_info["total_loss"] > 0

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer.train_config.BATCH_SIZE,)

    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change."


def test_train_step_muzero_per(trainer: Trainer, mock_per_batch: SampledBatchPER):
    """Test train step with PER batch."""
    trainer.model.to(trainer.device)
    initial_params = [p.clone() for p in trainer.model.parameters()]
    train_result = trainer.train_step(mock_per_batch)
    assert train_result is not None
    loss_info, td_errors = train_result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert loss_info["total_loss"] > 0  # Loss should still be positive

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer.train_config.BATCH_SIZE,)

    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change (PER)."


def test_train_step_empty_batch_muzero(trainer: Trainer):
    assert trainer.train_step([]) is None


File: tests\rl\__init__.py


File: tests\stats\test_collector.py
# File: tests/stats/test_collector.py
import logging
from collections import deque

import cloudpickle
import pytest
import ray

from muzerotriangle.stats import StatsCollectorActor
from muzerotriangle.utils.types import StepInfo  # Import StepInfo

# --- Fixtures ---


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    if not ray.is_initialized():
        ray.init(logging_level=logging.WARNING, num_cpus=1)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def stats_actor():
    """Provides a fresh StatsCollectorActor instance for each test."""
    actor = StatsCollectorActor.remote(max_history=5)
    # Ensure actor is initialized before returning
    ray.get(actor.clear.remote())  # Use a simple remote call to wait for init
    yield actor
    # Clean up the actor after the test
    ray.kill(actor, no_restart=True)


# --- Helper to create StepInfo ---
def create_step_info(step: int) -> StepInfo:
    """Creates a basic StepInfo dict for testing."""
    return {"global_step": step}


# --- Tests ---


def test_actor_initialization(stats_actor):
    """Test if the actor initializes correctly."""
    assert ray.get(stats_actor.get_data.remote()) == {}
    # Also check initial worker states
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_single_metric(stats_actor):
    """Test logging a single metric."""
    metric_name = "test_metric"
    value = 10.5
    step = 1
    # --- CHANGED: Pass StepInfo ---
    step_info = create_step_info(step)
    ray.get(stats_actor.log.remote(metric_name, value, step_info))
    # --- END CHANGED ---
    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert data[metric_name][0] == (step_info, value)
    # --- END CHANGED ---


def test_log_batch_metrics(stats_actor):
    """Test logging a batch of metrics."""
    # --- CHANGED: Pass StepInfo ---
    step_info_1 = create_step_info(1)
    step_info_2 = create_step_info(2)
    ray.get(stats_actor.log.remote("metric_a", 1.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_b", 2.5, step_info_1))
    ray.get(stats_actor.log.remote("metric_a", 1.1, step_info_2))
    # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert "metric_a" in data
    assert "metric_b" in data
    assert len(data["metric_a"]) == 2, (
        f"Expected 2 entries for metric_a, found {len(data['metric_a'])}"
    )
    assert len(data["metric_b"]) == 1
    # --- CHANGED: Check StepInfo in results ---
    assert data["metric_a"][0] == (step_info_1, 1.0)
    assert data["metric_a"][1] == (step_info_2, 1.1)
    assert data["metric_b"][0] == (step_info_1, 2.5)
    # --- END CHANGED ---


def test_max_history(stats_actor):
    """Test if the max_history constraint is enforced."""
    metric_name = "history_test"
    max_hist = 5  # Matches fixture
    for i in range(max_hist + 3):
        # --- CHANGED: Pass StepInfo ---
        step_info = create_step_info(i)
        ray.get(stats_actor.log.remote(metric_name, float(i), step_info))
        # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == max_hist
    # Check if the first elements were dropped
    # --- CHANGED: Check StepInfo in result ---
    expected_first_step_info = create_step_info(3)
    assert data[metric_name][0] == (expected_first_step_info, 3.0)
    expected_last_step_info = create_step_info(max_hist + 2)
    assert data[metric_name][-1] == (expected_last_step_info, float(max_hist + 2))
    # --- END CHANGED ---


def test_get_metric_data(stats_actor):
    """Test retrieving data for a specific metric."""
    # --- CHANGED: Pass StepInfo ---
    step_info_1 = create_step_info(1)
    step_info_2 = create_step_info(2)
    ray.get(stats_actor.log.remote("metric_1", 10.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_2", 20.0, step_info_1))
    ray.get(stats_actor.log.remote("metric_1", 11.0, step_info_2))
    # --- END CHANGED ---

    metric1_data = ray.get(stats_actor.get_metric_data.remote("metric_1"))
    metric2_data = ray.get(stats_actor.get_metric_data.remote("metric_2"))
    metric3_data = ray.get(stats_actor.get_metric_data.remote("metric_3"))

    assert isinstance(metric1_data, deque)
    assert len(metric1_data) == 2
    # --- CHANGED: Check StepInfo in results ---
    assert list(metric1_data) == [(step_info_1, 10.0), (step_info_2, 11.0)]
    # --- END CHANGED ---

    assert isinstance(metric2_data, deque)
    assert len(metric2_data) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert list(metric2_data) == [(step_info_1, 20.0)]
    # --- END CHANGED ---

    assert metric3_data is None


def test_clear_data(stats_actor):
    """Test clearing the collected data."""
    # --- CHANGED: Pass StepInfo ---
    step_info = create_step_info(1)
    ray.get(stats_actor.log.remote("metric_1", 10.0, step_info))
    # --- END CHANGED ---
    assert len(ray.get(stats_actor.get_data.remote())) == 1
    ray.get(stats_actor.clear.remote())
    assert ray.get(stats_actor.get_data.remote()) == {}
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}


def test_log_non_finite(stats_actor):
    """Test that non-finite values are not logged."""
    metric_name = "non_finite_test"
    # --- CHANGED: Pass StepInfo ---
    ray.get(stats_actor.log.remote(metric_name, float("inf"), create_step_info(1)))
    ray.get(stats_actor.log.remote(metric_name, float("-inf"), create_step_info(2)))
    ray.get(stats_actor.log.remote(metric_name, float("nan"), create_step_info(3)))
    step_info_4 = create_step_info(4)
    ray.get(stats_actor.log.remote(metric_name, 10.0, step_info_4))
    # --- END CHANGED ---

    data = ray.get(stats_actor.get_data.remote())
    assert metric_name in data
    assert len(data[metric_name]) == 1
    # --- CHANGED: Check StepInfo in result ---
    assert data[metric_name][0] == (step_info_4, 10.0)
    # --- END CHANGED ---


def test_get_set_state(stats_actor):
    """Test saving and restoring the actor's state."""
    # --- CHANGED: Pass StepInfo ---
    step_info_10 = create_step_info(10)
    step_info_11 = create_step_info(11)
    ray.get(stats_actor.log.remote("m1", 1.0, step_info_10))
    ray.get(stats_actor.log.remote("m2", 2.0, step_info_10))
    ray.get(stats_actor.log.remote("m1", 1.5, step_info_11))
    # --- END CHANGED ---

    state = ray.get(stats_actor.get_state.remote())

    # Verify state structure (basic check)
    assert isinstance(state, dict)
    assert "max_history" in state
    assert "_metrics_data_list" in state
    assert isinstance(state["_metrics_data_list"], dict)
    assert "m1" in state["_metrics_data_list"]
    assert isinstance(state["_metrics_data_list"]["m1"], list)
    # --- CHANGED: Check StepInfo in results ---
    assert state["_metrics_data_list"]["m1"] == [
        (step_info_10, 1.0),
        (step_info_11, 1.5),
    ], f"Actual m1 list: {state['_metrics_data_list']['m1']}"
    assert state["_metrics_data_list"]["m2"] == [(step_info_10, 2.0)], (
        f"Actual m2 list: {state['_metrics_data_list']['m2']}"
    )
    # --- END CHANGED ---

    # Use cloudpickle to simulate saving/loading
    pickled_state = cloudpickle.dumps(state)
    unpickled_state = cloudpickle.loads(pickled_state)

    # Create a new actor and restore state
    new_actor = StatsCollectorActor.remote(
        max_history=10
    )  # Different initial max_history
    ray.get(new_actor.set_state.remote(unpickled_state))

    # Verify restored state
    restored_data = ray.get(new_actor.get_data.remote())
    original_data = ray.get(
        stats_actor.get_data.remote()
    )  # Get original data again for comparison

    assert len(restored_data) == len(original_data)
    assert "m1" in restored_data
    assert "m2" in restored_data
    # Compare the deques after converting to lists
    assert list(restored_data["m1"]) == list(original_data["m1"])
    assert list(restored_data["m2"]) == list(original_data["m2"])

    # Check max_history was restored
    # Check behavior by adding more data
    # --- CHANGED: Pass StepInfo ---
    step_info_12 = create_step_info(12)
    step_info_13 = create_step_info(13)
    step_info_14 = create_step_info(14)
    step_info_15 = create_step_info(15)
    ray.get(new_actor.log.remote("m1", 2.0, step_info_12))
    ray.get(new_actor.log.remote("m1", 2.5, step_info_13))
    ray.get(new_actor.log.remote("m1", 3.0, step_info_14))
    ray.get(new_actor.log.remote("m1", 3.5, step_info_15))
    # --- END CHANGED ---

    restored_m1 = ray.get(new_actor.get_metric_data.remote("m1"))
    assert len(restored_m1) == 5  # Max history from loaded state
    # --- CHANGED: Check StepInfo in result ---
    assert restored_m1[0] == (step_info_11, 1.5)  # Check first element is correct
    # --- END CHANGED ---

    # Check that worker states were cleared on restore
    assert ray.get(new_actor.get_latest_worker_states.remote()) == {}

    ray.kill(new_actor, no_restart=True)


# --- Tests for Game State Handling ---
# Mock GameState class for testing state updates
class MockGameStateForStats:
    def __init__(self, step: int, score: float):
        self.current_step = step
        self.game_score = score
        # Add dummy attributes expected by the check in update_worker_game_state
        self.grid_data = True
        self.shapes = True


def test_update_and_get_worker_state(stats_actor):
    """Test updating and retrieving worker game states."""
    worker_id = 1
    state1 = MockGameStateForStats(step=10, score=5.0)
    state2 = MockGameStateForStats(step=11, score=6.0)

    # Initial state should be empty
    assert ray.get(stats_actor.get_latest_worker_states.remote()) == {}

    # Update state for worker 1
    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state1))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 10
    assert latest_states[worker_id].game_score == 5.0

    # Update state again for worker 1
    ray.get(stats_actor.update_worker_game_state.remote(worker_id, state2))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id].game_score == 6.0

    # Update state for worker 2
    worker_id_2 = 2
    state3 = MockGameStateForStats(step=5, score=2.0)
    ray.get(stats_actor.update_worker_game_state.remote(worker_id_2, state3))
    latest_states = ray.get(stats_actor.get_latest_worker_states.remote())
    assert worker_id in latest_states
    assert worker_id_2 in latest_states
    assert latest_states[worker_id].current_step == 11
    assert latest_states[worker_id_2].current_step == 5


File: tests\stats\__init__.py


File: tests\utils\test_sumtree.py
# File: tests/utils/test_sumtree.py
import logging

import pytest

from muzerotriangle.utils.sumtree import SumTree, sumtree_logger  # Import logger

# Configure logging for tests to see debug messages from SumTree
# logging.basicConfig(level=logging.DEBUG) # Set root logger level if needed
# sumtree_logger.setLevel(logging.DEBUG) # Set specific logger level


def dump_sumtree_state(tree: SumTree, test_name: str):
    """Helper function to print SumTree state for debugging."""
    print(f"\n--- SumTree State Dump ({test_name}) ---")
    print(f"  User Capacity: {tree.capacity}")
    print(f"  Internal Capacity: {tree._internal_capacity}")  # Log internal capacity
    print(f"  n_entries: {tree.n_entries}")
    print(f"  data_pointer: {tree.data_pointer}")
    print(f"  _max_priority: {tree._max_priority:.4f}")
    print(f"  total_priority (root): {tree.total():.4f}")
    # Only print the relevant part of the tree array
    tree_size = 2 * tree._internal_capacity - 1  # Use internal capacity
    print(
        f"  Tree array (size {len(tree.tree)}, showing up to {tree_size}): {tree.tree[:tree_size]}"
    )
    # Print only the populated part of the data array (up to n_entries)
    print(
        f"  Data array (size {len(tree.data)}, showing up to {tree.n_entries}): {tree.data[: tree.n_entries]}"
    )
    print("--- End Dump ---")


@pytest.fixture
def sum_tree_cap5() -> SumTree:
    """Provides a SumTree instance with user capacity 5."""
    # Internal capacity will be 8
    return SumTree(capacity=5)


def test_sumtree_init():
    tree_user_cap = 10
    tree = SumTree(capacity=tree_user_cap)
    internal_cap = 16  # Next power of 2 >= 10
    assert tree.capacity == tree_user_cap
    assert tree._internal_capacity == internal_cap
    assert len(tree.tree) == 2 * internal_cap - 1  # 31
    assert len(tree.data) == internal_cap  # 16
    assert tree.data_pointer == 0
    assert tree.n_entries == 0
    assert tree.total() == 0.0
    assert tree.max_priority == 1.0


def test_sumtree_add_single(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    # First add goes to data_pointer=0, tree_idx = 0 + 8 - 1 = 7
    tree_idx = sum_tree_cap5.add(0.5, "data1")
    assert tree_idx == 7
    assert sum_tree_cap5.n_entries == 1
    assert sum_tree_cap5.data_pointer == 1  # Wraps around user capacity 5
    assert sum_tree_cap5.total() == pytest.approx(0.5)
    assert sum_tree_cap5.max_priority == pytest.approx(0.5)
    assert sum_tree_cap5.tree[tree_idx] == 0.5
    assert sum_tree_cap5.data[0] == "data1"


def test_sumtree_add_multiple(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    data = ["d0", "d1", "d2", "d3", "d4"]
    expected_leaf_indices = [7, 8, 9, 10, 11]  # Based on internal capacity 8
    expected_max_priority = 0.0
    for i, (p, d) in enumerate(zip(priorities, data, strict=False)):
        tree_idx = sum_tree_cap5.add(p, d)
        assert tree_idx == expected_leaf_indices[i]
        expected_max_priority = max(expected_max_priority, p)
        assert sum_tree_cap5.n_entries == i + 1
        assert sum_tree_cap5.data_pointer == (i + 1) % 5  # Wrap around user capacity 5
        assert sum_tree_cap5.total() == pytest.approx(sum(priorities[: i + 1]))
        assert sum_tree_cap5.max_priority == pytest.approx(expected_max_priority)

    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 0
    # Data array has internal capacity, check only the first 5 slots
    assert sum_tree_cap5.data[:5] == data


def test_sumtree_add_overflow(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    data = ["d0", "d1", "d2", "d3", "d4"]
    expected_leaf_indices = [7, 8, 9, 10, 11]
    for i, (p, d) in enumerate(zip(priorities, data, strict=False)):
        tree_idx = sum_tree_cap5.add(p, d)
        assert tree_idx == expected_leaf_indices[i]

    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.total() == pytest.approx(sum(priorities))
    assert sum_tree_cap5.max_priority == pytest.approx(0.8)

    # Add one more, overwriting the first element (data_idx 0, tree_idx 7)
    tree_idx_5 = sum_tree_cap5.add(1.0, "d5")
    assert tree_idx_5 == 7  # Should overwrite leaf 7
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 1  # Wraps around user capacity 5
    assert sum_tree_cap5.data[0] == "d5"
    assert sum_tree_cap5.data[1] == "d1"  # Unchanged
    assert sum_tree_cap5.total() == pytest.approx(sum(priorities[1:]) + 1.0)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[7] == 1.0  # Check leaf node updated

    # Add another, overwriting the second element (data_idx 1, tree_idx 8)
    tree_idx_6 = sum_tree_cap5.add(0.05, "d6")
    assert tree_idx_6 == 8  # Should overwrite leaf 8
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 2  # Wraps around user capacity 5
    assert sum_tree_cap5.data[0] == "d5"
    assert sum_tree_cap5.data[1] == "d6"
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.05 + 0.2 + 0.8 + 0.4)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[8] == 0.05  # Check leaf node updated


def test_sumtree_update(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    tree_idx_0 = sum_tree_cap5.add(0.5, "data0")  # Leaf 7
    tree_idx_1 = sum_tree_cap5.add(0.3, "data1")  # Leaf 8
    assert tree_idx_0 == 7
    assert tree_idx_1 == 8
    assert sum_tree_cap5.total() == pytest.approx(0.8)
    assert sum_tree_cap5.max_priority == pytest.approx(0.5)

    sum_tree_cap5.update(tree_idx_0, 1.0)  # Update leaf 7
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.3)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[tree_idx_0] == pytest.approx(1.0)

    sum_tree_cap5.update(tree_idx_1, 0.1)  # Update leaf 8
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.1)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[tree_idx_1] == pytest.approx(0.1)


def test_sumtree_retrieve(sum_tree_cap5: SumTree):
    """Test the _retrieve method directly with internal capacity 8."""
    # --- Enable Debug Logging ---
    original_level = sumtree_logger.level
    sumtree_logger.setLevel(logging.DEBUG)
    # ---

    # Internal capacity = 8. Tree size = 15. Leaves = indices 7..14
    # User capacity = 5. Data indices = 0..4. n_entries = 5.
    # Mapping: Data 0 -> Leaf 7, Data 1 -> Leaf 8, ..., Data 4 -> Leaf 11

    data_map = {}
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]  # Sum = 2.0
    expected_leaf_indices = [7, 8, 9, 10, 11]
    for i, p in enumerate(priorities):
        data_id = f"d{i}"
        tree_idx = sum_tree_cap5.add(p, data_id)
        assert tree_idx == expected_leaf_indices[i]
        data_map[tree_idx] = data_id

    print(f"\n[test_retrieve] Tree state: {sum_tree_cap5.tree}")
    print(f"[test_retrieve] Data state: {sum_tree_cap5.data}")
    print(f"[test_retrieve] Data map (tree_idx -> data_id): {data_map}")

    # Expected Tree (cap 8, 5 entries):
    # Leaves: [0.1, 0.5, 0.2, 0.8, 0.4, 0.0, 0.0, 0.0] (Indices 7-14)
    # Lvl 2:  [0.6, 1.0, 0.4, 0.0] (Indices 3-6)
    # Lvl 1:  [1.6, 0.4] (Indices 1-2)
    # Root:   [2.0] (Index 0)
    # Full Tree: [2.0, 1.6, 0.4, 0.6, 1.0, 0.4, 0.0, 0.1, 0.5, 0.2, 0.8, 0.4, 0.0, 0.0, 0.0]

    # Test retrieval based on sample values
    # Cumulative sums: [0.1, 0.6, 0.8, 1.6, 2.0]
    test_cases = {
        0.05: 7,  # Should fall in the first bucket (index 7, prio 0.1)
        0.1: 8,  # Should fall in the second bucket (index 8, prio 0.5)
        0.15: 8,
        0.6: 9,  # Should fall in the third bucket (index 9, prio 0.2)
        0.7: 9,
        0.8: 10,  # Should fall in the fourth bucket (index 10, prio 0.8)
        1.5: 10,
        1.6: 11,  # Should fall in the fifth bucket (index 11, prio 0.4)
        1.99: 11,
    }

    try:
        for sample_value, expected_tree_idx in test_cases.items():
            print(f"[test_retrieve] Testing sample {sample_value:.4f}")
            retrieved_tree_idx = sum_tree_cap5._retrieve(0, sample_value)
            assert retrieved_tree_idx == expected_tree_idx, (
                f"Sample {sample_value:.4f} -> Expected {expected_tree_idx}, Got {retrieved_tree_idx}"
            )
    finally:
        # --- Restore Log Level ---
        sumtree_logger.setLevel(original_level)
        # ---


def test_sumtree_retrieve_with_zeros(sum_tree_cap5: SumTree):
    """Test _retrieve with zero priority nodes."""
    # --- Enable Debug Logging ---
    original_level = sumtree_logger.level
    sumtree_logger.setLevel(logging.DEBUG)
    # ---

    # Internal capacity = 8. Leaves = indices 7..14
    # User capacity = 5. Data indices = 0..4. n_entries = 5.
    # Mapping: Data 0 -> Leaf 7, ..., Data 4 -> Leaf 11

    data_map = {}
    priorities = [0.0, 0.4, 0.6, 0.0, 0.0]  # Sum = 1.0
    data = ["z0", "iA", "iB", "z1", "z2"]
    expected_leaf_indices = [7, 8, 9, 10, 11]
    for i, p in enumerate(priorities):
        data_id = data[i]
        tree_idx = sum_tree_cap5.add(p, data_id)
        assert tree_idx == expected_leaf_indices[i]
        data_map[tree_idx] = data_id

    print(f"\n[test_retrieve_zeros] Tree state: {sum_tree_cap5.tree}")
    print(f"[test_retrieve_zeros] Data state: {sum_tree_cap5.data}")
    print(f"[test_retrieve_zeros] Data map (tree_idx -> data_id): {data_map}")

    # Expected Tree (cap 8, 5 entries with zeros):
    # Leaves: [0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0] (Indices 7-14)
    # Lvl 2:  [0.4, 0.6, 0.0, 0.0] (Indices 3-6)
    # Lvl 1:  [1.0, 0.0] (Indices 1-2)
    # Root:   [1.0] (Index 0)
    # Full Tree: [1.0, 1.0, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Test retrieval based on sample values
    # Cumulative sums: [0.0, 0.4, 1.0, 1.0, 1.0]
    test_cases = {
        0.0: 8,  # Should skip zero-priority leaf 7 and land in leaf 8
        0.1: 8,  # Should land in leaf 8
        0.3: 8,
        0.399: 8,
        0.4: 9,  # Should land in leaf 9
        0.5: 9,
        0.99: 9,
    }

    try:
        for sample_value, expected_tree_idx in test_cases.items():
            print(f"[test_retrieve_zeros] Testing sample {sample_value:.4f}")
            retrieved_tree_idx = sum_tree_cap5._retrieve(0, sample_value)
            assert retrieved_tree_idx == expected_tree_idx, (
                f"Sample {sample_value:.4f} -> Expected {expected_tree_idx}, Got {retrieved_tree_idx}"
            )
    finally:
        # --- Restore Log Level ---
        sumtree_logger.setLevel(original_level)
        # ---


def test_sumtree_get_leaf_edge_cases(sum_tree_cap5: SumTree):
    """Test edge cases for get_leaf."""
    # --- Enable Debug Logging ---
    original_level = sumtree_logger.level
    sumtree_logger.setLevel(logging.DEBUG)
    # ---

    try:
        # Empty tree
        with pytest.raises(
            ValueError,
            match="Cannot sample from SumTree with zero or negative total priority",
        ):
            sum_tree_cap5.get_leaf(0.5)

        # Single item
        tree_idx_0 = sum_tree_cap5.add(1.0, "only_item")  # Should be leaf index 7
        assert tree_idx_0 == 7
        assert sum_tree_cap5.n_entries == 1
        assert sum_tree_cap5.total() == pytest.approx(1.0)
        print(f"\n[test_edge_cases] Single item tree: {sum_tree_cap5.tree}")

        # Test sampling at 0.0
        print("[test_edge_cases] Testing get_leaf(0.0)")
        idx0, p0, d0 = sum_tree_cap5.get_leaf(0.0)
        assert d0 == "only_item"
        assert p0 == pytest.approx(1.0)
        assert idx0 == tree_idx_0  # Should be 7

        # Test sampling close to total priority
        print("[test_edge_cases] Testing get_leaf(1.0 - eps)")
        idx1, p1, d1 = sum_tree_cap5.get_leaf(1.0 - 1e-9)
        assert d1 == "only_item"
        assert p1 == pytest.approx(1.0)
        assert idx1 == tree_idx_0

        # Test sampling exactly at total priority (should be clipped)
        print("[test_edge_cases] Testing get_leaf(1.0)")
        idx_exact, p_exact, d_exact = sum_tree_cap5.get_leaf(1.0)
        assert d_exact == "only_item"
        assert idx_exact == tree_idx_0

        # Test sampling above total priority (should be clipped)
        print("[test_edge_cases] Testing get_leaf(1.1)")
        idx_above, p_above, d_above = sum_tree_cap5.get_leaf(1.1)
        assert d_above == "only_item"
        assert idx_above == tree_idx_0

        # Test sampling below zero (should be clipped)
        print("[test_edge_cases] Testing get_leaf(-0.1)")
        idx_below, p_below, d_below = sum_tree_cap5.get_leaf(-0.1)
        assert d_below == "only_item"
        assert idx_below == tree_idx_0

        # Zero priority item
        sum_tree_cap5.reset()
        tree_idx_z0 = sum_tree_cap5.add(0.0, "z0")  # data_idx 0, tree_idx 7
        tree_idx_iA = sum_tree_cap5.add(0.4, "itemA")  # data_idx 1, tree_idx 8
        tree_idx_iB = sum_tree_cap5.add(0.6, "itemB")  # data_idx 2, tree_idx 9
        sum_tree_cap5.add(0.0, "z1")  # data_idx 3, tree_idx 10
        sum_tree_cap5.add(0.0, "z2")  # data_idx 4, tree_idx 11
        assert sum_tree_cap5.n_entries == 5
        assert sum_tree_cap5.total() == pytest.approx(1.0)
        print(f"[test_edge_cases] Zero priority tree: {sum_tree_cap5.tree}")
        print(f"[test_edge_cases] Data: {sum_tree_cap5.data}")

        # Sampling value < 0.4 should yield 'itemA' (index 8)
        print("[test_edge_cases] Testing get_leaf(0.3) with zero priority item")
        idx, p, d = sum_tree_cap5.get_leaf(0.3)
        assert d == "itemA"
        assert p == pytest.approx(0.4)
        assert idx == tree_idx_iA  # Should be 8

        # Sampling value >= 0.4 and < 1.0 should yield 'itemB' (index 9)
        print("[test_edge_cases] Testing get_leaf(0.5) with zero priority item")
        idx, p, d = sum_tree_cap5.get_leaf(0.5)
        assert d == "itemB"
        assert p == pytest.approx(0.6)
        assert idx == tree_idx_iB  # Should be 9

        # Test sampling exactly at boundary (0.4)
        print("[test_edge_cases] Testing get_leaf(0.4) boundary")
        idx_b, p_b, d_b = sum_tree_cap5.get_leaf(0.4)
        assert d_b == "itemB"  # Should land in the second non-zero bucket
        assert idx_b == tree_idx_iB  # Should be 9

        # Test sampling at 0.0
        print("[test_edge_cases] Testing get_leaf(0.0) boundary")
        idx_0, p_0, d_0 = sum_tree_cap5.get_leaf(0.0)
        assert d_0 == "itemA"  # Should pick the first non-zero element
        assert idx_0 == tree_idx_iA  # Should be 8

        # Test updating a zero-priority item
        sum_tree_cap5.update(tree_idx_z0, 0.1)  # Update "z0" priority (index 7)
        assert sum_tree_cap5.total() == pytest.approx(1.1)
        print(f"[test_edge_cases] After update tree: {sum_tree_cap5.tree}")
        print("[test_edge_cases] Testing get_leaf(0.05) after update")
        idx_up, p_up, d_up = sum_tree_cap5.get_leaf(0.05)
        assert d_up == "z0"
        assert idx_up == tree_idx_z0  # Should be 7
        assert p_up == pytest.approx(0.1)

    finally:
        # --- Restore Log Level ---
        sumtree_logger.setLevel(original_level)
        # ---


File: tests\utils\__init__.py


