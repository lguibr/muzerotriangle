# File: muzerotriangle/training/README.md
# Training Module (`muzerotriangle.training`)

## Purpose and Architecture

This module encapsulates the logic for setting up, running, and managing the MuZero reinforcement learning training pipeline. It aims to provide a cleaner separation of concerns compared to embedding all logic within the run scripts or a single orchestrator class.

-   **[`setup.py`](setup.py):** Contains `setup_training_components` which initializes Ray, detects resources, adjusts worker counts, loads configurations, and creates the core components bundle (`TrainingComponents`).
-   **[`components.py`](components.py):** Defines the `TrainingComponents` dataclass, a simple container to bundle all the necessary initialized objects (NN, Buffer, Trainer, DataManager, StatsCollector, Configs) required by the `TrainingLoop`.
-   **[`loop.py`](loop.py):** Defines the `TrainingLoop` class. This class contains the core asynchronous logic of the training loop itself:
    -   Managing the pool of `SelfPlayWorker` actors via `WorkerManager`.
    -   Submitting self-play tasks and collecting results (which include full game `Trajectory` data).
    -   Adding completed `Trajectory` objects to the `ExperienceBuffer`.
    -   Sampling batches of fixed-length sequences (`SampledBatch`) from the buffer.
    -   Triggering training steps on the `Trainer` with sampled sequences.
    -   Updating worker network weights periodically, passing the current `global_step`, and logging weight update events.
    -   Updating progress bars.
    -   Pushing state updates to the visualizer queue (if provided).
    -   Handling stop requests.
-   **[`worker_manager.py`](worker_manager.py):** Defines the `WorkerManager` class, responsible for creating, managing, submitting tasks to, and collecting results (`SelfPlayResult` containing `Trajectory`) from the `SelfPlayWorker` actors. Passes `global_step` to workers during weight updates.
-   **[`loop_helpers.py`](loop_helpers.py):** Contains helper functions used by `TrainingLoop` for tasks like logging rates, updating the visual queue, and logging results. Constructs `StepInfo` for asynchronous logging. Includes logic to log the weight update event. **Buffer size logging now reflects total steps.**
-   **[`runners.py`](runners.py):** Re-exports the main entry point functions (`run_training_headless_mode`, `run_training_visual_mode`).
-   **[`headless_runner.py`](headless_runner.py) / [`visual_runner.py`](visual_runner.py):** Contain the top-level logic for running training. They handle setup, load initial state (including MuZero buffer format), run the `TrainingLoop`, and manage cleanup.
-   **[`logging_utils.py`](logging_utils.py):** Contains helper functions for logging setup and MLflow integration.

This structure separates the high-level setup/teardown from the core iterative logic, accommodating the MuZero data flow.

## Exposed Interfaces

-   **Classes:** `TrainingLoop`, `TrainingComponents`, `WorkerManager`, `LoopHelpers`.
-   **Functions (from `runners.py`):** `run_training_headless_mode`, `run_training_visual_mode`.
-   **Functions (from `setup.py`):** `setup_training_components`.
-   **Functions (from `logging_utils.py`):** `setup_file_logging`, `log_configs_to_mlflow`, etc.

## Dependencies

-   **`muzerotriangle.config`**: All configuration classes.
-   **`muzerotriangle.nn`**: `NeuralNetwork` (MuZero version).
-   **`muzerotriangle.rl`**: `ExperienceBuffer`, `Trainer`, `SelfPlayWorker`, `SelfPlayResult` (all MuZero versions).
-   **`muzerotriangle.data`**: `DataManager`, `LoadedTrainingState` (handling MuZero buffer).
-   **`muzerotriangle.stats`**: `StatsCollectorActor`.
-   **`muzerotriangle.environment`**: `GameState`.
-   **`muzerotriangle.utils`**: Helper functions and types (including `Trajectory`, `SampledSequence`, `StepInfo`).
-   **`muzerotriangle.visualization`**: `ProgressBar`, `DashboardRenderer`.
-   **`ray`**: For parallelism.
-   **`mlflow`**: For experiment tracking.
-   **`torch`**: For neural network operations.
-   **Standard Libraries:** `logging`, `time`, `threading`, `queue`, `os`, `json`, `collections.deque`, `dataclasses`.

---

**Note:** Please keep this README updated when changing the structure of the training pipeline, the responsibilities of the components, or the way components interact, especially regarding trajectory handling and MuZero training logic.