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