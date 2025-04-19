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