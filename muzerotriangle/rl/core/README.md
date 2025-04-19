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