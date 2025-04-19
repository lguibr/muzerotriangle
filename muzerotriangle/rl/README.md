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