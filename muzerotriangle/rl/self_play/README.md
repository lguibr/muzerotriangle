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
        -   Asynchronously logs per-step statistics (`MCTS/Step_Visits`, `MCTS/Step_Depth`, `RL/Step_Reward`, **`RL/Current_Score`**) and reports its current `GameState` to the `StatsCollectorActor`.

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