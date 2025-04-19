# File: muzerotriangle/mcts/core/README.md
# MCTS Core Submodule (`muzerotriangle.mcts.core`)

## Purpose and Architecture

This submodule defines the fundamental building blocks and interfaces for the MuZero Monte Carlo Tree Search implementation.

-   **[`Node`](node.py):** The `Node` class is the cornerstone, representing a single state within the search tree. It stores the associated *hidden state* (`torch.Tensor`), the predicted *reward* to reach this state, parent/child relationships, the action that led to it, and crucial MCTS statistics (visit count, value sum, prior probability). The root node additionally holds the initial `GameState`. It provides properties like `value_estimate` (Q-value) and `is_expanded`.
-   **[`search`](search.py):** The `search.py` module contains the high-level `run_mcts_simulations` function. This function orchestrates the core MCTS loop for a given root node: performing initial inference, then repeatedly selecting leaves, expanding them using the network's dynamics and prediction functions, and backpropagating the results. It uses helper functions from the [`muzerotriangle.mcts.strategy`](../strategy/README.md) submodule. **Includes extensive DEBUG level logging** for tracing the simulation process. Handles potential gradient issues by detaching tensors before converting to NumPy.
-   **[`types`](types.py):** The `types.py` module defines essential type hints and protocols for the MCTS module, such as `ActionPolicyMapping`. The `ActionPolicyValueEvaluator` protocol is less relevant now as the `NeuralNetwork` interface is used directly.

## Exposed Interfaces

-   **Classes:**
    -   `Node`: The tree node class (MuZero version).
    -   `MCTSExecutionError`: Custom exception for MCTS failures.
-   **Functions:**
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network: NeuralNetwork, valid_actions_from_state: List[ActionType]) -> int`: Orchestrates the MCTS process. Returns the maximum depth reached during simulations.
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

**Note:** Please keep this README updated when modifying the `Node` structure, the `run_mcts_simulations` logic, or the interfaces used. Accurate documentation is crucial for maintainability. **Set the logger level for `muzerotriangle.mcts.core` to `DEBUG` to enable detailed tracing.**