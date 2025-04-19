# File: muzerotriangle/mcts/README.md
# Monte Carlo Tree Search Module (`muzerotriangle.mcts`)

## Purpose and Architecture

This module implements the Monte Carlo Tree Search algorithm, adapted for the MuZero framework. MCTS is used during self-play to explore the game tree, generate improved policies, and estimate state values, providing training targets for the neural network.

-   **Core Components ([`core/README.md`](core/README.md)):**
    -   `Node`: Represents a state in the search tree. In MuZero, nodes store the *hidden state* (`s_k`) predicted by the dynamics function, the predicted *reward* (`r_k`) to reach that state, and MCTS statistics (visit counts, value sum, prior probability). The root node holds the initial `GameState` and its corresponding initial hidden state (`s_0`) after the first inference.
    -   `search`: Contains the main `run_mcts_simulations` function orchestrating the selection, expansion, and backpropagation phases. It uses the `NeuralNetwork` interface for initial inference (`h+f`) and recurrent inference (`g+f`). Includes extensive DEBUG level logging to trace simulation steps, node selection, expansion, and backpropagation. Handles potential gradient issues by detaching tensors before converting to NumPy. **Note:** When starting with an unexpanded root node (typical for MuZero at each game step), the root's visit count will be `num_simulations + 1` due to an initial backpropagation of the root's predicted value before the main simulation loop.
    -   `config`: Defines the `MCTSConfig` class holding hyperparameters like the number of simulations, PUCT coefficient, temperature settings, Dirichlet noise parameters, and the discount factor (`gamma`).
    -   `types`: Defines necessary type hints and protocols, notably `ActionPolicyValueEvaluator` (though the `NeuralNetwork` interface is now used directly) and `ActionPolicyMapping`.
-   **Strategy Components ([`strategy/README.md`](strategy/README.md)):**
    -   `selection`: Implements the tree traversal logic (PUCT calculation, Dirichlet noise addition, leaf selection). Includes DEBUG level logging for PUCT scores and selected children. Uses value initialization (FPU) for unvisited nodes.
    -   `expansion`: Handles expanding leaf nodes using policy predictions from the network's prediction function (`f`). Includes DEBUG level logging for expansion details.
    -   `backpropagation`: Implements the process of updating node statistics back up the tree, incorporating predicted rewards and the discount factor. Includes DEBUG level logging for value propagation steps.
    -   `policy`: Provides functions to select the final action based on visit counts (`select_action_based_on_visits`) and to generate the policy target vector for training (`get_policy_target`).

## Exposed Interfaces

-   **Core:**
    -   `Node`: The tree node class (MuZero version).
    -   `MCTSConfig`: Configuration class (defined in [`muzerotriangle.config`](../config/README.md)).
    -   `run_mcts_simulations(root_node: Node, config: MCTSConfig, network: NeuralNetwork, valid_actions_from_state: List[ActionType]) -> int`: The main function to run MCTS. Returns the maximum depth reached during simulations.
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
    -   Used for Dirichlet noise and policy calculations. **Requires careful handling (e.g., `.detach()`) when converting from `torch.Tensor`.**
-   **`torch`**:
    -   Used for hidden states within `Node`.
-   **Standard Libraries:** `typing`, `math`, `logging`, `numpy`, `time`.

---

**Note:** Please keep this README updated when changing the MCTS algorithm phases (selection, expansion, backpropagation), the node structure, configuration options, or the interaction with the environment or neural network. Accurate documentation is crucial for maintainability. **Set the logger level for `muzerotriangle.mcts` to `DEBUG` to enable detailed tracing.**