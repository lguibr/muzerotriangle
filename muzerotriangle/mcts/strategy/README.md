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