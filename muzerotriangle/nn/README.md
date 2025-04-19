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
        -   `initial_inference(observation: StateType)`: Runs `h` and `f` to get `s_0`, `p_0`, `v_0`. Returns logits and hidden state.
        -   `recurrent_inference(hidden_state: Tensor, action: ActionType | Tensor)`: Runs `g` and `f` to get `s_k`, `r_k`, `p_k`, `v_k`. Returns logits and next hidden state.
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