# File: tests/mcts/conftest.py
import pytest
import torch

from muzerotriangle.config import EnvConfig, ModelConfig
from muzerotriangle.environment import GameState
from muzerotriangle.mcts.core.node import Node
from muzerotriangle.structs import Shape
from muzerotriangle.utils.types import StateType


@pytest.fixture
def real_game_state(mock_env_config: EnvConfig) -> GameState:
    return GameState(config=mock_env_config, initial_seed=123)


class MockMuZeroNetwork:
    def __init__(self, model_config, env_config):
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(env_config.ACTION_DIM)
        self.hidden_dim = model_config.HIDDEN_STATE_DIM
        self.device = torch.device("cpu")
        self.support = torch.linspace(
            model_config.VALUE_MIN,
            model_config.VALUE_MAX,
            model_config.NUM_VALUE_ATOMS,
            device=self.device,
        )
        r_max = float((model_config.REWARD_SUPPORT_SIZE - 1) // 2)
        r_min = -r_max
        self.reward_support = torch.linspace(
            r_min, r_max, model_config.REWARD_SUPPORT_SIZE, device=self.device
        )
        self.default_value = (model_config.VALUE_MAX + model_config.VALUE_MIN) / 2.0
        self.default_reward = 0.0
        self.model = self

    def _state_to_tensors(self, _state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        grid_shape = (
            1,
            self.model_config.GRID_INPUT_CHANNELS,
            self.env_config.ROWS,
            self.env_config.COLS,
        )
        other_shape = (1, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
        return torch.randn(grid_shape), torch.randn(other_shape)

    def _get_mock_logits(self, batch_size, num_classes):
        # Return uniform logits for mock
        return torch.zeros((batch_size, num_classes), device=self.device)

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def _logits_to_scalar(
        self, logits: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        probs = self._logits_to_probs(logits)
        support_expanded = support.expand_as(probs)
        scalar = torch.sum(probs * support_expanded, dim=-1)
        return scalar

    def initial_inference(
        self, observation: StateType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(observation, GameState):
            raise TypeError("Mock initial_inference expects StateType dict")
        batch_size = 1
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        reward_logits = torch.zeros(
            (batch_size, self.model_config.REWARD_SUPPORT_SIZE), device=self.device
        )
        hidden_state = torch.randn((batch_size, self.hidden_dim), device=self.device)
        return policy_logits, value_logits, reward_logits, hidden_state

    def dynamics(self, hidden_state, action):
        if hidden_state is None:
            raise ValueError("Dynamics received None hidden_state")
        batch_size = hidden_state.shape[0]
        next_hidden_state = torch.randn(
            (batch_size, self.hidden_dim), device=self.device
        )
        reward_logits = self._get_mock_logits(
            batch_size, self.model_config.REWARD_SUPPORT_SIZE
        )
        if isinstance(action, int):
            action_val = action
        elif isinstance(action, torch.Tensor) and action.numel() == 1:
            action_val = int(action.item())
        else:
            action_val = 0
        # Add slight deterministic change based on action for testing
        next_hidden_state[:, 0] += (action_val / self.action_dim) * 0.1
        return next_hidden_state, reward_logits

    def predict(self, hidden_state):
        batch_size = hidden_state.shape[0]
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        return policy_logits, value_logits

    def forward(
        self, grid_state: torch.Tensor, _other_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = grid_state.shape[0]
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        hidden_state = torch.randn((batch_size, self.hidden_dim), device=self.device)
        return policy_logits, value_logits, hidden_state

    def evaluate(self, _state):
        policy_logits = self._get_mock_logits(1, self.action_dim)
        value_logits = self._get_mock_logits(1, self.model_config.NUM_VALUE_ATOMS)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        value = self._logits_to_scalar(value_logits, self.support).item()
        policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
        return policy_map, value

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]


@pytest.fixture
def mock_muzero_network(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> MockMuZeroNetwork:
    return MockMuZeroNetwork(mock_model_config, mock_env_config)


@pytest.fixture
def root_node_real_state(real_game_state: GameState) -> Node:
    # Create root node with initial game state, but no hidden state yet
    return Node(initial_game_state=real_game_state)


@pytest.fixture
def node_with_hidden_state(mock_model_config: ModelConfig) -> Node:
    hidden_state = torch.randn((mock_model_config.HIDDEN_STATE_DIM,))
    return Node(prior=0.2, hidden_state=hidden_state, reward=0.1, action_taken=1)


@pytest.fixture
def expanded_root_node(
    root_node_real_state: Node, mock_muzero_network: MockMuZeroNetwork
) -> Node:
    """Expands the root node once using the mock network."""
    root = root_node_real_state
    game_state = root.initial_game_state
    assert game_state is not None

    # Perform initial inference to get hidden state and policy
    mock_state: StateType = {
        "grid": torch.randn(
            1,
            mock_muzero_network.model_config.GRID_INPUT_CHANNELS,
            mock_muzero_network.env_config.ROWS,
            mock_muzero_network.env_config.COLS,
        )
        .squeeze(0)
        .numpy(),
        "other_features": torch.randn(
            1, mock_muzero_network.model_config.OTHER_NN_INPUT_FEATURES_DIM
        )
        .squeeze(0)
        .numpy(),
    }
    policy_logits, value_logits_init, _, initial_hidden_state = (
        mock_muzero_network.initial_inference(mock_state)
    )

    # Assign hidden state and predicted value to root
    root.hidden_state = initial_hidden_state.squeeze(0)
    root.predicted_value = mock_muzero_network._logits_to_scalar(
        value_logits_init, mock_muzero_network.support
    ).item()

    # Expand children
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
    valid_actions = game_state.valid_actions()

    for action in valid_actions:
        prior = policy_map.get(action, 0.0)
        hs_batch = (
            root.hidden_state.unsqueeze(0) if root.hidden_state is not None else None
        )
        if hs_batch is None:
            continue
        next_hidden_state_tensor, reward_logits = mock_muzero_network.dynamics(
            hs_batch, action
        )
        reward = mock_muzero_network._logits_to_scalar(
            reward_logits, mock_muzero_network.reward_support
        ).item()
        child = Node(
            prior=prior,
            hidden_state=next_hidden_state_tensor.squeeze(0),
            reward=reward,
            parent=root,
            action_taken=action,
        )
        root.children[action] = child

    # NOTE: Visit counts and value sums are NOT set here, they start at 0
    # The test calling run_mcts_simulations will handle the initial backprop if needed.
    return root


@pytest.fixture
def deep_expanded_node_mock_state(
    expanded_root_node: Node, mock_muzero_network: MockMuZeroNetwork
) -> Node:
    """
    Creates a tree of depth 2 by expanding one child of the expanded_root_node.
    Visit counts and value sums are NOT manually set here.
    Slightly increases the prior of the expanded child to guide selection.
    """
    root = expanded_root_node
    if not root.children:
        pytest.skip("Cannot create deep tree, root has no children.")

    # Select the first child to expand (arbitrarily) and slightly boost its prior
    child_to_expand = None
    child_to_reduce = None
    boost_amount = 0.01
    found_first = False
    for child in root.children.values():
        if not found_first:
            child_to_expand = child
            child.prior_probability += boost_amount  # Boost the first child
            found_first = True
        elif child_to_reduce is None:
            child_to_reduce = child  # Find another child to reduce prior
            child.prior_probability -= boost_amount
            break  # Found both, exit loop

    # Ensure we found a child to expand
    if child_to_expand is None or child_to_expand.hidden_state is None:
        pytest.skip("Cannot create deep tree, a valid child to expand is needed.")
    # Ensure we found a child to reduce (if more than one child exists)
    if len(root.children) > 1 and child_to_reduce is None:
        pytest.skip("Could not find a second child to reduce prior for balancing.")

    # Predict for the child
    policy_logits_child, _ = mock_muzero_network.predict(
        child_to_expand.hidden_state.unsqueeze(0)
    )
    policy_probs_child = (
        torch.softmax(policy_logits_child, dim=-1).squeeze(0).cpu().numpy()
    )
    policy_map_child = {i: float(p) for i, p in enumerate(policy_probs_child)}

    # Mock valid actions for grandchild level (can be all actions for mock)
    valid_actions_child = list(range(mock_muzero_network.action_dim))

    # Expand the selected child
    for action in valid_actions_child:
        prior = policy_map_child.get(action, 0.0)
        hs_batch = child_to_expand.hidden_state.unsqueeze(0)
        next_hidden_state_tensor, reward_logits = mock_muzero_network.dynamics(
            hs_batch, action
        )
        reward = mock_muzero_network._logits_to_scalar(
            reward_logits, mock_muzero_network.reward_support
        ).item()
        grandchild = Node(
            prior=prior,
            hidden_state=next_hidden_state_tensor.squeeze(0),
            reward=reward,
            parent=child_to_expand,
            action_taken=action,
        )
        child_to_expand.children[action] = grandchild

    # NOTE: Visit counts and value sums remain 0 for child and grandchildren.
    return root


@pytest.fixture
def root_node_no_valid_actions(mock_env_config: EnvConfig) -> Node:
    """Creates a GameState where no valid actions should exist."""
    gs = GameState(config=mock_env_config, initial_seed=789)

    # Fill all UP cells
    for r in range(gs.env_config.ROWS):
        for c in range(gs.env_config.COLS):
            is_up = (r + c) % 2 != 0
            if is_up and not gs.grid_data.is_death(r, c):
                gs.grid_data._occupied_np[r, c] = True

    # Provide only UP shapes
    up_shape_1 = Shape([(0, 0, True)], (0, 255, 0))
    up_shape_2_adj = Shape([(0, 1, True)], (0, 0, 255))  # Example different UP shape

    gs.shapes = [None] * gs.env_config.NUM_SHAPE_SLOTS
    gs.shapes[0] = up_shape_1
    if gs.env_config.NUM_SHAPE_SLOTS > 1:
        gs.shapes[1] = up_shape_2_adj
    # Fill remaining slots with copies or other UP shapes
    for i in range(2, gs.env_config.NUM_SHAPE_SLOTS):
        gs.shapes[i] = up_shape_1.copy()

    assert not gs.valid_actions(), "Fixture setup failed: Valid actions still exist."

    return Node(initial_game_state=gs)
