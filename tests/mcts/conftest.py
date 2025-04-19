# File: tests/mcts/conftest.py

import pytest
import torch

from muzerotriangle.config import EnvConfig, ModelConfig
from muzerotriangle.environment import GameState
from muzerotriangle.mcts.core.node import Node
from muzerotriangle.utils.types import ActionType


@pytest.fixture
def real_game_state(mock_env_config: EnvConfig) -> GameState:
    return GameState(config=mock_env_config, initial_seed=123)


class MockMuZeroNetwork:
    # ... (init, _state_to_tensors, _get_mock_logits, _logits_to_scalar, initial_inference, predict, evaluate, evaluate_batch remain the same) ...
    def __init__(self, model_config, env_config):
        self.model_config = model_config
        self.env_config = env_config
        self.action_dim = int(env_config.ACTION_DIM)
        self.hidden_dim = model_config.HIDDEN_STATE_DIM  # type: ignore[call-overload]
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

    def _state_to_tensors(self, state):
        grid_shape = (
            1,
            self.model_config.GRID_INPUT_CHANNELS,
            self.env_config.ROWS,
            self.env_config.COLS,
        )
        other_shape = (1, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
        return torch.randn(grid_shape), torch.randn(other_shape)

    def _get_mock_logits(self, batch_size, num_classes):
        return torch.ones((batch_size, num_classes), device=self.device) / num_classes

    def _logits_to_scalar(self, logits, support):
        probs = torch.softmax(logits, dim=-1)
        support_expanded = support.expand_as(probs)
        return torch.sum(probs * support_expanded, dim=-1)

    def initial_inference(self, observation):
        if isinstance(observation, GameState):
            _ = self._state_to_tensors(observation)
        batch_size = 1
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        reward_logits = self._get_mock_logits(
            batch_size, self.model_config.REWARD_SUPPORT_SIZE
        )
        hidden_state = torch.randn((batch_size, self.hidden_dim), device=self.device)
        return policy_logits, value_logits, reward_logits, hidden_state

    def predict(self, hidden_state):
        batch_size = hidden_state.shape[0]
        policy_logits = self._get_mock_logits(batch_size, self.action_dim)
        value_logits = self._get_mock_logits(
            batch_size, self.model_config.NUM_VALUE_ATOMS
        )
        return policy_logits, value_logits

    def evaluate(self, state):
        policy_logits, value_logits, _, _ = self.initial_inference(state)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        value = self._logits_to_scalar(value_logits, self.support).item()
        policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
        return policy_map, value

    def evaluate_batch(self, states):
        return [self.evaluate(s) for s in states]

    def dynamics(
        self, hidden_state: torch.Tensor | None, action: ActionType | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Add None check ---
        if hidden_state is None:
            raise ValueError("Dynamics function received None hidden_state")
        # ---
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
        next_hidden_state[:, 0] += (action_val / self.action_dim) * 0.1
        return next_hidden_state, reward_logits


@pytest.fixture
def mock_muzero_network(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> MockMuZeroNetwork:
    return MockMuZeroNetwork(mock_model_config, mock_env_config)


@pytest.fixture
def root_node_real_state(real_game_state: GameState) -> Node:
    return Node(initial_game_state=real_game_state)


@pytest.fixture
def node_with_hidden_state(mock_model_config: ModelConfig) -> Node:
    hidden_state = torch.randn((mock_model_config.HIDDEN_STATE_DIM,))
    return Node(prior=0.2, hidden_state=hidden_state, reward=0.1, action_taken=1)


@pytest.fixture
def expanded_root_node(
    root_node_real_state: Node,
    mock_muzero_network: MockMuZeroNetwork,
    mock_env_config: EnvConfig,
) -> Node:
    root = root_node_real_state
    game_state = root.initial_game_state
    assert game_state is not None
    policy_logits, value_logits_init, _, initial_hidden_state = (
        mock_muzero_network.initial_inference(game_state)
    )
    root.hidden_state = initial_hidden_state.squeeze(0)
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
    valid_actions = game_state.valid_actions()
    for action in valid_actions:
        prior = policy_map.get(action, 0.0)
        # --- Add unsqueeze only if hidden_state is not None ---
        hs_batch = (
            root.hidden_state.unsqueeze(0) if root.hidden_state is not None else None
        )
        if hs_batch is None:
            continue  # Skip if root hidden state is None
        next_hidden_state_tensor, reward_logits = mock_muzero_network.dynamics(
            hs_batch, action
        )
        # ---
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
    root.visit_count = 1
    root.value_sum = mock_muzero_network._logits_to_scalar(
        value_logits_init, mock_muzero_network.support
    ).item()
    return root
