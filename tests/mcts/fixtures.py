# File: tests/mcts/fixtures.py
# This file might be deprecated if all fixtures moved to conftest.py
# Assuming it's still used, make the necessary changes:

from typing import Any

import pytest
import torch  # Import torch

# Use absolute imports for consistency
from muzerotriangle.config import EnvConfig, MCTSConfig
from muzerotriangle.mcts.core.node import Node


# --- Mock GameState --- (Keep as is)
class MockGameState:
    def __init__(
        self,
        current_step=0,
        is_terminal=False,
        outcome=0.0,
        valid_actions=None,
        env_config=None,
    ):
        self.current_step = current_step
        self._is_over = is_terminal
        self._outcome = outcome
        self.env_config = env_config if env_config else EnvConfig()
        action_dim_int = int(self.env_config.ACTION_DIM)  # type: ignore[call-overload]
        self._valid_actions = (
            valid_actions if valid_actions is not None else list(range(action_dim_int))
        )

    def is_over(self):
        return self._is_over

    def get_outcome(self):
        if not self._is_over:
            raise ValueError("Cannot get outcome of non-terminal state.")
        return self._outcome

    def valid_actions(self):
        return self._valid_actions

    def copy(self):
        return MockGameState(
            self.current_step,
            self._is_over,
            self._outcome,
            list(self._valid_actions),
            self.env_config,
        )

    def step(self, action):
        if action not in self._valid_actions:
            raise ValueError(f"Invalid action {action}")
        self.current_step += 1
        self._is_over = self.current_step >= 5
        self._outcome = 1.0 if self._is_over else 0.0
        return 0.0, self._is_over  # Return reward, done

    def __hash__(self):
        return hash((self.current_step, self._is_over, tuple(self._valid_actions)))

    def __eq__(self, other):
        return (
            isinstance(other, MockGameState)
            and self.current_step == other.current_step
            and self._is_over == other._is_over
            and self._valid_actions == other._valid_actions
        )


# --- Mock Network Evaluator --- (Keep as is)
class MockNetworkEvaluator:
    def __init__(self, default_policy=None, default_value=0.5, action_dim=3):
        self._default_policy = default_policy
        self._default_value = default_value
        self._action_dim = action_dim
        self.evaluation_history = []
        self.batch_evaluation_history = []

    def _get_policy(self, state):
        if self._default_policy:
            return self._default_policy
        valid_actions = state.valid_actions()
        prob = 1.0 / len(valid_actions) if valid_actions else 0
        return dict.fromkeys(valid_actions, prob) if valid_actions else {}

    def evaluate(self, state):
        self.evaluation_history.append(state)
        policy = self._get_policy(state)
        full_policy = dict.fromkeys(range(self._action_dim), 0.0)
        full_policy.update(policy)
        return full_policy, self._default_value

    def evaluate_batch(self, states):
        self.batch_evaluation_history.append(states)
        return [self.evaluate(s) for s in states]


# --- Pytest Fixtures --- (Adapt Node creation)
@pytest.fixture
def mock_env_config_local() -> EnvConfig:  # Renamed to avoid clash if imported
    return EnvConfig()


@pytest.fixture
def mock_mcts_config_local() -> MCTSConfig:  # Renamed
    return MCTSConfig()


@pytest.fixture
def mock_evaluator_local(
    mock_env_config_local: EnvConfig,
) -> MockNetworkEvaluator:  # Renamed
    action_dim_int = int(mock_env_config_local.ACTION_DIM)  # type: ignore[call-overload]
    return MockNetworkEvaluator(action_dim=action_dim_int)


@pytest.fixture
def root_node_mock_state_local(mock_env_config_local: EnvConfig) -> Node:  # Renamed
    state = MockGameState(env_config=mock_env_config_local)
    # --- Use initial_game_state ---
    return Node(initial_game_state=state)  # type: ignore[arg-type]


@pytest.fixture
def expanded_node_mock_state_local(
    root_node_mock_state_local: Node, mock_evaluator_local: MockNetworkEvaluator
) -> Node:  # Renamed
    root = root_node_mock_state_local
    mock_state: Any = root.initial_game_state  # Root holds GameState
    assert mock_state is not None  # Ensure game state exists
    policy, value = mock_evaluator_local.evaluate(mock_state)
    # --- Add dummy hidden state ---
    root.hidden_state = torch.randn(
        (32,)
    )  # Add dummy state after initial eval if needed

    for action in mock_state.valid_actions():
        prior = policy.get(action, 0.0)
        # --- Create child with hidden_state, reward, prior ---
        MockGameState(
            current_step=1, valid_actions=[0, 1], env_config=mock_state.env_config
        )
        # In reality, hidden_state and reward come from dynamics
        child_hidden_state = torch.randn((32,))
        child_reward = 0.05
        child = Node(
            # Use correct keywords
            prior=prior,
            hidden_state=child_hidden_state,
            reward=child_reward,
            parent=root,
            action_taken=action,
        )
        root.children[action] = child
    root.visit_count = 1
    # --- Use value_sum ---
    root.value_sum = value
    return root
