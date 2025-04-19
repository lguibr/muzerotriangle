# File: tests/nn/test_network.py
from typing import cast  # Import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from muzerotriangle.config import (  # Import TrainConfig
    EnvConfig,
    ModelConfig,
    TrainConfig,
)
from muzerotriangle.environment import GameState
from muzerotriangle.nn import MuZeroNet, NeuralNetwork
from muzerotriangle.utils.types import StateType
from tests.conftest import rng


# --- Fixtures ---
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.COMPILE_MODEL = True
    # Explicitly cast the copied and modified config
    return cast("TrainConfig", cfg)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> NeuralNetwork:
    nn = NeuralNetwork(model_config, env_config, train_config, device)
    nn.model.to(device)
    nn.model.eval()
    return nn


@pytest.fixture
def mock_game_state(env_config: EnvConfig) -> GameState:
    return GameState(config=env_config, initial_seed=123)


@pytest.fixture
def mock_game_state_batch(mock_game_state: GameState) -> list[GameState]:
    return [mock_game_state.copy() for _ in range(3)]


@pytest.fixture
def mock_state_type_nn(model_config: ModelConfig, env_config: EnvConfig) -> StateType:
    grid_shape = (model_config.GRID_INPUT_CHANNELS, env_config.ROWS, env_config.COLS)
    other_shape = (model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape).astype(np.float32),
        "other_features": rng.random(other_shape).astype(np.float32),
    }


# --- Tests ---
def test_nn_initialization_muzero(nn_interface: NeuralNetwork, device: torch.device):
    assert nn_interface is not None
    assert nn_interface.device == device
    model_to_check = getattr(nn_interface.model, "_orig_mod", nn_interface.model)
    assert isinstance(model_to_check, MuZeroNet)
    assert not nn_interface.model.training


@patch("muzerotriangle.nn.network.extract_state_features")
def test_initial_inference(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_state_type_nn: StateType,
    model_config: ModelConfig,
):
    mock_extract.return_value = mock_state_type_nn
    batch_size = 1
    policy_logits, value_logits, reward_logits, hidden_state = (
        nn_interface.initial_inference(mock_state_type_nn)
    )
    assert policy_logits.shape == (batch_size, nn_interface.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)
    assert hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32
    assert reward_logits.dtype == torch.float32
    assert hidden_state.dtype == torch.float32
    assert policy_logits.device == nn_interface.device


def test_recurrent_inference(nn_interface: NeuralNetwork, model_config: ModelConfig):
    batch_size = 4
    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=nn_interface.device
    )
    dummy_actions = torch.randint(
        0, nn_interface.action_dim, (batch_size,), device=nn_interface.device
    )
    policy_logits, value_logits, reward_logits, next_hidden_state = (
        nn_interface.recurrent_inference(dummy_hidden_state, dummy_actions)
    )
    assert policy_logits.shape == (batch_size, nn_interface.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)
    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32
    assert reward_logits.dtype == torch.float32
    assert next_hidden_state.dtype == torch.float32
    assert policy_logits.device == nn_interface.device


@patch("muzerotriangle.nn.network.extract_state_features")
def test_evaluate_single_muzero(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    mock_extract.return_value = mock_state_type_nn
    action_dim_int = int(env_config.ACTION_DIM)
    policy_map, value = nn_interface.evaluate(mock_game_state)
    mock_extract.assert_called_once_with(mock_game_state, nn_interface.model_config)
    assert isinstance(policy_map, dict)
    assert isinstance(value, float)
    assert len(policy_map) == action_dim_int
    assert abs(sum(policy_map.values()) - 1.0) < 1e-5


@patch("muzerotriangle.nn.network.extract_state_features")
def test_evaluate_batch_muzero(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state_batch: list[GameState],
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    mock_states = mock_game_state_batch
    batch_size = len(mock_states)
    mock_extract.side_effect = [
        {
            k: (v.copy() + i * 0.1 if isinstance(v, np.ndarray) else v)
            for k, v in mock_state_type_nn.items()
        }
        for i in range(batch_size)
    ]
    action_dim_int = int(env_config.ACTION_DIM)
    results = nn_interface.evaluate_batch(mock_states)
    assert mock_extract.call_count == batch_size
    assert isinstance(results, list)
    assert len(results) == batch_size
    for policy_map, value in results:
        assert isinstance(policy_map, dict)
        assert isinstance(value, float)
        assert len(policy_map) == action_dim_int
        assert abs(sum(policy_map.values()) - 1.0) < 1e-5


def test_get_set_weights_muzero(nn_interface: NeuralNetwork):
    initial_weights = nn_interface.get_weights()
    assert isinstance(initial_weights, dict)
    modified_weights = {}
    for k, v in initial_weights.items():
        modified_weights[k] = v + 0.1 if v.dtype.is_floating_point else v
    nn_interface.set_weights(modified_weights)
    new_weights = nn_interface.get_weights()
    for key in initial_weights:
        if initial_weights[key].dtype.is_floating_point:
            assert torch.allclose(modified_weights[key], new_weights[key], atol=1e-6), (
                f"Mismatch key {key}"
            )
        else:
            assert torch.equal(initial_weights[key], new_weights[key]), (
                f"Non-float mismatch key {key}"
            )
