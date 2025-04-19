# File: tests/rl/test_trainer.py
import random  # Add import

import numpy as np
import pytest
import torch

# ... (rest of imports and fixtures remain the same) ...
from muzerotriangle.config import EnvConfig, ModelConfig, TrainConfig
from muzerotriangle.nn import NeuralNetwork
from muzerotriangle.rl import Trainer
from muzerotriangle.utils.types import (
    SampledBatch,
    SampledSequence,
    StateType,
    TrajectoryStep,
)
from tests.conftest import rng


# --- Fixtures ---
# ... (fixtures remain the same) ...
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = False
    cfg.MUZERO_UNROLL_STEPS = 3
    cfg.POLICY_LOSS_WEIGHT = 1.0
    cfg.VALUE_LOSS_WEIGHT = 0.25
    cfg.REWARD_LOSS_WEIGHT = 1.0
    return cfg


@pytest.fixture
def nn_interface(
    model_config: ModelConfig, env_config: EnvConfig, train_config: TrainConfig
) -> NeuralNetwork:
    device = torch.device("cpu")
    nn = NeuralNetwork(model_config, env_config, train_config, device)
    nn.model.to(device)
    nn.model.eval()
    return nn


@pytest.fixture
def trainer(
    nn_interface: NeuralNetwork, train_config: TrainConfig, env_config: EnvConfig
) -> Trainer:
    return Trainer(nn_interface, train_config, env_config)


@pytest.fixture
def mock_state_type() -> StateType:
    return {
        "grid": rng.random((1, 3, 3)).astype(np.float32),
        "other_features": rng.random((10,)).astype(np.float32),
    }


@pytest.fixture
def mock_trajectory_step(
    mock_state_type: StateType, env_config: EnvConfig
) -> TrajectoryStep:
    action_dim = int(env_config.ACTION_DIM)
    return {
        "observation": mock_state_type,
        "action": random.randint(0, action_dim - 1) if action_dim > 0 else 0,
        "reward": random.uniform(-1, 1),
        "policy_target": dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
        else {},
        "value_target": random.uniform(-1, 1),
        "hidden_state": None,
    }  # type: ignore


@pytest.fixture
def mock_sequence(
    mock_trajectory_step: TrajectoryStep, train_config: TrainConfig
) -> SampledSequence:
    seq_len = train_config.MUZERO_UNROLL_STEPS + 1
    return [mock_trajectory_step.copy() for _ in range(seq_len)]


@pytest.fixture
def mock_batch(
    mock_sequence: SampledSequence, train_config: TrainConfig
) -> SampledBatch:
    return [mock_sequence[:] for _ in range(train_config.BATCH_SIZE)]


# --- Tests ---
# ... (test_trainer_initialization_muzero, test_prepare_batch_muzero remain the same) ...
def test_trainer_initialization_muzero(trainer: Trainer):
    assert trainer.nn is not None
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert hasattr(trainer, "scheduler")
    assert trainer.unroll_steps == trainer.train_config.MUZERO_UNROLL_STEPS


def test_prepare_batch_muzero(trainer: Trainer, mock_batch: SampledBatch):
    batch_size = trainer.train_config.BATCH_SIZE
    seq_len = trainer.unroll_steps + 1
    action_dim = int(trainer.env_config.ACTION_DIM)  # type: ignore
    prepared_data = trainer._prepare_batch(mock_batch)
    assert isinstance(prepared_data, dict)
    expected_keys = {
        "grids",
        "others",
        "actions",
        "rewards",
        "policy_targets",
        "value_targets",
    }
    assert set(prepared_data.keys()) == expected_keys
    assert prepared_data["grids"].shape == (
        batch_size,
        seq_len,
        trainer.model_config.GRID_INPUT_CHANNELS,
        trainer.env_config.ROWS,
        trainer.env_config.COLS,
    )
    assert prepared_data["others"].shape == (
        batch_size,
        seq_len,
        trainer.model_config.OTHER_NN_INPUT_FEATURES_DIM,
    )
    assert prepared_data["actions"].shape == (batch_size, seq_len)
    assert prepared_data["rewards"].shape == (batch_size, seq_len)
    assert prepared_data["policy_targets"].shape == (batch_size, seq_len, action_dim)
    assert prepared_data["value_targets"].shape == (batch_size, seq_len)
    for key in expected_keys:
        assert prepared_data[key].device == trainer.device


def test_train_step_muzero(trainer: Trainer, mock_batch: SampledBatch):
    trainer.model.to(trainer.device)
    initial_params = [p.clone() for p in trainer.model.parameters()]
    loss_info = trainer.train_step(mock_batch)
    assert loss_info is not None
    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert "reward_loss" in loss_info
    assert loss_info["total_loss"] > 0
    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change."


def test_train_step_empty_batch_muzero(trainer: Trainer):
    assert trainer.train_step([]) is None
