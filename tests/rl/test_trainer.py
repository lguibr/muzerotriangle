# File: tests/rl/test_trainer.py
# File: tests/rl/test_trainer.py
import random
from typing import cast  # Import cast

import numpy as np
import pytest
import torch

from muzerotriangle.config import (  # Import TrainConfig
    EnvConfig,
    ModelConfig,
    TrainConfig,
)
from muzerotriangle.nn import NeuralNetwork
from muzerotriangle.rl import Trainer
from muzerotriangle.utils.types import (
    SampledBatch,
    SampledBatchPER,  # Import PER type
    SampledSequence,
    StateType,
    TrajectoryStep,
)
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
    """Use MuZero defaults: PER enabled, Unroll steps."""
    cfg = mock_train_config.model_copy(deep=True)
    cfg.USE_PER = True  # Enable PER
    cfg.MUZERO_UNROLL_STEPS = 3
    cfg.N_STEP_RETURNS = 5
    cfg.POLICY_LOSS_WEIGHT = 1.0
    cfg.VALUE_LOSS_WEIGHT = 0.25
    cfg.REWARD_LOSS_WEIGHT = 1.0
    return cast("TrainConfig", cfg)


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
        "policy_target": (
            dict.fromkeys(range(action_dim), 1.0 / action_dim) if action_dim > 0 else {}
        ),
        "value_target": random.uniform(-1, 1),
        "n_step_reward_target": random.uniform(-1, 1),  # Add n-step target
        "hidden_state": None,
    }


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


@pytest.fixture
def mock_per_batch(
    mock_batch: SampledBatch, train_config: TrainConfig
) -> SampledBatchPER:
    batch_size = train_config.BATCH_SIZE
    return SampledBatchPER(
        sequences=mock_batch,
        indices=np.arange(batch_size, dtype=np.int32),
        weights=np.ones(batch_size, dtype=np.float32) * 0.5,  # Example weights
    )


# --- Tests ---
def test_trainer_initialization_muzero(trainer: Trainer):
    assert trainer.nn is not None
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert hasattr(trainer, "scheduler")
    assert trainer.unroll_steps == trainer.train_config.MUZERO_UNROLL_STEPS


def test_prepare_batch_muzero(trainer: Trainer, mock_batch: SampledBatch):
    batch_size = trainer.train_config.BATCH_SIZE
    seq_len = trainer.unroll_steps + 1
    action_dim = int(trainer.env_config.ACTION_DIM)
    prepared_data = trainer._prepare_batch(mock_batch)
    assert isinstance(prepared_data, dict)
    expected_keys = {
        "grids",
        "others",
        "actions",
        "n_step_rewards",  # Check for n-step key
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
    assert prepared_data["n_step_rewards"].shape == (batch_size, seq_len)  # Check shape
    assert prepared_data["policy_targets"].shape == (batch_size, seq_len, action_dim)
    assert prepared_data["value_targets"].shape == (batch_size, seq_len)
    for key in expected_keys:
        assert prepared_data[key].device == trainer.device


def test_train_step_muzero_uniform(trainer: Trainer, mock_batch: SampledBatch):
    """Test train step with uniform batch."""
    trainer.model.to(trainer.device)
    initial_params = [p.clone() for p in trainer.model.parameters()]
    train_result = trainer.train_step(mock_batch)
    assert train_result is not None
    loss_info, td_errors = train_result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert "policy_loss" in loss_info
    assert "value_loss" in loss_info
    assert "reward_loss" in loss_info
    assert loss_info["total_loss"] > 0

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer.train_config.BATCH_SIZE,)

    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change."


def test_train_step_muzero_per(trainer: Trainer, mock_per_batch: SampledBatchPER):
    """Test train step with PER batch."""
    trainer.model.to(trainer.device)
    initial_params = [p.clone() for p in trainer.model.parameters()]
    train_result = trainer.train_step(mock_per_batch)
    assert train_result is not None
    loss_info, td_errors = train_result

    assert isinstance(loss_info, dict)
    assert "total_loss" in loss_info
    assert loss_info["total_loss"] > 0  # Loss should still be positive

    assert isinstance(td_errors, np.ndarray)
    assert td_errors.shape == (trainer.train_config.BATCH_SIZE,)

    params_changed = False
    for p_initial, p_final in zip(
        initial_params, trainer.model.parameters(), strict=True
    ):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change (PER)."


def test_train_step_empty_batch_muzero(trainer: Trainer):
    assert trainer.train_step([]) is None
