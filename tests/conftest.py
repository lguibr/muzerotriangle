# File: tests/conftest.py
import random
from typing import cast

import numpy as np
import pytest
import torch
import torch.optim as optim

from muzerotriangle.config import EnvConfig, MCTSConfig, ModelConfig, TrainConfig
from muzerotriangle.nn import NeuralNetwork
from muzerotriangle.rl import ExperienceBuffer, Trainer

# Import MuZero Types
from muzerotriangle.utils.types import StateType, Trajectory, TrajectoryStep

# REMOVED: Experience

rng = np.random.default_rng()


# --- Fixtures --- (mock_env_config, mock_model_config, mock_train_config, mock_mcts_config remain the same) ---
@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    rows = 3
    cols = 3
    cols_per_row = [cols] * rows
    return EnvConfig(
        ROWS=rows,
        COLS=cols,
        COLS_PER_ROW=cols_per_row,
        NUM_SHAPE_SLOTS=1,
        MIN_LINE_LENGTH=3,
    )


@pytest.fixture(scope="session")
def mock_model_config(mock_env_config: EnvConfig) -> ModelConfig:
    int(mock_env_config.ACTION_DIM)  # type: ignore[call-overload]
    return ModelConfig(
        GRID_INPUT_CHANNELS=1,
        OTHER_NN_INPUT_FEATURES_DIM=10,
        HIDDEN_STATE_DIM=32,
        ACTION_ENCODING_DIM=8,
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=False,
        CONV_FILTERS=[4],
        CONV_KERNEL_SIZES=[3],
        CONV_STRIDES=[1],
        CONV_PADDING=[1],
        NUM_RESIDUAL_BLOCKS=0,
        RESIDUAL_BLOCK_FILTERS=4,
        USE_TRANSFORMER_IN_REP=False,
        REP_FC_DIMS_AFTER_ENCODER=[],
        DYNAMICS_NUM_RESIDUAL_BLOCKS=1,
        REWARD_HEAD_DIMS=[16],
        REWARD_SUPPORT_SIZE=5,
        PREDICTION_NUM_RESIDUAL_BLOCKS=1,
        POLICY_HEAD_DIMS=[16],
        VALUE_HEAD_DIMS=[16],
        NUM_VALUE_ATOMS=11,
        VALUE_MIN=-5.0,
        VALUE_MAX=5.0,
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    return TrainConfig(
        BATCH_SIZE=4,
        BUFFER_CAPACITY=100,
        MIN_BUFFER_SIZE_TO_TRAIN=10,
        USE_PER=False,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        MAX_TRAINING_STEPS=200,
    )  # Removed MuZero params for base mock


@pytest.fixture(scope="session")
def mock_mcts_config() -> MCTSConfig:
    # Add discount here
    return MCTSConfig(
        num_simulations=10,
        puct_coefficient=1.5,
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_anneal_steps=5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_search_depth=10,
        discount=0.99,
    )  # Added discount


@pytest.fixture(scope="session")
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape, dtype=np.float32),
        "other_features": rng.random(other_shape, dtype=np.float32),
    }


# --- REMOVED mock_experience ---


# --- ADD MuZero Data Fixtures ---
@pytest.fixture(scope="session")
def mock_trajectory_step_global(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> TrajectoryStep:
    """Creates a single mock TrajectoryStep (session scoped)."""
    action_dim = int(mock_env_config.ACTION_DIM)  # type: ignore[call-overload]
    return {
        "observation": mock_state_type,
        "action": random.randint(0, action_dim - 1) if action_dim > 0 else 0,
        "reward": random.uniform(-1, 1),
        "policy_target": dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
        else {},
        "value_target": random.uniform(-1, 1),
        "hidden_state": None,
    }


@pytest.fixture(scope="session")
def mock_trajectory_global(mock_trajectory_step_global: TrajectoryStep) -> Trajectory:
    """Creates a mock Trajectory (session scoped)."""
    # Use a fixed length relevant for training (e.g., unroll_steps + N)
    return [mock_trajectory_step_global.copy() for _ in range(10)]  # Example length 10


# --- END ADD MuZero Data Fixtures ---


@pytest.fixture(scope="session")
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    device = torch.device("cpu")
    nn = NeuralNetwork(mock_model_config, mock_env_config, mock_train_config, device)
    return nn


@pytest.fixture(scope="session")
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture(scope="session")
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    return cast("optim.Optimizer", mock_trainer.optimizer)


@pytest.fixture
def mock_experience_buffer(mock_train_config: TrainConfig) -> ExperienceBuffer:
    return ExperienceBuffer(mock_train_config)


@pytest.fixture
def filled_mock_buffer(
    mock_experience_buffer: ExperienceBuffer, mock_trajectory_global: Trajectory
) -> ExperienceBuffer:
    """Provides a buffer filled with mock trajectories."""
    # Fill based on total steps
    while mock_experience_buffer.total_steps < mock_experience_buffer.min_size_to_train:
        mock_experience_buffer.add(mock_trajectory_global[:])  # Add copy of trajectory
    return mock_experience_buffer
