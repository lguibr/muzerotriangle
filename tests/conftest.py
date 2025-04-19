import random
from typing import cast

import numpy as np
import pytest
import torch
import torch.optim as optim

# Use absolute imports as tests might be run from different contexts
from muzerotriangle.config import EnvConfig, MCTSConfig, ModelConfig, TrainConfig
from muzerotriangle.nn import NeuralNetwork
from muzerotriangle.rl import ExperienceBuffer, Trainer
from muzerotriangle.utils.types import Experience, StateType

# Use default NumPy random number generator
rng = np.random.default_rng()


@pytest.fixture(scope="session")
def mock_env_config() -> EnvConfig:
    """Provides a default, *valid* EnvConfig for tests (session-scoped)."""
    rows = 3
    cols = 3
    cols_per_row = [cols] * rows
    return EnvConfig(
        ROWS=rows,
        COLS=cols,
        COLS_PER_ROW=cols_per_row,
        NUM_SHAPE_SLOTS=1,
        MIN_LINE_LENGTH=3,
        # Keep default rewards for now
    )


@pytest.fixture(scope="session")
def mock_model_config(mock_env_config: EnvConfig) -> ModelConfig:
    """Provides a default ModelConfig compatible with MuZero tests."""
    action_dim_int = int(mock_env_config.ACTION_DIM) # type: ignore[call-overload]

    # Simple config for testing MuZero components
    return ModelConfig(
        # Input/Shared
        GRID_INPUT_CHANNELS=1,
        OTHER_NN_INPUT_FEATURES_DIM=10, # Keep this compatible with mock_state_type
        HIDDEN_STATE_DIM=32,            # Smaller hidden state for tests
        ACTION_ENCODING_DIM=8,          # Smaller action embedding
        ACTIVATION_FUNCTION="ReLU",
        USE_BATCH_NORM=False,           # Disable BN for simpler testing initially

        # Representation (h)
        CONV_FILTERS=[4],               # Very simple CNN
        CONV_KERNEL_SIZES=[3],
        CONV_STRIDES=[1],
        CONV_PADDING=[1],
        NUM_RESIDUAL_BLOCKS=0,          # No ResBlocks in CNN part
        RESIDUAL_BLOCK_FILTERS=4,       # Matches last conv
        USE_TRANSFORMER_IN_REP=False,   # No Transformer in representation
        REP_FC_DIMS_AFTER_ENCODER=[], # Direct projection after CNN encoder + other feats

        # Dynamics (g)
        DYNAMICS_NUM_RESIDUAL_BLOCKS=1, # Minimal blocks (placeholder linear layers)
        REWARD_HEAD_DIMS=[16],          # Simple reward head
        REWARD_SUPPORT_SIZE=5,          # Small reward support (-2 to +2)

        # Prediction (f)
        PREDICTION_NUM_RESIDUAL_BLOCKS=1, # Minimal blocks (placeholder linear layers)
        POLICY_HEAD_DIMS=[16],          # Simple policy head
        VALUE_HEAD_DIMS=[16],           # Simple value head
        NUM_VALUE_ATOMS=11,             # Small value support
        VALUE_MIN=-5.0,
        VALUE_MAX=5.0,
    )


@pytest.fixture(scope="session")
def mock_train_config() -> TrainConfig:
    """Provides a default TrainConfig for tests (session-scoped)."""
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
        # MuZero Specific (Defaults will need adding to TrainConfig later)
        # UNROLL_STEPS=5,
        # POLICY_LOSS_WEIGHT=1.0,
        # VALUE_LOSS_WEIGHT=0.25,
        # REWARD_LOSS_WEIGHT=1.0,
        # ... other TrainConfig defaults ...
    )


@pytest.fixture(scope="session")
def mock_mcts_config() -> MCTSConfig:
    """Provides a default MCTSConfig for tests (session-scoped)."""
    return MCTSConfig(
        num_simulations=10,
        puct_coefficient=1.5,
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_anneal_steps=5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        max_search_depth=10,
    )


@pytest.fixture(scope="session")
def mock_state_type(
    mock_model_config: ModelConfig, mock_env_config: EnvConfig
) -> StateType:
    """Creates a mock StateType dictionary with correct shapes."""
    grid_shape = (
        mock_model_config.GRID_INPUT_CHANNELS,
        mock_env_config.ROWS,
        mock_env_config.COLS,
    )
    # Use the dimension from the updated mock_model_config
    other_shape = (mock_model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape, dtype=np.float32),
        "other_features": rng.random(other_shape, dtype=np.float32),
    }


@pytest.fixture(scope="session")
def mock_experience(
    mock_state_type: StateType, mock_env_config: EnvConfig
) -> Experience:
    """Creates a mock Experience tuple."""
    action_dim = int(mock_env_config.ACTION_DIM) # type: ignore[call-overload]
    policy_target = (
        dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
        else {0: 1.0}
    )
    value_target = random.uniform(-1, 1)
    return (mock_state_type, policy_target, value_target)


@pytest.fixture(scope="session")
def mock_nn_interface(
    mock_model_config: ModelConfig,
    mock_env_config: EnvConfig,
    mock_train_config: TrainConfig,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance with the MuZeroNet."""
    device = torch.device("cpu")
    # Use the MuZero-compatible mock configs
    nn_interface = NeuralNetwork(
        mock_model_config, mock_env_config, mock_train_config, device
    )
    return nn_interface


@pytest.fixture(scope="session")
def mock_trainer(
    mock_nn_interface: NeuralNetwork,
    mock_train_config: TrainConfig,
    mock_env_config: EnvConfig,
) -> Trainer:
    """Provides a Trainer instance."""
    # Trainer will need updates for MuZero loss later
    return Trainer(mock_nn_interface, mock_train_config, mock_env_config)


@pytest.fixture(scope="session")
def mock_optimizer(mock_trainer: Trainer) -> optim.Optimizer:
    """Provides the optimizer from the mock_trainer."""
    return cast("optim.Optimizer", mock_trainer.optimizer)


@pytest.fixture
def mock_experience_buffer(mock_train_config: TrainConfig) -> ExperienceBuffer:
    """Provides an ExperienceBuffer instance."""
    # Buffer will need updates for MuZero trajectories later
    return ExperienceBuffer(mock_train_config)


@pytest.fixture
def filled_mock_buffer(
    mock_experience_buffer: ExperienceBuffer, mock_experience: Experience
) -> ExperienceBuffer:
    """Provides a buffer filled with some mock experiences."""
    for _ in range(mock_experience_buffer.min_size_to_train + 5):
        state_copy: StateType = {
            "grid": mock_experience[0]["grid"].copy(),
            "other_features": mock_experience[0]["other_features"].copy(),
        }
        state_copy["grid"] += (
            rng.standard_normal(state_copy["grid"].shape, dtype=np.float32) * 0.1
        )
        exp_copy: Experience = (state_copy, mock_experience[1], random.uniform(-1, 1))
        mock_experience_buffer.add(exp_copy)
    return mock_experience_buffer