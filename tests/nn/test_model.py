# File: tests/nn/test_model.py
import pytest
import torch

# Use absolute imports for consistency
from muzerotriangle.config import EnvConfig, ModelConfig
from muzerotriangle.nn.model import MuZeroNet  # Import MuZeroNet


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    return mock_model_config


@pytest.fixture
def model(model_config: ModelConfig, env_config: EnvConfig) -> MuZeroNet:
    """Provides an instance of the MuZeroNet model."""
    return MuZeroNet(model_config, env_config)


def test_muzero_model_initialization(
    model: MuZeroNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test if the MuZeroNet model initializes without errors."""
    assert model is not None
    assert model.action_dim == int(env_config.ACTION_DIM)  # type: ignore[call-overload]
    assert model.hidden_dim == model_config.HIDDEN_STATE_DIM
    # Add more checks for internal components if needed
    assert hasattr(model, "representation_encoder")
    assert hasattr(model, "representation_projector")
    assert hasattr(model, "action_encoder")
    assert hasattr(model, "dynamics_core")
    assert hasattr(model, "reward_head")
    assert hasattr(model, "prediction_core")
    assert hasattr(model, "policy_head")
    assert hasattr(model, "value_head")


def test_representation_function(
    model: MuZeroNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test the representation function (h)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)
    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        hidden_state = model.represent(dummy_grid, dummy_other)

    assert hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert hidden_state.dtype == torch.float32


def test_dynamics_function(model: MuZeroNet, model_config: ModelConfig):
    """Test the dynamics function (g)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=device
    )
    # Test with batch of actions
    dummy_actions = torch.randint(0, model.action_dim, (batch_size,), device=device)

    with torch.no_grad():
        next_hidden_state, reward_logits = model.dynamics(
            dummy_hidden_state, dummy_actions
        )

    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)
    assert next_hidden_state.dtype == torch.float32
    assert reward_logits.dtype == torch.float32


def test_dynamics_function_single_action(model: MuZeroNet, model_config: ModelConfig):
    """Test the dynamics function (g) with a single action."""
    batch_size = 1  # Test with batch size 1
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=device
    )
    # Test with single integer action
    dummy_action_int = model.action_dim // 2

    with torch.no_grad():
        next_hidden_state, reward_logits = model.dynamics(
            dummy_hidden_state, dummy_action_int
        )

    assert next_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
    assert reward_logits.shape == (batch_size, model_config.REWARD_SUPPORT_SIZE)


def test_prediction_function(model: MuZeroNet, model_config: ModelConfig):
    """Test the prediction function (f)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    dummy_hidden_state = torch.randn(
        (batch_size, model_config.HIDDEN_STATE_DIM), device=device
    )

    with torch.no_grad():
        policy_logits, value_logits = model.predict(dummy_hidden_state)

    assert policy_logits.shape == (batch_size, model.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert policy_logits.dtype == torch.float32
    assert value_logits.dtype == torch.float32


def test_forward_initial_inference(
    model: MuZeroNet, model_config: ModelConfig, env_config: EnvConfig
):
    """Test the main forward pass for initial inference (h + f)."""
    batch_size = 4
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    grid_shape = (
        batch_size,
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (batch_size, model_config.OTHER_NN_INPUT_FEATURES_DIM)
    dummy_grid = torch.randn(grid_shape, device=device)
    dummy_other = torch.randn(other_shape, device=device)

    with torch.no_grad():
        policy_logits, value_logits, initial_hidden_state = model(
            dummy_grid, dummy_other
        )

    assert policy_logits.shape == (batch_size, model.action_dim)
    assert value_logits.shape == (batch_size, model_config.NUM_VALUE_ATOMS)
    assert initial_hidden_state.shape == (batch_size, model_config.HIDDEN_STATE_DIM)
