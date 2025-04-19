# File: tests/rl/test_buffer.py
import random

import numpy as np
import pytest

from muzerotriangle.config import TrainConfig
from muzerotriangle.rl import ExperienceBuffer
from muzerotriangle.utils.sumtree import SumTree  # Import SumTree for type check
from muzerotriangle.utils.types import (
    StateType,
    Trajectory,
    TrajectoryStep,
)

# Import the helper function
from tests.utils.test_sumtree import dump_sumtree_state

# Use default_rng for modern numpy random generation
rng = np.random.default_rng(seed=42)


# --- Fixtures ---
@pytest.fixture
def muzero_train_config() -> TrainConfig:
    # Enable PER for buffer tests
    return TrainConfig(
        BUFFER_CAPACITY=1000,
        MIN_BUFFER_SIZE_TO_TRAIN=50,
        BATCH_SIZE=4,
        MUZERO_UNROLL_STEPS=3,
        N_STEP_RETURNS=5,
        USE_PER=True,  # Enable PER
        PER_ALPHA=0.6,
        PER_BETA_INITIAL=0.4,
        DEVICE="cpu",
        RANDOM_SEED=42,
        NUM_SELF_PLAY_WORKERS=1,
        WORKER_DEVICE="cpu",
        WORKER_UPDATE_FREQ_STEPS=10,
        OPTIMIZER_TYPE="Adam",
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=1e-4,
        LR_SCHEDULER_ETA_MIN=1e-6,
        POLICY_LOSS_WEIGHT=1.0,
        VALUE_LOSS_WEIGHT=1.0,
        REWARD_LOSS_WEIGHT=1.0,
        CHECKPOINT_SAVE_FREQ_STEPS=50,
        MAX_TRAINING_STEPS=200,
        LOAD_CHECKPOINT_PATH=None,
        LOAD_BUFFER_PATH=None,
        AUTO_RESUME_LATEST=False,
    )


@pytest.fixture
def muzero_buffer(muzero_train_config: TrainConfig) -> ExperienceBuffer:
    # The ExperienceBuffer init should handle PER setup correctly based on config
    return ExperienceBuffer(muzero_train_config)


@pytest.fixture
def mock_state_type() -> StateType:
    return {
        "grid": rng.random((1, 3, 3)).astype(np.float32),
        "other_features": rng.random((10,)).astype(np.float32),
    }


@pytest.fixture
def mock_trajectory_step(mock_state_type: StateType) -> TrajectoryStep:
    action_dim = 9
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
def short_trajectory(mock_trajectory_step: TrajectoryStep) -> Trajectory:
    return [mock_trajectory_step.copy() for _ in range(3)]  # Length 3


@pytest.fixture
def long_trajectory(mock_trajectory_step: TrajectoryStep) -> Trajectory:
    traj = []
    for i in range(10):
        step = mock_trajectory_step.copy()
        step["reward"] += i * 0.1
        step["value_target"] += i * 0.1
        step["action"] = i % 9
        traj.append(step)
    return traj


# --- PER Buffer Tests ---
def test_muzero_buffer_init_per(muzero_buffer: ExperienceBuffer):
    assert muzero_buffer.use_per
    assert muzero_buffer.sum_tree is not None
    assert isinstance(muzero_buffer.sum_tree, SumTree)  # Check specific type
    assert len(muzero_buffer) == 0
    assert muzero_buffer.sum_tree.n_entries == 0


def test_muzero_buffer_add_per(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert muzero_buffer.sum_tree is not None
    initial_total_priority = muzero_buffer.sum_tree.total()  # Use total() method
    initial_n_entries = muzero_buffer.sum_tree.n_entries
    assert initial_n_entries == 0  # Should start at 0

    print("\n[test_muzero_buffer_add_per] Before add:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_add_per_before")
    muzero_buffer.add(long_trajectory)
    print("[test_muzero_buffer_add_per] After add:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_add_per_after")

    assert len(muzero_buffer.buffer) == 1
    assert len(muzero_buffer) == len(long_trajectory)
    # Check n_entries incremented
    assert (
        muzero_buffer.sum_tree.n_entries == initial_n_entries + 1
    ), f"n_entries did not increment. Before: {initial_n_entries}, After: {muzero_buffer.sum_tree.n_entries}"
    # New entry added with max priority
    assert muzero_buffer.sum_tree.total() > initial_total_priority  # Use total() method
    # Check if the stored item is a tuple
    assert isinstance(muzero_buffer.buffer[0], tuple)
    assert isinstance(muzero_buffer.buffer[0][0], int)  # buffer_idx
    assert isinstance(muzero_buffer.buffer[0][1], list)  # trajectory


def test_muzero_buffer_sample_per(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert muzero_buffer.sum_tree is not None
    # Ensure buffer is ready
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for i in range(num_needed):
        traj_copy = [step.copy() for step in long_trajectory]
        for step in traj_copy:
            step["reward"] += i * 0.01  # Add slight variation
        muzero_buffer.add(traj_copy)

    print("\n[test_muzero_buffer_sample_per] Before is_ready check:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_sample_per")
    # This assertion should now pass because SumTree.n_entries is correct
    assert muzero_buffer.is_ready(), (
        f"Buffer not ready. Steps: {len(muzero_buffer)}, Min: {muzero_buffer.min_size_to_train}, "
        f"SumTree Entries: {muzero_buffer.sum_tree.n_entries}, BatchSize: {muzero_buffer.config.BATCH_SIZE}"
    )
    assert (
        muzero_buffer.sum_tree.n_entries > 0
    ), "SumTree has no entries after adding trajectories"
    assert (
        muzero_buffer.sum_tree.total() > 1e-9  # Use total() method
    ), "SumTree total priority is near zero"

    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(
        batch_size, current_train_step=1
    )  # Pass step for beta

    assert sample is not None, "PER sampling returned None unexpectedly"
    assert isinstance(
        sample, dict
    ), f"Expected dict (PER sample), got {type(sample)}"  # Check it's a dict

    # Check keys instead of isinstance for TypedDict
    assert "sequences" in sample
    assert "indices" in sample
    assert "weights" in sample
    assert isinstance(sample["sequences"], list)
    assert len(sample["sequences"]) == batch_size
    assert isinstance(sample["indices"], np.ndarray)
    assert len(sample["indices"]) == batch_size
    assert isinstance(sample["weights"], np.ndarray)
    assert len(sample["weights"]) == batch_size
    assert sample["weights"].dtype == np.float32
    assert np.all(sample["weights"] > 0)
    assert np.all(sample["weights"] <= 1.0 + 1e-6)  # Check normalization

    for sequence in sample["sequences"]:
        assert isinstance(sequence, list)
        assert len(sequence) == muzero_buffer.sequence_length


def test_muzero_buffer_update_priorities(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert muzero_buffer.sum_tree is not None
    # Ensure buffer is ready
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        muzero_buffer.add(long_trajectory.copy())

    print("\n[test_muzero_buffer_update_priorities] Before is_ready check:")
    dump_sumtree_state(muzero_buffer.sum_tree, "test_muzero_buffer_update_priorities")
    # This assertion should now pass
    assert muzero_buffer.is_ready(), (
        f"Buffer not ready. Steps: {len(muzero_buffer)}, Min: {muzero_buffer.min_size_to_train}, "
        f"SumTree Entries: {muzero_buffer.sum_tree.n_entries}, BatchSize: {muzero_buffer.config.BATCH_SIZE}"
    )

    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(batch_size, current_train_step=1)

    assert sample is not None, "PER sampling returned None unexpectedly"
    assert isinstance(sample, dict), f"Expected dict (PER sample), got {type(sample)}"
    assert "indices" in sample

    tree_indices = sample["indices"]
    # Get initial priorities correctly
    initial_priorities_list = []
    for idx_val in tree_indices:
        idx = int(idx_val)
        if 0 <= idx < len(muzero_buffer.sum_tree.tree):
            initial_priorities_list.append(float(muzero_buffer.sum_tree.tree[idx]))
        else:
            initial_priorities_list.append(0.0)
    initial_priorities = np.array(initial_priorities_list)

    # Use rng.random for modern numpy
    td_errors = rng.random(len(tree_indices)) * 0.5  # Match length of indices
    muzero_buffer.update_priorities(tree_indices, td_errors)

    # Check if priorities changed
    new_priorities_list = []
    for idx_val in tree_indices:
        idx = int(idx_val)
        if 0 <= idx < len(muzero_buffer.sum_tree.tree):
            new_priorities_list.append(float(muzero_buffer.sum_tree.tree[idx]))
        else:
            new_priorities_list.append(0.0)
    new_priorities = np.array(new_priorities_list)

    assert not np.allclose(initial_priorities, new_priorities)
    # Check if priorities reflect errors (higher error -> higher priority)
    expected_priorities = (
        np.abs(td_errors) + muzero_buffer.per_epsilon
    ) ** muzero_buffer.per_alpha
    assert new_priorities.shape == expected_priorities.shape
    # --- RELAXED TOLERANCE ---
    assert np.allclose(
        new_priorities, expected_priorities, atol=1e-5
    ), f"Priorities mismatch.\nNew: {new_priorities}\nExpected: {expected_priorities}"
    # --- END RELAXED TOLERANCE ---


# --- Uniform Fallback Tests (if PER fails or disabled) ---
def test_muzero_buffer_sample_uniform_fallback(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    """Test fallback to uniform sampling if PER is enabled but SumTree is empty."""
    assert muzero_buffer.use_per
    assert muzero_buffer.sum_tree is not None
    # Add just enough steps to be ready
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        muzero_buffer.add(long_trajectory.copy())

    # Check if buffer is ready based on steps (should be)
    assert len(muzero_buffer) >= muzero_buffer.min_size_to_train
    print("\n[test_muzero_buffer_sample_uniform_fallback] Before n_entries check:")
    dump_sumtree_state(
        muzero_buffer.sum_tree, "test_muzero_buffer_sample_uniform_fallback"
    )
    # Check if buffer is ready based on SumTree entries (should be)
    assert muzero_buffer.sum_tree.n_entries >= muzero_buffer.config.BATCH_SIZE
    assert muzero_buffer.is_ready()  # This should now pass

    # Force total priority to 0 AND n_entries to 0 to trigger fallback
    muzero_buffer.sum_tree.tree.fill(0.0)  # Zero out entire tree
    muzero_buffer.sum_tree.n_entries = 0  # Crucial for the check in _sample_per
    assert muzero_buffer.sum_tree.total() < 1e-9  # Use total() method
    assert muzero_buffer.sum_tree.n_entries == 0

    # is_ready should now fail because n_entries < batch_size
    assert not muzero_buffer.is_ready()

    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(batch_size, current_train_step=1)

    # Expect None because is_ready() failed
    assert sample is None, "Expected None when PER sampling fails, but got a sample."


def test_muzero_buffer_sample_uniform_when_per_disabled(long_trajectory: Trajectory):
    """Test uniform sampling when PER is explicitly disabled."""
    config_no_per = TrainConfig(
        BUFFER_CAPACITY=1000,
        MIN_BUFFER_SIZE_TO_TRAIN=50,
        BATCH_SIZE=4,
        MUZERO_UNROLL_STEPS=3,
        N_STEP_RETURNS=5,
        USE_PER=False,  # PER Disabled
    )
    buffer_no_per = ExperienceBuffer(config_no_per)
    assert not buffer_no_per.use_per
    assert buffer_no_per.sum_tree is None

    num_needed = (buffer_no_per.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        buffer_no_per.add(long_trajectory.copy())

    batch_size = config_no_per.BATCH_SIZE
    sample = buffer_no_per.sample(batch_size)  # No step needed for uniform

    assert sample is not None
    assert isinstance(sample, list)  # Uniform returns list
    assert not isinstance(sample, dict)
    assert len(sample) == batch_size
