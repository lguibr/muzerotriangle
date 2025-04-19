# File: tests/rl/test_buffer.py
import random  # Add import
from collections import deque

import numpy as np
import pytest

from muzerotriangle.config import TrainConfig
from muzerotriangle.rl import ExperienceBuffer
from muzerotriangle.utils.types import StateType, Trajectory, TrajectoryStep
from tests.conftest import rng


# --- Fixtures --- (remain the same) ---
@pytest.fixture
def muzero_train_config() -> TrainConfig:
    return TrainConfig(
        BUFFER_CAPACITY=1000,
        MIN_BUFFER_SIZE_TO_TRAIN=50,
        BATCH_SIZE=4,
        MUZERO_UNROLL_STEPS=3,
        USE_PER=False,
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
        "policy_target": dict.fromkeys(range(action_dim), 1.0 / action_dim)
        if action_dim > 0
        else {},
        "value_target": random.uniform(-1, 1),
        "hidden_state": None,
    }


@pytest.fixture
def short_trajectory(mock_trajectory_step: TrajectoryStep) -> Trajectory:
    return [mock_trajectory_step.copy() for _ in range(3)]


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


# --- Uniform Buffer Tests (MuZero Adaptation) --- (remain the same) ---
def test_muzero_buffer_init(muzero_buffer: ExperienceBuffer):
    assert isinstance(muzero_buffer.buffer, deque)
    assert muzero_buffer.capacity == 1000
    assert len(muzero_buffer) == 0
    assert not muzero_buffer.is_ready()
    assert muzero_buffer.sequence_length == 4


def test_muzero_buffer_add_trajectory(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert len(muzero_buffer) == 0
    muzero_buffer.add(long_trajectory)
    assert len(muzero_buffer.buffer) == 1
    assert len(muzero_buffer) == len(long_trajectory)
    assert muzero_buffer.buffer[0] == long_trajectory


def test_muzero_buffer_add_batch_trajectories(
    muzero_buffer: ExperienceBuffer,
    long_trajectory: Trajectory,
    short_trajectory: Trajectory,
):
    batch = [long_trajectory, short_trajectory, long_trajectory.copy()]
    muzero_buffer.add_batch(batch)
    assert len(muzero_buffer.buffer) == 3
    expected_total_steps = (
        len(long_trajectory) + len(short_trajectory) + len(long_trajectory)
    )
    assert len(muzero_buffer) == expected_total_steps


def test_muzero_buffer_capacity(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    traj_len = len(long_trajectory)
    num_trajs_to_fill = muzero_buffer.capacity // traj_len + 1
    for i in range(num_trajs_to_fill + 5):
        traj_copy = [step.copy() for step in long_trajectory]
        for step in traj_copy:
            step["reward"] += i * 0.01
            muzero_buffer.add(traj_copy)
    assert len(muzero_buffer) <= muzero_buffer.capacity
    assert len(muzero_buffer.buffer) < num_trajs_to_fill + 5
    first_added_reward_start = long_trajectory[0]["reward"] + 0 * 0.01
    present = any(
        t[0]["reward"] == first_added_reward_start for t in muzero_buffer.buffer if t
    )
    assert not present


def test_muzero_buffer_is_ready(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    assert not muzero_buffer.is_ready()
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for _ in range(num_needed):
        muzero_buffer.add(long_trajectory.copy())
        assert muzero_buffer.is_ready()


def test_muzero_buffer_sample_sequence(
    muzero_buffer: ExperienceBuffer, long_trajectory: Trajectory
):
    num_needed = (muzero_buffer.min_size_to_train // len(long_trajectory)) + 1
    for i in range(num_needed):
        traj_copy = [step.copy() for step in long_trajectory]
    for step in traj_copy:
        step["reward"] += i * 0.01
        muzero_buffer.add(traj_copy)
    batch_size = muzero_buffer.config.BATCH_SIZE
    sample = muzero_buffer.sample(batch_size)
    assert sample is not None
    assert isinstance(sample, list)
    assert len(sample) == batch_size
    for sequence in sample:
        assert isinstance(sequence, list)
        assert len(sequence) == muzero_buffer.sequence_length
        assert all(isinstance(step, dict) for step in sequence)


def test_muzero_buffer_sample_not_ready(muzero_buffer: ExperienceBuffer):
    assert muzero_buffer.sample(muzero_buffer.config.BATCH_SIZE) is None


def test_muzero_buffer_sample_with_short_trajectories(
    muzero_buffer: ExperienceBuffer, short_trajectory: Trajectory
):
    for _ in range(10):
        muzero_buffer.add(short_trajectory.copy())
    assert len(muzero_buffer) >= muzero_buffer.min_size_to_train
    sample = muzero_buffer.sample(muzero_buffer.config.BATCH_SIZE)
    assert sample is None or len(sample) == 0
