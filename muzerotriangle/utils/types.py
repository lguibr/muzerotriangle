# File: muzerotriangle/utils/types.py
from collections import deque
from collections.abc import Mapping

import numpy as np
from typing_extensions import TypedDict

# --- Core State & Action ---


class StateType(TypedDict):
    grid: np.ndarray  # (C, H, W) float32
    other_features: np.ndarray  # (OtherFeatDim,) float32


ActionType = int

# --- MCTS & Policy ---

PolicyTargetMapping = Mapping[ActionType, float]
PolicyValueOutput = tuple[
    Mapping[ActionType, float], float
]  # (Policy Map, Expected Scalar Value)

# --- MuZero Trajectory Data ---


class TrajectoryStep(TypedDict):
    """Data stored for a single step in a game trajectory."""

    observation: StateType  # Observation o_t from the environment
    action: ActionType  # Action a_{t+1} taken after observation
    reward: float  # Actual reward r_{t+1} received from environment
    policy_target: PolicyTargetMapping  # MCTS policy target pi_t at step t
    value_target: float  # MCTS value target z_t (e.g., root value) at step t
    hidden_state: (
        np.ndarray | None
    )  # Optional: Store hidden state s_t from NN for debugging/analysis


# A complete game trajectory
Trajectory = list[TrajectoryStep]

# --- Training Data ---

# A sequence sampled from a trajectory for training
# Contains K unroll steps + 1 initial step = K+1 steps total
SampledSequence = list[TrajectoryStep]
SampledBatch = list[SampledSequence]  # Batch of sequences

# --- Statistics ---


class StepInfo(TypedDict, total=False):
    """Dictionary to hold various step counters associated with a metric."""

    global_step: int
    buffer_size: int  # Can now represent total steps or trajectories in buffer
    game_step_index: int
    # Add other relevant step types if needed


StatsCollectorData = dict[str, deque[tuple[StepInfo, float]]]

# --- REMOVED: PER Types (temporarily disabled) ---
# class PERBatchSample(TypedDict):
#    batch: ExperienceBatch # This would need changing to SampledBatch
#    indices: np.ndarray
#    weights: np.ndarray
