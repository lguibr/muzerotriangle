# File: muzerotriangle/utils/types.py
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
    n_step_reward_target: float  # N-step discounted reward target R_t^{(N)}
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


# --- Prioritized Experience Replay (PER) ---
class SampledBatchPER(TypedDict):
    """Data structure for samples from PER buffer."""

    sequences: SampledBatch  # The batch of sampled sequences
    indices: np.ndarray  # Indices in the SumTree for priority updates
    weights: np.ndarray  # Importance sampling weights


# --- Statistics ---


class StepInfo(TypedDict, total=False):
    """Dictionary to hold various step counters associated with a metric."""

    global_step: int
    buffer_size: int  # Can now represent total steps or trajectories in buffer
    game_step_index: int
    # Add other relevant step types if needed


StatsCollectorData = dict[str, deque[tuple[StepInfo, float]]]
