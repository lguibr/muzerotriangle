# File: muzerotriangle/rl/core/buffer.py
import logging
import random
from collections import deque

from ...config import TrainConfig

# from ...utils.sumtree import SumTree # REMOVED: PER disabled
from ...utils.types import (
    SampledBatch,  # Use SampledBatch
    Trajectory,
)

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience Replay Buffer for MuZero. Stores complete game trajectories.
    Samples sequences of fixed length for training.
    **Currently uses uniform sampling only (PER disabled).**
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.capacity = (
            config.BUFFER_CAPACITY
        )  # Capacity in terms of total steps/transitions
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN  # Min total steps
        self.unroll_steps = config.MUZERO_UNROLL_STEPS
        self.sequence_length = self.unroll_steps + 1  # K unroll steps + 1 initial state

        self.buffer: deque[Trajectory] = deque()
        self.total_steps = 0  # Track total steps stored across all trajectories

        # --- REMOVED: PER attributes ---
        # self.use_per = config.USE_PER
        # if self.use_per: ...
        # ---

        logger.info(
            f"MuZero Experience buffer initialized with uniform sampling. "
            f"Capacity (total steps): {self.capacity}, Sequence Length: {self.sequence_length}"
        )

    def add(self, trajectory: Trajectory):
        """Adds a complete trajectory to the buffer."""
        if not trajectory:
            logger.warning("Attempted to add an empty trajectory to the buffer.")
            return

        traj_len = len(trajectory)
        # --- Check capacity and evict oldest trajectories if needed ---
        while self.total_steps + traj_len > self.capacity and len(self.buffer) > 0:
            removed_traj = self.buffer.popleft()
            self.total_steps -= len(removed_traj)
            logger.debug(
                f"Buffer capacity reached. Removed oldest trajectory (len: {len(removed_traj)}). Current total steps: {self.total_steps}"
            )
        # ---

        self.buffer.append(trajectory)
        self.total_steps += traj_len
        logger.debug(
            f"Added trajectory (len: {traj_len}). Buffer trajectories: {len(self.buffer)}, Total steps: {self.total_steps}"
        )

    def add_batch(self, trajectories: list[Trajectory]):
        """Adds a batch of trajectories."""
        for traj in trajectories:
            self.add(traj)  # Use single add logic for capacity check

    def sample(
        self,
        batch_size: int,
        _current_train_step: int | None = None,  # Step unused for uniform
    ) -> SampledBatch | None:  # Returns batch of sequences
        """
        Samples a batch of sequences uniformly.
        Each sequence has length `unroll_steps + 1`.
        """
        if not self.is_ready():
            return None

        sampled_sequences: SampledBatch = []
        attempts = 0
        max_attempts = batch_size * 5  # Allow some failed samples

        while len(sampled_sequences) < batch_size and attempts < max_attempts:
            attempts += 1
            # 1. Sample a trajectory uniformly
            traj_index = random.randrange(len(self.buffer))
            trajectory = self.buffer[traj_index]

            # 2. Sample a starting index within the trajectory
            # Need at least sequence_length steps available
            if len(trajectory) < self.sequence_length:
                # logger.debug(f"Skipping trajectory {traj_index} (len {len(trajectory)} < required {self.sequence_length})")
                continue  # Skip short trajectories

            start_index = random.randrange(len(trajectory) - self.sequence_length + 1)

            # 3. Extract the sequence
            sequence = trajectory[start_index : start_index + self.sequence_length]

            # 4. Basic validation (ensure correct length)
            if len(sequence) == self.sequence_length:
                sampled_sequences.append(sequence)
            else:
                logger.warning(
                    f"Sampled sequence has incorrect length {len(sequence)} (expected {self.sequence_length}) from traj {traj_index}, index {start_index}."
                )

        if len(sampled_sequences) < batch_size:
            logger.warning(
                f"Could only sample {len(sampled_sequences)} sequences after {attempts} attempts (batch size {batch_size}). Buffer might contain many short trajectories."
            )
            # Return partial batch if needed? Or None? Let's return partial.
            # return None

        return sampled_sequences

    def __len__(self) -> int:
        """Returns the total number of steps stored in the buffer."""
        return self.total_steps

    def is_ready(self) -> bool:
        """Checks if the buffer has enough total steps to start training."""
        # Also ensure there are enough trajectories to form a batch
        return self.total_steps >= self.min_size_to_train and len(self.buffer) > 0

    # --- REMOVED: PER methods ---
    # def _get_priority(self, error: float) -> float: ...
    # def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray): ...
    # ---
