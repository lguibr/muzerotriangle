# File: muzerotriangle/rl/core/buffer.py
import logging
import random
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt  # Import numpy typing

from ...utils.sumtree import SumTree
from ...utils.types import (
    SampledBatch,
    SampledBatchPER,  # Import PER batch type
    Trajectory,
)

if TYPE_CHECKING:
    from ...config import TrainConfig

    pass

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience Replay Buffer for MuZero. Stores complete game trajectories.
    Samples sequences of fixed length for training.
    Supports Prioritized Experience Replay (PER).
    """

    def __init__(self, config: "TrainConfig"):
        self.config = config
        self.capacity = (
            config.BUFFER_CAPACITY
        )  # Capacity in terms of total steps/transitions
        self.min_size_to_train = config.MIN_BUFFER_SIZE_TO_TRAIN  # Min total steps
        self.unroll_steps = config.MUZERO_UNROLL_STEPS
        self.sequence_length = self.unroll_steps + 1  # K unroll steps + 1 initial state

        # --- Data Storage ---
        # Stores tuples of (unique_buffer_index, trajectory)
        self.buffer: deque[tuple[int, Trajectory]] = deque()
        self.tree_idx_to_buffer_idx: dict[
            int, int
        ] = {}  # Maps SumTree leaf index to unique buffer_idx
        self.buffer_idx_to_tree_idx: dict[
            int, int
        ] = {}  # Maps unique buffer_idx to SumTree leaf index
        self.next_buffer_idx = 0  # Monotonically increasing index for unique ID
        self.total_steps = 0

        # --- PER Attributes ---
        self.use_per = config.USE_PER
        self.sum_tree: SumTree | None = None  # Initialize as None
        if self.use_per:
            # Estimate SumTree capacity based on trajectories, not steps
            # A better estimate might be needed, but this is a starting point
            estimated_avg_traj_len = 50  # Heuristic, adjust as needed
            estimated_num_trajectories = max(
                1, config.BUFFER_CAPACITY // estimated_avg_traj_len
            )
            # Give SumTree more capacity than just estimated trajectories
            sumtree_capacity = int(estimated_num_trajectories * 1.5)
            # Ensure it's large enough for batch size and some minimum
            sumtree_capacity = max(
                sumtree_capacity, config.BATCH_SIZE * 10
            )  # Increased buffer
            sumtree_capacity = max(sumtree_capacity, 1000)  # Absolute minimum
            self.sum_tree = SumTree(sumtree_capacity)
            self.per_alpha = config.PER_ALPHA
            self.per_beta_initial = config.PER_BETA_INITIAL
            self.per_beta_final = config.PER_BETA_FINAL
            self.per_beta_anneal_steps = config.PER_BETA_ANNEAL_STEPS
            self.per_epsilon = config.PER_EPSILON
            logger.info(
                f"MuZero Experience buffer initialized with PER. "
                f"Capacity (total steps): {self.capacity}, Sequence Length: {self.sequence_length}, "
                f"SumTree Capacity (trajectories): {sumtree_capacity}"
            )
        else:
            logger.info(
                f"MuZero Experience buffer initialized with uniform sampling. "
                f"Capacity (total steps): {self.capacity}, Sequence Length: {self.sequence_length}"
            )

    def _get_priority(self, error: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates priority from TD error array using PER parameters."""
        if not self.use_per:
            # Should not be called if PER is disabled, but return default if it is
            return np.ones_like(error)
        # Use np.abs for potential array input
        error_abs = np.abs(error)
        return (error_abs + self.per_epsilon) ** self.per_alpha

    def add(self, trajectory: Trajectory):
        """Adds a complete trajectory to the buffer and SumTree (if PER enabled)."""
        if not trajectory:
            logger.warning("Attempted to add an empty trajectory.")
            return

        traj_len = len(trajectory)
        if traj_len < self.sequence_length:
            logger.debug(
                f"Skipping short trajectory (length {traj_len} < {self.sequence_length})"
            )
            return  # Don't add trajectories shorter than the required sequence length

        buffer_idx = self.next_buffer_idx  # Use unique ID for this trajectory
        self.next_buffer_idx += 1

        # --- Eviction Logic ---
        while self.total_steps + traj_len > self.capacity and len(self.buffer) > 0:
            removed_buffer_idx, removed_traj = self.buffer.popleft()
            self.total_steps -= len(removed_traj)
            if self.use_per and self.sum_tree:
                removed_tree_idx = self.buffer_idx_to_tree_idx.pop(
                    removed_buffer_idx, None
                )
                if removed_tree_idx is not None:
                    # Update priority to 0 in the tree, effectively removing it from sampling
                    # We don't actually remove the node, just nullify its priority
                    self.sum_tree.update(removed_tree_idx, 0.0)
                    # Remove the mapping from tree_idx back to buffer_idx
                    if removed_tree_idx in self.tree_idx_to_buffer_idx:
                        del self.tree_idx_to_buffer_idx[removed_tree_idx]
                else:
                    # This might happen if buffer_idx wasn't added correctly or already evicted
                    logger.warning(
                        f"Could not find tree index for evicted buffer index {removed_buffer_idx}"
                    )
            logger.debug(
                f"Buffer capacity reached. Removed oldest trajectory (BufferIdx: {removed_buffer_idx}, Len: {len(removed_traj)}). Current total steps: {self.total_steps}"
            )
        # ---

        # Add new trajectory
        self.buffer.append((buffer_idx, trajectory))
        self.total_steps += traj_len

        if self.use_per and self.sum_tree:
            # Calculate initial priority (e.g., max priority or based on initial TD error if available)
            # Using max priority ensures new samples are likely to be picked soon
            priority = self.sum_tree.max_priority  # Use the property here
            if (
                priority == 0
            ):  # Handle the case where the tree is empty or only has 0-priority items
                priority = 1.0  # Assign a default high priority for new samples

            # Add the buffer_idx to the SumTree with the calculated priority
            tree_idx = self.sum_tree.add(priority, buffer_idx)
            if tree_idx is not None:
                # Store the mapping between tree_idx and buffer_idx
                self.tree_idx_to_buffer_idx[tree_idx] = buffer_idx
                self.buffer_idx_to_tree_idx[buffer_idx] = tree_idx
            else:
                logger.error(
                    f"SumTree add returned None index for buffer_idx {buffer_idx}. PER might be inconsistent."
                )

        logger.debug(
            f"Added trajectory (BufferIdx: {buffer_idx}, Len: {traj_len}). Buffer trajectories: {len(self.buffer)}, Total steps: {self.total_steps}"
        )

    def add_batch(self, trajectories: list[Trajectory]):
        """Adds a batch of trajectories."""
        for traj in trajectories:
            self.add(traj)

    def _anneal_beta(self, current_train_step: int) -> float:
        """Linearly anneals PER beta."""
        if (
            not self.use_per
            or self.per_beta_anneal_steps is None
            or self.per_beta_anneal_steps <= 0
            or self.config.MAX_TRAINING_STEPS  # Use self.config here
            is None  # Avoid division by zero if MAX_TRAINING_STEPS is None
        ):
            return self.per_beta_initial

        # Ensure anneal_steps doesn't exceed total steps
        anneal_steps = min(
            self.per_beta_anneal_steps,
            self.config.MAX_TRAINING_STEPS,  # Use self.config
        )
        fraction = min(1.0, current_train_step / anneal_steps)
        beta = self.per_beta_initial + fraction * (
            self.per_beta_final - self.per_beta_initial
        )
        return beta

    def sample(
        self,
        batch_size: int,
        current_train_step: int | None = None,
    ) -> SampledBatchPER | SampledBatch | None:
        """
        Samples a batch of sequences. Uses PER if enabled, otherwise uniform.
        Returns SampledBatchPER if PER is used, SampledBatch otherwise, or None if not ready or sampling fails.
        """
        if not self.is_ready():
            logger.debug(
                f"Buffer not ready for sampling. Steps: {self.total_steps}/{self.min_size_to_train}"
            )
            return None

        if self.use_per:
            return self._sample_per(batch_size, current_train_step)
        else:
            return self._sample_uniform(batch_size)

    def _sample_uniform(self, batch_size: int) -> SampledBatch | None:
        """Uniformly samples sequences."""
        sampled_sequences: SampledBatch = []
        attempts = 0
        max_attempts = batch_size * 20  # Increased attempts

        if len(self.buffer) == 0:
            logger.warning("Uniform sample called on empty buffer.")
            return None

        # Create a list of (deque_index, trajectory_length) for trajectories long enough
        eligible_trajectories = [
            (idx, len(traj))
            for idx, (_, traj) in enumerate(self.buffer)
            if len(traj) >= self.sequence_length
        ]

        if not eligible_trajectories:
            logger.warning(
                f"No trajectories long enough ({self.sequence_length}) for uniform sampling."
            )
            return None

        while len(sampled_sequences) < batch_size and attempts < max_attempts:
            attempts += 1
            # Sample a trajectory index uniformly from eligible ones
            traj_deque_idx, traj_len = random.choice(eligible_trajectories)
            _, trajectory = self.buffer[traj_deque_idx]  # Access deque by index

            # Sample a valid start index for the sequence
            start_index = random.randrange(traj_len - self.sequence_length + 1)
            sequence = trajectory[start_index : start_index + self.sequence_length]

            if len(sequence) == self.sequence_length:
                sampled_sequences.append(sequence)
            else:
                # This should not happen if start_index logic is correct
                logger.error(
                    f"Uniform Sample: Sequence incorrect length {len(sequence)} (expected {self.sequence_length}). Traj len: {traj_len}, Start: {start_index}"
                )

        if len(sampled_sequences) < batch_size:
            logger.warning(
                f"Uniform Sample: Could only sample {len(sampled_sequences)} sequences after {attempts} attempts."
            )

        return sampled_sequences if sampled_sequences else None

    def _sample_per(
        self, batch_size: int, current_train_step: int | None
    ) -> SampledBatchPER | None:
        """Samples sequences using Prioritized Experience Replay."""
        if (
            self.sum_tree is None
            or self.sum_tree.n_entries == 0
            or self.sum_tree.total() <= 0  # Use total() method
        ):
            logger.warning(
                f"PER sample called but SumTree empty or total priority zero. "
                f"n_entries: {self.sum_tree.n_entries if self.sum_tree else 'None'}, "
                f"total_priority: {self.sum_tree.total() if self.sum_tree else 'None'}. Cannot sample."
            )
            return None

        if current_train_step is None:
            logger.warning(
                "PER sample requires current_train_step for beta annealing. Using initial beta."
            )
            beta = self.per_beta_initial
        else:
            beta = self._anneal_beta(current_train_step)

        sampled_sequences_list: SampledBatch = []
        # Initialize as lists to append easily
        tree_indices_list: list[int] = []
        priorities_list: list[float] = []
        buffer_indices_sampled_list: list[int] = []  # Store the unique buffer_idx

        segment = self.sum_tree.total() / batch_size  # Use total() method
        attempts = 0
        max_attempts_per_sample = (
            20  # Limit attempts per sample to avoid infinite loops
        )
        sampled_count = 0

        # Create a temporary mapping from buffer_idx to deque index for quick lookup
        # This avoids iterating the deque repeatedly inside the loop
        buffer_idx_to_deque_idx = {
            buf_idx: i for i, (buf_idx, _) in enumerate(self.buffer)
        }

        while (
            sampled_count < batch_size
            and attempts < max_attempts_per_sample * batch_size
        ):
            attempts += 1
            a = segment * sampled_count
            b = segment * (sampled_count + 1)
            # Ensure b doesn't exceed total priority due to floating point issues
            b = min(b, self.sum_tree.total())  # Use total() method
            # Ensure a < b even with floating point issues
            if a >= b:
                if (
                    self.sum_tree.total() > 1e-9
                ):  # Avoid division by zero if total_priority is tiny
                    a = b - (self.sum_tree.total() * 1e-6)  # Sample very close to b
                else:
                    a = 0.0  # Sample from the beginning if total priority is ~0

            value = random.uniform(a, b)

            try:
                # get_leaf returns (tree_idx, priority, buffer_idx)
                tree_idx, priority, buffer_idx = self.sum_tree.get_leaf(value)
            except (IndexError, ValueError, RuntimeError) as e:
                logger.warning(
                    f"PER sample: SumTree get_leaf failed for value {value}. Error: {e}. Retrying."
                )
                continue

            if not isinstance(buffer_idx, int):
                logger.warning(
                    f"PER sample: SumTree returned invalid buffer_idx {buffer_idx} (type: {type(buffer_idx)}). Retrying."
                )
                continue

            # Check if we already sampled this trajectory in this batch (optional, but can improve diversity)
            # if buffer_idx in buffer_indices_sampled_list:
            #     continue

            # Find the trajectory in the deque using the buffer_idx
            deque_idx = buffer_idx_to_deque_idx.get(buffer_idx)
            if deque_idx is None:
                logger.error(
                    f"PER sample: Trajectory for buffer_idx {buffer_idx} (TreeIdx: {tree_idx}) not found in deque map. SumTree/Deque inconsistent! Setting priority to 0."
                )
                # Attempt to recover by removing the bad index from the tree
                self.sum_tree.update(tree_idx, 0.0)
                if tree_idx in self.tree_idx_to_buffer_idx:
                    del self.tree_idx_to_buffer_idx[tree_idx]
                if buffer_idx in self.buffer_idx_to_tree_idx:
                    del self.buffer_idx_to_tree_idx[buffer_idx]
                continue

            _, trajectory = self.buffer[deque_idx]

            if len(trajectory) < self.sequence_length:
                logger.debug(
                    f"PER Sample: Trajectory {buffer_idx} too short ({len(trajectory)} < {self.sequence_length}). Skipping."
                )
                # Optionally reduce priority of short trajectories?
                # self.sum_tree.update(tree_idx, self.per_epsilon ** self.per_alpha)
                continue

            # Sample a valid start index for the sequence
            start_index = random.randrange(len(trajectory) - self.sequence_length + 1)
            sequence = trajectory[start_index : start_index + self.sequence_length]

            if len(sequence) == self.sequence_length:
                sampled_sequences_list.append(sequence)
                tree_indices_list.append(tree_idx)
                priorities_list.append(priority)
                buffer_indices_sampled_list.append(buffer_idx)  # Store the unique ID
                sampled_count += 1
            else:
                logger.error(
                    f"PER Sample: Sequence incorrect length {len(sequence)} (expected {self.sequence_length}). BufferIdx: {buffer_idx}, TreeIdx: {tree_idx}"
                )

        if sampled_count == 0:
            logger.warning(
                f"PER Sample: Could not sample any valid sequences after {attempts} attempts."
            )
            return None
        if sampled_count < batch_size:
            logger.warning(
                f"PER Sample: Could only sample {sampled_count} sequences out of {batch_size} requested after {attempts} attempts."
            )
            # Trim lists to the actual number sampled
            sampled_sequences_list = sampled_sequences_list[:sampled_count]
            tree_indices_list = tree_indices_list[:sampled_count]
            priorities_list = priorities_list[:sampled_count]
            # buffer_indices_sampled_list = buffer_indices_sampled_list[:sampled_count] # Not used later, but good practice

        # Convert lists to numpy arrays
        # --- FIXED: Use typed arrays directly ---
        tree_indices: npt.NDArray[np.int32] = np.array(
            tree_indices_list, dtype=np.int32
        )
        priorities_np: npt.NDArray[np.float32] = np.array(
            priorities_list, dtype=np.float32
        )
        # --- END FIXED ---

        # Calculate IS weights
        sampling_probabilities = priorities_np / max(
            self.sum_tree.total(),
            1e-9,  # Use total() method
        )  # Avoid division by zero
        weights = np.power(
            max(self.sum_tree.n_entries, 1) * sampling_probabilities + 1e-9, -beta
        )
        max_weight = np.max(weights) if len(weights) > 0 else 1.0
        weights /= max(max_weight, 1e-9)  # Avoid division by zero

        return SampledBatchPER(
            sequences=sampled_sequences_list,
            indices=tree_indices,  # Use typed array
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Updates priorities of sampled experiences based on TD errors."""
        if not self.use_per or self.sum_tree is None:
            return
        if len(tree_indices) != len(td_errors):
            logger.error(
                f"PER update failed: Mismatch length indices ({len(tree_indices)}) vs errors ({len(td_errors)})"
            )
            return

        # Ensure td_errors is a numpy array for vectorized operations
        td_errors_np = np.asarray(td_errors)
        # Calculate priorities using the internal method
        priorities_array: npt.NDArray[np.float64] = self._get_priority(td_errors_np)

        if len(priorities_array) != len(tree_indices):
            logger.error(
                f"PER update failed: Mismatch length indices ({len(tree_indices)}) vs calculated priorities ({len(priorities_array)})"
            )
            return

        for i in range(len(tree_indices)):
            idx = int(tree_indices[i])  # Cast numpy int to python int
            p = float(priorities_array[i])  # Cast numpy float to python float
            # Check index validity before updating
            if not (0 <= idx < len(self.sum_tree.tree)):
                logger.warning(
                    f"PER update: Invalid tree index {idx} provided. Skipping update for this index."
                )
                continue
            try:
                self.sum_tree.update(idx, p)
            except IndexError:
                # This might happen if the tree structure is somehow corrupted
                logger.error(
                    f"PER update: Error updating tree index {idx} with priority {p}. Skipping.",
                    exc_info=True,
                )

    def __len__(self) -> int:
        """Returns the total number of steps (transitions) stored in the buffer."""
        return self.total_steps

    def is_ready(self) -> bool:
        """Checks if the buffer has enough total steps and trajectories to start training."""
        sufficient_steps = self.total_steps >= self.min_size_to_train
        sufficient_trajectories = True
        if self.use_per:
            if self.sum_tree is None:
                sufficient_trajectories = False  # Cannot sample if tree is missing
            else:
                # Need enough trajectories in the tree to form at least one batch
                sufficient_trajectories = (
                    self.sum_tree.n_entries >= self.config.BATCH_SIZE
                )
        return sufficient_steps and sufficient_trajectories
