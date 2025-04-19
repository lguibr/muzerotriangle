# File: muzerotriangle/utils/sumtree.py
import logging
import sys
import numpy as np

# Use a dedicated logger for SumTree internal debugging
sumtree_logger = logging.getLogger("muzerotriangle.utils.sumtree_internal")
# Keep logging level higher for less noise during normal runs
# Set to DEBUG locally if needed for deep dives
sumtree_logger.setLevel(logging.WARNING)
if not sumtree_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    sumtree_logger.addHandler(handler)


class SumTree:
    """
    Simple SumTree implementation for efficient prioritized sampling.
    Stores priorities and allows sampling proportional to priority.
    Does NOT handle the calculation of priority from error (e.g., (error+eps)^alpha).
    Handles circular buffer logic for data storage using a Python list.
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        # Tree size is 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Data storage: Use a Python list, initialized with None
        self.data: list[object | None] = [None] * capacity
        # data_pointer points to the next index to write to in self.data
        self.data_pointer = 0
        # n_entries tracks the number of valid entries (up to capacity)
        self.n_entries = 0
        # _max_priority tracks the maximum priority ever added/updated
        self._max_priority = 0.0  # Initialize to 0.0
        sumtree_logger.debug(f"SumTree initialized with capacity {capacity}")

    def reset(self):
        """Resets the tree and data."""
        self.tree.fill(0.0)
        # Recreate the list
        self.data = [None] * self.capacity
        self.data_pointer = 0
        self.n_entries = 0
        self._max_priority = 0.0  # Reset to 0.0
        sumtree_logger.debug("SumTree reset.")

    def _propagate(self, tree_idx: int, change: float):
        """Propagates priority change up the tree."""
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        # Stop propagation when root is reached
        if parent != 0:
            self._propagate(parent, change)

    def _update_leaf(self, tree_idx: int, priority: float):
        """Updates a leaf node and propagates the change."""
        if not (self.capacity - 1 <= tree_idx < 2 * self.capacity - 1):
            msg = (
                f"Invalid tree_idx {tree_idx} for leaf update. Capacity={self.capacity}"
            )
            sumtree_logger.error(msg)
            raise IndexError(msg)

        if priority < 0:
            sumtree_logger.warning(
                f"Attempted to update with negative priority {priority}. Clamping to 0."
            )
            priority = 0.0
        elif not np.isfinite(priority):
            sumtree_logger.warning(
                f"Attempted to update with non-finite priority {priority}. Clamping to 0."
            )
            priority = 0.0

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate change upwards starting from the updated leaf index
        # No need to check for tree_idx == 0 because leaf indices start at capacity - 1
        self._propagate(tree_idx, change)
        # Update max priority seen so far (compare only with new priority)
        self._max_priority = max(self._max_priority, priority)

    def add(self, priority: float, data: object) -> int:
        """Adds data with a given priority. Returns the tree index."""
        if self.capacity == 0:
            raise ValueError("Cannot add to a SumTree with zero capacity.")

        tree_idx = self.data_pointer + self.capacity - 1
        sumtree_logger.debug(
            f"Add START: prio={priority:.4f}, data_ptr={self.data_pointer}, n_entries={self.n_entries}, capacity={self.capacity}"
        )
        sumtree_logger.debug(f"Add: Calculated tree_idx={tree_idx}")

        self.data[self.data_pointer] = data
        sumtree_logger.debug(f"Add: Stored data at data_idx={self.data_pointer}")

        self.update(
            tree_idx, priority
        )  # Use update which handles propagation and max_priority
        sumtree_logger.debug(
            f"Add: Updated leaf {tree_idx} with priority {priority:.4f}, _max_p={self._max_priority:.4f}"
        )

        # Update data_pointer and n_entries
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        # Only increment n_entries if we haven't filled the buffer yet
        if self.n_entries < self.capacity:
            self.n_entries += 1
            sumtree_logger.debug(f"Add: Incremented n_entries to {self.n_entries}")
        # else: # No need for else, data_pointer wrap handles overwrite
        #      sumtree_logger.debug(
        #         f"Add: n_entries ({self.n_entries}) remains at capacity"
        #     )

        sumtree_logger.debug(
            f"Add END: data_ptr={self.data_pointer}, n_entries={self.n_entries}"
        )
        return tree_idx

    def update(self, tree_idx: int, priority: float):
        """Public method to update priority at a given tree index."""
        self._update_leaf(tree_idx, priority)

    # --- CORRECTED Iterative _retrieve ---
    def _retrieve(self, tree_idx: int, sample_value: float) -> int:
        """Finds the leaf index for a given sample value using binary search on the tree."""
        # Start from the root (tree_idx = 0)
        current_idx = 0
        while True:
            left_child_idx = 2 * current_idx + 1
            right_child_idx = left_child_idx + 1

            # If left child index is out of bounds, we are at a leaf node
            if left_child_idx >= len(self.tree):
                return current_idx  # This is the leaf index

            left_sum = self.tree[left_child_idx]

            # If the sample value is less than the priority of the left subtree, go left
            # Use strict inequality to handle zero-priority nodes correctly
            if sample_value < left_sum:
                current_idx = left_child_idx
            # Otherwise, go right, adjusting the sample value
            else:
                sample_value -= left_sum
                current_idx = right_child_idx

    # --- END CORRECTED ---

    def get_leaf(self, value: float) -> tuple[int, float, object]:
        """
        Finds the leaf node index, priority, and associated data for a given sample value.
        """
        sumtree_logger.debug(f"GetLeaf START: value={value:.4f}")
        total_p = self.total()  # Use the method here
        if (
            total_p <= 0
        ):  # Check for <= 0 to handle potential negative priorities if clamping fails
            raise ValueError(
                f"Cannot sample from SumTree with zero or negative total priority ({total_p}). n_entries: {self.n_entries}"
            )
        sumtree_logger.debug(f"GetLeaf: TotalP={total_p:.4f}")

        # Clamp value to be within [0, total_p)
        value = np.clip(
            value, 0, total_p - 1e-9
        )  # Use epsilon to avoid hitting exact total_p

        sumtree_logger.debug(f"GetLeaf: Using value={value:.4f} for retrieval")

        leaf_tree_idx = self._retrieve(0, value)
        sumtree_logger.debug(f"GetLeaf: Retrieved leaf_tree_idx={leaf_tree_idx}")
        data_idx = leaf_tree_idx - (self.capacity - 1)
        sumtree_logger.debug(f"GetLeaf: Calculated data_idx={data_idx}")

        # Check if the data index is valid given the number of entries
        if not (0 <= data_idx < self.n_entries):
            # This can happen if the tree is not full and the sampling process
            # lands in a region corresponding to an empty slot.
            # This indicates an issue either in sampling logic or tree structure.
            tree_dump = self.tree[
                self.capacity - 1 : self.capacity - 1 + self.n_entries
            ]
            sumtree_logger.error(
                f"GetLeaf: Invalid data_idx {data_idx} retrieved for tree_idx {leaf_tree_idx}. "
                f"n_entries={self.n_entries}, capacity={self.capacity}. "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}. "
                f"Leaf priorities (first {self.n_entries}): {tree_dump}"
            )
            # Re-running retrieve might help if it was a transient float issue, but unlikely.
            # A better approach might be to resample, but that logic belongs in the caller (ExperienceBuffer).
            # For now, raise an error to indicate the problem clearly.
            raise IndexError(
                f"Retrieved data_idx {data_idx} is out of bounds for n_entries {self.n_entries}."
            )

        priority = self.tree[leaf_tree_idx]
        data = self.data[data_idx]  # Retrieve from list
        sumtree_logger.debug(f"GetLeaf: Found priority={priority:.4f}, data={data}")

        # Check for None data, which indicates an uninitialized slot was sampled
        # This check should ideally be redundant now with the n_entries check above.
        if data is None:
            tree_dump = self.tree[
                self.capacity - 1 : self.capacity - 1 + self.n_entries
            ]
            sumtree_logger.error(
                f"Sampled None data at data_idx {data_idx} (tree_idx {leaf_tree_idx}). "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}, "
                f"n_entries: {self.n_entries}, data_pointer: {self.data_pointer}. "
                f"Leaf priorities (first {self.n_entries}): {tree_dump}"
            )
            raise RuntimeError(
                f"Sampled None data at data_idx {data_idx} (tree_idx {leaf_tree_idx}). "
                f"This indicates an issue with the SumTree state or sampling logic. "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}"
            )

        return leaf_tree_idx, priority, data

    def total(self) -> float:
        """Returns the total priority (root node value)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns the maximum priority seen so far, or 1.0 if empty."""
        # Return 1.0 if empty, ensuring new items get added with non-zero priority
        # If not empty, return the actual max seen.
        return float(self._max_priority) if self.n_entries > 0 else 1.0
