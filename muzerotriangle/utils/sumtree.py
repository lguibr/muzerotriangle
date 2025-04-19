# File: muzerotriangle/utils/sumtree.py
import logging
import sys

import numpy as np

# Use a dedicated logger for SumTree internal debugging
sumtree_logger = logging.getLogger("muzerotriangle.utils.sumtree_internal")
sumtree_logger.setLevel(logging.WARNING)  # Default level
if not sumtree_logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    sumtree_logger.addHandler(handler)


class SumTree:
    """
    Simple SumTree implementation for efficient prioritized sampling.
    Stores priorities and allows sampling proportional to priority.
    Handles circular buffer logic for data storage using a Python list.
    Uses internal capacity padding to the next power of 2 for simplified tree structure.
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

        # User-facing capacity
        self.capacity = capacity
        # Internal capacity (power of 2) for the tree structure
        self._internal_capacity = 1
        while self._internal_capacity < capacity:
            self._internal_capacity *= 2

        # Tree size based on internal capacity
        self.tree = np.zeros(2 * self._internal_capacity - 1)
        # Data storage size based on internal capacity (though only user capacity is used)
        self.data: list[object | None] = [None] * self._internal_capacity
        # data_pointer points to the next index to write to in self.data (wraps around user capacity)
        self.data_pointer = 0
        # n_entries tracks the number of valid entries (up to user capacity)
        self.n_entries = 0
        # _max_priority tracks the maximum priority ever added/updated
        self._max_priority = 0.0
        sumtree_logger.debug(
            f"SumTree initialized with user_capacity={capacity}, internal_capacity={self._internal_capacity}"
        )

    def reset(self):
        """Resets the tree and data."""
        self.tree.fill(0.0)
        # Recreate the data list based on internal capacity
        self.data = [None] * self._internal_capacity
        self.data_pointer = 0
        self.n_entries = 0
        self._max_priority = 0.0
        sumtree_logger.debug("SumTree reset.")

    def _propagate(self, tree_idx: int, change: float):
        """Propagates priority change up the tree."""
        parent = (tree_idx - 1) // 2
        if parent < 0:
            return
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _update_leaf(self, tree_idx: int, priority: float):
        """Updates a leaf node and propagates the change."""
        if not (
            self._internal_capacity - 1 <= tree_idx < 2 * self._internal_capacity - 1
        ):
            msg = f"Invalid tree_idx {tree_idx} for leaf update. InternalCapacity={self._internal_capacity}"
            sumtree_logger.error(msg)
            raise IndexError(msg)

        if priority < 0 or not np.isfinite(priority):
            priority = 0.0

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self._max_priority = max(self._max_priority, priority)

    def add(self, priority: float, data: object) -> int:
        """Adds data with a given priority. Returns the tree index."""
        if self.capacity == 0:  # Check user capacity
            raise ValueError("Cannot add to a SumTree with zero capacity.")

        # Calculate tree index based on data_pointer and internal capacity
        tree_idx = self.data_pointer + self._internal_capacity - 1

        # Store data at data_pointer index
        self.data[self.data_pointer] = data
        # Update the corresponding leaf in the tree
        self.update(tree_idx, priority)

        # Update data_pointer, wrapping around the *user* capacity
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        # Increment n_entries up to the *user* capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

        return tree_idx

    def update(self, tree_idx: int, priority: float):
        """Public method to update priority at a given tree index."""
        self._update_leaf(tree_idx, priority)

    # --- CORRECTED Iterative Retrieve with Strict Inequality ---
    def _retrieve(self, tree_idx: int, sample_value: float) -> int:
        """Finds the leaf index for a given sample value using binary search on the tree."""
        current_idx = tree_idx  # Start search from the provided index (usually 0)
        sumtree_logger.debug(
            f"Retrieve START: initial_idx={current_idx}, sample_value={sample_value:.6f}"
        )

        while True:
            left_child_idx = 2 * current_idx + 1
            right_child_idx = left_child_idx + 1
            sumtree_logger.debug(
                f"  Loop: current_idx={current_idx}, sample_value={sample_value:.6f}"
            )

            # If left child index is out of bounds, current_idx is a leaf
            if left_child_idx >= len(self.tree):
                sumtree_logger.debug(
                    f"  Leaf condition met: left_child_idx={left_child_idx} >= tree_len={len(self.tree)}. Returning leaf_idx={current_idx}"
                )
                break

            left_sum = self.tree[left_child_idx]
            sumtree_logger.debug(
                f"    left_child_idx={left_child_idx}, left_sum={left_sum:.6f}"
            )

            # --- Use strict less than comparison ---
            if sample_value < left_sum:
                # --- End change ---
                sumtree_logger.debug(
                    f"    Condition TRUE: {sample_value:.6f} < {left_sum:.6f}. Going LEFT."
                )
                current_idx = left_child_idx
            else:
                sumtree_logger.debug(
                    f"    Condition FALSE: {sample_value:.6f} >= {left_sum:.6f}. Going RIGHT."
                )
                sample_value -= left_sum
                sumtree_logger.debug(f"      Adjusted sample_value={sample_value:.6f}")
                # Ensure right child exists before assigning
                if right_child_idx >= len(self.tree):
                    sumtree_logger.warning(
                        f"      Right child index {right_child_idx} out of bounds! Tree len={len(self.tree)}. Breaking loop at idx={current_idx}."
                    )
                    break
                current_idx = right_child_idx
                sumtree_logger.debug(f"      New current_idx={current_idx}")

        sumtree_logger.debug(f"Retrieve END: Returning leaf_idx={current_idx}")
        return current_idx

    # --- End CORRECTED Iterative Retrieve ---

    def get_leaf(self, value: float) -> tuple[int, float, object]:
        """
        Finds the leaf node index, priority, and associated data for a given sample value.
        """
        total_p = self.total()
        if total_p <= 0:
            raise ValueError(
                f"Cannot sample from SumTree with zero or negative total priority ({total_p}). n_entries: {self.n_entries}"
            )

        # Clamp value to be within [0, total_p) using epsilon
        value = np.clip(value, 0, total_p - 1e-9)

        # Start retrieval from the root (index 0)
        leaf_tree_idx = self._retrieve(0, value)

        # Ensure returned index is actually a leaf index based on internal capacity
        if not (
            self._internal_capacity - 1
            <= leaf_tree_idx
            < 2 * self._internal_capacity - 1
        ):
            sumtree_logger.error(
                f"GetLeaf: _retrieve returned non-leaf index {leaf_tree_idx}. "
                f"InternalCapacity={self._internal_capacity}, Sampled value: {value:.4f}, Total P: {total_p:.4f}."
            )
            # Fallback: Find the leftmost leaf based on internal capacity
            leaf_tree_idx = self._internal_capacity - 1

        data_idx = leaf_tree_idx - (self._internal_capacity - 1)

        # Check if the data index corresponds to a valid *entry* (within user capacity and n_entries)
        if not (0 <= data_idx < self.n_entries):
            tree_dump = self.tree[
                self._internal_capacity - 1 : self._internal_capacity
                - 1
                + self.n_entries
            ]
            sumtree_logger.error(
                f"GetLeaf: Invalid data_idx {data_idx} (from leaf_tree_idx {leaf_tree_idx}) retrieved. "
                f"n_entries={self.n_entries}, user_capacity={self.capacity}. "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}. "
                f"Leaf priorities (first {self.n_entries}): {tree_dump}"
            )
            raise IndexError(
                f"Retrieved data_idx {data_idx} is out of bounds for n_entries {self.n_entries}."
            )

        priority = self.tree[leaf_tree_idx]
        # Retrieve data using the calculated data_idx (which is within [0, user_capacity))
        data = self.data[data_idx]

        if data is None:
            # This should ideally not happen if data_idx < n_entries check passed
            tree_dump = self.tree[
                self._internal_capacity - 1 : self._internal_capacity
                - 1
                + self.n_entries
            ]
            sumtree_logger.error(
                f"Sampled None data at data_idx {data_idx} (tree_idx {leaf_tree_idx}). "
                f"Sampled value: {value:.4f}, Total P: {total_p:.4f}, "
                f"n_entries: {self.n_entries}, data_pointer: {self.data_pointer}. "
                f"Leaf priorities (first {self.n_entries}): {tree_dump}"
            )
            raise RuntimeError(
                f"Sampled None data at data_idx {data_idx} (tree_idx {leaf_tree_idx}). "
            )

        return leaf_tree_idx, priority, data

    def total(self) -> float:
        """Returns the total priority (root node value)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns the maximum priority seen so far, or 1.0 if empty."""
        return float(self._max_priority) if self.n_entries > 0 else 1.0
