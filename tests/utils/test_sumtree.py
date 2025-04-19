# File: tests/utils/test_sumtree.py
import logging
import random
from collections import Counter

import numpy as np
import pytest

from muzerotriangle.utils.sumtree import SumTree

# Configure logging for tests to see debug messages from SumTree
logging.basicConfig(
    level=logging.INFO
)  # Set to INFO for less verbose output during tests
sumtree_logger = logging.getLogger("muzerotriangle.utils.sumtree_internal")
sumtree_logger.setLevel(logging.WARNING)  # Keep SumTree logs quiet unless warning/error


def dump_sumtree_state(tree: SumTree, test_name: str):
    """Helper function to print SumTree state for debugging."""
    print(f"\n--- SumTree State Dump ({test_name}) ---")
    print(f"  Capacity: {tree.capacity}")
    print(f"  n_entries: {tree.n_entries}")
    print(f"  data_pointer: {tree.data_pointer}")
    print(f"  _max_priority: {tree._max_priority:.4f}")
    print(f"  total_priority (root): {tree.total():.4f}")  # Use total() method
    # Only print the relevant part of the tree array
    tree_size = 2 * tree.capacity - 1
    print(
        f"  Tree array (size {len(tree.tree)}, showing up to {tree_size}): {tree.tree[:tree_size]}"
    )
    # Print only the populated part of the data array
    print(
        f"  Data array (size {len(tree.data)}, showing up to {tree.n_entries}): {tree.data[:tree.n_entries]}"
    )
    print(f"--- End Dump ---")


@pytest.fixture
def sum_tree_cap5() -> SumTree:
    """Provides a SumTree instance with capacity 5."""
    return SumTree(capacity=5)


def test_sumtree_init():
    tree = SumTree(capacity=10)
    assert tree.capacity == 10
    assert len(tree.tree) == 19  # 2*10 - 1
    assert len(tree.data) == 10
    assert tree.data_pointer == 0
    assert tree.n_entries == 0
    assert tree.total() == 0.0  # Use total() method
    assert tree.max_priority == 1.0  # Default max priority when empty


def test_sumtree_add_single(sum_tree_cap5: SumTree):
    tree_idx = sum_tree_cap5.add(0.5, "data1")
    assert tree_idx == 4  # Index of the first leaf in a tree of capacity 5
    assert sum_tree_cap5.n_entries == 1
    assert sum_tree_cap5.data_pointer == 1
    assert sum_tree_cap5.total() == pytest.approx(0.5)  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(0.5)
    assert sum_tree_cap5.tree[tree_idx] == 0.5
    assert sum_tree_cap5.data[0] == "data1"


def test_sumtree_add_multiple(sum_tree_cap5: SumTree):
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    data = ["d0", "d1", "d2", "d3", "d4"]
    expected_max_priority = 0.0  # Start with 0.0
    for i, (p, d) in enumerate(zip(priorities, data)):
        sum_tree_cap5.add(p, d)
        expected_max_priority = max(expected_max_priority, p)
        assert sum_tree_cap5.n_entries == i + 1
        assert sum_tree_cap5.data_pointer == (i + 1) % 5
        assert sum_tree_cap5.total() == pytest.approx(
            sum(priorities[: i + 1])
        )  # Use total() method
        assert sum_tree_cap5.max_priority == pytest.approx(expected_max_priority)

    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 0
    assert sum_tree_cap5.data == data


def test_sumtree_add_overflow(sum_tree_cap5: SumTree):
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    data = ["d0", "d1", "d2", "d3", "d4"]
    for p, d in zip(priorities, data):
        sum_tree_cap5.add(p, d)

    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.total() == pytest.approx(sum(priorities))  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(0.8)

    # Add one more, overwriting the first element (data_idx 0, tree_idx 4)
    sum_tree_cap5.add(1.0, "d5")
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 1
    assert sum_tree_cap5.data[0] == "d5"
    assert sum_tree_cap5.data[1] == "d1"  # Unchanged
    assert sum_tree_cap5.total() == pytest.approx(
        sum(priorities[1:]) + 1.0
    )  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[4] == 1.0  # Check leaf node updated

    # Add another, overwriting the second element (data_idx 1, tree_idx 5)
    sum_tree_cap5.add(0.05, "d6")
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 2
    assert sum_tree_cap5.data[0] == "d5"
    assert sum_tree_cap5.data[1] == "d6"
    assert sum_tree_cap5.total() == pytest.approx(
        1.0 + 0.05 + 0.2 + 0.8 + 0.4
    )  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(
        1.0
    )  # Max priority doesn't decrease
    assert sum_tree_cap5.tree[5] == 0.05  # Check leaf node updated


def test_sumtree_update(sum_tree_cap5: SumTree):
    tree_idx_0 = sum_tree_cap5.add(0.5, "data0")
    tree_idx_1 = sum_tree_cap5.add(0.3, "data1")
    assert sum_tree_cap5.total() == pytest.approx(0.8)  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(0.5)

    sum_tree_cap5.update(tree_idx_0, 1.0)
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.3)  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[tree_idx_0] == pytest.approx(1.0)

    sum_tree_cap5.update(tree_idx_1, 0.1)
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.1)  # Use total() method
    assert sum_tree_cap5.max_priority == pytest.approx(
        1.0
    )  # Max priority doesn't decrease
    assert sum_tree_cap5.tree[tree_idx_1] == pytest.approx(0.1)


def test_sumtree_retrieve(sum_tree_cap5: SumTree):
    """Test the _retrieve method directly."""
    # Tree structure for capacity 5:
    #       0
    #     /   \
    #    1     2
    #   / \   / \
    #  3   4 5   6   <- This level doesn't exist for capacity 5
    # Indices: 0=root, 1=left, 2=right, 3=left-left, 4=left-right, 5=right-left, 6=right-right
    # For capacity 5, tree size is 9. Leaves are indices 4, 5, 6, 7, 8.
    # Data indices: 0, 1, 2, 3, 4

    # Add items
    data_map = {}
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    for i, p in enumerate(priorities):
        data_id = f"i{i}"
        tree_idx = sum_tree_cap5.add(p, data_id)
        data_map[tree_idx] = data_id

    print(f"\n[test_retrieve] Tree state: {sum_tree_cap5.tree}")
    print(f"[test_retrieve] Data state: {sum_tree_cap5.data}")
    print(f"[test_retrieve] Data map (tree_idx -> data_id): {data_map}")

    # Expected tree: [2.0, 1.3, 0.7, 1.2, 0.1, 0.5, 0.2, 0.8, 0.4]
    # Leaf indices:  4    5    6    7    8
    # Data indices:  0    1    2    3    4
    # Priorities:    0.1  0.5  0.2  0.8  0.4
    # Data:         i0   i1   i2   i3   i4
    # Cumulative sums for leaves: [0.1, 0.6, 0.8, 1.6, 2.0]

    # Test retrieval based on sample values
    test_cases = {
        0.05: 4,  # Should fall in the first bucket (index 4, priority 0.1)
        0.1: 5,  # Should fall in the second bucket (index 5, priority 0.5)
        0.15: 5,
        0.6: 6,  # Should fall in the third bucket (index 6, priority 0.2)
        0.7: 6,
        0.8: 7,  # Should fall in the fourth bucket (index 7, priority 0.8)
        1.5: 7,
        1.6: 8,  # Should fall in the fifth bucket (index 8, priority 0.4)
        1.99: 8,
    }

    for sample_value, expected_tree_idx in test_cases.items():
        print(f"[test_retrieve] Testing sample {sample_value:.4f}")
        retrieved_tree_idx = sum_tree_cap5._retrieve(0, sample_value)
        assert (
            retrieved_tree_idx == expected_tree_idx
        ), f"Sample {sample_value:.4f} -> Expected {expected_tree_idx}, Got {retrieved_tree_idx}"


def test_sumtree_retrieve_with_zeros(sum_tree_cap5: SumTree):
    """Test _retrieve with zero priority nodes."""
    # Add items
    data_map = {}
    priorities = [0.0, 0.4, 0.6, 0.0, 0.0]
    data = ["z0", "iA", "iB", "z1", "z2"]
    for i, p in enumerate(priorities):
        data_id = data[i]
        tree_idx = sum_tree_cap5.add(p, data_id)
        data_map[tree_idx] = data_id

    print(f"\n[test_retrieve_zeros] Tree state: {sum_tree_cap5.tree}")
    print(f"[test_retrieve_zeros] Data state: {sum_tree_cap5.data}")
    print(f"[test_retrieve_zeros] Data map (tree_idx -> data_id): {data_map}")

    # Expected tree: [1.0, 0.4, 0.6, 0.0, 0.0, 0.4, 0.6, 0.0, 0.0]
    # Leaf indices:  4    5    6    7    8
    # Data indices:  0    1    2    3    4
    # Priorities:    0.0  0.4  0.6  0.0  0.0
    # Data:         z0   iA   iB   z1   z2
    # Cumulative sums for leaves: [0.0, 0.4, 1.0, 1.0, 1.0]

    # Test retrieval based on sample values
    test_cases = {
        0.0: 5,  # Should skip zero-priority leaf 4 and land in leaf 5
        0.1: 5,  # Should land in leaf 5
        0.3: 5,  # Should land in leaf 5
        0.399: 5,  # Should land in leaf 5
        0.4: 6,  # Should land in leaf 6
        0.5: 6,
        0.99: 6,  # Should land in leaf 6
    }

    for sample_value, expected_tree_idx in test_cases.items():
        print(f"[test_retrieve_zeros] Testing sample {sample_value:.4f}")
        retrieved_tree_idx = sum_tree_cap5._retrieve(0, sample_value)
        assert (
            retrieved_tree_idx == expected_tree_idx
        ), f"Sample {sample_value:.4f} -> Expected {expected_tree_idx}, Got {retrieved_tree_idx}"


def test_sumtree_get_leaf_edge_cases(sum_tree_cap5: SumTree):
    """Test edge cases for get_leaf."""
    # Empty tree
    # --- UPDATED REGEX ---
    with pytest.raises(
        ValueError,
        match="Cannot sample from SumTree with zero or negative total priority",
    ):
        sum_tree_cap5.get_leaf(0.5)
    # --- END UPDATED REGEX ---

    # Single item
    tree_idx_0 = sum_tree_cap5.add(1.0, "only_item")
    assert sum_tree_cap5.n_entries == 1
    assert sum_tree_cap5.total() == pytest.approx(1.0)  # Use total() method
    print(f"\n[test_edge_cases] Single item tree: {sum_tree_cap5.tree}")
    # Test sampling at 0.0
    print("[test_edge_cases] Testing get_leaf(0.0)")
    idx0, p0, d0 = sum_tree_cap5.get_leaf(0.0)
    assert d0 == "only_item"
    assert p0 == pytest.approx(1.0)
    assert idx0 == tree_idx_0

    # Test sampling close to total priority
    print("[test_edge_cases] Testing get_leaf(1.0 - eps)")
    idx1, p1, d1 = sum_tree_cap5.get_leaf(1.0 - 1e-9)
    assert d1 == "only_item"
    assert p1 == pytest.approx(1.0)
    assert idx1 == tree_idx_0

    # Test sampling exactly at total priority (should be clipped by get_leaf)
    print("[test_edge_cases] Testing get_leaf(1.0)")
    idx_exact, p_exact, d_exact = sum_tree_cap5.get_leaf(1.0)
    assert d_exact == "only_item"
    assert idx_exact == tree_idx_0

    # Test sampling above total priority (should be clipped by get_leaf)
    print("[test_edge_cases] Testing get_leaf(1.1)")
    idx_above, p_above, d_above = sum_tree_cap5.get_leaf(1.1)
    assert d_above == "only_item"
    assert idx_above == tree_idx_0

    # Test sampling below zero (should be clipped by get_leaf)
    print("[test_edge_cases] Testing get_leaf(-0.1)")
    idx_below, p_below, d_below = sum_tree_cap5.get_leaf(-0.1)
    assert d_below == "only_item"
    assert idx_below == tree_idx_0

    # Zero priority item
    sum_tree_cap5.reset()
    tree_idx_z0 = sum_tree_cap5.add(0.0, "z0")  # data_idx 0, tree_idx 4
    tree_idx_iA = sum_tree_cap5.add(0.4, "itemA")  # data_idx 1, tree_idx 5
    tree_idx_iB = sum_tree_cap5.add(0.6, "itemB")  # data_idx 2, tree_idx 6
    tree_idx_z1 = sum_tree_cap5.add(0.0, "z1")  # data_idx 3, tree_idx 7
    tree_idx_z2 = sum_tree_cap5.add(0.0, "z2")  # data_idx 4, tree_idx 8
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.total() == pytest.approx(1.0)  # Use total() method
    print(f"[test_edge_cases] Zero priority tree: {sum_tree_cap5.tree}")
    print(f"[test_edge_cases] Data: {sum_tree_cap5.data}")

    # Sampling value < 0.4 should yield 'itemA' (index 5)
    print("[test_edge_cases] Testing get_leaf(0.3) with zero priority item")
    idx, p, d = sum_tree_cap5.get_leaf(0.3)
    assert d == "itemA"
    assert p == pytest.approx(0.4)
    assert idx == tree_idx_iA

    # Sampling value >= 0.4 and < 1.0 should yield 'itemB' (index 6)
    print("[test_edge_cases] Testing get_leaf(0.5) with zero priority item")
    idx, p, d = sum_tree_cap5.get_leaf(0.5)
    assert d == "itemB"
    assert p == pytest.approx(0.6)
    assert idx == tree_idx_iB

    # Test sampling exactly at boundary (0.4)
    print("[test_edge_cases] Testing get_leaf(0.4) boundary")
    idx_b, p_b, d_b = sum_tree_cap5.get_leaf(0.4)
    assert d_b == "itemB"
    assert idx_b == tree_idx_iB

    # Test sampling at 0.0
    print("[test_edge_cases] Testing get_leaf(0.0) boundary")
    idx_0, p_0, d_0 = sum_tree_cap5.get_leaf(0.0)
    assert d_0 == "itemA"  # Should pick the first non-zero element
    assert idx_0 == tree_idx_iA

    # Test updating a zero-priority item
    sum_tree_cap5.update(tree_idx_z0, 0.1)  # Update "zero_item" priority
    assert sum_tree_cap5.total() == pytest.approx(1.1)  # Use total() method
    print(f"[test_edge_cases] After update tree: {sum_tree_cap5.tree}")
    print("[test_edge_cases] Testing get_leaf(0.05) after update")
    idx_up, p_up, d_up = sum_tree_cap5.get_leaf(0.05)
    assert d_up == "z0"  # Corrected expected data
    assert idx_up == tree_idx_z0
    assert p_up == pytest.approx(0.1)
