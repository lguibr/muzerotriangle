# File: tests/utils/test_sumtree.py
import logging

import pytest

from muzerotriangle.utils.sumtree import SumTree, sumtree_logger  # Import logger

# Configure logging for tests to see debug messages from SumTree
# logging.basicConfig(level=logging.DEBUG) # Set root logger level if needed
# sumtree_logger.setLevel(logging.DEBUG) # Set specific logger level


def dump_sumtree_state(tree: SumTree, test_name: str):
    """Helper function to print SumTree state for debugging."""
    print(f"\n--- SumTree State Dump ({test_name}) ---")
    print(f"  User Capacity: {tree.capacity}")
    print(f"  Internal Capacity: {tree._internal_capacity}")  # Log internal capacity
    print(f"  n_entries: {tree.n_entries}")
    print(f"  data_pointer: {tree.data_pointer}")
    print(f"  _max_priority: {tree._max_priority:.4f}")
    print(f"  total_priority (root): {tree.total():.4f}")
    # Only print the relevant part of the tree array
    tree_size = 2 * tree._internal_capacity - 1  # Use internal capacity
    print(
        f"  Tree array (size {len(tree.tree)}, showing up to {tree_size}): {tree.tree[:tree_size]}"
    )
    # Print only the populated part of the data array (up to n_entries)
    print(
        f"  Data array (size {len(tree.data)}, showing up to {tree.n_entries}): {tree.data[: tree.n_entries]}"
    )
    print("--- End Dump ---")


@pytest.fixture
def sum_tree_cap5() -> SumTree:
    """Provides a SumTree instance with user capacity 5."""
    # Internal capacity will be 8
    return SumTree(capacity=5)


def test_sumtree_init():
    tree_user_cap = 10
    tree = SumTree(capacity=tree_user_cap)
    internal_cap = 16  # Next power of 2 >= 10
    assert tree.capacity == tree_user_cap
    assert tree._internal_capacity == internal_cap
    assert len(tree.tree) == 2 * internal_cap - 1  # 31
    assert len(tree.data) == internal_cap  # 16
    assert tree.data_pointer == 0
    assert tree.n_entries == 0
    assert tree.total() == 0.0
    assert tree.max_priority == 1.0


def test_sumtree_add_single(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    # First add goes to data_pointer=0, tree_idx = 0 + 8 - 1 = 7
    tree_idx = sum_tree_cap5.add(0.5, "data1")
    assert tree_idx == 7
    assert sum_tree_cap5.n_entries == 1
    assert sum_tree_cap5.data_pointer == 1  # Wraps around user capacity 5
    assert sum_tree_cap5.total() == pytest.approx(0.5)
    assert sum_tree_cap5.max_priority == pytest.approx(0.5)
    assert sum_tree_cap5.tree[tree_idx] == 0.5
    assert sum_tree_cap5.data[0] == "data1"


def test_sumtree_add_multiple(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    data = ["d0", "d1", "d2", "d3", "d4"]
    expected_leaf_indices = [7, 8, 9, 10, 11]  # Based on internal capacity 8
    expected_max_priority = 0.0
    for i, (p, d) in enumerate(zip(priorities, data, strict=False)):
        tree_idx = sum_tree_cap5.add(p, d)
        assert tree_idx == expected_leaf_indices[i]
        expected_max_priority = max(expected_max_priority, p)
        assert sum_tree_cap5.n_entries == i + 1
        assert sum_tree_cap5.data_pointer == (i + 1) % 5  # Wrap around user capacity 5
        assert sum_tree_cap5.total() == pytest.approx(sum(priorities[: i + 1]))
        assert sum_tree_cap5.max_priority == pytest.approx(expected_max_priority)

    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 0
    # Data array has internal capacity, check only the first 5 slots
    assert sum_tree_cap5.data[:5] == data


def test_sumtree_add_overflow(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]
    data = ["d0", "d1", "d2", "d3", "d4"]
    expected_leaf_indices = [7, 8, 9, 10, 11]
    for i, (p, d) in enumerate(zip(priorities, data, strict=False)):
        tree_idx = sum_tree_cap5.add(p, d)
        assert tree_idx == expected_leaf_indices[i]

    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.total() == pytest.approx(sum(priorities))
    assert sum_tree_cap5.max_priority == pytest.approx(0.8)

    # Add one more, overwriting the first element (data_idx 0, tree_idx 7)
    tree_idx_5 = sum_tree_cap5.add(1.0, "d5")
    assert tree_idx_5 == 7  # Should overwrite leaf 7
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 1  # Wraps around user capacity 5
    assert sum_tree_cap5.data[0] == "d5"
    assert sum_tree_cap5.data[1] == "d1"  # Unchanged
    assert sum_tree_cap5.total() == pytest.approx(sum(priorities[1:]) + 1.0)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[7] == 1.0  # Check leaf node updated

    # Add another, overwriting the second element (data_idx 1, tree_idx 8)
    tree_idx_6 = sum_tree_cap5.add(0.05, "d6")
    assert tree_idx_6 == 8  # Should overwrite leaf 8
    assert sum_tree_cap5.n_entries == 5
    assert sum_tree_cap5.data_pointer == 2  # Wraps around user capacity 5
    assert sum_tree_cap5.data[0] == "d5"
    assert sum_tree_cap5.data[1] == "d6"
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.05 + 0.2 + 0.8 + 0.4)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[8] == 0.05  # Check leaf node updated


def test_sumtree_update(sum_tree_cap5: SumTree):
    # Internal capacity = 8. Leaves start at index 7.
    tree_idx_0 = sum_tree_cap5.add(0.5, "data0")  # Leaf 7
    tree_idx_1 = sum_tree_cap5.add(0.3, "data1")  # Leaf 8
    assert tree_idx_0 == 7
    assert tree_idx_1 == 8
    assert sum_tree_cap5.total() == pytest.approx(0.8)
    assert sum_tree_cap5.max_priority == pytest.approx(0.5)

    sum_tree_cap5.update(tree_idx_0, 1.0)  # Update leaf 7
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.3)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[tree_idx_0] == pytest.approx(1.0)

    sum_tree_cap5.update(tree_idx_1, 0.1)  # Update leaf 8
    assert sum_tree_cap5.total() == pytest.approx(1.0 + 0.1)
    assert sum_tree_cap5.max_priority == pytest.approx(1.0)
    assert sum_tree_cap5.tree[tree_idx_1] == pytest.approx(0.1)


def test_sumtree_retrieve(sum_tree_cap5: SumTree):
    """Test the _retrieve method directly with internal capacity 8."""
    # --- Enable Debug Logging ---
    original_level = sumtree_logger.level
    sumtree_logger.setLevel(logging.DEBUG)
    # ---

    # Internal capacity = 8. Tree size = 15. Leaves = indices 7..14
    # User capacity = 5. Data indices = 0..4. n_entries = 5.
    # Mapping: Data 0 -> Leaf 7, Data 1 -> Leaf 8, ..., Data 4 -> Leaf 11

    data_map = {}
    priorities = [0.1, 0.5, 0.2, 0.8, 0.4]  # Sum = 2.0
    expected_leaf_indices = [7, 8, 9, 10, 11]
    for i, p in enumerate(priorities):
        data_id = f"d{i}"
        tree_idx = sum_tree_cap5.add(p, data_id)
        assert tree_idx == expected_leaf_indices[i]
        data_map[tree_idx] = data_id

    print(f"\n[test_retrieve] Tree state: {sum_tree_cap5.tree}")
    print(f"[test_retrieve] Data state: {sum_tree_cap5.data}")
    print(f"[test_retrieve] Data map (tree_idx -> data_id): {data_map}")

    # Expected Tree (cap 8, 5 entries):
    # Leaves: [0.1, 0.5, 0.2, 0.8, 0.4, 0.0, 0.0, 0.0] (Indices 7-14)
    # Lvl 2:  [0.6, 1.0, 0.4, 0.0] (Indices 3-6)
    # Lvl 1:  [1.6, 0.4] (Indices 1-2)
    # Root:   [2.0] (Index 0)
    # Full Tree: [2.0, 1.6, 0.4, 0.6, 1.0, 0.4, 0.0, 0.1, 0.5, 0.2, 0.8, 0.4, 0.0, 0.0, 0.0]

    # Test retrieval based on sample values
    # Cumulative sums: [0.1, 0.6, 0.8, 1.6, 2.0]
    test_cases = {
        0.05: 7,  # Should fall in the first bucket (index 7, prio 0.1)
        0.1: 8,  # Should fall in the second bucket (index 8, prio 0.5)
        0.15: 8,
        0.6: 9,  # Should fall in the third bucket (index 9, prio 0.2)
        0.7: 9,
        0.8: 10,  # Should fall in the fourth bucket (index 10, prio 0.8)
        1.5: 10,
        1.6: 11,  # Should fall in the fifth bucket (index 11, prio 0.4)
        1.99: 11,
    }

    try:
        for sample_value, expected_tree_idx in test_cases.items():
            print(f"[test_retrieve] Testing sample {sample_value:.4f}")
            retrieved_tree_idx = sum_tree_cap5._retrieve(0, sample_value)
            assert retrieved_tree_idx == expected_tree_idx, (
                f"Sample {sample_value:.4f} -> Expected {expected_tree_idx}, Got {retrieved_tree_idx}"
            )
    finally:
        # --- Restore Log Level ---
        sumtree_logger.setLevel(original_level)
        # ---


def test_sumtree_retrieve_with_zeros(sum_tree_cap5: SumTree):
    """Test _retrieve with zero priority nodes."""
    # --- Enable Debug Logging ---
    original_level = sumtree_logger.level
    sumtree_logger.setLevel(logging.DEBUG)
    # ---

    # Internal capacity = 8. Leaves = indices 7..14
    # User capacity = 5. Data indices = 0..4. n_entries = 5.
    # Mapping: Data 0 -> Leaf 7, ..., Data 4 -> Leaf 11

    data_map = {}
    priorities = [0.0, 0.4, 0.6, 0.0, 0.0]  # Sum = 1.0
    data = ["z0", "iA", "iB", "z1", "z2"]
    expected_leaf_indices = [7, 8, 9, 10, 11]
    for i, p in enumerate(priorities):
        data_id = data[i]
        tree_idx = sum_tree_cap5.add(p, data_id)
        assert tree_idx == expected_leaf_indices[i]
        data_map[tree_idx] = data_id

    print(f"\n[test_retrieve_zeros] Tree state: {sum_tree_cap5.tree}")
    print(f"[test_retrieve_zeros] Data state: {sum_tree_cap5.data}")
    print(f"[test_retrieve_zeros] Data map (tree_idx -> data_id): {data_map}")

    # Expected Tree (cap 8, 5 entries with zeros):
    # Leaves: [0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0] (Indices 7-14)
    # Lvl 2:  [0.4, 0.6, 0.0, 0.0] (Indices 3-6)
    # Lvl 1:  [1.0, 0.0] (Indices 1-2)
    # Root:   [1.0] (Index 0)
    # Full Tree: [1.0, 1.0, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Test retrieval based on sample values
    # Cumulative sums: [0.0, 0.4, 1.0, 1.0, 1.0]
    test_cases = {
        0.0: 8,  # Should skip zero-priority leaf 7 and land in leaf 8
        0.1: 8,  # Should land in leaf 8
        0.3: 8,
        0.399: 8,
        0.4: 9,  # Should land in leaf 9
        0.5: 9,
        0.99: 9,
    }

    try:
        for sample_value, expected_tree_idx in test_cases.items():
            print(f"[test_retrieve_zeros] Testing sample {sample_value:.4f}")
            retrieved_tree_idx = sum_tree_cap5._retrieve(0, sample_value)
            assert retrieved_tree_idx == expected_tree_idx, (
                f"Sample {sample_value:.4f} -> Expected {expected_tree_idx}, Got {retrieved_tree_idx}"
            )
    finally:
        # --- Restore Log Level ---
        sumtree_logger.setLevel(original_level)
        # ---


def test_sumtree_get_leaf_edge_cases(sum_tree_cap5: SumTree):
    """Test edge cases for get_leaf."""
    # --- Enable Debug Logging ---
    original_level = sumtree_logger.level
    sumtree_logger.setLevel(logging.DEBUG)
    # ---

    try:
        # Empty tree
        with pytest.raises(
            ValueError,
            match="Cannot sample from SumTree with zero or negative total priority",
        ):
            sum_tree_cap5.get_leaf(0.5)

        # Single item
        tree_idx_0 = sum_tree_cap5.add(1.0, "only_item")  # Should be leaf index 7
        assert tree_idx_0 == 7
        assert sum_tree_cap5.n_entries == 1
        assert sum_tree_cap5.total() == pytest.approx(1.0)
        print(f"\n[test_edge_cases] Single item tree: {sum_tree_cap5.tree}")

        # Test sampling at 0.0
        print("[test_edge_cases] Testing get_leaf(0.0)")
        idx0, p0, d0 = sum_tree_cap5.get_leaf(0.0)
        assert d0 == "only_item"
        assert p0 == pytest.approx(1.0)
        assert idx0 == tree_idx_0  # Should be 7

        # Test sampling close to total priority
        print("[test_edge_cases] Testing get_leaf(1.0 - eps)")
        idx1, p1, d1 = sum_tree_cap5.get_leaf(1.0 - 1e-9)
        assert d1 == "only_item"
        assert p1 == pytest.approx(1.0)
        assert idx1 == tree_idx_0

        # Test sampling exactly at total priority (should be clipped)
        print("[test_edge_cases] Testing get_leaf(1.0)")
        idx_exact, p_exact, d_exact = sum_tree_cap5.get_leaf(1.0)
        assert d_exact == "only_item"
        assert idx_exact == tree_idx_0

        # Test sampling above total priority (should be clipped)
        print("[test_edge_cases] Testing get_leaf(1.1)")
        idx_above, p_above, d_above = sum_tree_cap5.get_leaf(1.1)
        assert d_above == "only_item"
        assert idx_above == tree_idx_0

        # Test sampling below zero (should be clipped)
        print("[test_edge_cases] Testing get_leaf(-0.1)")
        idx_below, p_below, d_below = sum_tree_cap5.get_leaf(-0.1)
        assert d_below == "only_item"
        assert idx_below == tree_idx_0

        # Zero priority item
        sum_tree_cap5.reset()
        tree_idx_z0 = sum_tree_cap5.add(0.0, "z0")  # data_idx 0, tree_idx 7
        tree_idx_iA = sum_tree_cap5.add(0.4, "itemA")  # data_idx 1, tree_idx 8
        tree_idx_iB = sum_tree_cap5.add(0.6, "itemB")  # data_idx 2, tree_idx 9
        sum_tree_cap5.add(0.0, "z1")  # data_idx 3, tree_idx 10
        sum_tree_cap5.add(0.0, "z2")  # data_idx 4, tree_idx 11
        assert sum_tree_cap5.n_entries == 5
        assert sum_tree_cap5.total() == pytest.approx(1.0)
        print(f"[test_edge_cases] Zero priority tree: {sum_tree_cap5.tree}")
        print(f"[test_edge_cases] Data: {sum_tree_cap5.data}")

        # Sampling value < 0.4 should yield 'itemA' (index 8)
        print("[test_edge_cases] Testing get_leaf(0.3) with zero priority item")
        idx, p, d = sum_tree_cap5.get_leaf(0.3)
        assert d == "itemA"
        assert p == pytest.approx(0.4)
        assert idx == tree_idx_iA  # Should be 8

        # Sampling value >= 0.4 and < 1.0 should yield 'itemB' (index 9)
        print("[test_edge_cases] Testing get_leaf(0.5) with zero priority item")
        idx, p, d = sum_tree_cap5.get_leaf(0.5)
        assert d == "itemB"
        assert p == pytest.approx(0.6)
        assert idx == tree_idx_iB  # Should be 9

        # Test sampling exactly at boundary (0.4)
        print("[test_edge_cases] Testing get_leaf(0.4) boundary")
        idx_b, p_b, d_b = sum_tree_cap5.get_leaf(0.4)
        assert d_b == "itemB"  # Should land in the second non-zero bucket
        assert idx_b == tree_idx_iB  # Should be 9

        # Test sampling at 0.0
        print("[test_edge_cases] Testing get_leaf(0.0) boundary")
        idx_0, p_0, d_0 = sum_tree_cap5.get_leaf(0.0)
        assert d_0 == "itemA"  # Should pick the first non-zero element
        assert idx_0 == tree_idx_iA  # Should be 8

        # Test updating a zero-priority item
        sum_tree_cap5.update(tree_idx_z0, 0.1)  # Update "z0" priority (index 7)
        assert sum_tree_cap5.total() == pytest.approx(1.1)
        print(f"[test_edge_cases] After update tree: {sum_tree_cap5.tree}")
        print("[test_edge_cases] Testing get_leaf(0.05) after update")
        idx_up, p_up, d_up = sum_tree_cap5.get_leaf(0.05)
        assert d_up == "z0"
        assert idx_up == tree_idx_z0  # Should be 7
        assert p_up == pytest.approx(0.1)

    finally:
        # --- Restore Log Level ---
        sumtree_logger.setLevel(original_level)
        # ---
