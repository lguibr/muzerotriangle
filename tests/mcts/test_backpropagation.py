# File: tests/mcts/test_backpropagation.py
import pytest
import torch

from muzerotriangle.mcts.core.node import Node
from muzerotriangle.mcts.strategy import backpropagation


@pytest.fixture
def simple_tree() -> Node:
    """Creates a simple 3-level tree for backpropagation testing."""
    # Root (s0)
    root = Node(hidden_state=torch.tensor([0.0]), reward=0.0)
    # Child 1 (s1, reached via action a1, reward r1=0.1)
    child1 = Node(
        parent=root,
        action_taken=1,
        hidden_state=torch.tensor([1.0]),
        reward=0.1,
        prior=0.6,
    )
    # Child 2 (s2, reached via action a2, reward r2=0.2)
    child2 = Node(
        parent=root,
        action_taken=2,
        hidden_state=torch.tensor([2.0]),
        reward=0.2,
        prior=0.4,
    )
    # Grandchild (s3, reached from s1 via action a3, reward r3=0.3)
    grandchild = Node(
        parent=child1,
        action_taken=3,
        hidden_state=torch.tensor([3.0]),
        reward=0.3,
        prior=0.9,
    )
    root.children = {1: child1, 2: child2}
    child1.children = {3: grandchild}
    return root


def test_backpropagate_from_leaf(simple_tree: Node):
    """Test backpropagation from a leaf node (grandchild)."""
    root = simple_tree
    child1 = root.children[1]
    grandchild = child1.children[3]
    leaf_value = 0.8  # Value predicted at the leaf state s3
    discount = 0.9

    # Backpropagate from grandchild
    depth = backpropagation.backpropagate_value(grandchild, leaf_value, discount)

    # Check grandchild stats
    assert grandchild.visit_count == 1
    assert grandchild.value_sum == pytest.approx(leaf_value)
    assert grandchild.value_estimate == pytest.approx(leaf_value)

    # Check child1 stats
    # Expected value at child1 = r3 + gamma * V(s3) = 0.3 + 0.9 * 0.8 = 0.3 + 0.72 = 1.02
    assert child1.visit_count == 1
    assert child1.value_sum == pytest.approx(1.02)
    assert child1.value_estimate == pytest.approx(1.02)

    # Check root stats
    # Expected value at root = r1 + gamma * V(s1) = 0.1 + 0.9 * 1.02 = 0.1 + 0.918 = 1.018
    assert root.visit_count == 1
    assert root.value_sum == pytest.approx(1.018)
    assert root.value_estimate == pytest.approx(1.018)

    # Check depth
    assert depth == 2  # Path: grandchild -> child1 -> root


def test_backpropagate_from_intermediate(simple_tree: Node):
    """Test backpropagation from an intermediate node (child2)."""
    root = simple_tree
    child2 = root.children[2]
    leaf_value = 0.5  # Value predicted at state s2
    discount = 0.9

    # Backpropagate from child2
    depth = backpropagation.backpropagate_value(child2, leaf_value, discount)

    # Check child2 stats
    assert child2.visit_count == 1
    assert child2.value_sum == pytest.approx(leaf_value)
    assert child2.value_estimate == pytest.approx(leaf_value)

    # Check root stats
    # Expected value at root = r2 + gamma * V(s2) = 0.2 + 0.9 * 0.5 = 0.2 + 0.45 = 0.65
    assert root.visit_count == 1
    assert root.value_sum == pytest.approx(0.65)
    assert root.value_estimate == pytest.approx(0.65)

    # Check depth
    assert depth == 1  # Path: child2 -> root


def test_backpropagate_multiple_visits(simple_tree: Node):
    """Test backpropagation with multiple visits through the same nodes."""
    root = simple_tree
    child1 = root.children[1]
    grandchild = child1.children[3]
    discount = 0.9

    # First backpropagation from grandchild
    backpropagation.backpropagate_value(grandchild, 0.8, discount)
    # Second backpropagation from grandchild (different leaf value)
    backpropagation.backpropagate_value(grandchild, 0.6, discount)

    # Check grandchild stats
    assert grandchild.visit_count == 2
    assert grandchild.value_sum == pytest.approx(0.8 + 0.6)
    assert grandchild.value_estimate == pytest.approx((0.8 + 0.6) / 2)

    # Check child1 stats
    # Value 1 = 0.3 + 0.9 * 0.8 = 1.02
    # Value 2 = 0.3 + 0.9 * 0.6 = 0.84
    assert child1.visit_count == 2
    assert child1.value_sum == pytest.approx(1.02 + 0.84)
    assert child1.value_estimate == pytest.approx((1.02 + 0.84) / 2)

    # Check root stats
    # Value 1 = 0.1 + 0.9 * 1.02 = 1.018
    # Value 2 = 0.1 + 0.9 * 0.84 = 0.856
    assert root.visit_count == 2
    assert root.value_sum == pytest.approx(1.018 + 0.856)
    assert root.value_estimate == pytest.approx((1.018 + 0.856) / 2)

    # Backpropagate from child2
    backpropagation.backpropagate_value(root.children[2], 0.5, discount)

    # Check child2 stats
    assert root.children[2].visit_count == 1
    assert root.children[2].value_sum == pytest.approx(0.5)

    # Check root stats again
    # Value 3 = 0.2 + 0.9 * 0.5 = 0.65
    assert root.visit_count == 3
    assert root.value_sum == pytest.approx(1.018 + 0.856 + 0.65)
    assert root.value_estimate == pytest.approx((1.018 + 0.856 + 0.65) / 3)
