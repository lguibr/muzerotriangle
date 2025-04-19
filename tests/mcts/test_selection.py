# File: tests/mcts/test_selection.py
# File: tests/mcts/test_selection.py
import math

import pytest
import torch

# Import from top-level conftest implicitly, or specific fixtures if needed
from muzerotriangle.config import MCTSConfig, ModelConfig
from muzerotriangle.mcts.core.node import Node
from muzerotriangle.mcts.strategy import selection


# --- Test PUCT Calculation ---
@pytest.mark.usefixtures("mock_mcts_config")
def test_puct_calculation_basic(
    mock_mcts_config: MCTSConfig, node_with_hidden_state: Node
):
    """Test basic PUCT score calculation."""
    if node_with_hidden_state.hidden_state is None:
        pytest.skip("Node needs hidden state")
    parent = Node(hidden_state=torch.randn_like(node_with_hidden_state.hidden_state))
    parent.visit_count = 25
    child = node_with_hidden_state
    child.parent = parent
    child.visit_count = 5
    child.value_sum = 3.0
    child.prior_probability = 0.2
    score, q_value, exploration = selection.calculate_puct_score(
        parent, child, mock_mcts_config
    )
    assert q_value == pytest.approx(0.6)
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.2 * (math.sqrt(25) / (1 + 5))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


@pytest.mark.usefixtures("mock_mcts_config")
def test_puct_calculation_unvisited_child(
    mock_mcts_config: MCTSConfig, node_with_hidden_state: Node
):
    """Test PUCT score for an unvisited child node."""
    if node_with_hidden_state.hidden_state is None:
        pytest.skip("Node needs hidden state")
    parent = Node(hidden_state=torch.randn_like(node_with_hidden_state.hidden_state))
    parent.visit_count = 10
    child = node_with_hidden_state
    child.parent = parent
    child.visit_count = 0
    child.value_sum = 0.0
    child.prior_probability = 0.5
    score, q_value, exploration = selection.calculate_puct_score(
        parent, child, mock_mcts_config
    )
    assert q_value == 0.0
    expected_exploration = (
        mock_mcts_config.puct_coefficient * 0.5 * (math.sqrt(10) / (1 + 0))
    )
    assert exploration == pytest.approx(expected_exploration)
    assert score == pytest.approx(q_value + exploration)


# --- Test Child Selection ---
def test_select_child_node_basic(
    expanded_root_node: Node, mock_mcts_config: MCTSConfig
):
    """Test selecting the best child based on PUCT."""
    parent = expanded_root_node
    parent.visit_count = 10  # Set parent visits for calculation

    # Ensure there are at least two children to compare
    if len(parent.children) < 2:
        pytest.skip("Requires at least two children for meaningful selection test.")

    # Assign different stats to make selection deterministic for the test
    child_list = list(parent.children.values())
    child0 = child_list[0]
    child1 = child_list[1]

    # Make child1 clearly better according to PUCT
    child0.visit_count = 5
    child0.value_sum = 0.8 * child0.visit_count  # Q = 0.8
    child0.prior_probability = 0.1

    child1.visit_count = 1
    child1.value_sum = 0.5 * child1.visit_count  # Q = 0.5
    child1.prior_probability = 0.6  # Higher prior, lower visits -> higher exploration

    # Calculate scores manually to verify selection logic
    score0, _, _ = selection.calculate_puct_score(parent, child0, mock_mcts_config)
    score1, _, _ = selection.calculate_puct_score(parent, child1, mock_mcts_config)

    selected_child = selection.select_child_node(parent, mock_mcts_config)

    # Assert that the child with the higher score was selected
    if score1 > score0:
        assert selected_child is child1
    else:
        assert selected_child is child0


def test_select_child_node_no_children(
    root_node_real_state: Node, mock_mcts_config: MCTSConfig
):
    parent = root_node_real_state
    assert not parent.children
    with pytest.raises(selection.SelectionError):
        selection.select_child_node(parent, mock_mcts_config)


# --- Test Dirichlet Noise ---
def test_add_dirichlet_noise(expanded_root_node: Node, mock_mcts_config: MCTSConfig):
    node = expanded_root_node
    if not node.children:
        pytest.skip("Node needs children to test noise.")

    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.dirichlet_alpha = 0.5
    config_copy.dirichlet_epsilon = 0.25
    n_children = len(node.children)
    original_priors = {a: c.prior_probability for a, c in node.children.items()}

    selection.add_dirichlet_noise(node, config_copy)

    new_priors = {a: c.prior_probability for a, c in node.children.items()}
    mixed_sum = sum(new_priors.values())
    assert len(new_priors) == n_children
    priors_changed = False
    for action, new_p in new_priors.items():
        assert 0.0 <= new_p <= 1.0
        if abs(new_p - original_priors[action]) > 1e-9:
            priors_changed = True
    assert priors_changed, "Priors did not change after adding noise"
    assert mixed_sum == pytest.approx(1.0, abs=1e-6)


# --- Test Traversal ---
def test_traverse_to_leaf_unexpanded(
    root_node_real_state: Node, mock_mcts_config: MCTSConfig
):
    leaf, depth = selection.traverse_to_leaf(root_node_real_state, mock_mcts_config)
    assert leaf is root_node_real_state
    assert depth == 0


def test_traverse_to_leaf_expanded(
    expanded_root_node: Node, mock_mcts_config: MCTSConfig
):
    root = expanded_root_node
    if not root.children:
        pytest.skip("Root node fixture did not expand.")
    for child in root.children.values():
        assert not child.is_expanded  # Ensure children are leaves initially
    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)
    assert leaf in root.children.values()
    assert depth == 1


def test_traverse_to_leaf_max_depth(
    expanded_root_node: Node,
    mock_mcts_config: MCTSConfig,
    mock_model_config: ModelConfig,
):
    root = expanded_root_node
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 0
    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    assert leaf is root
    assert depth == 0

    config_copy.max_search_depth = 1
    if not root.children:
        pytest.skip("Root has no children.")
    child0 = next(iter(root.children.values()))
    hidden_dim = mock_model_config.HIDDEN_STATE_DIM
    if child0.hidden_state is None:
        child0.hidden_state = torch.randn((hidden_dim,))
    gc_state = (
        torch.randn_like(child0.hidden_state)
        if child0.hidden_state is not None
        else None
    )
    # Ensure grandchild action is different from child action if possible
    gc_action = 0 if child0.action_taken != 0 else 1
    child0.children[gc_action] = Node(
        hidden_state=gc_state, parent=child0, action_taken=gc_action
    )

    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    assert leaf in root.children.values()  # Should stop at depth 1
    assert depth == 1


# Use the corrected fixture name
def test_traverse_to_leaf_deeper_muzero(
    deep_expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    root = deep_expanded_node_mock_state
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 10  # Allow deep traversal

    # Find the child that was expanded in the fixture
    expanded_child = None
    for child in root.children.values():
        if child.children:
            expanded_child = child
            break
    assert expanded_child is not None, "Fixture error: No expanded child found"
    assert expanded_child.children, "Fixture error: Expanded child has no children"

    # Find an expected leaf (grandchild) - Removed as selection isn't guaranteed
    # expected_leaf = next(iter(expanded_child.children.values()), None)
    # assert expected_leaf is not None, "Fixture error: No grandchild found"

    # Traverse and check
    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    # --- FIXED ASSERTION ---
    assert leaf in expanded_child.children.values(), (
        "Returned leaf is not one of the expected grandchildren"
    )
    # --- END FIXED ASSERTION ---
    assert depth == 2
