# File: tests/mcts/test_selection.py
import math

import pytest
import torch

# Import from top-level conftest implicitly, or specific fixtures if needed
from muzerotriangle.config import MCTSConfig, ModelConfig  # Import ModelConfig
from muzerotriangle.mcts.core.node import Node
from muzerotriangle.mcts.strategy import selection

# --- REMOVED: Local fixture imports (use global ones) ---
# from .conftest import (EnvConfig, MockGameState, deep_expanded_node_mock_state,
#                       expanded_root_node, mock_mcts_config, node_with_hidden_state, root_node_real_state)


# --- Test PUCT Calculation ---
@pytest.mark.usefixtures("mock_mcts_config")  # Indicate fixture usage if not argument
def test_puct_calculation_basic(
    mock_mcts_config: MCTSConfig, node_with_hidden_state: Node
):
    """Test basic PUCT score calculation."""
    # --- Add check for hidden_state ---
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
    # --- Add check for hidden_state ---
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


# --- Test Child Selection --- (No changes needed here)
# ... test_select_child_node_basic, test_select_child_node_no_children ...
def test_select_child_node_basic(
    expanded_root_node: Node, mock_mcts_config: MCTSConfig
):
    parent = expanded_root_node
    parent.visit_count = 10
    if 0 not in parent.children or 1 not in parent.children:
        pytest.skip("Req children missing")
    child0 = parent.children[0]
    child0.visit_count = 1
    child0.value_sum = 0.8
    child0.prior_probability = 0.1
    child1 = parent.children[1]
    child1.visit_count = 5
    child1.value_sum = 0.5
    child1.prior_probability = 0.6
    selected_child = selection.select_child_node(parent, mock_mcts_config)
    assert selected_child is child0


def test_select_child_node_no_children(
    root_node_real_state: Node, mock_mcts_config: MCTSConfig
):
    parent = root_node_real_state
    assert not parent.children
    with pytest.raises(selection.SelectionError):
        selection.select_child_node(parent, mock_mcts_config)


# --- Test Dirichlet Noise (No changes needed here) ---
# ... test_add_dirichlet_noise ...
def test_add_dirichlet_noise(expanded_root_node: Node, mock_mcts_config: MCTSConfig):
    node = expanded_root_node
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
    assert priors_changed
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
    for child in root.children.values():
        assert not child.is_expanded
    leaf, depth = selection.traverse_to_leaf(root, mock_mcts_config)
    assert leaf in root.children.values()
    assert depth == 1


def test_traverse_to_leaf_max_depth(
    expanded_root_node: Node,
    mock_mcts_config: MCTSConfig,
    mock_model_config: ModelConfig,  # Add model_config fixture
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
    # --- Use ModelConfig for hidden dim ---
    hidden_dim = mock_model_config.HIDDEN_STATE_DIM
    if child0.hidden_state is None:
        child0.hidden_state = torch.randn((hidden_dim,))
    gc_state = (
        torch.randn_like(child0.hidden_state)
        if child0.hidden_state is not None
        else None
    )
    child0.children[0] = Node(hidden_state=gc_state, parent=child0, action_taken=0)
    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    assert leaf in root.children.values()
    assert depth == 1


# --- test_traverse_to_leaf_deeper_muzero uses fixture from conftest ---
def test_traverse_to_leaf_deeper_muzero(
    deep_expanded_node_mock_state: Node, mock_mcts_config: MCTSConfig
):
    root = deep_expanded_node_mock_state
    config_copy = mock_mcts_config.model_copy(deep=True)
    config_copy.max_search_depth = 10
    assert 0 in root.children
    child0 = root.children[0]
    assert child0.children
    preferred_gc_action = 1
    if 1 not in child0.children and child0.children:
        preferred_gc_action = next(iter(child0.children.keys()))
    elif not child0.children:
        pytest.fail("Fixture error: Child 0 has no children")
    expected_leaf = child0.children.get(preferred_gc_action)
    assert expected_leaf is not None
    leaf, depth = selection.traverse_to_leaf(root, config_copy)
    assert leaf is expected_leaf
    assert depth == 2
