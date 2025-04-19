# File: tests/mcts/test_search.py
from typing import Any

import pytest

from muzerotriangle.config import MCTSConfig
from muzerotriangle.mcts import run_mcts_simulations
from muzerotriangle.mcts.core.node import Node

# Use fixtures from top-level conftest implicitly


def test_run_mcts_simulations_basic(
    root_node_real_state: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    root = root_node_real_state
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    valid_actions = root.initial_game_state.valid_actions()
    if not valid_actions:
        pytest.skip("Initial state has no valid actions.")
    test_config = mock_mcts_config.model_copy(update={"num_simulations": 5})
    max_depth = run_mcts_simulations(
        root_node=root,
        config=test_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )
    # Visit count should be 1 (initial backprop) + num_simulations
    assert root.visit_count == 1 + test_config.num_simulations
    assert root.is_expanded
    assert len(root.children) > 0
    assert max_depth >= 0


def test_run_mcts_simulations_on_terminal_state(
    root_node_real_state: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    root = root_node_real_state
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    root.initial_game_state.game_over = True
    valid_actions = root.initial_game_state.valid_actions()
    max_depth = run_mcts_simulations(
        root_node=root,
        config=mock_mcts_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )
    assert max_depth == 0
    assert root.visit_count == 0
    assert not root.is_expanded


def test_run_mcts_simulations_no_valid_actions(
    root_node_no_valid_actions: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    """Test running MCTS when the root state has no valid actions."""
    root = root_node_no_valid_actions
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")

    valid_actions = root.initial_game_state.valid_actions()
    assert not valid_actions

    # Run simulations - it should perform initial inference but fail expansion
    max_depth = run_mcts_simulations(
        root_node=root,
        config=mock_mcts_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )

    # Check state after running simulations
    # Initial inference happens, value is backpropagated once.
    # Simulation loop runs, but expansion fails each time. Backprop happens each time.
    expected_visits = 1 + mock_mcts_config.num_simulations
    assert root.visit_count == expected_visits, (
        f"Root visit count should be 1 + num_simulations ({expected_visits})"
    )
    assert not root.children, "Root should have no children when no valid actions exist"
    assert not root.is_expanded, "Root should not be expanded (no children added)"
    assert max_depth >= 0  # Depth reflects initial inference/backprop


def test_run_mcts_simulations_visits_and_depth(
    expanded_root_node: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    """Test visit counts and depth after running simulations on an expanded node."""
    root = expanded_root_node
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    valid_actions = root.initial_game_state.valid_actions()
    if not valid_actions or not root.children:
        pytest.skip("Expanded root node fixture invalid.")

    # Root visit count starts at 0 in the simplified fixture
    assert root.visit_count == 0

    num_sims = 10
    test_config = mock_mcts_config.model_copy(update={"num_simulations": num_sims})

    max_depth = run_mcts_simulations(
        root_node=root,
        config=test_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )

    # Root visit count should be exactly num_sims (no initial backprop needed as it was expanded)
    assert root.visit_count == num_sims, (
        f"Expected root visits {num_sims}, got {root.visit_count}"
    )
    # Total visits across children should equal root visits
    total_child_visits = sum(c.visit_count for c in root.children.values())
    assert total_child_visits == root.visit_count, (
        f"Sum of child visits ({total_child_visits}) != root visits ({root.visit_count})"
    )
    # Check that depth is at least 1 (since we selected children)
    assert max_depth >= 1, f"Expected depth >= 1, got {max_depth}"


def test_run_mcts_simulations_max_depth_limit(
    deep_expanded_node_mock_state: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    """Test that MCTS respects the max_search_depth limit."""
    root = deep_expanded_node_mock_state
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    valid_actions = root.initial_game_state.valid_actions()

    # Root visit count starts at 0 in the simplified fixture
    assert root.visit_count == 0

    # Set max depth to 1
    num_sims = 10
    test_config = mock_mcts_config.model_copy(
        update={"num_simulations": num_sims, "max_search_depth": 1}
    )

    max_depth_reached = run_mcts_simulations(
        root_node=root,
        config=test_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )

    # The maximum depth reached during *traversal* should be 1
    assert max_depth_reached == 1, (
        f"Expected max depth reached to be 1, got {max_depth_reached}"
    )
    # Root visits should still be num_sims (as each sim starts from root and backprops)
    assert root.visit_count == num_sims, (
        f"Expected root visits {num_sims}, got {root.visit_count}"
    )
    # Check that grandchildren were NOT visited (because traversal stopped at depth 1)
    for child in root.children.values():
        if child.children:
            for grandchild in child.children.values():
                assert grandchild.visit_count == 0, (
                    f"Grandchild {grandchild.action_taken} was visited, but max depth was 1."
                )
