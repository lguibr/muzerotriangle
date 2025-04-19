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
    assert root.visit_count >= test_config.num_simulations
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
    # --- ADJUSTED ASSERTION ---
    # The root node's hidden_state and predicted_value are set during initial inference.
    # expand_node is called, but should return early without adding children if valid_actions is empty.
    assert not root.children, "Root should have no children when no valid actions exist"
    # is_expanded checks if self.children is non-empty.
    assert not root.is_expanded, "Root should not be expanded (no children added)"
    # --- END ADJUSTED ASSERTION ---
    assert max_depth >= 0  # Depth reflects initial inference/backprop
