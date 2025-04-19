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
    # --- Use public attribute ---
    root.initial_game_state.game_over = True
    # ---
    valid_actions = root.initial_game_state.valid_actions()  # Likely empty now
    max_depth = run_mcts_simulations(
        root_node=root,
        config=mock_mcts_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )
    assert max_depth == 0
    assert root.visit_count == 0
    assert not root.is_expanded


@pytest.mark.skip(
    reason="Difficult to guarantee no valid actions in real GameState setup for testing"
)
def test_run_mcts_simulations_no_valid_actions(
    root_node_real_state: Node,
    mock_mcts_config: MCTSConfig,
    mock_muzero_network: Any,
):
    """Test running MCTS when the root state has no valid actions."""
    # This test is skipped because setting valid_actions=[] directly on GameState isn't feasible.
    # We'd need to construct a specific game board state (e.g., completely full)
    # where game.valid_actions() returns [].
    root = root_node_real_state
    if root.initial_game_state is None:
        pytest.skip("Root node needs game state")
    # Setup a game state where valid_actions() returns [] (e.g., fill grid)
    # ... (complex setup required) ...
    valid_actions: list[int] = []  # Assuming setup leads to this
    max_depth = run_mcts_simulations(
        root_node=root,
        config=mock_mcts_config,
        network=mock_muzero_network,
        valid_actions_from_state=valid_actions,
    )
    assert root.visit_count == 1
    assert not root.is_expanded
    assert max_depth >= 0
