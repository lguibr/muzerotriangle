# File: tests/mcts/test_expansion.py
from typing import Any  # Import Any

import pytest
import torch

from muzerotriangle.mcts.core.node import Node
from muzerotriangle.mcts.strategy import expansion

# Use fixtures from conftest implicitly or explicitly if needed
# from .conftest import MockMuZeroNetwork, Node, mock_muzero_network, node_with_hidden_state


# Use Any for mock network type hint
def test_expand_node_basic(node_with_hidden_state: Node, mock_muzero_network: Any):
    node = node_with_hidden_state
    hidden_state = node.hidden_state
    assert hidden_state is not None
    policy_logits, _ = mock_muzero_network.predict(hidden_state.unsqueeze(0))
    policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
    valid_actions = list(range(mock_muzero_network.action_dim))
    assert not node.is_expanded
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert node.is_expanded
    assert len(node.children) == len(valid_actions)
    for action in valid_actions:
        assert action in node.children
        child = node.children[action]
        assert child.parent is node
        assert child.action_taken == action
        assert child.prior_probability == pytest.approx(policy_map[action])
        assert child.hidden_state is not None
        assert child.hidden_state.shape == hidden_state.shape
        assert isinstance(child.reward, float)
        assert not child.is_expanded
        assert child.visit_count == 0
        assert child.value_sum == 0.0
        if action > 0:
            assert not torch.equal(child.hidden_state, hidden_state)


def test_expand_node_no_valid_actions(
    node_with_hidden_state: Node, mock_muzero_network: Any
):
    node = node_with_hidden_state
    policy_map = {0: 1.0}
    valid_actions: list[int] = []
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert not node.is_expanded


def test_expand_node_already_expanded(
    node_with_hidden_state: Node, mock_muzero_network: Any
):
    node = node_with_hidden_state
    policy_map = {0: 1.0}
    valid_actions = [0]
    child_state = (
        torch.randn_like(node.hidden_state) if node.hidden_state is not None else None
    )
    node.children[0] = Node(hidden_state=child_state, parent=node, action_taken=0)
    assert node.is_expanded
    original_children = node.children.copy()
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert node.children == original_children


def test_expand_node_missing_hidden_state(mock_muzero_network: Any):
    node = Node(parent=None, action_taken=None)
    policy_map = {0: 1.0}
    valid_actions = [0]
    expansion.expand_node(node, policy_map, mock_muzero_network, valid_actions)
    assert not node.is_expanded
