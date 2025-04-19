# File: muzerotriangle/mcts/strategy/backpropagation.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(
    leaf_node: "Node",
    value_from_leaf: float,
    discount: float,
) -> int:
    """
    Propagates the simulation value back up the tree from the leaf node.
    In MuZero, the value incorporates predicted rewards along the path.
    G = r_{k+1} + gamma * r_{k+2} + ... + gamma^N * v_L
    Returns the depth of the backpropagation path.
    """
    current_node: Node | None = leaf_node
    depth = 0
    value_to_propagate: float = value_from_leaf
    logger.debug(
        f"  [Backprop Start] Leaf Node (Action={leaf_node.action_taken if leaf_node.action_taken is not None else 'LeafRoot'}, Visits={leaf_node.visit_count}), StartValue={value_from_leaf:.4f}"
    )

    while current_node is not None:
        action_str_bp = (
            current_node.action_taken
            if current_node.action_taken is not None
            else "Root"
        )
        visits_before_bp = current_node.visit_count
        value_sum_before_bp = current_node.value_sum
        logger.debug(
            f"    [Backprop Pre ] Node({action_str_bp:<4}, Depth={depth}): V_b={visits_before_bp:<4}, Sum_b={value_sum_before_bp: .3f}, PropG={value_to_propagate:.4f}"
        )

        current_node.visit_count += 1
        current_node.value_sum += value_to_propagate

        visits_after_bp = current_node.visit_count
        value_sum_after_bp = current_node.value_sum
        q_after = current_node.value_estimate
        logger.debug(
            f"    [Backprop Post] Node({action_str_bp:<4}, Depth={depth}): V {visits_before_bp}->{visits_after_bp}, Sum {value_sum_before_bp:.3f}->{value_sum_after_bp:.3f} (Q={q_after:.3f})"
        )

        if current_node.parent is not None:
            # Apply discount and add reward *before* moving to parent
            value_to_propagate = current_node.reward + discount * value_to_propagate
            logger.debug(
                f"      Value for parent: r_k={current_node.reward:.3f} + gamma={discount:.2f} * G_child={value_to_propagate / discount:.4f} = {value_to_propagate:.4f}"
            )
        else:
            # Reached the root node
            logger.debug("  [Backprop End] Reached root node.")
            break

        current_node = current_node.parent
        depth += 1

    return depth
