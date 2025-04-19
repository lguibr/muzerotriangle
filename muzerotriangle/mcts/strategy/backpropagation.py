# File: muzerotriangle/mcts/strategy/backpropagation.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(
    leaf_node: "Node",
    value_from_leaf: float,  # Value estimate v_L from f(s_L) or terminal reward
    discount: float,  # Gamma discount factor
) -> int:
    """
    Propagates the simulation value back up the tree from the leaf node.
    In MuZero, the value incorporates predicted rewards along the path.
    G = r_{k+1} + gamma * r_{k+2} + ... + gamma^N * v_L
    Returns the depth of the backpropagation path.
    """
    current_node: Node | None = leaf_node
    path_str = []
    depth = 0
    value_to_propagate: float = value_from_leaf  # Start with v_L or terminal reward

    logger.debug(
        f"[Backprop] Starting backprop from leaf node {leaf_node.action_taken} "
        f"with initial value_from_leaf={value_from_leaf:.4f} and discount={discount}"
    )

    while current_node is not None:
        # --- MuZero Modification ---
        # The value estimate Q(s,a) stored at the parent relates to the value G
        # derived *from* the current_node.
        # The value_to_propagate represents the value starting from *this* node's state.
        # We add this value to the node's statistics.
        # When moving to the parent, we discount the current value_to_propagate
        # and add the predicted reward 'r' that led *to* the current_node.
        # ---

        q_before = current_node.value_estimate
        total_val_before = current_node.value_sum
        visits_before = current_node.visit_count

        current_node.visit_count += 1
        current_node.value_sum += value_to_propagate  # Add the calculated G

        q_after = current_node.value_estimate
        total_val_after = current_node.value_sum
        visits_after = current_node.visit_count

        action_str = (
            f"Act={current_node.action_taken}"
            if current_node.action_taken is not None
            else "Root"
        )
        path_str.append(
            f"N({action_str},R={current_node.reward:.2f},V={visits_after},Q={q_after:.3f})"
        )

        logger.debug(
            f"  [Backprop] Depth {depth}: Node({action_str}), "
            f"Visits: {visits_before} -> {visits_after}, "
            f"PropagatedG={value_to_propagate:.4f}, "
            f"ValueSum: {total_val_before:.3f} -> {total_val_after:.3f}, "
            f"Q: {q_before:.3f} -> {q_after:.3f}"
        )

        # --- MuZero: Calculate value for the parent ---
        # G_{parent} = r_{current} + gamma * G_{current}
        # where r_{current} is the reward predicted by g() for reaching current_node
        if current_node.parent is not None:  # Don't update G beyond the root
            value_to_propagate = current_node.reward + discount * value_to_propagate
            logger.debug(
                f"    [Backprop] PrevG={value_to_propagate / discount - current_node.reward:.4f} -> "
                f"NextG (for parent) = r_k({current_node.reward:.3f}) + gamma({discount}) * PrevG = {value_to_propagate:.4f}"
            )
        # ---

        current_node = current_node.parent
        depth += 1

    logger.debug(f"[Backprop] Finished. Path: {' <- '.join(reversed(path_str))}")
    return depth
