# File: muzerotriangle/mcts/strategy/expansion.py
import logging
from typing import TYPE_CHECKING

import torch  # Import torch for tensor check

from ..core.node import Node
from ..core.types import ActionPolicyMapping

if TYPE_CHECKING:
    from muzerotriangle.nn import NeuralNetwork  # Need network for dynamics
    from muzerotriangle.utils.types import ActionType


logger = logging.getLogger(__name__)


def expand_node(
    node_to_expand: "Node",  # Node containing s_k
    policy_prediction: ActionPolicyMapping,  # Policy p_k from f(s_k)
    network: "NeuralNetwork",  # Network interface to call dynamics (g)
    valid_actions: list["ActionType"] | None = None,  # Pass valid actions explicitly
):
    """Expands a leaf node in the MuZero search tree."""
    # ... (initial checks remain the same) ...
    if node_to_expand.is_expanded:
        return
    hidden_state_k = node_to_expand.hidden_state
    if hidden_state_k is None:
        logger.error(
            f"[Expand] Node {node_to_expand.action_taken} has no hidden state."
        )
        return

    logger.debug(f"[Expand] Expanding Node via action: {node_to_expand.action_taken}")
    actions_to_expand: list[ActionType]
    if valid_actions is not None:
        actions_to_expand = [a for a in valid_actions if a in policy_prediction]
        # ... (warnings about missing actions remain the same) ...
        if not actions_to_expand:
            return
    else:
        actions_to_expand = list(policy_prediction.keys())
        if not actions_to_expand:
            return
        logger.warning("[Expand] Expanding based on policy keys only.")

    children_created = 0
    for action in actions_to_expand:
        prior = policy_prediction.get(action, 0.0)
        if prior < 0:
            prior = 0.0

        try:
            # Ensure hidden_state_k has batch dim
            if not isinstance(hidden_state_k, torch.Tensor):
                logger.error(f"Hidden state is not a tensor: {type(hidden_state_k)}")
                continue  # Skip if state is invalid
            hidden_state_k_batch = (
                hidden_state_k.unsqueeze(0)
                if hidden_state_k.dim() == 1
                else hidden_state_k
            )

            # --- Call dynamics via the underlying model ---
            hidden_state_k_plus_1, reward_logits = network.model.dynamics(
                hidden_state_k_batch, action
            )
            # ---

            hidden_state_k_plus_1 = hidden_state_k_plus_1.squeeze(0)
            reward_logits = reward_logits.squeeze(0)
            reward_k_plus_1 = network._logits_to_scalar(
                reward_logits.unsqueeze(0), network.reward_support
            ).item()

        except Exception as e:
            logger.error(
                f"[Expand] Error calling dynamics for action {action}: {e}",
                exc_info=True,
            )
            continue

        child = Node(
            prior=prior,
            hidden_state=hidden_state_k_plus_1,
            reward=reward_k_plus_1,
            parent=node_to_expand,
            action_taken=action,
        )
        node_to_expand.children[action] = child
        # ... (logging remains the same) ...
        children_created += 1

    logger.debug(f"[Expand] Node expanded with {children_created} children.")
