# File: muzerotriangle/mcts/strategy/expansion.py
import logging
from typing import TYPE_CHECKING

import torch

from ..core.node import Node
from ..core.types import ActionPolicyMapping

if TYPE_CHECKING:
    from muzerotriangle.nn import NeuralNetwork
    from muzerotriangle.utils.types import ActionType


logger = logging.getLogger(__name__)


def expand_node(
    node_to_expand: "Node",
    policy_prediction: ActionPolicyMapping,
    network: "NeuralNetwork",
    valid_actions: list["ActionType"] | None = None,
):
    """
    Expands a leaf node in the MuZero search tree.
    If valid_actions is provided, only those actions are considered for expansion.
    Otherwise, actions with non-zero prior probability in policy_prediction are considered.
    """
    parent_action = (
        node_to_expand.action_taken
        if node_to_expand.action_taken is not None
        else "Root"
    )
    depth = 0
    temp_node = node_to_expand
    while temp_node.parent:
        depth += 1
        temp_node = temp_node.parent

    logger.debug(
        f"[Expand] Attempting expansion for Node (Action={parent_action}, Depth={depth}, Visits={node_to_expand.visit_count})"
    )

    if node_to_expand.is_expanded:
        logger.debug(
            f"[Expand] Node (Action={parent_action}) already expanded. Skipping."
        )
        return
    hidden_state_k = node_to_expand.hidden_state
    if hidden_state_k is None:
        logger.error(
            f"[Expand] Node (Action={parent_action}) has no hidden state. Cannot expand."
        )
        return

    if valid_actions is not None:
        actions_to_expand_set = set(valid_actions)
        logger.debug(
            f"[Expand] Using provided valid_actions: {len(actions_to_expand_set)} actions."
        )
        if not actions_to_expand_set:
            logger.warning(
                f"[Expand] No valid actions provided for node {parent_action}. Node will remain unexpanded."
            )
            return
    else:
        # Expand based on policy prediction keys (actions with non-zero prior)
        actions_to_expand_set = {
            a for a, p in policy_prediction.items() if p > 0
        }  # Filter zero priors
        # Log the policy map being used
        policy_log_str = ", ".join(
            f"{a}:{p:.3f}" for a, p in sorted(policy_prediction.items()) if p > 1e-4
        )
        logger.debug(
            f"[Expand] Using policy prediction keys ({len(actions_to_expand_set)} actions with >0 prior). Policy: {{{policy_log_str}}}"
        )
        if not actions_to_expand_set:
            logger.warning(
                f"[Expand] No actions found with non-zero prior in policy prediction for node {parent_action}. Node will remain unexpanded."
            )
            return

    children_created = 0
    for action in actions_to_expand_set:
        prior = policy_prediction.get(action, 0.0)
        if prior < 0:
            logger.warning(
                f"[Expand] Negative prior {prior} for action {action}. Clamping to 0."
            )
            prior = 0.0
        # Skip actions with zero prior unless they were explicitly provided as valid
        if prior == 0.0 and valid_actions is None:
            continue

        logger.debug(
            f"  [Expand Child] Action={action}, Prior={prior:.4f}. Calling dynamics..."
        )
        try:
            if not isinstance(hidden_state_k, torch.Tensor):
                logger.error(
                    f"  [Expand Child] Hidden state is not a tensor: {type(hidden_state_k)}"
                )
                continue
            hidden_state_k_batch = (
                hidden_state_k.unsqueeze(0)
                if hidden_state_k.dim() == 1
                else hidden_state_k
            )

            action_tensor = torch.tensor(
                [action], dtype=torch.long, device=network.device
            )
            hidden_state_k_plus_1, reward_logits = network.model.dynamics(
                hidden_state_k_batch, action_tensor
            )

            hidden_state_k_plus_1 = hidden_state_k_plus_1.squeeze(0).detach()
            reward_logits = reward_logits.squeeze(0)
            reward_k_plus_1 = network._logits_to_scalar(
                reward_logits.unsqueeze(0), network.reward_support
            ).item()
            logger.debug(
                f"  [Expand Child] Dynamics success. Action={action}, Reward={reward_k_plus_1:.3f}"
            )

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
        children_created += 1

    logger.debug(
        f"[Expand] Node (Action={parent_action}, Depth={depth}) finished expansion. Children created: {children_created}. Total children: {len(node_to_expand.children)}. IsExpanded: {node_to_expand.is_expanded}"
    )
