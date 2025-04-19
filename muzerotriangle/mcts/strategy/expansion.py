# File: muzerotriangle/mcts/strategy/expansion.py
import logging
from typing import TYPE_CHECKING, List, Optional

import torch  # Import torch for tensor check

from ..core.node import Node
from ..core.types import ActionPolicyMapping

if TYPE_CHECKING:
    # Use NeuralNetwork interface type hint
    from muzerotriangle.nn import NeuralNetwork
    from muzerotriangle.utils.types import ActionType


logger = logging.getLogger(__name__)


def expand_node(
    node_to_expand: "Node",  # Node containing s_k
    policy_prediction: ActionPolicyMapping,  # Policy p_k from f(s_k)
    network: "NeuralNetwork",  # Network interface to call dynamics (g)
    valid_actions: Optional[List["ActionType"]] = None,  # Pass valid actions explicitly
):
    """
    Expands a leaf node in the MuZero search tree.
    If valid_actions is provided, only those actions are considered for expansion.
    Otherwise, actions with non-zero prior probability in policy_prediction are considered.
    """
    if node_to_expand.is_expanded:
        logger.debug(f"Node {node_to_expand.action_taken} already expanded. Skipping.")
        return
    hidden_state_k = node_to_expand.hidden_state
    if hidden_state_k is None:
        logger.error(
            f"[Expand] Node {node_to_expand.action_taken} has no hidden state."
        )
        return

    logger.debug(f"[Expand] Expanding Node via action: {node_to_expand.action_taken}")

    # --- MODIFIED: Determine actions to expand based on valid_actions first ---
    if valid_actions is not None:
        # If valid_actions are provided (e.g., for root), only expand these
        actions_to_expand_set = set(valid_actions)
        if not actions_to_expand_set:
            logger.warning(
                f"[Expand] No valid actions provided for node {node_to_expand.action_taken}. Node will remain unexpanded."
            )
            return  # Do not create children if valid_actions is empty
    else:
        # If valid_actions not provided, use policy prediction keys (for internal nodes)
        # Filter by non-zero prior? Optional, but can reduce unnecessary dynamics calls.
        # Let's keep expanding all actions predicted by the policy for now.
        actions_to_expand_set = set(policy_prediction.keys())
        if not actions_to_expand_set:
            logger.warning(
                f"[Expand] No actions found in policy prediction for node {node_to_expand.action_taken}. Node will remain unexpanded."
            )
            return
    # --- END MODIFIED ---

    children_created = 0
    for action in actions_to_expand_set:
        # Get prior probability for this action from the prediction
        prior = policy_prediction.get(action, 0.0)
        if prior < 0:
            logger.warning(f"Negative prior {prior} for action {action}. Clamping.")
            prior = 0.0
        # Optional: Skip expansion if prior is zero?
        # if prior <= 1e-6:
        #     continue

        try:
            if not isinstance(hidden_state_k, torch.Tensor):
                logger.error(f"Hidden state is not a tensor: {type(hidden_state_k)}")
                continue
            # Ensure hidden state is batched for dynamics call
            hidden_state_k_batch = (
                hidden_state_k.unsqueeze(0)
                if hidden_state_k.dim() == 1
                else hidden_state_k
            )

            # Call dynamics on the underlying model
            # Ensure action is a tensor for the model
            action_tensor = torch.tensor(
                [action], dtype=torch.long, device=network.device
            )
            hidden_state_k_plus_1, reward_logits = network.model.dynamics(
                hidden_state_k_batch, action_tensor
            )

            # Remove batch dimension and calculate scalar reward
            hidden_state_k_plus_1 = hidden_state_k_plus_1.squeeze(
                0
            ).detach()  # Detach here
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

        # Create the child node
        child = Node(
            prior=prior,
            hidden_state=hidden_state_k_plus_1,
            reward=reward_k_plus_1,
            parent=node_to_expand,
            action_taken=action,
        )
        node_to_expand.children[action] = child
        logger.debug(
            f"  [Expand] Created child for action {action}: Prior={prior:.4f}, Reward={reward_k_plus_1:.3f}"
        )
        children_created += 1

    logger.debug(f"[Expand] Node expanded with {children_created} children.")
