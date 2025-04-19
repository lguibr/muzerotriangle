# File: muzerotriangle/mcts/core/search.py
import logging
from typing import TYPE_CHECKING

from ...config import MCTSConfig
from ...utils.types import ActionType
from ..strategy import backpropagation, expansion, selection
from .node import Node

if TYPE_CHECKING:
    from ...nn import NeuralNetwork

logger = logging.getLogger(__name__)


class MCTSExecutionError(Exception):
    pass


def run_mcts_simulations(
    root_node: Node,
    config: MCTSConfig,
    network: "NeuralNetwork",
    valid_actions_from_state: list[ActionType],
) -> int:
    """Runs MuZero MCTS simulations."""
    if root_node.initial_game_state is None:
        raise MCTSExecutionError("Root node needs initial_game_state.")
    if root_node.initial_game_state.is_over():
        return 0

    max_depth_overall = 0
    sim_success_count = 0
    sim_error_count = 0

    # Initial Root Inference and Expansion
    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node initial inference...")
        try:
            # Note: initial_inference expects StateType dict, not GameState
            state_dict = network._state_to_tensors(root_node.initial_game_state)
            policy_logits, value_logits, _, initial_hidden_state = (
                network.initial_inference(state_dict)
            )  # type: ignore

            root_node.hidden_state = initial_hidden_state.squeeze(0)
            policy_probs = (
                network._logits_to_probs(policy_logits).squeeze(0).cpu().numpy()
            )
            root_node.predicted_value = network._logits_to_scalar(
                value_logits, network.support
            ).item()
            policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
            expansion.expand_node(
                root_node, policy_map, network, valid_actions_from_state
            )

            if root_node.predicted_value is not None:
                # --- Use config.discount ---
                depth_bp = backpropagation.backpropagate_value(
                    root_node, root_node.predicted_value, config.discount
                )
                max_depth_overall = max(max_depth_overall, depth_bp)
            selection.add_dirichlet_noise(root_node, config)
        except Exception as e:
            raise MCTSExecutionError(f"Initial root inference failed: {e}") from e
    elif root_node.visit_count == 0:
        selection.add_dirichlet_noise(root_node, config)

    # Simulation Loop
    for sim in range(config.num_simulations):
        current_node = root_node
        search_path = [current_node]
        depth = 0
        try:
            # Selection
            while current_node.is_expanded:
                if (
                    config.max_search_depth is not None
                    and depth >= config.max_search_depth
                ):
                    break
                current_node = selection.select_child_node(current_node, config)
                search_path.append(current_node)
                depth += 1
            leaf_node = current_node
            max_depth_overall = max(max_depth_overall, depth)

            # Expansion & Prediction
            value_for_backprop = 0.0
            if leaf_node.hidden_state is not None:
                # --- Call predict via underlying model ---
                policy_logits, value_logits = network.model.predict(
                    leaf_node.hidden_state.unsqueeze(0)
                )
                # ---
                leaf_node.predicted_value = network._logits_to_scalar(
                    value_logits, network.support
                ).item()
                value_for_backprop = leaf_node.predicted_value
                policy_probs = (
                    network._logits_to_probs(policy_logits).squeeze(0).cpu().numpy()
                )
                policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
                expansion.expand_node(
                    leaf_node, policy_map, network, valid_actions_from_state
                )
            elif (
                leaf_node.is_root and leaf_node.predicted_value is not None
            ):  # Handle root being leaf
                value_for_backprop = leaf_node.predicted_value
            else:
                logger.error(
                    f"Leaf node state invalid: is_root={leaf_node.is_root}, hidden_state={leaf_node.hidden_state is None}"
                )

            # Backpropagation
            # --- Use config.discount ---
            _ = backpropagation.backpropagate_value(
                leaf_node, value_for_backprop, config.discount
            )
            sim_success_count += 1
        except Exception as e:
            sim_error_count += 1
            logger.error(f"Error in simulation {sim + 1}: {e}", exc_info=True)

    # Final Logging (remains the same)
    # ... (logging code) ...
    if sim_error_count > config.num_simulations * 0.1:
        raise MCTSExecutionError(
            f"MCTS failed: High error rate ({sim_error_count} errors)."
        )

    return max_depth_overall
