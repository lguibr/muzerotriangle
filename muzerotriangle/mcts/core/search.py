# File: muzerotriangle/mcts/core/search.py
import logging
from typing import TYPE_CHECKING, List

import torch
import numpy as np  # Ensure numpy is imported

from ...config import MCTSConfig
from ...features import extract_state_features  # Import feature extractor
from ...utils.types import ActionType, StateType  # Import StateType
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
    valid_actions_from_state: List[ActionType],
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
            # Extract features first, pass StateType dict
            state_dict: StateType = extract_state_features(
                root_node.initial_game_state, network.model_config
            )
            # Ensure tensors are on the correct device
            # initial_inference expects StateType dict with numpy arrays
            # The conversion to tensor happens inside initial_inference
            (
                policy_logits,
                value_logits,
                _,
                initial_hidden_state,
            ) = network.initial_inference(
                state_dict  # type: ignore[arg-type] # MyPy seems confused here
            )

            root_node.hidden_state = initial_hidden_state.squeeze(
                0
            ).detach()  # Detach here
            policy_probs = (
                network._logits_to_probs(policy_logits)
                .squeeze(0)
                .detach()  # Detach here
                .cpu()
                .numpy()
            )
            root_node.predicted_value = network._logits_to_scalar(
                value_logits, network.support
            ).item()

            # Filter policy map by valid actions for root
            full_policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
            filtered_policy_map = {
                action: full_policy_map.get(action, 0.0)
                for action in valid_actions_from_state
            }

            # Pass valid actions AND the filtered policy map for the initial root expansion
            expansion.expand_node(
                root_node,
                filtered_policy_map,  # Pass filtered map
                network,
                valid_actions_from_state,  # Pass valid actions explicitly
            )

            if root_node.predicted_value is not None:
                depth_bp = backpropagation.backpropagate_value(
                    root_node, root_node.predicted_value, config.discount
                )
                max_depth_overall = max(max_depth_overall, depth_bp)
            # Apply noise only if children were actually created
            if root_node.children:
                selection.add_dirichlet_noise(root_node, config)
        except Exception as e:
            raise MCTSExecutionError(f"Initial root inference failed: {e}") from e
    elif (
        root_node.visit_count == 0 and root_node.children
    ):  # Apply noise only if children exist
        # Apply noise only if root hasn't been visited yet in this MCTS run
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

            # --- ADDED CHECK: Don't expand if leaf represents a terminal state ---
            # We can only reliably check this for the root node in MuZero MCTS
            is_terminal_leaf = False
            if leaf_node.is_root and leaf_node.initial_game_state is not None:
                # Check if the game state has no valid actions
                if not leaf_node.initial_game_state.valid_actions():
                    is_terminal_leaf = True
                    logger.debug(
                        f"Leaf node (root) has no valid actions. Treating as terminal."
                    )
            # --- END ADDED CHECK ---

            # Expansion & Prediction
            value_for_backprop = 0.0
            # --- MODIFIED: Only expand/predict if not terminal ---
            if not is_terminal_leaf and leaf_node.hidden_state is not None:
                # --- END MODIFIED ---
                # Ensure hidden_state is on the correct device and has batch dim
                hidden_state_batch = leaf_node.hidden_state.to(
                    network.device
                ).unsqueeze(0)
                policy_logits, value_logits = network.model.predict(hidden_state_batch)

                leaf_node.predicted_value = network._logits_to_scalar(
                    value_logits, network.support
                ).item()
                value_for_backprop = leaf_node.predicted_value
                policy_probs = (
                    network._logits_to_probs(policy_logits)
                    .squeeze(0)
                    .detach()  # Detach here
                    .cpu()
                    .numpy()
                )
                policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
                # Pass valid_actions_from_state here as well, as the leaf node might be the root
                # or we might need to re-evaluate valid actions if the game state was available
                # For MuZero, we rely on the policy mask from the network if available,
                # otherwise, we might need the game state if the node represents a real state.
                # Since we only expand based on the policy prediction, we don't strictly need
                # valid_actions here unless we want to mask the policy further.
                # Let's assume the policy prediction already accounts for valid actions.
                expansion.expand_node(
                    leaf_node,
                    policy_map,
                    network,
                    valid_actions=None,  # Pass None here as we use the network's policy
                )
            elif (
                leaf_node.is_root and leaf_node.predicted_value is not None
            ):  # Handle root being leaf
                value_for_backprop = leaf_node.predicted_value
            # --- ADDED: Handle terminal leaf value ---
            elif is_terminal_leaf:
                # If it's terminal, the value is known (e.g., 0 or from game outcome if available)
                # For simplicity, let's use 0, assuming no outcome is readily available here.
                value_for_backprop = 0.0
                logger.debug("Using 0.0 as value for terminal leaf node.")
            # --- END ADDED ---
            else:
                # This case should ideally not happen if the game hasn't ended before expansion
                # If it's a terminal state reached during simulation, the value should be the actual outcome
                # For MuZero's pure MCTS (without explicit game state simulation),
                # we rely on the network's value prediction. If hidden_state is None here, it's an error.
                logger.error(
                    f"Leaf node state invalid: is_root={leaf_node.is_root}, hidden_state is None, is_terminal={is_terminal_leaf}"
                )
                value_for_backprop = 0.0  # Fallback, but indicates an issue

            # Backpropagation
            _ = backpropagation.backpropagate_value(
                leaf_node, value_for_backprop, config.discount
            )
            sim_success_count += 1
        except Exception as e:
            sim_error_count += 1
            logger.error(f"Error in simulation {sim + 1}: {e}", exc_info=True)
            # Optionally break or continue based on error tolerance
            # break

    if sim_error_count > config.num_simulations * 0.1:  # Allow up to 10% errors
        logger.warning(
            f"MCTS completed with {sim_error_count} errors out of {config.num_simulations} simulations."
        )
        # Decide if this should be a fatal error
        # raise MCTSExecutionError(f"MCTS failed: High error rate ({sim_error_count} errors).")

    return max_depth_overall
