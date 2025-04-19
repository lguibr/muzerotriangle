# File: muzerotriangle/mcts/core/search.py
import logging
from typing import TYPE_CHECKING

from ...config import MCTSConfig
from ...features import extract_state_features
from ...utils.types import ActionType, StateType
from ..strategy import backpropagation, expansion, selection
from ..strategy.backpropagation import backpropagate_value
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
    """
    Runs MuZero MCTS simulations starting from the root_node.

    Args:
        root_node: The root node of the search, containing the current state representation.
                   If unexpanded, it should contain the initial_game_state.
        config: MCTS configuration parameters.
        network: The neural network interface for evaluations.
        valid_actions_from_state: A list of valid actions from the root state.

    Returns:
        int: The maximum search depth reached during the simulations.
    """
    if root_node.initial_game_state is None and root_node.hidden_state is None:
        raise MCTSExecutionError("Root node needs initial_game_state or hidden_state.")
    if (
        root_node.initial_game_state is not None
        and root_node.initial_game_state.is_over()
    ):
        logger.debug("[MCTS] Root node is terminal. Skipping simulations.")
        return 0

    max_depth_overall = 0
    sim_success_count = 0
    sim_error_count = 0
    initial_backprop_done = False

    # Initial Root Inference and Expansion (if root is unexpanded)
    if not root_node.is_expanded:
        logger.debug("[MCTS] Root node not expanded. Performing initial inference...")
        if root_node.initial_game_state is None:
            raise MCTSExecutionError(
                "Unexpanded root node requires initial_game_state for inference."
            )
        try:
            state_dict: StateType = extract_state_features(
                root_node.initial_game_state, network.model_config
            )
            (
                policy_logits,
                value_logits,
                _,
                initial_hidden_state,
            ) = network.initial_inference(
                state_dict  # type: ignore[arg-type]
            )

            root_node.hidden_state = initial_hidden_state.squeeze(0).detach()
            policy_probs = (
                network._logits_to_probs(policy_logits)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            root_node.predicted_value = network._logits_to_scalar(
                value_logits, network.support
            ).item()
            logger.debug(
                f"[MCTS] Initial inference complete. Root predicted value: {root_node.predicted_value:.4f}"
            )

            full_policy_map = {i: float(p) for i, p in enumerate(policy_probs)}
            filtered_policy_map = {
                action: full_policy_map.get(action, 0.0)
                for action in valid_actions_from_state
            }

            expansion.expand_node(
                root_node,
                filtered_policy_map,
                network,
                valid_actions_from_state,
            )

            # Backpropagate the initial root value prediction
            if root_node.predicted_value is not None:
                logger.debug("[MCTS] Backpropagating initial root value...")
                depth_bp = backpropagate_value(
                    root_node, root_node.predicted_value, config.discount
                )
                max_depth_overall = max(max_depth_overall, depth_bp)
                initial_backprop_done = True  # Mark that the +1 visit occurred
                logger.debug(
                    f"[MCTS] Initial backprop complete. Depth: {depth_bp}, Root visits: {root_node.visit_count}"
                )
            if root_node.children:
                logger.debug("[MCTS] Adding Dirichlet noise to root priors...")
                selection.add_dirichlet_noise(root_node, config)
            else:
                logger.warning("[MCTS] Root node has no children after expansion!")
        except Exception as e:
            logger.error(
                f"Error during MCTS initial inference/expansion: {e}", exc_info=True
            )
            raise MCTSExecutionError(
                f"Initial root inference/expansion failed: {e}"
            ) from e
    elif root_node.visit_count == 0 and root_node.children:
        # If root was expanded previously but has 0 visits (e.g., loaded checkpoint), add noise
        logger.debug("[MCTS] Root expanded but unvisited. Adding Dirichlet noise...")
        selection.add_dirichlet_noise(root_node, config)

    # Simulation Loop
    for sim in range(config.num_simulations):
        logger.debug(
            f"[MCTS Sim {sim + 1}/{config.num_simulations}] Starting simulation..."
        )
        current_node = root_node
        search_path = [current_node]
        depth = 0
        try:
            # Selection Phase
            logger.debug(f"[MCTS Sim {sim + 1}] --- Selection Phase ---")
            while current_node.is_expanded:
                if (
                    config.max_search_depth is not None
                    and depth >= config.max_search_depth
                ):
                    logger.debug(
                        f"[MCTS Sim {sim + 1}] Max search depth ({config.max_search_depth}) reached at depth {depth}. Stopping selection."
                    )
                    break
                selected_child = selection.select_child_node(current_node, config)
                logger.debug(
                    f"[MCTS Sim {sim + 1}] Selected child action: {selected_child.action_taken} (Depth {depth + 1})"
                )
                current_node = selected_child
                search_path.append(current_node)
                depth += 1
            leaf_node = current_node
            max_depth_overall = max(max_depth_overall, depth)
            leaf_action = (
                leaf_node.action_taken if leaf_node.action_taken is not None else "Root"
            )
            logger.debug(
                f"[MCTS Sim {sim + 1}] Selected Leaf Node (Action={leaf_action}, Depth={depth}, Expanded={leaf_node.is_expanded}, Visits={leaf_node.visit_count})"
            )

            # Check if leaf is terminal (based on game state if root, or potentially network prediction if needed)
            is_terminal_leaf = False
            if (
                leaf_node.is_root
                and leaf_node.initial_game_state is not None
                and not leaf_node.initial_game_state.valid_actions()
            ):
                is_terminal_leaf = True
                logger.debug(
                    f"[MCTS Sim {sim + 1}] Leaf node is terminal (root state)."
                )

            # Expansion & Prediction Phase
            logger.debug(f"[MCTS Sim {sim + 1}] --- Expansion/Prediction Phase ---")
            value_for_backprop = 0.0
            if not is_terminal_leaf and leaf_node.hidden_state is not None:
                logger.debug(
                    f"[MCTS Sim {sim + 1}] Leaf not terminal. Performing recurrent inference..."
                )
                hidden_state_batch = leaf_node.hidden_state.to(
                    network.device
                ).unsqueeze(0)
                policy_logits, value_logits = network.model.predict(hidden_state_batch)

                leaf_node.predicted_value = network._logits_to_scalar(
                    value_logits, network.support
                ).item()
                value_for_backprop = leaf_node.predicted_value
                logger.debug(
                    f"[MCTS Sim {sim + 1}] Leaf predicted value: {value_for_backprop:.4f}"
                )
                policy_probs = (
                    network._logits_to_probs(policy_logits)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                policy_map = {i: float(p) for i, p in enumerate(policy_probs)}

                # Expand the leaf node
                expansion.expand_node(
                    leaf_node,
                    policy_map,
                    network,
                    valid_actions=None,  # Let expansion use policy map keys
                )
            elif leaf_node.is_root and leaf_node.predicted_value is not None:
                # If we stopped at the root (e.g., max depth 0), use its initial prediction
                value_for_backprop = leaf_node.predicted_value
                logger.debug(
                    f"[MCTS Sim {sim + 1}] Using initial root predicted value for backprop: {value_for_backprop:.4f}"
                )
            elif is_terminal_leaf:
                value_for_backprop = 0.0  # Terminal nodes have a value of 0
                logger.debug(
                    f"[MCTS Sim {sim + 1}] Using terminal value (0.0) for backprop."
                )
            else:
                # This case indicates an issue, e.g., non-root node without hidden state
                logger.error(
                    f"[MCTS Sim {sim + 1}] Leaf node state invalid: is_root={leaf_node.is_root}, hidden_state is None, is_terminal={is_terminal_leaf}. Using 0.0 for backprop."
                )
                value_for_backprop = 0.0

            # Backpropagation Phase
            logger.debug(f"[MCTS Sim {sim + 1}] --- Backpropagation Phase ---")
            depth_bp = backpropagation.backpropagate_value(
                leaf_node, value_for_backprop, config.discount
            )
            logger.debug(
                f"[MCTS Sim {sim + 1}] Backpropagation complete. Path depth: {depth_bp}"
            )
            sim_success_count += 1
        except Exception as e:
            sim_error_count += 1
            logger.error(f"Error in simulation {sim + 1}: {e}", exc_info=True)
            # Optionally break or continue depending on desired robustness
            # break

    logger.debug("[MCTS] Finished simulations. Final root children stats:")
    if root_node.children:
        for action, child in sorted(root_node.children.items()):
            logger.debug(
                f"  Child Action={action}: Visits={child.visit_count}, Q={child.value_estimate:.4f}, P={child.prior_probability:.4f}"
            )
    else:
        logger.debug("  Root node has no children after simulations.")

    if sim_error_count > config.num_simulations * 0.1:
        logger.warning(
            f"MCTS completed with {sim_error_count} errors out of {config.num_simulations} simulations."
        )

    # Note: Root visit count will be num_simulations + 1 if the root was initially unexpanded
    # because of the initial backpropagation of the root's predicted value.
    # If the root was already expanded (e.g., tree reuse), visit count would be num_simulations.
    expected_visits = config.num_simulations + (1 if initial_backprop_done else 0)
    logger.debug(
        f"[MCTS] Completed {sim_success_count} simulations. Max depth reached: {max_depth_overall}. Root visits: {root_node.visit_count} (Expected ~{expected_visits})"
    )
    return max_depth_overall
