# File: muzerotriangle/mcts/strategy/selection.py
import logging
import math
import random

import numpy as np

from ...config import MCTSConfig
from ..core.node import Node

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class SelectionError(Exception):
    """Custom exception for errors during node selection."""

    pass


def calculate_puct_score(
    parent_node: "Node",
    child_node: "Node",
    config: MCTSConfig,
) -> tuple[float, float, float]:
    """
    Calculates the PUCT score for a child node, used for selection from the parent.
    Score = Q(parent, action_to_child) + U(parent, action_to_child)
    Uses parent's value estimate for unvisited children (Value Initialization / FPU).
    """
    # --- Value Initialization / FPU ---
    if child_node.visit_count == 0:
        # Use parent's value estimate as initial Q for unvisited children.
        # If parent is root and also unvisited (only during initial call before backprop), use 0.
        q_value = parent_node.value_estimate if parent_node.visit_count > 0 else 0.0
        # Log that FPU/Value Init is being used
        logger.debug(
            f"      PUCT Calc (FPU): Action={child_node.action_taken}, Using Parent Q={q_value:.4f} (Child Visits=0)"
        )
    else:
        q_value = child_node.value_estimate
    # --- End Value Initialization ---

    prior = child_node.prior_probability
    parent_visits = parent_node.visit_count
    child_visits = child_node.visit_count

    # Ensure parent_visits is at least 1 for the sqrt calculation to avoid issues at root
    parent_visits_adjusted = max(1, parent_visits)

    # Use adjusted count for sqrt
    parent_sqrt_term = math.sqrt(parent_visits_adjusted)

    exploration_term = (
        config.puct_coefficient * prior * (parent_sqrt_term / (1 + child_visits))
    )
    score = q_value + exploration_term

    if not np.isfinite(score):
        logger.warning(
            f"Non-finite PUCT score calculated: Q={q_value}, P={prior}, Np={parent_visits}, Nc={child_visits}. Resetting exploration term to 0."
        )
        score = q_value
        exploration_term = 0.0

    # Add detailed logging for Q and U terms
    logger.debug(
        f"      PUCT Calc: Action={child_node.action_taken}, Q={q_value:.4f}, P={prior:.4f}, Np={parent_visits}(adj={parent_visits_adjusted}), Nc={child_visits} -> U={exploration_term:.4f}, Score={score:.4f}"
    )

    return score, q_value, exploration_term


def add_dirichlet_noise(node: Node, config: MCTSConfig):
    """Adds Dirichlet noise to the prior probabilities of the children of this node."""
    if (
        config.dirichlet_alpha <= 0.0
        or config.dirichlet_epsilon <= 0.0
        or not node.children
        or len(node.children) <= 1
    ):
        return

    actions = list(node.children.keys())
    noise = rng.dirichlet([config.dirichlet_alpha] * len(actions))
    eps = config.dirichlet_epsilon

    noisy_priors_sum = 0.0
    logger.debug(f"  [Noise] Adding noise (alpha={config.dirichlet_alpha}, eps={eps})")
    for i, action in enumerate(actions):
        child = node.children[action]
        original_prior = child.prior_probability
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]
        noisy_priors_sum += child.prior_probability
        logger.debug(
            f"    Action {action}: P_orig={original_prior:.4f}, Noise={noise[i]:.4f} -> P_new={child.prior_probability:.4f}"
        )

    # Normalize if sum is not close to 1
    if abs(noisy_priors_sum - 1.0) > 1e-6:
        norm_factor = noisy_priors_sum if noisy_priors_sum > 1e-9 else 1.0
        logger.warning(
            f"  [Noise] Priors sum to {noisy_priors_sum:.6f} after noise. Normalizing."
        )
        if norm_factor > 1e-9:
            for action in actions:
                node.children[action].prior_probability /= norm_factor
        else:
            logger.warning(
                "  [Noise] Sum of priors after noise is near zero. Resetting to uniform."
            )
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                node.children[action].prior_probability = uniform_prob


def select_child_node(node: Node, config: MCTSConfig) -> Node:
    """
    Selects the child node with the highest PUCT score. Assumes noise already added if root.
    Raises SelectionError if no valid child can be selected.
    """
    if not node.children:
        raise SelectionError(f"Cannot select child from node {node} with no children.")

    best_score = -float("inf")
    best_children: list[Node] = []

    parent_action = node.action_taken if node.action_taken is not None else "Root"
    # Log parent visit count before loop
    logger.debug(
        f"  [SelectChild] Parent Node (Action={parent_action}, Visits={node.visit_count}, Value={node.value_estimate:.4f}, Children={len(node.children)}): Calculating PUCT scores..."
    )

    for action, child in node.children.items():
        score, q, exp_term = calculate_puct_score(node, child, config)

        # Log entry moved inside calculate_puct_score

        if not np.isfinite(score):
            logger.warning(
                f"    [SelectChild] Non-finite PUCT score ({score}) calculated for child action {action}. Skipping."
            )
            continue

        if score > best_score:
            best_score = score
            best_children = [child]
        elif abs(score - best_score) < 1e-9:  # Handle ties by collecting all best
            best_children.append(child)

    if not best_children:
        child_details = [
            f"Act={a}, N={c.visit_count}, P={c.prior_probability:.4f}, Q={c.value_estimate:.3f}"
            for a, c in node.children.items()
        ]
        state_info = (
            f"Root Step {node.initial_game_state.current_step}"
            if node.is_root and node.initial_game_state
            else f"Node Action {node.action_taken}"
        )
        logger.error(
            f"Could not select best child for {state_info}. Child details: {child_details}"
        )
        if not node.children:
            raise SelectionError(
                f"Cannot select child from node {state_info} with no children (should have been caught earlier)."
            )
        logger.warning(
            f"All child scores were non-finite or no children found. Selecting a random child for node {state_info}."
        )
        if not node.children:
            raise SelectionError(
                f"Node {state_info} has no children to select from, even randomly."
            )
        selected_child = random.choice(list(node.children.values()))
    else:
        # Break ties randomly among the best children
        selected_child = random.choice(best_children)

    logger.debug(
        f"  [SelectChild] --> Selected Child: Action {selected_child.action_taken}, Score {best_score:.4f}"
    )
    return selected_child


def traverse_to_leaf(root_node: Node, config: MCTSConfig) -> tuple[Node, int]:
    """
    Traverses the tree from root to a leaf node using PUCT selection.
    A leaf is defined as a node that has not been expanded.
    Stops also if the maximum search depth has been reached.
    Note: Terminal state check now happens during expansion/prediction.
    Raises SelectionError if child selection fails during traversal.
    Returns the leaf node and the depth reached.
    """
    current_node = root_node
    depth = 0
    stop_reason = "Unknown"
    logger.debug(f"  [Traverse] Starting traversal from node: {current_node}")

    while current_node.is_expanded:
        logger.debug(
            f"  [Traverse] Depth {depth}: Node Action={current_node.action_taken if current_node.action_taken is not None else 'Root'} (Visits={current_node.visit_count}) is expanded. Selecting child..."
        )
        if config.max_search_depth is not None and depth >= config.max_search_depth:
            stop_reason = f"Max Depth Reached ({config.max_search_depth})"
            logger.debug(f"  [Traverse] Stop reason: {stop_reason}")
            break

        try:
            selected_child = select_child_node(current_node, config)
            current_node = selected_child
            depth += 1
            logger.debug(
                f"  [Traverse] Depth {depth}: Moved to child node {current_node.action_taken}"
            )
        except SelectionError as e:
            stop_reason = f"Child Selection Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Error during child selection: {e}. Breaking traverse.",
                exc_info=False,
            )
            logger.warning(
                f"  [Traverse] Returning current node {current_node.action_taken} due to SelectionError."
            )
            break
        except Exception as e:
            stop_reason = f"Unexpected Error: {e}"
            logger.error(
                f"  [Traverse] Depth {depth}: Unexpected error during child selection: {e}. Breaking traverse.",
                exc_info=True,
            )
            logger.warning(
                f"  [Traverse] Returning current node {current_node.action_taken} due to Unexpected Error."
            )
            break
    else:
        stop_reason = "Unexpanded Leaf"
        logger.debug(f"  [Traverse] Stop reason: {stop_reason}")

    logger.debug(
        f"  [Traverse] Finished traversal. Leaf: {current_node.action_taken}, Depth: {depth}, Stop Reason: {stop_reason}"
    )
    return current_node, depth
