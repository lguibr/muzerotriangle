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
    parent_node: "Node",  # The node *from* which we are selecting a child
    child_node: "Node",  # The child node being evaluated
    config: MCTSConfig,
) -> tuple[float, float, float]:
    """
    Calculates the PUCT score for a child node, used for selection from the parent.
    Score = Q(parent, action_to_child) + U(parent, action_to_child)
    Q value is the average value derived from simulations passing through the child.
    U value is the exploration bonus based on the child's prior and visit counts.
    """
    # Q(s, a) is the value estimate of the child node itself
    # It represents the expected return *after* taking 'action_taken' from the parent.
    q_value = child_node.value_estimate

    # P(a|s) is the prior probability stored in the child node
    prior = child_node.prior_probability
    parent_visits = parent_node.visit_count
    child_visits = child_node.visit_count

    # Exploration bonus U(s, a)
    # Use max(1, parent_visits) to avoid math domain error if parent_visits is 0 (though it shouldn't be if we're selecting a child)
    exploration_term = (
        config.puct_coefficient
        * prior
        * (math.sqrt(max(1, parent_visits)) / (1 + child_visits))
    )
    score = q_value + exploration_term

    # Ensure score is finite
    if not np.isfinite(score):
        logger.warning(
            f"Non-finite PUCT score calculated (Q={q_value}, P={prior}, ChildN={child_visits}, ParentN={parent_visits}, Exp={exploration_term}). Defaulting to Q-value."
        )
        score = q_value
        exploration_term = 0.0

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
    for i, action in enumerate(actions):
        child = node.children[action]
        original_prior = child.prior_probability
        child.prior_probability = (1 - eps) * child.prior_probability + eps * noise[i]
        noisy_priors_sum += child.prior_probability
        logger.debug(
            f"  [Noise] Action {action}: OrigP={original_prior:.4f}, Noise={noise[i]:.4f} -> NewP={child.prior_probability:.4f}"
        )

    # Re-normalize priors
    if abs(noisy_priors_sum - 1.0) > 1e-6:
        logger.debug(
            f"Re-normalizing priors after Dirichlet noise (Sum={noisy_priors_sum:.6f})"
        )
        norm_factor = noisy_priors_sum if noisy_priors_sum > 1e-9 else 1.0
        if norm_factor > 1e-9:
            for action in actions:
                node.children[action].prior_probability /= norm_factor
        else:
            # If sum is zero (e.g., all priors were zero), distribute uniformly
            logger.warning(
                "Sum of priors after noise is near zero. Resetting to uniform."
            )
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                node.children[action].prior_probability = uniform_prob

    logger.debug(
        f"[Noise] Added Dirichlet noise (alpha={config.dirichlet_alpha}, eps={eps}) to {len(actions)} root node priors."
    )


def select_child_node(node: Node, config: MCTSConfig) -> Node:
    """
    Selects the child node with the highest PUCT score. Assumes noise already added if root.
    Raises SelectionError if no valid child can be selected.
    """
    if not node.children:
        raise SelectionError(f"Cannot select child from node {node} with no children.")

    best_score = -float("inf")
    best_children: list[Node] = []  # Use a list for tie-breaking

    if logger.isEnabledFor(logging.DEBUG):
        # Check if root node to display step correctly
        state_info = (
            f"Step={node.initial_game_state.current_step}"
            if node.is_root and node.initial_game_state
            else f"Action={node.action_taken}"
        )
        logger.debug(
            f"  [Select] Selecting child for Node (Visits={node.visit_count}, Children={len(node.children)}, {state_info}):"
        )

    for action, child in node.children.items():
        # Pass the parent (current node) and the child being evaluated
        score, q, exp_term = calculate_puct_score(node, child, config)

        if logger.isEnabledFor(logging.DEBUG):
            log_entry = (
                f"    Act={action}, Score={score:.4f} "
                f"(Q={q:.3f}, P={child.prior_probability:.4f}, N={child.visit_count}, Exp={exp_term:.4f})"
            )
            logger.debug(log_entry)  # Log each child's score

        if not np.isfinite(score):
            logger.warning(
                f"    [Select] Non-finite PUCT score ({score}) calculated for child action {action}. Skipping."
            )
            continue

        if score > best_score:
            best_score = score
            best_children = [child]  # Start a new list with this best child
        # Tie-breaking: if scores are equal, add to the list
        elif abs(score - best_score) < 1e-9:  # Use tolerance for float comparison
            best_children.append(child)

    if not best_children:  # Changed from checking best_child is None
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
        # Fallback: if all scores are non-finite or no children found, pick a random child
        if not node.children:
            raise SelectionError(
                f"Cannot select child from node {node} with no children (should have been caught earlier)."
            )
        logger.warning(
            f"All child scores were non-finite or no children found. Selecting a random child for node {state_info}."
        )
        # Ensure we actually have children before choosing randomly
        if not node.children:
            raise SelectionError(
                f"Node {state_info} has no children to select from, even randomly."
            )
        selected_child = random.choice(list(node.children.values()))
    else:
        # If there are ties, select randomly among the best
        selected_child = random.choice(best_children)

    logger.debug(
        f"  [Select] --> Selected Child: Action {selected_child.action_taken}, Score {best_score:.4f}, Q-value {selected_child.value_estimate:.3f}"
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
    state_info = (
        f"Root Step {root_node.initial_game_state.current_step}"
        if root_node.is_root and root_node.initial_game_state
        else f"Node Action {root_node.action_taken}"
    )
    logger.debug(f"[Traverse] --- Start Traverse (Start Node: {state_info}) ---")
    stop_reason = "Unknown"

    while current_node.is_expanded:  # Traverse while node has children
        state_info = (
            f"Root Step {current_node.initial_game_state.current_step}"
            if current_node.is_root and current_node.initial_game_state
            else f"Node Action {current_node.action_taken}"
        )
        logger.debug(
            f"  [Traverse] Depth {depth}: Considering Node {state_info} (Expanded={current_node.is_expanded})"
        )

        if config.max_search_depth is not None and depth >= config.max_search_depth:
            stop_reason = "Max Depth Reached"
            logger.debug(
                f"  [Traverse] Depth {depth}: Hit MAX DEPTH ({config.max_search_depth}). Stopping traverse."
            )
            break

        # Node is expanded and below max depth - select child
        try:
            selected_child = select_child_node(current_node, config)
            logger.debug(
                f"  [Traverse] Depth {depth}: Selected child with action {selected_child.action_taken}"
            )
            current_node = selected_child
            depth += 1
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
        # Loop finished because node is not expanded (it's a leaf)
        stop_reason = "Unexpanded Leaf"
        state_info = (
            f"Root Step {current_node.initial_game_state.current_step}"
            if current_node.is_root and current_node.initial_game_state
            else f"Node Action {current_node.action_taken}"
        )
        logger.debug(
            f"  [Traverse] Depth {depth}: Node {state_info} is LEAF (not expanded). Stopping traverse."
        )

    state_info_final = (
        f"Root Step {current_node.initial_game_state.current_step}"
        if current_node.is_root and current_node.initial_game_state
        else f"Node Action {current_node.action_taken}"
    )
    logger.debug(
        f"[Traverse] --- End Traverse: Reached Node {state_info_final} at Depth {depth}. Reason: {stop_reason} ---"
    )
    return current_node, depth
