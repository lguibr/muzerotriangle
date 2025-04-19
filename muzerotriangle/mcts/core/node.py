# File: muzerotriangle/mcts/core/node.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from muzerotriangle.environment import GameState
    from muzerotriangle.utils.types import ActionType

logger = logging.getLogger(__name__)


class Node:
    """
    Represents a node in the Monte Carlo Search Tree for MuZero.
    Stores hidden state, predicted reward, and MCTS statistics.
    The root node holds the actual GameState.
    """

    def __init__(
        self,
        prior: float = 0.0,
        # --- MuZero Specific ---
        hidden_state: torch.Tensor | None = None,  # Stores s_k
        reward: float = 0.0,  # Stores r_k (predicted reward to reach this state)
        initial_game_state: GameState | None = None,  # Only for root node
        # --- Common MCTS ---
        parent: Node | None = None,
        action_taken: ActionType | None = None,  # Action a_k that led to this state s_k
    ):
        self.parent = parent
        self.action_taken = action_taken
        self.prior_probability = prior

        # State Representation
        self.hidden_state = hidden_state  # Tensor representing the state s_k
        self.initial_game_state = initial_game_state  # For root node observation
        self.reward = reward  # Predicted reward r_k from g() to reach this state

        # MCTS Statistics
        self.visit_count: int = 0
        self.value_sum: float = (
            0.0  # Sum of backed-up values (G_i) from simulations passing through here
        )
        self.children: dict[ActionType, Node] = {}

        # --- Cached values from prediction function f(s_k) ---
        # These are calculated when the node is expanded or selected
        self.predicted_value: float | None = None  # Cached v_k

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_expanded(self) -> bool:
        """Checks if the node has children."""
        return bool(self.children)

    @property
    def value_estimate(self) -> float:
        """
        Calculates the Q-value estimate Q(s,a) for the *action* leading to this node.
        Average of values G backpropagated through this node.
        Returns 0 if unvisited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_state(self) -> torch.Tensor | GameState:
        """Returns the representation of the state (hidden_state or GameState for root)."""
        if self.is_root and self.initial_game_state is not None:
            return self.initial_game_state
        elif self.hidden_state is not None:
            return self.hidden_state
        else:
            # This should ideally not happen for a non-root node after expansion
            raise ValueError(
                "Node state is missing (neither initial_game_state nor hidden_state is set)."
            )

    def __repr__(self) -> str:
        state_desc = (
            f"Root(Step={self.initial_game_state.current_step})"
            if self.is_root and self.initial_game_state
            else f"HiddenState(shape={self.hidden_state.shape if self.hidden_state is not None else 'None'})"
        )
        return (
            f"Node(Action={self.action_taken}, State={state_desc}, "
            f"Reward={self.reward:.2f}, Visits={self.visit_count}, "
            f"Value={self.value_estimate:.3f}, Prior={self.prior_probability:.4f}, "
            f"Children={len(self.children)})"
        )
