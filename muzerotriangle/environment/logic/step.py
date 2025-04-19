# File: muzerotriangle/environment/logic/step.py
import logging
import random
from typing import TYPE_CHECKING

# Correct import path for constants
from ...structs.constants import COLOR_TO_ID_MAP, NO_COLOR_ID
from .. import shapes as ShapeLogic

# Import the logic submodule correctly
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ...config import EnvConfig
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def calculate_reward(
    placed_count: int,
    unique_coords_cleared: set[tuple[int, int]],
    is_game_over: bool,
    config: "EnvConfig",
) -> float:
    """
    Calculates the step reward based on the new specification (v3).

    Args:
        placed_count: Number of triangles successfully placed.
        unique_coords_cleared: Set of unique (r, c) coordinates cleared this step.
        is_game_over: Boolean indicating if the game ended *after* this step.
        config: Environment configuration containing reward constants.

    Returns:
        The calculated step reward.
    """
    reward = 0.0

    # 1. Placement Reward
    reward += placed_count * config.REWARD_PER_PLACED_TRIANGLE

    # 2. Line Clear Reward
    reward += len(unique_coords_cleared) * config.REWARD_PER_CLEARED_TRIANGLE

    # 3. Survival Reward OR Game Over Penalty
    if is_game_over:
        reward += config.PENALTY_GAME_OVER
    else:
        reward += config.REWARD_PER_STEP_ALIVE

    logger.debug(
        f"Calculated Reward: Placement({placed_count * config.REWARD_PER_PLACED_TRIANGLE:.3f}) "
        f"+ LineClear({len(unique_coords_cleared) * config.REWARD_PER_CLEARED_TRIANGLE:.3f}) "
        f"+ {'GameOver' if is_game_over else 'Survival'}({config.PENALTY_GAME_OVER if is_game_over else config.REWARD_PER_STEP_ALIVE:.3f}) "
        f"= {reward:.3f}"
    )
    return reward


def execute_placement(
    game_state: "GameState", shape_idx: int, r: int, c: int, rng: random.Random
) -> float:
    """
    Places a shape, clears lines, updates game state (NumPy arrays), potentially
    triggers a batch refill, checks for game over, and calculates reward.

    Args:
        game_state: The current game state (will be modified).
        shape_idx: Index of the shape to place.
        r: Target row for placement.
        c: Target column for placement.
        rng: Random number generator for shape refilling.

    Returns:
        The reward obtained for this step.
    """
    shape = game_state.shapes[shape_idx]
    if not shape:
        logger.error(f"Attempted to place an empty shape slot: {shape_idx}")
        return 0.0

    # Use the NumPy-based can_place from GridLogic
    if not GridLogic.can_place(game_state.grid_data, shape, r, c):
        logger.error(f"Invalid placement attempted: Shape {shape_idx} at ({r},{c})")
        return 0.0

    # --- Place the shape ---
    placed_coords: set[tuple[int, int]] = set()
    placed_count = 0
    color_id = COLOR_TO_ID_MAP.get(shape.color, NO_COLOR_ID)
    if color_id == NO_COLOR_ID:
        logger.warning(f"Shape color {shape.color} not found in COLOR_TO_ID_MAP!")
        color_id = 0

    for dr, dc, _ in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        if game_state.grid_data.valid(tri_r, tri_c):
            if (
                not game_state.grid_data._death_np[tri_r, tri_c]
                and not game_state.grid_data._occupied_np[tri_r, tri_c]
            ):
                game_state.grid_data._occupied_np[tri_r, tri_c] = True
                game_state.grid_data._color_id_np[tri_r, tri_c] = color_id
                placed_coords.add((tri_r, tri_c))
                placed_count += 1
            else:
                logger.error(
                    f"Placement conflict at ({tri_r},{tri_c}) during execution, though can_place was true."
                )
        else:
            logger.error(
                f"Invalid coordinates ({tri_r},{tri_c}) encountered during placement execution."
            )

    game_state.shapes[shape_idx] = None  # Remove shape from slot
    game_state.pieces_placed_this_episode += 1

    # --- Check and clear lines ---
    lines_cleared_count, unique_coords_cleared, _ = GridLogic.check_and_clear_lines(
        game_state.grid_data, placed_coords
    )
    game_state.triangles_cleared_this_episode += len(unique_coords_cleared)

    # --- Update Score (Optional tracking) ---
    game_state.game_score += placed_count + len(unique_coords_cleared) * 2

    # --- Refill shapes if all slots are empty (Batch Refill Logic) ---
    if all(s is None for s in game_state.shapes):
        logger.debug("All shape slots empty, triggering batch refill.")
        ShapeLogic.refill_shape_slots(game_state, rng)  # Call batch refill

    # --- Check for game over AFTER placement and potential refill ---
    # This ensures game over is only triggered if no moves are possible with the *current or newly refilled* shapes
    if not game_state.valid_actions():
        game_state.game_over = True
        logger.info(
            f"Game over detected after placing shape {shape_idx} and potential refill."
        )

    # --- Calculate Reward based on the final outcome of this step ---
    step_reward = calculate_reward(
        placed_count=placed_count,
        unique_coords_cleared=unique_coords_cleared,
        is_game_over=game_state.game_over,  # Use the potentially updated game_over status
        config=game_state.env_config,
    )

    return step_reward
