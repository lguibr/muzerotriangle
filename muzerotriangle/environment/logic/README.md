# File: muzerotriangle/environment/logic/README.md
# Environment Logic Submodule (`muzerotriangle.environment.logic`)

## Purpose and Architecture

This submodule contains higher-level game logic that operates on the `GameState` and its components (`GridData`, `Shape`). It bridges the gap between basic actions/rules and the overall game flow.

-   **`actions.py`:**
    -   `get_valid_actions`: Determines all possible valid moves (shape placements) from the current `GameState` by iterating through available shapes and grid positions, checking placement validity using [`GridLogic.can_place`](../grid/logic.py). Returns a list of encoded `ActionType` integers.
-   **`step.py`:**
    -   `execute_placement`: Performs the core logic when a shape is placed. It updates the `GridData` (occupancy and color), checks for and clears completed lines using [`GridLogic.check_and_clear_lines`](../grid/logic.py), calculates the reward for the step using `calculate_reward`, updates the game score and step counters. **It then checks if all shape slots are empty and triggers a batch refill using [`ShapeLogic.refill_shape_slots`](../shapes/logic.py) if they are. Finally, it checks if the game is over based on the potentially refilled shapes.**
    -   `calculate_reward`: Calculates the reward based on the number of triangles placed, triangles cleared, and whether the game ended.

## Exposed Interfaces

-   **Functions:**
    -   `get_valid_actions(game_state: GameState) -> List[ActionType]`
    -   `execute_placement(game_state: GameState, shape_idx: int, r: int, c: int, rng: random.Random) -> float`
    -   `calculate_reward(placed_count: int, unique_coords_cleared: Set[Tuple[int, int]], is_game_over: bool, config: EnvConfig) -> float`

## Dependencies

-   **[`muzerotriangle.environment.core`](../core/README.md)**:
    -   `GameState`: The primary object operated upon.
    -   `ActionCodec`: Used by `get_valid_actions`.
-   **[`muzerotriangle.environment.grid`](../grid/README.md)**:
    -   `GridData`, `GridLogic`: Used for placement checks and line clearing.
-   **[`muzerotriangle.environment.shapes`](../shapes/README.md)**:
    -   `Shape`, `ShapeLogic`: Used for shape refilling.
-   **[`muzerotriangle.config`](../../config/README.md)**:
    -   `EnvConfig`: Used for reward calculation and action encoding.
-   **[`muzerotriangle.structs`](../../structs/README.md)**:
    -   `Shape`, `Triangle`, `COLOR_TO_ID_MAP`, `NO_COLOR_ID`.
-   **[`muzerotriangle.utils`](../../utils/README.md)**:
    -   `ActionType`.
-   **Standard Libraries:** `typing`, `random`, `logging`.

---

**Note:** Please keep this README updated when changing the logic for determining valid actions, executing placements (including reward calculation, **batch shape refilling**, and game over check timing), or modifying dependencies.