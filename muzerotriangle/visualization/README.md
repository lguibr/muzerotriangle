# File: muzerotriangle/visualization/README.md
# Visualization Module (`muzerotriangle.visualization`)

## Purpose and Architecture

This module handles all visual aspects of the MuZeroTriangle project, primarily using Pygame for rendering the game board, pieces, and training progress.

-   **Core ([`core/README.md`](core/README.md)):** Contains the main rendering classes (`Visualizer`, `GameRenderer`, `DashboardRenderer`), layout logic (`layout.py`), color definitions (`colors.py`), font loading (`fonts.py`), and coordinate mapping utilities (`coord_mapper.py`).
    -   `Visualizer`: Manages the display for interactive modes (play, debug).
    -   `GameRenderer`: Renders a single game instance, used by `DashboardRenderer`.
    -   `DashboardRenderer`: Manages the complex layout for training visualization, displaying multiple worker games, plots, and progress bars. **Progress bars now show specific information (model/params on train bar, global stats on buffer bar).**
-   **Drawing ([`drawing/README.md`](drawing/README.md)):** Contains specific functions for drawing individual elements like the grid, shapes, previews, HUD, and highlights. These are used by the core renderers.
-   **UI ([`ui/README.md`](ui/README.md)):** Contains reusable UI elements like `ProgressBar`.

## Exposed Interfaces

-   **Classes:**
    -   `Visualizer`: Main renderer for interactive modes.
    -   `GameRenderer`: Renderer for a single worker's game state.
    -   `DashboardRenderer`: Renderer for training visualization.
    -   `ProgressBar`: UI element for progress display.
-   **Functions:**
    -   `load_fonts`: Loads Pygame fonts.
    -   Various drawing functions (see `drawing/README.md`).
    -   Layout functions (see `core/README.md`).
    -   Coordinate mapping functions (see `core/README.md`).
-   **Modules:**
    -   `colors`: Provides color constants.

## Dependencies

-   **[`muzerotriangle.config`](../config/README.md)**: `VisConfig`, `EnvConfig`, `ModelConfig`.
-   **[`muzerotriangle.environment`](../environment/README.md)**: `GameState`, `GridData`.
-   **[`muzerotriangle.structs`](../structs/README.md)**: `Triangle`, `Shape`, color constants.
-   **[`muzerotriangle.stats`](../stats/README.md)**: `Plotter`, `StatsCollectorActor`.
-   **[`muzerotriangle.utils`](../utils/README.md)**: `geometry`, `helpers`, `types`.
-   **`pygame`**: Core library for graphics rendering.
-   **`matplotlib`**: Used by `Plotter` for generating plots.
-   **`numpy`**: Used for plot data handling.
-   **Standard Libraries:** `typing`, `logging`, `math`, `time`, `queue`, `io`.

---

**Note:** Keep this README updated with any changes to the visualization structure, components, or dependencies.