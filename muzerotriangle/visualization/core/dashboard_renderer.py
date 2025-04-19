# File: muzerotriangle/visualization/core/dashboard_renderer.py
import logging
import math
from collections import deque  # Import deque
from typing import TYPE_CHECKING, Any

import pygame
import ray  # Import ray

from ...environment import GameState
from ...stats import Plotter
from ...utils.types import StatsCollectorData  # Import StatsCollectorData
from ..drawing import hud as hud_drawing
from ..ui import ProgressBar  # Import ProgressBar
from . import colors, layout
from .game_renderer import GameRenderer

if TYPE_CHECKING:
    from ...config import EnvConfig, ModelConfig, VisConfig
    from ...stats import StatsCollectorActor

logger = logging.getLogger(__name__)


class DashboardRenderer:
    """Renders the training dashboard, including multiple game states and plots."""

    def __init__(
        self,
        screen: pygame.Surface,
        vis_config: "VisConfig",
        env_config: "EnvConfig",
        fonts: dict[str, pygame.font.Font | None],
        stats_collector_actor: ray.actor.ActorHandle | None,
        model_config: "ModelConfig",
        total_params: int | None = None,
        trainable_params: int | None = None,
    ):
        self.screen = screen
        self.vis_config = vis_config
        self.env_config = env_config
        self.fonts = fonts
        self.stats_collector_actor = stats_collector_actor
        self.model_config = model_config
        self.total_params = total_params
        self.trainable_params = trainable_params

        self.layout_rects: dict[str, pygame.Rect] | None = None
        self.worker_sub_rects: dict[int, pygame.Rect] = {}
        self.last_worker_grid_size = (0, 0)
        self.last_num_workers = 0

        self.game_renderer = GameRenderer(vis_config, env_config, fonts)
        self.plotter = Plotter()
        self.progress_bar_height_per_bar = 25
        self.num_progress_bars = 2
        self.progress_bar_spacing = 2
        self.progress_bars_total_height = (
            (
                (self.progress_bar_height_per_bar * self.num_progress_bars)
                + (self.progress_bar_spacing * (self.num_progress_bars - 1))
            )
            if self.num_progress_bars > 0
            else 0
        )
        self._layout_calculated_for_size: tuple[int, int] = (0, 0)

    def ensure_layout(self):
        """Recalculates layout if screen size changes."""
        current_w, current_h = self.screen.get_size()
        current_size = (current_w, current_h)
        if (
            self.layout_rects is None
            or self._layout_calculated_for_size != current_size
        ):
            self.layout_rects = layout.calculate_training_layout(
                current_w,
                current_h,
                self.vis_config,
                progress_bars_total_height=self.progress_bars_total_height,
            )
            self._layout_calculated_for_size = current_size
            logger.info(f"Recalculated dashboard layout for {current_size}")
            # Reset worker layout cache as well
            self.last_worker_grid_size = (0, 0)
            self.worker_sub_rects = {}
        return self.layout_rects if self.layout_rects is not None else {}

    def _calculate_worker_sub_layout(self, worker_grid_area, worker_ids):
        """Calculates the positions and sizes for each worker's game view."""
        area_w, area_h = worker_grid_area.size
        num_workers = len(worker_ids)

        # Only recalculate if size or number of workers changes
        if (
            area_w,
            area_h,
        ) == self.last_worker_grid_size and num_workers == self.last_num_workers:
            return

        self.last_worker_grid_size = (area_w, area_h)
        self.last_num_workers = num_workers
        self.worker_sub_rects = {}

        if area_h <= 10 or area_w <= 10 or num_workers <= 0:
            logger.warning(
                f"Worker grid area too small or zero workers: {area_w}x{area_h}, {num_workers} workers"
            )
            return

        # Simple grid layout calculation
        cols = int(math.ceil(math.sqrt(num_workers)))
        rows = math.ceil(num_workers / cols)
        cell_w = area_w // cols
        cell_h = area_h // rows

        logger.info(
            f"Calculated worker sub-layout: {rows}x{cols}. Cell: {cell_w}x{cell_h}"
        )

        sorted_worker_ids = sorted(worker_ids)
        for i, worker_id in enumerate(sorted_worker_ids):
            row = i // cols
            col = i % cols
            worker_area_x = worker_grid_area.left + col * cell_w
            worker_area_y = worker_grid_area.top + row * cell_h
            worker_rect = pygame.Rect(worker_area_x, worker_area_y, cell_w, cell_h)
            self.worker_sub_rects[worker_id] = worker_rect.clip(worker_grid_area)

    def render(
        self,
        worker_states: dict[int, GameState],
        global_stats: dict[str, Any] | None = None,
    ):
        """Renders the entire training dashboard."""
        self.screen.fill(colors.DARK_GRAY)
        layout_rects = self.ensure_layout()
        if not layout_rects:
            return

        worker_grid_area = layout_rects.get("worker_grid")
        plots_rect = layout_rects.get("plots")
        progress_bar_area_rect = layout_rects.get("progress_bar_area")
        hud_rect = layout_rects.get("hud")

        # --- Render Worker Grids ---
        if (
            worker_grid_area
            and worker_grid_area.width > 0
            and worker_grid_area.height > 0
        ):
            worker_ids = list(worker_states.keys())
            if not worker_ids and global_stats and "num_workers" in global_stats:
                # If no states yet, create placeholders based on expected worker count
                worker_ids = list(range(global_stats["num_workers"]))

            self._calculate_worker_sub_layout(worker_grid_area, worker_ids)

            for worker_id, worker_area_rect in self.worker_sub_rects.items():
                game_state = worker_states.get(worker_id)
                # Extract step-specific stats for this worker if available
                worker_step_stats = None
                if global_stats and "latest_worker_stats" in global_stats:
                    worker_step_stats = global_stats["latest_worker_stats"].get(
                        worker_id
                    )

                # --- CORRECTED TYPO ---
                self.game_renderer.render_worker_state(
                    self.screen,
                    worker_area_rect,
                    worker_id,
                    game_state,
                    worker_step_stats=worker_step_stats,
                )
                # --- END CORRECTION ---
                pygame.draw.rect(self.screen, colors.GRAY, worker_area_rect, 1)

        # --- Render Plots ---
        if plots_rect and plots_rect.width > 0 and plots_rect.height > 0:
            stats_data_for_plot: StatsCollectorData | None = (
                global_stats.get("stats_data") if global_stats else None
            )
            plot_surface = None
            if stats_data_for_plot is not None:
                has_any_metric_data = any(
                    isinstance(dq, deque) and dq  # Use imported deque
                    for key, dq in stats_data_for_plot.items()
                    if not key.startswith("Internal/")
                )
                if has_any_metric_data:
                    plot_surface = self.plotter.get_plot_surface(
                        stats_data_for_plot,
                        int(plots_rect.width),
                        int(plots_rect.height),
                    )

            if plot_surface:
                self.screen.blit(plot_surface, plots_rect.topleft)
            else:  # Draw placeholder if no data or plotter failed
                pygame.draw.rect(self.screen, colors.DARK_GRAY, plots_rect)
                plot_font = self.fonts.get("help")
                if plot_font:
                    wait_text = (
                        "Plot Area (Waiting for data...)"
                        if stats_data_for_plot is None
                        else "Plot Area (No data yet)"
                    )
                    wait_surf = plot_font.render(wait_text, True, colors.LIGHT_GRAY)
                    wait_rect = wait_surf.get_rect(center=plots_rect.center)
                    self.screen.blit(wait_surf, wait_rect)
                pygame.draw.rect(self.screen, colors.GRAY, plots_rect, 1)

        # --- Render Progress Bars ---
        if progress_bar_area_rect and global_stats:
            current_y = progress_bar_area_rect.top
            progress_bar_font = self.fonts.get("help")
            if progress_bar_font:
                bar_width = progress_bar_area_rect.width
                bar_x = progress_bar_area_rect.left
                bar_height = self.progress_bar_height_per_bar

                # Training Progress Bar
                train_progress = global_stats.get("train_progress")
                if isinstance(train_progress, ProgressBar):  # Use imported ProgressBar
                    # Construct info string for training bar
                    train_info_parts = []
                    if self.model_config:
                        model_str = f"CNN:{len(self.model_config.CONV_FILTERS)}L"
                        if self.model_config.NUM_RESIDUAL_BLOCKS > 0:
                            model_str += (
                                f"/Res:{self.model_config.NUM_RESIDUAL_BLOCKS}L"
                            )
                        if (
                            self.model_config.USE_TRANSFORMER_IN_REP
                            and self.model_config.REP_TRANSFORMER_LAYERS > 0
                        ):
                            model_str += (
                                f"/TF:{self.model_config.REP_TRANSFORMER_LAYERS}L"
                            )
                        train_info_parts.append(model_str)
                    if self.total_params is not None:
                        train_info_parts.append(
                            f"Params:{self.total_params / 1e6:.1f}M"
                        )
                    train_bar_info_str = " | ".join(train_info_parts)

                    train_progress.render(
                        self.screen,
                        (bar_x, current_y),
                        int(bar_width),
                        bar_height,
                        progress_bar_font,
                        border_radius=3,
                        info_line=train_bar_info_str,
                    )
                    current_y += bar_height + self.progress_bar_spacing

                # Buffer Progress Bar
                buffer_progress = global_stats.get("buffer_progress")
                if isinstance(buffer_progress, ProgressBar):  # Use imported ProgressBar
                    # Construct info string for buffer bar
                    buffer_info_parts = []
                    updates = global_stats.get("worker_weight_updates", "?")
                    episodes = global_stats.get("episodes_played", "?")
                    sims = global_stats.get("total_simulations_run", "?")
                    num_workers = global_stats.get("num_workers", "?")
                    pending_tasks = global_stats.get("num_pending_tasks", "?")

                    buffer_info_parts.append(f"Weight Updates: {updates}")
                    buffer_info_parts.append(f"Episodes: {episodes}")
                    if isinstance(sims, (int, float)):
                        sims_str = (
                            f"{sims / 1e6:.1f}M"
                            if sims >= 1e6
                            else (
                                f"{sims / 1e3:.0f}k" if sims >= 1000 else str(int(sims))
                            )
                        )
                        buffer_info_parts.append(f"Simulations: {sims_str}")
                    else:
                        buffer_info_parts.append(f"Simulations: {sims}")

                    if isinstance(pending_tasks, int) and isinstance(num_workers, int):
                        buffer_info_parts.append(
                            f"Workers: {num_workers-pending_tasks}/{num_workers} Active"
                        )
                    else:
                        buffer_info_parts.append(f"Workers: ?/?")

                    buffer_bar_info_str = " | ".join(buffer_info_parts)

                    buffer_progress.render(
                        self.screen,
                        (bar_x, current_y),
                        int(bar_width),
                        bar_height,
                        progress_bar_font,
                        border_radius=3,
                        info_line=buffer_bar_info_str,
                    )

        # Render HUD (always last)
        hud_drawing.render_hud(
            surface=self.screen,
            mode="training_visual",
            fonts=self.fonts,
            display_stats=None,  # HUD doesn't need detailed stats anymore
        )
