# File: muzerotriangle/visualization/core/layout.py
import logging
import math

import pygame

from ...config import VisConfig

logger = logging.getLogger(__name__)


def calculate_interactive_layout(
    screen_width: int, screen_height: int, vis_config: VisConfig
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for interactive modes (play/debug).
    Places grid on the left and preview on the right.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    hud_h = vis_config.HUD_HEIGHT
    preview_w = vis_config.PREVIEW_AREA_WIDTH

    available_h = max(0, sh - hud_h - 2 * pad)
    available_w = max(0, sw - 3 * pad)  # Account for padding around grid and preview

    grid_w = max(0, available_w - preview_w)
    grid_h = available_h

    grid_rect = pygame.Rect(pad, pad, grid_w, grid_h)
    preview_rect = pygame.Rect(grid_rect.right + pad, pad, preview_w, grid_h)
    hud_rect = pygame.Rect(pad, grid_rect.bottom + pad, sw - 2 * pad, hud_h)

    # Clip rects to screen bounds to prevent errors with small windows
    screen_rect = pygame.Rect(0, 0, sw, sh)
    grid_rect = grid_rect.clip(screen_rect)
    preview_rect = preview_rect.clip(screen_rect)
    hud_rect = hud_rect.clip(screen_rect)

    logger.debug(
        f"Interactive Layout calculated: Grid={grid_rect}, Preview={preview_rect}, HUD={hud_rect}"
    )

    return {
        "grid": grid_rect,
        "preview": preview_rect,
        "hud": hud_rect,
    }


def calculate_training_layout(
    screen_width: int,
    screen_height: int,
    vis_config: VisConfig,
    progress_bars_total_height: int,
) -> dict[str, pygame.Rect]:
    """
    Calculates layout rectangles for training visualization mode.
    Worker grid top, progress bars bottom (above HUD), plots fill middle.
    """
    sw, sh = screen_width, screen_height
    pad = vis_config.PADDING
    hud_h = vis_config.HUD_HEIGHT

    # Calculate total available height excluding top/bottom padding and HUD
    total_content_height = sh - hud_h - 2 * pad

    # Allocate space for worker grid (e.g., 30% of available height)
    worker_grid_h = int(total_content_height * 0.3)
    worker_grid_w = sw - 2 * pad
    worker_grid_rect = pygame.Rect(pad, pad, worker_grid_w, worker_grid_h)

    # Allocate space for progress bars at the bottom, just above the HUD
    pb_area_y = sh - hud_h - pad - progress_bars_total_height
    pb_area_w = sw - 2 * pad
    progress_bar_area_rect = pygame.Rect(
        pad, pb_area_y, pb_area_w, progress_bars_total_height
    )

    # Plot area fills the remaining space between worker grid and progress bars
    plot_area_y = worker_grid_rect.bottom + pad
    plot_area_h = max(0, progress_bar_area_rect.top - plot_area_y - pad)
    plot_area_w = sw - 2 * pad
    plot_rect = pygame.Rect(pad, plot_area_y, plot_area_w, plot_area_h)

    # HUD area at the very bottom
    hud_rect = pygame.Rect(pad, sh - hud_h - pad, sw - 2 * pad, hud_h)

    # Clip all rects to screen bounds
    screen_rect = pygame.Rect(0, 0, sw, sh)
    worker_grid_rect = worker_grid_rect.clip(screen_rect)
    plot_rect = plot_rect.clip(screen_rect)
    progress_bar_area_rect = progress_bar_area_rect.clip(screen_rect)
    hud_rect = hud_rect.clip(screen_rect)

    logger.debug(
        f"Training Layout calculated: WorkerGrid={worker_grid_rect}, PlotRect={plot_rect}, ProgressBarArea={progress_bar_area_rect}, HUD={hud_rect}"
    )

    return {
        "worker_grid": worker_grid_rect,
        "plots": plot_rect,
        "progress_bar_area": progress_bar_area_rect,
        "hud": hud_rect,
    }
