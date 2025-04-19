# File: muzerotriangle/visualization/drawing/hud.py
from typing import Any

import pygame

from ..core import colors


def render_hud(
    surface: pygame.Surface,
    mode: str,
    fonts: dict[str, pygame.font.Font | None],
    display_stats: dict[str, Any] | None = None,
) -> None:
    """
    Renders the Heads-Up Display (HUD) at the bottom of the screen.

    Args:
        surface: The Pygame surface to draw on.
        mode: The current mode ('play', 'debug', 'training_visual').
        fonts: A dictionary containing loaded Pygame fonts.
        display_stats: Optional dictionary containing stats to display (used in training mode).
    """
    screen_width, screen_height = surface.get_size()
    hud_height = 40  # Define HUD height or get from config if available
    hud_rect = pygame.Rect(0, screen_height - hud_height, screen_width, hud_height)
    pygame.draw.rect(surface, colors.GRAY, hud_rect)  # Draw HUD background

    font = fonts.get("help")
    if not font:
        return  # Cannot render text without font

    # Common text for all modes
    common_text = "[ESC] Quit"

    # Mode-specific text
    if mode == "play":
        mode_text = " | [Click] Select/Place Shape"
    elif mode == "debug":
        mode_text = " | [Click] Toggle Cell"
    elif mode == "training_visual":
        mode_text = " | Training Mode"  # Keep it simple for training view
    else:
        mode_text = ""

    full_text = common_text + mode_text

    # Render and blit the text
    text_surface = font.render(full_text, True, colors.WHITE)
    text_rect = text_surface.get_rect(center=hud_rect.center)
    surface.blit(text_surface, text_rect)

    # Display additional stats in training mode if provided
    if mode == "training_visual" and display_stats:
        stats_font = fonts.get("help") or font  # Use help font or fallback
        stats_text_parts = []

        # Example stats to display (customize as needed)
        if "global_step" in display_stats:
            stats_text_parts.append(f"Step: {display_stats['global_step']:,}")
        if "episodes_played" in display_stats:
            stats_text_parts.append(f"Eps: {display_stats['episodes_played']:,}")
        if "total_simulations_run" in display_stats:
            sims = display_stats["total_simulations_run"]
            sims_str = (
                f"{sims / 1e6:.1f}M"
                if sims >= 1e6
                else (f"{sims / 1e3:.0f}k" if sims >= 1000 else str(int(sims)))
            )
            stats_text_parts.append(f"Sims: {sims_str}")
        if "buffer_size" in display_stats and "buffer_capacity" in display_stats:
            stats_text_parts.append(
                f"Buffer: {display_stats['buffer_size']:,}/{display_stats['buffer_capacity']:,}"
            )
        if "num_active_workers" in display_stats and "num_workers" in display_stats:
            stats_text_parts.append(
                f"Workers: {display_stats['num_active_workers']}/{display_stats['num_workers']}"
            )

        stats_text = " | ".join(stats_text_parts)
        if stats_text:
            stats_surf = stats_font.render(stats_text, True, colors.YELLOW)
            # Position stats text to the left of the help text if space allows, otherwise below
            stats_rect = stats_surf.get_rect(
                midleft=(hud_rect.left + 10, hud_rect.centery)
            )
            if stats_rect.right > text_rect.left - 10:  # Check for overlap
                stats_rect.topleft = (
                    hud_rect.left + 10,
                    hud_rect.top + 2,
                )  # Position above if overlapping
                text_rect.topleft = (hud_rect.left + 10, stats_rect.bottom + 2)
            surface.blit(stats_surf, stats_rect)
