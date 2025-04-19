# File: muzerotriangle/training/visual_runner.py
import logging
import queue
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Optional

import mlflow
import pygame
import ray

from .. import config, environment, visualization
from ..config import APP_NAME, PersistenceConfig, TrainConfig

# Import Trajectory type
from ..utils.sumtree import SumTree  # Import SumTree for re-initialization
from ..utils.types import Trajectory
from .components import TrainingComponents
from .logging_utils import (
    Tee,
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)

# Queue for communication between training thread and visualization thread
visual_state_queue: queue.Queue[Optional[dict[int, Any]]] = queue.Queue(maxsize=5)


def _initialize_mlflow(persist_config: PersistenceConfig, run_name: str) -> bool:
    # (No changes needed here)
    try:
        mlflow_abs_path = persist_config.get_mlflow_abs_path()
        Path(mlflow_abs_path).mkdir(parents=True, exist_ok=True)
        mlflow_tracking_uri = persist_config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(APP_NAME)
        logger.info(f"Set MLflow tracking URI to: {mlflow_tracking_uri}")
        logger.info(f"Set MLflow experiment to: {APP_NAME}")
        mlflow.start_run(run_name=run_name)
        active_run = mlflow.active_run()
        if active_run:
            logger.info(f"MLflow Run started (ID: {active_run.info.run_id}).")
            return True
        else:
            logger.error("MLflow run failed to start.")
            return False
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        return False


def _load_and_apply_initial_state(components: TrainingComponents) -> TrainingLoop:
    """Loads initial state (checkpoint, MuZero buffer) and applies it."""
    loaded_state = components.data_manager.load_initial_state()
    # Pass the visual queue to the TrainingLoop
    training_loop = TrainingLoop(components, visual_state_queue=visual_state_queue)

    # --- Apply Checkpoint Data (No change needed here) ---
    if loaded_state.checkpoint_data:
        cp_data = loaded_state.checkpoint_data
        logger.info(
            f"Applying loaded checkpoint data (Run: {cp_data.run_name}, Step: {cp_data.global_step})"
        )
        if cp_data.model_state_dict:
            components.nn.set_weights(cp_data.model_state_dict)
        if cp_data.optimizer_state_dict:
            try:
                components.trainer.load_optimizer_state(
                    cp_data.optimizer_state_dict
                )  # Use helper
                logger.info("Optimizer state loaded and moved to device.")
            except Exception as opt_load_err:
                logger.error(
                    f"Could not load optimizer state: {opt_load_err}. Optimizer might reset."
                )
        if (
            cp_data.stats_collector_state
            and components.stats_collector_actor is not None
        ):
            try:
                set_state_ref = components.stats_collector_actor.set_state.remote(
                    cp_data.stats_collector_state
                )
                ray.get(set_state_ref, timeout=5.0)
                logger.info("StatsCollectorActor state restored.")
            except Exception as e:
                logger.error(
                    f"Error restoring StatsCollectorActor state: {e}", exc_info=True
                )
        training_loop.set_initial_state(
            cp_data.global_step, cp_data.episodes_played, cp_data.total_simulations_run
        )
    else:
        logger.info("No checkpoint data loaded. Starting fresh.")
        training_loop.set_initial_state(0, 0, 0)

    # --- Apply MuZero Buffer Data ---
    if loaded_state.buffer_data:
        logger.info("Loading MuZero buffer data...")
        reconstructed_buffer_data: list[tuple[int, Trajectory]] = []
        # Ensure buffer_idx mapping is rebuilt correctly
        components.buffer.buffer.clear()
        components.buffer.tree_idx_to_buffer_idx.clear()
        components.buffer.buffer_idx_to_tree_idx.clear()
        components.buffer.total_steps = 0
        components.buffer.next_buffer_idx = 0
        if components.buffer.use_per and components.buffer.sum_tree:
            components.buffer.sum_tree.reset()  # Reset sumtree

        for i, traj in enumerate(loaded_state.buffer_data.trajectories):
            # Re-add trajectories to ensure buffer and sumtree are consistent
            components.buffer.add(traj)

        # Verify counts after loading
        logger.info(
            f"MuZero Buffer loaded. Trajectories in deque: {len(components.buffer.buffer)}, "
            f"Total Steps: {components.buffer.total_steps}, "
            f"SumTree Entries: {components.buffer.sum_tree.n_entries if components.buffer.sum_tree else 'N/A'}"
        )
        if training_loop.buffer_fill_progress:
            training_loop.buffer_fill_progress.set_current_steps(len(components.buffer))
    else:
        logger.info("No buffer data loaded.")

    components.nn.model.train()
    return training_loop


def _save_final_state(training_loop: TrainingLoop):
    """Saves the final training state."""
    # (No changes needed here, uses DataManager)
    if not training_loop:
        return
    components = training_loop.components
    logger.info("Saving final training state...")
    try:
        components.data_manager.save_training_state(
            nn=components.nn,
            optimizer=components.trainer.optimizer,
            stats_collector_actor=components.stats_collector_actor,
            buffer=components.buffer,
            global_step=training_loop.global_step,
            episodes_played=training_loop.episodes_played,
            total_simulations_run=training_loop.total_simulations_run,
            is_final=True,
        )
    except Exception as e_save:
        logger.error(f"Failed to save final training state: {e_save}", exc_info=True)


# --- ADDED: Define _training_loop_thread_func ---
def _training_loop_thread_func(training_loop: TrainingLoop):
    """Target function for the training loop thread."""
    try:
        training_loop.initialize_workers()
        training_loop.run()
    except Exception as e:
        logger.critical(f"Exception in training loop thread: {e}", exc_info=True)
        training_loop.training_exception = e  # Store exception
    finally:
        # Signal visualization thread to exit by putting None in the queue
        if training_loop.visual_state_queue:
            try:
                training_loop.visual_state_queue.put_nowait(None)
            except queue.Full:
                logger.warning("Visual queue full when trying to send exit signal.")
        logger.info("Training loop thread finished.")


# --- END ADDED ---


def run_training_visual_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline with visualization."""
    main_thread_exception = None
    train_thread = None
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    file_handler = None
    tee_stdout = None
    tee_stderr = None
    ray_initialized_by_setup = False
    mlflow_run_active = False
    total_params: int | None = None
    trainable_params: int | None = None

    try:
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "visual"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
        )

        # Redirect stdout/stderr to also go to the log file
        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if file_handler and hasattr(file_handler, "stream") and file_handler.stream:
            tee_stdout = Tee(
                original_stdout,
                file_handler.stream,
                main_stream_for_fileno=original_stdout,
            )
            tee_stderr = Tee(
                original_stderr,
                file_handler.stream,
                main_stream_for_fileno=original_stderr,
            )
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            print("--- Stdout/Stderr redirected to console and log file ---")
            logger.info("Stdout/Stderr redirected to console and log file.")
        else:
            logger.error(
                "Could not redirect stdout/stderr: File handler stream not available."
            )

        components, ray_initialized_by_setup = setup_training_components(
            train_config_override, persist_config_override
        )
        if not components:
            raise RuntimeError("Failed to initialize training components.")

        mlflow_run_active = _initialize_mlflow(
            components.persist_config, components.train_config.RUN_NAME
        )
        if mlflow_run_active:
            log_configs_to_mlflow(components)
            total_params, trainable_params = count_parameters(components.nn.model)
            logger.info(
                f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}"
            )
            mlflow.log_param("model_total_params", total_params)
            mlflow.log_param("model_trainable_params", trainable_params)
        else:
            logger.warning("MLflow initialization failed, proceeding without MLflow.")

        training_loop = _load_and_apply_initial_state(components)

        # Start the training loop in a separate thread
        train_thread = threading.Thread(
            target=_training_loop_thread_func, args=(training_loop,), daemon=True
        )
        train_thread.start()
        logger.info("Training loop thread launched.")

        # Initialize Pygame and the DashboardRenderer
        vis_config = config.VisConfig()
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode(
            (vis_config.SCREEN_WIDTH, vis_config.SCREEN_HEIGHT), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"{config.APP_NAME} - Training ({components.train_config.RUN_NAME})"
        )
        clock = pygame.time.Clock()
        fonts = visualization.load_fonts()
        dashboard_renderer = visualization.DashboardRenderer(
            screen,
            vis_config,
            components.env_config,
            fonts,
            components.stats_collector_actor,
            components.model_config,
            total_params=total_params,
            trainable_params=trainable_params,
        )

        # Main visualization loop
        running = True
        # --- ADDED: Initialize loop variables ---
        current_worker_states: dict[int, environment.GameState] = {}
        current_global_stats: dict[str, Any] = {}
        # --- END ADDED ---
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.VIDEORESIZE:
                    try:
                        w, h = max(640, event.w), max(240, event.h)
                        screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        dashboard_renderer.screen = screen
                        dashboard_renderer.layout_rects = None  # Force recalculation
                    except pygame.error as e:
                        logger.error(f"Error resizing window: {e}")

            # Check if the training thread is still alive
            if train_thread and not train_thread.is_alive():
                logger.warning("Training loop thread terminated.")
                running = False
                if training_loop and training_loop.training_exception:
                    logger.error(
                        f"Training thread terminated due to exception: {training_loop.training_exception}"
                    )
                    main_thread_exception = training_loop.training_exception

            # Get data from the queue (non-blocking)
            try:
                visual_data = visual_state_queue.get_nowait()
                if visual_data is None:  # Signal to exit
                    running = False
                    logger.info("Received exit signal from training thread.")
                    continue
                # Update local copies for rendering
                current_global_stats = visual_data.pop(-1, {})
                current_worker_states = visual_data
            except queue.Empty:
                # No new data, just redraw with existing data
                pass
            except Exception as e:
                logger.error(f"Error getting data from visual queue: {e}")
                time.sleep(0.1)  # Avoid busy-waiting on error
                continue

            # Render the dashboard
            screen.fill(visualization.colors.DARK_GRAY)
            dashboard_renderer.render(current_worker_states, current_global_stats)
            pygame.display.flip()
            clock.tick(vis_config.FPS)

        # Signal the training loop to stop if it hasn't already
        if training_loop and not training_loop.stop_requested.is_set():
            training_loop.request_stop()

        # Wait for the training thread to finish
        if train_thread and train_thread.is_alive():
            logger.info("Waiting for training thread to finish...")
            train_thread.join(timeout=10)  # Wait up to 10 seconds
            if train_thread.is_alive():
                logger.warning("Training thread did not terminate gracefully.")

        # Determine exit code based on exceptions
        if main_thread_exception:
            exit_code = 1
        elif training_loop and training_loop.training_exception:
            exit_code = 1
        elif training_loop and training_loop.training_complete:
            exit_code = 0
        else:
            exit_code = 1  # Interrupted or other issue

    except Exception as e:
        logger.critical(f"An unhandled error occurred in visual training script: {e}")
        traceback.print_exc()
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            # Ensure final state is saved even if visualization loop crashed
            if (
                not training_loop.training_complete
                and not training_loop.training_exception
            ):
                _save_final_state(training_loop)

            training_loop.cleanup_actors()  # Ensure actors are cleaned up

            if main_thread_exception:
                final_status = "VIS_FAILED"
                error_msg = str(main_thread_exception)
            elif training_loop.training_exception:
                final_status = "TRAIN_FAILED"
                error_msg = str(training_loop.training_exception)
            elif training_loop.training_complete:
                final_status = "COMPLETED"
            else:
                final_status = "INTERRUPTED"
        else:
            final_status = "SETUP_FAILED"

        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", final_status)
                if error_msg:
                    mlflow.log_param("error_message", error_msg)
                mlflow.end_run()
                logger.info(f"MLflow Run ended. Final Status: {final_status}")
            except Exception as mlf_end_err:
                logger.error(f"Error ending MLflow run: {mlf_end_err}")

        if ray_initialized_by_setup and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shut down by visual runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        pygame.quit()

        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                get_root_logger().removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Visual training finished with exit code {exit_code}.")
    return exit_code
