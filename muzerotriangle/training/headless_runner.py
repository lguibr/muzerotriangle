# File: muzerotriangle/training/headless_runner.py
import logging
import sys
import traceback
from collections import deque
from pathlib import Path

import mlflow
import ray

from ..config import APP_NAME, PersistenceConfig, TrainConfig

# from ..utils.sumtree import SumTree # REMOVED: PER disabled
from .components import TrainingComponents
from .logging_utils import (
    get_root_logger,
    log_configs_to_mlflow,
    setup_file_logging,
)
from .loop import TrainingLoop
from .setup import count_parameters, setup_training_components

logger = logging.getLogger(__name__)


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
    # Pass None for visual queue in headless mode
    training_loop = TrainingLoop(components, visual_state_queue=None)

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
        # --- CHANGED: Rebuild buffer with trajectories ---
        components.buffer.buffer = deque(loaded_state.buffer_data.trajectories)
        components.buffer.total_steps = loaded_state.buffer_data.total_steps
        logger.info(
            f"MuZero buffer loaded. Trajectories: {len(components.buffer.buffer)}, Total Steps: {len(components.buffer)}"
        )
        # --- END CHANGED ---
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


def run_training_headless_mode(
    log_level_str: str,
    train_config_override: TrainConfig,
    persist_config_override: PersistenceConfig,
) -> int:
    """Runs the training pipeline in headless mode."""
    # (Rest of the function remains largely the same, setup/cleanup logic is independent of buffer format)
    training_loop: TrainingLoop | None = None
    components: TrainingComponents | None = None
    exit_code = 1
    log_file_path = None
    file_handler = None
    ray_initialized_by_setup = False
    mlflow_run_active = False

    try:
        log_file_path = setup_file_logging(
            persist_config_override, train_config_override.RUN_NAME, "headless"
        )
        log_level = logging.getLevelName(log_level_str.upper())
        logger.info(
            f"Logging {logging.getLevelName(log_level)} and higher messages to console and: {log_file_path}"
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
        training_loop.initialize_workers()
        training_loop.run()

        if training_loop.training_complete:
            exit_code = 0
        elif training_loop.training_exception:
            exit_code = 1
        else:
            exit_code = 1

    except Exception as e:
        logger.critical(
            f"An unhandled error occurred during headless training setup or execution: {e}"
        )
        traceback.print_exc()
        if mlflow_run_active:
            try:
                mlflow.log_param("training_status", "SETUP_FAILED")
                mlflow.log_param("error_message", str(e))
            except Exception as mlf_err:
                logger.error(f"Failed to log setup error status to MLflow: {mlf_err}")
        exit_code = 1

    finally:
        final_status = "UNKNOWN"
        error_msg = ""
        if training_loop:
            _save_final_state(training_loop)
            training_loop.cleanup_actors()
            if training_loop.training_exception:
                final_status = "FAILED"
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
                logger.info("Ray shut down by headless runner.")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}", exc_info=True)

        root_logger = get_root_logger()
        file_handler = next(
            (h for h in root_logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if file_handler:
            try:
                file_handler.flush()
                file_handler.close()
                root_logger.removeHandler(file_handler)
            except Exception as e_close:
                print(f"Error closing log file handler: {e_close}", file=sys.__stderr__)

        logger.info(f"Headless training finished with exit code {exit_code}.")
    return exit_code
