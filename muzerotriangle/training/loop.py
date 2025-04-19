# File: muzerotriangle/training/loop.py
# File: muzerotriangle/training/loop.py
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any  # Import Optional

# Use MuZero types
from ..rl.types import SelfPlayResult
from .loop_helpers import LoopHelpers
from .worker_manager import WorkerManager

if TYPE_CHECKING:
    from ..visualization.ui import ProgressBar
    from .components import TrainingComponents

logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Manages the core asynchronous MuZero training loop logic.
    Coordinates worker tasks (trajectory generation), buffer (trajectory storage),
    trainer (sequence-based training), and visualization.
    Handles PER sampling and priority updates.
    """

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: (
            queue.Queue[dict[int, Any] | None] | None
        ) = None,  # Make optional
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.train_config = components.train_config

        self.buffer = components.buffer
        self.trainer = components.trainer
        self.data_manager = components.data_manager  # Add data manager

        self.global_step = 0
        self.episodes_played = 0
        self.total_simulations_run = 0
        self.worker_weight_updates_count = 0
        self.start_time = time.time()
        self.stop_requested = threading.Event()
        self.training_complete = False
        self.training_exception: Exception | None = None  # Make optional

        self.train_step_progress: ProgressBar | None = None  # Make optional
        self.buffer_fill_progress: ProgressBar | None = None  # Make optional

        self.worker_manager = WorkerManager(components)
        self.loop_helpers = LoopHelpers(
            components, self.visual_state_queue, self._get_loop_state
        )

        logger.info("MuZero TrainingLoop initialized.")

    def _get_loop_state(self) -> dict[str, Any]:
        """Provides current loop state to helpers."""
        return {
            "global_step": self.global_step,
            "episodes_played": self.episodes_played,
            "total_simulations_run": self.total_simulations_run,
            "worker_weight_updates": self.worker_weight_updates_count,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "num_active_workers": self.worker_manager.get_num_active_workers(),
            "num_pending_tasks": self.worker_manager.get_num_pending_tasks(),
            "train_progress": self.train_step_progress,
            "buffer_progress": self.buffer_fill_progress,
            "start_time": self.start_time,
            "num_workers": self.train_config.NUM_SELF_PLAY_WORKERS,
        }

    def set_initial_state(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Sets the initial state counters after loading."""
        self.global_step = global_step
        self.episodes_played = episodes_played
        self.total_simulations_run = total_simulations
        self.worker_weight_updates_count = (
            global_step // self.train_config.WORKER_UPDATE_FREQ_STEPS
        )
        self.train_step_progress, self.buffer_fill_progress = (
            self.loop_helpers.initialize_progress_bars(
                global_step,
                len(self.buffer),
                self.start_time,
            )
        )
        self.loop_helpers.reset_rate_counters(
            global_step, episodes_played, total_simulations
        )
        logger.info(
            f"TrainingLoop initial state set: Step={global_step}, Episodes={episodes_played}, Sims={total_simulations}, "
            f"WeightUpdates={self.worker_weight_updates_count}, BufferSteps={len(self.buffer)}"
        )

    def initialize_workers(self):
        """Initializes self-play workers using WorkerManager."""
        self.worker_manager.initialize_workers()

    def request_stop(self):
        """Signals the training loop to stop gracefully."""
        if not self.stop_requested.is_set():
            logger.info("Stop requested for TrainingLoop.")
            self.stop_requested.set()

    def _process_self_play_result(self, result: SelfPlayResult, worker_id: int):
        """Processes a validated MuZero result (trajectory) from a worker."""
        logger.debug(
            f"Processing result from worker {worker_id} "
            f"(Ep Steps: {result.episode_steps}, Score: {result.final_score:.2f}, "
            f"Trajectory Len: {len(result.trajectory)})"
        )

        if not result.trajectory:
            logger.warning(
                f"Worker {worker_id} returned an empty trajectory. Skipping."
            )
            return

        try:
            self.buffer.add(result.trajectory)
            logger.debug(
                f"Added trajectory (len {len(result.trajectory)}) from worker {worker_id} to buffer. "
                f"Buffer total steps: {len(self.buffer)}"
            )
        except Exception as e:
            logger.error(
                f"Error adding trajectory to buffer from worker {worker_id}: {e}",
                exc_info=True,
            )
            return

        if self.buffer_fill_progress:
            self.buffer_fill_progress.set_current_steps(len(self.buffer))
        self.episodes_played += 1
        self.total_simulations_run += result.total_simulations

    def _run_training_step(self) -> bool:
        """Runs one MuZero training step using sampled sequences."""
        if not self.buffer.is_ready():
            return False

        # --- Sample Sequences (Handles PER internally) ---
        sampled_batch_data = self.buffer.sample(
            self.train_config.BATCH_SIZE,
            current_train_step=self.global_step,  # Pass step for PER beta
        )
        if not sampled_batch_data:
            return False

        # --- Train on Sequences ---
        # Trainer handles PER weights internally if batch_sample is SampledBatchPER
        train_result = self.trainer.train_step(sampled_batch_data)

        if train_result:
            loss_info, td_errors = train_result
            self.global_step += 1
            if self.train_step_progress:
                self.train_step_progress.set_current_steps(self.global_step)

            # --- Update PER Priorities ---
            if (
                self.train_config.USE_PER
                and isinstance(sampled_batch_data, dict)
                and "indices" in sampled_batch_data
            ):
                tree_indices = sampled_batch_data["indices"]
                if len(tree_indices) == len(td_errors):
                    self.buffer.update_priorities(tree_indices, td_errors)
                else:
                    logger.error(
                        f"PER Update Error: Mismatch between tree indices ({len(tree_indices)}) and TD errors ({len(td_errors)})"
                    )

            # Log results
            self.loop_helpers.log_training_results_async(
                loss_info, self.global_step, self.total_simulations_run
            )

            # Check for worker weight update
            if self.global_step % self.train_config.WORKER_UPDATE_FREQ_STEPS == 0:
                try:
                    self.worker_manager.update_worker_networks(self.global_step)
                    self.worker_weight_updates_count += 1
                    self.loop_helpers.log_weight_update_event(self.global_step)
                except Exception as update_err:
                    logger.error(f"Failed to update worker networks: {update_err}")

            if self.global_step % 100 == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"Loss(T={loss_info['total_loss']:.3f} P={loss_info['policy_loss']:.3f} "
                    f"V={loss_info['value_loss']:.3f} R={loss_info['reward_loss']:.3f}) "
                    f"Ent={loss_info['entropy']:.3f}"
                )
            return True
        else:
            logger.warning(
                f"Training step {self.global_step + 1} failed (Trainer returned None)."
            )
            return False

    def run(self):
        """Main MuZero training loop."""
        max_steps_info = (
            f"Target steps: {self.train_config.MAX_TRAINING_STEPS}"
            if self.train_config.MAX_TRAINING_STEPS is not None
            else "Running indefinitely"
        )
        logger.info(f"Starting MuZero TrainingLoop run... {max_steps_info}")
        self.start_time = time.time()

        try:
            self.worker_manager.submit_initial_tasks()
            last_save_time = time.time()

            while not self.stop_requested.is_set():
                # Check max steps
                if (
                    self.train_config.MAX_TRAINING_STEPS is not None
                    and self.global_step >= self.train_config.MAX_TRAINING_STEPS
                ):
                    logger.info(
                        f"Reached MAX_TRAINING_STEPS ({self.train_config.MAX_TRAINING_STEPS}). Stopping."
                    )
                    self.training_complete = True
                    self.request_stop()
                    break

                # Training Step
                training_happened = False
                if self.buffer.is_ready():
                    training_happened = self._run_training_step()
                else:
                    time.sleep(0.1)

                if self.stop_requested.is_set():
                    break

                # Handle Completed Worker Tasks
                wait_timeout = 0.01 if training_happened else 0.2
                completed_tasks = self.worker_manager.get_completed_tasks(wait_timeout)

                for worker_id, result_or_error in completed_tasks:
                    if isinstance(result_or_error, SelfPlayResult):
                        try:
                            self._process_self_play_result(result_or_error, worker_id)
                        except Exception as proc_err:
                            logger.error(
                                f"Error processing result: {proc_err}", exc_info=True
                            )
                    elif isinstance(result_or_error, Exception):
                        logger.error(
                            f"Worker {worker_id} task failed: {result_or_error}"
                        )
                    else:
                        logger.error(
                            f"Unexpected item from worker {worker_id}: {type(result_or_error)}"
                        )
                    self.worker_manager.submit_task(worker_id)

                if self.stop_requested.is_set():
                    break

                # Periodic Tasks
                self.loop_helpers.update_visual_queue()
                self.loop_helpers.log_progress_eta()
                self.loop_helpers.calculate_and_log_rates()

                # Checkpointing
                current_time = time.time()
                if (
                    self.global_step > 0
                    and self.global_step % self.train_config.CHECKPOINT_SAVE_FREQ_STEPS
                    == 0
                    and current_time - last_save_time > 60
                ):
                    logger.info(f"Saving checkpoint at step {self.global_step}...")
                    self.data_manager.save_training_state(
                        nn=self.components.nn,
                        optimizer=self.trainer.optimizer,
                        stats_collector_actor=self.components.stats_collector_actor,
                        buffer=self.buffer,
                        global_step=self.global_step,
                        episodes_played=self.episodes_played,
                        total_simulations_run=self.total_simulations_run,
                    )
                    last_save_time = current_time

                if not completed_tasks and not training_happened:
                    time.sleep(0.02)

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Stopping.")
        except Exception as e:
            logger.critical(f"Unhandled exception in TrainingLoop: {e}", exc_info=True)
            self.training_exception = e
        finally:
            self.request_stop()
            self.training_complete = (
                self.training_complete and self.training_exception is None
            )
            logger.info(
                f"TrainingLoop finished. Complete: {self.training_complete}, Exception: {self.training_exception is not None}"
            )

    def cleanup_actors(self):
        """Cleans up worker actors."""
        self.worker_manager.cleanup_actors()
