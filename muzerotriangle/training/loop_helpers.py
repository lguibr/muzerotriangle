# File: muzerotriangle/training/loop_helpers.py
import logging
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any  # Import Optional

import ray

from ..environment import GameState
from ..stats.plot_definitions import WEIGHT_UPDATE_METRIC_KEY
from ..utils import format_eta

# Import MuZero types
from ..visualization.core import colors  # Keep colors import
from ..visualization.ui import ProgressBar

if TYPE_CHECKING:
    from ..utils.types import StatsCollectorData, StepInfo
    from .components import TrainingComponents

logger = logging.getLogger(__name__)

VISUAL_UPDATE_INTERVAL = 0.2
STATS_FETCH_INTERVAL = 0.5
VIS_STATE_FETCH_TIMEOUT = 0.1
RATE_CALCULATION_INTERVAL = 5.0


class LoopHelpers:
    """Provides helper functions for the MuZero TrainingLoop."""

    def __init__(
        self,
        components: "TrainingComponents",
        visual_state_queue: queue.Queue[dict[int, Any] | None] | None,  # Make optional
        get_loop_state_func: Callable[[], dict[str, Any]],
    ):
        self.components = components
        self.visual_state_queue = visual_state_queue
        self.get_loop_state = get_loop_state_func

        self.stats_collector_actor = components.stats_collector_actor
        self.train_config = components.train_config
        self.trainer = components.trainer

        self.last_visual_update_time = 0.0
        self.last_stats_fetch_time = 0.0
        self.latest_stats_data: StatsCollectorData = {}

        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = 0
        self.last_rate_calc_episodes = 0
        self.last_rate_calc_sims = 0

    def reset_rate_counters(
        self, global_step: int, episodes_played: int, total_simulations: int
    ):
        """Resets counters used for rate calculation."""
        self.last_rate_calc_time = time.time()
        self.last_rate_calc_step = global_step
        self.last_rate_calc_episodes = episodes_played
        self.last_rate_calc_sims = total_simulations

    def initialize_progress_bars(
        self, global_step: int, buffer_total_steps: int, start_time: float
    ) -> tuple[ProgressBar, ProgressBar]:
        """Initializes and returns progress bars."""
        train_total = self.train_config.MAX_TRAINING_STEPS or 1
        # Buffer progress now tracks total steps vs capacity
        buffer_total = self.train_config.BUFFER_CAPACITY

        train_pb = ProgressBar(
            "Training Steps", train_total, start_time, global_step, colors.GREEN
        )
        # Use buffer_total_steps for initial buffer progress
        buffer_pb = ProgressBar(
            "Buffer Steps", buffer_total, start_time, buffer_total_steps, colors.ORANGE
        )
        return train_pb, buffer_pb

    def _fetch_latest_stats(self):
        """Fetches the latest stats data from the actor."""
        current_time = time.time()
        if current_time - self.last_stats_fetch_time < STATS_FETCH_INTERVAL:
            return
        self.last_stats_fetch_time = current_time
        if self.stats_collector_actor:
            try:
                data_ref = self.stats_collector_actor.get_data.remote()  # type: ignore
                self.latest_stats_data = ray.get(data_ref, timeout=1.0)
            except Exception as e:
                logger.warning(f"Failed to fetch latest stats: {e}")

    def calculate_and_log_rates(self):
        """Calculates and logs rates. Buffer size now represents total steps."""
        current_time = time.time()
        time_delta = current_time - self.last_rate_calc_time
        if time_delta < RATE_CALCULATION_INTERVAL:
            return

        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        episodes_played = loop_state["episodes_played"]
        total_simulations = loop_state["total_simulations_run"]
        current_buffer_total_steps = int(
            loop_state["buffer_size"]
        )  # This is total steps

        steps_delta = global_step - self.last_rate_calc_step
        episodes_delta = episodes_played - self.last_rate_calc_episodes
        sims_delta = total_simulations - self.last_rate_calc_sims

        steps_per_sec = steps_delta / time_delta if time_delta > 0 else 0.0
        episodes_per_sec = episodes_delta / time_delta if time_delta > 0 else 0.0
        sims_per_sec = sims_delta / time_delta if time_delta > 0 else 0.0

        if self.stats_collector_actor:
            # Buffer size context uses total steps
            step_info_buffer: StepInfo = {
                "global_step": global_step,
                "buffer_size": current_buffer_total_steps,
            }
            step_info_global: StepInfo = {"global_step": global_step}

            rate_stats: dict[str, tuple[float, StepInfo]] = {
                "Rate/Episodes_Per_Sec": (episodes_per_sec, step_info_buffer),
                "Rate/Simulations_Per_Sec": (sims_per_sec, step_info_buffer),
                "Buffer/Size": (
                    float(current_buffer_total_steps),
                    step_info_buffer,
                ),  # Log total steps
            }
            log_msg_steps = "Steps/s=N/A"
            if steps_delta > 0:
                rate_stats["Rate/Steps_Per_Sec"] = (steps_per_sec, step_info_global)
                log_msg_steps = f"Steps/s={steps_per_sec:.2f}"

            try:
                self.stats_collector_actor.log_batch.remote(rate_stats)  # type: ignore
                logger.debug(
                    f"Logged rates/buffer at step {global_step} / buffer_steps {current_buffer_total_steps}: "
                    f"{log_msg_steps}, Eps/s={episodes_per_sec:.2f}, Sims/s={sims_per_sec:.1f}, "
                    f"BufferSteps={current_buffer_total_steps}"
                )
            except Exception as e:
                logger.error(f"Failed to log rate/buffer stats: {e}")

        self.reset_rate_counters(global_step, episodes_played, total_simulations)

    def log_progress_eta(self):
        """Logs progress and ETA based on training steps."""
        # (No change needed here, uses training steps)
        loop_state = self.get_loop_state()
        global_step = loop_state["global_step"]
        train_progress = loop_state["train_progress"]
        if global_step == 0 or global_step % 100 != 0 or not train_progress:
            return

        elapsed_time = time.time() - loop_state["start_time"]
        steps_since_load = global_step - train_progress.initial_steps
        steps_per_sec = 0.0
        self._fetch_latest_stats()
        rate_dq = self.latest_stats_data.get("Rate/Steps_Per_Sec")
        if rate_dq:
            steps_per_sec = rate_dq[-1][1]
        elif elapsed_time > 1 and steps_since_load > 0:
            steps_per_sec = steps_since_load / elapsed_time

        target_steps = self.train_config.MAX_TRAINING_STEPS
        target_steps_str = f"{target_steps:,}" if target_steps else "Infinite"
        progress_str = f"Step {global_step:,}/{target_steps_str}"
        eta_str = format_eta(train_progress.get_eta_seconds())
        buffer_total_steps = loop_state["buffer_size"]
        buffer_capacity = loop_state["buffer_capacity"]
        buffer_fill_perc = (
            (buffer_total_steps / buffer_capacity) * 100 if buffer_capacity > 0 else 0.0
        )
        total_sims = loop_state["total_simulations_run"]
        total_sims_str = (
            f"{total_sims / 1e6:.2f}M"
            if total_sims >= 1e6
            else (f"{total_sims / 1e3:.1f}k" if total_sims >= 1000 else str(total_sims))
        )
        num_pending = loop_state["num_pending_tasks"]

        logger.info(
            f"Progress: {progress_str}, Episodes: {loop_state['episodes_played']:,}, Total Sims: {total_sims_str}, "
            f"Buffer Steps: {buffer_total_steps:,}/{buffer_capacity:,} ({buffer_fill_perc:.1f}%), "
            f"Pending Tasks: {num_pending}, Speed: {steps_per_sec:.2f} steps/sec, ETA: {eta_str}"
        )

    def update_visual_queue(self):
        """Fetches latest states/stats and puts them onto the visual queue."""
        # (No change needed here, passes loop state and stats data)
        if not self.visual_state_queue or not self.stats_collector_actor:
            return
        current_time = time.time()
        if current_time - self.last_visual_update_time < VISUAL_UPDATE_INTERVAL:
            return
        self.last_visual_update_time = current_time

        latest_worker_states: dict[int, GameState] = {}
        try:
            states_ref = self.stats_collector_actor.get_latest_worker_states.remote()  # type: ignore
            latest_worker_states = ray.get(states_ref, timeout=VIS_STATE_FETCH_TIMEOUT)
            if not isinstance(latest_worker_states, dict):
                latest_worker_states = {}
        except Exception as e:
            logger.warning(f"Failed to fetch worker states: {e}")
            latest_worker_states = {}

        self._fetch_latest_stats()

        visual_data: dict[int, Any] = {}
        for worker_id, state in latest_worker_states.items():
            if isinstance(state, GameState):
                visual_data[worker_id] = state

        visual_data[-1] = {
            **self.get_loop_state(),
            "stats_data": self.latest_stats_data,
        }

        if not visual_data or len(visual_data) == 1:
            return  # Only global data

        try:
            while self.visual_state_queue.qsize() > 2:
                self.visual_state_queue.get_nowait()
            self.visual_state_queue.put_nowait(visual_data)
        except queue.Full:
            logger.warning("Visual queue full.")
        except Exception as qe:
            logger.error(f"Error putting in visual queue: {qe}")

    # --- REMOVED: validate_experiences (handled by SelfPlayResult) ---

    def log_training_results_async(
        self, loss_info: dict[str, float], global_step: int, total_simulations: int
    ) -> None:
        """Logs MuZero training results asynchronously."""
        current_lr = self.trainer.get_current_lr()
        loop_state = self.get_loop_state()
        buffer_total_steps = int(loop_state["buffer_size"])  # Use total steps

        if self.stats_collector_actor:
            step_info: StepInfo = {
                "global_step": global_step,
                "buffer_size": buffer_total_steps,
            }
            stats_batch: dict[str, tuple[float, StepInfo]] = {
                # --- CHANGED: Log MuZero Losses ---
                "Loss/Total": (loss_info["total_loss"], step_info),
                "Loss/Policy": (loss_info["policy_loss"], step_info),
                "Loss/Value": (loss_info["value_loss"], step_info),
                "Loss/Reward": (loss_info["reward_loss"], step_info),  # Add reward loss
                # --- END CHANGED ---
                "Loss/Entropy": (
                    loss_info.get("entropy", 0.0),
                    step_info,
                ),  # Keep entropy if calculated
                "LearningRate": (current_lr, step_info),
                "Progress/Total_Simulations": (float(total_simulations), step_info),
            }
            # --- REMOVED: PER Beta ---
            # if per_beta is not None: stats_batch["PER/Beta"] = (per_beta, step_info)
            try:
                self.stats_collector_actor.log_batch.remote(stats_batch)  # type: ignore
            except Exception as e:
                logger.error(f"Failed to log batch to StatsCollectorActor: {e}")

    def log_weight_update_event(self, global_step: int) -> None:
        """Logs the event of a worker weight update."""
        # (No change needed here)
        if self.stats_collector_actor:
            try:
                step_info: StepInfo = {"global_step": global_step}
                update_metric = {WEIGHT_UPDATE_METRIC_KEY: (1.0, step_info)}
                self.stats_collector_actor.log_batch.remote(update_metric)  # type: ignore
                logger.info(f"Logged worker weight update event at step {global_step}.")
            except Exception as e:
                logger.error(f"Failed to log weight update event: {e}")
