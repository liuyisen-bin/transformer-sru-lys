#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""Video recorder utility for logging training videos to WandB and saving locally."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


class VideoRecorder:
    """Records video frames during training and logs them to WandB.

    This class handles video recording during RL training, capturing frames
    at specified intervals and uploading them to Weights & Biases for visualization.

    Video recording only works when:
    1. Recording is explicitly enabled via `enable()`
    2. The environment has `render_mode="rgb_array"`
    3. WandB logger is being used (logger_type == "wandb")
    4. The writer has `log_video` method available

    Example:
        >>> recorder = VideoRecorder(env, video_length=200, video_interval=1000)
        >>> recorder.enable()
        >>> # In training loop:
        >>> if recorder.should_record(iteration):
        >>>     recorder.start_recording()
        >>> if recorder.is_recording:
        >>>     recorder.capture_frame()
        >>> recorder.log_video(writer, iteration, logger_type)
    """

    def __init__(
        self,
        env: VecEnv,
        video_length: int = 200,
        video_interval: int = 2000,
        fps: int = 30,
        save_local: bool = True,
        log_dir: str | None = None,
    ):
        """Initialize the video recorder.

        Args:
            env: The vectorized environment to record from.
            video_length: Number of environment steps per video. Defaults to 200.
            video_interval: Number of training iterations between recordings. Defaults to 2000.
            fps: Frames per second for the recorded video. Defaults to 30.
            save_local: Whether to save videos locally as MP4 files. Defaults to True.
            log_dir: Directory to save videos. If None, videos are only uploaded to WandB.
        """
        self.env = env
        self.video_length = video_length
        self.video_interval = video_interval
        self.fps = fps
        self.save_local = save_local
        self.log_dir = log_dir

        self._enabled = False
        self._is_recording = False
        self._frames: list[np.ndarray] = []
        self._step_count = 0
        self._current_iteration = 0

        # Create videos directory if local saving is enabled
        if self.save_local and self.log_dir is not None:
            self.video_dir = Path(self.log_dir) / "videos"
            self.video_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        """Whether video recording is enabled."""
        return self._enabled

    @property
    def is_recording(self) -> bool:
        """Whether currently recording frames."""
        return self._is_recording

    def enable(self):
        """Enable video recording."""
        self._enabled = True
        self._reset_state()
        save_info = f", saving to {self.video_dir}" if self.save_local and self.log_dir else ""
        print(f"[INFO] Video recording enabled: {self.video_length} steps every {self.video_interval} iterations{save_info}")

    def disable(self):
        """Disable video recording."""
        self._enabled = False
        self._reset_state()

    def _reset_state(self):
        """Reset recording state."""
        self._is_recording = False
        self._frames = []
        self._step_count = 0

    def _can_render(self) -> bool:
        """Check if the environment can render rgb_array frames."""
        return hasattr(self.env, "render_mode") and self.env.render_mode == "rgb_array"

    def should_record(self, iteration: int) -> bool:
        """Check if we should start recording at this iteration.

        Args:
            iteration: Current training iteration.

        Returns:
            True if recording should start at this iteration.
        """
        result = self._enabled and iteration % self.video_interval == 0
        if result:
            print(f"[DEBUG VideoRecorder] should_record({iteration}) = True (enabled={self._enabled}, interval={self.video_interval})")
        return result

    def start_recording(self):
        """Start a new video recording session."""
        if not self._enabled:
            return

        self._is_recording = True
        self._frames = []
        self._step_count = 0
        print(f"[DEBUG VideoRecorder] Started recording (can_render={self._can_render()}, render_mode={getattr(self.env, 'render_mode', None)})")

    def capture_frame(self):
        """Capture a frame from the environment if recording.

        Should be called once per environment step during rollout.
        Automatically stops recording when video_length is reached.
        """
        if not self._is_recording:
            return

        if self._step_count >= self.video_length:
            return

        if not self._can_render():
            self._step_count += 1
            return

        frame = self.env.unwrapped.render()
        if frame is not None:
            self._frames.append(frame)

        self._step_count += 1

    def is_complete(self) -> bool:
        """Check if the video has captured enough frames.

        Returns:
            True if video has reached video_length frames, False otherwise.
        """
        return self._step_count >= self.video_length

    def _save_video_local(self, iteration: int) -> bool:
        """Save video to local MP4 file.

        Args:
            iteration: Current training iteration for filename.

        Returns:
            True if video was saved successfully, False otherwise.
        """
        if not self.save_local or self.log_dir is None:
            return False

        if len(self._frames) == 0:
            return False

        try:
            import imageio

            video_path = self.video_dir / f"train_video_iter_{iteration:06d}.mp4"
            frames_array = np.array(self._frames)

            # Save as MP4 using imageio
            imageio.mimwrite(video_path, frames_array, fps=self.fps, quality=8, macro_block_size=1)

            print(f"[INFO] Saved video to {video_path} ({len(self._frames)} frames, {len(self._frames)/self.fps:.1f}s)")
            return True
        except ImportError:
            print("[WARNING] imageio not installed, skipping local video save. Install with: pip install imageio[ffmpeg]")
            return False
        except Exception as e:
            print(f"[WARNING] Failed to save video locally: {e}")
            return False

    def log_video(self, writer, iteration: int, logger_type: str) -> bool:
        """Log the recorded video to WandB and/or save locally.

        Args:
            writer: The summary writer (WandB, TensorBoard, etc.).
            iteration: Current training iteration for logging step.
            logger_type: Type of logger being used ("wandb", "tensorboard", "neptune").

        Returns:
            True if video was successfully logged or saved, False otherwise.
        """
        print(f"[DEBUG VideoRecorder] log_video called: iter={iteration}, is_recording={self._is_recording}, frames={len(self._frames)}, step_count={self._step_count}")
        if not self._is_recording:
            return False

        if len(self._frames) == 0:
            print(f"[DEBUG VideoRecorder] No frames captured! Resetting state.")
            self._reset_state()
            return False

        success = False

        # Save video locally if enabled
        if self.save_local:
            local_success = self._save_video_local(iteration)
            success = success or local_success

        # Upload to WandB if available
        if logger_type == "wandb" and hasattr(writer, "log_video"):
            writer.log_video(self._frames, step=iteration, fps=self.fps, tag="Media/video")
            print(f"[INFO] Uploaded video to WandB with {len(self._frames)} frames at iteration {iteration}")
            success = True

        # Reset state after logging
        self._reset_state()
        return success
