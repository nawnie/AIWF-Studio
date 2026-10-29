from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from aiwf.infrastructure.video.export import select_codec
from aiwf.infrastructure.video.processing import VideoError, _frame_to_rgb_uint8, _resolve_ffmpeg, _transcode_to_h264


class StreamingVideoWriter:
    """Stream RGB frames into FFmpeg without retaining the full clip in RAM."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        width: int,
        height: int,
        input_fps: float,
        target_fps: float | None = None,
        audio_source: str | Path | None = None,
        crf: int = 18,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.input_fps = float(input_fps)
        self.target_fps = float(target_fps) if target_fps else None
        self.audio_source = Path(audio_source) if audio_source else None
        self.crf = int(crf)
        self.frames_written = 0
        self._ffmpeg = _resolve_ffmpeg()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file = None
        self._cv2_writer = None
        self._temporary = self.output_path.with_name(
            f".{self.output_path.stem}.streaming{self.output_path.suffix or '.mp4'}"
        )
        if self._ffmpeg:
            self._start_ffmpeg()
        else:
            self._start_opencv()

    def _start_ffmpeg(self) -> None:
        codec, pix_fmt = select_codec()
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.input_fps:.8f}",
            "-i",
            "pipe:0",
        ]
        has_audio = self.audio_source is not None and self.audio_source.is_file()
        if has_audio:
            cmd.extend(["-i", str(self.audio_source), "-map", "0:v:0", "-map", "1:a:0?"])
        if self.target_fps and abs(self.target_fps - self.input_fps) > 0.001:
            cmd.extend(["-vf", f"fps={self.target_fps:g}"])
        if "nvenc" in codec:
            cmd.extend(
                [
                    "-c:v",
                    codec,
                    "-preset",
                    "p5",
                    "-tune",
                    "hq",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(self.crf),
                    "-b:v",
                    "0",
                ]
            )
        else:
            cmd.extend(["-c:v", codec, "-preset", "medium", "-crf", str(self.crf)])
        cmd.extend(["-pix_fmt", pix_fmt])
        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        cmd.extend(["-movflags", "+faststart", str(self._temporary)])

        fd, log_name = tempfile.mkstemp(prefix="aiwf-rife-ffmpeg-", suffix=".log")
        os.close(fd)
        self._stderr_file = open(log_name, "wb")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
            creationflags=creationflags,
        )

    def _start_opencv(self) -> None:
        try:
            import cv2
        except Exception as exc:
            raise VideoError("Neither FFmpeg nor OpenCV video encoding is available.") from exc
        fps = self.target_fps or self.input_fps
        self._cv2_writer = cv2.VideoWriter(
            str(self._temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.width, self.height),
        )
        if not self._cv2_writer.isOpened():
            raise VideoError("Could not open a streaming video encoder.")

    def write(self, frame) -> None:
        rgb = _frame_to_rgb_uint8(frame)
        if rgb.shape[:2] != (self.height, self.width):
            try:
                import cv2

                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            except Exception as exc:
                raise VideoError(
                    f"RIFE output frame changed size from {self.width}x{self.height} to "
                    f"{rgb.shape[1]}x{rgb.shape[0]}."
                ) from exc
        rgb = np.ascontiguousarray(rgb)
        if self._process is not None:
            if self._process.stdin is None:
                raise VideoError("FFmpeg frame pipe is unavailable.")
            try:
                self._process.stdin.write(rgb.tobytes())
            except BrokenPipeError as exc:
                raise VideoError("FFmpeg closed the RIFE frame pipe early.") from exc
        else:
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            self._cv2_writer.write(bgr)
        self.frames_written += 1

    def close(self) -> Path:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            return_code = self._process.wait()
            if self._stderr_file is not None:
                log_path = Path(self._stderr_file.name)
                self._stderr_file.close()
            else:
                log_path = None
            if return_code != 0:
                detail = ""
                if log_path and log_path.is_file():
                    detail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                self.abort()
                raise VideoError(f"FFmpeg streaming encode failed with code {return_code}.\n{detail}")
            if log_path:
                log_path.unlink(missing_ok=True)
        elif self._cv2_writer is not None:
            self._cv2_writer.release()
            _transcode_to_h264(self._temporary)

        if not self._temporary.is_file() or self._temporary.stat().st_size <= 0:
            raise VideoError("Streaming encoder produced no output.")
        os.replace(self._temporary, self.output_path)
        return self.output_path

    def abort(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.kill()
            except OSError:
                pass
        if self._cv2_writer is not None:
            try:
                self._cv2_writer.release()
            except Exception:
                pass
        if self._stderr_file is not None and not self._stderr_file.closed:
            self._stderr_file.close()
        self._temporary.unlink(missing_ok=True)

    def __enter__(self) -> "StreamingVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
