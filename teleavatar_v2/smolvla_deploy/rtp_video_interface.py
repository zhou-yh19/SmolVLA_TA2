#!/usr/bin/env python3
"""
Receive a Teleavatar H265 RTP video stream as RGB images.

This mirrors the image-facing part of TeleavatarROS2Interface: the latest
decoded frame is kept under a lock, and get_observation() returns image copies.
The interface exposes both the composite frame and six split camera views.
Test-only file writing lives in test_video.py.

The GStreamer streaming thread hands over only a reference to the decoded
sample; the RGB conversion and the crops are computed by whichever thread asks
for them. The policy consumes about one frame per chunk while the stream runs
at ~45fps, and converting every frame in the callback cost ~20MB of copies per
frame on a thread that then competed with inference for the GIL.
"""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Dict, Iterable
from typing import Optional

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib
    from gi.repository import Gst
except ImportError as exc:
    raise RuntimeError(
        "PyGObject/GStreamer Python bindings are required. Install python3-gi "
        "and gir1.2-gstreamer-1.0, or run with a Python that has gi available."
    ) from exc


Gst.init(None)


TELEAVATAR_SPLIT_REGIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("head_right_eye", (0.1250, 0.0000, 0.8750, 0.3529411765)),
    ("head_left_eye", (0.1250, 0.3529411765, 0.8750, 0.7058823529)),
    ("right_wrist_left_eye", (0.0000, 0.7058823529, 0.5000, 0.8529411765)),
    ("right_wrist_right_eye", (0.5000, 0.7058823529, 1.0000, 0.8529411765)),
    ("left_wrist_left_eye", (0.0000, 0.8529411765, 0.5000, 1.0000)),
    ("left_wrist_right_eye", (0.5000, 0.8529411765, 1.0000, 1.0000)),
)


class RTPH265VideoInterface:
    """Thread-safe image interface backed by a GStreamer RTP/H265 decoder."""

    def __init__(
        self,
        *,
        camera_name: str = "head_camera",
        port: int = 8890,
        payload: int = 96,
        udp_buffer_size: int = 20_000_000,
        decoder: str = "nvh265dec max-display-delay=0",
        log_interval_s: float = 2.0,
    ):
        self.camera_name = camera_name
        self.port = port
        self.payload = payload
        self.udp_buffer_size = udp_buffer_size
        self.decoder = decoder
        self.log_interval_s = log_interval_s
        self.split_regions = TELEAVATAR_SPLIT_REGIONS
        self.split_image_names = tuple(name for name, _box in self.split_regions)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.lock = threading.Lock()
        # The newest decoded sample and when it arrived. Views are cut from it
        # on demand; `_converted` memoizes them for the sample they came from.
        self._latest_sample: Optional[Gst.Sample] = None
        self._latest_timestamp: Optional[float] = None
        self._converted: tuple[Optional[Gst.Sample], Dict[str, np.ndarray]] = (None, {})

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._rtp_probe: Optional[Gst.Element] = None
        self._first_frame = threading.Event()
        self._eos_or_error = threading.Event()
        self._stop_requested = False

        self._frame_count = 0
        self._rtp_packet_count = 0
        self._h265_buffer_count = 0
        self._stats_start_time: Optional[float] = None
        self._last_log_time = time.monotonic()
        self._last_log_frame_count = 0
        self._decode_start_times: deque[float] = deque(maxlen=512)
        self._latest_decode_latency_ms: Optional[float] = None

    def start(self) -> None:
        """Start receiving images in a background GStreamer thread."""
        if self._thread and self._thread.is_alive():
            return

        self._reset_runtime_state()
        pipeline_description = self._build_pipeline()
        self.logger.info("Starting GStreamer pipeline: %s", pipeline_description)

        pipeline = Gst.parse_launch(pipeline_description)
        if not isinstance(pipeline, Gst.Pipeline):
            raise RuntimeError("GStreamer description did not create a Pipeline")

        appsink = pipeline.get_by_name("decoded_frames")
        if appsink is None:
            raise RuntimeError("Failed to find appsink named decoded_frames")
        appsink.connect("new-sample", self._image_callback)

        rtp_probe = pipeline.get_by_name("rtp_probe")
        if rtp_probe is not None:
            rtp_probe.connect("handoff", self._on_rtp_packet)
        self._rtp_probe = rtp_probe

        h265_probe = pipeline.get_by_name("h265_probe")
        if h265_probe is not None:
            h265_probe.connect("handoff", self._on_h265_buffer)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._pipeline = pipeline
        self._loop = GLib.MainLoop()
        self._first_frame.clear()
        self._eos_or_error.clear()
        self._stop_requested = False

        self._thread = threading.Thread(target=self._run_loop, name="rtp-h265-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop receiving images."""
        self._stop_requested = True

        if self._thread is not None:
            # quit() is a no-op on a loop that has not started running yet,
            # so retry until the loop thread actually exits (bounded).
            deadline = time.monotonic() + 2.0
            while self._thread.is_alive() and time.monotonic() < deadline:
                if self._loop is not None and self._loop.is_running():
                    self._loop.quit()
                self._thread.join(timeout=0.1)
            if self._thread.is_alive():
                self.logger.warning("RTP decoder thread did not exit within 2s")

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None

        self._loop = None
        self.logger.info("Stopped decoder after %d frames", self._frame_count)

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        """Wait for the first decoded image, matching TeleavatarROS2Interface."""
        self.logger.info("Waiting for initial video frame...")
        if self._first_frame.wait(timeout=timeout):
            self.logger.info("Initial video frame received")
            return True

        self.logger.error(
            "Timeout waiting for video after %.1fs: rtp_packets=%d h265_buffers=%d",
            timeout,
            self._rtp_packet_count,
            self._h265_buffer_count,
        )
        return False

    def get_observation(self) -> Optional[dict]:
        """Return current image observation, or None before the first frame."""
        images = self.get_latest_images()
        if not images:
            return None
        return {"images": images}

    def get_latest_image(self) -> Optional[np.ndarray]:
        """Return the latest composite RGB image."""
        return self.get_views((self.camera_name,)).get(self.camera_name)

    def get_latest_images(self) -> Dict[str, np.ndarray]:
        """Return all latest images, including the composite and split crops."""
        return self.get_views((self.camera_name, *self.split_image_names))

    def get_latest_images_with_timestamps(self) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Return latest image copies and their receive timestamps."""
        images = self.get_latest_images()
        timestamps = self.get_image_timestamps()
        return images, {name: timestamps[name] for name in images}

    def get_views(self, names: Iterable[str]) -> Dict[str, np.ndarray]:
        """Return copies of the requested views cut from the newest decoded frame.

        Only the views asked for are converted, so a caller that needs three
        crops does not pay for the composite and the other three. Empty before
        the first frame.
        """
        names = tuple(names)
        with self.lock:
            sample = self._latest_sample
            cached_sample, cached = self._converted
        if sample is None:
            return {}
        if cached_sample is not sample:
            cached = {}
        missing = [name for name in names if name not in cached]
        if missing:
            fresh = dict(cached)

            def cut(frame: np.ndarray) -> None:
                # `frame` only lives while the buffer is mapped, so every view
                # handed out is a copy made in here.
                for name in missing:
                    fresh[name] = frame.copy() if name == self.camera_name else self._crop(frame, name)

            _with_mapped_rgb(sample, cut)
            with self.lock:
                if self._latest_sample is sample:
                    self._converted = (sample, fresh)
            cached = fresh
        return {name: cached[name].copy() for name in names if name in cached}

    def get_image_timestamps(self) -> Dict[str, float]:
        """Return timestamps without touching the multi-megabyte RGB frames."""
        with self.lock:
            timestamp = self._latest_timestamp
        if timestamp is None:
            return {}
        return {name: timestamp for name in (self.camera_name, *self.split_image_names)}

    def has_initial_frame(self) -> bool:
        return self._first_frame.is_set()

    def stream_ended(self) -> bool:
        """True once the pipeline hit EOS or an unrecoverable error (no new frames will arrive)."""
        return self._eos_or_error.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for EOS/error. Returns True if the stream ended."""
        return self._eos_or_error.wait(timeout=timeout)

    def _reset_runtime_state(self) -> None:
        self._frame_count = 0
        self._rtp_packet_count = 0
        self._h265_buffer_count = 0
        self._stats_start_time = None
        self._last_log_time = time.monotonic()
        self._last_log_frame_count = 0
        self._decode_start_times.clear()
        self._latest_decode_latency_ms = None
        with self.lock:
            self._latest_sample = None
            self._latest_timestamp = None
            self._converted = (None, {})

    def _build_pipeline(self) -> str:
        caps = (
            "application/x-rtp,"
            "media=video,"
            "clock-rate=90000,"
            "encoding-name=H265,"
            f"payload={self.payload}"
        )
        decode = (
            f"udpsrc port={self.port} buffer-size={self.udp_buffer_size} caps=\"{caps}\" "
            "! identity name=rtp_probe signal-handoffs=true silent=true "
            "! rtph265depay "
            "! h265parse config-interval=-1 disable-passthrough=true "
            "! video/x-h265,stream-format=byte-stream,alignment=au "
            "! identity name=h265_probe signal-handoffs=true silent=true "
            f"! {self.decoder} "
            "! videoconvert n-threads=4 "
            "! video/x-raw,format=RGB "
        )
        appsink = (
            "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 "
            "! appsink name=decoded_frames emit-signals=true max-buffers=1 drop=true sync=false"
        )
        return f"{decode} ! {appsink}"

    def _run_loop(self) -> None:
        if self._pipeline is None or self._loop is None:
            raise RuntimeError("Pipeline was not initialized")

        pipeline = self._pipeline
        loop = self._loop
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self.logger.error("Failed to set GStreamer pipeline to PLAYING")
            self._eos_or_error.set()
            return

        try:
            # stop() may have been requested before the loop started running;
            # quit() would have been lost, so check the flag first (stop()
            # also keeps retrying quit() until this thread exits).
            if not self._stop_requested:
                loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)

    def _image_callback(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        decoded_at = time.monotonic()

        try:
            shape = _sample_shape(sample)
        except Exception:
            self.logger.exception("Decoded sample is not an RGB frame")
            return Gst.FlowReturn.OK

        self._update_decode_latency(decoded_at)

        # Keep only a reference: the sample is converted when a consumer asks.
        with self.lock:
            self._latest_sample = sample
            self._latest_timestamp = time.time()

        now = time.monotonic()
        self._frame_count += 1
        if not self._first_frame.is_set():
            self._first_frame.set()
            # The RTP packet count only serves bring-up ("is anything arriving
            # at all?"). Past the first frame it would cost ~1700 Python
            # callbacks per second on the streaming thread.
            if self._rtp_probe is not None:
                self._rtp_probe.set_property("signal-handoffs", False)
        self._log_stats(shape, now)
        return Gst.FlowReturn.OK

    def _on_rtp_packet(self, _identity: Gst.Element, _buffer: Gst.Buffer) -> None:
        self._rtp_packet_count += 1

    def _on_h265_buffer(self, _identity: Gst.Element, _buffer: Gst.Buffer) -> None:
        self._h265_buffer_count += 1
        self._decode_start_times.append(time.monotonic())

    def _update_decode_latency(self, decoded_at: float) -> None:
        if not self._decode_start_times:
            return
        self._latest_decode_latency_ms = max(0.0, (decoded_at - self._decode_start_times.popleft()) * 1000.0)

    def _crop(self, frame: np.ndarray, name: str) -> np.ndarray:
        height, width = frame.shape[:2]
        box = dict(self.split_regions)[name]
        x1, y1, x2, y2 = _normalized_box_to_pixels(box, width, height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Split region {name!r} is empty for a {width}x{height} frame")
        return frame[y1:y2, x1:x2].copy()

    def _log_stats(self, shape: tuple[int, ...], now: float) -> None:
        if self._stats_start_time is None:
            self._stats_start_time = now
            self._last_log_time = now
            self._last_log_frame_count = self._frame_count
            return

        elapsed = now - self._last_log_time
        if elapsed < self.log_interval_s:
            return

        frames_since_last_log = self._frame_count - self._last_log_frame_count
        interval_fps = frames_since_last_log / elapsed if elapsed > 0.0 else 0.0
        overall_elapsed = now - self._stats_start_time
        overall_fps = (self._frame_count - 1) / overall_elapsed if overall_elapsed > 0.0 else 0.0
        self._last_log_time = now
        self._last_log_frame_count = self._frame_count

        latency = "n/a" if self._latest_decode_latency_ms is None else f"{self._latest_decode_latency_ms:.1f} ms"
        self.logger.info(
            "camera=%s frames=%d h265_buffers=%d shape=%s fps=%.2f overall_fps=%.2f decode_latency=%s",
            self.camera_name,
            self._frame_count,
            self._h265_buffer_count,
            shape,
            interval_fps,
            overall_fps,
            latency,
        )

    def _on_bus_message(self, _bus: Gst.Bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self.logger.error("GStreamer error from %s: %s", message.src.get_name(), error)
            if debug:
                self.logger.error("GStreamer debug info: %s", debug)
            self._eos_or_error.set()
            if self._loop is not None:
                self._loop.quit()
        elif message.type == Gst.MessageType.EOS:
            self._eos_or_error.set()
            if self._loop is not None:
                self._loop.quit()


def _sample_shape(sample: Gst.Sample) -> tuple[int, int, int]:
    caps = sample.get_caps()
    if caps is None or caps.get_size() == 0:
        raise RuntimeError("Decoded sample has no caps")
    structure = caps.get_structure(0)
    if structure.get_value("format") != "RGB":
        raise RuntimeError(f"Expected RGB sample, got {structure.get_value('format')}")
    return int(structure.get_value("height")), int(structure.get_value("width")), 3


def _with_mapped_rgb(sample: Gst.Sample, consume) -> None:
    """Map the sample's buffer and hand `consume` an HWC uint8 view of it.

    The view is only valid inside the call; `consume` must copy what it keeps.
    """
    height, width, _ = _sample_shape(sample)
    buffer = sample.get_buffer()
    if buffer is None:
        raise RuntimeError("Decoded sample has no buffer")

    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("Failed to map decoded frame buffer")
    try:
        row_bytes = width * 3
        raw = np.frombuffer(map_info.data, dtype=np.uint8)
        stride = raw.size // height if raw.size % height == 0 and raw.size // height >= row_bytes else row_bytes
        if raw.size < height * stride:
            raise RuntimeError(f"Decoded buffer holds {raw.size} bytes, expected at least {height * stride}")
        consume(raw[: height * stride].reshape((height, stride))[:, :row_bytes].reshape((height, width, 3)))
    finally:
        buffer.unmap(map_info)


def _normalized_box_to_pixels(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, round(x1 * width))),
        max(0, min(height, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )
