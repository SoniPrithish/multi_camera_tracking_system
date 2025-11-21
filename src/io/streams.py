"""
Video stream handling with bounded queues, RTSP reconnect, and health monitoring.
Supports MP4 files, RTSP streams, HTTP streams, and webcams.
"""

import cv2
import time
import logging
import threading
from queue import Queue, Empty, Full
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Stream connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class StreamHealth:
    """Health metrics for a stream."""
    status: StreamStatus = StreamStatus.DISCONNECTED
    fps: float = 0.0
    frame_count: int = 0
    drop_count: int = 0
    reconnect_count: int = 0
    last_frame_time: float = 0.0
    latency_ms: float = 0.0
    error_message: str = ""
    
    @property
    def drop_rate(self) -> float:
        """Calculate frame drop rate."""
        total = self.frame_count + self.drop_count
        return self.drop_count / total if total > 0 else 0.0


@dataclass
class StreamConfig:
    """Configuration for a video stream."""
    path: str
    camera_id: str
    queue_size: int = 30
    reconnect_delay: float = 2.0
    max_reconnects: int = 10
    target_fps: Optional[float] = None
    resize: Optional[Tuple[int, int]] = None  # (width, height)


class VideoStream:
    """
    Single video stream with bounded queue and auto-reconnect.
    Runs frame capture in a separate thread to prevent blocking.
    """
    
    def __init__(self, config: StreamConfig):
        """
        Initialize video stream.
        
        Args:
            config: Stream configuration
        """
        self.config = config
        self.camera_id = config.camera_id
        self.path = config.path
        
        # Frame queue with bounded size (drops old frames when full)
        self.queue: Queue = Queue(maxsize=config.queue_size)
        
        # Stream state
        self.health = StreamHealth()
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # FPS tracking
        self._frame_times: List[float] = []
        self._last_read_time = 0.0
        
    def start(self):
        """Start the stream capture thread."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started stream: {self.camera_id} ({self.path})")
    
    def stop(self):
        """Stop the stream capture thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        self.health.status = StreamStatus.DISCONNECTED
        logger.info(f"Stopped stream: {self.camera_id}")
    
    def read(self, timeout: float = 0.1) -> Optional[Any]:
        """
        Read the latest frame from the queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Frame as numpy array, or None if no frame available
        """
        try:
            frame = self.queue.get(timeout=timeout)
            self._last_read_time = time.time()
            return frame
        except Empty:
            return None
    
    def get_latest(self) -> Optional[Any]:
        """Get the most recent frame, discarding older ones."""
        frame = None
        try:
            while True:
                frame = self.queue.get_nowait()
        except Empty:
            pass
        return frame
    
    @property
    def fps(self) -> float:
        """Get stream FPS."""
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 25.0
        return 25.0
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get stream resolution (width, height)."""
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        reconnect_count = 0
        
        while self._running:
            # Connect to stream
            if not self._connect():
                reconnect_count += 1
                self.health.reconnect_count = reconnect_count
                
                if reconnect_count >= self.config.max_reconnects:
                    self.health.status = StreamStatus.FAILED
                    self.health.error_message = "Max reconnection attempts reached"
                    logger.error(f"Stream {self.camera_id}: max reconnects reached")
                    break
                
                time.sleep(self.config.reconnect_delay)
                continue
            
            reconnect_count = 0
            
            # Frame capture loop
            while self._running and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Stream {self.camera_id}: failed to read frame")
                    self.health.status = StreamStatus.RECONNECTING
                    break
                
                # Resize if configured
                if self.config.resize is not None:
                    frame = cv2.resize(frame, self.config.resize)
                
                # Update health metrics
                now = time.time()
                self._update_fps(now)
                self.health.frame_count += 1
                self.health.last_frame_time = now
                
                # Add to queue, drop oldest if full
                try:
                    self.queue.put_nowait(frame)
                except Full:
                    # Drop oldest frame
                    try:
                        self.queue.get_nowait()
                        self.health.drop_count += 1
                    except Empty:
                        pass
                    self.queue.put_nowait(frame)
                
                # Frame rate limiting
                if self.config.target_fps is not None:
                    target_interval = 1.0 / self.config.target_fps
                    elapsed = time.time() - now
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)
            
            # Release and prepare for reconnect
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            if self._running:
                self.health.status = StreamStatus.RECONNECTING
                time.sleep(self.config.reconnect_delay)
    
    def _connect(self) -> bool:
        """Attempt to connect to the video source."""
        self.health.status = StreamStatus.CONNECTING
        
        try:
            # Handle different source types
            path = self.path
            if path.isdigit():
                path = int(path)
            
            self.cap = cv2.VideoCapture(path)
            
            # Set buffer size for RTSP streams
            if isinstance(self.path, str) and self.path.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                self.health.error_message = f"Failed to open: {self.path}"
                logger.warning(f"Stream {self.camera_id}: {self.health.error_message}")
                return False
            
            self.health.status = StreamStatus.CONNECTED
            self.health.error_message = ""
            logger.info(f"Stream {self.camera_id}: connected ({self.resolution})")
            return True
            
        except Exception as e:
            self.health.error_message = str(e)
            self.health.status = StreamStatus.FAILED
            logger.error(f"Stream {self.camera_id}: connection error: {e}")
            return False
    
    def _update_fps(self, now: float):
        """Update FPS calculation."""
        self._frame_times.append(now)
        # Keep only last second of frame times
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        self.health.fps = len(self._frame_times)


class MultiVideoReader:
    """
    Multi-camera video reader with synchronized frame retrieval.
    Manages multiple VideoStream instances.
    """
    
    def __init__(self, inputs: List[Dict[str, Any]]):
        """
        Initialize multi-video reader.
        
        Args:
            inputs: List of input configurations with 'path' and 'camera_id' keys
        """
        self.streams: Dict[str, VideoStream] = {}
        
        for inp in inputs:
            config = StreamConfig(
                path=str(inp.get('path', '')),
                camera_id=inp.get('camera_id', f'cam_{len(self.streams)}'),
                queue_size=inp.get('queue_size', 30),
                reconnect_delay=inp.get('reconnect_delay', 2.0),
                max_reconnects=inp.get('max_reconnects', 10),
                target_fps=inp.get('target_fps'),
                resize=inp.get('resize'),
            )
            stream = VideoStream(config)
            self.streams[config.camera_id] = stream
        
        # Start all streams
        for stream in self.streams.values():
            stream.start()
    
    def read(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Read frames from all cameras.
        
        Args:
            timeout: Maximum time to wait for frames
            
        Returns:
            Dict mapping camera_id to frame, or None if all streams ended
        """
        frames = {}
        any_active = False
        
        for cam_id, stream in self.streams.items():
            if stream.health.status in (StreamStatus.CONNECTED, StreamStatus.RECONNECTING):
                any_active = True
                frame = stream.read(timeout=timeout / len(self.streams))
                if frame is not None:
                    frames[cam_id] = frame
        
        if not any_active and len(frames) == 0:
            return None
        
        return frames if frames else {}
    
    def fps(self, cam_id: str) -> float:
        """Get FPS for a specific camera."""
        stream = self.streams.get(cam_id)
        return stream.fps if stream else 25.0
    
    def get_health(self) -> Dict[str, StreamHealth]:
        """Get health metrics for all streams."""
        return {cam_id: stream.health for cam_id, stream in self.streams.items()}
    
    def close(self):
        """Stop all streams and release resources."""
        for stream in self.streams.values():
            stream.stop()
        logger.info("All streams closed")


class AdaptiveResolutionManager:
    """
    Manages adaptive resolution scaling based on processing latency.
    Reduces resolution when latency exceeds budget.
    """
    
    def __init__(
        self,
        base_resolution: Tuple[int, int] = (1280, 720),
        min_resolution: Tuple[int, int] = (640, 360),
        target_latency_ms: float = 100.0,
        scale_steps: List[float] = [1.0, 0.75, 0.5],
    ):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.target_latency_ms = target_latency_ms
        self.scale_steps = sorted(scale_steps, reverse=True)
        
        self.current_scale = 1.0
        self._latency_history: List[float] = []
    
    def update(self, latency_ms: float) -> Tuple[int, int]:
        """
        Update with current latency and return recommended resolution.
        
        Args:
            latency_ms: Current processing latency in milliseconds
            
        Returns:
            Recommended (width, height) resolution
        """
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 30:
            self._latency_history.pop(0)
        
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        # Adjust scale based on latency
        if avg_latency > self.target_latency_ms * 1.5:
            # Scale down
            current_idx = self.scale_steps.index(self.current_scale) if self.current_scale in self.scale_steps else 0
            if current_idx < len(self.scale_steps) - 1:
                self.current_scale = self.scale_steps[current_idx + 1]
        elif avg_latency < self.target_latency_ms * 0.5:
            # Scale up
            current_idx = self.scale_steps.index(self.current_scale) if self.current_scale in self.scale_steps else len(self.scale_steps) - 1
            if current_idx > 0:
                self.current_scale = self.scale_steps[current_idx - 1]
        
        w = max(int(self.base_resolution[0] * self.current_scale), self.min_resolution[0])
        h = max(int(self.base_resolution[1] * self.current_scale), self.min_resolution[1])
        
        return (w, h)
    
    @property
    def current_resolution(self) -> Tuple[int, int]:
        """Get current resolution."""
        return (
            int(self.base_resolution[0] * self.current_scale),
            int(self.base_resolution[1] * self.current_scale)
        )
