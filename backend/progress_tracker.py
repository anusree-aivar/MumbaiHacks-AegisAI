"""
Progress Tracking System for Verification Workflow

This module provides a thread-safe progress tracking system that allows
the verification workflow to report progress updates in real-time.
"""

import threading
from typing import Dict, Optional, Callable
from datetime import datetime


class ProgressTracker:
    """
    Thread-safe progress tracker for verification workflow.
    
    Tracks progress through different stages of the verification process
    and allows callbacks to be registered for progress updates.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._progress = 0
        self._stage = "Initializing..."
        self._callbacks = []
        self._details = {}
    
    def update(self, progress: int, stage: str, details: Optional[Dict] = None):
        """
        Update progress and notify callbacks.
        
        Args:
            progress: Progress percentage (0-100)
            stage: Current stage description
            details: Optional additional details
        """
        with self._lock:
            self._progress = min(100, max(0, progress))
            self._stage = stage
            self._details = details or {}
            
            # Notify all callbacks
            for callback in self._callbacks:
                try:
                    callback(self._progress, self._stage, self._details)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
    
    def get_status(self) -> Dict:
        """Get current progress status."""
        with self._lock:
            return {
                "progress": self._progress,
                "stage": self._stage,
                "details": self._details.copy(),
                "timestamp": datetime.now().isoformat()
            }
    
    def register_callback(self, callback: Callable):
        """Register a callback to be notified of progress updates."""
        with self._lock:
            self._callbacks.append(callback)
    
    def clear_callbacks(self):
        """Clear all registered callbacks."""
        with self._lock:
            self._callbacks.clear()


# Global progress tracker instance
_global_tracker: Optional[ProgressTracker] = None
_tracker_lock = threading.Lock()


def get_progress_tracker() -> ProgressTracker:
    """Get or create the global progress tracker instance."""
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = ProgressTracker()
        return _global_tracker


def reset_progress_tracker():
    """Reset the global progress tracker."""
    global _global_tracker
    with _tracker_lock:
        if _global_tracker is not None:
            # Reset the existing instance instead of creating a new one
            with _global_tracker._lock:
                _global_tracker._progress = 0
                _global_tracker._stage = "Initializing..."
                _global_tracker._details = {}
        else:
            _global_tracker = ProgressTracker()
