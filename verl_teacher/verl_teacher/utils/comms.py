""" Utility classes and functions for communication between the teacher and the student processes """
from typing import Any, Dict, List, Optional, Tuple
import ray
import time

__all__ = ["Aggregator", "Sender"]

@ray.remote
class Aggregator:
    """
    Aggregates incoming dict messages into fixed-size batches.
    Parent polls via poll_batch() (non-blocking: returns None if nothing ready).
    """

    def __init__(self, batch_size: int, max_buffer: int = 100_000):
        self.batch_size = int(batch_size)
        self.max_buffer = int(max_buffer)

        self.buf: List[Dict[str, Any]] = []
        self.ready: List[Tuple[int, List[Dict[str, Any]]]] = []
        self._next_batch_id = 0
        self._seen_total = 0  # total items accepted

    def add(self, d: Dict[str, Any]) -> Dict[str, int]:
        payload = d.get("payload", {})
        # flatten dict → items
        for k, v in payload.items():
            self.buf.append((k, v))
        self._seen_total += len(payload)

        # Form as many full batches as possible
        while len(self.buf) >= self.batch_size:
            batch = self.buf[: self.batch_size]
            del self.buf[: self.batch_size]
            bid = self._next_batch_id
            self._next_batch_id += 1
            self.ready.append((bid, batch))

        # Avoid unbounded growth if producers outrun consumer
        if len(self.buf) > self.max_buffer:
            drop = len(self.buf) - self.max_buffer
            del self.buf[:drop]

        return {
            "buffered": len(self.buf),
            "ready": len(self.ready),
            "seen_total": self._seen_total,
            "next_batch_id": self._next_batch_id,
        }

    def poll_batch(self) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
        if not self.ready:
            return None
        return self.ready.pop(0)

    def flush_partial(self) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
        """
        Optional helper: force-return a final partial batch (if any).
        Useful for tests / shutdown behavior.
        """
        if self.buf:
            bid = self._next_batch_id
            self._next_batch_id += 1
            batch = self.buf
            self.buf = []
            return (bid, batch)
        return None

    def stats(self) -> Dict[str, int]:
        return {
            "buffered": len(self.buf),
            "ready": len(self.ready),
            "seen_total": self._seen_total,
            "next_batch_id": self._next_batch_id,
        }

import json, os
# @ray.remote
class OfflineAggregator:
    """
    This is the version of Aggregator used when the teacher is running in "offline" mode, where it 
    reads from a static dataset instead of receiving messages from student processes.
    """
    def __init__(self, data_dir: str, batch_size: int):
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Each file of training data corresponds to a different training step
        self.curr_step = 1
        # Initialize a buffer with no size limit; we will only add one file's worth of data at a time, 
        # and refill the buffer any time its size drops below the batch size.
        self.buf: List[Dict[str, Any]] = []
        self.add_next_step_data()

    def add_next_step_data(self):
        """Fill the buffer with data from a list of files."""
        file_path = os.path.join(self.data_dir, f"step_{self.curr_step}.jsonl")
        if not os.path.exists(file_path):
            return False
        with open(file_path, "r") as f:
            self.buf.extend([json.loads(line) for line in f.readlines()])
            self.curr_step += 1
            return True

    def poll_batch(self) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
        if len(self.buf) < self.batch_size:
            # Try to add more data if available
            if self.add_next_step_data():
                pass
            else:
                # No more data available, return what we have (even if it's less than a full batch)
                if self.buf:
                    batch = self.buf
                    self.buf = []
                    return (self.curr_step - 1, batch)
                else:
                    return None
        # Return the next batch
        batch = self.buf[: self.batch_size]
        self.buf = self.buf[self.batch_size :]
        return (self.curr_step - 1, batch)
    
    def get_buf(self) -> List[Dict[str, Any]]:
        """Helper method for debugging: return the current buffer contents."""
        return self.buf

    def add(self):
        raise NotImplementedError("OfflineAggregator does not support add()")
    
    def flush_partial(self):
        raise NotImplementedError("OfflineAggregator does not support flush_partial()")