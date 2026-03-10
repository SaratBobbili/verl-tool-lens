""" Utility classes and functions for communication between the teacher and the student processes """
import socket
from typing import Any, Dict, List, Optional, Tuple
import ray
import time
from collections import deque
import uuid

__all__ = ["Aggregator", "OfflineAggregator", "Dispatcher"]

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
        self._next_batch_id = 0 # NOTE: This is replaced with the step index in OfflineAggregator (which starts at 1 instead of 0)
        self._seen_total = 0  # total items accepted

        self.instance_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        print(
            f"[Aggregator.__init__] instance_id={self.instance_id} "
            f"pid={self.pid} host={self.hostname} batch_size={self.batch_size}"
        )

    def identity(self):
        return {
            "instance_id": self.instance_id,
            "pid": self.pid,
            "hostname": self.hostname,
            "batch_size": self.batch_size,
        }

    def add(self, d: Dict[str, Any]) -> Dict[str, int]:
        # TODO: In the future, when there are two students at once, we will need the input d to have a 
        # student_id field so that we can maintain separate buffers for each student.
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
        if len(self.buf) > 0:
            breakpoint()
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

@ray.remote 
class Dispatcher:
    """
    Maintains a set of curated batches of training data for each student, and dispatches them upon request.
    Note that there will be some communication overhead in the request/response compared to giving each student
    their own actor for this purpose, but the method of using a single actor on the teacher side makes it easy 
    to ensure that the freshest batches are always sent to the students.
    The batches ready to be dispatched are stored in a separate deque for each student, and the most recently
    added batch is always dispatched first to maximize freshness.
    """
    def __init__(self):
        self._counter = 0  # simple counter to generate unique batch IDs
        self.batches_available = {}
        self.maxlen = 100  # absolute max number of batches to keep in memory for each student; old batches will be dropped if this is exceeded

    def register_student(self, student_id: str):
        """ Used by students to register themselves with the teacher. """
        assert student_id in [0, 1], "Students ID should be 0 for the weak student and 1 for the strong student"
        assert student_id not in self.batches_available.keys(), f"Student {student_id} is already registered"
        self.batches_available[student_id] = deque(maxlen=self.maxlen)

    def submit_batch(self, batch_desc: Dict[str, Any]):
        """
        Used by the teacher to submit a new batch descriptor to be dispatched to students upon request.
        The batch_desc dict should contain the batch itself and the ID of the student it is intended for
        """
        # print(f"Received new batch descriptor: {batch_desc}")
        # TODO: include the step that the student model was on when the batch was generated (determined
        # based on the step numbers received through the Aggregator) so that the student process can determine
        # if a batch is too stale
        student_id = batch_desc.get("student_id")
        assert student_id in self.batches_available.keys(), f"Student {student_id} is not registered"
        self.batches_available[student_id].append(batch_desc)

    def request_batches(self, student_id: str, need_k: int, *, student_step: int = -1, stats: dict | None = None):
        """
        Used by students
        Return 'need_k' batch descriptors. Keep this FAST.
        Prefer returning IDs/refs, not large tensors.
        """
        out = []
        for i in range(need_k):
            batch = self.batches_available[student_id].pop() if self.batches_available[student_id] else None
            if batch is None:
                break
            self._counter += 1
            out.append(
                {
                    "student_step": student_step,
                    "batch_id": self._counter,
                    "batch": batch,
                }
            )
        return out