import collections
from typing import Optional

import ray

from verl.protocol import DataProto

HRL_ROLE_REPLAY_NAME = "hrl_role_replay"


@ray.remote
class HRLRoleReplay:
    """Per-role in-memory buffer for HRL rollouts."""

    def __init__(self) -> None:
        self._queues: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    def push_batch(self, data: DataProto) -> None:
        """Push every sample from the batch into its role queue."""
        for idx in range(len(data)):
            item = data[idx : idx + 1]
            role_field = item.non_tensor_batch.get("model_role")
            role = "selector"
            if role_field is not None and len(role_field) > 0:
                role = str(role_field[0])
            self._queues[role].append(item)

    def pop(self, role: str, max_batch: int = 1) -> list[DataProto]:
        """Pop up to max_batch samples for a role."""
        if max_batch <= 0:
            return []
        queue = self._queues.get(role)
        if queue is None:
            return []
        popped: list[DataProto] = []
        for _ in range(min(max_batch, len(queue))):
            popped.append(queue.popleft())
        return popped

    def size(self, role: Optional[str] = None):
        """Report queue sizes."""
        if role is None:
            return {key: len(queue) for key, queue in self._queues.items()}
        return len(self._queues.get(role, ()))
