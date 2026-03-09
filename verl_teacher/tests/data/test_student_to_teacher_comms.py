import time
from typing import Any, Dict, List
import numpy as np

import pytest
import ray

from verl_teacher.utils.comms import Aggregator

@ray.remote
class Sender:
    """
    Simulates a realistic producer actor that sends many messages over time.
    """

    def run(
        self,
        agg,
        sender_id: str,
        n: int,
        msg_batch_size: int = 1,
        start_seq: int = 0,
        jitter_s: float = 0.0,
    ) -> int:
        """
        Sends n dict messages to the Aggregator. Returns n when done.
        """
        for i in range(n):
            random_pass_rates = np.random.uniform(0, 1, size=msg_batch_size)
            msg = {"sender": sender_id, "seq": start_seq + i, "payload": {indx: random_pass_rates[indx] for indx in range(msg_batch_size)}}
            # Fire-and-forget send (still an RPC, but no ray.get here).
            agg.add.remote(msg)

            # Optional jitter to produce more realistic interleavings
            if jitter_s > 0:
                time.sleep(jitter_s)
        return n

@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    # Local Ray runtime for pytest. In CI, keep it lightweight.
    ray.init(
        num_cpus=8,
        include_dashboard=False,
        ignore_reinit_error=True,
        # log_to_driver=False,
    )
    yield
    ray.shutdown()


def _poll_until_done(
    agg,
    expected_total_msgs: int,
    batch_size: int,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Polls agg.poll_batch() until we've collected expected_total_msgs.
    Uses ray.wait(timeout=0) to avoid blocking on slow RPCs.
    """
    collected: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s

    poll_ref = agg.poll_batch.remote()
    while len(collected) < expected_total_msgs:
        if time.monotonic() > deadline:
            stats = ray.get(agg.stats.remote())
            raise TimeoutError(
                f"Timed out polling aggregator: collected={len(collected)}/{expected_total_msgs}, stats={stats}"
            )

        ready, _ = ray.wait([poll_ref], timeout=0.0)
        if not ready:
            time.sleep(poll_interval_s)
            continue

        maybe = ray.get(ready[0])
        poll_ref = agg.poll_batch.remote()  # issue next poll immediately

        if maybe is None:
            time.sleep(poll_interval_s)
            continue

        bid, batch = maybe
        assert isinstance(bid, int)
        assert isinstance(batch, list)

        # Correctness invariants for every returned batch
        # (All batches except possibly a forced flush should be exactly batch_size)
        assert len(batch) == batch_size, f"Got non-full batch from poll_batch: len={len(batch)}"
        collected.extend(batch)

    return collected


def test_aggregator_batches_from_multiple_senders():
    """
    Realistic scenario:
      - Multiple Ray actors (senders) emit dict messages concurrently
      - Aggregator batches them
      - Parent polls without blocking for readiness
      - Verify we receive exactly the expected messages, no loss/duplication
    """
    batch_size = 10
    num_senders = 6
    msgs_per_sender = 37  # intentionally not divisible by batch_size
    msg_batch_size = 2 # Each message is a dict of 2 items, each representing a different data item
    total_msgs = num_senders * msgs_per_sender
    total_items = total_msgs * msg_batch_size

    # The aggregator unpacks each message into msg_batch_size individual items,
    # so batching operates on total_items, not total_msgs.
    # We will only be able to retrieve full batches via poll_batch().
    # So we expect floor(total_items / batch_size) * batch_size through polling,
    # and the remainder will still be buffered (unless we flush explicitly).
    full_batch_items = (total_items // batch_size) * batch_size
    remainder = total_items - full_batch_items

    agg = Aggregator.remote(batch_size=batch_size)

    senders = [Sender.remote() for _ in range(num_senders)]
    # Start all senders concurrently (with a bit of jitter to create interleaving)
    send_refs = [
        s.run.remote(agg, sender_id=f"s{i}", n=msgs_per_sender, msg_batch_size=msg_batch_size, start_seq=0, jitter_s=0.0005)
        for i, s in enumerate(senders)
    ]

    # Poll until all full-batch items are collected
    collected = _poll_until_done(
        agg,
        expected_total_msgs=full_batch_items,
        batch_size=batch_size,
        timeout_s=15.0,
        poll_interval_s=0.005,
    )

    # Ensure senders finished (i.e., aggregator should have seen all total messages)
    sent_counts = ray.get(send_refs)
    # Here we check the number of messages, not the total number of items sent
    assert sum(sent_counts) == total_msgs

    stats = ray.get(agg.stats.remote())
    # Here we consider the total number of items seeen by the aggregator
    assert stats["seen_total"] == total_items
    assert stats["ready"] == 0  # we've drained all full batches
    assert stats["buffered"] == remainder

    # Validate we drained exactly full_batch_items
    assert len(collected) == full_batch_items

    # Now flush the remainder and validate those too
    tail = ray.get(agg.flush_partial.remote())
    if remainder == 0:
        assert tail is None
    else:
        assert tail is not None
        _, tail_batch = tail
        assert len(tail_batch) == remainder
        collected.extend(tail_batch)

    # After flush, no buffered items should remain
    stats2 = ray.get(agg.stats.remote())
    assert stats2["buffered"] == 0
    assert len(collected) == total_items

def test_aggregator_multiple_batches_back_to_back():
    """
    Stress-ish test:
      - Many messages quickly, ensuring the aggregator can queue multiple ready batches
      - Poll drains them correctly
    """
    batch_size = 8
    agg = Aggregator.remote(batch_size=batch_size)

    num_senders = 4
    msgs_per_sender = 64  # divisible by batch_size for clean drain
    msg_batch_size = 2
    total_msgs = num_senders * msgs_per_sender
    total_items = total_msgs * msg_batch_size
    assert total_items % batch_size == 0

    senders = [Sender.remote() for _ in range(num_senders)]
    send_refs = [
        s.run.remote(agg, sender_id=f"s{i}", n=msgs_per_sender, msg_batch_size=msg_batch_size, start_seq=i * 10_000, jitter_s=0.0)
        for i, s in enumerate(senders)
    ]

    collected = _poll_until_done(
        agg,
        expected_total_msgs=total_items,
        batch_size=batch_size,
        timeout_s=15.0,
        poll_interval_s=0.001,
    )

    ray.get(send_refs)

    stats = ray.get(agg.stats.remote())
    assert stats["seen_total"] == total_items
    assert stats["buffered"] == 0
    assert stats["ready"] == 0

    # Verify we collected exactly the expected number of items
    assert len(collected) == total_items
    