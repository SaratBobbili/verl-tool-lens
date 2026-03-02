""" 
Tests the function which takes batches output by the aggregator and converts them into DataProto objects which
are used by the update_teacher function.

TODO: This file is useless right now since it overlaps a lot with test_special_dp_teacher_train.py. I will use
it later to test converting actual output from Aggregator into DataProto and running a training step with it,
but I can't do that until I have a way to load data samples by their unique IDs.
"""
import numpy as np
from verl import DataProto


def test_construct_from_dummy_data():
    # Construct dummy batch data; input_ids will be a random tensor of shape (batch_size, pad_len), the 
    # attention mask and response_mask will be 1s for all tokens, and position_ids will be a range from 
    # 0 to pad_len-1. The scores will be a random tensor of shape (batch_size,).
    batch_size = 4
    pad_len = 8

    input_ids = [[i + j for j in range(pad_len)] for i in range(batch_size)]
    attention_mask = [[1] * pad_len for _ in range(batch_size)]
    position_ids = [[j for j in range(pad_len)] for _ in range(batch_size)]
    response_mask = [[1] * pad_len for _ in range(batch_size)]
    scores = np.random.uniform(size=(batch_size,)).tolist()

    data = DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "scores": scores,
            },
            meta_info={},
        )

    # Now test running a training step with this data
