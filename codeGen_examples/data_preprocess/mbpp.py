# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a copy of the taco.py file in the data_preprocess directory of the main branch, with my edits
(mainly edits to the prompt)
"""
import fire
import os
import datasets, glob
from pathlib import Path

MBPP_prompt_single_turn = """\
You will be given a programming task, and you must write a function to perform that task in a markdown code block like this: ```python\nyour code here\n```. IMPORTANT: your response should only include a single markdown code block. 
"""

# TODO: Adapt this for the case where tests are available
execution_prompt = """\
Answer the given coding question. You must conduct reasoning about the problem and then provide the final program as answer. 
During the thinking process, you can write test cases or test your current solutions using a testing tool. if you want to test any python code, writing it inside ```python and ``` tags following with "```output". 
The code between "```python" and "``````output" will then be executed, and the terminal output (standard output and standard error) will be provided to you. 
Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. 
If you find no further code execution needed, you can then give your final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."
naive_execution_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The Assistant can reason with the help of Python code. If the Assistant wants to run any Python code, it writes it inside ```python and ``` tags, and makes sure to follow it with "```output", meaning that it is requesting the code to be executed. Then the result of execution will be provided to the Assistant between "```output" and "```" for the python code block that it follows. The Assistant can test Python codes as many times as it wants. If the Assistant finds no further code execution needed, it can then give the final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything.
"""



def main(
    local_dir: str = 'data/',
    single_turn: bool = True,
    tests_available: bool = False
):
    local_dir = Path(local_dir) / "MBPP-sanitized"
    if single_turn:
        local_dir = local_dir.parent / (local_dir.name + '-single-turn')
    elif tests_available:
        raise NotImplementedError("tests_available=True is not yet implemented")
        local_dir = local_dir.parent / (local_dir.name + '-with-tests')
        system_prompt = None
    else:
        local_dir = local_dir.parent / (local_dir.name + '-with-execution')
        system_prompt = naive_execution_prompt
    local_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.load_dataset('google-research-datasets/mbpp', "sanitized", split='train+test')
    test_dataset = datasets.load_dataset('google-research-datasets/mbpp', "sanitized", split='validation')

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('prompt')
            # MBPP gives test cases as assert statements
            tests = example['test_list']
            inputs_outputs = None
            data = {
                # Data source does not actually seem to matter
                "data_source": "mbpp",
                "prompt": [
                    {
                        "role": "system",
                        "content": MBPP_prompt_single_turn,
                    },
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ""
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': str(example['task_id']),
                    "question": question_raw,
                    "test_cases": tests,
                    "inputs_outputs": inputs_outputs,
                }
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(test_dataset)} testing samples")
    print(f"Example of a training sample:")
    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved to {len(train_dataset)} training samples to {local_dir}/train.parquet")
    print(f"Saved to {len(test_dataset)} testing samples to {local_dir}/test.parquet")

if __name__ == '__main__':
    fire.Fire(main)
