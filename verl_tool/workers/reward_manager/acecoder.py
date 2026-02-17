# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
import regex as re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
import time
import torch
import signal

from verl import DataProto
from .reward_score import _default_compute_score

import asyncio
from verl.utils.reward_score.prime_code import compute_score as prime_code_compute_score
from verl.workers.reward_manager.prime import parallel_compute_score_async
from verl.workers.reward_manager import register

import hashlib
import random
import os
import json
import subprocess
import time
import ast
from pathlib import Path
from collections import Counter

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

def check_syntax(code_string):
    try:
        # Attempt to parse the code string
        ast.parse(code_string)
        return True
    # except SyntaxError as e:
    except Exception as e:
        # If a SyntaxError is raised, the code is not valid
        # print(f"Syntax error in code: {e}")
        return False
    
def parse_code(action: str, mode="all"):
    """
    Parse the raw action string (which is the llm response) into an actual action and its contents.
    Ensures that the parsed code is valid and safe for execution.
    
    Args:
        action: Raw action string containing Python code
        
    Returns:
        Tuple containing the extracted code and a validity flag
    """
    # Try to find Python code in various formats
    all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
    
    if not all_valid_python_code:
        all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
    
    if len(all_valid_python_code) == 0:
        return ""
    
    if mode == "all":
        parsed_code = "\n".join([code for code in all_valid_python_code if check_syntax(code)])
    elif mode == "first":
        # Use the first code block found
        parsed_code = all_valid_python_code[0]
    elif mode == "last":
        # Use the last code block found
        parsed_code = all_valid_python_code[-1]
    elif mode == "all_in_last_turn":
        # parse all the code blocks only in the last assistant turn
        # find the last assistant turn
        last_turn_start_idx = action.rfind('<|im_start|>assistant')
        if last_turn_start_idx == -1:
            last_turn = action
        else:
            last_turn = action[last_turn_start_idx:]
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", last_turn, re.DOTALL)
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", last_turn, re.DOTALL)
        if len(all_valid_python_code) == 0:
            return ""
        parsed_code = "\n".join([code for code in all_valid_python_code if check_syntax(code)])
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'all', 'first', 'last', or 'all_in_last_turn'.")
    
    parsed_code = parsed_code.strip(' \n')
    return parsed_code

def prime_code_compute_score_async(data_source, solution_str, ground_truth, extra_info=None):
    res = prime_code_compute_score(solution_str, ground_truth, continuous=True)
    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

@register("acecoder")
class AceCoderRewardManager:
    """
    The Reward Manager used in https://github.com/TIGER-AI-Lab/AceCoder
    """
    name = "acecoder"
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.step_idx = None
        self.n_workers = 1
        self.binary = True
        self.parse_code_mode = "last" # "all", "first", "last"
        self.add_format_think_penalty = False # -0.5 if not begines with <think> and end with </think>
        self.add_format_answer_penalty = False # -0.5 if not having <answer> </answer>
        self.add_valid_action_penalty = False # -1.0 if num turns > 0 not action not valid
        self.add_unfinished_traj_penalty = False # -0.25 if the traj is not finished
        self.add_no_tool_interact_penalty = False # -1.0 if the traj's num turn is 0, no interaction at all
        self.add_code_exec_penalty = False # -0.25 if the execution has an error.
        self.reward_fn_key = reward_fn_key
        self.record_dir = Path("/home/codeGen_curriculum/verl_step_records/acecoder_records")
        self.record_dir.mkdir(parents=True, exist_ok=True)

        try:
            from acecoder import evaluate_test_cases
        except ImportError:
            raise ImportError("`from acecoder import evaluate_test_cases` failed, please install acecoder to use test_case rule")

    def get_acecoder_data_score(self, data: DataProto, response_str, prompt_str, extracted_answers, test_cases):
        scores = [{} for _ in range(len(data))]
        data_sources = data.non_tensor_batch['data_source']

        question_hashes = [hash_string(question) for question in prompt_str]
        assert len(response_str) == len(test_cases) == len(data_sources)

        samples = [
            {
                'task_id': question_hash,
                'prompt': question,
                'output': answer,
                'original_response': response,
                'tests': list(test_case),
                '_identifier': f"{question_hash}_{i}"
            }
            for i, (question_hash, question, answer, test_case, response)
            in enumerate(zip(question_hashes, prompt_str, extracted_answers, test_cases, response_str))
        ]

        temp_file = self.record_dir / f"step-{self.step_idx}_{hash_string(''.join(question_hashes))}_{len(response_str[0])}_{time.time_ns()}.jsonl"
        with open(temp_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        output_file = Path(temp_file).with_suffix(".eval_results_binary.jsonl").absolute()

        # Store logs next to outputs for post-mortem debugging.
        log_file = Path(str(output_file) + ".log")

        # Build argv list (NO shell=True)
        argv = [
            "python", "-m", "acecoder.eval_test_cases",
            "--samples", str(temp_file),
            "--n_workers", str(self.n_workers),
            "--extract_solution", "True",
            "--output_file", str(output_file),
            "--test_details", "True",
            "--i_just_wanna_run", "True",
            "--min_time_limit", "1",
            "--gt_time_limit_factor", "1",
        ]

        # Choose a timeout. Prefer a config on self, else a reasonable default.
        # You can tune this. The important part is: it must exist.
        timeout_s = getattr(self, "acecoder_timeout_s", None)
        if timeout_s is None:
            # Heuristic: base + per-sample. Adjust to your workload.
            timeout_s = 30 + 5 * len(samples)

        start = time.time()
        rc = None
        timed_out = False

        # Open logs in append mode so repeated runs don't lose context.
        with open(log_file, "ab", buffering=0) as lf:
            lf.write(
                (f"\n=== step={self.step_idx} start={time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"samples={len(samples)} timeout_s={timeout_s} ===\n"
                f"CMD: {' '.join(argv)}\n").encode()
            )

            # Start subprocess in its own process group so we can kill all descendants.
            # preexec_fn=os.setsid works on Linux.
            proc = subprocess.Popen(
                argv,
                stdout=lf,
                stderr=lf,
                preexec_fn=os.setsid,
                cwd=None,
                env=os.environ.copy(),
            )

            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                lf.write(f"\n[TIMEOUT] Exceeded {timeout_s}s. Killing process group...\n".encode())

                # Kill the whole process group.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception as e:
                    lf.write(f"[WARN] SIGTERM killpg failed: {e}\n".encode())

                # Give it a moment to die gracefully, then SIGKILL.
                try:
                    rc = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except Exception as e:
                        lf.write(f"[WARN] SIGKILL killpg failed: {e}\n".encode())
                    rc = proc.wait(timeout=5)
            except Exception as e:
                # Any other unexpected error: ensure process group is cleaned up.
                lf.write(f"\n[ERROR] Exception while waiting: {e}\n".encode())
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                raise
            finally:
                end = time.time()
                lf.write(f"=== end rc={rc} elapsed={end-start:.2f}s timed_out={timed_out} ===\n".encode())

        print(f"Step {self.step_idx}: acecoder evaluation script took {time.time() - start:.2f} seconds for {len(samples)} samples. rc={rc} timeout={timed_out}")

        # If timed out or failed, return a safe fallback score and keep logs for inspection.
        if timed_out or rc != 0 or not output_file.exists():
            # Conservative fallback: treat as failure. (You can choose neutral instead.)
            for i in range(len(scores)):
                scores[i]['pass_rate'] = 0.0
                scores[i]['binary_pass_rate'] = 0.0
                scores[i]['score'] = -1.0 if getattr(self, "binary", False) else 0.0
                scores[i]['acecoder_error'] = "timeout" if timed_out else f"rc={rc}"
                scores[i]['acecoder_log'] = str(log_file)
            # Cleanup temp input to avoid disk bloat; keep output/log for debugging.
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return scores

        # Parse output file
        with open(output_file, "r") as f:
            all_samples_results = [json.loads(x) for x in f]

        pass_rates = [x['eval_results']['pass_rate'] for x in all_samples_results]
        code_error = [x['eval_results']['code_error'] for x in all_samples_results]
        test_case_error = [
            [x['eval_results']['details'][i]['reason'] for i in range(len(x['eval_results']['details']))]
            for x in all_samples_results
        ]

        print(f"Step {self.step_idx}: acecoder evaluation script error statistics for {len(samples)} samples.")
        num_empty = sum(1 for code in extracted_answers if code.strip(' \n') == '')
        if len(extracted_answers) > 0:
            print(f" - Empty code: {num_empty} ({num_empty / len(extracted_answers) * 100:.2f}%)")
        if len(code_error) > 0:
            print(f" - Syntax error: {sum(1 for x in code_error if x)} ({len([x for x in code_error if x]) / len(code_error) * 100:.2f}%)")
        print(" - Test case error:")
        counter = Counter()
        for errs in test_case_error:
            if errs:
                counter.update(errs)
        if len(test_case_error) > 0:
            for k, v in counter.items():
                print(f"   - {k}: {v} ({v / len(test_case_error) * 100:.2f}%)")

        # Cleanup. Keep log_file by default (it’s valuable); remove if you prefer.
        try:
            os.remove(temp_file)
            os.remove(output_file)
            os.remove(log_file)  # uncomment if you want logs removed on success
        except Exception:
            pass

        for i in range(len(scores)):
            scores[i]['pass_rate'] = pass_rates[i]
            scores[i]['binary_pass_rate'] = 1.0 if pass_rates[i] == 1.0 else 0.0
            if getattr(self, "binary", False):
                scores[i]['score'] = 1.0 if pass_rates[i] == 1.0 else -1.0
            else:
                scores[i]['score'] = pass_rates[i]

        return scores

    def get_prime_code_data_score(self, data: DataProto, response_str, prompt_str, extracted_answers, test_cases):
        scores = [{} for _ in range(len(data))]
        data_sources = data.non_tensor_batch['data_source']
        
        sequences_str = extracted_answers
        ground_truth = test_cases
        data_sources = ["taco"] * len(sequences_str)
        extra_info = [None] * len(sequences_str)
        pass_rates = asyncio.run(
            parallel_compute_score_async(
                prime_code_compute_score_async,
                sequences_str,
                ground_truth,
                data_sources,
                extra_info=extra_info,
                num_processes=self.n_workers,
            )
        ) # list of 1.0 or 0.0
        for i in range(len(scores)):
            scores[i]['pass_rate'] = pass_rates[i]
            scores[i]['binary_pass_rate'] = 1.0 if pass_rates[i] == 1.0 else 0.0
            if self.binary:
                scores[i]['score'] = 1.0 if pass_rates[i] == 1.0 else -1.0
            else:
                scores[i]['score'] = pass_rates[i]
        return scores
    
    def add_additional_penalties(self, response: str, data_i, scores_i: dict):
        # 1.4 format penalty
        if self.add_format_think_penalty:
            match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if not match or not response.startswith("<think>") or response.count("<think>") != 1 or response.count("</think>") != 1:
                scores_i['score'] -= 0.5
                scores_i['think_format_penalty'] = 1
            else:
                scores_i['think_format_penalty'] = 0
        if self.add_format_answer_penalty:
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if not match or not response.endswith("</answer>") or response.count("<answer>") != 1 or response.count("</answer>") != 1:
                scores_i['score'] -= 0.5
                scores_i['answer_format_penalty'] = 1
            else:
                scores_i['answer_format_penalty'] = 0
        if "turns_stats" in data_i.non_tensor_batch:
            if self.add_valid_action_penalty:
                num_turn = data_i.non_tensor_batch["turns_stats"]
                num_valid_action = data_i.non_tensor_batch["valid_action_stats"]
                if num_valid_action < num_turn:
                    scores_i['score'] -= 1.0 
                    scores_i['valid_action_penalty'] = 1
                else:
                    scores_i['valid_action_penalty'] = 0
            if self.add_unfinished_traj_penalty:
                is_active = data_i.non_tensor_batch["active_mask"]
                if is_active:
                    scores_i['score'] -= 0.25
                    scores_i['unfinished_traj_penalty'] = 1
                else:
                    scores_i['unfinished_traj_penalty'] = 0
            if self.add_no_tool_interact_penalty:
                num_valid_action = data_i.non_tensor_batch["valid_action_stats"]
                if num_valid_action == 0:
                    scores_i['score'] -= 1.0
                    scores_i['no_tool_interact_penalty'] = 1
                else:
                    scores_i['no_tool_interact_penalty'] = 0
            if self.add_code_exec_penalty:
                keywords = ["ERROR:\nTraceback", "Execution timed out"]
                if any(keyword in response for keyword in keywords):
                    scores_i['score'] -= 0.25
                    scores_i['exec_error'] = 1
                else:
                    scores_i['exec_error'] = 0
        
        return scores_i
        
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        # TODO: implement new reward computing & statistic mechanism
        scores = [{} for _ in range(len(data))]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        if "turns_stats" in data.non_tensor_batch:
            num_turn = data.non_tensor_batch["turns_stats"]
            num_valid_action = data.non_tensor_batch["valid_action_stats"]
            is_active = data.non_tensor_batch["active_mask"]
            is_done = [not is_active[i] for i in range(len(is_active))]
            
        already_print_data_sources = {}
        
        # retrieve the list of prompt_token_ids and their length
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        # retrieve the list of response ids and their valid length
        response_ids = data.batch['responses']
        valid_prompt_length = data.batch['attention_mask'][:, :prompt_length].sum(dim=-1)
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        
        # with open("test.json", 'w') as f:
        #     # batch decode the list of responses and prompts
        #     response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=False)
        #     prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)
        #     json.dump({
        #         "response_ids": response_ids.tolist(),
        #         "prompt_ids": prompt_ids.tolist(),
        #         "response_str": response_str,
        #         "prompt_str": prompt_str,  
        #     }, f, indent=4)
            
        # batch decode the list of responses and prompts
        response_str = [self.tokenizer.decode(response_ids[i][:valid_response_length[i].item()], skip_special_tokens=False) for i in range(len(data))]
        prompt_str = [self.tokenizer.decode(prompt_ids[i][-valid_prompt_length[i].item():], skip_special_tokens=False) for i in range(len(data))]
        # response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        
        # extract the answer for the list of responses
        extracted_answers = [re.sub(r"<think>(.|\n)*?</think>", "", response) for response in response_str]
        extracted_answers = [parse_code(response, self.parse_code_mode) for response in extracted_answers]
        
        # retrieve the list of ground truths/test cases
        test_cases = []
        acecoder_data_idxs = []
        prime_code_data_idxs = []
        for i in range(len(data)):
            if data[i].non_tensor_batch['extra_info'].get("inputs_outputs"):
                test_cases.append(data[i].non_tensor_batch['extra_info']['inputs_outputs'])
                prime_code_data_idxs.append(i)
            elif data[i].non_tensor_batch['extra_info'].get("test_cases"):
                test_cases.append(data[i].non_tensor_batch['extra_info']['test_cases'])
                acecoder_data_idxs.append(i)
            else:
                raise ValueError(f"Cannot find test cases for data {i} in {data[i].non_tensor_batch['extra_info']}")

        # 1.1 process acecoder data
        if len(acecoder_data_idxs) > 0:
            acecoder_data = data[acecoder_data_idxs]
            acecoder_response_str = [response_str[i] for i in acecoder_data_idxs]
            acecoder_prompt_str = [prompt_str[i] for i in acecoder_data_idxs]
            acecoder_test_cases = [test_cases[i] for i in acecoder_data_idxs]
            acecoder_extracted_answers = [extracted_answers[i] for i in acecoder_data_idxs]
            acecoder_scores = self.get_acecoder_data_score(acecoder_data, acecoder_response_str, acecoder_prompt_str, acecoder_extracted_answers, acecoder_test_cases)
            print(f"Step {self.step_idx}: {len(acecoder_data_idxs)} acecoder data scores")
            print(" - Average pass rate: ", sum([x['pass_rate'] for x in acecoder_scores]) / len(acecoder_scores))
            print(" - Average binary pass rate: ", sum([x['binary_pass_rate'] for x in acecoder_scores]) / len(acecoder_scores))
            print(" - Average score: ", sum([x['score'] for x in acecoder_scores]) / len(acecoder_scores))
        else:
            acecoder_scores = []
        
        # 1.2 
        if len(prime_code_data_idxs) > 0:
            prime_code_data = data[prime_code_data_idxs]
            prime_code_response_str = [response_str[i] for i in prime_code_data_idxs]
            prime_code_prompt_str = [prompt_str[i] for i in prime_code_data_idxs]
            prime_code_test_cases = [test_cases[i] for i in prime_code_data_idxs]
            prime_code_extracted_answers = [extracted_answers[i] for i in prime_code_data_idxs]
            prime_code_scores = self.get_prime_code_data_score(prime_code_data, prime_code_response_str, prime_code_prompt_str, prime_code_extracted_answers, prime_code_test_cases)
            print(f"Step {self.step_idx}: {len(prime_code_data_idxs)} prime code data scores")
            print(" - Average pass rate: ", sum([x['pass_rate'] for x in prime_code_scores]) / len(prime_code_scores))
            print(" - Average binary pass rate: ", sum([x['binary_pass_rate'] for x in prime_code_scores]) / len(prime_code_scores))
            print(" - Average score: ", sum([x['score'] for x in prime_code_scores]) / len(prime_code_scores))
        else:
            prime_code_scores = []
        
        # 1.3 merge the scores
        idxs_map = sorted([(idx, i, 'acecoder') for i, idx in enumerate(acecoder_data_idxs)] + [(idx, i, 'prime_code') for i, idx in enumerate(prime_code_data_idxs)], key=lambda x: x[0])
        for i in range(len(data)):
            if idxs_map[i][2] == "acecoder":
                scores[i] = acecoder_scores[idxs_map[i][1]]
            else:
                scores[i] = prime_code_scores[idxs_map[i][1]]
                
        # 1.4 additional penalty
        for i in range(len(data)):
            scores[i] = self.add_additional_penalties(response_str[i], data[i], scores[i])       
            

        for i, score in enumerate(scores):
            if isinstance(score, dict):
                reward_tensor[i, valid_response_length[i].item() - 1] = score['score']
                for k, v in score.items():
                    reward_extra_info[k].append(v)
            else:
                reward_tensor[i, valid_response_length[i].item() - 1] = score
        
        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor