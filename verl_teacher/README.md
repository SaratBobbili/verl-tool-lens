# Repository Structure
- experiments/ contains subfolders with standalone experiments for testing the performance of different configurations.
- tests/ contains pytest harnesses for testing the correctness of the verl_teacher code. You will need to have pytest installed in order to run any of these.
- verl_teacher/ contains the source code and config defaults

# Main Entrypoint
The main entrypoint for running the teacher (whether you are training it offline or in parallel with a student model) is verl_teacher/main_teacher.py. This file instantiates the TeacherRunner class (analogous to the RayPPOTrainer) from teacher_runner.py to do training and inference (currently only training is implemented). The launch_teacher.sh script under experiments/offline_training/ gives an example of how to launch main_teacher.py for offline training, and the script under experiments/online_training/ with the same name gives an example of how to launch main_teacher.py for online training.

# Preparing Training Data for the Teacher
In online training, the teacher must wait for new data to come from the student processes before the teacher can be updated. The Aggregator class defined in verl_teacher/utils/comms.py handles this by asynchronously recieving data from the students and allowing the main teacher process to poll it to check whether enough data is available to form a complete training batch. The Aggregator should receive batches with prompt IDs (rather than the prompts themselves) along with empirical success rates and store them in its buffer until they are passed on to the main process. The main process will then convert the prompt IDs into tokenized prompts from the TeacherRunner's train_dataset so that they can be put into a DataProto object (see line 558 of teacher_runner.py).

In offline training, the OfflineAggregator class, which offers the same interface as the Aggregator but simply loads data from .jsonl files, is used instead.

# Sending Curated Batches to the Students
The method for sending batches to the student processes will mirror the Aggregator class; I envision creating a "Dispatcher" class which maintains batches of available training data for each student and sends them to the student processes upon request (using a "get" function instead of the "add" function from before). The student processes will have to take these batches and convert the prompt IDs to tokenized prompts, then use those batches instead of the batches they are pulling from their dataloaders currently.

# Inference-Time Routing
Inference will be done on a single node with 4 GPUs. My plan for inference right now is to make all routing decisions before initializing the student models on any GPUs. This means that the router model can use all 4 GPUs which should allow it to finish quickly. After that, we will know how what the workload for each student model is, and we can allocate GPUs to them accordingly. This plan is going to require significant modifications to the standard evaluation script (currently math_eval.py under math-evaluation-harness) since it was designed for a single model. Note that I have switched this submodule to a personal fork so that it can be modified.

I have already started modifying math_eval.py to do routing using a TeacherScoreWorker (this is in a copy called math_eval_with_routing.py). This needs to be continued until the TeacherScoreWorker (or a whole worker group) can occupy all 4 GPUs at once. Then we can add in the student workers and test how much time can actually be saved by using the router as opposed to giving everything to the big model, as well as what is the cost in accuracy.

## TODOs
- Set up a router inference pipeline that fully utilizes all available GPUs in math_eval_with_routing.py
    - Note that in addition to doing inference to get the predicted success rates, there needs to be some logic to convert those to routing decisions (i.e. 0 for small model, 1 for big model)
- Decide how to allocate GPUs to models based on the routing results and implement that (also in math_eval_with_routing.py)