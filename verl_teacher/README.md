# Repository Structure
- experiments/ contains subfolders with standalone experiments for testing the performance of different configurations.
- tests/ contains pytest harnesses for testing the correctness of the verl_teacher code. You will need to have pytest installed in order to run any of these.
- verl_teacher/ contains the source code and config defaults

# Main Entrypoint
The main entrypoint for running the teacher (whether you are training it offline or in parallel with a student model) is verl_teacher/main_teacher.py. This file instantiates the TeacherRunner class (analogous to the RayPPOTrainer) from teacher_runner.py to do training and inference (currently only training is implemented). The launch_teacher.sh script under experiments/router_accuracy/small_model_test/ gives an example of how to launch main_teacher.py for training.

# Preparing Training Data for the Teacher
In online training, the teacher must wait for new data to come from the student processes before the teacher can be updated. The Aggregator class defined in verl_teacher/utils/comms.py handles this by asynchronously recieving data from the students and allowing the main teacher process to poll it to check whether enough data is available. The Aggregator should receive batches with prompt IDs (rather than the prompts themselves) along with empirical success rates and store them in its buffer until they are passed on to the main process. The main process will then convert the prompt IDs into tokenized prompts from the TeacherRunner's train_dataset so that they can be put into a DataProto object (see line 558 of teacher_runner.py).

In order for training data to be sent to the Aggregator in the first place, the ray_trainer.py file in the base verl code needs to be modified to send it. I have started the code for this at line 1186, which so far just computes the empirical success rates (ESRs). It will also need to associate the ESRs with the prompt IDs (currently the index it is using is the uid which will not be consistent across processes) and then execute the add function of the Aggregator (you can leave this part to me since I know how to access the Aggregator as a Ray actor from a different process).

In offline training, the OfflineAggregator class, which offers the same interface as the Aggregator but simply loads data from .jsonl files, is used instead. Nothing needs to be modified for this.

# Sending Curated Batches to the Students
The method for sending batches to the student processes will mirror the Aggregator class; I envision creating a "Dispatcher" class which maintains batches of available training data for each student and sends them to the student processes upon request (using a "get" function instead of the "add" function from before). The student processes will have to take these batches and convert the prompt IDs to tokenized prompts, then use those batches instead of the batches they are pulling from their dataloaders currently.