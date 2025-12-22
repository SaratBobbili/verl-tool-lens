# IGPO Usage
This code overwrites VeRL's RayPPOTrainer class with a RayPPOTrainerIGPO class which additionally computes the information gain for each turn and uses it in a custom advantage computation. The modified advantage computation is linked directly to the RayPPOTrainerIGPO class, so it uses the original GRPO AdvantageEstimator class; therefore you should put ```algorithm.adv_estimator=grpo``` in your config. Eventually I hope to control whether IGPO is used through a config option, but right now this is the easiest way. All code relevant to the IGPO implementation is currently located in verl_tool/trainer/ppo/ray_trainer_igpo.py.

If you are testing on math problems, add ```algorithm.task_type="math"``` to your config. This will ensure that the ground-truth answer is wrapped in \boxed{} to match the expected format. The default task_type is designed for searchR1 and uses ```<answer><\answer>``` tags to wrap the ground-truth answer.

The default value for the discount factor $\gamma$ is 1.0. To modify it, use the ```algorithm.gamma``` parameter in your config.

Note that this code has only been tested on the searchR1 environment with retrieval tool, and it has not been tested on any models outside of the Qwen-2.5 family (in fact, it is unlikely to work on any other models because it assumes a specific tokenizer). Also, is SearchR1 some questions have multiple valid answers, but my code will only compute the information gain for the first answer in the list.

# Totally Professional Disclaimer
This algorithm is provided as is and as available, without any representations or warranties of any kind, express or implied. I make no guarantees regarding the condition, performance, accuracy, completeness, or reliability of the algorithm or any results obtained from its use.

I am not responsible for any errors, omissions, damages, losses, or misleading results that may arise from the use, misuse, or inability to use this algorithm. Any reliance you place on the algorithm or its outputs is strictly at your own risk. It is the user’s responsibility to verify all results and determine the suitability of the algorithm for their intended purpose (but honestly, who has time for that these days?).