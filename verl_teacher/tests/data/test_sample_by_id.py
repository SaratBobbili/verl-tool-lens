import unittest
import os, torch
from hydra import compose, initialize_config_dir  
from hydra.core.global_hydra import GlobalHydra  

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local

from verl_teacher.main_teacher import create_rl_dataset

class TestTeacherScoreWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        

    @classmethod
    def tearDownClass(cls):
        

    def setUp(self):
        """Set up test fixtures - note that most of this code is copied from main_teacher.py"""
        ### This code is used to load the same config which main_ppo.py would load
        GlobalHydra.instance().clear()  
        try:  
            # This requires an absolute path
            with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "..", "config")):  
                config = compose(config_name="teacher_runner", overrides=[
                    "teacher.ppo_micro_batch_size_per_gpu=256",
                    "teacher.model.path=Qwen/Qwen2.5-1.5B-Instruct"
                ])  
        finally:  
            GlobalHydra.instance().clear()
                
        local_path = copy_to_local(
            config.teacher.model.path, use_shm=config.teacher.model.get("use_shm", False)
        )
        # Instantiate the tokenizer and processor.
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()

        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        config.save_pretrained(self.temp_dir)

        self.config = FSDPTeacherConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            forward_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            cliprange_value=0.5,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            ulysses_sequence_parallel_size=1,
            rollout_n=1,
            optim=FSDPOptimizerConfig(lr=1e-6),
            model=FSDPTeacherModelCfg(
                path="Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct",
                fsdp_config=FSDPEngineConfig(fsdp_size=-1),
                use_remove_padding=False,
            ),
        )
        assert self.world_size <= 4 // 2

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dataset(self, batch_size=2, seq_len=10, response_len=5):
        """Create test data for compute_scores method"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        responses = torch.randint(0, 1000, (batch_size, response_len), dtype=torch.long)
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            batch_size=[batch_size],
        )

        data = DataProto(
            batch=batch, meta_info={"micro_batch_size": 2, "max_token_len": seq_len, "use_dynamic_bsz": False}
        )

        return data

    def test_init_model(self):
        """Test TeacherScoreWorker.init_model() method"""
        worker = TeacherScoreWorker(self.config)
        worker.init_model()