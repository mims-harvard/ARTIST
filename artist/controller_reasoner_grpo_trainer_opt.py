#!/usr/bin/env python3
"""
Custom GRPO Trainer for Controller-Reasoner Framework

Handles two-phase training: iterative loop + G rollouts for consistency

Features:
- Controller-Reasoner iterative generation with segment selection
- Group-based advantage calculation for reasoner rollouts
- Class-balanced sampling for imbalanced datasets (see BALANCED_SAMPLING_README.md)
  * Weighted random sampling with configurable class weights
  * Oversampling minority classes to match majority
  * Compatible with distributed training and GRPO's generation reuse
"""
import random
import os
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from trl import GRPOTrainer
from trl.trainer.utils import pad
from accelerate.utils import gather
import warnings
import copy
from segment_memory import SegmentMemory, SegmentMemoryManager
from controller_reasoner_generation_opt import ControllerReasonerGenerator
from controller_reasoner_rewards import ControllerReward, ReasonerReward
from trl.trainer.grpo_config import GRPOConfig
from dataclasses import dataclass
from qwents_utils import get_instruction_for_stage
import re
from trl.extras.profiling import profiling_decorator
from trl.trainer.grpo_trainer import nanmin, nanmax
from tools_qwents import TS_TOOLS
from trl.trainer.utils import pad
import time
from tqdm import tqdm
import contextlib
import torch
import psutil
import gc
import json
from torch.nn.utils.rnn import pad_sequence

def clean_input_data(inputs):
    """Simple cleanup for post-loss freeing of GPU tensors."""
    if not isinstance(inputs, dict):
        return
    for k, v in list(inputs.items()):
        if isinstance(v, dict):
            for kk, vv in list(v.items()):
                if torch.is_tensor(vv) or isinstance(vv, (list, tuple)):
                    v[kk] = None
            inputs[k] = None
        elif torch.is_tensor(v):
            inputs[k] = None
    gc.collect()

def print_gpu_memory(tag=""):
    """Prints detailed CUDA + system memory stats."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"\n[GPU Memory] {tag}")
        print(f"  Allocated: {alloc:.2f} MB | Reserved: {reserved:.2f} MB")
        print(f"  Max Allocated: {max_alloc:.2f} MB | Max Reserved: {max_reserved:.2f} MB")
    else:
        print("\n[GPU Memory] No CUDA device detected.")

    # Optional: also show CPU (host) memory
    vm = psutil.virtual_memory()
    print(f"  CPU RAM Used: {(vm.total - vm.available) / 1024**2:.2f} MB / {vm.total / 1024**2:.2f} MB total")
    print("-" * 60)


@dataclass
class ControllerReasonerGRPOConfig(GRPOConfig):
    """
    Custom GRPO config for Controller-Reasoner framework
    
    Bypasses two validations that don't apply to our custom generation:
    1. num_generations >= 2 (we use num_generations=1, no prompt repetition)
    2. generation_batch_size % num_generations == 0 (we compute advantages differently)
    """
    temperature: float = 0.7
    reasoner_temperature: float = 0.7
    controller_temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    min_p = None
    repetition_penalty: float = 1.0
    extended_prompt: bool = False
    
    # Class balancing options
    use_balanced_sampling: bool = False
    sampling_strategy: str = "weighted"  # Options: "weighted", "oversample", "stratified"
    balance_alpha: float = 1.0  # Power for computing class weights (1.0 = inverse frequency)
    
    # Advantage computation options
    use_rloo_advantages: bool = False  # If True, use RLOO (Leave-One-Out) advantages; otherwise use regular GRPO advantages
    def __post_init__(self):
        """
        Override __post_init__ to bypass incompatible validations
        
        We handle generation and advantages ourselves, so we skip:
        - "GRPO requires at least 2 generations per prompt" check (line 619-623)
        - "generation_batch_size must be divisible by num_generations" check (line 613-619)
        """
        
        # === STEP 1: Call TrainingArguments.__post_init__ (grandparent) ===
        # This sets up basic training args like bf16, distributed settings, etc.
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        super(GRPOConfig, self).__post_init__()  # Skip GRPOConfig, call TrainingArguments
        
        # === STEP 2: Set up generation batch size (copied from GRPOConfig) ===
        num_processes = self.world_size
        
        # Calculate generation_batch_size and steps_per_generation
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes #* self.steps_per_generation
            )
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Validate divisibility by global batch size (keep this check)
            global_batch_size = self.per_device_train_batch_size * num_processes
            if self.generation_batch_size % global_batch_size != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by "
                    f"global batch size ({global_batch_size})."
                )
            self.steps_per_generation = self.generation_batch_size // global_batch_size
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes #* self.steps_per_generation
            )
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' cannot both be configured at the same time"
            )
        
        # === STEP 3: SKIP the two problematic validations ===
        # We don't validate:
        # 1. generation_batch_size % num_generations == 0  ← SKIPPED
        # 2. num_generations >= 2                          ← SKIPPED
        #
        # Reason: We use custom generation logic where:
        # - num_generations=1 (no prompt repetition)
        # - We generate 1 controller + G reasoners per question
        # - Advantages computed separately (batch-level for controller, group-level for reasoner)
        
        print(f"\n{'='*60}")
        print(f"Controller-Reasoner GRPO Configuration:")
        print(f"  per_device_train_batch_size: {self.per_device_train_batch_size}")
        print(f"  num_generations: {self.num_generations} (custom - bypassing validation)")
        print(f"  generation_batch_size: {self.generation_batch_size}")
        print(f"  steps_per_generation: {self.steps_per_generation}")
        print(f"  world_size: {self.world_size}")
        print(f"  use_rloo_advantages: {self.use_rloo_advantages}")
        print(f"  ℹStandard GRPO validations bypassed (using custom generation)")
        print(f"{'='*60}\n")


class BalancedRandomSampler(torch.utils.data.Sampler):
    """
    Weighted random sampler for class imbalance.
    Samples with replacement according to class weights.
    """
    def __init__(self, dataset, labels, alpha=1.0, num_samples=None, seed=None):
        """
        Args:
            dataset: The dataset to sample from
            labels: List/array of labels for each sample
            alpha: Power for computing weights. 1.0 = inverse frequency, <1.0 = less aggressive
            num_samples: Number of samples per epoch (default: len(dataset))
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.labels = labels
        self.alpha = alpha
        self.num_samples = num_samples or len(dataset)
        self.seed = seed
        
        # Compute class weights
        from collections import Counter
        import numpy as np
        
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)
        
        # Compute inverse frequency weights
        class_weights = {}
        for label, count in label_counts.items():
            class_weights[label] = (total_samples / (num_classes * count)) ** alpha
        
        # Assign weight to each sample
        self.weights = torch.DoubleTensor([class_weights[label] for label in labels])
        
        print(f"\nBalanced Sampling Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Class distribution:")
        for label, count in sorted(label_counts.items()):
            weight = class_weights[label]
            print(f"    Class {label}: {count} samples ({100*count/total_samples:.1f}%), weight: {weight:.3f}")
        print()
        
    def __iter__(self):
        # Set seed for reproducibility across processes
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            generator = None
            
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=generator)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


class OversamplingBalancedSampler(torch.utils.data.Sampler):
    """
    Oversampling sampler that duplicates minority class samples to match majority class.
    Samples without replacement in each epoch.
    """
    def __init__(self, dataset, labels, seed=None):
        """
        Args:
            dataset: The dataset to sample from
            labels: List/array of labels for each sample
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.labels = labels
        self.seed = seed
        
        from collections import defaultdict
        import numpy as np
        
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Find max class size
        max_class_size = max(len(indices) for indices in class_indices.values())
        
        # Oversample each class to match max class size
        self.balanced_indices = []
        for label, indices in sorted(class_indices.items()):
            # Repeat indices to match max class size
            num_repeats = max_class_size // len(indices)
            remainder = max_class_size % len(indices)
            
            self.balanced_indices.extend(indices * num_repeats)
            if remainder > 0:
                if seed is not None:
                    rng = np.random.RandomState(seed)
                    self.balanced_indices.extend(rng.choice(indices, remainder, replace=False).tolist())
                else:
                    self.balanced_indices.extend(np.random.choice(indices, remainder, replace=False).tolist())
        
        print(f"\nOversampling Statistics:")
        print(f"  Original dataset size: {len(labels)}")
        print(f"  Balanced dataset size: {len(self.balanced_indices)}")
        print(f"  Max class size: {max_class_size}")
        print(f"  Class distribution after oversampling:")
        for label, indices in sorted(class_indices.items()):
            print(f"    Class {label}: {len(indices)} -> {max_class_size} samples")
        print()
        
    def __iter__(self):
        # Shuffle the balanced indices
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            indices = torch.randperm(len(self.balanced_indices), generator=generator)
        else:
            indices = torch.randperm(len(self.balanced_indices))
        
        return iter([self.balanced_indices[i] for i in indices])
    
    def __len__(self):
        return len(self.balanced_indices)


class BalancedRepeatSampler(torch.utils.data.Sampler):
    """
    Custom RepeatSampler that respects the underlying balanced sampler's indices.
    
    Unlike TRL's RepeatSampler which ignores the data_source's __iter__ and generates
    its own indices, this sampler actually uses the indices from the balanced sampler.
    
    Args:
        base_sampler: The balanced sampler to wrap (BalancedRandomSampler or OversamplingBalancedSampler)
        mini_repeat_count: Number of times to repeat each index per batch
        batch_size: Number of unique indices per batch
        repeat_count: Number of times to repeat the full sampling process
    """
    def __init__(
        self,
        base_sampler: torch.utils.data.Sampler,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
    ):
        self.base_sampler = base_sampler
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(base_sampler)
    
    def __iter__(self):
        # CRITICAL: Get indices from the balanced sampler (TRL's RepeatSampler doesn't do this!)
        # This call triggers base_sampler.__iter__(), which does the shuffling:
        #   - BalancedRandomSampler: Uses torch.multinomial for random weighted sampling
        #   - OversamplingBalancedSampler: Uses torch.randperm to shuffle balanced indices
        # So we DO get randomized batches - the shuffling happens in the base sampler!
        base_indices = list(self.base_sampler)
        
        # Group shuffled indices into batches
        batches = [base_indices[i:i + self.batch_size] 
                   for i in range(0, len(base_indices), self.batch_size)]
        
        # Keep only complete batches
        batches = [batch for batch in batches if len(batch) == self.batch_size]
        
        # Apply repeat logic (same as TRL's RepeatSampler)
        for batch in batches:
            for _ in range(self.repeat_count):
                for index in batch:
                    for _ in range(self.mini_repeat_count):
                        yield index
    
    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count


class ControllerReasonerGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO trainer for Controller-Reasoner framework
    
    Training Flow:
    1. For each question, run Controller-Reasoner loop until accept or max rounds
    2. Run G reasoner rollouts with final segment list
    3. Calculate role-specific rewards
    4. Unified policy update
    """
    
    def __init__(
        self,
        task,
        generation_handler,
        controller_reward_fn,
        reasoner_reward_fn,
        max_rounds: int = 20,
        num_rollouts: int = 8,  # G in the paper
        first_seg_trials: int = 5,
        include_full_ts_initially: bool = False,
        use_conversation_history: bool = False,
        controller_temperature: float = 0.9,
        reasoner_temperature: float = 0.7,
        num_loops_per_generation: int = 1, # for the controller
        model=None,
        extended_prompt: bool = False,
        reasoner_max_new_tokens: int = 512,
        use_uncertainty_prompt: bool = False,
        lora_weights_path: str = None,  
        lora_cfg: dict = None,          
        **kwargs
    ):
        self.task = task
        if model is None:
            raise ValueError("Pass `model=` when constructing ControllerReasonerGRPOTrainer")
        
        # Ensure model class is registered
        import transformers as _tf
        setattr(_tf, model.__class__.__name__, model.__class__)
        
        if not getattr(model.config, "architectures", None):
            model.config.architectures = [model.__class__.__name__]
        else:
            model.config.architectures[0] = model.__class__.__name__
        
        kwargs["model"] = model
        
        # Initialize base GRPO trainer with combined reward functions
        combined_reward_fns = [controller_reward_fn, reasoner_reward_fn]
        kwargs["reward_funcs"] = combined_reward_fns
        
        # Save beta, temporarily set to 0 to skip ref_model loading in parent __init__
        # (GRPOTrainer doesn't pass trust_remote_code to AutoConfig.from_pretrained)
        args = kwargs.get("args")
        original_beta = args.beta if args else 0.0
        if args and args.beta != 0.0:
            args.beta = 0.0

        super().__init__(**kwargs)
        self._lora_weights_path = lora_weights_path
        self._lora_cfg = lora_cfg or {}
        # Restore beta and load ref_model ourselves with trust_remote_code=True
        if original_beta != 0.0:
            self.beta = original_beta
            if args:
                args.beta = original_beta
            self._setup_reference_model()
        self._logs = {
            "prompt": [],
            "completion": [],
            "rewards": {},
            "advantages": [],
            "images": []
        }
        self._controller_logs = []
        self._reasoner_logs = []
        
        self.generation_handler = generation_handler
        self.num_loops_per_generation = num_loops_per_generation
        print(f"num_loops_per_generation: {self.num_loops_per_generation}")
        print("total rollouts of the reasoner: ", num_rollouts * num_loops_per_generation)
        self.controller_reward_fn = controller_reward_fn
        self.reasoner_reward_fn = reasoner_reward_fn
        self.max_rounds = max_rounds
        self.num_rollouts = num_rollouts
        self.include_full_ts_initially = include_full_ts_initially
        self.controller_temperature = controller_temperature
        self.reasoner_temperature = reasoner_temperature
        self.extended_prompt = extended_prompt
        self.reasoner_max_new_tokens = reasoner_max_new_tokens
        self.use_uncertainty_prompt = use_uncertainty_prompt
        # Store advantage computation method from args
        self.use_rloo_advantages = args.use_rloo_advantages if args else False
        # Initialize generator
        self.cr_generator = ControllerReasonerGenerator(
            model=self.model,
            tokenizer=self.processing_class,
            generation_handler=generation_handler,
            max_rounds=max_rounds,
            first_seg_trials=first_seg_trials,
            task_name=self.task,
            include_full_ts_initially=include_full_ts_initially,
            use_conversation_history=use_conversation_history,
            controller_temperature=controller_temperature,
            reasoner_temperature=reasoner_temperature,
            extended_prompt=extended_prompt,
            reasoner_max_new_tokens=reasoner_max_new_tokens,
            use_uncertainty_prompt=use_uncertainty_prompt
        )

    def _setup_reference_model(self):
        """Load reference model with trust_remote_code=True for KL divergence"""
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        
        def remap_peft_keys(state_dict, model):
            """Remap PEFT keys to match current model structure"""
            new_state_dict = {}
            model_keys = set(model.state_dict().keys())
            
            for key, value in state_dict.items():
                new_key = key
                
                if key in model_keys:
                    new_state_dict[key] = value
                    continue
                
                # Try adding .base_layer
                if '.base_layer.' not in key:
                    candidate = key.replace('.weight', '.base_layer.weight')
                    if candidate in model_keys:
                        new_key = candidate
                # Try removing .base_layer
                elif '.base_layer.' in key:
                    candidate = key.replace('.base_layer.weight', '.weight')
                    if candidate in model_keys:
                        new_key = candidate
                
                new_state_dict[new_key] = value
            
            return new_state_dict
        
        # Get model path from the model's config
        model_path = self.model.config._name_or_path
        
        print(f"Loading reference model from {model_path}")
        
        # Load base model
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_safetensors=False
        )
        
        # Check if we have SFT weights to load (passed from wrapper)
        lora_weights_path = getattr(self, '_lora_weights_path', None)
        
        if lora_weights_path:
            print(f"Loading SFT weights into reference model from {lora_weights_path}")
            
            # Get LoRA config from the main model (if it's a PeftModel)
            # Or use default config
            lora_cfg = getattr(self, '_lora_cfg', {})
            lora_r = lora_cfg.get('r', 8)
            lora_alpha = lora_cfg.get('alpha', 16)
            lora_dropout = lora_cfg.get('dropout', 0.02)
            lora_target_modules = lora_cfg.get('target_modules', 
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            
            # Apply LoRA to ref model
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.ref_model = get_peft_model(self.ref_model, lora_config)
            
            # Load checkpoint weights
            checkpoint = torch.load(lora_weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                            for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Remap keys for PEFT version compatibility
            state_dict = remap_peft_keys(state_dict, self.ref_model)
            
            missing, unexpected = self.ref_model.load_state_dict(state_dict, strict=False)
            print(f"Ref model loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # Merge and unload (bake LoRA into base weights)
            self.ref_model = self.ref_model.merge_and_unload()
            print("Reference model LoRA merged")
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Prepare with accelerator
        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        print("Reference model ready")

    @contextlib.contextmanager
    def generation_mode(self):
        """Context manager to set up model for generation"""
        
        # Add this ONE line:
        base_model = self._unwrap_model(self.model)
        
        # Save original states - use base_model where needed
        was_training = self.model.training
        original_use_cache = base_model.config.use_cache  # Changed
        
        # Check gradient checkpointing - use base_model
        grad_ckpt_enabled = False
        if hasattr(base_model, 'is_gradient_checkpointing'):  # Changed
            grad_ckpt_enabled = base_model.is_gradient_checkpointing
        if hasattr(base_model, '_gradient_checkpointing_func'):  # Changed
            grad_ckpt_enabled = grad_ckpt_enabled or (base_model._gradient_checkpointing_func is not None)
        if hasattr(base_model.config, 'use_gradient_checkpointing'):  # Changed
            grad_ckpt_enabled = grad_ckpt_enabled or base_model.config.use_gradient_checkpointing
        
        try:
            self.model.eval()
            base_model.config.use_cache = True  # Changed
            
            if grad_ckpt_enabled:
                print("Temporarily disabling gradient checkpointing for generation...")
                base_model.gradient_checkpointing_disable()  # Changed
            
            with torch.no_grad():
                yield
        
        finally:
            if was_training:
                self.model.train()
            base_model.config.use_cache = original_use_cache  # Changed
            
            if grad_ckpt_enabled:
                print("Re-enabling gradient checkpointing...")
                base_model.gradient_checkpointing_enable()  # Changed

    def _get_train_sampler(self, dataset=None):
        """
        Override to add class-balanced sampling support.
        
        This wraps the balanced base sampler with GRPO's RepeatSampler logic
        to maintain compatibility with the GRPO training loop.
        """
        if dataset is None:
            dataset = self.train_dataset
            
        # Check if balanced sampling is enabled
        if not getattr(self.args, 'use_balanced_sampling', False):
            # Use default GRPO sampler
            return super()._get_train_sampler(dataset)
        
        print(f"\n{'='*60}")
        print(f"Setting up class-balanced sampling:")
        print(f"  Strategy: {self.args.sampling_strategy}")
        print(f"  Balance alpha: {self.args.balance_alpha}")
        
        # Extract labels from dataset
        labels = []
        for i in tqdm(range(len(dataset)), desc="Extracting labels"):
            example = dataset[i]
            # Extract the label from metadata
            if "metadata" in example and "correct_answer" in example["metadata"]:
                label = example["metadata"]["correct_answer"]
                # For MCQ, normalize label to integer
                if isinstance(label, str):
                    # Assuming labels are like "A", "B", "C", "D" or "0", "1", "2", "3"
                    if label.isdigit():
                        label = int(label)
                    if label in ("yes", "no"):
                        label = 1 if label == "yes" else 0
                    else:
                        # Convert A->0, B->1, C->2, D->3
                        label = ord(label.upper()) - ord('A')
                labels.append(label)
            else:
                raise ValueError(f"Could not find label in dataset example {i}")
        
        # Create balanced sampler based on strategy
        strategy = self.args.sampling_strategy
        seed = self.args.seed
        
        if strategy == "weighted":
            # Weighted random sampling
            base_sampler = BalancedRandomSampler(
                dataset=dataset,
                labels=labels,
                alpha=self.args.balance_alpha,
                num_samples=len(dataset),
                seed=seed
            )
        elif strategy == "oversample":
            # Oversample minority classes
            base_sampler = OversamplingBalancedSampler(
                dataset=dataset,
                labels=labels,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}. Use 'weighted' or 'oversample'")
        
        # Use our custom BalancedRepeatSampler that respects the balanced sampler's indices
        # (TRL's RepeatSampler doesn't call the base sampler's __iter__, so it ignores our balancing)
        repeat_sampler = BalancedRepeatSampler(
            base_sampler=base_sampler,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
        )
        
        print(f"  Balanced sampler created successfully!")
        print(f"{'='*60}\n")
        
        return repeat_sampler

    def _fill_reasoner_log_rows(
        self,
        reasoner_log_rows: list[dict],
        reasoner_rewards,
        reasoner_advantages,
        reasoner_components: list[dict],
    ):
        if len(reasoner_log_rows) == 0:
            return
        rewards_list = reasoner_rewards
        for idx, (tot, comps, adv) in enumerate(zip(rewards_list, reasoner_components, reasoner_advantages)):
            row = reasoner_log_rows[idx]
            row["reward_total"] = float(tot)
            row["reward_accuracy"] = float(comps.get("accuracy_score_weighted", 0.0))
            row["reward_format"] = float(comps.get("format_score_weighted", 0.0))
            row["reward_uncertainty"] = float(comps.get("uncertainty_bonus_raw", 0.0))
            row["advantage"] = float(adv)

    def _build_controller_log_rows(
        self,
        controller_rewards,
        controller_advantages,
        controller_components: list[dict],
        controller_metadata: list[dict],
        all_controller_full_conv: list[list],
        all_controller_gt_segs: list[list],
    ):
        rows = []
        rewards_list = controller_rewards
        advantages_list = controller_advantages
        for qi, (tot, comps, meta, conv, adv, gt_segs) in enumerate(
            zip(rewards_list, controller_components, controller_metadata, all_controller_full_conv, advantages_list, all_controller_gt_segs)
        ):
            # extract last prompt + all responses
            last_prompt = conv[-2][1]['content'] if len(conv) > 1 else ""
            all_responses = []
            for i,msg_l in enumerate(conv):
                if i % 2 == 1:
                    all_responses.append(msg_l['content'])

            rows.append({
                "step": int(self.state.global_step),
                "role": "controller",
                "question_index": meta['question_idx'],
                "rollout_index": meta['rollout_idx'],
                "reward_total": float(tot),
                "r_consistency": float(comps.get("consistency_score_weighted", 0.0)),
                "r_efficiency": float(comps.get("efficiency_penalty_weighted", 0.0)),
                "r_segment_explore": float(comps.get("segment_explore_score_raw", 0.0)),
                "r_format": float(comps.get("format_score_weighted", 0.0)),
                "r_trajectory": float(comps.get("trajectory_score_weighted", 0.0)),
                "consistency_score_raw": float(comps.get("consistency_score_raw", 0.0)),
                "num_trials": int(meta.get("num_trials", 0)),
                "format_error": bool(meta.get("format_error", False)),
                "has_accept": bool(meta.get("has_accept", False)),
                # new fields:
                "controller_last_prompt": last_prompt,
                "controller_responses": json.dumps(all_responses, ensure_ascii=False),
                "advantage": float(adv),
                "gt_segs": json.dumps(meta.get("gt_segs", [[-1,-1]]), ensure_ascii=False),
                "pred_segs": json.dumps(meta['all_segments'], ensure_ascii=False),
                "GT_answer": meta.get("gold_answer", ""),
            })
        return rows

    def _run_controller_loops(self, lightning_examples, prompts_text, inputs):
        """
        Runs the controller-reasoner loop per question for `num_loops_per_generation`.
        Returns:
        loop_results: list of dicts with per-loop results
        question_accuracies: list[int] 0/1 per loop (final reasoner vs gold)
        """
        loop_results = []
        question_accuracies = []

        print_gpu_memory("_generate_and_score_completions: before controller reasoner loop")
        self.controller_reward_fn.set_step(self.state.global_step)
        with self.generation_mode(), torch.no_grad():
            for q_idx, (le, prompt_text) in enumerate(
                tqdm(zip(lightning_examples, prompts_text), total=len(lightning_examples), desc="Controller Loop")
            ):
                ts_data = le["ts"]
                ts_tensor = self.cr_generator.generation_handler._prepare_ts_tensor(ts_data)
                raw_ts = le.get("raw_ts", ts_data)
                context = le["context"]
                gold_answer = le["label"]
                timestamps = inputs[q_idx]["metadata"].get("timestamps")

                formatted_context = get_instruction_for_stage('mcq', self.task) + "\n" + context
                question_ids = self.processing_class.encode(formatted_context, add_special_tokens=False)

                per_question_loops = []
                for _ in range(self.num_loops_per_generation):
                    result = self.cr_generator.generate_controller_reasoner_loop(
                        question=formatted_context,
                        ts_data=ts_tensor,
                        gold_answer=gold_answer,
                        question_ids=question_ids,
                        timestamps=timestamps,
                        raw_ts=raw_ts,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    per_question_loops.append({
                        "result": result,
                        "gold_answer": gold_answer,
                        "q_idx": q_idx,
                    })
                    if len(result["reasoner_answers"]) > 0:
                        pred = self._normalize_answer(result["reasoner_answers"][-1]['answer'])
                        question_accuracies.append(1 if pred == self._normalize_answer(gold_answer) else 0)
                    else:
                        question_accuracies.append(0)

                loop_results.extend(per_question_loops)
                del ts_tensor
        
        print_gpu_memory("_generate_and_score_completions: after controller reasoner loop")
        return loop_results, question_accuracies
    
    def _run_controller_loops_batched(self, lightning_examples, prompts_text, inputs):
        """
        Runs the controller-reasoner loop per question for `num_loops_per_generation`.
        Returns:
        loop_results: list of dicts with per-loop results
        question_accuracies: list[int] 0/1 per loop (final reasoner vs gold)
        """
        loop_results = []
        question_accuracies = []

        print_gpu_memory("_generate_and_score_completions: before controller reasoner loop")
        self.controller_reward_fn.set_step(self.state.global_step)
        with self.generation_mode(), torch.no_grad():
            for q_idx, (le, prompt_text) in enumerate(
                tqdm(zip(lightning_examples, prompts_text), total=len(lightning_examples), desc="Controller Loop")
            ):
                ts_data = le["ts"]
                gold_answer = le["label"]
                
                if isinstance(ts_data, np.ndarray):
                    ts_tensor = torch.from_numpy(ts_data).float()
                elif isinstance(ts_data, list):
                    ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
                else:
                    ts_tensor = ts_data
                
                # Ensure proper shape: [seq_len, n_vars] or [seq_len]
                if ts_tensor.dim() == 1:
                    ts_tensor = ts_tensor.unsqueeze(-1)  # [seq_len, 1]

                raw_ts = le.get("raw_ts", ts_data)
                if self.task in ["ECG_QA_S_VERIFY", "ECG_QA_S_QUERY", "ECG_QA_MIXED"]:
                    # question = le["question"]
                    # clinical_context = le["clinical_context"]
                    # options = le["options"]
                    # ts_text = le["ts_text"]
                    # gold_answer = 'no' if gold_answer == 'A' else 'yes'
                    context = le["context"]
                    # context = f"Clinical Context: {clinical_context}\nQuestion: {question}\nOptions: {options}\nTime Series information: {ts_text}"
                elif self.task == "SLEEPQA":
                    question = le["question"]
                    options = le["options"]
                    options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
                    ts_text = le["ts_text"]
                    context = f"Question: {question}\nOptions: {options_text}\nTime Series information: {ts_text}"
                else:
                    context = le["context"]
                
                
                timestamps = inputs[q_idx]["metadata"].get("timestamps")

                formatted_context = get_instruction_for_stage('mcq', self.task) + "\n" + context

                results = self.cr_generator.generate_controller_reasoner_loop_batched(
                question=formatted_context,
                ts_data=ts_tensor,
                gold_answer=gold_answer,
                timestamps=timestamps,
                raw_ts=raw_ts,
                temperature=self.controller_temperature,
                top_p=self.top_p,
                num_loops=self.num_loops_per_generation,  # This controls batch size
                )
                for result in results:
                    loop_results.append({
                        "result": result,
                        "gold_answer": gold_answer,
                        "q_idx": q_idx,               
                    })
                    
                    # Accuracy check - SAME CODE, just inside the loop
                    if len(result["reasoner_answers"]) > 0:
                        pred = self._normalize_answer(result["reasoner_answers"][-1]['answer'])
                        question_accuracies.append(1 if pred == self._normalize_answer(gold_answer) else 0)
                    else:
                        question_accuracies.append(0)
                
                del ts_tensor
        print_gpu_memory("_generate_and_score_completions: after controller reasoner loop")
        return loop_results, question_accuracies

    def _build_reasoner_rollouts(self, lightning_examples, inputs, loop_results):
        """
        Using the loop_results, run G reasoner rollouts (per controller rollout).
        Returns a big nested structure (per-question lists) + controller-side rollup metrics.
        """
        all_controller_completions = []
        all_controller_full_conv = []
        all_controller_gt_segs = []
        all_controller_ts_data = []
        all_controller_question_ids = []
        controller_metadata = []
        all_controller_prompt_ids = [] 
        all_controller_completion_ids = []

        all_reasoner_prompts_formatted = []
        all_reasoner_ts_data = []
        all_reasoner_question_ids = []
        all_reasoner_segment_encodings = []
        all_reasoner_segments = []
        reasoner_metadata = []
        all_reasoner_completions = []
        all_reasoner_ids = []
        all_reasoner_rollout_counts = []
        all_reasoner_prompt_ids = []
        
        controller_consistencies_all = []
        controller_num_rounds_all = []
        controller_format_errors_all = []

        with self.generation_mode(), torch.no_grad():
            for q_idx, le in enumerate(
                tqdm(lightning_examples, total=len(lightning_examples), desc="Reasoner Rollouts")
            ):
                ts_data = le["ts"]
                gold_answer = le["label"]
                
                if isinstance(ts_data, np.ndarray):
                    ts_tensor = torch.from_numpy(ts_data).float()
                elif isinstance(ts_data, list):
                    ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
                else:
                    ts_tensor = ts_data
                
                # Ensure proper shape: [seq_len, n_vars] or [seq_len]
                if ts_tensor.dim() == 1:
                    ts_tensor = ts_tensor.unsqueeze(-1)  # [seq_len, 1]
                # ts_tensor = ts_data.to(self.model.device) if isinstance(ts_data, torch.Tensor) else ts_data
                raw_ts = le.get("raw_ts", ts_data)
                
                if self.task in ["ECG_QA_S_VERIFY", "ECG_QA_S_QUERY", "ECG_QA_MIXED"]:
                    question = le["question"]
                    clinical_context = le["clinical_context"]
                    options = le["options"]
                    ts_text = le["ts_text"]
                    # gold_answer = 'no' if gold_answer == 'A' else 'yes'
                    context = le["context"]
                    # context = f"Clinical Context: {clinical_context}\nQuestion: {question}\nOptions: {options}\nTime Series information: {ts_text}"
                elif self.task == "SLEEPQA":
                    question = le["question"]
                    options = le["options"]
                    options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
                    ts_text = le["ts_text"]
                    context = f"Question: {question}\nOptions: {options_text}\nTime Series information: {ts_text}"
                else:
                    context = le["context"]
                
                
                timestamps = inputs[q_idx]["metadata"].get("timestamps")
                reasoner_question_counts = [] 

                # per-question containers
                ctrl_completions_q, ctrl_full_conv_q, ctrl_ts_q, ctrl_qids_q, ctrl_meta_q, ctrl_gt_segs_q, ctrl_prompt_ids_q, ctrl_completion_ids_q = [], [], [], [], [], [], [], []
                rsn_prompts_q, rsn_ts_q, rsn_qids_q, rsn_segenc_q, rsn_segs_q, rsn_meta_q, rsn_compl_q, rsn_ids_q, rsn_prompt_ids_q = [], [], [], [], [], [], [], [], []

                # print(f"\n[DEBUG] Question {q_idx}: Processing {self.num_loops_per_generation} controller rollouts")
                for rep in range(self.num_loops_per_generation):
                    result = loop_results[q_idx * self.num_loops_per_generation + rep]["result"]
                    formatted_context = get_instruction_for_stage('mcq', self.task) + "\n" + context

                    # defaults when no reasoner output or no final segments
                    consistency_score = 0.0
                    reasoner_rollout_count = 0
                    if len(result["reasoner_completions"]) > 0 and len(result.get("final_segments", [])) > 0:
                        r0_completion = result["reasoner_completions"][-1]
                        r0_answer = result["reasoner_answers"][-1]['answer']
                        # segment_encodings = [enc.cpu() if enc.is_cuda else enc for enc in result["segment_encodings"]]
                        # segment_encodings_gpu = [enc.to(self.model.device) for enc in segment_encodings]
                        segments = result["final_segments"]

                        rollout_completions, rollout_answers, rollout_prompt_ids = self.cr_generator.generate_reasoner_rollouts_batched(
                            question=formatted_context,
                            ts_data=ts_tensor,
                            segments=segments,
                            num_rollouts=self.num_rollouts - 1,
                            timestamps=timestamps,
                            raw_ts=raw_ts,
                            temperature=self.reasoner_temperature
                        )
                        torch.cuda.empty_cache()

                        rollout_completions.insert(0, r0_completion)
                        rollout_answers.insert(0, r0_answer)
                        consistency_score = self._compute_consistency(rollout_answers, gold_answer)
                        reasoner_rollout_count = len(rollout_completions)


                        for ri, (completion, answer) in enumerate(zip(rollout_completions, rollout_answers)):
                            rsn_compl_q.append(completion)
                            rsn_ids_q.append(torch.tensor(self.processing_class.encode(completion, add_special_tokens=False)))
                            rsn_prompts_q.append(result.get("final_reasoner_prompt"))
                            rsn_ts_q.append(le["ts"])
                            rsn_segs_q.append(segments)
                            rsn_meta_q.append({"gold_answer": gold_answer, "predicted_answer": answer})
                            rsn_prompt_ids_q.append(rollout_prompt_ids)
                            # reasoner_log_rows.append({
                            #     "step": int(self.state.global_step),
                            #     "role": "reasoner",
                            #     "question_index": q_idx,
                            #     "rollout_index": ri * self.num_loops_per_generation + rep,
                            #     "prompt": result.get("final_reasoner_prompt"),
                            #     "completion": completion.strip(),
                            #     "pred_answer": answer,
                            #     "gold_answer": gold_answer,
                            #     "reward_total": None,
                            #     "reward_accuracy": None,
                            #     "reward_format": None,
                            # })

                    # controller-side per rollout
                    reasoner_question_counts.append(reasoner_rollout_count)
                    ctrl_completions_q.append(result["controller_completions"])
                    ctrl_full_conv_q.append(result["full_conv_for_controller"])
                    ctrl_prompt_ids_q.append(result["controller_prompt_ids"])
                    ctrl_completion_ids_q.append(result["controller_completion_ids"])
                    ctrl_gt_segs_q.append(le.get("gt_segs", [[-1, -1]]))
                    ctrl_ts_q.append(le["ts"])
                    ctrl_meta_q.append({
                        "consistency_score": consistency_score,
                        "include_full_ts_initially": self.include_full_ts_initially,
                        "num_trials": result["num_rounds"],
                        "gold_answer": gold_answer,
                        "format_error": result.get('format_error_occurred'),
                        "format_error_round": result.get("format_error_round"),
                        "has_accept": result.get("has_accept", False),
                        "hit_max_rounds": result.get("hit_max_rounds", False),
                        "all_segments": result["final_segments"],
                        "ts_length": len(ts_data),
                        "question_idx": q_idx,
                        "rollout_idx": rep,
                    })
                    controller_consistencies_all.append(float(consistency_score))
                    controller_num_rounds_all.append(int(result["num_rounds"]))
                    controller_format_errors_all.append(1 if result.get("format_error_occurred") else 0)

                # append per-question
                all_controller_completions.append(ctrl_completions_q)
                all_controller_full_conv.append(ctrl_full_conv_q)
                all_controller_gt_segs.append(ctrl_gt_segs_q)
                all_controller_ts_data.append(ctrl_ts_q)
                all_controller_question_ids.append(ctrl_qids_q)
                controller_metadata.append(ctrl_meta_q)
                all_controller_prompt_ids.append(ctrl_prompt_ids_q)
                all_controller_completion_ids.append(ctrl_completion_ids_q)

                all_reasoner_completions.append(rsn_compl_q)
                all_reasoner_ids.append(rsn_ids_q)
                all_reasoner_prompts_formatted.append(rsn_prompts_q)
                all_reasoner_prompt_ids.append(rsn_prompt_ids_q)
                all_reasoner_ts_data.append(rsn_ts_q)
                all_reasoner_question_ids.append(rsn_qids_q)
                all_reasoner_segment_encodings.append(rsn_segenc_q)
                all_reasoner_segments.append(rsn_segs_q)
                reasoner_metadata.append(rsn_meta_q)
                all_reasoner_rollout_counts.append(reasoner_question_counts)

                del ts_tensor
                torch.cuda.empty_cache()

        return (
            all_controller_completions, all_controller_full_conv, all_controller_gt_segs,
            all_controller_ts_data, all_controller_question_ids, controller_metadata,
            all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
            all_reasoner_segment_encodings, all_reasoner_segments, reasoner_metadata,
            all_reasoner_completions, all_reasoner_ids,
            controller_consistencies_all, controller_num_rounds_all, controller_format_errors_all,
            all_reasoner_rollout_counts, 
            all_controller_prompt_ids, all_controller_completion_ids, all_reasoner_prompt_ids,
        )

    def _loo_advantages(self, r: torch.Tensor) -> torch.Tensor:
        """
        Leave-one-out baseline advantages for a single question/group.
        r: shape (K,)
        Returns: shape (K,)
        """
        K = r.numel()
        if K <= 1:
            return torch.zeros_like(r)
        sum_r = r.sum()
        baseline = (sum_r - r) / (K - 1)
        return r - baseline  # centered relative to peers

    @staticmethod
    def per_gpu_global_norm(x: torch.Tensor, eps: float = 1e-8):
        """
        Per-GPU global normalization.
        x: 1D tensor of advantages on THIS GPU for the current step.
        """
        if x.numel() <= 1:
            return torch.zeros_like(x)

        mean = x.mean()
        std = x.std(unbiased=False).clamp_min(eps)
        return (x - mean) / std



    def _compute_controller_rewards_advantages(
        self,
        all_controller_full_conv, all_controller_completions, all_controller_ts_data,
        all_controller_question_ids, controller_metadata, controller_log_rows,
        all_controller_prompt_ids, all_controller_completion_ids,
    ):
        """
        Per-question GRPO normalization for controller. Flattens across rollouts.
        Returns:
        selected_full_conv, selected_ts, selected_qids, selected_advs (all flat)
        controller_rewards (flat tensor), controller_advantages (flat tensor),
        controller_selected_completions (flat list of texts for length metrics)
        """
        device = self.accelerator.device
        controller_selected_full_conv = []
        controller_selected_ts_data = []
        controller_selected_qids = []
        controller_selected_advantages = []
        controller_selected_completions = []
        per_q_rewards = []
        controller_selected_prompt_ids = []      # ← NEW
        controller_selected_completion_ids = []  # ← NEW

        num_questions = len(all_controller_full_conv)
        assert len(all_controller_completions) == num_questions
        assert len(all_controller_ts_data) == num_questions
        assert len(all_controller_question_ids) == num_questions

        for q_idx in range(num_questions): # number fo questions
            q_full_conv = all_controller_full_conv[q_idx]
            q_compls = all_controller_completions[q_idx]
            q_ts = all_controller_ts_data[q_idx]
            q_meta = controller_metadata[q_idx]
            q_prompt_ids = all_controller_prompt_ids[q_idx]        # ← NEW
            q_completion_ids = all_controller_completion_ids[q_idx]  # ← NEW

            q_rewards, q_components = self.controller_reward_fn(
                prompts=[], completions=q_compls, metadata=q_meta
            )
            q_rewards_t = torch.tensor(q_rewards)
            per_q_rewards.append(q_rewards_t)

            if len(q_rewards_t) <= 1:
                q_advs = torch.zeros_like(q_rewards_t)
            else:
                q_advs = (q_rewards_t - q_rewards_t.mean()) / (q_rewards_t.std() + 1e-8)

            for ridx in range(len(q_full_conv)): # number of controller rollouts for this question
                controller_selected_full_conv.append(q_full_conv[ridx])
                controller_selected_ts_data.append(q_ts[ridx])
                # controller_selected_qids.append(q_qids[ridx])
                controller_selected_advs = q_advs[ridx]
                controller_selected_advantages.append(controller_selected_advs)
                controller_selected_completions.extend(q_compls[ridx])
                controller_selected_prompt_ids.append(q_prompt_ids[ridx])
                controller_selected_completion_ids.append(q_completion_ids[ridx])
                # expand W&B controller rows (your builder expects lists)
                controller_log_rows.extend(
                    self._build_controller_log_rows(
                        controller_rewards=[float(q_rewards_t[ridx].item())],
                        controller_advantages=[float(controller_selected_advs.item())],
                        controller_components=[q_components[ridx]],
                        controller_metadata=[q_meta[ridx]],
                        all_controller_full_conv=[q_full_conv[ridx]],
                        all_controller_gt_segs=[[[-1, -1]]],
                    )
                )

        controller_advantages = (
            torch.stack(controller_selected_advantages)
            if controller_selected_advantages else torch.zeros(0)
        )
        controller_rewards = (
            torch.cat([r.to(torch.float32) for r in per_q_rewards])
            if per_q_rewards else torch.zeros(0)
        )

        return (
            controller_selected_full_conv, controller_selected_ts_data, controller_selected_qids,
            controller_advantages, controller_rewards, controller_selected_completions,
            controller_selected_prompt_ids, controller_selected_completion_ids,
        )

    def _compute_controller_rewards_advantages_rloo(
        self,
        all_controller_full_conv, all_controller_completions, all_controller_ts_data,
        all_controller_question_ids, controller_metadata, controller_log_rows,
        all_controller_prompt_ids, all_controller_completion_ids,
    ):
        """
        Per-question GRPO normalization for controller. Flattens across rollouts.
        Returns:
        selected_full_conv, selected_ts, selected_qids, selected_advs (all flat)
        controller_rewards (flat tensor), controller_advantages (flat tensor),
        controller_selected_completions (flat list of texts for length metrics)
        """
        device = self.accelerator.device
        controller_selected_full_conv = []
        controller_selected_ts_data = []
        controller_selected_qids = []
        controller_selected_advantages = []
        controller_selected_completions = []
        per_q_rewards = []
        controller_selected_prompt_ids = []      # ← NEW
        controller_selected_completion_ids = []  # ← NEW

        num_questions = len(all_controller_full_conv)
        assert len(all_controller_completions) == num_questions
        assert len(all_controller_ts_data) == num_questions
        assert len(all_controller_question_ids) == num_questions

        for q_idx in range(num_questions): # number fo questions
            q_full_conv = all_controller_full_conv[q_idx]
            q_compls = all_controller_completions[q_idx]
            q_ts = all_controller_ts_data[q_idx]
            q_meta = controller_metadata[q_idx]
            q_prompt_ids = all_controller_prompt_ids[q_idx]        # ← NEW
            q_completion_ids = all_controller_completion_ids[q_idx]  # ← NEW

            q_rewards, q_components = self.controller_reward_fn(
                prompts=[], completions=q_compls, metadata=q_meta
            )
            q_rewards_t = torch.tensor(q_rewards)
            per_q_rewards.append(q_rewards_t)

            if len(q_rewards_t) <= 1:
                q_advs = torch.zeros_like(q_rewards_t)
            else:
                q_advs = self._loo_advantages(q_rewards_t)  # shape (K,)

            for ridx in range(len(q_full_conv)): # number of controller rollouts for this question
                controller_selected_full_conv.append(q_full_conv[ridx])
                controller_selected_ts_data.append(q_ts[ridx])
                controller_selected_advs = q_advs[ridx]
                controller_selected_advantages.append(controller_selected_advs)
                controller_selected_completions.extend(q_compls[ridx])
                controller_selected_prompt_ids.append(q_prompt_ids[ridx])
                controller_selected_completion_ids.append(q_completion_ids[ridx])
                # expand W&B controller rows (your builder expects lists)
                controller_log_rows.extend(
                    self._build_controller_log_rows(
                        controller_rewards=[float(q_rewards_t[ridx].item())],
                        controller_advantages=[float(controller_selected_advs.item())],
                        controller_components=[q_components[ridx]],
                        controller_metadata=[q_meta[ridx]],
                        all_controller_full_conv=[q_full_conv[ridx]],
                        all_controller_gt_segs=[[[-1, -1]]],
                    )
                )

        controller_advantages = (
            torch.stack(controller_selected_advantages)
            if controller_selected_advantages else torch.zeros(0)
        )
        if controller_advantages.numel() > 1:
            controller_advantages = self.per_gpu_global_norm(controller_advantages)

        controller_rewards = (
            torch.cat([r.to(torch.float32) for r in per_q_rewards])
            if per_q_rewards else torch.zeros(0)
        )

        return (
            controller_selected_full_conv, controller_selected_ts_data, controller_selected_qids,
            controller_advantages, controller_rewards, controller_selected_completions,
            controller_selected_prompt_ids, controller_selected_completion_ids,
        )


    def _sample_and_flatten_reasoners(
        self,
        all_controller_full_conv,
        all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
        all_reasoner_segment_encodings, all_reasoner_segments, all_reasoner_ids, reasoner_metadata,
        all_reasoner_completions, 
        all_reasoner_rollout_counts,
        all_reasoner_prompt_ids,
    ):
        """
        Samples reasoners per controller rollout (size = reasoner_keep, default = G),
        flattens them, computes rewards & advantages, and builds padded tensors.
        Returns a dict with everything needed by loss/metrics/output.
        """
        device = self.accelerator.device
        reasoner_keep = getattr(self, "reasoner_sample_size", None) or self.num_rollouts

        flat_prompts, flat_ts, flat_qids = [], [], []
        flat_segencs, flat_segs, flat_ids, flat_meta, flat_completions, flat_prompt_ids = [], [], [], [], [], []
        kept_per_ctrl = []
        rng = np.random.default_rng()
        flat_reasoner_log_rows = []
        group_lengths = []

        num_questions = len(all_controller_full_conv)
        for q_idx in range(num_questions):
            q_prompts = all_reasoner_prompts_formatted[q_idx]
            q_ts = all_reasoner_ts_data[q_idx]
            q_qids = all_reasoner_question_ids[q_idx]
            q_segencs = all_reasoner_segment_encodings[q_idx]
            q_segs = all_reasoner_segments[q_idx]
            q_ids = all_reasoner_ids[q_idx]
            q_meta = reasoner_metadata[q_idx]
            q_completions = all_reasoner_completions[q_idx]
            q_prompt_ids = all_reasoner_prompt_ids[q_idx]
            num_ctrl_rollouts_q = len(all_controller_full_conv[q_idx])

            if num_ctrl_rollouts_q == 0:
                continue
            
            G = self.num_rollouts
            counts = all_reasoner_rollout_counts[q_idx]
            valid = [i for i, c in enumerate(counts) if c > 0]
            if not valid:
                continue
            
            c = int(rng.choice(valid))
            start = sum(counts[:c])
            glen  = counts[c]
            end   = min(start + glen, len(q_ids))
            
            # print(f"\n[DEBUG] _sample_and_flatten_reasoners - Question {q_idx}:")
            # print(f"  - Total prompts available: {len(q_prompts)}")
            # print(f"  - Num controller rollouts: {num_ctrl_rollouts_q}")
            # print(f"  - Reasoner counts per controller: {counts}")
            # print(f"  - Selected controller rollout: {c}")
            # print(f"  - Selected range: [{start}, {end}) with length {glen}")
            # print(f"  - Reasoner prompts being selected:")
            # for i in range(start, end):
            #     if i < len(q_prompts):
            #         print(f"    [{i}]: {q_prompts[i][:150]}...")

            group_lengths.append(glen)
            for local_j, i in enumerate(range(start, end)):
                flat_prompts.append(q_prompts[i])
                flat_prompt_ids.append(q_prompt_ids[i])
                flat_ts.append(q_ts[i])
                flat_qids.append(q_qids[i])
                flat_segencs.append(q_segencs[i])
                flat_segs.append(q_segs[i])
                flat_ids.append(q_ids[i])
                flat_meta.append(q_meta[i])
                flat_completions.append(q_completions[i])
                flat_reasoner_log_rows.append({
                    "step": int(self.state.global_step),
                    "role": "reasoner",
                    "question_index": q_idx,
                    "controller_rollout_index": c,
                    "rollout_index_in_group": local_j,
                    "prompt": q_prompts[i],
                    "completion": q_completions[i],
                    "pred_answer": q_meta[i]["predicted_answer"],
                    "gold_answer": q_meta[i]["gold_answer"],
                    "reward_total": None,
                    "reward_accuracy": None,
                    "reward_format": None,
                })

        total_reasoners = len(flat_ids)
        if total_reasoners == 0:  # NEW
            print("WARNING: No reasoner completions kept.")  # NEW
            return flat_reasoner_log_rows,{  # NEW
                "total_reasoners": 0,
                "rewards_cpu": torch.empty(0),  # stays on CPU  # NEW
                "advantages": torch.empty(0, device=device),   # for loss on GPU  # NEW
                "completion_ids": torch.zeros((0, 1), dtype=torch.long, device=device),  # NEW
                "completion_mask": torch.zeros((0, 1), dtype=torch.long, device=device), # NEW
                "prompt_mask": torch.zeros((0, 1), dtype=torch.long, device=device),     # NEW
                "prompts_formatted": flat_prompts, "ts": flat_ts, "qids": flat_qids,               # NEW
                "segencs": flat_segencs, "segs": flat_segs                                # NEW
            }

        # === rewards (CPU) + advantages (GPU) ===  # NEW
        rewards_list, components = self.reasoner_reward_fn(  # NEW
            prompts=[], completions=flat_completions,
            completion_ids=[ids.tolist() for ids in flat_ids],
            metadata=flat_meta
        )
        rewards_cpu = torch.tensor(rewards_list)  # keep on CPU to save VRAM  # NEW
        rewards_gpu = rewards_cpu.to(device)      # only for advantages calc  # NEW

        # not grouping anymore -> use batch-wise advantages  # NEW
        adv_chunks, cur = [], 0
        for gl in group_lengths:
            r = rewards_gpu[cur:cur+gl]
            adv_chunks.append(torch.zeros_like(r) if gl <= 1 else (r - r.mean()) / (r.std() + 1e-8))
            cur += gl
        advantages = torch.cat(adv_chunks, dim=0)

        # === tensors for loss ===  # NEW
        completion_ids = pad(
            flat_ids, padding_value=self.pad_token_id, padding_side="right"
        ).to(device)  # NEW
        completion_mask = (completion_ids != self.pad_token_id).long()  # NEW

        prompt_inputs = self.processing_class(  # NEW
            text=flat_prompts, return_tensors="pt",
            padding=True, padding_side="left", add_special_tokens=False
        )
        
        prompt_mask_new = prompt_inputs["attention_mask"]#.to(device)  # NEW

        # === W&B rows ===  # NEW
        self._fill_reasoner_log_rows(  # NEW
            flat_reasoner_log_rows,
            rewards_cpu.detach().tolist(),
            advantages.detach().cpu().tolist(),
            components
        )

        # single return dict — no second update  # NEW
        return flat_reasoner_log_rows, {  # NEW
            "total_reasoners": total_reasoners,
            "rewards_cpu": rewards_cpu,                  # CPU for metrics  # NEW
            "effective_G": G,
            "advantages": advantages,                    # GPU for loss     # NEW
            "completion_ids": completion_ids,            # GPU              # NEW
            "prompt_ids": flat_prompt_ids,
            "completion_mask": completion_mask,          # GPU              # NEW
            "prompt_mask": prompt_mask_new,                  # GPU              # NEW
            "prompts_formatted": flat_prompts, "ts": flat_ts, "qids": flat_qids,    # lists  # NEW
            "segencs": flat_segencs, "segs": flat_segs                     # lists  # NEW
        }

    
    def _sample_and_flatten_reasoners_dynamic(
        self,
        all_controller_full_conv,
        all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
        all_reasoner_segment_encodings, all_reasoner_segments, all_reasoner_ids, reasoner_metadata,
        all_reasoner_completions, 
        all_reasoner_rollout_counts,
        all_reasoner_prompt_ids,
    ):
        """
        Samples reasoners per controller rollout (size = reasoner_keep, default = G),
        flattens them, computes rewards & advantages, and builds padded tensors.
        Returns a dict with everything needed by loss/metrics/output.
        """
        device = self.accelerator.device
        reasoner_keep = getattr(self, "reasoner_sample_size", None) or self.num_rollouts

        flat_prompts, flat_ts, flat_qids = [], [], []
        flat_segencs, flat_segs, flat_ids, flat_meta, flat_completions, flat_prompt_ids = [], [], [], [], [], []
        kept_per_ctrl = []
        rng = np.random.default_rng()
        flat_reasoner_log_rows = []
        group_lengths = []

        num_questions = len(all_controller_full_conv)
        for q_idx in range(num_questions):
            q_prompts = all_reasoner_prompts_formatted[q_idx]
            q_ts = all_reasoner_ts_data[q_idx]
            q_qids = all_reasoner_question_ids[q_idx]
            q_segencs = all_reasoner_segment_encodings[q_idx]
            q_segs = all_reasoner_segments[q_idx]
            q_ids = all_reasoner_ids[q_idx]
            q_meta = reasoner_metadata[q_idx]
            q_completions = all_reasoner_completions[q_idx]
            q_prompt_ids = all_reasoner_prompt_ids[q_idx]
            num_ctrl_rollouts_q = len(all_controller_full_conv[q_idx])

            if num_ctrl_rollouts_q == 0:
                continue
            
            G = self.num_rollouts
            counts = all_reasoner_rollout_counts[q_idx]
            valid = [i for i, c in enumerate(counts) if c > 0]
            if not valid:
                continue
            
            # compute rewards for all reasoners completions
            q_rewards_list, _ = self.reasoner_reward_fn(
            prompts=[],
            completions=q_completions,
            completion_ids=[ids.tolist() for ids in q_ids],
            metadata=q_meta,
            )
            q_rewards = torch.tensor(q_rewards_list, dtype=torch.float32) 
            
            # compute group std per controller rollout
            group_stds = []
            group_ranges = []  # list of (start, end) for each controller rollout index
            cur = 0
            for ci, glen in enumerate(counts):
                start = cur
                end = min(cur + glen, len(q_rewards))
                cur += glen
                group_ranges.append((start, end))

                if glen <= 0:
                    group_stds.append(0.0)
                    continue

                r = q_rewards[start:end]
                # std=0 if all same reward; for binary accuracy this means all-correct or all-wrong.
                std = float(r.std(unbiased=False)) if r.numel() > 1 else 0.0
                group_stds.append(std)
            
            # convert stds to probabilities
            eps = 1e-6
            stds_valid = np.array([group_stds[i] for i in valid], dtype=np.float64)

            # Optional: sharpen/soften with temperature (smaller => more peaked)
            temperature = 1.0
            weights = (stds_valid + eps) ** (1.0 / max(temperature, 1e-6))

            # Fallback: if all stds are ~0 (no informative groups), use uniform over valid
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                p = None  # numpy => uniform if p=None
            else:
                p = weights / weights.sum()

            # sample controller rollout index c by std-weighted probability
            c = int(rng.choice(valid, p=p))

            # --- 4) select chosen group slice ---
            start, end = group_ranges[c]
            glen = end - start
            if glen <= 0:
                continue
        
            group_lengths.append(glen)
            for local_j, i in enumerate(range(start, end)):
                flat_prompts.append(q_prompts[i])
                flat_prompt_ids.append(q_prompt_ids[i])
                flat_ts.append(q_ts[i])
                flat_segs.append(q_segs[i])
                flat_ids.append(q_ids[i])
                flat_meta.append(q_meta[i])
                flat_completions.append(q_completions[i])
                flat_reasoner_log_rows.append({
                    "step": int(self.state.global_step),
                    "role": "reasoner",
                    "question_index": q_idx,
                    "controller_rollout_index": c,
                    "rollout_index_in_group": local_j,
                    "prompt": q_prompts[i],
                    "completion": q_completions[i],
                    "pred_answer": q_meta[i]["predicted_answer"],
                    "gold_answer": q_meta[i]["gold_answer"],
                    "reward_total": None,
                    "reward_accuracy": None,
                    "reward_format": None,
                })

        total_reasoners = len(flat_ids)
        if total_reasoners == 0:  # NEW
            print("WARNING: No reasoner completions kept.")  # NEW
            return flat_reasoner_log_rows,{  # NEW
                "total_reasoners": 0,
                "rewards_cpu": torch.empty(0),  # stays on CPU  # NEW
                "advantages": torch.empty(0, device=device),   # for loss on GPU  # NEW
                "completion_ids": torch.zeros((0, 1), dtype=torch.long, device=device),  # NEW
                "completion_mask": torch.zeros((0, 1), dtype=torch.long, device=device), # NEW
                "prompt_mask": torch.zeros((0, 1), dtype=torch.long, device=device),     # NEW
                "prompts_formatted": flat_prompts, "ts": flat_ts, "qids": flat_qids,               # NEW
                "segencs": flat_segencs, "segs": flat_segs                                # NEW
            }

        # === rewards (CPU) + advantages (GPU) ===  # NEW
        rewards_list, components = self.reasoner_reward_fn(  # NEW
            prompts=[], completions=flat_completions,
            completion_ids=[ids.tolist() for ids in flat_ids],
            metadata=flat_meta
        )
        rewards_cpu = torch.tensor(rewards_list)  # keep on CPU to save VRAM  # NEW
        rewards_gpu = rewards_cpu.to(device)      # only for advantages calc  # NEW

        # not grouping anymore -> use batch-wise advantages  # NEW
        adv_chunks, cur = [], 0
        for gl in group_lengths:
            r = rewards_gpu[cur:cur+gl]
            adv_chunks.append(torch.zeros_like(r) if gl <= 1 else (r - r.mean()) / (r.std() + 1e-8))
            cur += gl
        advantages = torch.cat(adv_chunks, dim=0)

        # === tensors for loss ===  # NEW
        completion_ids = pad(
            flat_ids, padding_value=self.pad_token_id, padding_side="right"
        ).to(device)  # NEW
        completion_mask = (completion_ids != self.pad_token_id).long()  # NEW

        prompt_inputs = self.processing_class(  # NEW
            text=flat_prompts, return_tensors="pt",
            padding=True, padding_side="left", add_special_tokens=False
        )
        
        prompt_mask_new = prompt_inputs["attention_mask"]#.to(device)  # NEW

        # === W&B rows ===  # NEW
        self._fill_reasoner_log_rows(  # NEW
            flat_reasoner_log_rows,
            rewards_cpu.detach().tolist(),
            advantages.detach().cpu().tolist(),
            components
        )

        # single return dict — no second update  # NEW
        return flat_reasoner_log_rows, {  # NEW
            "total_reasoners": total_reasoners,
            "rewards_cpu": rewards_cpu,                  # CPU for metrics  # NEW
            "effective_G": G,
            "advantages": advantages,                    # GPU for loss     # NEW
            "completion_ids": completion_ids,            # GPU              # NEW
            "prompt_ids": flat_prompt_ids,
            "completion_mask": completion_mask,          # GPU              # NEW
            "prompt_mask": prompt_mask_new,                  # GPU              # NEW
            "prompts_formatted": flat_prompts, "ts": flat_ts, "qids": flat_qids,    # lists  # NEW
            "segencs": flat_segencs, "segs": flat_segs                     # lists  # NEW
        }
    
    def _sample_and_flatten_reasoners_dynamic_rloo(
        self,
        all_controller_full_conv,
        all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
        all_reasoner_segment_encodings, all_reasoner_segments, all_reasoner_ids, reasoner_metadata,
        all_reasoner_completions, 
        all_reasoner_rollout_counts,
        all_reasoner_prompt_ids,
    ):
        """
        Samples reasoners per controller rollout (size = reasoner_keep, default = G),
        flattens them, computes rewards & advantages, and builds padded tensors.
        Returns a dict with everything needed by loss/metrics/output.
        """
        device = self.accelerator.device
        reasoner_keep = getattr(self, "reasoner_sample_size", None) or self.num_rollouts

        flat_prompts, flat_ts, flat_qids = [], [], []
        flat_segencs, flat_segs, flat_ids, flat_meta, flat_completions, flat_prompt_ids = [], [], [], [], [], []
        kept_per_ctrl = []
        rng = np.random.default_rng()
        flat_reasoner_log_rows = []
        group_lengths = []

        num_questions = len(all_controller_full_conv)
        for q_idx in range(num_questions):
            q_prompts = all_reasoner_prompts_formatted[q_idx]
            q_ts = all_reasoner_ts_data[q_idx]
            q_qids = all_reasoner_question_ids[q_idx]
            q_segencs = all_reasoner_segment_encodings[q_idx]
            q_segs = all_reasoner_segments[q_idx]
            q_ids = all_reasoner_ids[q_idx]
            q_meta = reasoner_metadata[q_idx]
            q_completions = all_reasoner_completions[q_idx]
            q_prompt_ids = all_reasoner_prompt_ids[q_idx]
            num_ctrl_rollouts_q = len(all_controller_full_conv[q_idx])

            if num_ctrl_rollouts_q == 0:
                continue
            
            G = self.num_rollouts
            counts = all_reasoner_rollout_counts[q_idx]
            valid = [i for i, c in enumerate(counts) if c > 0]
            if not valid:
                continue
            
            # compute rewards for all reasoners completions
            q_rewards_list, _ = self.reasoner_reward_fn(
            prompts=[],
            completions=q_completions,
            completion_ids=[ids.tolist() for ids in q_ids],
            metadata=q_meta,
            )
            q_rewards = torch.tensor(q_rewards_list, dtype=torch.float32) 
            
            # compute group std per controller rollout
            group_stds = []
            group_ranges = []  # list of (start, end) for each controller rollout index
            cur = 0
            for ci, glen in enumerate(counts):
                start = cur
                end = min(cur + glen, len(q_rewards))
                cur += glen
                group_ranges.append((start, end))

                if glen <= 0:
                    group_stds.append(0.0)
                    continue

                r = q_rewards[start:end]
                # std=0 if all same reward; for binary accuracy this means all-correct or all-wrong.
                std = float(r.std(unbiased=False)) if r.numel() > 1 else 0.0
                group_stds.append(std)
            
            # convert stds to probabilities
            eps = 1e-6
            stds_valid = np.array([group_stds[i] for i in valid], dtype=np.float64)

            # Optional: sharpen/soften with temperature (smaller => more peaked)
            temperature = 1.0
            weights = (stds_valid + eps) ** (1.0 / max(temperature, 1e-6))

            # Fallback: if all stds are ~0 (no informative groups), use uniform over valid
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                p = None  # numpy => uniform if p=None
            else:
                p = weights / weights.sum()

            # sample controller rollout index c by std-weighted probability
            c = int(rng.choice(valid, p=p))

            # --- 4) select chosen group slice ---
            start, end = group_ranges[c]
            glen = end - start
            if glen <= 0:
                continue
        
            group_lengths.append(glen)
            for local_j, i in enumerate(range(start, end)):
                flat_prompts.append(q_prompts[i])
                flat_prompt_ids.append(q_prompt_ids[i])
                flat_ts.append(q_ts[i])
                flat_qids.append(q_qids[i])
                flat_segencs.append(q_segencs[i])
                flat_segs.append(q_segs[i])
                flat_ids.append(q_ids[i])
                flat_meta.append(q_meta[i])
                flat_completions.append(q_completions[i])
                flat_reasoner_log_rows.append({
                    "step": int(self.state.global_step),
                    "role": "reasoner",
                    "question_index": q_idx,
                    "controller_rollout_index": c,
                    "rollout_index_in_group": local_j,
                    "prompt": q_prompts[i],
                    "completion": q_completions[i],
                    "pred_answer": q_meta[i]["predicted_answer"],
                    "gold_answer": q_meta[i]["gold_answer"],
                    "reward_total": None,
                    "reward_accuracy": None,
                    "reward_format": None,
                })

        total_reasoners = len(flat_ids)
        if total_reasoners == 0:  # NEW
            print("WARNING: No reasoner completions kept.")  # NEW
            return flat_reasoner_log_rows,{  # NEW
                "total_reasoners": 0,
                "rewards_cpu": torch.empty(0),  # stays on CPU  # NEW
                "advantages": torch.empty(0, device=device),   # for loss on GPU  # NEW
                "completion_ids": torch.zeros((0, 1), dtype=torch.long, device=device),  # NEW
                "completion_mask": torch.zeros((0, 1), dtype=torch.long, device=device), # NEW
                "prompt_mask": torch.zeros((0, 1), dtype=torch.long, device=device),     # NEW
                "prompts_formatted": flat_prompts, "ts": flat_ts, "qids": flat_qids,               # NEW
                "segencs": flat_segencs, "segs": flat_segs                                # NEW
            }

        # === rewards (CPU) + advantages (GPU) ===  # NEW
        rewards_list, components = self.reasoner_reward_fn(  # NEW
            prompts=[], completions=flat_completions,
            completion_ids=[ids.tolist() for ids in flat_ids],
            metadata=flat_meta
        )
        rewards_cpu = torch.tensor(rewards_list)  # keep on CPU to save VRAM  # NEW
        rewards_gpu = rewards_cpu.to(device)      # only for advantages calc  # NEW

        # not grouping anymore -> use batch-wise advantages  # NEW
        adv_chunks, cur = [], 0
        for gl in group_lengths:
            r = rewards_gpu[cur:cur+gl]
            if gl <= 1:
                adv = torch.zeros_like(r)
            else:
                # Leave-one-out baseline
                adv = r - (r.sum() - r) / (gl - 1)
            adv_chunks.append(adv)
            cur += gl
        advantages = torch.cat(adv_chunks, dim=0)
        if advantages.numel() > 1:
            advantages = self.per_gpu_global_norm(advantages)
        # === tensors for loss ===  # NEW
        completion_ids = pad(
            flat_ids, padding_value=self.pad_token_id, padding_side="right"
        ).to(device)  # NEW
        completion_mask = (completion_ids != self.pad_token_id).long()  # NEW

        prompt_inputs = self.processing_class(  # NEW
            text=flat_prompts, return_tensors="pt",
            padding=True, padding_side="left", add_special_tokens=False
        )
        
        prompt_mask_new = prompt_inputs["attention_mask"]#.to(device)  # NEW

        # === W&B rows ===  # NEW
        self._fill_reasoner_log_rows(  # NEW
            flat_reasoner_log_rows,
            rewards_cpu.detach().tolist(),
            advantages.detach().cpu().tolist(),
            components
        )

        # single return dict — no second update  # NEW
        return flat_reasoner_log_rows, {  # NEW
            "total_reasoners": total_reasoners,
            "rewards_cpu": rewards_cpu,                  # CPU for metrics  # NEW
            "effective_G": G,
            "advantages": advantages,                    # GPU for loss     # NEW
            "completion_ids": completion_ids,            # GPU              # NEW
            "prompt_ids": flat_prompt_ids,
            "completion_mask": completion_mask,          # GPU              # NEW
            "prompt_mask": prompt_mask_new,                  # GPU              # NEW
            "prompts_formatted": flat_prompts, "ts": flat_ts, "qids": flat_qids,    # lists  # NEW
            "segencs": flat_segencs, "segs": flat_segs                     # lists  # NEW
        }

    def _compute_and_log_metrics(
        self,
        mode,
        question_accuracies_list,
        controller_selected_completions,
        controller_rewards,
        controller_advantages,
        reasoner_pack,
        controller_consistencies_all,
        controller_num_rounds_all,
        controller_format_errors_all,
    ):
        """
        Computes all metrics (token counts, lengths, rewards, accuracies, consistency, rounds, format errors)
        and updates self._metrics in-place. Returns nothing.
        """
        device = self.accelerator.device

        # controller token-length metric
        temp_controller_ids = [
            torch.tensor(self.processing_class.encode(txt, add_special_tokens=False))
            for txt in controller_selected_completions
        ]
        controller_completion_ids = (
            pad(temp_controller_ids, padding_value=self.pad_token_id, padding_side="right").to(device)
            if temp_controller_ids else torch.zeros((0, 1), dtype=torch.long, device=device)
        )
        controller_completion_mask = (controller_completion_ids != self.pad_token_id).long()
        controller_prompt_mask = torch.ones_like(controller_completion_mask)

        # token accounting
        if mode == "train":
            controller_tokens = (controller_completion_mask.sum() + controller_prompt_mask.sum()).item()
            reasoner_tokens = (
                (reasoner_pack["completion_mask"].sum() + reasoner_pack["prompt_mask"].sum()).item()
                if reasoner_pack["total_reasoners"] > 0 else 0
            )
            total_tokens = controller_tokens + reasoner_tokens
            self.state.num_input_tokens_seen += self.accelerator.gather(
                torch.tensor(total_tokens, device=device)
            ).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # lengths
        gathered_ctrl_lengths = self.gather_and_compute(controller_completion_mask.sum(1))
        self._metrics[mode]["completions/controller_mean_length"] = [gathered_ctrl_lengths.float().mean().item()]
        if reasoner_pack["total_reasoners"] > 0:
            gathered_rsn_lengths = self.gather_and_compute(reasoner_pack["completion_mask"].sum(1))
            self._metrics[mode]["completions/reasoner_mean_length"] = [gathered_rsn_lengths.float().mean().item()]
            self._metrics[mode]["completions/mean_length"] = [
                torch.cat([gathered_ctrl_lengths, gathered_rsn_lengths]).float().mean().item()
            ]
        else:
            self._metrics[mode]["completions/reasoner_mean_length"] = [0.0]
            self._metrics[mode]["completions/mean_length"] = [gathered_ctrl_lengths.float().mean().item()]

        # rewards
        if controller_rewards.numel() > 0:
            self._metrics[mode]["controller/reward_mean"] = [controller_rewards.mean().item()]
            self._metrics[mode]["controller/reward_std"] = [controller_rewards.std().item()]
        else:
            self._metrics[mode]["controller/reward_mean"] = [0.0]
            self._metrics[mode]["controller/reward_std"] = [0.0]

        if reasoner_pack["total_reasoners"] > 0:
            gathered_rsn_rewards = self.gather_and_compute(reasoner_pack["rewards_cpu"])
            self._metrics[mode]["reasoner/reward_mean"] = [gathered_rsn_rewards.mean().item()]
            self._metrics[mode]["reasoner/reward_std"] = [gathered_rsn_rewards.std().item()]
            # group std across effective_G per controller rollout
            reasoner_group_stds = []
            num_ctrl_total = len(controller_advantages)  # one advantage per controller rollout kept
            G = reasoner_pack["effective_G"]
            for i in range(num_ctrl_total):
                s = i * G
                e = s + G
                if e <= len(gathered_rsn_rewards):
                    reasoner_group_stds.append(gathered_rsn_rewards[s:e].std().item())
            self._metrics[mode]["reasoner/group_std_mean"] = [float(np.mean(reasoner_group_stds)) if reasoner_group_stds else 0.0]
            all_rewards = torch.cat([controller_rewards.detach().cpu(), gathered_rsn_rewards])
        else:
            self._metrics[mode]["reasoner/reward_mean"] = [0.0]
            self._metrics[mode]["reasoner/reward_std"] = [0.0]
            self._metrics[mode]["reasoner/group_std_mean"] = [0.0]
            all_rewards = controller_rewards.detach().cpu()
        self._metrics[mode]["reward"] = [all_rewards.mean().item()]

        # Advantage stats (controller)
        if controller_advantages.numel() > 0:
            self._metrics[mode]["controller/advantage_mean"] = [controller_advantages.mean().item()]
            self._metrics[mode]["controller/advantage_std"] = [controller_advantages.std().item() if controller_advantages.numel() > 1 else 0.0]
        else:
            self._metrics[mode]["controller/advantage_mean"] = [0.0]
            self._metrics[mode]["controller/advantage_std"] = [0.0]

        # Advantage stats (reasoner)
        if reasoner_pack["total_reasoners"] > 0:
            self._metrics[mode]["reasoner/advantage_mean"] = [reasoner_pack["advantages"].mean().item()]
            self._metrics[mode]["reasoner/advantage_std"] = [reasoner_pack["advantages"].std().item() if reasoner_pack["advantages"].numel() > 1 else 0.0]
        else:
            self._metrics[mode]["reasoner/advantage_mean"] = [0.0]
            self._metrics[mode]["reasoner/advantage_std"] = [0.0]

        # Reward variance
        if controller_rewards.numel() > 1:
            self._metrics[mode]["controller/reward_var"] = [controller_rewards.var().item()]
        else:
            self._metrics[mode]["controller/reward_var"] = [0.0]
        
        if reasoner_pack["total_reasoners"] > 1:
            self._metrics[mode]["reasoner/reward_var"] = [reasoner_pack["rewards_cpu"].var().item()]
        else:
            self._metrics[mode]["reasoner/reward_var"] = [0.0]

        # accuracies
        self._metrics[mode]["question_accuracies_mean"] = [
            float(np.mean(question_accuracies_list)) if len(question_accuracies_list) > 0 else 0.0
        ]

        # consistency / rounds / format errors
        if controller_consistencies_all:
            gathered_consistency = self.gather_and_compute(torch.tensor(controller_consistencies_all, device=device))
            self._metrics[mode]["consistency_mean"] = [gathered_consistency.float().mean().item()]
        else:
            self._metrics[mode]["consistency_mean"] = [0.0]

        if controller_num_rounds_all:
            gathered_num_rounds = self.gather_and_compute(torch.tensor(controller_num_rounds_all, device=device))
            self._metrics[mode]["num_rounds_mean"] = [gathered_num_rounds.float().mean().item()]
        else:
            self._metrics[mode]["num_rounds_mean"] = [0.0]

        if controller_format_errors_all:
            gathered_format_errors = self.gather_and_compute(torch.tensor(controller_format_errors_all, device=device))
            self._metrics[mode]["format_errors"] = [int(gathered_format_errors.sum().item())]
            self._metrics[mode]["format_error_rate"] = [gathered_format_errors.float().mean().item()]
        else:
            self._metrics[mode]["format_errors"] = [0]
            self._metrics[mode]["format_error_rate"] = [0.0]

        # cleanup
        del controller_completion_mask, controller_prompt_mask, controller_completion_ids

    # NEW
    def _package_output(
        self,
        controller_selected_full_conv, controller_selected_ts_data, controller_selected_qids,
        controller_advantages, controller_ref_logps_per_question, all_controller_prompt_ids, all_controller_completion_ids,
        reasoner_pack
    ):
        """
        Creates the final output dict for the trainer step.
        """
        device = self.accelerator.device
        output = {
            "controller": {
                "advantages": controller_advantages.detach(),
                "ts_data_list": controller_selected_ts_data,
                "all_controller_full_conv_list": controller_selected_full_conv,
                "ref_logps_per_question": controller_ref_logps_per_question,
                "prompt_ids_list": all_controller_prompt_ids,
                "completion_ids_list": all_controller_completion_ids,
            },
            "reasoner": {
                "completion_ids": reasoner_pack["completion_ids"].detach(),
                "completion_mask": reasoner_pack["completion_mask"].detach(),
                "advantages": reasoner_pack["advantages"].detach() if reasoner_pack["total_reasoners"] > 0 else torch.empty(0, device=device),
                "prompts_formatted": reasoner_pack["prompts_formatted"],
                "ts_data_list": reasoner_pack["ts"],
                "segments_list": reasoner_pack["segs"],
            }
        }
        return output

    def _generate_and_score_completions(self, inputs):
        """
        Refactored: orchestrates sub-steps with smaller helpers.
        Logic identical to the long version (controller GRPO by question; reasoner sampled groups).
        """
        start_time = time.time()
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        lightning_examples = [item['metadata']['lightning_example'] for item in inputs]
        prompts_text = [item['prompt'] for item in inputs]

        print(f"\n{'='*60}")
        print(f"Starting Generation & Scoring for {len(lightning_examples)} questions")
        print(f"{'='*60}")

        controller_log_rows = []
        reasoner_log_rows = []

        # 1) Controller loops
        loop_results, question_accuracies_list = self._run_controller_loops_batched(lightning_examples, prompts_text, inputs)

        valid_results = []
        skipped_count = 0
        
        for result in loop_results:
            if result.get("skipped", False) or result.get("exclude_from_loss", False):
                skipped_count += 1
                continue
            valid_results.append(result)
        
        if skipped_count > 0:
            print(f"Excluded {skipped_count}/{len(loop_results)} samples from loss")
        
        if len(valid_results) == 0:
            print(f"All samples skipped! Returning zero loss batch.")
            return self._create_zero_loss_batch()
        
        # Continue with only valid_results
        loop_results = valid_results
        
        # 2) Reasoner rollouts (and collect nested per-question structures)
        (
            all_controller_completions, all_controller_full_conv, all_controller_gt_segs,
            all_controller_ts_data, all_controller_question_ids, controller_metadata,
            all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
            all_reasoner_segment_encodings, all_reasoner_segments, reasoner_metadata,
            all_reasoner_completions, all_reasoner_ids,
            controller_consistencies_all, controller_num_rounds_all, controller_format_errors_all,
            all_reasoner_rollout_counts, all_controller_prompt_ids, all_controller_completion_ids,
            all_reasoner_prompt_ids,
        ) = self._build_reasoner_rollouts(lightning_examples, inputs, loop_results)

        # 3) Rewards & advantages
        reward_start = time.time()
        # if self.use_rloo_advantages:
        #     (
        #         controller_selected_full_conv,
        #         controller_selected_ts_data,
        #         controller_selected_qids,
        #         controller_advantages,
        #         controller_rewards,
        #         controller_selected_completions,
        #         controller_selected_prompt_ids, controller_selected_completion_ids,
        #     ) = self._compute_controller_rewards_advantages_rloo(
        #         all_controller_full_conv, all_controller_completions,
        #         all_controller_ts_data, all_controller_question_ids,
        #         controller_metadata, controller_log_rows,
        #         all_controller_prompt_ids, all_controller_completion_ids,
        #     )
        # else:
        (
            controller_selected_full_conv,
            controller_selected_ts_data,
            controller_selected_qids,
            controller_advantages,
            controller_rewards,
            controller_selected_completions,
            controller_selected_prompt_ids, controller_selected_completion_ids,
        ) = self._compute_controller_rewards_advantages(
            all_controller_full_conv, all_controller_completions,
            all_controller_ts_data, all_controller_question_ids,
            controller_metadata, controller_log_rows,
            all_controller_prompt_ids, all_controller_completion_ids,
        )
        
        if self.use_rloo_advantages:
            reasoner_log_rows, reasoner_pack = self._sample_and_flatten_reasoners_dynamic_rloo(
                all_controller_full_conv,
                all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
                all_reasoner_segment_encodings, all_reasoner_segments, all_reasoner_ids, reasoner_metadata,
                all_reasoner_completions, all_reasoner_rollout_counts, all_reasoner_prompt_ids,
            )
        else:   
            reasoner_log_rows, reasoner_pack = self._sample_and_flatten_reasoners_dynamic(
            all_controller_full_conv,
            all_reasoner_prompts_formatted, all_reasoner_ts_data, all_reasoner_question_ids,
            all_reasoner_segment_encodings, all_reasoner_segments, all_reasoner_ids, reasoner_metadata,
            all_reasoner_completions, all_reasoner_rollout_counts, all_reasoner_prompt_ids,
        )
        reward_elapsed = time.time() - reward_start

        # 4) Metrics
        metrics_start = time.time()
        self._compute_and_log_metrics(
            mode=mode,
            question_accuracies_list=question_accuracies_list,
            controller_selected_completions=controller_selected_completions,
            controller_rewards=controller_rewards,
            controller_advantages=controller_advantages,
            reasoner_pack=reasoner_pack,
            controller_consistencies_all=controller_consistencies_all,
            controller_num_rounds_all=controller_num_rounds_all,
            controller_format_errors_all=controller_format_errors_all,
        )
        metrics_elapsed = time.time() - metrics_start

        # 5) Compute reference model logprobs for KL divergence (if beta > 0)
        controller_ref_logps_per_question = None
        reasoner_ref_logps = None

        if self.beta != 0.0:
            print(f"Computing reference model logprobs for KL divergence (beta={self.beta})")
            
            # Controller reference logprobs
            # with torch.no_grad():
                # controller_ref_logps_per_question = self._compute_ref_logps_for_controller(
                #     controller_selected_full_conv,
                #     controller_selected_ts_data,
                #     controller_selected_qids,
                # )
            
            # Reasoner reference logprobs
            if reasoner_pack["total_reasoners"] > 0:
                with torch.no_grad():
                    reasoner_ref_logps = self._compute_ref_logps_for_reasoner(
                        prompts_formatted=reasoner_pack["prompts_formatted"],
                        completion_ids=reasoner_pack["completion_ids"],
                        completion_mask=reasoner_pack["completion_mask"],
                        ts_data_list=reasoner_pack["ts"],
                        question_ids_list=reasoner_pack["qids"],
                        segments_list=reasoner_pack["segs"],
                        segment_encodings_list=reasoner_pack["segencs"],
                    )

        # Package output
        output = self._package_output(
            controller_selected_full_conv, controller_selected_ts_data, controller_selected_qids,
            controller_advantages, controller_ref_logps_per_question, controller_selected_prompt_ids, controller_selected_completion_ids,
            reasoner_pack
        )
        
        # Add reasoner ref logprobs to output
        if reasoner_ref_logps is not None:
            output["reasoner"]["ref_per_token_logps"] = reasoner_ref_logps
  
        del loop_results
        # Cleanup + logging tables
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_memory("_generate_and_score_completions: end of function")

        if self.accelerator.is_main_process:
            self._controller_logs.extend(controller_log_rows)
            self._reasoner_logs.extend(reasoner_log_rows)

        return output

    def _flatten_groups(self, groups):
        """Flatten List[List[T]] -> List[T] without copying tensor payloads."""
        flat = []
        for g in groups:
            flat.extend(g)
        return flat

    def _create_zero_loss_batch(self):
        """Return output structure when all samples are skipped"""
        device = self.accelerator.device
        return {
            "controller": {
                "advantages": torch.empty(0, device=device),
                "ts_data_list": [],
                "all_controller_full_conv_list": [],
                "ref_logps_per_question": None,
                "prompt_ids_list": [],
                "completion_ids_list": [],
            },
            "reasoner": {
                "completion_ids": torch.empty(0, 0, dtype=torch.long, device=device),
                "completion_mask": torch.empty(0, 0, device=device),
                "advantages": torch.empty(0, device=device),
                "prompts_formatted": [],
                "ts_data_list": [],
                "segments_list": [],
            },
        }

    def _is_empty_batch(self, inputs):
        """Check if batch has no data to train on"""
        controller = inputs.get("controller", {})
        reasoner = inputs.get("reasoner", {})
        
        # Check controller - advantages is a tensor
        controller_adv = controller.get("advantages")
        controller_empty = (controller_adv is None or controller_adv.numel() == 0)
        
        # Check reasoner - advantages is a tensor
        reasoner_adv = reasoner.get("advantages")
        reasoner_empty = (reasoner_adv is None or reasoner_adv.numel() == 0)
        
        return controller_empty and reasoner_empty
        
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss for both controller and reasoner"""
        # print_gpu_memory("compute_loss: before compute_loss_for_controller")
        if return_outputs:
            raise ValueError("Does not support returning outputs")
        
        if self._is_empty_batch(inputs):
            print("Empty batch - returning zero loss")
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        # Compute loss for controller
        controller_loss = self._compute_loss_for_controller(model, inputs["controller"])
        print_gpu_memory("compute_loss: after compute_loss_for_controller")
        # Compute loss for reasoner
        reasoner_batch_size = len(inputs["reasoner"]["completion_ids"])
        if reasoner_batch_size == 0:
            print("No reasoner completions, skipping reasoner loss")
            clean_input_data(inputs)
            return controller_loss  / self.current_gradient_accumulation_steps
        else:
            reasoner_loss = self._compute_loss(model, inputs["reasoner"])
            print_gpu_memory("compute_loss: after compute_loss_for_reasoner")
            # Combine (could use different weights here!)
            total_loss = (controller_loss + reasoner_loss) / self.current_gradient_accumulation_steps
            print(f"   Total loss: {total_loss.item()}")
            print(f"   Controller loss: {controller_loss.item()}")
            print(f"   Reasoner loss: {reasoner_loss.item()}")  
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"   WARNING: Loss is {total_loss.item()}!")
                print(f"   Controller loss: {controller_loss.item()}")
                print(f"   Reasoner loss: {reasoner_loss.item()}")
                # # Return a small stable loss to prevent crash
                # return torch.tensor(0.01, device=total_loss.device, requires_grad=True)

            clean_input_data(inputs)
            print_gpu_memory("compute_loss: after compute_loss")
            return total_loss

    def _compute_loss(self, model, inputs):
        """
        Compute loss with TS embedding injection
        
        Overrides TRL's _compute_loss to handle custom TS embeddings
        """
        # print_gpu_memory("_compute_loss: before function")
        completion_ids = inputs["completion_ids"]
        # prompt_ids = inputs["prompt_ids"]
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]
        # old_per_token_logps = inputs.get("old_per_token_logps")
        
        # Get metadata for TS injection
        prompts_formatted = inputs["prompts_formatted"]
        ts_data_list = inputs["ts_data_list"]
        segments_list = inputs["segments_list"]

        batch_size = completion_ids.size(0)
        device = model.device
        
        # Compute per-token log probs and entropies WITH TS INJECTION
        per_token_logps_list = []
        entropies_list = []
        
        for i in range(batch_size):
            # Get data for this sample
            prompt_formatted = prompts_formatted[i]
            completion_ids_i = completion_ids[i]
            completion_mask_i = completion_mask[i]
            # prompt_ids_i = prompt_ids[i]

            ts_data_i = ts_data_list[i]
            segments_i = segments_list[i]
            
            # Get completion length (without padding)
            completion_length = completion_mask_i.sum().item()
            completion_ids_i_nopad = completion_ids_i[:completion_length].unsqueeze(0)

            A_i = advantages[i].item()
            if A_i == 0.0 and self.beta == 0.0 and self.top_entropy_quantile == 1.0:
                full_completion_length = completion_mask_i.size(0)
                per_token_logps_list.append(torch.zeros(full_completion_length, device=device))
                entropies_list.append(torch.zeros(full_completion_length, device=device))
                continue
            
            # Compute log probs and entropy for this sample
            logps_i, entropy_i = self._compute_logps_with_ts_injection(
                prompt_formatted=prompt_formatted,
                # prompt_ids=prompt_ids_i,
                completion_ids=completion_ids_i,
                completion_mask=completion_mask_i,
                ts_data=ts_data_i,
                segments=segments_i,
                model=model,
                return_entropies=True,
            )
            logps_i = logps_i.squeeze(0)
            entropy_i = entropy_i.squeeze(0)
            
            # # Pad to match completion_ids shape
            # if logps_i.size(0) < completion_ids.size(1):
            #     pad_length = completion_ids.size(1) - logps_i.size(0)
            #     logps_i = torch.cat([
            #         logps_i,
            #         torch.zeros(pad_length, device=device)
            #     ], dim=0)
            #     entropy_i = torch.cat([
            #         entropy_i,
            #         torch.zeros(pad_length, device=device)
            #     ], dim=0)
            
            per_token_logps_list.append(logps_i)
            entropies_list.append(entropy_i)
        
        per_token_logps = pad(
        per_token_logps_list,
        padding_value=0.0,
        padding_side="right"
        )
        entropies = pad(
        entropies_list,
        padding_value=0.0,
        padding_side="right"
        )
        

        B, Lp = per_token_logps.size()
        if completion_mask.size(1) != Lp:
            print(f"\n[DEBUG] Shape mismatch in _compute_loss:")
            print(f"  Completion mask shape: {completion_mask.shape}")
            print(f"  Per token logps shape: {per_token_logps.shape}")
            print(f"  Batch size: {B}")
            print(f"  Number of prompts: {len(prompts_formatted)}")
            
            # Check for duplicate prompts
            print(f"\n  Checking prompt uniqueness:")
            for idx, prompt in enumerate(prompts_formatted):
                # Show segment info from each prompt
                if "### Segment" in prompt:
                    seg_start = prompt.find("### Segment")
                    seg_end = prompt.find("\n\n", seg_start) if seg_start != -1 else -1
                    if seg_start != -1 and seg_end != -1:
                        print(f"    Prompt {idx} segments: {prompt[seg_start:seg_end][:200]}")
                    else:
                        print(f"    Prompt {idx}: Segment info not found properly")
                else:
                    print(f"    Prompt {idx}: No segment marker found")
            
            # Check if prompts are actually the same
            if len(prompts_formatted) > 1:
                prompts_equal = all(p == prompts_formatted[0] for p in prompts_formatted)
                print(f"  All prompts equal: {prompts_equal}")

            lengths = torch.tensor([x.size(0) for x in per_token_logps_list], device=device)
            new_completion_mask = (torch.arange(Lp, device=device).unsqueeze(0) < lengths.unsqueeze(1)).long()
            completion_mask = new_completion_mask.to(device)
        else:
            completion_mask = completion_mask.to(device)
        # # If old_per_token_logps not provided, use detached current
        # if old_per_token_logps is None:
        #     old_per_token_logps = per_token_logps.detach()
        # SPELL-style loss: Direct policy gradient (no ratio, no clipping)
        per_token_loss = -per_token_logps * advantages.unsqueeze(1)
        # Entropy masking
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, 
                completion_mask, 
                1 - self.top_entropy_quantile
            )
            per_token_loss = per_token_loss * entropy_mask
        else:
            entropy_mask = None
        
        # KL divergence
        if self.beta != 0.0:
            ref_per_token_logps = inputs.get("ref_per_token_logps", None)
            if ref_per_token_logps is not None:
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - 
                    (ref_per_token_logps - per_token_logps) - 1
                )
                per_token_loss = per_token_loss + self.beta * per_token_kl

        # Aggregate loss
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            # loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            # Group normalization (SPELL style)
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            # loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            # loss = loss / self.current_gradient_accumulation_steps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Log metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        
        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count
        
        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())
        
        # Entropy metric, should we detach here? TODO
        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        
        del per_token_logps, entropies
        del completion_ids, completion_mask, advantages, prompts_formatted, ts_data_list, segments_list
        del per_token_logps_list, entropies_list
        # print_gpu_memory("_compute_loss: after function")
        return loss

    def log(self, logs: dict[str, float], start_time = None) -> None:
        '''Override to add controller/reasoner wandb tables'''
        from typing import Optional
        
        # Log grad_norm post-clipping (base trainer logs pre-clip as 'grad_norm')
        if hasattr(self, '_grad_norm') and self._grad_norm is not None:
            # Post-clip norm is min(pre_clip_norm, max_grad_norm)
            max_norm = getattr(self.args, 'max_grad_norm', None)
            if max_norm is not None and max_norm > 0:
                logs["grad_norm_post_clip"] = min(float(self._grad_norm), max_norm)
            else:
                logs["grad_norm_post_clip"] = float(self._grad_norm)
        
        # Call parent log first (handles metrics aggregation and base logging)
        super().log(logs, start_time)
        
        # Log controller/reasoner tables to wandb
        if self.accelerator.is_main_process and self.args.report_to and "wandb" in self.args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    # Log controller table if we have data
                    if self._controller_logs:
                        import pandas as pd
                        controller_df = pd.DataFrame(self._controller_logs)
                        wandb.log({
                            "controller_completions": wandb.Table(dataframe=controller_df)
                        })
                        print(f"Logged {len(self._controller_logs)} controller samples to wandb")
                        self._controller_logs.clear()
                    
                    # Log reasoner table if we have data
                    if self._reasoner_logs:
                        import pandas as pd
                        reasoner_df = pd.DataFrame(self._reasoner_logs)
                        reasoner_df.dropna(inplace=True)
                        if len(reasoner_df) > 20:
                            reasoner_df = reasoner_df.sample(n=20, random_state=42)
                        wandb.log({
                            "reasoner_completions": wandb.Table(dataframe=reasoner_df)
                        })
                        print(f"Logged {len(self._reasoner_logs)} reasoner samples to wandb")
                        self._reasoner_logs.clear()
            except Exception as e:
                print(f"Warning: Failed to log controller/reasoner tables to wandb: {e}")
                import traceback
                traceback.print_exc()
                self._controller_logs.clear()
                self._reasoner_logs.clear()
        else:
            self._controller_logs.clear()
            self._reasoner_logs.clear()

    def _unwrap_model(self, model):
        """Unwrap DDP/DeepSpeed wrapper if present"""
        return model.module if hasattr(model, 'module') else model

    def compute_logprobs_for_controller(self, model, full_conv_list, ts_data, return_entropies=False):
        """
        Compute log probabilities for controller completions.
        
        Args:
            model: The model to use
            full_conv_list: List of [messages, completion, messages, completion, ...]
            ts_data: Time series tensor
            question_ids: Question token IDs
            return_entropies: Whether to return entropies
            
        Returns:
            per_token_logps_list: List of per-token log probs for each completion
            entropies_list: List of entropies for each completion
        """
        try:
            model = self._unwrap_model(model)
            per_token_logps_list = []
            entropies_list = []
            device = model.device
            
            ts_length = len(ts_data)
            ts_tensor = torch.tensor(ts_data, dtype=torch.float32)

            ts_segs = [[0, ts_length]]  # Full TS
            ts_for_model = self.cr_generator._prepare_ts_segments_for_model(
                ts_tensor.unsqueeze(0),
                [ts_segs]
            )
            if ts_for_model is not None:
                if self.task == "TSQA" or self.task == "TRQA_MIXED":    
                    ts_for_model = ts_for_model.to(device, dtype=torch.bfloat16)
                else:
                    ts_for_model = ts_for_model.to(device)
            
            j = 0
            while j < len(full_conv_list):
                # Get prompt (messages) and completion
                prompt_txt = full_conv_list[j]
                completion_txt = full_conv_list[j + 1]
                
                # Format prompt with chat template
                prompt_txt = self.processing_class.apply_chat_template(
                    prompt_txt,
                    tokenize=False,
                    tools=TS_TOOLS,
                    add_generation_prompt=True,  # Add generation prompt for proper formatting
                    enable_thinking=True
                )
                
                completion_txt = self.processing_class.apply_chat_template(
                    [completion_txt],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True
                )
                
                # Tokenize prompt and completion
                prompt_ids = self.processing_class.encode(prompt_txt, add_special_tokens=False)
                completion_ids = self.processing_class.encode(completion_txt, add_special_tokens=False)
                
                prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
                completion_tensor = torch.tensor(completion_ids, device=device, dtype=torch.long).unsqueeze(0)
                
                prompt_len = prompt_tensor.size(1)
                completion_len = completion_tensor.size(1)
                
                # Concatenate prompt + completion for full sequence
                total_ids = torch.cat([prompt_tensor, completion_tensor], dim=1)
                total_attn = torch.ones_like(total_ids)
                
                outputs = model(
                    input_ids=total_ids,
                    attention_mask=total_attn,
                    timeseries=ts_for_model,
                    return_dict=True,
                    use_cache=False
                )
                
                logits = outputs.logits
                
                # Shift for next-token prediction
                logits = logits[:, :-1, :]
                logits = logits[:, -completion_len:, :]
                # We want logits that predict the completion tokens
                # These are at positions [prompt_len-1 : prompt_len + completion_len - 1]
                # logits = logits[:, prompt_len - 1:prompt_len + completion_len - 1, :]
                
                # Apply temperature
                logits = logits / self.controller_temperature
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"NaN/Inf after temperature scaling at round {j // 2 + 1}!")
                    print(f"Temperature: {self.controller_temperature}")
                    return None, None
                
                # Compute log probs
                log_probs = torch.log_softmax(logits, dim=-1)
                per_token_logps = log_probs.gather(-1, completion_tensor.unsqueeze(-1)).squeeze(-1)
                
                if torch.isnan(log_probs).any():
                    print(f"NaN in log_probs at round {j // 2 + 1}!")
                    return None, None
                
                if torch.isinf(log_probs).any():
                    print(f"Inf in log_probs at round {j // 2 + 1}!")
                    return None, None
                
                if torch.isnan(per_token_logps).any():
                    print(f"NaN in per_token_logps at round {j // 2 + 1}!")
                    return None, None
                
                # Check for extreme values
                if per_token_logps.min() < -100:
                    print(f"Extremely small log probs at round {j // 2 + 1}: min={per_token_logps.min().item():.4f}")
                
                if per_token_logps.max() > 0:
                    print(f"POSITIVE log probs at round {j // 2 + 1}! max={per_token_logps.max().item():.4f}")
                    return None, None
                
                # Compute entropy
                probs = torch.softmax(logits, dim=-1)
                entropies = -(probs * log_probs).sum(dim=-1)
                
                if torch.isnan(probs).any():
                    print(f"NaN in probs at round {j // 2 + 1}!")
                    return None, None
                
                # Store results for this round
                per_token_logps_list.append(per_token_logps.squeeze(0))
                entropies_list.append(entropies.squeeze(0))
                
                j += 2  # Move to next round
                
                # Cleanup
                del logits, log_probs, probs, total_ids, total_attn, outputs
                del prompt_tensor, completion_tensor
            
            # Final cleanup
            del ts_for_model
            torch.cuda.empty_cache()
            
            return per_token_logps_list, entropies_list
            
        except Exception as e:
            print(f"Error in compute_logprobs_for_controller: {e}")
            import traceback
            traceback.print_exc()
            raise e


    def _compute_loss_for_controller(self, model, inputs, return_entropies=False):
        """
        Version with KL divergence support
        """
        advantages = inputs["advantages"]
        all_controller_full_conv_list = inputs["all_controller_full_conv_list"]
        ts_data_list = inputs["ts_data_list"]
        ref_logps_per_question = inputs.get("ref_logps_per_question", None)  # Use .get() for safety
        completion_ids_list = inputs["completion_ids_list"] 
        prompt_ids_list = inputs["prompt_ids_list"]
        num_questions = len(all_controller_full_conv_list)
        device = model.device
        # print_gpu_memory("_compute_loss_for_controller: before function")

        print(f"Number of questions: {num_questions}")
        print(f"Advantages: {advantages.cpu().tolist()}")
        print(f"Advantages stats: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}, min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")
        
        if torch.isnan(advantages).any():
            print(f"        NaN in advantages before any computation!")
            print(f"   Advantages: {advantages}")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        if advantages.abs().sum() < 1e-8:
            print(f"        WARNING: All advantages are nearly zero!")
            print(f"   This can cause issues - adding small noise")
            advantages = advantages + torch.randn_like(advantages) * 1e-5
        
        question_losses = []
        all_entropies_for_logging = []
        
        for question_idx in range(num_questions):
            print(f"\n--- Processing question {question_idx + 1}/{num_questions} ---")
            prompt_ids_i = prompt_ids_list[question_idx]
            completion_ids_i = completion_ids_list[question_idx]
            full_conv_list = all_controller_full_conv_list[question_idx]
            ts_data_i = ts_data_list[question_idx]
            question_advantage = advantages[question_idx]

            print(f"  Question advantage: {question_advantage.item():.6f}")
            # Policy model log probs
            per_token_logps_list, entropies_list = self.compute_logprobs_for_controller(
                model, full_conv_list, ts_data_i,
            )
            if per_token_logps_list is None or entropies_list is None:
                print(f"        compute_logprobs_for_controller returned None!")
                return torch.tensor(0.0, device=device, requires_grad=True)
            entropies = pad(entropies_list, padding_value=0.0, padding_side="right")
            
            # Mask
            original_lengths = [len(logps) for logps in per_token_logps_list]
            # Pad
            per_token_logps = pad(per_token_logps_list, padding_value=0.0, padding_side="right")
            
            if torch.isnan(per_token_logps).any():
                print(f"        NaN in per_token_logps after padding!")
                print(f"     Original per_token_logps_list lengths: {[x.shape for x in per_token_logps_list]}")
                for i, logps in enumerate(per_token_logps_list):
                    if torch.isnan(logps).any():
                        print(f"     NaN found in element {i}: {logps}")
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            print(f"  Logprobs stats: min={per_token_logps.min().item():.4f}, max={per_token_logps.max().item():.4f}, mean={per_token_logps.mean().item():.4f}")
            
            # Create mask based on actual lengths
            completion_mask = torch.zeros_like(per_token_logps)
            for idx, orig_len in enumerate(original_lengths):
                completion_mask[idx, :orig_len] = 1.0
            effective_mask = completion_mask
            
            if self.top_entropy_quantile < 1.0:
                entropy_mask = self.get_high_entropy_mask(
                    entropies, completion_mask.long(), 1 - self.top_entropy_quantile
                )
                effective_mask = effective_mask * entropy_mask.float()
            
            print(f"  Effective mask sum: {effective_mask.sum().item()}")
            # Per-token loss (define BEFORE using it!)
            per_token_loss = -per_token_logps * question_advantage
            
            if torch.isnan(per_token_loss).any():
                print(f"        NaN in per_token_loss!")
                print(f"     per_token_logps: {per_token_logps}")
                print(f"     question_advantage: {question_advantage}")
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            print(f"  Per-token loss (before KL): min={per_token_loss.min().item():.4f}, max={per_token_loss.max().item():.4f}")
            
            # Add KL penalty (if enabled)
            if self.beta != 0.0 and ref_logps_per_question is not None:
                ref_logps = ref_logps_per_question[question_idx]  # Already padded
                
                # Ensure same shape
                if ref_logps.size() != per_token_logps.size():
                    max_len = max(per_token_logps.size(1), ref_logps.size(1))
                    if per_token_logps.size(1) < max_len:
                        per_token_logps = torch.cat([
                            per_token_logps,
                            torch.zeros(per_token_logps.size(0), max_len - per_token_logps.size(1), device=device)
                        ], dim=1)
                    if ref_logps.size(1) < max_len:
                        ref_logps = torch.cat([
                            ref_logps,
                            torch.zeros(ref_logps.size(0), max_len - ref_logps.size(1), device=device)
                        ], dim=1)
                
                # Compute KL divergence
                per_token_kl = (
                    torch.exp(ref_logps - per_token_logps) -
                    (ref_logps - per_token_logps) - 1
                )
                per_token_loss = per_token_loss + self.beta * per_token_kl
            
            # Question loss
            masked_loss = per_token_loss * effective_mask
            question_loss = masked_loss.sum() / effective_mask.sum().clamp(min=1.0)

            if torch.isnan(masked_loss).any():
                print(f"        NaN in masked_loss!")
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            denominator = effective_mask.sum().clamp(min=1.0)
            
            # ★ DEBUG CHECK 11: Question loss
            if torch.isnan(question_loss):
                print(f"        NaN in question_loss!")
                print(f"     masked_loss sum: {masked_loss.sum().item()}")
                print(f"     denominator: {denominator.item()}")
                return torch.tensor(0.0, device=device, requires_grad=True)

            print(f"  ✓ Question loss: {question_loss.item():.6f}")
            question_losses.append(question_loss)
            for e in entropies_list:
                if e.numel() > 0:
                    all_entropies_for_logging.append(e.detach())
            
            del per_token_logps_list, entropies_list
            del per_token_logps, entropies, completion_mask, effective_mask, masked_loss, denominator, per_token_loss

        
        # Average across questions
        loss = torch.stack(question_losses).mean()
        print(f"\n{'='*80}")
        print(f"Final controller loss: {loss.item():.6f}")
        print(f"{'='*80}\n")
        # print_gpu_memory("_compute_loss_for_controller: after function")
        # Logging
        mode = "train" if self.model.training else "eval"
        if all_entropies_for_logging:
            all_entropies_flat = torch.cat([e.flatten() for e in all_entropies_for_logging if len(e) > 0])
            if len(all_entropies_flat) > 0:
                mean_entropy = all_entropies_flat.mean()
                self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        del all_entropies_for_logging
        return loss

    def _compute_ref_logps_for_controller(
        self,
        controller_selected_full_conv,
        controller_selected_ts_data,
        controller_selected_qids,
    ):
        """
        Compute reference model log probabilities for controller KL divergence.
        Uses the frozen reference model (or base model with LoRA disabled).
        
        Returns:
            List of padded reference logprobs tensors, one per question
        """
        if self.beta == 0.0:
            return None
        
        device = self.accelerator.device
        ref_logps_per_question = []
        
        for q_idx, full_conv_list in enumerate(controller_selected_full_conv):
            ts_data = controller_selected_ts_data[q_idx]
            
            # Compute logprobs with reference model
            if self.ref_model is not None:
                # Using separate reference model
                ref_logps_list, _ = self.compute_logprobs_for_controller(
                    self.ref_model, full_conv_list, ts_data
                )
            else:
                # Using LoRA - disable adapter for reference
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logps_list, _ = self.compute_logprobs_for_controller(
                        self.model, full_conv_list, ts_data
                    )
            
            # Pad and stack
            ref_logps_padded = pad(ref_logps_list, padding_value=0.0, padding_side="right")
            ref_logps_per_question.append(ref_logps_padded.detach())
        
        return ref_logps_per_question

    def _compute_ref_logps_for_reasoner(
        self,
        prompts_formatted,
        completion_ids,
        completion_mask,
        ts_data_list,
        question_ids_list,
        segments_list,
        segment_encodings_list,
    ):
        """
        Compute reference model log probabilities for reasoner completions.
        
        Returns:
            Padded tensor of reference logprobs (batch_size, max_seq_len)
        """
        if self.beta == 0.0:
            return None
        
        device = self.accelerator.device
        batch_size = completion_ids.size(0)
        ref_logps_list = []
        
        for i in range(batch_size):
            prompt = prompts_formatted[i]
            comp_ids_i = completion_ids[i]
            comp_mask_i = completion_mask[i]
            ts_data = ts_data_list[i]
            segments = segments_list[i]
            
            # Compute with reference model
            if self.ref_model is not None:
                ref_logps, _ = self._compute_logps_with_ts_injection(
                    prompt_formatted=prompt,
                    completion_ids=comp_ids_i,
                    completion_mask=comp_mask_i,
                    ts_data=ts_data,
                    segments=segments,
                    model=self.ref_model,
                    return_entropies=True,
                )
            else:
                # LoRA - disable adapter
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logps, _ = self._compute_logps_with_ts_injection(
                        prompt_formatted=prompt,
                        completion_ids=comp_ids_i,
                        completion_mask=comp_mask_i,
                        ts_data=ts_data,
                        segments=segments,
                        model=self.model,
                        return_entropies=True,
                        
                    )
            
            ref_logps_list.append(ref_logps.squeeze(0).detach())
        
        # Pad and stack
        ref_logps_padded = pad(ref_logps_list, padding_value=0.0, padding_side="right")
        return ref_logps_padded.detach()

    def _compute_logps_with_ts_injection(
        self,
        prompt_formatted,  # Full prompt string with chat template applied
        # prompt_ids,
        completion_ids,  # Shape: (1, completion_len)
        completion_mask,  # Shape: (1, completion_len)
        ts_data,
        segments=None,  # None for controller (full TS), list for reasoner
        model=None,
        return_entropies=False,
    ):
        """
        Compute log probs for completion with TS embedding injection
        
        Args:
            prompt_formatted: Formatted prompt string (has TS_PLACEHOLDER)
            completion_ids: Completion token IDs (1, L)
            ts_data: Time series data
            question_ids: Question token IDs
            segments: Segment indices (for reasoner) or None (for controller)
            segment_encodings: Pre-encoded segments
            model: Model to use (default: self.model)
        
        Returns:
            per_token_logps: Log probs for completion tokens (1, L)
        """ 
        # print_gpu_memory("_compute_logps_with_ts_injection: before function")
        device = model.device
        model = self._unwrap_model(model)
        # Tokenize the prompt
        prompt_ids = self.processing_class.encode(prompt_formatted, add_special_tokens=False)
        prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0)
        completion_ids = completion_ids.to(device).unsqueeze(0)
        completion_mask = completion_mask.to(device).unsqueeze(0)
        prompt_mask = torch.ones_like(prompt_tensor)
        # completion_mask = torch.ones_like(completion_ids)
        ts_length = len(ts_data)
        ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
        # Determine segments
        if segments is None:
            ts_segs = [[0, ts_length]]
        else:
            # Reasoner: use provided segments
            ts_segs = segments
        
        # Prepare TS for model
        ts_for_model = self.cr_generator._prepare_ts_segments_for_model(
            ts_tensor.unsqueeze(0),
            [ts_segs]
        )
        if ts_for_model is not None:
            if self.task == "TSQA" or self.task == "TRQA_MIXED":
                ts_for_model = ts_for_model.to(device, dtype=torch.bfloat16)
            else:
                ts_for_model = ts_for_model.to(device)
        
        # Concatenate prompt + completion
        total_ids = torch.cat([prompt_tensor, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        

        outputs = model(
            input_ids=total_ids,
            attention_mask=attention_mask,
            timeseries=ts_for_model,
            return_dict=True,
            use_cache=False
        )
         
        # logits = model.lm_head(outputs.last_hidden_state)
        logits = outputs.logits
        
        # Remove last position (predicts next token)
        logits = logits[:, :-1, :]
        
        # Keep only completion part
        completion_length = completion_ids.size(1)
        logits = logits[:, -completion_length:, :]
        
        # Divide by temperature
        logits = logits / self.reasoner_temperature # temperature # self.temperature
        
        # Compute log probs
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logps = log_probs.gather(-1, completion_ids.to(device).unsqueeze(-1)).squeeze(-1)
        
        # Compute entropy
        probs = torch.softmax(logits, dim=-1)
        entropies = -(probs * log_probs).sum(dim=-1)
        
        del logits, log_probs, probs, outputs
        del ts_tensor, attention_mask, ts_for_model
        torch.cuda.empty_cache()

        # print_gpu_memory("_compute_logps_with_ts_injection: after function")
        if return_entropies:
            return per_token_logps, entropies
        else:
            return per_token_logps

    def _build_conversation_for_controller(self, full_conv_list, round_idx):
        """Split conversation into prompt (context) + completion (response)"""
        from tools_qwents import TS_TOOLS
        TS_PLACEHOLDER = "<ts><ts/>" 
        
        # Flatten structure
        messages = []
        if isinstance(full_conv_list[0], list):
            messages.extend(full_conv_list[0])
            messages.extend(full_conv_list[1:])
        else:
            messages = full_conv_list
        
        # Split at this round
        if round_idx == 0:
            prompt_messages = messages[:2]  # system + user
            completion_message = messages[2]
        else:
            prompt_end = 2 + round_idx * 2
            prompt_messages = messages[:prompt_end]
            completion_message = messages[prompt_end]
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, tools=TS_TOOLS,
            add_generation_prompt=True, enable_thinking=True
        )
        
        # Verify TS_PLACEHOLDER preserved
        if TS_PLACEHOLDER not in prompt_text:
            raise ValueError(f"TS_PLACEHOLDER lost during chat template!")
        
        return prompt_text, completion_message["content"]

    def gather_and_compute(self, tensor_or_list, compute_fn=None):
        """Move to GPU, gather, compute, return CPU result, cleanup"""
        device = self.accelerator.device
        if isinstance(tensor_or_list, list):
            tensor = torch.tensor(tensor_or_list, device=device)
        else:
            tensor = tensor_or_list.to(device)
        
        # gathered = self.accelerator.gather(tensor)
        
        if compute_fn:
            result = compute_fn(tensor)
        else:
            result = tensor
        
        # Move result to CPU and cleanup GPU tensors
        if isinstance(result, torch.Tensor):
            result = result.cpu()
        
        return result

    def _compute_logps_with_ts_injection_for_controller(
        self,
        full_conv_ids,  # Full conversation history for controller
        question_ids,
        ts_data,
        model=None,
        return_entropies=False,
    ):
        """
        Compute log probs for full conversation sequence with TS injection
    
        Args:
            input_ids: Token IDs for full conversation (B, seq_len)
            ts_data: Time series data
            question_ids: Question token IDs
            model: Model to use
            return_entropies: Whether to return entropies
        
        Returns:
            per_token_logps: Log probs for all tokens (B, seq_len-1)
            entropies: Entropies (B, seq_len-1) [if return_entropies=True]
        """ 
        device = model.device
        TS_PLACEHOLDER = "<ts><ts/>" 
        ts_placeholder_id = self.tokenizer.encode(TS_PLACEHOLDER, add_special_tokens=False)[0]
        # print_gpu_memory("_compute_logps_with_ts_injection_for_controller: before function")
        # Tokenize the prompt
        # full_conv_ids = torch.tensor(full_conv_ids, device=device).unsqueeze(0)
        
        ph_mask = (full_conv_ids == ts_placeholder_id)
        num_placeholders = ph_mask.sum().item()
        
        # Get text embeddings
        text_emb = model.model.embed_tokens(full_conv_ids)
        ts_tensor = self.cr_generator.generation_handler._prepare_ts_tensor(ts_data)
        question_emb = self.cr_generator.generation_handler._get_question_embedding(question_ids)
        full_ts_emb = self.cr_generator.generation_handler.model.encode_ts(ts_tensor, question_emb)
        # Find TS_PLACEHOLDER position and inject
        final_emb, final_attn, _ = self.cr_generator.generation_handler.model.inject_ts_segments_and_labels(
            text_embeddings=text_emb,
            placeholder_mask=ph_mask,
            ts_embeddings=[[full_ts_emb]*num_placeholders],  # Batch format
            labels=None
        )
        
        # Forward pass with injected embeddings
        outputs = model.model(inputs_embeds=final_emb,attention_mask=final_attn, return_dict=True)
        logits = model.lm_head(outputs.last_hidden_state)
        
        # Clean up intermediate tensors
        del text_emb, final_emb, outputs

        # Shift for next-token prediction
        logits = logits[:, :-1, :]  # (B, seq_len-1, vocab_size)
        target_ids = full_conv_ids[:, 1:]  # (B, seq_len-1)
        
        # Compute log probs
        logits = logits / self.controller_temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        # print_gpu_memory("_compute_logps_with_ts_injection_for_controller: after function")

        if return_entropies:
            probs = torch.softmax(logits, dim=-1)
            entropies = -(probs * log_probs).sum(dim=-1)
            return per_token_logps, entropies
        else:
            return per_token_logps
        
        
    def _compute_consistency(self, rollout_answers: List[str], gold_answer: str) -> float:
        """
        Compute consistency score: fraction of rollouts that match gold answer
        
        Args:
            rollout_answers: List of predicted answers from rollouts
            gold_answer: Gold/correct answer
        
        Returns:
            Consistency score in [0, 1]
        """
        if not rollout_answers:
            return 0.0
        
        # Normalize gold answer
        gold_normalized = self._normalize_answer(gold_answer)
        
        # Count correct rollouts
        correct_count = 0
        for answer in rollout_answers:
            pred_normalized = self._normalize_answer(answer)
            if pred_normalized == gold_normalized and len(pred_normalized) > 0:
                correct_count += 1
        
        # Return fraction
        consistency = correct_count / len(rollout_answers)
        return float(consistency)

    def _prepare_inputs(self, generation_batch):
        """
        Generate fresh completions each step (like SPELL)
        """
        mode = "train" if self.model.training else "eval"
        
        # Generate fresh completions
        generation_batch = self._generate_and_score_completions(generation_batch)
        
        if self._is_empty_batch(generation_batch):
            print("Empty batch detected - skipping shuffle")
            return generation_batch
        
        if mode == "train":
            # Shuffle to decorrelate
            from trl.trainer.grpo_trainer import shuffle_sequence_dict
            generation_batch["controller"] = shuffle_sequence_dict(generation_batch["controller"])
            generation_batch["reasoner"] = shuffle_sequence_dict(generation_batch["reasoner"])
        
        return generation_batch

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison
        
        Handles:
        - Case insensitivity
        - Punctuation removal
        - Whitespace normalization
        - MCQ letter extraction (A, B, C, D)
        """
        import string
        
        if not answer:
            return ""
        
        answer = str(answer).strip().lower()
        
        # Check if MCQ (single letter)
        mcq_pattern = r'^([a-d])\b'
        mcq_match = re.match(mcq_pattern, answer)
        if mcq_match:
            return mcq_match.group(1)
        
        # Remove punctuation
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        
        # Normalize whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer


    def _calculate_batch_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Batch-level advantage normalization (for controller)
        Like REINFORCE++ / SPELL questioner
        """
        if len(rewards) == 1:
            return torch.zeros_like(rewards)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        if std_reward < 1e-8:
            return torch.zeros_like(rewards)
        
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        return advantages


    def _calculate_group_advantages(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        """
        Group-level advantage normalization (for reasoner)
        Like GRPO / SPELL responder
        
        Formula: A_i = (r_i - mean(r_group)) / std(r_group)
        """
        num_groups = len(rewards) // group_size
        advantages = torch.zeros_like(rewards)
        zero_var_count = 0

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            
            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()
            
            if group_std < 1e-8:
                zero_var_count += 1
                # No variance in group - all get zero advantage
                advantages[start_idx:end_idx] = 0.0
            else:
                advantages[start_idx:end_idx] = (group_rewards - group_mean) / (group_std + 1e-8)
        
        print(f"Zero-variance groups: {zero_var_count}/{num_groups} ({zero_var_count/num_groups*100:.1f}%)")
        return advantages


class ControllerReasonerGRPOTrainerWrapper:
    """
    Wrapper class for easier initialization
    Similar to your QwenTSGRPOTrainer pattern
    """
    
    def __init__(
        self,
        model_path: str,
        lightning_task,
        task: str = "1TS",
        training_stage: str = "mcq",
        w_cot: bool = False,
        use_lora: bool = True,
        use_4bit: bool = False,
        lora_weights_path: str = None,
        output_dir: str = "./controller_reasoner_output",
        lora_cfg: dict = None,
        adapter_type: str = None,
        extra_model_cfg: dict = None,
        reward_config: dict = None,
        max_rounds: int = 5,
        num_rollouts: int = 8,
        first_seg_trials: int = 5,
        include_full_ts_initially: bool = True,
        use_conversation_history: bool = False,
        freeze_ts_encoder: bool = True,
        controller_temperature: float = 0.9,
        reasoner_temperature: float = 0.7,
        num_loops_per_generation: int = 1,
        gradient_accumulation_steps: int = 1,
        extended_prompt: bool = False,
        reasoner_max_new_tokens: int = 512,
        use_uncertainty_prompt: bool = False,
    ):
        self.model_path = model_path
        self.lightning_task = lightning_task
        self.task = task
        self.training_stage = training_stage
        self.w_cot = w_cot
        self.use_lora = use_lora
        self.use_4bit = use_4bit
        self.lora_weights_path = lora_weights_path
        self.output_dir = output_dir
        self.lora_cfg = lora_cfg or {}
        self.adapter_type = adapter_type
        self.extra_model_cfg = extra_model_cfg or {}
        self.reward_config = reward_config or {}
        self.max_rounds = max_rounds
        self.num_rollouts = num_rollouts
        self.include_full_ts_initially = include_full_ts_initially
        self.use_conversation_history = use_conversation_history
        self.freeze_ts_encoder = freeze_ts_encoder
        self.first_seg_trials = first_seg_trials
        self.controller_temperature = controller_temperature
        self.reasoner_temperature = reasoner_temperature
        self.num_loops_per_generation = num_loops_per_generation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.extended_prompt = extended_prompt
        self.reasoner_max_new_tokens = reasoner_max_new_tokens
        self.use_uncertainty_prompt = use_uncertainty_prompt
        # Initialize components
        self._setup_tokenizer()
        self._setup_model()
        self._setup_dataset()
        # self._setup_generation_handler()
        self._setup_rewards()
        self._setup_reference_model()
    
    def _setup_tokenizer(self):
        """Setup tokenizer"""
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _setup_model(self):
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel
        
        def remap_peft_keys(state_dict, model):
            new_state_dict = {}
            model_keys = set(model.state_dict().keys())
            
            for key, value in state_dict.items():
                new_key = key
                
                if key in model_keys:
                    new_state_dict[key] = value
                    continue
                
                # Try adding .base_layer
                if '.base_layer.' not in key:
                    candidate = key.replace('.weight', '.base_layer.weight')
                    if candidate in model_keys:
                        new_key = candidate
                # Try removing .base_layer
                elif '.base_layer.' in key:
                    candidate = key.replace('.base_layer.weight', '.weight')
                    if candidate in model_keys:
                        new_key = candidate
                
                new_state_dict[new_key] = value
            
            return new_state_dict
        
        print(f"Loading model from {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_safetensors=False
        )
        
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Model type: {type(self.model).__name__}")
        
        # Load LoRA/SFT weights if provided
        if self.lora_weights_path:
            # Step 1: Apply LoRA first (creates the layers)
            self._apply_lora()
            
            # Step 2: Load checkpoint weights
            print(f"Loading SFT checkpoint from {self.lora_weights_path}")
            checkpoint = torch.load(self.lora_weights_path, map_location='cpu')
            
            # Handle Lightning checkpoint format
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present (Lightning adds this)
                state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                            for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Remap keys for new PEFT format (.weight -> .base_layer.weight)
            state_dict = remap_peft_keys(state_dict, self.model)
            
            # Try strict=True first (safest if checkpoint matches model exactly)
            try:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
                print(f"\n{'='*60}")
                print(f"✅ Checkpoint loaded successfully with strict=True")
                print(f"  (Perfect match - no missing or unexpected keys)")
                print(f"{'='*60}\n")
            except RuntimeError as e:
                # If strict=True fails, fall back to strict=False with validation
                print(f"\n⚠️  Strict loading failed, falling back to lenient mode...")
                print(f"  Error: {str(e)[:200]}")
                
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                
                # Validate loading results
                print(f"\n{'='*60}")
                print(f"Checkpoint Loading Summary (lenient mode):")
                print(f"  Missing keys: {len(missing)}")
                print(f"  Unexpected keys: {len(unexpected)}")
                
                # Check for critical missing keys (base model weights, not just LoRA)
                critical_missing = [k for k in missing if not any(x in k for x in ['lora', 'peft', 'adapter'])]
                if critical_missing:
                    print(f"\n  ⚠️  WARNING: {len(critical_missing)} critical keys missing (non-LoRA weights):")
                    for key in critical_missing[:10]:
                        print(f"      - {key}")
                    if len(critical_missing) > 10:
                        print(f"      ... and {len(critical_missing) - 10} more")
                
                # Show expected missing keys (LoRA-specific, which is OK if we're initializing new LoRA)
                expected_missing = [k for k in missing if any(x in k for x in ['lora', 'peft', 'adapter'])]
                if expected_missing:
                    print(f"\n  ℹ️  {len(expected_missing)} LoRA/adapter keys missing (expected if initializing new LoRA)")
                
                # Check unexpected keys (might indicate model mismatch)
                if unexpected:
                    print(f"\n  ⚠️  WARNING: {len(unexpected)} unexpected keys found:")
                    for key in unexpected[:10]:
                        print(f"      - {key}")
                    if len(unexpected) > 10:
                        print(f"      ... and {len(unexpected) - 10} more")
                    print(f"  This might indicate a model architecture mismatch!")
                
                # Final validation
                if critical_missing:
                    print(f"\n  ❌ ERROR: Critical weights are missing! Model may not work correctly.")
                    raise ValueError(f"Failed to load {len(critical_missing)} critical weights. Check checkpoint compatibility.")
                elif unexpected and len(unexpected) > 50:  # Many unexpected keys suggests a mismatch
                    print(f"\n  ⚠️  WARNING: Many unexpected keys detected. Proceeding with caution...")
                else:
                    print(f"\n  ✅ Checkpoint loaded successfully (with lenient mode)")
                print(f"{'='*60}\n")
            
        # Freeze TS encoder components
        self.model = self.model.merge_and_unload()
        self._freeze_ts_components()
        
        # Apply LoRA if requested (for training)
        if self.use_lora and not self.lora_weights_path:
            self._apply_lora()
    
    def _load_sft_weights(self):
        """Load SFT weights (LoRA and/or TS encoder)"""
        from peft import PeftModel
        
        print(f"Loading SFT weights from {self.lora_weights_path}")
        
        # Check if it's a .ckpt file (full checkpoint) or a directory (LoRA weights)
        if self.lora_weights_path.endswith('.ckpt'):
            print("Loading full checkpoint...")
            checkpoint = torch.load(self.lora_weights_path, map_location='cpu')
            
            # Extract state dict (handle Lightning checkpoint format)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present (Lightning adds this)
                state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                              for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            # Try strict=True first (safest if checkpoint matches model exactly)
            try:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
                print(f"\n{'='*60}")
                print(f"✅ Checkpoint loaded successfully with strict=True")
                print(f"  (Perfect match - no missing or unexpected keys)")
                print(f"{'='*60}\n")
            except RuntimeError as e:
                # If strict=True fails, fall back to strict=False with validation
                print(f"\n⚠️  Strict loading failed, falling back to lenient mode...")
                print(f"  Error: {str(e)[:200]}")
                
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                
                # Validate loading results
                print(f"\n{'='*60}")
                print(f"Checkpoint Loading Summary (lenient mode):")
                print(f"  Missing keys: {len(missing)}")
                print(f"  Unexpected keys: {len(unexpected)}")
                
                # Check for critical missing keys (base model weights, not just LoRA)
                critical_missing = [k for k in missing if not any(x in k for x in ['lora', 'peft', 'adapter'])]
                if critical_missing:
                    print(f"\n  ⚠️  WARNING: {len(critical_missing)} critical keys missing (non-LoRA weights):")
                    for key in critical_missing[:10]:
                        print(f"      - {key}")
                    if len(critical_missing) > 10:
                        print(f"      ... and {len(critical_missing) - 10} more")
                
                # Show expected missing keys (LoRA-specific, which is OK if we're initializing new LoRA)
                expected_missing = [k for k in missing if any(x in k for x in ['lora', 'peft', 'adapter'])]
                if expected_missing:
                    print(f"\n  ℹ️  {len(expected_missing)} LoRA/adapter keys missing (expected if initializing new LoRA)")
                
                # Check unexpected keys (might indicate model mismatch)
                if unexpected:
                    print(f"\n  ⚠️  WARNING: {len(unexpected)} unexpected keys found:")
                    for key in unexpected[:10]:
                        print(f"      - {key}")
                    if len(unexpected) > 10:
                        print(f"      ... and {len(unexpected) - 10} more")
                    print(f"  This might indicate a model architecture mismatch!")
                
                # Final validation
                if critical_missing:
                    print(f"\n  ❌ ERROR: Critical weights are missing! Model may not work correctly.")
                    raise ValueError(f"Failed to load {len(critical_missing)} critical weights. Check checkpoint compatibility.")
                elif unexpected and len(unexpected) > 50:  # Many unexpected keys suggests a mismatch
                    print(f"\n  ⚠️  WARNING: Many unexpected keys detected. Proceeding with caution...")
                else:
                    print(f"\n  ✅ Checkpoint loaded successfully (with lenient mode)")
                print(f"{'='*60}\n")
        else:
            # It's a LoRA weights directory
            print("Loading LoRA weights from directory...")
            self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path)
            
            # Load TS encoder weights if they exist
            ts_encoder_path = os.path.join(self.lora_weights_path, 'ts_encoder.bin')
            if os.path.exists(ts_encoder_path):
                print(f"Loading TS encoder weights from {ts_encoder_path}")
                ts_encoder_weights = torch.load(ts_encoder_path, map_location='cpu')
                
                # Get base model for loading
                base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
                if hasattr(base_model, 'ts_encoder'):
                    base_model.ts_encoder.load_state_dict(ts_encoder_weights)
                    print("Loaded TS encoder weights")
        
        print("SFT weights loaded successfully")
    
    def _apply_lora(self):
        """Apply LoRA to the model for training"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_r = self.lora_cfg.get('r', 16)
        lora_alpha = self.lora_cfg.get('alpha', 32)
        lora_dropout = self.lora_cfg.get('dropout', 0.1)
        lora_target_modules = self.lora_cfg.get('target_modules', 
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}")
        print(f"Target modules: {lora_target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA applied successfully")
        self._print_trainable_summary()
    
    def _print_trainable_summary(self):
        """Print trainable parameter summary"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print("\n=== TRAINABLE PARAMETERS ===")
        print(f"Trainable: {trainable:,}")
        print(f"Total: {total:,}")
        print(f"Percentage: {100 * trainable / total:.2f}%")
        print("=" * 40)


    def _setup_reference_model(self):
        """
        Setup reference model for KL penalty
        Only created if beta > 0 in training config
        """
        # Reference model will be created in trainer init based on beta
        # We don't create it here to avoid memory overhead if not needed
        pass

    def _freeze_ts_components(self):
        """Freeze TS encoder - LLM remains trainable for full param RL training"""
        frozen_params = 0
        trainable_params = 0
        for param in self.model.parameters():
            param.requires_grad = True
        
        base_model = self.model
        print(f"Model type: {type(base_model).__name__}")
        
        if hasattr(base_model, 'ts_encoder'):
            print("Found ts_encoder - freezing...")
            for name, param in base_model.ts_encoder.named_parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            
            base_model.ts_encoder.eval()
            print(f"  ts_encoder frozen and set to eval mode")
        else:
            print("WARNING: ts_encoder not found!")
        
        # Count all params
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                if 'ts_encoder' not in name:
                    print(f"  WARNING: Non-ts_encoder param frozen: {name}")
        
        total_params = frozen_params + trainable_params
        
        print(f"\n{'='*60}")
        print(f"Full Parameter RL Training Summary:")
        print(f"  Frozen (ts_encoder):       {frozen_params:,} parameters")
        print(f"  Trainable (LLM):           {trainable_params:,} parameters")
        print(f"  Total:                     {total_params:,} parameters")
        if total_params > 0:
            print(f"  Trainable ratio:           {trainable_params/total_params*100:.2f}%")
        print(f"{'='*60}\n")
    
    def _setup_dataset(self):
        """Setup dataset (reuse your existing logic)"""
        from datasets import Dataset
        from multimodal import MultimodalMCQDataset, MultimodalOpenDataset
        from tools_qwents import TS_TOOLS
        TS_PLACEHOLDER = "<ts><ts/>" 
        
        print("Setting up RL dataset")
        
        lightning_data = self.lightning_task.load("train")
        
        # Create Lightning dataset
        mcq_datasets = ["RCW", "TSQA_TF", "TRQA", "TSQA", "TIMERBED_ECG", "TIMERBED_RCW", "ETI", "1TS", "2TS", "TOY_DS_W_MULTIPLE_SEGS", "TOY_DS_EASY", "TOY_DS_MEDIUM", "TOY_DS_HARD", "TOY_DS_MIXED", "ECG_QA_S_VERIFY", "ECG_QA_S_QUERY", "ECG_QA_MIXED","ECG_QA_S_QUERY", "SLEEPQA", "TRQA_MIXED"]
        if self.task in mcq_datasets:
            lightning_dataset = MultimodalMCQDataset(
                lightning_data,
                tokenizer=self.tokenizer,
                context_columns=self.lightning_task.context_columns,
                context_prefix=self.lightning_task.context_prefix,
                ts_column=self.lightning_task.ts_column,
                format_abc_mcq=getattr(self.lightning_task, 'format_abc_mcq', True),
                label_column=self.lightning_task.label_column,
                options_column=getattr(self.lightning_task, 'options_column', 'options'),
                task_name=self.task,
                w_cot=self.w_cot,
                partition="train",
                scale_ts=True,
                shuffle_labels=self.lightning_task.shuffle_labels
            )
        else:
            lightning_dataset = MultimodalOpenDataset(
                lightning_data,
                tokenizer=self.tokenizer,
                context_columns=self.lightning_task.context_columns,
                context_prefix=self.lightning_task.context_prefix,
                ts_column=self.lightning_task.ts_column,
                label_column=self.lightning_task.label_column,
                task_name=self.task,
                partition="train",
                scale_ts=True
            )
        
        # Convert to TRL format
        trl_examples = []
        for i in range(len(lightning_dataset)):
            lightning_example = lightning_dataset[i]
            
            context = lightning_example["context"]
            if self.task == "2TS":
                formatted_context = f"Given two time series (original and modified), {context}"
            else:
                formatted_context = context
            
            options = lightning_example.get("options", [])
            is_tf = len(options) == 2 
            instruction = get_instruction_for_stage(self.training_stage, self.task)
            sys_content = 'you are a helpful assistant that can answer questions about time series data.'
            formatted_context = instruction + "\n" + formatted_context
            
            initial_messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": f"{TS_PLACEHOLDER}\n{formatted_context}"}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                initial_messages, 
                tokenize=False, 
                tools=TS_TOOLS,
                add_generation_prompt=True, 
                enable_thinking=True
            )
            
            trl_example = {
                "prompt": prompt,
                "metadata": {
                    "lightning_example": lightning_example,
                    "correct_answer": lightning_example["label"],
                    "task": self.task,
                    "training_stage": self.training_stage,
                    "timestamps": lightning_example.get("timestamps")
                }
            }
            trl_examples.append(trl_example)
        
        self.train_dataset = Dataset.from_list(trl_examples)
        print(f"Created RL dataset with {len(self.train_dataset)} examples")
    
    def _setup_generation_handler(self):
        """Setup generation handler"""
        from qwents_generation_handler import QwenTSGenerationHandler
        
        self.generation_handler = QwenTSGenerationHandler(
            model=self.model,
            tokenizer=self.tokenizer,
            task=self.task
        )
    
    def _setup_rewards(self):
        """Setup reward functions"""
        # Controller reward
        lambda_penalty = self.reward_config.get("lambda_penalty", 0.1)
        penalty_type = self.reward_config.get("penalty_type", "linear")
        exploration_bonus = self.reward_config.get("exploration_bonus", 0.5)
        exploration_steps = self.reward_config.get("exploration_steps", 300)
        segment_explore_bonus_weight = self.reward_config.get("segment_explore_bonus_weight", 0.0)
        accuracy_weight = self.reward_config.get("accuracy_weight", 3.0)
        format_weight = self.reward_config.get("format_weight", 1.0)
        consistency_weight = self.reward_config.get("consistency_weight", 3.0)
        gentle_reward = self.reward_config.get("gentle_reward", False)
        use_exploration_decay = self.reward_config.get("use_exploration_decay", False)
        uncertainty_reward = self.reward_config.get("uncertainty_reward", 0.0)

        self.controller_reward = ControllerReward(
            lambda_penalty=lambda_penalty,
            penalty_type=penalty_type,
            format_weight=format_weight,    
            consistency_weight=consistency_weight,
            max_trials=self.max_rounds,
            exploration_bonus=exploration_bonus,
            exploration_steps=exploration_steps,
            gentle_reward=gentle_reward,
            segment_explore_bonus_weight=segment_explore_bonus_weight,
            use_exploration_decay=use_exploration_decay
        )
        
        # Reasoner reward
        accuracy_weight = self.reward_config.get("accuracy_weight", 3.0)
        format_weight = self.reward_config.get("format_weight", 1.0)
        
        self.reasoner_reward = ReasonerReward(
            accuracy_weight=accuracy_weight,
            format_weight=format_weight,
            uncertainty_reward=uncertainty_reward
        )
    
    def train(self, **training_args):
        """Run GRPO training"""
        from trl import GRPOConfig
        
        print("Starting Controller-Reasoner GRPO training")
        
        default_config = {
            "output_dir": self.output_dir,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-6,
            "num_train_epochs": 2,
            "max_prompt_length": 2048,
            "logging_steps": 5,
            "save_steps": 200,
            "eval_steps": 200,
            "warmup_steps": 50,
            "gradient_checkpointing": True,
            "bf16": True,
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "group_by_length": False,
            "beta": 0.0,
            "loss_type": "bnpo",  # "bnpo" for group normalization
            "lr_scheduler_type": "linear",  # Default to linear scheduler (can be overridden in config)
            # Class balancing options
            "use_balanced_sampling": False,
            "sampling_strategy": "weighted",  # Options: "weighted", "oversample"
            "balance_alpha": 1.0,  # Power for computing class weights
            # Advantage computation method
            "use_rloo_advantages": False,  # If True, use RLOO advantages; otherwise use regular GRPO advantages
            # Required for custom models when using beta > 0 (KL divergence)
            "model_init_kwargs": {"trust_remote_code": True},
        }
        
        cfg = {**default_config, **training_args}
        
        # Type coercion
        self._coerce_types(cfg)
        
        grpo_config = ControllerReasonerGRPOConfig(**cfg)
        
        # Create trainer
        trainer = ControllerReasonerGRPOTrainer(
            task=self.task,
            model=self.model,
            args=grpo_config,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
            generation_handler=None,
            controller_reward_fn=self.controller_reward,
            reasoner_reward_fn=self.reasoner_reward,
            max_rounds=self.max_rounds,
            num_rollouts=self.num_rollouts,
            first_seg_trials=self.first_seg_trials,
            include_full_ts_initially=self.include_full_ts_initially,
            use_conversation_history=self.use_conversation_history,
            controller_temperature=self.controller_temperature,
            reasoner_temperature=self.reasoner_temperature,
            num_loops_per_generation=self.num_loops_per_generation,
            extended_prompt=self.extended_prompt,
            reasoner_max_new_tokens=self.reasoner_max_new_tokens,
            use_uncertainty_prompt=self.use_uncertainty_prompt,
            lora_weights_path=self.lora_weights_path,  # ← ADD
            lora_cfg=self.lora_cfg,                    
        )
        if 'resume_from_checkpoint' in training_args:
            resume_from_checkpoint = training_args['resume_from_checkpoint']
        else:
            resume_from_checkpoint = None
        
        try:
            print("Starting training...")
            print(f"Training config:")
            print(f"   Batch size: {grpo_config.per_device_train_batch_size}")
            print(f"   Max rounds: {self.max_rounds}")
            print(f"   Num rollouts (G): {self.num_rollouts}")
            print(f"   Include full TS initially: {self.include_full_ts_initially}")
            print(f"   Max grad norm: {grpo_config.max_grad_norm}")
            print(f"   LR scheduler: {grpo_config.lr_scheduler_type}")
            if resume_from_checkpoint:
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                trainer.train()
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Save final model
        try:
            if self.use_lora:
                final_lora_dir = os.path.join(self.output_dir, "final_lora")
                trainer.save_model(final_lora_dir)
                print(f"LoRA weights saved to {final_lora_dir}")
            else:
                final_model_dir = os.path.join(self.output_dir, "final_model")
                trainer.save_model(final_model_dir)
                print(f"Model saved to {final_model_dir}")
            
            print(f"Training completed! Results saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Training completed but saving failed: {e}")

    def _coerce_types(self, cfg: dict):
        """Type coercion for config"""
        ints = ("per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs",
                "logging_steps", "save_steps", "eval_steps", "warmup_steps", "dataloader_num_workers",
                "max_prompt_length", "max_completion_length", "num_generations")
        floats = ("learning_rate", "weight_decay", "beta", "temperature", "controller_temperature", 
                  "reasoner_temperature", "balance_alpha", "max_grad_norm", "warmup_ratio")
        bools = ("bf16", "fp16", "remove_unused_columns", "group_by_length", "gradient_checkpointing",
                 "use_balanced_sampling", "extended_prompt", "use_rloo_advantages")
        
        for k in ints:
            if k in cfg and cfg[k] is not None:
                cfg[k] = int(cfg[k])
        for k in floats:
            if k in cfg and cfg[k] is not None:
                cfg[k] = float(cfg[k])
        for k in bools:
            if k in cfg and isinstance(cfg[k], str):
                cfg[k] = cfg[k].strip().lower() in {"1", "true", "yes", "y", "on"}

        if "resume_from_checkpoint" in cfg:
            val = cfg["resume_from_checkpoint"]
            if isinstance(val, bool):
                # True means auto-detect, False/None means don't resume
                cfg["resume_from_checkpoint"] = val
            elif isinstance(val, str):
                # Check if it's a boolean-like string
                if val.lower() in {"true", "yes", "1", "on"}:
                    cfg["resume_from_checkpoint"] = True
                elif val.lower() in {"false", "no", "0", "off", "none"}:
                    cfg["resume_from_checkpoint"] = None
                else:
                    # It's a path
                    cfg["resume_from_checkpoint"] = str(val)
            else:
                cfg["resume_from_checkpoint"] = None


