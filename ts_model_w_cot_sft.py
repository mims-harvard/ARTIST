"""
SFT Training for Qwen3TS with CoT and Tools
Adapted from qwen3_ts_model_w_cot.py for the new Qwen3TS architecture
Uses <ts><ts/> placeholders with processor-based encoding
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from torch.utils.data import DataLoader
import numpy as np
import json
from typing import List, Dict, Optional
from datetime import datetime
import copy
import re
import random
import json
from datetime import datetime

# Import Qwen3TS components
from bases import MultimodalModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
# Import existing dataset classes
from multimodal import MultimodalMCQDataset, MultimodalOpenDataset
from utils_general import read_jsonl
from ts_model_w_cot_sft import mask_non_assistant, post_process_conversation, get_first_user_message, extract_ts_segments, merge_consecutive_thinks, merge_consecutive_thinks_new
from qwen_utils import (
    extract_answer, 
    extract_answer_letter, 
    extract_answer_letter_from_text,
    parse_tool_segs,
)
import math
import gc
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnAny(StoppingCriteria):
    def __init__(self, tokenizer, stops, initial_length):
        self.tokenizer = tokenizer
        self.stops = stops
        self.initial_length = initial_length  # Length of prompt
        
    def __call__(self, input_ids, scores, **kwargs):
        # Only decode tokens generated AFTER the initial prompt
        if input_ids.shape[1] <= self.initial_length:
            return False
            
        # Decode only the newly generated tokens
        new_tokens = input_ids[0, self.initial_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        for stop in self.stops:
            if stop in text:
                # print(f"STOPPING: Found '{stop}' in generated text")
                return True
        return False

# Tool definition for Qwen3TS
TOOLS = [{
    "type": "function",
    "function": {
        "name": "timeseries_zoom_in_tool",
        "description": "Zoom in on a TS segment [x1, y1].",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_seg": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["ts_seg"]
        }
    }
}]


class Qwen3TSDataModule(pl.LightningDataModule):
    """DataModule for Qwen3TS SFT using existing dataset classes"""
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        task_type: str = "1TS",
        batch_size: int = 4,
        num_workers: int = 4,
        w_cot: bool = False,
        training_stage: str = "mcq",
        tokenizer = None,
        **dataset_kwargs
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.task_type = task_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.w_cot = w_cot
        self.training_stage = training_stage
        self.tokenizer = tokenizer
        self.dataset_kwargs = dataset_kwargs
    
    def _load_data(self, data_path: str):
        """Load data from JSON/JSONL file"""
        return read_jsonl(data_path)
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_data = self._load_data(self.train_data_path)
            
            # Use MultimodalMCQDataset for MCQ tasks
            if self.training_stage == "mcq" or self.training_stage == "reasoning":
                self.train_dataset = MultimodalMCQDataset(
                    train_data,
                    tokenizer=self.tokenizer,
                    task_name=self.task_type,
                    w_cot=self.w_cot,
                    partition="train",
                    **self.dataset_kwargs
                )
            else:  # alignment stage
                self.train_dataset = MultimodalOpenDataset(
                    train_data,
                    tokenizer=self.tokenizer,
                    task_name=self.task_type,
                    w_cot=self.w_cot,
                    partition="train",
                    **self.dataset_kwargs
                )
            
            if self.val_data_path:
                val_data = self._load_data(self.val_data_path)
                
                if self.training_stage == "mcq" or self.training_stage == "reasoning":
                    self.val_dataset = MultimodalMCQDataset(
                        val_data,
                        tokenizer=self.tokenizer,
                        task_name=self.task_type,
                        w_cot=self.w_cot,
                        partition="val",
                        **self.dataset_kwargs
                    )
                else:
                    self.val_dataset = MultimodalOpenDataset(
                        val_data,
                        tokenizer=self.tokenizer,
                        task_name=self.task_type,
                        w_cot=self.w_cot,
                        partition="val",
                        **self.dataset_kwargs
                    )
            else:
                self.val_dataset = None
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._simple_collate
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._simple_collate
        )
    
    def _simple_collate(self, batch):
        """Simple collate that preserves the structure from MultimodalMCQDataset"""
        # Batch is a list of dicts with keys: context, ts, label, cot_in_format, etc.
        collated = {
            'context': [item['context'] for item in batch],
            'ts': [item['ts'] for item in batch],
            'label': [item['label'] for item in batch],
        }
        
        # Add cot_in_format if present
        if 'cot_in_format' in batch[0]:
            collated['cot_in_format'] = [item.get('cot_in_format', None) for item in batch]
        
        return collated


class Qwen3TSLightning(MultimodalModel):
    """Lightning module for Qwen3TS with CoT and tool support"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-3B",
        ts_config: dict = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        batch_size: int = 4,
        w_cot: bool = False,
        training_stage: str = "mcq",
        task: str = "1TS",  # Match CLI's task parameter name
        # LoRA parameters
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list = None,
        freeze_ts_encoder: bool = False,
        lora_weights_path: str = None,
        # Additional parameters for CLI compatibility
        random_segment_selection: bool = False,
        total_ts: bool = False,
        regular_sft: bool = False,
        eval_passk: bool = False,
        num_samples_per_question: int = 1,
        passk: str = "1,2,4,8",
        max_eval_samples: int = -1,
        adapter_type: str = "itformer",
        cot_no_tools: bool = False,
        **kwargs
    ):
        MultimodalModel.__init__(self, **kwargs)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.w_cot = w_cot
        self.training_stage = training_stage
        self.task = task  # Use 'task' to match CLI
        self.freeze_ts_encoder = freeze_ts_encoder
        # LoRA settings
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Additional parameters for CLI compatibility
        self.random_segment_selection = random_segment_selection
        self.total_ts = total_ts
        self.regular_sft = regular_sft
        self.eval_passk = eval_passk
        self.num_samples_per_question = num_samples_per_question
        self.passk = passk
        self.max_eval_samples = max_eval_samples
        self.lora_weights_path = lora_weights_path
        self.cot_no_tools = cot_no_tools
        # Load tokenizer and processor
        print(f"Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.processor = Qwen3TSProcessor(tokenizer=self.tokenizer)
        
        # Load model
        print(f"Loading Qwen3TS model from: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_safetensors=False
        )
        
        # Override TS config if provided
        if ts_config:
            print(f"Overriding TS config with: {ts_config}")
            for key, value in ts_config.items():
                setattr(self.model.config.ts, key, value)
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Apply LoRA if requested
        if self.use_lora:
            self._apply_lora()
        
        # Load LoRA weights if path provided
        if self.lora_weights_path:
            # Check if it's a .ckpt file (full checkpoint) or a directory (LoRA weights)
            if self.lora_weights_path.endswith('.ckpt'):
                print(f"Loading full checkpoint from: {self.lora_weights_path}")
                checkpoint = torch.load(self.lora_weights_path, map_location='cpu')
                self.load_state_dict(checkpoint['state_dict'], strict=False)
                print("Loaded full checkpoint successfully")
            else:
                # It's a LoRA weights directory
                if not self.use_lora:
                    self.use_lora = True
                    self._apply_lora()
                self._load_lora_weights()

        
        # Load custom weights if path provided (and no LoRA weights)
        # if self.custom_weights_path and not self.lora_weights_path:
        #     self._load_custom_weights(self.custom_weights_path)

        # Unfreeze TS encoder (always trainable)
        if not self.freeze_ts_encoder:
            print("TS encoder will be trained")
            for param in self.model.ts_encoder.parameters():
                param.requires_grad = True
        else:
            print("TS encoder is frozen")
        
        self.save_hyperparameters()
        self._print_trainable_summary()
    
    def _print_trainable_summary(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print("\n=== TRAINABLE PARAMETERS ===")
        print(f"Trainable: {trainable:,}")
        print(f"Total: {total:,}")
        print(f"Percentage: {100 * trainable / total:.2f}%")
        
        if self.use_lora:
            lora_params = sum(p.numel() for n, p in self.model.named_parameters() 
                            if 'lora' in n.lower() and p.requires_grad)
            print(f"LoRA parameters: {lora_params:,}")
        
        print("=" * 40)
    
    def _apply_lora(self):
        """Apply LoRA to the language model"""
        print(f"Applying LoRA with r={self.lora_r}, alpha={self.lora_alpha}")
        print(f"Target modules: {self.lora_target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied successfully")
    
    def save_weights(self, output_dir: str):
        """Save LoRA weights and TS encoder weights"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(output_dir)
            print(f"Saved LoRA weights to {output_dir}")
        
        # Get base model (unwrap LoRA if present)
        if hasattr(self.model, 'get_base_model'):
            base_model = self.model.get_base_model()
        else:
            base_model = self.model
        
        # Save TS encoder weights
        if not self.freeze_ts_encoder:
            ts_encoder_path = os.path.join(output_dir, 'ts_encoder.bin')
            torch.save(base_model.ts_encoder.state_dict(), ts_encoder_path)
            print(f"Saved TS encoder weights to {ts_encoder_path}")
        else:
            print("TS encoder is frozen, not saving encoder weights")
    
    def _print_trainable_summary(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print("\n=== TRAINABLE PARAMETERS ===")
        print(f"Trainable: {trainable:,}")
        print(f"Total: {total:,}")
        print(f"Percentage: {100 * trainable / total:.2f}%")
        print("=" * 40)
    
    def _load_lora_weights(self):
        """Load LoRA weights from saved path"""
        if not os.path.exists(self.lora_weights_path):
            raise FileNotFoundError(f"LoRA weights path not found: {self.lora_weights_path}")
        
        print(f"Loading LoRA weights from {self.lora_weights_path}")
        
        if isinstance(self.model, PeftModel):
            base_model = self.model.get_base_model()
        else:
            base_model = self.model
        
        peft_model = PeftModel.from_pretrained(base_model, self.lora_weights_path)
        self.model = peft_model
        
        # Get the base model for loading additional weights
        model_for_loading = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        
        # Load TS encoder weights if they exist
        ts_encoder_path = os.path.join(self.lora_weights_path, 'ts_encoder.bin')
        if os.path.exists(ts_encoder_path):
            ts_encoder_weights = torch.load(ts_encoder_path, map_location='cpu')
            if hasattr(model_for_loading, 'ts_encoder'):
                model_for_loading.ts_encoder.load_state_dict(ts_encoder_weights)
                print("Loaded TS encoder weights")
        
        print("LoRA weights loaded successfully")
    
    def _load_custom_weights(self, checkpoint_dir: str):
        """Load custom components (TS encoder) without LoRA"""
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Custom weights path not found: {checkpoint_dir}")
        
        print(f"Loading custom weights from {checkpoint_dir}")
        
        # For non-LoRA models, use the model directly
        model_for_loading = self.model
        
        # Load TS encoder weights if they exist
        ts_encoder_path = os.path.join(checkpoint_dir, 'ts_encoder.bin')
        if os.path.exists(ts_encoder_path):
            ts_encoder_weights = torch.load(ts_encoder_path, map_location='cpu')
            if hasattr(model_for_loading, 'ts_encoder'):
                model_for_loading.ts_encoder.load_state_dict(ts_encoder_weights)
                print(f"Loaded TS encoder weights")
        
        print("Custom weights loaded successfully")

    def _detect_trailing_padding(self, segment: torch.Tensor) -> int:
        """
        Detect how many consecutive zeros are at the end of the segment (padding).
        
        Args:
            segment: [seg_len] time series segment
            
        Returns:
            num_padding: Number of trailing zeros (padding length)
        """
        seg_len = segment.shape[0]
        
        # Count consecutive zeros from the end
        num_padding = 0
        for i in range(seg_len - 1, -1, -1):
            if segment[i] == 0:
                num_padding += 1
            else:
                break
        
        return num_padding

    def _get_instruction_for_stage(self, training_stage: str) -> str:
        """Get instruction text based on training stage and task type"""
        if training_stage == "mcq":
            if self.task == "ETI":
                return "The following is a multiple-choice question. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
            elif self.task == "1TS":
                return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
            elif self.task == "2TS":
                return "The following is a multiple-choice question about two time series. Please select and provide the correct answer from options available. Only return the correct answer letter."
            else:
                return "The following is a multiple-choice question. Please select and provide the correct answer. Only return the correct answer letter."
        elif training_stage == "alignment":
            return "You are provided with a time series data and a question about it. Please analyze the time series data and provide a detailed answer to the following question."
        elif training_stage == "reasoning":
            if self.task == "1TS":
                return "Please analyze the time series data step by step, explaining your reasoning, then provide your final answer to the following question."
            elif self.task == "2TS":
                return "Please analyze the two time series step by step, explaining your reasoning, then provide your final answer to the following question."
            else:
                return "Please analyze the data step by step, explaining your reasoning, then provide your final answer."
        else:
            raise ValueError(f"Unknown training_stage: {training_stage}")
    
    def create_prompt(self, context: str, ts_length: int) -> str:
        """Create formatted prompt for Qwen3TS"""
        return f"""
You are a time series expert. Analyze ONLY the given time series data and answer the question.

# Output Schema (STRICT)
<think>One–two sentences describing your reasoning and how you came to the answer.</think>
<answer>
[Direct answer ONLY - the first line must be exactly one letter from the set of available options.]
</answer>

# Rules (MANDATORY)
- No text outside <think> and <answer>.
- In <think>, explain your reasoning and reference the time series.
- In <answer>, the first line must be exactly one letter from the set of available options.

# Time Series
### Segment 0: Timesteps [0, {ts_length}]
<ts><ts/>

# Question
{context}
""".strip()
    
    def replace_answer(self, s: str, actual: str) -> str:
        """Replace answer in CoT content"""
        pattern = r'<answer>.*?</answer>'
        if re.search(pattern, s):
            return re.sub(pattern, f'<answer>{actual}</answer>', s, count=1)
        else:
            return s + f'\n<answer>{actual}</answer>'
    
    def _prepare_ts_segments_for_model(self, ts_tensor: torch.Tensor, ts_segs: List) -> torch.Tensor:
        """
        Prepare time series segments with mask channel for model input.
        
        The model expects: [batch_size, max_num_segments, max_seg_len, 2]
        where channel 0 = values, channel 1 = mask (1=valid, 0=padding)
        
        Args:
            ts_tensor: [batch, full_seq_len] original time series
            ts_segs: List of lists with (start, end) tuples per sample
        
        Returns:
            ts_for_model: [batch_size, max_num_segments, max_seg_len, 2] with values and mask
        """
        # print(f"DEBUG _prepare_ts_segments_for_model:")
        # print(f"  ts_tensor.shape: {ts_tensor.shape}")
        # print(f"  len(ts_segs): {len(ts_segs)}")
        # print(f"  ts_segs: {ts_segs}")
        if ts_segs is None or len(ts_segs) == 0:
            return None
        
        batch_size = len(ts_segs)
        
        # First pass: collect all segments and find max dimensions
        all_batch_segments = []
        max_num_segments = 0
        max_seg_len = 0
        
        for batch_idx, segments in enumerate(ts_segs):
            batch_segments = []
            
            if segments is None or len(segments) == 0:
                all_batch_segments.append([])
                print("no segments found")
                continue
            
            if ts_tensor.dim() == 4:
                ts_sample = ts_tensor[batch_idx][0]  # [seq_len]
            else: # dim == 3
                ts_sample = ts_tensor[batch_idx]  # [seq_len]
            for start, end in segments:
                # Extract segment
                segment = ts_sample[start:end]  # [seg_len]
                batch_segments.append(segment)
                max_seg_len = max(max_seg_len, segment.shape[0])
            
            max_num_segments = max(max_num_segments, len(batch_segments))
            all_batch_segments.append(batch_segments)
        
        if max_num_segments == 0 or max_seg_len == 0:
            print("no segments found")
            raise ValueError("no segments found")
        
        # Second pass: create padded tensor
        # Shape: [batch_size, max_num_segments, max_seg_len, 2]
        total_num_segments = sum(len(segs) for segs in all_batch_segments)
        ts_for_model = torch.zeros(
            total_num_segments, 
            max_seg_len, 
            2,  # [values, mask]
            dtype=ts_tensor[0].dtype,
            device=ts_tensor[0].device
        )
        seg_list_padded = []
        mask_list = []
        seg_counter = 0
        for batch_idx, batch_segments in enumerate(all_batch_segments):
            for seg_idx, segment in enumerate(batch_segments):
                seg_len = segment.shape[0]
                if segment.dim() == 2:
                    segment = segment.squeeze(-1)
                
                # Pad segment to max_seg_len
                if seg_len < max_seg_len:
                    # Pad with last value
                    # padding = segment[-1].repeat(max_seg_len - seg_len)
                    padding = torch.zeros(max_seg_len - seg_len, dtype=ts_tensor[0].dtype, device=ts_tensor[0].device)
                    padded_segment = torch.cat([segment, padding])
                else:
                    padded_segment = segment
                
                # Create mask (1 for valid, 0 for padding)
                # num_padding = self._detect_trailing_padding(padded_segment)
                # if num_padding > 0:
                #     seg_len = seg_len - num_padding
                # mask = torch.zeros(max_seg_len, dtype=ts_tensor[0].dtype, device=ts_tensor[0].device)
                # mask[:seg_len] = 1.0
                if self.task in ["TSQA", "TRQA", "ETI"]:
                    mask = torch.zeros(max_seg_len, dtype=ts_tensor[0].dtype, device=ts_tensor[0].device)
                    mask[:seg_len] = 1.0
                else: # you just ran rcw wo it, so remember this in the inference
                    num_padding = self._detect_trailing_padding(padded_segment)
                    if num_padding > 0:
                        seg_len = seg_len - num_padding
                    mask = torch.zeros(max_seg_len, dtype=ts_tensor[0].dtype, device=ts_tensor[0].device)
                    mask[:seg_len] = 1.0
                
                # Fill in the tensor
                ts_for_model[seg_counter, :, 0] = padded_segment  # values
                ts_for_model[seg_counter, :, 1] = mask  # mask
                seg_counter += 1   
        
        return ts_for_model

    def _prepare_ts(self, ts_batch):
        """Prepare time series data for Qwen3TS"""
        if self.task == "2TS":
            # 2TS: stack each sample's two arrays into [2, seq_len, n_vars]
            ts_tensor_pairs = []
            for pair in ts_batch:
                pair_tensors = []
                for t in pair:
                    ts_tensor = torch.from_numpy(t).float() if isinstance(t, np.ndarray) else torch.tensor(t, dtype=torch.float32)
                    if ts_tensor.dim() == 1:
                        ts_tensor = ts_tensor.unsqueeze(-1)
                    pair_tensors.append(ts_tensor)
                ts_tensor_pairs.append(torch.stack(pair_tensors, dim=0))
            ts_batch_tensor = torch.stack(ts_tensor_pairs, dim=0).to(self.device)
        else:
            # 1TS
            ts_tensors = []
            for t in ts_batch:
                ts_tensor = torch.from_numpy(t).float() if isinstance(t, np.ndarray) else torch.tensor(t, dtype=torch.float32)
                if ts_tensor.dim() == 1:
                    ts_tensor = ts_tensor.unsqueeze(-1)
                ts_tensors.append(ts_tensor)
            ts_batch_tensor = torch.stack(ts_tensors, dim=0).to(self.device)
            
        return ts_batch_tensor

    def qwen_preprocess(
        self,
        sources,
        ts_tensor: torch.Tensor,
        tokenizer,
        partition: str = 'train',
        enable_thinking: bool = True,
        regular_sft: bool = False,
    ) -> Dict:

        # System message
        system_message = {
            'role': 'system',
            'content':'you are a helpful assistant that can answer questions about time series data.'
        }
        
        conversations = []
        questions = []
        ts_segments_list = []
        segs_names_list = []
        
        for i, source in enumerate(sources):
            # Build message list
            messages = [system_message]
            
            # Extract segments to get accurate count of valid tool calls
            if not self.cot_no_tools:
                ts_segs, invalid_segments_count, msg_w_tool_idx = extract_ts_segments(source, ts_tensor[i])
            else:
                ts_segs = [[0, len(ts_tensor[i])]]
                invalid_segments_count = 0
                msg_w_tool_idx = {}
            # Add after-tool messages based on actual valid segments
            j = 1 # tool call index starts from 1
            for i, msg in enumerate(source):
                messages.append(msg)
                if not self.cot_no_tools:
                    if i in msg_w_tool_idx.keys():
                        content_str = ""
                        for seg in msg_w_tool_idx[i]:
                            content_str += f"Segment ({seg[0]},{seg[1]}) of time series: <ts><ts/>\n"
                        messages.append({
                            "role": "user",
                            "content": f"{content_str}Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. "
                        })
            
            # Process messages
       
            messages = merge_consecutive_thinks(messages)

            # Step 2: Extract TS segments if training/validation
            if partition != 'test':
                ts_segments_list.append(ts_segs)
            
            # Step 3: Extract question for encoding
            first_user_message = get_first_user_message(messages)
            question = first_user_message.replace("<ts><ts/>", "").strip() #next((m for m in messages if m.get("role") == "user" and TS_PLACEHOLDER in m.get("content", "")), None)
            if question:
                question_ids = tokenizer.encode(question, add_special_tokens=False)
            else:
                question_ids = []
            questions.append(question_ids)
            
            # Step 4: Normalize messages for Qwen format
            # messages = normalize_for_qwen(messages)
            
            # Step 5: Apply chat template
            add_generation_prompt = (partition == 'test')
            if enable_thinking and not self.cot_no_tools:
                conversation = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    tools=TOOLS,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking
                )
            elif regular_sft or self.cot_no_tools:
                conversation = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking
                )
            else:
                conversation = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
            conversation1 = post_process_conversation(conversation)
            conversations.append(conversation1)
        
        # Tokenize all conversations
        tokenized = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        
        # Create labels
        targets = input_ids.clone()
        targets[attention_mask == 0] = -100  # Mask padding
        
        # Mask non-assistant tokens for training
        if partition != 'test':
            targets = mask_non_assistant(conversations, tokenizer, targets)
        
        return {
            "input_ids": input_ids,
            "labels": targets,
            "attention_mask": attention_mask,
            "ts_segs": ts_segments_list if partition != 'test' else None,
            "question_ids": questions,
            "invalid_segments_count": invalid_segments_count,
            "segs_names": segs_names_list if partition != 'test' else None
        }

    def prepare_inputs(self, batch, partition='train'):
        """Prepare inputs for Qwen3TS - matches your prepare_inputs style"""
        contexts = batch['context']
        labels = batch['label']
        ts_data = batch['ts']
        cot_data = batch.get('cot_in_format', [None] * len(contexts))
        
        # Get instruction based on training stage
        instruction = self._get_instruction_for_stage(self.training_stage)
        
        # Format contexts
        if self.task == "2TS":
            formatted_contexts = [f"Given two time series (original and modified), {x}" for x in contexts]
        else:
            formatted_contexts = contexts
        
        # Add instruction
        formatted_contexts = [instruction + "\n" + x for x in formatted_contexts]
        ts_tensor = self._prepare_ts(ts_data)
        
        # Build conversation sources
        if self.w_cot:
            sources = []
            all_timeseries = []
            
            for i, (context, lbl, ts, cot_item) in enumerate(zip(formatted_contexts, labels, ts_data, cot_data)):
                all_conv = []
                
                prompt = self.create_prompt(context, ts_tensor[i].shape[0])
                all_conv.append({"role": "user", "content": prompt})
                
                # Add CoT if available
                if cot_item is not None:
                    cot_copy = copy.deepcopy(cot_item)
                    # Fix answer in last message
                    last_content = cot_copy[-1]['content']
                    correct_content = self.replace_answer(last_content, lbl)
                    cot_copy[-1]['content'] = correct_content
                    all_conv.extend(cot_copy)
                
                sources.append(all_conv)
                # all_timeseries.append(ts_np)
        else:
            sources = []
            all_timeseries = []
            
            for context, lbl, ts in zip(formatted_contexts, labels, ts_data):
                all_conv = []
                
                # Convert ts to numpy
                if isinstance(ts, torch.Tensor):
                    ts_np = ts.cpu().numpy()
                else:
                    ts_np = np.array(ts, dtype=np.float32)
                
                # Simple format
                prompt = f"<ts><ts/>\n{context}"
                all_conv.append({"role": "user", "content": prompt})
                all_conv.append({"role": "assistant", "content": f"<answer>{lbl}</answer>"})
                
                sources.append(all_conv)
                # all_timeseries.append(ts_np)
        
        data_dicts = self.qwen_preprocess(sources, ts_tensor, self.tokenizer, partition=partition, enable_thinking=self.w_cot, regular_sft=self.regular_sft)
        input_ids = data_dicts["input_ids"].to(self.device)
        attention_mask = data_dicts["attention_mask"].to(self.device)
        labels_tensor = data_dicts["labels"].to(self.device)
        ts_segs = data_dicts["ts_segs"]  # Extract the segments from tool calls

        ts_for_model = self._prepare_ts_segments_for_model(ts_tensor, ts_segs)
        if ts_for_model is not None:
            ts_for_model = ts_for_model.to(self.device)
        
        return input_ids, attention_mask, ts_for_model, labels_tensor
    
    def forward(self, batch, partition='train'):
        input_ids, attention_mask, timeseries, labels = self.prepare_inputs(batch, partition)

        outputs = self.model(
            input_ids=input_ids,
            timeseries=timeseries,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss, outputs.logits
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, partition='train')
        self.log('train/loss', loss, on_step=True, on_epoch=True, 
                prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch, partition='val')
        self.log('val/loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def generate(self, context: List[str], label: List[str], ts: List[np.ndarray], cot: List[str] = None, **kwargs):
        """
        Tool-aware generation for Qwen3TS that handles embedding injection for tool calls.
        Adapted from QwenTS generate method.
        """
        device = self.device
        tok = self.tokenizer
        ts_placeholder_id = tok.encode("<ts><ts/>", add_special_tokens=False)[0]
        ts_tensors = self._prepare_ts(ts)
        
        if self.task == "2TS":
            formatted_context = [f"Given two time series (original and modified), {x}" for x in context]
        else:
            formatted_context = context
        
        # Get instruction based on training stage
        instruction = self._get_instruction_for_stage(self.training_stage)
        formatted_context = [instruction + "\n" + x for x in formatted_context]
        
        # System prompt and tools definition
        sys_content = 'you are a helpful assistant that can answer questions about time series data.'
    
        # Generation params
        gen_params = {
            'max_new_tokens': kwargs.get("gen_tokens", 2048),
            'do_sample': True,
            'temperature': kwargs.get("temperature", 0.7),
            'top_p': kwargs.get("top_p", 1.0),
            'eos_token_id': tok.eos_token_id,
            'pad_token_id': tok.pad_token_id
        }

        results = []
        
        # === NON-COT MODE: Simple generation without tools ===
        if not self.w_cot:
            stopper = StopOnAny(tok, ["</answer>", "<|im_end|>"])
            
            for question_text, gold, ts_tensor in zip(formatted_context, label, ts_tensors):
                L = ts_tensor.shape[1]
                
                context_text = self.create_prompt(question_text, L)
                
                # Initialize conversation
                initial_messages = [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": context_text}
                ]
                
                conv = tok.apply_chat_template(initial_messages, tokenize=False, 
                                            add_generation_prompt=True, enable_thinking=False)
                
                initial_tokens = tok(conv, return_tensors="pt", padding=False, truncation=True,
                                    max_length=tok.model_max_length).to(device)
                
                # Prepare timeseries input with full segment
                ts_segs = [[0, L]]
                ts_for_model = self._prepare_ts_segments_for_model(ts_tensor.unsqueeze(0), [ts_segs])
                if ts_for_model is not None:
                    ts_for_model = ts_for_model.to(device)
                
                input_len = initial_tokens.input_ids.size(1) 
                # Single generation call
                out = self.model.generate(
                    input_ids=initial_tokens.input_ids,
                    attention_mask=initial_tokens.attention_mask,
                    timeseries=ts_for_model,
                    stopping_criteria=[stopper] if stopper else None,
                    **gen_params
                )
                new_tokens = out[0, input_len:]
                generated_text = tok.decode(new_tokens, skip_special_tokens=False)
                
                # Extract answer
                answer = extract_answer(generated_text)
                if answer is None:
                    answer = extract_answer_letter_from_text(generated_text) or generated_text.strip()
                
                # MCQ post-processing
                if self.training_stage == "mcq":
                    answer = extract_answer_letter(answer)
                
                results.append({
                    "context": f"<ts><ts/>\n{question_text}",
                    "result": answer,
                    "raw_result": generated_text,
                    "label": gold,
                    "ts": ts_tensor.cpu().numpy()
                })
            
            torch.cuda.empty_cache()
            gc.collect()
            return results

        elif self.cot_no_tools:
            stopper = StopOnAny(tok, ["</answer>", "<|im_end|>"], 0)
            
            for question_text, gold, ts_tensor in zip(formatted_context, label, ts_tensors):
                L = ts_tensor.shape[0] if ts_tensor.dim() == 1 else ts_tensor.shape[1]
                
                context_text = self.create_prompt(question_text, L)
                
                # Initialize conversation WITHOUT tools but WITH thinking enabled
                initial_messages = [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": context_text}
                ]
                
                # Apply template WITHOUT tools but WITH thinking enabled
                conv = tok.apply_chat_template(
                    initial_messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=True  # Enable <think> tags
                    # NOTE: No tools parameter - this is the key difference
                )
                
                initial_tokens = tok(conv, return_tensors="pt", padding=False, truncation=True,
                                    max_length=tok.model_max_length).to(device)
                
                # Prepare timeseries input with full segment
                ts_segs = [[0, L]]
                ts_for_model = self._prepare_ts_segments_for_model(ts_tensor.unsqueeze(0), [ts_segs])
                if ts_for_model is not None:
                    ts_for_model = ts_for_model.to(device)
                
                input_len = initial_tokens.input_ids.size(1)
                stopper.initial_length = input_len
                
                # Single generation call (no tool loop)
                out = self.model.generate(
                    input_ids=initial_tokens.input_ids,
                    attention_mask=initial_tokens.attention_mask,
                    timeseries=ts_for_model,
                    stopping_criteria=StoppingCriteriaList([stopper]),
                    **gen_params
                )
                new_tokens = out[0, input_len:]
                generated_text = tok.decode(new_tokens, skip_special_tokens=False)
                
                # Extract answer
                answer = extract_answer(generated_text)
                if answer is None:
                    answer = extract_answer_letter_from_text(generated_text) or generated_text.strip()
                
                # MCQ post-processing
                if self.training_stage == "mcq":
                    answer = extract_answer_letter(answer)
                
                results.append({
                    "context": f"<ts><ts/>\n{question_text}",
                    "result": answer,
                    "raw_result": generated_text,
                    "label": gold,
                    "ts": ts_tensor.cpu().numpy()
                })
            
            torch.cuda.empty_cache()
            gc.collect()
            return results

        # === COT MODE: Tool-aware generation ===
        max_rounds = kwargs.get("max_rounds", 4)
        enable_thinking = kwargs.get("enable_thinking", True)
        

        for question_text, gold, ts_tensor in zip(formatted_context, label, ts_tensors):
            L = ts_tensor.shape[0]
            
            context_text = self.create_prompt(question_text, L)
            
            # === Initialize conversation ===
            initial_messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": context_text}
            ]
            conv = tok.apply_chat_template(initial_messages, tokenize=False, tools=TOOLS,
                                            add_generation_prompt=True, enable_thinking=enable_thinking)
            
            initial_tokens = tok(conv, return_tensors="pt", padding=False, truncation=True,
                                max_length=tok.model_max_length).to(device)
            
            # Prepare initial timeseries with full segment            
            current_input_ids = initial_tokens.input_ids
            current_attention_mask = initial_tokens.attention_mask
                        
            # === Generation loop ===
            final_text = ""
            answer = None
            generated_segs = set([(0, L)])  # Track segments we've already used
            all_segments = [[0, L]] 

            for round_num in range(max_rounds):
                ts_for_model = self._prepare_ts_segments_for_model(ts_tensor.squeeze(-1).unsqueeze(0), [all_segments])
                if ts_for_model is not None:
                    ts_for_model = ts_for_model.to(device)
                # Generate continuation
                prompt_len = current_input_ids.size(1)
                stopper = StopOnAny(tok, ["</tool_call>", "</answer>", "<|im_end|>"],prompt_len)
                # In your generate method, before calling model.generate:
                # print(f"DEBUG before generation:")
                # print(f"  ts_for_model.shape: {ts_for_model.shape if ts_for_model is not None else None}")
                # print(f"  Number of <ts> tokens in input: {(current_input_ids == ts_placeholder_id).sum()}"
                
                out = self.model.generate(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    timeseries=ts_for_model,
                    stopping_criteria=[stopper] if stopper else None,
                    **gen_params
                )
                
                new_tokens = out[0, prompt_len:]
                new_text = tok.decode(new_tokens, skip_special_tokens=False)
                final_text += new_text
                
                # Check for answer
                if (answer := extract_answer(new_text)) is not None:    #Try on new text
                    break
                
                # Check for tool calls
                tool_segs, _ = parse_tool_segs(new_text)
                if not tool_segs:
                    # No tool calls and no answer - update state and continue
                    current_input_ids = out
                    current_attention_mask = torch.ones_like(out)
                    continue

                
                # === Handle tool calls ===
                # Add generated response to input
                current_input_ids = out
                current_attention_mask = torch.ones_like(out)
                
                # Prepare valid segments from tool calls
                valid_segs = []
                for start, end in tool_segs:
                    start, end = max(0, min(start, L)), max(0, min(end, L))
                    if end > start and (start, end) not in generated_segs:
                        valid_segs.append([start, end])
                        generated_segs.add((start, end))
                        all_segments.append([start, end])
                
                # Format after-tool message
                content_str = ""
                for valid_seg in valid_segs:
                    content_str += f"Segment ({valid_seg[0]},{valid_seg[1]}) of time series: <ts><ts/>\n"
                after_msg = [{"role": "user", "content": f"{content_str}Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. "}]
                after_formatted = tok.apply_chat_template(after_msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                after_tokens = tok(after_formatted, return_tensors="pt", add_special_tokens=False).to(device)
                
                # Concatenate after-tool message
                current_input_ids = torch.cat([current_input_ids, after_tokens.input_ids], dim=1)
                current_attention_mask = torch.cat([current_attention_mask, after_tokens.attention_mask], dim=1)
                
                # Prepare timeseries with new segments
                # if valid_segs:
                #     ts_for_model = self._prepare_ts_segments_for_model(ts_tensor.squeeze(-1).unsqueeze(0), [valid_segs])
                #     if ts_for_model is not None:
                #         ts_for_model = ts_for_model.to(device)
            
            if answer is None:
                answer = extract_answer(final_text)  # Try on full text

            # If still None, try more aggressive extraction
            # if answer is None:
            #     # Try to extract letter from anywhere in the text
            #     answer = extract_answer_letter_from_text(final_text)

            # If STILL None, use the entire generated text as fallback
            if answer is None:
                print(f"WARNING: Could not extract answer, using raw text")
                answer = final_text.strip()

            # MCQ post-processing - now guaranteed to have a string
            if self.training_stage == "mcq" and answer is not None:
                answer = extract_answer_letter(answer)  # Fallback to original if extraction fails
            
            results.append({
                "context": f"<ts><ts/>\n{question_text}",
                "result": answer,
                "raw_result": final_text,
                "label": gold,
                "ts": ts_tensor.cpu().numpy()
            })
        
        torch.cuda.empty_cache()
        gc.collect()
        return results

    def test_step(self, batch, batch_idx):
        self.model.eval()

        # Extract data from batch
        contexts = batch['context']
        labels = batch['label']
        ts_data = batch['ts']

        # Convert to lists if needed
        if not isinstance(contexts, list):
            contexts = [contexts] if isinstance(contexts, str) else contexts.tolist()
        if not isinstance(labels, list):
            labels = [labels] if isinstance(labels, str) else labels.tolist()

        # Convert time series to numpy arrays
        ts_arrays = []
        if not isinstance(ts_data, list):
            ts_data = [ts_data]

        for ts in ts_data:
            if hasattr(ts, 'cpu'):
                ts_arrays.append(ts.cpu().numpy())
            elif hasattr(ts, 'numpy'):
                ts_arrays.append(ts.numpy())
            else:
                ts_arrays.append(np.array(ts))

        # Optional: limit number of evaluated samples globally (runtime saver)
        max_eval = getattr(self, "max_eval_samples", None)
        if max_eval != -1:
            seen = getattr(self, "_eval_seen", 0)
            remaining = max_eval - seen
            if remaining <= 0:
                return

            contexts = contexts[:remaining]
            labels = labels[:remaining]
            ts_arrays = ts_arrays[:remaining]
            setattr(self, "_eval_seen", seen + len(contexts))

        # Local helper: unbiased pass@k estimator from n samples with c correct
        import math
        def _pass_at_k(n: int, c: int, k: int) -> float:
            if c <= 0:
                return 0.0
            if n - c < k:
                return 1.0
            return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

        def _parse_k_values(passk_str: str, n: int):
            ks = []
            for x in str(passk_str).split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    ks.append(int(x))
                except ValueError:
                    continue
            ks = sorted(set(k for k in ks if 1 <= k <= n))
            return ks if ks else [1]

        def _set_seed(seed: int):
            """Set seed for reproducibility"""
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        try:
            # ===== DEFAULT: single-sample evaluation (accuracy, f1, etc.) =====
            if not getattr(self, "eval_passk", False):
                raw_results = self.generate(contexts, labels, ts_arrays)

                processed_results = []
                for result in raw_results:
                    extracted_answer = result['result']

                    processed_result = {
                        'context': result['context'],
                        'result': "" if extracted_answer is None else str(extracted_answer),
                        'raw_result': result['raw_result'],
                        'label': "" if result['label'] is None else str(result['label']),
                        'ts': result['ts'].cpu().numpy() if hasattr(result['ts'], "cpu") else result['ts'],
                    }
                    processed_results.append(processed_result)

                # Calculate batch accuracy
                correct = 0
                for r in processed_results:
                    if r['result'].strip().lower() == r['label'].strip().lower():
                        correct += 1
                batch_accuracy = correct / len(processed_results) if processed_results else 0.0

                # Log batch metrics
                batch_size = len(processed_results)
                self.log('test_accuracy', batch_accuracy, on_step=True, on_epoch=True,
                        prog_bar=True, batch_size=batch_size)

                result = {
                    'predictions': processed_results,
                    'batch_idx': batch_idx,
                    'accuracy': batch_accuracy,
                    'attention_data': None,
                    'mode': 'single'
                }
                if not hasattr(self, '_test_step_outputs'):
                    self._test_step_outputs = []
                self._test_step_outputs.append(result)
                return

            # ===== PASS@K EVALUATION (explicit opt-in only) =====
            n = int(getattr(self, "num_samples_per_question", 1))
            n = max(1, n)
            k_values = _parse_k_values(getattr(self, "passk", "1"), n)
            
            # Base seed for reproducibility (can be set as a hyperparameter)
            base_seed = getattr(self, "eval_seed", 42)

            passk_records = []
            # For each example, collect n independent generations
            for q_idx, (ctx, gold, ts_arr) in enumerate(zip(contexts, labels, ts_arrays)):
                samples = []
                correct_flags = []
                seeds_used = []

                for sample_idx in range(n):
                    # Compute seed for this specific (question, sample) combination
                    # Same question + same sample_idx = same seed (reproducible)
                    # Different questions get different seeds (diversity across questions)
                    # Different samples of same question get different seeds (diversity for pass@k)
                    seed = base_seed + 2020 + sample_idx
                    _set_seed(seed)
                    seeds_used.append(seed)
                    
                    one = self.generate([ctx], [gold], [ts_arr])[0]
                    pred = one.get("result", None)
                    pred = "" if pred is None else str(pred).strip()

                    samples.append({
                        "pred": pred,
                        "raw_result": one.get("raw_result", ""),
                        "seed": seed,
                    })
                    correct_flags.append(pred.strip().lower() == str(gold).strip().lower())

                c = int(sum(correct_flags))
                q_rec = {
                    "question": ctx,
                    "gold_answer": gold,
                    "n": n,
                    "k_values": k_values,
                    "num_correct": c,
                    "samples": samples,
                    "correct_flags": correct_flags,
                    "seeds": seeds_used,
                    "accuracy_at_1": float(correct_flags[0]) if len(correct_flags) > 0 else 0.0,
                    "any_correct": bool(c > 0),
                    "passk": {f"pass@{k}": _pass_at_k(n, c, k) for k in k_values},
                }
                passk_records.append(q_rec)

            # Aggregate per-batch pass@k means (optional but useful for progress bar)
            batch_passk_means = {}
            for k in k_values:
                key = f"pass@{k}"
                vals = [q["passk"][key] for q in passk_records]
                batch_passk_means[key] = float(sum(vals) / max(1, len(vals)))

            # Also track accuracy@1 for sanity
            acc1_vals = [q["accuracy_at_1"] for q in passk_records]
            batch_acc1 = float(sum(acc1_vals) / max(1, len(acc1_vals)))

            # Log batch metrics
            batch_size = len(passk_records)
            self.log('test/accuracy@1', batch_acc1, on_step=True, on_epoch=True,
                    prog_bar=True, batch_size=batch_size)
            for k in k_values:
                key = f"pass@{k}"
                self.log(f'test/{key}', batch_passk_means[key], on_step=True, on_epoch=True,
                        prog_bar=(k == min(k_values)), batch_size=batch_size)

            result = {
                'passk_records': passk_records,
                'batch_idx': batch_idx,
                'mode': 'passk',
                'k_values': k_values,
                'n': n,
                'base_seed': base_seed,
                'batch_passk_means': batch_passk_means,
                'batch_acc1': batch_acc1,
            }
            if not hasattr(self, '_test_step_outputs'):
                self._test_step_outputs = []
            self._test_step_outputs.append(result)

        except Exception as e:
            print(f"Error in test_step batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'error': str(e),
                'batch_idx': batch_idx,
                'predictions': [],
                'mode': 'error'
            }
            if not hasattr(self, '_test_step_outputs'):
                self._test_step_outputs = []

            self._test_step_outputs.append(result)

    def on_test_epoch_end(self):
        """
        Called by Lightning after all test batches are processed.
        """
        outputs = getattr(self, "_test_step_outputs", [])
        print(f"\nTest epoch completed. Processing {len(outputs)} batch outputs...")

        # ---------- output directory ----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_name = getattr(self, "encoder_name", "unknown")
        default_root = getattr(self.trainer, "default_root_dir", "experiments")
        base_dir = os.path.join(default_root, f"test_results_{encoder_name}")
        output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        errors = []

        # ===================== STANDARD (NON pass@k) ==========================
        if not getattr(self, "eval_passk", False):
            all_predictions = []
            total_correct = 0
            total_samples = 0

            for batch_output in outputs:
                if "error" in batch_output:
                    errors.append(f"Batch {batch_output['batch_idx']}: {batch_output['error']}")
                    continue

                preds = batch_output.get("predictions", [])
                all_predictions.extend(preds)

                for p in preds:
                    total_samples += 1
                    if p["result"].strip().lower() == p["label"].strip().lower():
                        total_correct += 1

            final_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            print("\nFINAL TEST RESULTS (single-sample)")
            print(f"  Total samples: {total_samples}")
            print(f"  Accuracy: {final_accuracy:.4f}")
            print(f"  Errors: {len(errors)}")

            self.log("test/accuracy_final", final_accuracy)
            self.log("test/num_samples", total_samples)

            with open(os.path.join(output_dir, "predictions.json"), "w") as f:
                json.dump(all_predictions, f, indent=2, default=str)

            with open(os.path.join(output_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        "accuracy": final_accuracy,
                        "num_samples": total_samples,
                        "errors": errors,
                        "timestamp": timestamp,
                    },
                    f,
                    indent=2,
                )

            print(f"\nSaved results to {output_dir}")
            self._test_step_outputs = []
            return

        # =========================== PASS@K ===================================

        all_records = []
        base_seed = getattr(self, "eval_seed", 42)
        
        for batch_output in outputs:
            if "error" in batch_output:
                errors.append(f"Batch {batch_output['batch_idx']}: {batch_output['error']}")
                continue
            all_records.extend(batch_output.get("passk_records", []))

        if len(all_records) == 0:
            print("No pass@k records collected.")
            self._test_step_outputs = []
            return

        # Aggregate pass@k (mean over questions)
        k_values = all_records[0]["k_values"]
        n = all_records[0]["n"]
        
        mean_passk = {}
        for k in k_values:
            key = f"pass@{k}"
            mean_passk[key] = float(
                sum(r["passk"][key] for r in all_records) / len(all_records)
            )

        # Empirical accuracy@1 (first sample only)
        acc1 = float(
            sum(r["accuracy_at_1"] for r in all_records) / len(all_records)
        )

        any_correct_rate = float(
            sum(1.0 if r["any_correct"] else 0.0 for r in all_records) / len(all_records)
        )

        # ============== Aggregate metrics over all n samples ==============
        all_preds = []
        all_labels = []
        all_correct_flags = []
        all_seeds = []
        
        # Per-seed tracking
        per_seed_results = {i: {"correct": 0, "total": 0, "preds": [], "labels": []} for i in range(n)}
        
        for record in all_records:
            gold = record["gold_answer"]
            for sample_idx, sample in enumerate(record["samples"]):
                pred = sample["pred"]
                seed = sample.get("seed", None)
                is_correct = record["correct_flags"][sample_idx]
                
                all_preds.append(pred.strip().lower())
                all_labels.append(str(gold).strip().lower())
                all_correct_flags.append(is_correct)
                all_seeds.append(seed)
                
                # Track per-seed (sample index) results
                per_seed_results[sample_idx]["total"] += 1
                per_seed_results[sample_idx]["preds"].append(pred.strip().lower())
                per_seed_results[sample_idx]["labels"].append(str(gold).strip().lower())
                if is_correct:
                    per_seed_results[sample_idx]["correct"] += 1
        
        # Average accuracy over all n samples
        avg_accuracy_all_samples = float(sum(all_correct_flags)) / len(all_correct_flags) if all_correct_flags else 0.0
        
        # Calculate per-sample (per-seed) metrics
        per_sample_metrics = {}
        try:
            from sklearn.metrics import precision_recall_fscore_support
            
            for sample_idx in range(n):
                sample_data = per_seed_results[sample_idx]
                if sample_data["total"] == 0:
                    continue
                    
                accuracy = sample_data["correct"] / sample_data["total"]
                
                # Calculate F1, precision, recall for this sample/seed
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    sample_data["labels"], sample_data["preds"], average='macro', zero_division=0
                )
                
                # Get the actual seed used for the first question at this sample index
                sample_seed = None
                for record in all_records:
                    if sample_idx < len(record.get("seeds", [])):
                        # Extract just the sample offset part of the seed
                        sample_seed = record["seeds"][sample_idx]
                        break
                
                per_sample_metrics[f"sample_{sample_idx + 1}"] = {
                    "seed_offset": sample_idx,  # The offset from base_seed
                    "example_seed": sample_seed,  # An example actual seed used
                    "accuracy": float(accuracy),
                    "macro_precision": float(precision_macro),
                    "macro_recall": float(recall_macro),
                    "macro_f1": float(f1_macro),
                    "total": sample_data["total"],
                    "correct": sample_data["correct"],
                }
        except ImportError:
            # Fallback without sklearn
            for sample_idx in range(n):
                sample_data = per_seed_results[sample_idx]
                if sample_data["total"] == 0:
                    continue
                accuracy = sample_data["correct"] / sample_data["total"]
                per_sample_metrics[f"sample_{sample_idx + 1}"] = {
                    "seed_offset": sample_idx,
                    "accuracy": float(accuracy),
                    "total": sample_data["total"],
                    "correct": sample_data["correct"],
                }

        # Calculate aggregate metrics using sklearn
        try:
            from sklearn.metrics import precision_recall_fscore_support
            
            unique_labels = sorted(set(all_labels))
            
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro', zero_division=0
            )
            
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                all_labels, all_preds, average=None, labels=unique_labels, zero_division=0
            )
            
            per_class_metrics = {}
            for idx, label in enumerate(unique_labels):
                per_class_metrics[label] = {
                    "precision": float(precision_per_class[idx]),
                    "recall": float(recall_per_class[idx]),
                    "f1": float(f1_per_class[idx]),
                    "support": int(support_per_class[idx])
                }
            
            aggregate_metrics = {
                "avg_accuracy_all_samples": avg_accuracy_all_samples,
                "macro_precision": float(precision_macro),
                "macro_recall": float(recall_macro),
                "macro_f1": float(f1_macro),
                "weighted_precision": float(precision_weighted),
                "weighted_recall": float(recall_weighted),
                "weighted_f1": float(f1_weighted),
                "micro_precision": float(precision_micro),
                "micro_recall": float(recall_micro),
                "micro_f1": float(f1_micro),
                "per_class": per_class_metrics,
                "per_sample": per_sample_metrics,
                "total_predictions": len(all_preds),
                "num_questions": len(all_records),
                "num_samples_per_question": n,
                "base_seed": base_seed,
            }
            
        except ImportError:
            print("Warning: sklearn not available, skipping F1/precision/recall calculation")
            aggregate_metrics = {
                "avg_accuracy_all_samples": avg_accuracy_all_samples,
                "per_sample": per_sample_metrics,
                "total_predictions": len(all_preds),
                "num_questions": len(all_records),
                "num_samples_per_question": n,
                "base_seed": base_seed,
            }

        # ============== Print results ==============
        print("\n" + "="*70)
        print("FINAL PASS@K RESULTS")
        print("="*70)
        print(f"  Questions evaluated: {len(all_records)}")
        print(f"  n (samples/question): {n}")
        print(f"  Total predictions: {len(all_preds)}")
        print(f"  Base seed: {base_seed}")
        print("-"*70)
        print("  Pass@K Metrics:")
        print(f"    Accuracy@1 (first sample): {acc1:.4f}")
        for k in k_values:
            print(f"    Pass@{k}: {mean_passk[f'pass@{k}']:.4f}")
        print(f"    Any-correct rate: {any_correct_rate:.4f}")
        print("-"*70)
        print("  Aggregate Metrics (over all samples):")
        print(f"    Average Accuracy: {avg_accuracy_all_samples:.4f}")
        if "macro_f1" in aggregate_metrics:
            print(f"    Macro Precision: {aggregate_metrics['macro_precision']:.4f}")
            print(f"    Macro Recall: {aggregate_metrics['macro_recall']:.4f}")
            print(f"    Macro F1: {aggregate_metrics['macro_f1']:.4f}")
            print(f"    Weighted F1: {aggregate_metrics['weighted_f1']:.4f}")
        print("-"*70)
        print("  Per-Sample Metrics (by seed offset):")
        print(f"  {'Sample':<10} {'Seed Offset':<12} {'Accuracy':<10} {'Macro P':<10} {'Macro R':<10} {'Macro F1':<10}")
        print(f"  {'-'*62}")
        for sample_name, metrics in sorted(per_sample_metrics.items()):
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('macro_precision', 'N/A')
            rec = metrics.get('macro_recall', 'N/A')
            f1 = metrics.get('macro_f1', 'N/A')
            seed_off = metrics.get('seed_offset', 'N/A')
            
            prec_str = f"{prec:.4f}" if isinstance(prec, float) else prec
            rec_str = f"{rec:.4f}" if isinstance(rec, float) else rec
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
            
            print(f"  {sample_name:<10} {seed_off:<12} {acc:<10.4f} {prec_str:<10} {rec_str:<10} {f1_str:<10}")
        print(f"  Errors: {len(errors)}")
        print("="*70)

        # ============== Log to Lightning ==============
        self.log("test/accuracy@1", acc1)
        self.log("test/any_correct_rate", any_correct_rate)
        self.log("test/avg_accuracy_all_samples", avg_accuracy_all_samples)
        
        for k in k_values:
            self.log(f"test/pass@{k}", mean_passk[f"pass@{k}"])
        
        if "macro_f1" in aggregate_metrics:
            self.log("test/macro_precision", aggregate_metrics["macro_precision"])
            self.log("test/macro_recall", aggregate_metrics["macro_recall"])
            self.log("test/macro_f1", aggregate_metrics["macro_f1"])
            self.log("test/weighted_f1", aggregate_metrics["weighted_f1"])
        
        # Log per-sample metrics
        for sample_name, metrics in per_sample_metrics.items():
            self.log(f"test/{sample_name}_accuracy", metrics["accuracy"])
            if "macro_f1" in metrics:
                self.log(f"test/{sample_name}_macro_f1", metrics["macro_f1"])

        # ============== Save results ==============
        with open(os.path.join(output_dir, "passk_records.json"), "w") as f:
            json.dump(all_records, f, indent=2, default=str)

        summary = {
            "num_questions": len(all_records),
            "n": n,
            "k_values": k_values,
            "base_seed": base_seed,
            "accuracy_at_1": acc1,
            "any_correct_rate": any_correct_rate,
            "passk": mean_passk,
            "aggregate_metrics": aggregate_metrics,
            "errors": errors,
            "timestamp": timestamp,
        }
        
        with open(os.path.join(output_dir, "passk_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved pass@k results to {output_dir}")
        self._test_step_outputs = []
        return


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training for Qwen3TS")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="1TS")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B",
                        help="Path to Qwen3TS model directory or HuggingFace model ID")
    parser.add_argument("--w_cot", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./qwen3ts_experiments")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--training_stage", type=str, default="mcq")
    parser.add_argument("--eval_passk", action="store_true")
    parser.add_argument("--num_samples_per_question", type=int, default=1)
    parser.add_argument("--passk", type=str, default="1,2,4,8")
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    
    args = parser.parse_args()
    
    pl.seed_everything(42)
    
    print("=" * 80)
    print("Qwen3TS SFT Training with CoT and Tools")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Task: {args.task_type}")
    print(f"CoT: {args.w_cot}")
    print(f"Stage: {args.training_stage}")
    print("=" * 80)
    
    # Initialize model
    model = Qwen3TSLightning(
        model_path=args.model_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        w_cot=args.w_cot,
        training_stage=args.training_stage,
        task_type=args.task_type,  # Pass task_type
        lora_weights_path=args.lora_weights_path,
    )
    
    # Initialize data
    data_module = Qwen3TSDataModule(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        task_type=args.task_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        w_cot=args.w_cot,
        training_stage=args.training_stage,
        tokenizer=model.tokenizer  # Pass tokenizer from model
    )
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{timestamp}")
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(exp_dir, "checkpoints"),
            filename="{epoch}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(monitor="val/loss", patience=3, mode="min"),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup logger
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project="qwen3ts-sft",
            name=args.experiment_name,
            save_dir=exp_dir
        )
    else:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=exp_dir, name="logs")
    
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=exp_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, data_module)
    
    print(f"\n✅ Training complete! Results saved to {exp_dir}")


if __name__ == "__main__":
    main()