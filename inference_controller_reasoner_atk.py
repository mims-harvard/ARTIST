#!/usr/bin/env python3
"""
Controller-Reasoner Inference Script

CRITICAL: This script imports directly from the training code to ensure
data processing is identical between training and inference.

Usage:
    python controller_reasoner_inference.py \
        --model_path /path/to/model \
        --task TSQA \
        --dataset_config config.yaml \
        --output_file predictions.json
        
Make sure the training code directory is in your PYTHONPATH or run from the training code directory.
"""

import os
import sys
import torch
import argparse
import yaml
import json
import re
import string
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
from math import comb

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# ============================================================================
# IMPORTS FROM TRAINING CODE - These ensure identical data processing
# ============================================================================
from segment_memory import SegmentMemory, SegmentMemoryManager
from train_controller_reasoner_opt import _import_class
from qwents_utils import get_instruction_for_stage
from controller_reasoner_generation_opt import (
    # Stopping criteria classes
    ControllerStoppingCriteria,
    ReasonerStoppingCriteria,
    # Constants
    TS_PLACEHOLDER,
    TS_TOOLS,
    ControllerReasonerGenerator,
    # # TS preparation functions
    # _prepare_ts_segments_for_model,
    # _detect_trailing_padding,
    # # Prompt creation functions
    # _create_first_turn_controller_prompt,
    # _create_controller_prompt,
    # _create_controller_prompt_full_ts,
    # _create_reasoner_prompt,
    # # Parsing functions
    # _parse_controller_decision,
    # _parse_reasoner_response,
)


# ============================================================================
# PASS@K UTILITY
# ============================================================================
def pass_at_k_spell(n: int, c: int, k: int) -> float:
    """SPELL / Codex unbiased pass@k estimator."""
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ============================================================================
# CONTROLLER -REASONER INFERENCE
# ============================================================================
class ControllerReasonerInference:
    """
    Inference handler for Controller-Reasoner models.
    
    Uses functions imported directly from training code to ensure identical data processing.
    """
    
    def __init__(
        self,
        model_path: str,
        lora_weights_path: str = None,
        task: str = "1TS",
        training_stage: str = "mcq",
        max_rounds: int = 5,
        first_seg_trials: int = 5,
        controller_temperature: float = 0.3,
        reasoner_temperature: float = 0.3,
        top_p: float = 0.9,
        max_format_error_retries: int = 3,
        include_full_ts_initially: bool = False,
        reasoner_max_new_tokens: int = 768,
        device: str = "cuda",
        use_flash_attention: bool = True,
        lora_cfg: dict = None,
    ):
        self.model_path = model_path
        self.lora_weights_path = lora_weights_path
        self.task = task
        self.training_stage = training_stage
        self.max_rounds = max_rounds
        self.first_seg_trials = first_seg_trials
        self.controller_temperature = controller_temperature
        self.reasoner_temperature = reasoner_temperature
        self.top_p = top_p
        self.max_format_error_retries = max_format_error_retries
        self.include_full_ts_initially = include_full_ts_initially
        self.reasoner_max_new_tokens = reasoner_max_new_tokens
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_flash_attention = use_flash_attention
        self.lora_cfg = lora_cfg or {}
        
            
        print(f"{'='*60}")
        print("Controller-Reasoner Inference")
        print(f"{'='*60}")
        print(f"Model path: {model_path}")
        print(f"Task: {task}")
        print(f"Max rounds: {max_rounds}")
        print(f"Controller temperature: {controller_temperature}")
        print(f"Reasoner temperature: {reasoner_temperature}")
        print(f"Include full TS initially: {include_full_ts_initially}")
        print(f"{'='*60}\n")
        
        self._setup_tokenizer()
        self._setup_model()
        self.cr_generator = ControllerReasonerGenerator(
        model=None,
        tokenizer=self.tokenizer,
        generation_handler=None,
        max_rounds=self.max_rounds,
        first_seg_trials=self.first_seg_trials,
        task_name=self.task,
        include_full_ts_initially=self.include_full_ts_initially,
        use_conversation_history=False,
        controller_temperature=self.controller_temperature,
        reasoner_temperature=self.reasoner_temperature,
        extended_prompt=False,
        reasoner_max_new_tokens=self.reasoner_max_new_tokens,
        use_uncertainty_prompt=False
        )
    
    def _setup_tokenizer(self):
        """Load tokenizer - must match training."""
        print(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _setup_model(self):
        """Load model - must match training exactly."""
        print(f"Loading model from {self.model_path}")
        
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        
        if self.lora_weights_path and os.path.exists(self.lora_weights_path):
            print(f"Loading weights from {self.lora_weights_path}")
            self._load_weights()
        
        self.model.eval()
        print(f"Model loaded: {type(self.model).__name__}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
    
    def _load_weights(self):
        """Load trained weights (checkpoint or LoRA)."""
        if self.lora_weights_path.endswith('.ckpt'):
            print("Loading from Lightning checkpoint...")
            checkpoint = torch.load(self.lora_weights_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                              for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
            has_lora = any('lora' in k.lower() for k in state_dict.keys())
            
            if has_lora:
                print("Checkpoint contains LoRA weights, applying LoRA config...")
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.lora_cfg.get('r', 8),
                    lora_alpha=self.lora_cfg.get('alpha', 16),
                    lora_dropout=self.lora_cfg.get('dropout', 0.02),
                    target_modules=self.lora_cfg.get('target_modules', 
                        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            if has_lora:
                print("Merging LoRA weights...")
                self.model = self.model.merge_and_unload()
        
        elif os.path.exists(os.path.join(self.lora_weights_path, "adapter_config.json")):
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(self.model, self.lora_weights_path, is_trainable=False)
            print("Merging LoRA weights...")
            self.model = self.model.merge_and_unload()
        
        else:
            print(f"Warning: Could not determine weight format at {self.lora_weights_path}")

    # ========================================================================
    # TIME SERIES PREPARATION - Uses imported training functions
    # ========================================================================
    
    def _prepare_ts_tensor(self, ts_data) -> torch.Tensor:
        """Convert time series data to tensor."""
        if isinstance(ts_data, np.ndarray):
            ts_tensor = torch.from_numpy(ts_data).float()
        elif isinstance(ts_data, list):
            ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
        elif isinstance(ts_data, torch.Tensor):
            ts_tensor = ts_data.float()
        else:
            raise TypeError(f"Unsupported ts_data type: {type(ts_data)}")
        
        if ts_tensor.dim() == 1:
            ts_tensor = ts_tensor.unsqueeze(-1)
        
        return ts_tensor

    # ========================================================================
    # GENERATION - Uses imported training functions
    # ========================================================================
    
    def _generate(
        self,
        prompt: str,
        ts_tensor: torch.Tensor,
        segments: List[List[int]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stopping_criteria: StoppingCriteriaList,
        do_sample: bool = True,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        """generation - uses imported _prepare_ts_segments_for_model from training."""
        with torch.no_grad():
            toks = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            input_ids = toks.input_ids
            attn_mask = toks.attention_mask
            initial_length = input_ids.shape[1]
            
            ts_len = ts_tensor.shape[0]
            segs_for_model = [[0, ts_len]] if segments is None else segments
                        
            # Use imported function from training
            ts_for_model = self.cr_generator._prepare_ts_segments_for_model(ts_tensor.unsqueeze(0), [segs_for_model])
            
            if ts_for_model is not None:
                # if self.task == 'TSQA':
                ts_for_model = ts_for_model.to(self.model.device, dtype=torch.bfloat16)
                # else:
                #     ts_for_model = ts_for_model.to(self.model.device)
            
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                timeseries=ts_for_model,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            new_tokens = gen_ids[0, initial_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return text, input_ids.cpu(), new_tokens.cpu(), initial_length

    # ========================================================================
    # ANSWER NORMALIZATION
    # ========================================================================
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        
        answer = answer.strip().lower()
        mcq_match = re.match(r'^([a-d])\b', answer)
        if mcq_match:
            return mcq_match.group(1)
        
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer

    # ========================================================================
    # MAIN INFERENCE LOOP - Uses imported functions from training
    # ========================================================================
    
    def generate_controller_reasoner_loop(
        self, 
        question: str, 
        ts_data, 
        gold_answer: str = "",
        raw_ts=None, 
        timestamps=None, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Run the Controller-Reasoner loop using imported training functions."""
        ts_tensor = ts_data #.unsqueeze(0) if ts_data.dim() == 2 else ts_data
        ts_length = ts_tensor.shape[0]
        
        # Use imported SegmentMemory from training
        segment_memory = SegmentMemory(ts_length)
        
        controller_completions = []
        reasoner_completions = []
        reasoner_answers = []
        controller_prompts = []
        reasoner_prompts = []
        full_conv_for_controller = []
        
        previous_reasoner_answer = None
        format_error_occurred = False
        format_error_round = None
        decision = {"decision": "error"}
        
        # Main controller-reasoner loop
        for round_num in range(self.max_rounds):
            # Build controller prompt using imported functions from training
            if round_num == 0 and not self.include_full_ts_initially:
                controller_prompt = self.cr_generator._create_first_turn_controller_prompt(
                    question=question, segment_memory=segment_memory, ts=np.array(raw_ts),
                )
            elif round_num == 0 and self.include_full_ts_initially:
                controller_prompt = self.cr_generator._create_controller_prompt_full_ts(
                    question=question, segment_memory=segment_memory, current_segments=segment_memory.get_all_segments(), previous_reasoner_answer=previous_reasoner_answer
                )
            else:
                controller_prompt = self.cr_generator._create_controller_prompt(
                    question=question, current_segments=segment_memory.get_all_segments(), segment_memory=segment_memory, previous_reasoner_answer=previous_reasoner_answer,
                    ts=np.array(raw_ts),
                )
            
            sys_content = 'you are a helpful assistant that can answer questions about time series data.'
            controller_messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": controller_prompt}
            ]
            
            controller_prompt_formatted = self.tokenizer.apply_chat_template(
                controller_messages, tokenize=False, tools=TS_TOOLS,
                add_generation_prompt=True, enable_thinking=True,
            )
            controller_prompts.append(controller_prompt_formatted)
            full_conv_for_controller.append(controller_messages)
            
            toks_tmp = self.tokenizer(controller_prompt_formatted, return_tensors="pt", padding=True, padding_side="left")
            initial_length = toks_tmp.input_ids.shape[1]
            
            # Use imported ControllerStoppingCriteria from training
            stop_controller = StoppingCriteriaList([
                ControllerStoppingCriteria(self.tokenizer, initial_length=initial_length)
            ])
            
            # Round 0 must retrieve with retries
            if round_num == 0:
                trials = 0
                while True:
                    controller_response, _, _, _ = self._generate(
                        prompt=controller_prompt_formatted,
                        ts_tensor=ts_tensor,
                        segments=None,
                        max_new_tokens=256,
                        temperature=self.controller_temperature if self.controller_temperature > 0 else 1.0,
                        top_p=top_p,
                        stopping_criteria=stop_controller,
                        do_sample=(self.controller_temperature > 0),
                    )
                    # Use imported _parse_controller_decision from training
                    decision = self.cr_generator._parse_controller_decision(controller_response)
                    trials += 1
                    
                    if decision["decision"] == "retrieve":
                        break
                    if trials >= self.first_seg_trials:
                        format_error_occurred = True
                        format_error_round = round_num
                        break
                
                if format_error_occurred:
                    controller_completions.append(controller_response)
                    break
            else:
                controller_response, _, _, _ = self._generate(
                    prompt=controller_prompt_formatted,
                    ts_tensor=ts_tensor,
                    segments=None,
                    max_new_tokens=256,
                    temperature=self.controller_temperature if self.controller_temperature > 0 else 1.0,
                    top_p=top_p,
                    stopping_criteria=stop_controller,
                    do_sample=(self.controller_temperature > 0),
                )
                # Use imported _parse_controller_decision from training
                decision = self.cr_generator._parse_controller_decision(controller_response)
            
            controller_completions.append(controller_response)
            full_conv_for_controller.append({"role": "assistant", "content": controller_response})
            
            if decision["decision"] == "error":
                format_error_occurred = True
                format_error_round = round_num
                break
            
            if decision["decision"] == "accept":
                break
            
            if decision["decision"] == "retrieve":
                segment = list(decision["segment"])
                start, end = int(segment[0]), int(segment[1])
                start = max(0, start)
                end = min(end, ts_length)
                segment = [start, end]
                
                if not segment_memory.add_segment(segment):
                    format_error_occurred = True
                    format_error_round = round_num
                    break
                
                # Use imported _create_reasoner_prompt from training
                reasoner_prompt = self.cr_generator._create_reasoner_prompt(
                    question=question, segments_data=segment_memory.get_all_segments()
                )
                sys_content_reasoner = 'you are a helpful assistant that can answer questions about time series data.'
                messages_reasoner = [
                    {"role": "system", "content": sys_content_reasoner},
                    {"role": "user", "content": reasoner_prompt}
                ]
                reasoner_prompt_formatted = self.tokenizer.apply_chat_template(
                    messages_reasoner, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                reasoner_prompts.append(reasoner_prompt_formatted)
                
                toks_tmp = self.tokenizer(reasoner_prompt_formatted, return_tensors="pt", padding=True, padding_side="left")
                initial_length = toks_tmp.input_ids.shape[1]
                
                # Use imported ReasonerStoppingCriteria from training
                stop_reasoner = StoppingCriteriaList([
                    ReasonerStoppingCriteria(self.tokenizer, initial_length=initial_length)
                ])
                
                reasoner_response, _, _, _ = self._generate(
                    prompt=reasoner_prompt_formatted,
                    ts_tensor=ts_tensor,
                    segments=segment_memory.get_all_segments(),
                    max_new_tokens=self.reasoner_max_new_tokens,
                    temperature=self.reasoner_temperature if self.reasoner_temperature > 0 else 1.0,
                    top_p=top_p,
                    stopping_criteria=stop_reasoner,
                    do_sample=(self.reasoner_temperature > 0),
                )
                
                reasoner_completions.append(reasoner_response)
                # Use imported _parse_reasoner_response from training
                parsed = self.cr_generator._parse_reasoner_response(reasoner_response)
                previous_reasoner_answer = reasoner_response
                reasoner_answers.append(parsed)
        
        final_segments = segment_memory.get_all_segments()
        hit_max_rounds = (
            len(controller_completions) >= self.max_rounds
            and not format_error_occurred
            and decision.get("decision") != "accept"
        )
        
        segment_memory.clear()
        
        return {
            "controller_completions": controller_completions,
            "reasoner_completions": reasoner_completions,
            "full_conv_for_controller": full_conv_for_controller,
            "reasoner_answers": reasoner_answers,
            "final_segments": final_segments,
            "num_rounds": len(controller_completions),
            "format_error_occurred": format_error_occurred,
            "format_error_round": format_error_round,
            "hit_max_rounds": hit_max_rounds,
            "has_accept": (decision.get("decision") == "accept") if not format_error_occurred else False,
            "final_controller_prompt": controller_prompts[-1] if controller_prompts else None,
            "final_reasoner_prompt": reasoner_prompts[-1] if reasoner_prompts else None,
        }

    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def predict_single(
        self, 
        ts_data, 
        context: str, 
        gold_answer: str = None,
        raw_ts=None, 
        timestamps=None, 
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run inference on a single example."""
        num_retries = 0
        result = None
        
        for attempt in range(self.max_format_error_retries + 1):
            with torch.no_grad():
                result = self.generate_controller_reasoner_loop(
                    question=context,
                    ts_data=ts_data,
                    gold_answer=gold_answer or "",
                    raw_ts=raw_ts if raw_ts is not None else ts_data,
                    timestamps=timestamps or {},
                    temperature=self.controller_temperature,
                    top_p=self.top_p,
                )
            
            if not result["format_error_occurred"]:
                break
            num_retries += 1
        
        final_answer = ""
        if result["reasoner_completions"]:
            # Use imported _parse_reasoner_response from training
            parsed = self.cr_generator._parse_reasoner_response(result["reasoner_completions"][-1])
            final_answer = self._normalize_answer(parsed["answer"])
        
        output = {
            "question": context,
            "final_answer": final_answer,
            "num_rounds": result["num_rounds"],
            "final_segments": result["final_segments"],
            "format_error": result["format_error_occurred"],
            "num_format_error_retries": num_retries,
            "hit_max_rounds": result["hit_max_rounds"],
            "has_accept": result["has_accept"],
            "controller_completions": result["controller_completions"],
            "reasoner_completions": result["reasoner_completions"],
            "reasoner_answers": [a["answer"] for a in result["reasoner_answers"]],
            "points_used": self._points_used_from_segments(result.get("final_segments")) / max(1, len(ts_data))
        }
        
        if gold_answer:
            normalized_gold = self._normalize_answer(gold_answer)
            output["gold_answer"] = gold_answer
            output["correct"] = (final_answer == normalized_gold)
        
        if verbose:
            # print(f"Question: {context[:100]}...")
            print(f"Final answer: {final_answer}, Gold: {normalized_gold}, Correct: {output.get('correct')}")
        
        return output
    
    def predict_dataset(
        self, 
        dataset, 
        max_samples: int = None, 
        output_file: str = None,
        verbose: bool = False, 
        save_every: int = 500,
        num_samples_per_question: int = 1, 
        passk_values: List[int] = None
    ):
        """Run inference on a dataset with pass@k support."""
        predictions = []
        points_used_all_sum = 0
        points_used_all_count = 0
        num_questions = min(len(dataset), max_samples) if max_samples else len(dataset)
        
        n = max(1, int(num_samples_per_question))
        ks = sorted({int(k) for k in (passk_values or [1]) if int(k) >= 1 and int(k) <= n}) or [1]
        
        print(f"Running inference on {num_questions} questions (n={n}, k={ks})")
        all_preds = []
        all_golds = []
        for i in tqdm(range(num_questions)):
            try:
                sample = dataset[i]
                sample_runs = []
                
                for s in range(n):
                    ts_data = sample["ts"]
                    if isinstance(ts_data, np.ndarray):
                        ts_tensor = torch.from_numpy(ts_data).float()
                    elif isinstance(ts_data, list):
                        ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
                    else:
                        ts_tensor = ts_data
                    
                    # Ensure proper shape: [seq_len, n_vars] or [seq_len]
                    if ts_tensor.dim() == 1:
                        ts_tensor = ts_tensor.unsqueeze(-1)  # [seq_len, 1]

                    gold_answer = sample["label"]
                    raw_ts = sample.get("raw_ts", ts_data)
                    if self.task == "SLEEPQA":
                        question = sample["question"]
                        options = sample["options"]
                        options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
                        ts_text = sample["ts_text"]
                        context = f"Question: {question}\nOptions: {options_text}\nTime Series information: {ts_text}"
                    else:
                        context = sample["context"]
                    
                    formatted_context = get_instruction_for_stage('mcq', self.task) + "\n" + context
                    result = self.predict_single(ts_tensor, formatted_context, gold_answer, raw_ts, None, verbose)
                    result["sample_id"] = s
                    sample_runs.append(result)
                
                q_record = {
                    "question_idx": i,
                    "question": sample.get("context"),
                    "gold_answer": sample.get("label"),
                    "n": n,
                    "k_values": ks,
                    "samples": sample_runs,
                }
                
                if sample.get("label") is not None:
                    correct_flags = [bool(r.get("correct", False)) for r in sample_runs]
                    c = int(sum(correct_flags))
                    q_record["num_correct"] = c
                    q_record["accuracy_at_1"] = float(correct_flags[0]) if correct_flags else 0.0
                    q_record["passk"] = {f"pass@{k}": float(pass_at_k_spell(n, c, k)) for k in ks}
                    q_record["any_correct"] = bool(c > 0)

                    if sample_runs:
                        for run in sample_runs:
                            pred = run.get("final_answer", "")
                            gold = run.get("gold_answer", "")
                            normalized_gold = self._normalize_answer(gold)
                            all_preds.append(pred)
                            all_golds.append(normalized_gold)
                
                predictions.append(q_record)
                points_used_all_sum += sum(float(r.get("points_used", 0.0)) for r in sample_runs)
                points_used_all_count += len(sample_runs)
                
                if (i + 1) % 10 == 0:
                    num_with_labels = sum(1 for p in predictions if "accuracy_at_1" in p)
                    if num_with_labels > 0:
                        avg_acc = sum(p.get("accuracy_at_1", 0.0) for p in predictions) / num_with_labels
                        avg_points = points_used_all_sum / points_used_all_count if points_used_all_count > 0 else 0
                        
                        # Compute precision, recall, F1
                        prec, rec, f1, class_metrics = self._compute_precision_recall_f1(all_preds, all_golds)
                        
                        metrics_str = f"[Q {i+1}/{num_questions}] Acc@1: {avg_acc:.4f}"
                        for k in ks:
                            avg_passk = sum(p.get("passk", {}).get(f"pass@{k}", 0.0) for p in predictions) / num_with_labels
                            metrics_str += f" | Pass@{k}: {avg_passk:.4f}"
                        metrics_str += f" | Any: {sum(1.0 if p.get('any_correct', False) else 0.0 for p in predictions) / num_with_labels:.4f}"
                        metrics_str += f" | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}"
                        metrics_str += f" | Pts: {avg_points:.3f}"
                        
                        print(f"\n{metrics_str}")
                        
                        # Optionally print per-class breakdown
                        if verbose and class_metrics:
                            class_str = "  Per-class: " + " | ".join(
                                f"{cls}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}"
                                for cls, m in sorted(class_metrics.items())
                            )
                            print(class_str)

                if output_file and (i + 1) % save_every == 0:
                    with open(output_file.replace(".json", f"_temp_{i+1}.json"), "w") as f:
                        json.dump(predictions, f, indent=2, default=str)
            
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                import traceback
                traceback.print_exc()
                predictions.append({"error": str(e), "question_idx": i})
        
        metrics = self._compute_final_metrics(predictions, ks, n, points_used_all_sum, points_used_all_count)
        if all_preds and all_golds:
            prec, rec, f1, class_metrics = self._compute_precision_recall_f1(all_preds, all_golds)
            metrics["precision_macro"] = prec
            metrics["recall_macro"] = rec
            metrics["f1_macro"] = f1
            metrics["per_class_metrics"] = class_metrics
        
        if output_file:
            with open(output_file, "w") as f:
                json.dump({"predictions": predictions, "metrics": metrics}, f, indent=2, default=str)
            print(f"Saved to {output_file}")
        
        return predictions, metrics
    
    def _compute_final_metrics(self, predictions, ks, n, points_sum, points_count):
        """Compute final metrics."""
        metrics = {}
        valid = [p for p in predictions if "accuracy_at_1" in p]
        if not valid:
            return metrics
        
        denom = len(valid)
        metrics["accuracy_at_1"] = sum(p.get("accuracy_at_1", 0.0) for p in valid) / denom
        metrics["n"] = n
        metrics["k_values"] = ks
        
        for k in ks:
            vals = [p.get("passk", {}).get(f"pass@{k}", 0.0) for p in valid]
            metrics[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0
        
        metrics["any_correct_rate"] = sum(1.0 if p.get("any_correct") else 0.0 for p in valid) / denom
        
        if points_count > 0:
            metrics["avg_points_used"] = points_sum / points_count
        
        print(f"\n{'='*60}")
        print("METRICS")
        print(f"{'='*60}")
        print(f"Accuracy@1: {metrics['accuracy_at_1']:.4f}")
        for k in ks:
            print(f"Pass@{k}: {metrics[f'pass@{k}']:.4f}")
        print(f"Any correct rate: {metrics['any_correct_rate']:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _points_used_from_segments(self, final_segments) -> int:
        """Calculate total points used."""
        if not final_segments:
            return 0
        total = 0
        for seg in final_segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                s, e = int(seg[0]), int(seg[1])
                total += max(0, e - s + 1)
        return total
    
    def _compute_precision_recall_f1(self, preds: List[str], golds: List[str]) -> Tuple[float, float, float, Dict]:
        """
        Compute macro-averaged precision, recall, F1 and per-class metrics.
        
        Returns:
            (precision_macro, recall_macro, f1_macro, per_class_metrics_dict)
        """
        if not preds or not golds:
            return 0.0, 0.0, 0.0, {}
        
        # Get all unique classes
        all_classes = sorted(set(golds) | set(preds))
        
        # Compute per-class metrics
        class_metrics = {}
        for cls in all_classes:
            tp = sum(1 for p, g in zip(preds, golds) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(preds, golds) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(preds, golds) if p != cls and g == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn  # Number of actual instances of this class
            }
        
        # Macro average (only over classes that appear in gold labels)
        gold_classes = sorted(set(golds))
        if gold_classes:
            precision_macro = sum(class_metrics[cls]["precision"] for cls in gold_classes) / len(gold_classes)
            recall_macro = sum(class_metrics[cls]["recall"] for cls in gold_classes) / len(gold_classes)
            f1_macro = sum(class_metrics[cls]["f1"] for cls in gold_classes) / len(gold_classes)
        else:
            precision_macro, recall_macro, f1_macro = 0.0, 0.0, 0.0
        
        return precision_macro, recall_macro, f1_macro, class_metrics

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Controller-Reasoner Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--task", type=str, default="1TS")
    parser.add_argument("--training_stage", type=str, default="mcq")
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--output_file", type=str, default="predictions.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--first_seg_trials", type=int, default=5)
    parser.add_argument("--controller_temperature", type=float, default=0.3)
    parser.add_argument("--reasoner_temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--include_full_ts_initially", action="store_true")
    parser.add_argument("--reasoner_max_new_tokens", type=int, default=768)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_flash_attention", action="store_true")
    parser.add_argument("--num_samples_per_question", type=int, default=1)
    parser.add_argument("--passk", type=str, default="1,2,4,8")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    args = parser.parse_args()
    
    lora_cfg = {
        'r': args.lora_r,
        'alpha': args.lora_alpha,
        'dropout': 0.02,
        'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
    
    inference = ControllerReasonerInference(
        model_path=args.model_path,
        lora_weights_path=args.lora_weights,
        task=args.task,
        training_stage=args.training_stage,
        max_rounds=args.max_rounds,
        first_seg_trials=args.first_seg_trials,
        controller_temperature=args.controller_temperature,
        reasoner_temperature=args.reasoner_temperature,
        top_p=args.top_p,
        include_full_ts_initially=args.include_full_ts_initially,
        reasoner_max_new_tokens=args.reasoner_max_new_tokens,
        device=args.device,
        use_flash_attention=not args.no_flash_attention,
        lora_cfg=lora_cfg,
    )
    
    if args.dataset_config:
        with open(args.dataset_config, 'r') as f:
            config = yaml.safe_load(f)
        
        from importlib import import_module
        dataset_class = config["dataset"]["class_path"]
        TaskClass = _import_class(dataset_class)
        init_args = config["dataset"]["init_args"] or {}
        dataset_task = TaskClass(**init_args)
        # module_name, class_name = dataset_class.rsplit(".", 1)
        # module = import_module(module_name)
        # DatasetClass = getattr(module, class_name)
        
        test_data = dataset_task.load("test")
        
        try:
            from multimodal import MultimodalMCQDataset
            dataset = MultimodalMCQDataset(
                test_data,
                tokenizer=inference.tokenizer,
                context_columns=dataset_task.context_columns,
                context_prefix=dataset_task.context_prefix,
                ts_column=dataset_task.ts_column,
                format_abc_mcq=getattr(dataset_task, 'format_abc_mcq', True),
                label_column=dataset_task.label_column,
                options_column=getattr(dataset_task, 'options_column', 'options'),
                task_name=args.task,
                w_cot=dataset_task.w_cot,
                partition="test",
                scale_ts=True,
                shuffle_labels=dataset_task.shuffle_labels
            )
        except ImportError:
            dataset = test_data
        
        ks = [int(x) for x in args.passk.split(",") if x.strip()]
        n = max(ks) if ks else args.num_samples_per_question
        
        predictions, metrics = inference.predict_dataset(
            dataset, args.max_samples, args.output_file, args.verbose, 500, n, ks
        )
        print(f"Metrics: {metrics}")
    else:
        print("No dataset config provided.")


if __name__ == "__main__":
    main()