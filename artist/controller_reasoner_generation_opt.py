#!/usr/bin/env python3
"""
Generation logic for Controller-Reasoner framework
"""

import torch
import json
import re
from typing import List, Dict, Tuple, Optional
from segment_memory import SegmentMemory, SegmentMemoryManager
from tools_qwents import TS_TOOLS
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from qwents_generation_handler import ToolStoppingCriteria
from scipy.signal import argrelextrema
from torch.nn.utils.rnn import pad_sequence
TS_PLACEHOLDER = "<ts><ts/>" 

class ControllerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, initial_length: int):
        """
        Args:
            tokenizer: The tokenizer
            initial_length: Length of the prompt (so we only check newly generated tokens)
        """
        self.tokenizer = tokenizer
        self.initial_length = initial_length
        self.stop_id_seqs = [
            tokenizer.encode("</tool_call>", add_special_tokens=False),
            tokenizer.encode("</answer>", add_special_tokens=False),
            tokenizer.encode("<|im_end|>", add_special_tokens=False),
        ]
    
    def __call__(self, input_ids, scores, **kwargs):
        batch_size = input_ids.shape[0]
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for i in range(batch_size):
            # Only check tokens generated AFTER the initial prompt
            if input_ids.shape[1] <= self.initial_length:
                continue
            
            # Get only the newly generated tokens
            new_tokens = input_ids[i, self.initial_length:].tolist()
            
            # Check if any stop sequence appears in the new tokens
            for stop in self.stop_id_seqs:
                L = len(stop)
                if L <= len(new_tokens) and new_tokens[-L:] == stop:
                    is_done[i] = True
                    break
        
        return is_done

class ReasonerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, initial_length: int, string_check_interval: int = 5):
        """
        Args:
            tokenizer: The tokenizer
            initial_length: Length of the prompt (so we only check newly generated tokens)
            string_check_interval: How often to run the slower string-based check
        """
        self.tokenizer = tokenizer
        self.initial_length = initial_length
        self.string_check_interval = string_check_interval
        self.call_count = 0
        self.stop_strings = ["</answer>", "<|im_end|>"]
        self.stop_id_seqs = [
            tokenizer.encode("</answer>", add_special_tokens=False),
            tokenizer.encode("<|im_end|>", add_special_tokens=False),
        ]
    
    def __call__(self, input_ids, scores, **kwargs):
        self.call_count += 1
        batch_size = input_ids.shape[0]
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for i in range(batch_size):
            # Only check tokens generated AFTER the initial prompt
            if input_ids.shape[1] <= self.initial_length:
                continue
            
            # Get only the newly generated tokens
            new_tokens = input_ids[i, self.initial_length:].tolist()
            
            # Token-based check (fast)
            for stop in self.stop_id_seqs:
                L = len(stop)
                if L <= len(new_tokens) and new_tokens[-L:] == stop:
                    is_done[i] = True
                    break
            
            # String-based fallback (slower, run less frequently)
            if not is_done[i] and (self.call_count % self.string_check_interval == 0):
                # Only decode the newly generated tokens
                decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                for stop_str in self.stop_strings:
                    if stop_str in decoded:
                        is_done[i] = True
                        break
        
        return is_done

def _compute_extrema(ts: np.ndarray, order: int = 3):
    """
    Compute local minima and maxima for a 1D time series.
    
    Args:
        ts (np.ndarray): 1D time series array
        order (int): How many points to compare on each side (smoothing degree)
    Returns:
        minima, maxima: lists of (index, value)
    """
    if ts.ndim > 1:
        ts = ts.squeeze()
    if len(ts) < 3:
        return [], []
    
    # find local minima and maxima using relative comparison
    minima_idx = argrelextrema(ts, np.less_equal, order=order)[0]
    maxima_idx = argrelextrema(ts, np.greater_equal, order=order)[0]
    
    minima = [(int(i), float(ts[i])) for i in minima_idx]
    maxima = [(int(i), float(ts[i])) for i in maxima_idx]
    
    return minima, maxima


class ControllerReasonerGenerator:
    """
    Manages the iterative Controller-Reasoner loop
    """
    def __init__(self, 
                 model,
                 tokenizer,
                 generation_handler,
                 max_rounds: int = 5,
                 first_seg_trials: int = 5,
                 task_name: str = "ETI",
                 include_full_ts_initially: bool = False,
                 use_conversation_history: bool = False,
                 controller_temperature: float = 0.9,
                 reasoner_temperature: float = 0.7,
                 extended_prompt: bool = False,
                 reasoner_max_new_tokens: int = 512,
                 use_uncertainty_prompt: bool = False):
        """
        Args:
            include_full_ts_initially: If True, reasoner sees full TS in round 1
            use_conversation_history: If True, controller sees full conversation
            use_uncertainty_prompt: If True, uses v2 prompts with uncertainty guidance for reasoner
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_handler = generation_handler
        self.max_rounds = max_rounds
        self.include_full_ts_initially = include_full_ts_initially
        self.use_conversation_history = use_conversation_history
        self.first_seg_trials = first_seg_trials
        self.task_name = task_name
        self.controller_temperature = controller_temperature
        self.reasoner_temperature = reasoner_temperature
        self.extended_prompt = extended_prompt
        self.reasoner_max_new_tokens = reasoner_max_new_tokens
        self.use_uncertainty_prompt = use_uncertainty_prompt
        # self.segment_encoding_cache = {}  # {(start, end): encoding}
    
    def _get_or_encode_segment(
        self,
        ts_data,
        question_ids,
        segment
    ):
        """
        Get cached segment encoding or encode it
        """
        # Encode new segment
        start, end = segment
        
        # Prepare TS tensor
        ts_tensor = ts_data #self.generation_handler._prepare_ts_tensor(ts_data)
        
        # Get question embedding
        q_tensor = torch.tensor(question_ids, device=self.model.device).unsqueeze(0)
        q_emb = self.model.model.embed_tokens(q_tensor)
        
        # Encode segment
        seg_encoding = self.generation_handler._encode_ts_segment(
            ts_tensor,
            q_emb,
            [start, end]
        )

        seg_encoding = seg_encoding.detach().cpu()
        del q_tensor, q_emb
            
        return seg_encoding

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
        MIN_SEGMENT_SIZE =  8 #self.model.config.ts.get('patch_size', 16)

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
            
           
            ts_sample = ts_tensor[batch_idx].squeeze(-1)  # [seq_len]
           
            for start, end in segments:
                # Extract segment
                if end - start < MIN_SEGMENT_SIZE:
                    ts_len = ts_sample.shape[0] #if ts_tensor.dim() == 1 else ts_tensor.shape[-1]
                    end = min(start + MIN_SEGMENT_SIZE, ts_len)
                    start = max(0, end - MIN_SEGMENT_SIZE)

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
                
                # Pad segment to max_seg_len
                if seg_len < max_seg_len:
                    # Pad with last value
                    # padding = segment[-1].repeat(max_seg_len - seg_len)
                    padding = torch.zeros(max_seg_len - seg_len, dtype=ts_tensor[0].dtype, device=ts_tensor[0].device)
                    padded_segment = torch.cat([segment, padding])
                else:
                    padded_segment = segment
                
                # Create mask (1 for valid, 0 for padding)
               
                if self.task_name in ["TSQA", "TRQA_MIXED", "TRQA", "ETI"]:
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

    def _generate(
        self,
        prompt: str,
        ts_tensor: torch.Tensor,
        segments,                      # None => full TS, else List[[start,end], ...]
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stopping_criteria: StoppingCriteriaList,
        do_sample: bool = True,
    ):
        """
        generation that uses model._prepare_ts_segments_for_model + timeseries=...
        Returns: decoded_text, prompt_ids_tensor, completion_ids_tensor
        """
        with torch.no_grad():
            toks = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.device)

            input_ids = toks.input_ids
            attn_mask = toks.attention_mask
            initial_length = input_ids.shape[1]

            # Decide injected segments:
            # - Controller typically wants FULL TS embedding injected at the TS placeholder(s).
            # - Reasoner wants only selected segments.
            if segments is None:
                # Full TS segment (inclusive/exclusive depends on your tool convention; keep consistent)
                ts_len = ts_tensor.shape[-1] if ts_tensor.dim() > 1 else ts_tensor.shape[0]
                segs_for_model = [[0, ts_len]]
            else:
                segs_for_model = segments

            # Prepare batch-shaped ts input for the model
            # Expecting _prepare_ts_segments_for_model(ts_batch, segs_batch)
            # where segs_batch is a list length B, each is list of segments.
            if ts_tensor.dim() == 1:
                ts_batch = ts_tensor.unsqueeze(0).unsqueeze(0)   # (B=1, C=1, L)
            elif ts_tensor.dim() == 2:
                ts_batch = ts_tensor.unsqueeze(0)                # (B=1, C, L)
            else:
                # already has batch?
                ts_batch = ts_tensor

            ts_for_model = self._prepare_ts_segments_for_model(
                ts_batch,                # (B, C, L) or compatible
                [segs_for_model],        # batch wrapper
            )
            if ts_for_model is not None:
                if self.task_name == "TSQA" or self.task_name == "TRQA_MIXED" or self.task_name == "TRQA":
                    ts_for_model = ts_for_model.to(self.model.device, dtype=torch.bfloat16)
                else:
                    ts_for_model = ts_for_model.to(self.model.device)

            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                timeseries=ts_for_model,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            new_tokens = gen_ids[0, initial_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            return text, input_ids.detach().cpu(), new_tokens.detach().cpu(), initial_length


    def generate_controller_reasoner_loop(
        self,
        question,
        ts_data,
        gold_answer,
        timestamps=None,
        raw_ts=None,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,   # kept for signature compatibility; generate() uses cache internally
    ):
        ts_tensor = ts_data.to(self.model.device) if isinstance(ts_data, torch.Tensor) else ts_data
        if not isinstance(ts_tensor, torch.Tensor):
            raise TypeError("ts_data must be a torch.Tensor for generation.")

        # Infer length (L)
        ts_length = ts_tensor.shape[0]

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
        decision = {"decision": "error"}  # default

        # -------- optional: initial full-TS reasoner answer --------
        if self.include_full_ts_initially:
            full_ts_segment = [0, ts_length]
            if not segment_memory.add_segment(full_ts_segment):
                return {
                    "controller_completions": [],
                    "reasoner_completions": [],
                    "full_conv_for_controller": [],
                    "reasoner_answers": [],
                    "final_segments": [],
                    "segment_encodings": [],
                    "num_rounds": 0,
                    "format_error_occurred": True,
                    "format_error_round": 0,
                    "hit_max_rounds": False,
                    "has_accept": False,
                    "final_controller_prompt": None,
                    "final_reasoner_prompt": None,
                }

            reasoner_prompt = self._create_reasoner_prompt(
                question=question,
                segments_data=[full_ts_segment],
            )
            sys_content_reasoner = "you are a helpful assistant that can answer questions about time series data."
            messages_reasoner = [
                {"role": "system", "content": sys_content_reasoner},
                {"role": "user", "content": reasoner_prompt},
            ]
            reasoner_prompt_formatted = self.tokenizer.apply_chat_template(
                messages_reasoner,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            reasoner_prompts.append(reasoner_prompt_formatted)

            # stopping criteria based on the prompt length
            toks_tmp = self.tokenizer(reasoner_prompt_formatted, return_tensors="pt")
            initial_length = toks_tmp.input_ids.shape[1]
            stop = StoppingCriteriaList([
                ReasonerStoppingCriteria(self.tokenizer, initial_length=initial_length)
            ])

            reasoner_response, _, _, _ = self._generate(
                prompt=reasoner_prompt_formatted,
                ts_tensor=ts_tensor,
                segments=[full_ts_segment],
                max_new_tokens=self.reasoner_max_new_tokens,
                temperature=self.reasoner_temperature if self.reasoner_temperature > 0 else 1.0,
                top_p=top_p,
                stopping_criteria=stop,
                do_sample=(self.reasoner_temperature > 0),
            )

            reasoner_completions.append(reasoner_response)
            parsed = self._parse_reasoner_response(reasoner_response)
            previous_reasoner_answer = reasoner_response
            reasoner_answers.append(parsed)

        # ---------------- main controller-reasoner loop ----------------
        for round_num in range(self.max_rounds):
            # Build controller prompt
            if round_num == 0:
                if self.include_full_ts_initially:
                    controller_prompt = self._create_controller_prompt_generic_mcq_full_ts(
                        question=question,
                        current_segments=segment_memory.get_all_segments(),
                        segment_memory=segment_memory,
                        previous_reasoner_answer=previous_reasoner_answer,
                    )
                else:
                    controller_prompt = self._create_first_turn_controller_prompt(
                        question=question,
                        segment_memory=segment_memory,
                        ts=np.array(raw_ts) if raw_ts is not None else None,
                    )
            else:
                controller_prompt = self._create_controller_prompt(
                    question=question,
                    current_segments=segment_memory.get_all_segments(),
                    segment_memory=segment_memory,
                    previous_reasoner_answer=previous_reasoner_answer,
                    ts=np.array(raw_ts) if raw_ts is not None else None,
                )

            sys_content = "you are a helpful assistant that can answer questions about time series data."
            controller_messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": controller_prompt},
            ]

            controller_prompt_formatted = self.tokenizer.apply_chat_template(
                controller_messages,
                tokenize=False,
                tools=TS_TOOLS,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            controller_prompts.append(controller_prompt_formatted)
            full_conv_for_controller.append(controller_messages)

            # stopping criteria based on this prompt length
            toks_tmp = self.tokenizer(controller_prompt_formatted, return_tensors="pt")
            initial_length = toks_tmp.input_ids.shape[1]
            stop_controller = StoppingCriteriaList([
                ControllerStoppingCriteria(self.tokenizer, initial_length=initial_length)
            ])

            # Round 0 must retrieve; allow retries (format discipline)
            if round_num == 0:
                trials = 0
                while True:
                    controller_response, _, _, _ = self._generate(
                        prompt=controller_prompt_formatted,
                        ts_tensor=ts_tensor,
                        segments=None,  # FULL TS injection for controller
                        max_new_tokens=256,
                        temperature=self.controller_temperature if self.controller_temperature > 0 else 1.0,
                        top_p=top_p,
                        stopping_criteria=stop_controller,
                        do_sample=(self.controller_temperature > 0),
                    )
                    decision = self._parse_controller_decision(controller_response)
                    trials += 1

                    if decision["decision"] == "retrieve":
                        break

                    if trials >= self.first_seg_trials:
                        format_error_occurred = True
                        format_error_round = round_num
                        break

                if format_error_occurred:
                    controller_completions.append(controller_response)
                    full_conv_for_controller.append({"role": "assistant", "content": controller_response})
                    break

            else:
                controller_response, _, _, _ = self._generate(
                    prompt=controller_prompt_formatted,
                    ts_tensor=ts_tensor,
                    segments=None,  # FULL TS injection for controller
                    max_new_tokens=256,
                    temperature=self.controller_temperature if self.controller_temperature > 0 else 1.0,
                    top_p=top_p,
                    stopping_criteria=stop_controller,
                    do_sample=(self.controller_temperature > 0),
                )
                decision = self._parse_controller_decision(controller_response)

            controller_completions.append(controller_response)
            full_conv_for_controller.append({"role": "assistant", "content": controller_response})

            # Handle decision
            if decision["decision"] == "error":
                format_error_occurred = True
                format_error_round = round_num
                break

            if decision["decision"] == "accept":
                break

            if decision["decision"] == "retrieve":
                segment = list(decision["segment"])
                if not segment_memory._is_valid_segment(segment):
                    format_error_occurred = True
                    format_error_round = round_num
                    break

                # clamp and store
                start, end = int(segment[0]), int(segment[1])
                start = max(0, start)
                end = min(end, ts_length)
                segment = [start, end]

                if not segment_memory.add_segment(segment):
                    format_error_occurred = True
                    format_error_round = round_num
                    break

                # Reasoner prompt using selected segments
                reasoner_prompt = self._create_reasoner_prompt(
                    question=question,
                    segments_data=segment_memory.get_all_segments(),
                )
                sys_content_reasoner = "you are a helpful assistant that can answer questions about time series data."
                messages_reasoner = [
                    {"role": "system", "content": sys_content_reasoner},
                    {"role": "user", "content": reasoner_prompt},
                ]
                reasoner_prompt_formatted = self.tokenizer.apply_chat_template(
                    messages_reasoner,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                reasoner_prompts.append(reasoner_prompt_formatted)

                toks_tmp = self.tokenizer(reasoner_prompt_formatted, return_tensors="pt")
                initial_length = toks_tmp.input_ids.shape[1]
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
                parsed = self._parse_reasoner_response(reasoner_response)
                previous_reasoner_answer = reasoner_response
                reasoner_answers.append(parsed)

        final_controller_prompt = controller_prompts[-1] if controller_prompts else None
        final_reasoner_prompt = reasoner_prompts[-1] if reasoner_prompts else None
        final_segments = segment_memory.get_all_segments()

        final_segment_encodings = []

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
            "segment_encodings": final_segment_encodings,
            "num_rounds": len(controller_completions),
            "format_error_occurred": format_error_occurred,
            "format_error_round": format_error_round,
            "hit_max_rounds": hit_max_rounds,
            "has_accept": (decision.get("decision") == "accept") if not format_error_occurred else False,
            "final_controller_prompt": final_controller_prompt,
            "final_reasoner_prompt": final_reasoner_prompt,
        }

    
    def generate_reasoner_rollouts(
        self,
        question: str,
        ts_data,
        segments,
        segment_encodings,
        question_ids: List[int],
        num_rollouts: int = 8,
        timestamps = None,
        raw_ts = None,
        temperature = 0.7,
        top_p = 0.9,
        ):
        """
        Generate G rollouts with cached segment encodings
        """
        completions = []
        answers = []
        ts_tensor = ts_data #self.generation_handler._prepare_ts_tensor(ts_data)
        # Get all segment encodings (should be cached from main loop)
        # segment_encodings = [
        #     self.segment_encoding_cache.get(seg) or 
        #     self._get_or_encode_segment(ts_tensor, question_ids, seg)
        #     for seg in segments
        # ]
        
        # print(f"\nGenerating {num_rollouts} reasoner rollouts with {len(segments)} segments...")
        reasoner_prompt = self._create_reasoner_prompt(
                question=question,
                segments_data=segments,
        )
        sys_content_reasoner = 'you are a helpful assistant that can answer questions about time series data.'
        messages_reasoner = [
            {"role": "system", "content": sys_content_reasoner},
            {"role": "user", "content": reasoner_prompt}
        ]
        
        reasoner_prompt_formatted = self.tokenizer.apply_chat_template(
            messages_reasoner, 
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=True
        )
        
       
        for i in range(num_rollouts):
            # TODO: make sure the no_grad is sutiable here
            with torch.no_grad():   
                completion, _ = self.generation_handler.generate_single_shot(
                    prompt=reasoner_prompt_formatted,
                    ts_data=ts_tensor,
                    question_ids=question_ids,
                    segments=segments,
                    segment_encodings=segment_encodings,  # Reuse cached encodings
                    max_new_tokens=self.reasoner_max_new_tokens,
                    temperature=self.reasoner_temperature,
                    do_sample=True,
                    top_p=top_p,
                )
            
            completions.append(completion)
            
            parsed = self._parse_reasoner_response(completion)
            answers.append(parsed["answer"])
            
            # if (i + 1) % 2 == 0:
            #     print(f"  Completed {i+1}/{num_rollouts} rollouts")
        
        return completions, answers
    
    def generate_reasoner_rollouts_batched(
        self,
        question: str,
        ts_data,
        segments,
        num_rollouts: int = 8,
        timestamps = None,
        raw_ts = None,
        temperature = 0.7,
        top_p = 0.9,
        micro_batch_size = None,
        max_new_tokens = 512,
        ):
        """
        Generate G rollouts with cached segment encodings
        """
        completions = []
        answers = []
        prompt_ids_list = []

        reasoner_prompt = self._create_reasoner_prompt(
                question=question,
                segments_data=segments,
        )
        sys_content_reasoner = 'you are a helpful assistant that can answer questions about time series data.'
        messages_reasoner = [
            {"role": "system", "content": sys_content_reasoner},
            {"role": "user", "content": reasoner_prompt}
        ]
        
        reasoner_prompt_formatted = self.tokenizer.apply_chat_template(
            messages_reasoner, 
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=True
        )
        
        with torch.no_grad():
            toks = self.tokenizer(reasoner_prompt_formatted, return_tensors="pt").to(self.model.device)
            prompt_ids = toks.input_ids.cpu()
            initial_length = toks.input_ids.shape[1]
            
            stop = StoppingCriteriaList([
                ReasonerStoppingCriteria(
                    self.tokenizer,
                    initial_length=initial_length
                )
            ])
            mb = micro_batch_size if micro_batch_size is not None else num_rollouts

            start = 0
            while start < num_rollouts:
                end = min(start + mb, num_rollouts)
                this_bs = end - start

                # Repeat input tokens for batch
                input_ids_batch = toks.input_ids.repeat(this_bs, 1)
                attention_mask_batch = toks.attention_mask.repeat(this_bs, 1)
                
                # ← FIX: Prepare timeseries for the BATCHED input
                # Each sample in batch needs its own timeseries entry
                ts_tensor = ts_data.unsqueeze(0) if ts_data.dim() == 2 else ts_data
                
                # Repeat ts_data and segments for batch
                ts_batch_input = ts_tensor.repeat(this_bs, 1, 1) if ts_tensor.dim() == 3 else ts_tensor.repeat(this_bs, 1)
                segments_batch = [segments] * this_bs  # List of segments for each sample
                
                ts_for_model = self._prepare_ts_segments_for_model(
                    ts_batch_input,
                    segments_batch  # ← Pass list with one entry per batch sample
                )
                if ts_for_model is not None:
                    if self.task_name == "TSQA" or self.task_name == "TRQA_MIXED":
                        ts_for_model = ts_for_model.to(self.model.device, dtype=torch.bfloat16)
                    else:
                        ts_for_model = ts_for_model.to(self.model.device)

                gen_ids = self.model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    timeseries=ts_for_model,  # ← Correctly batched timeseries
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stop,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode only the newly generated tokens
                for i in range(this_bs):
                    new_tokens = gen_ids[i, initial_length:]
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    completions.append(text)
                    parsed = self._parse_reasoner_response(text)
                    answers.append(parsed.get("answer"))
                    prompt_ids_list.append(prompt_ids)

                # Free ASAP
                del input_ids_batch, attention_mask_batch, gen_ids, ts_for_model
                torch.cuda.empty_cache()
                start = end

            # Cleanup
            del toks
            torch.cuda.empty_cache()

        return completions, answers, prompt_ids

    def _create_first_turn_controller_prompt(self, question, segment_memory, ts):
        if self.task_name in ["ETI", "TIMERBED_RCW"]:
            if self.extended_prompt:
                return self._create_first_turn_controller_prompt_ETI_extended(question, ts)
            else:
                return self._create_first_turn_controller_prompt_ETI(question, segment_memory)
        elif self.task_name in ["TIMERBED_ECG"]:
            if self.extended_prompt:
                return self._create_first_turn_controller_prompt_ECG_extended(question, segment_memory, ts)
            else:
                return self._create_first_turn_controller_prompt_ECG(question, segment_memory)
        # elif self.task_name in ["ECG_QA_S_VERIFY"]:
        #     if self.extended_prompt:
        #         return self._create_first_turn_controller_prompt_ECG_QA_VERIFY_extended(question, segment_memory, ts)
        #     else:
        #         return self._create_first_turn_controller_prompt_ECG_v2(question, segment_memory)
        else:
            return self._create_first_turn_controller_prompt_ETI(question, segment_memory)
    
    def _create_reasoner_prompt(self, question, segments_data):
        if self.task_name in ["TIMERBED_RCW"]:
            if self.use_uncertainty_prompt:
                return self._create_reasoner_prompt_ETI_v2(question, segments_data)
            else:
                return self._create_reasoner_prompt_ETI(question, segments_data)
        elif self.task_name in ["TIMERBED_ECG"]:
            return self._create_reasoner_prompt_ECG(question, segments_data)
        # elif self.task_name in ["ECG_QA_S_VERIFY"]:
        #     return self._create_reasoner_prompt_ECG_QA_VERIFY_v2(question, segments_data)
        elif self.task_name in ["ECG_QA_S_VERIFY"]:
            return self._create_reasoner_prompt_generic_MCQ_v5(question, segments_data)
        elif self.task_name in ["TSQA_TF"]:
            return self._create_reasoner_prompt_generic_TF(question, segments_data)
        elif self.task_name in ["TRQA_MIXED"]:
            return self._create_reasoner_prompt_generic_MCQ_v4(question, segments_data)
        else:
            # return self._create_reasoner_prompt_generic_MCQ(question, segments_data)
            return self._create_reasoner_prompt_generic_MCQ_v3(question, segments_data)
    
    def _create_controller_prompt(self, question, current_segments, segment_memory: SegmentMemory, previous_reasoner_answer: Optional[str], ts: np.ndarray):
        if self.task_name in ["ETI", "TIMERBED_RCW"]:
            if self.extended_prompt:
                return self._create_controller_prompt_ETI_extended(question, current_segments, segment_memory, previous_reasoner_answer, ts)
            else:
                return self._create_controller_prompt_ETI(question, current_segments, segment_memory, previous_reasoner_answer)
        elif self.task_name in ["TIMERBED_ECG"]:
            if self.extended_prompt:
                return self._create_controller_prompt_ECG_extended(question, current_segments, segment_memory, previous_reasoner_answer, ts)
            else:
                return self._create_controller_prompt_ECG_v2(question, current_segments, segment_memory, previous_reasoner_answer)
        else:
            return self._create_controller_prompt_ETI(question, current_segments, segment_memory, previous_reasoner_answer)
    
    def create_reasoner_mcq_generic_sft_style_prompt(self, question, segments):
        segment_placeholders = "\n".join([TS_PLACEHOLDER for _ in segments])
        prompt = f"{segment_placeholders}\n{question}"
        return prompt

    def _create_first_turn_controller_prompt_scratch(self, question, segment_memory):
        total_ts_length = segment_memory.ts_length

        prompt = f"""# ROLE
        You are a controller for a time-series QA system. Output must follow the schema exactly.

        # Inputs
        - Full TS embedding:
        {TS_PLACEHOLDER}

        - Time series length: {total_ts_length} timesteps
        - Question: {question}

        # Task
        Select the **single** most informative segment to answer the question.

        # Output Format (STRICT)
        <think>One sentence (≤20 words) explaining why this segment is most informative.</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        # Rules (MANDATORY)
        - No text outside the two tags above.
        - Exactly one tool call; no ACCEPT in round 1.
        - You may choose ANY segment length: from 10 steps up to the full series.
        - ts_seg MUST be two **integers**: [start, end]. **No math or variables** (bad: "[150, 150+50]").
        - Example: if you want a 10-step window, compute it yourself and output integers (good: "[381, 391]").
        - 0 ≤ start < end ≤ {total_ts_length - 1}.

        # Examples
        Retrieve (good):
        <think>Trend change around the suspected event must be inspected.</think>
        <tool_call>
        {{"name":"timeseries_zoom_in_tool","arguments":{{"ts_seg":[260,340]}}}}
        </tool_call>

        Make your decision now."""

        return prompt
    
    def _create_first_turn_controller_prompt_ETI(self, question, segment_memory):
        total_ts_length = segment_memory.ts_length

        prompt = f"""# TASK
        You are a time series analysis expert. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.

        ## Instructions
        1. You receive the FULL time series embedding and the question.
        2. Review the question and analyze the full time series embedding, decide which segment is the most important to answer the question accurately.

        ## Full Time Series Embedding
        {TS_PLACEHOLDER}

        ## Current Status
        - **Time series length**: {total_ts_length} timesteps
        - **Question**: {question}

        ## Output Format
        Your response should be in the following format:
        <think>...</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>
        
        ## Guidelines
        in the <think> section, you should provide the concise reasoning (1 sentence) for why you are retrieving this segment. be concise and to the point.
        in the <tool_call> section, you should provide the segment for the tool call.
        ## Constraints
        - Valid range for the segment: [0, {total_ts_length - 1}]
        - start and end should be integers
        - You can call `timeseries_zoom_in_tool` ONCE in this round
        - Your output MUST include a <think> block (1 sentence) BEFORE the tool call.

        Only the above format is valid for this round.
        Make your decision now:
        """
        return prompt
    
    def _create_first_turn_controller_prompt_ECG_QA_VERIFY(self, question, segment_memory):
        total_ts_length = segment_memory.ts_length

        prompt = f"""# TASK
        You are a time-series analysis Controller. Your job is to decide which segment of a time series an LLM must inspect to answer a question.

        You MUST output a tool call. NOTHING ELSE.

        # What You Get
        - Full time series embedding:
        {TS_PLACEHOLDER}
        - Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        - Question: {question}

        # What You Must Do
        1. Read the question.
        2. Inspect the full embedding.
        3. Decide the single most relevant segment.
        4. Output ONLY:
        <think>YOUR ONE-SENTENCE INTERNAL REASONING</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool",
            "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        # HARD RULES (must follow)
        - Output MUST contain a <think> block FIRST.
        - Then a <tool_call> block containing valid JSON.
        - You MUST call `timeseries_zoom_in_tool` exactly once.
        - start and end MUST be integers.
        - start >= 0
        - end <= {total_ts_length - 1}
        - Nothing outside these tags is allowed.

        Make your decision and produce the required format NOW.
        """
        return prompt

    def _create_first_turn_controller_prompt_ECG_QA_VERIFY_extended(self, question, segment_memory, ts):
        total_ts_length = segment_memory.ts_length

        # --- Prep & length ---
        ts = ts.squeeze() if ts is not None and ts.ndim > 1 else ts
        N = int(len(ts)) if ts is not None else int(segment_memory.ts_length)

        # --- Extrema (indices only) ---
        if ts is None or N < 3:
            minima_idx, maxima_idx = [], []
        else:
            from scipy.signal import argrelextrema
            order = max(1, int(0.1 * N))
            minima_idx = [int(i) for i in argrelextrema(ts, np.less_equal, order=order)[0].tolist()]
            maxima_idx = [int(i) for i in argrelextrema(ts, np.greater_equal, order=order)[0].tolist()]

        # Helper to clamp segments
        def _seg(lo, hi):
            lo = max(0, int(lo))
            hi = min(N - 1, int(hi))
            if lo > hi:
                lo, hi = hi, lo
            return f"[{lo}, {hi}]"

        # --- Build illustrative example segments ---
        example_segments = []
        if len(maxima_idx) >= 2:
            example_segments.append(_seg(maxima_idx[0] - 100, maxima_idx[0] + 100))
            example_segments.append(_seg(maxima_idx[1] - 20,  maxima_idx[1] + 300))
            example_segments.append(_seg(0, maxima_idx[0] + 150))

            if len(maxima_idx) >= 3:
                example_segments.append(_seg(maxima_idx[2] - 50,  maxima_idx[2] + 500))

        if len(minima_idx) >= 2:
            example_segments.append(_seg(minima_idx[0] + 100, minima_idx[0] + 200))
            example_segments.append(_seg(minima_idx[1], minima_idx[1] + 800))

        # Fallback if no extrema found
        if not example_segments:
            example_segments = [
                _seg(0, min(199, N - 1)),
                _seg(max(0, N // 2 - 100), min(N // 2 + 100, N - 1)),
                _seg(max(0, N - 600), N - 1),
            ]

        example_segments_text = ", ".join(example_segments)

        # Extrema info block
        extrema_info = ""
        if minima_idx or maxima_idx:
            extrema_info = (
                "\n        # Key Points in the Time Series (indices)\n"
                f"        - Local minima indices: {minima_idx}\n"
                f"        - Local maxima indices: {maxima_idx}\n"
                f"        # Example segments based on key points: {example_segments_text}\n"
            )

        # --- MAIN PROMPT ---
        prompt = f"""# TASK
        You are a time-series analysis Controller. Your job is to decide which segment of a time series an LLM must inspect to answer a question.

        You MUST output a tool call. NOTHING ELSE.

        # What You Get
        - Full time series embedding:
        {TS_PLACEHOLDER}
        - Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        - Question: {question}
        {extrema_info}

        # What You Must Do
        1. Read the question.
        2. Inspect the full embedding.
        3. Use key points and example windows if helpful.
        4. Decide the single most relevant segment.
        5. Output ONLY:
        <think>YOUR ONE-SENTENCE INTERNAL REASONING</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool",
            "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        # HARD RULES (must follow)
        - Output MUST contain a <think> block FIRST.
        - Then a <tool_call> block containing valid JSON.
        - You MUST call `timeseries_zoom_in_tool` exactly once.
        - start and end MUST be integers.
        - start >= 0
        - end <= {total_ts_length - 1}
        - Nothing outside these tags is allowed.

        Make your decision and produce the required format NOW.
        """

        return prompt

    def _create_first_turn_controller_prompt_ECG(self, question, segment_memory):
        total_ts_length = segment_memory.ts_length

        prompt = f"""# TASK
        You are a time-series analysis Controller. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.

        ## Instructions
        1. You receive the FULL time series embedding and the question.
        2. Review the question and analyze the full time series embedding, decide which segment is the most important to answer the question accurately.

        ## Inputs
        • Full time series embedding:
        {TS_PLACEHOLDER}
        • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        • Question: {question}

        ## Output Format
        Your response should be in the following format:
        <think>...</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>
        
        ## Guidelines
        in the <think> section, you should provide the concise reasoning (1 sentence) for why you are retrieving this segment. be concise and to the point.
        in the <tool_call> section, you should provide the segment for the tool call.
        ## Constraints
        - Valid range for the segment: [0, {total_ts_length - 1}]
        - start and end should be integers
        - You can call `timeseries_zoom_in_tool` ONCE in this round
        - Your output MUST include a <think> block (1 sentence) BEFORE the tool call.

        Only the above format is valid for this round.
        Make your decision now:
        """
        return prompt
    
    def _create_first_turn_controller_prompt_ECG_v2(self, question, segment_memory):
        total_ts_length = segment_memory.ts_length

        prompt = f"""# TASK
        You are a time-series optimizer CONTROLLER. You assist a REASONER that tries to answer a question about a time series. 
        The REASONER only sees the segment of the time series that YOU choose.

        # YOUR GOAL
        Your goal is to choose the optimal most informative segment ([start, end]) of the time series that will enable the REASONER to answer the question accurately.
        Use your inherent knowledge about the question domain and context as well as your understanding of the time series to identify the most informative region.

        # INPUTS
        • Full time series embedding:
        {TS_PLACEHOLDER}
        • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        • Question: {question}

       # OUTPUT FORMAT (STRICT)
        <think>Your reasoning for why more evidence is needed and what information is missing.</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>
        <answer>
        Short explanation of why this segment is the most informative for solving the question and what missing evidence it provides based on time series analysis and your inherent knowledge about the time series, question domain and context.
        </answer>

        # GUIDELINES
        - In <think>, your reasoning for why you are retrieving this segment.
        - In <tool_call>, the segment for the tool call.
        - In <answer>, the short explanation of why this segment is the most informative for solving the question and what missing evidence it provides based on time series analysis and your inherent knowledge about the time series, question domain and context.

        # CONSTRAINTS
        - Valid indices: 0 ≤ start ≤ end ≤ {total_ts_length - 1}, integers only.
        - Exactly one <think>, one <tool_call>, and one <answer>.
        - You must call "timeseries_zoom_in_tool" exactly once.

# MAKE YOUR SELECTION NOW:
        """
        return prompt
    # def _create_first_turn_controller_prompt_ECG_extended(self, question, segment_memory, ts):
    #     total_ts_length = segment_memory.ts_length

    #     # --- Prep & length ---
    #     ts = ts.squeeze() if ts.ndim > 1 else ts
    #     N = int(len(ts))

    #     # --- Extrema (ALL of them) ---
    #     if N < 3:
    #         minima, maxima = [], []
    #     else:
    #         order = int(0.1*N)
    #         mins = argrelextrema(ts, np.less_equal, order=order)[0]
    #         maxs = argrelextrema(ts, np.greater_equal, order=order)[0]
    #         minima = [(int(i), round(float(ts[i]), 4)) for i in mins]
    #         maxima = [(int(i), round(float(ts[i]), 4)) for i in maxs]

    #     extrema_info = ""
    #     if minima or maxima:
    #         extrema_info = (
    #             "## Key Points in the Time Series\n"
    #             f"- **Local Minima (index → value)**: {minima}\n"
    #             f"- **Local Maxima (index → value)**: {maxima}\n"
    #         )
    #     if len(maxima) >=3 and len(minima) >=2:
    #         example_segments = [
    #         f"[0, {maxima[0][0]}]",                                # from start to first peak
    #         f"[{maxima[1][0]}, {maxima[1][0] + 100}]",             # short window after first max
    #         f"[{minima[0][0]}, {minima[0][0] + 200}]",             # window around a trough
    #         f"[{maxima[2][0]}, {maxima[2][0] + 500}]",             # medium-length segment near 2nd peak
    #         f"[{minima[1][0]}, {minima[1][0] + 800}]"              # larger region starting from a trough
    #         ]
    #         example_segments_text = ", ".join(example_segments)
    #     else:
    #         example_segments_text = ""

    #     prompt = f"""# TASK
    #     You are a time-series analysis Controller. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.

    #     ## Instructions
    #     1. You receive the FULL time series embedding and the question.
    #     2. Review the question and analyze the full time series embedding, decide which segment is the most important to answer the question accurately.

    #     ## Inputs
    #     • Full time series embedding:
    #     {TS_PLACEHOLDER}
    #     • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
    #     • Question: {question}

    #     ## Output Format
    #     Your response should be in the following format:
    #     <think>...</think>
    #     <tool_call>
    #     {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
    #     </tool_call>
        
    #     ## Guidelines
    #     in the <think> section, you should provide the concise reasoning (1 sentence) for why you are retrieving this segment. be concise and to the point.
    #     in the <tool_call> section, you should provide the segment for the tool call.
    #     You can choose different segment sizes to help capture different aspects of the time series.
    #     For example: {example_segments_text} are all valid segments.
    #     You can use the key points in the time series for segment selection:
    #     {extrema_info}
        
        
    #     ## Constraints
    #     - Valid range for the segment is between 0 and {total_ts_length - 1}
    #     - start and end should be integers
    #     - You can call `timeseries_zoom_in_tool` ONCE in this round
    #     - Your output MUST include a <think> block (1 sentence) BEFORE the tool call.

    #     Only the above format is valid for this round.
    #     Make your decision now:
    #     """
    #     return prompt
    def _create_first_turn_controller_prompt_ECG_extended(self, question, segment_memory, ts):
        total_ts_length = segment_memory.ts_length

        # --- Prep & length ---
          # --- Prep & length ---
        ts = ts.squeeze() if ts is not None and ts.ndim > 1 else ts
        N = int(len(ts)) if ts is not None else int(segment_memory.ts_length)

        # --- Extrema (indices only) ---
        if ts is None or N < 3:
            minima_idx, maxima_idx = [], []
        else:
            from scipy.signal import argrelextrema
            order = max(1, int(0.1 * N))
            minima_idx = [int(i) for i in argrelextrema(ts, np.less_equal, order=order)[0].tolist()]
            maxima_idx = [int(i) for i in argrelextrema(ts, np.greater_equal, order=order)[0].tolist()]

        def _seg(lo, hi):
            lo = max(0, int(lo))
            hi = min(N - 1, int(hi))
            if lo > hi:
                lo, hi = hi, lo
            return f"[{lo}, {hi}]"

        # Build illustrative example segments from points (only if we have enough points)
        example_segments = []
        if len(maxima_idx) >= 3:
            # centered windows around peaks
            example_segments.append(_seg(maxima_idx[0] - 100, maxima_idx[0] + 100))   # small window around first peak
            example_segments.append(_seg(maxima_idx[1] - 20,       maxima_idx[1] + 300))   # short window after second peak
            example_segments.append(_seg(maxima_idx[2] - 50,       maxima_idx[2] + 500))   # medium window starting at third peak
            # from start to a key peak
            example_segments.append(_seg(0, maxima_idx[0]+150))                           # start to first peak

        if len(minima_idx) >= 2:
            # windows around troughs
            example_segments.append(_seg(minima_idx[0], minima_idx[0] + 200))
            example_segments.append(_seg(minima_idx[1], minima_idx[1] + 800))

        # Fallback examples if extrema are scarce
        if not example_segments:
            example_segments = [
                _seg(0, min(199, N - 1)),
                _seg(max(0, N // 2 - 100), min(N // 2 + 100, N - 1)),
                _seg(max(0, N - 600), N - 1),
            ]

        example_segments_text = ", ".join(example_segments)

        # Extrema indices block (indices only; no values)
        extrema_info = ""
        if minima_idx or maxima_idx:
            extrema_info = (
                "## Key Points in the Time Series (indices)\n"
                f"- Local minima indices: {minima_idx}\n"
                f"- Local maxima indices: {maxima_idx}\n"
            )

        prompt = f"""# TASK
        You are a time-series analysis Controller. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.

        ## Instructions
        1. You receive the FULL time series embedding and the question.
        2. Review the question and analyze the full time series embedding, decide which segment is the most important to answer the question accurately.

        ## Inputs
        • Full time series embedding:
        {TS_PLACEHOLDER}
        • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        • Question: {question}

        ## Output Format
        Your response should be in the following format:
        <think>...</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>
        
        ## Guidelines
        in the <think> section, you should provide the concise reasoning (1 sentence) for why you are retrieving this segment. be concise and to the point.
        in the <tool_call> section, you should provide the segment for the tool call.
        You can choose different segment sizes to help capture different aspects of the time series.
        You may use key points (indices) to define segments. For a point at index i, a centered window example is [i minus 100, i plus 100].
        key points for segment selection:
        {extrema_info}
        Examples of segments using the key points: {example_segments_text}
        
        
        ## Constraints
        - Valid range for the segment is between 0 and {total_ts_length - 1}
        - start and end should be integers
        - You can call `timeseries_zoom_in_tool` ONCE in this round
        - Your output MUST include a <think> block (1 sentence) BEFORE the tool call.

        Only the above format is valid for this round.
        Make your decision now:
        """
        return prompt

    def _create_first_turn_controller_prompt_ETI_extended(self, question: str, ts: np.ndarray):
        """
        First-turn controller prompt:
        - computes ALL local minima/maxima (order=3)
        - provides exact segment-size ranges (small/medium/large) in timesteps
        - keeps original format/constraints/tool-call style
        """
        # --- Prep & length ---
        ts = ts.squeeze() if ts.ndim > 1 else ts
        N = int(len(ts))

        # --- Extrema (ALL of them) ---
        if N < 3:
            minima, maxima = [], []
        else:
            order = int(0.1*N)
            mins = argrelextrema(ts, np.less_equal, order=order)[0]
            maxs = argrelextrema(ts, np.greater_equal, order=order)[0]
            minima = [(int(i), round(float(ts[i]), 4)) for i in mins]
            maxima = [(int(i), round(float(ts[i]), 4)) for i in maxs]

        extrema_info = ""
        if minima or maxima:
            extrema_info = (
                "### Key Points in the Time Series\n"
                f"- **Local Minima (index → value)**: {minima}\n"
                f"- **Local Maxima (index → value)**: {maxima}\n"
            )

        # --- Exact segment-size ranges (inclusive length = end - start + 1) ---
        # Percent bands: Small 5–15%, Medium 20–40%, Large ≥50%
        def _band(lo_pct, hi_pct=None):
            lo = max(1, int(round(lo_pct * N)))
            hi = N if hi_pct is None else max(lo, int(round(hi_pct * N)))
            return lo, hi

        small_lo,  small_hi  = _band(0.05, 0.15)
        med_lo,    med_hi    = _band(0.20, 0.40)
        large_lo,  large_hi  = _band(0.50, None)

        size_guidance = (
            "### Segment Size Guidance\n"
            "You can choose any segment size you want, from 10 steps up to the full series. Some example sizes are provided below:\n"
            f"- **Small**  (≈5–15%): {small_lo}, {small_hi} timesteps\n"
            f"- **Medium** (≈20–40%): {med_lo}, {med_hi} timesteps\n"
            f"- **Large**  (≥50%):   {large_lo}, {large_hi} timesteps\n"
        )

        # --- Build prompt ---
        prompt = f"""# TASK
        You are a time series analysis expert. Your task is to decide which time series segment an LLM needs to accurately answer a question related to a time series.

        ## Instructions
        1. You receive the FULL time series embedding and the question.
        2. Review the question and analyze the full time series embedding, then decide which segment is the most important to answer the question accurately.

        ## Full Time Series Embedding
        {TS_PLACEHOLDER}

        ## Current Status
        - **Time series length**: {N} timesteps
        - **Question**: {question}

    
        ## Output Format
        Your response must strictly follow:
        <think>...</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        ## Guidelines
        - In `<think>`, briefly justify your choice (1 short sentence).
        - In `<tool_call>`, specify the segment indices.
        - **Inclusive length** rule: `(end - start + 1)` must match the chosen size range.
        - Valid index range: [0, {N - 1}]
        - `start` and `end` must be integers.
        - Call `timeseries_zoom_in_tool` **once** in this round.
        - Always include the `<think>` block **before** the tool call.
        
        ## Information for selecting the segment
        {extrema_info}

        {size_guidance}

        Only the above format is valid for this round.
        Make your decision now:
        """
        return prompt

    def _create_controller_prompt_scratch(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
    ) -> str:
        """Create controller prompt with TS_PLACEHOLDER for injection"""
        
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join([
                f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)
            ])
        else:
            current_segments_text = "  - No segments provided yet"
        
        # Get coverage stats
        total_ts_length = segment_memory.ts_length
        # covered_steps = len(segment_memory.get_covered_timesteps())
        # coverage_pct = (covered_steps / total_ts_length) * 100 if total_ts_length > 0 else 0
        
        # Format previous answer
        if previous_reasoner_answer:
            answer_section = f"""
        ## LLM's Previous Answer
        ```
        {previous_reasoner_answer}
        ```
        Evaluate if the answer is accurate. If incomplete/uncertain, retrieve ONE additional segment.
        """
        else:
            answer_section = """
        ## LLM's Previous Answer
        No answer yet.
        """
            
        prompt = f"""# ROLE
        You are a controller for a time-series QA system. You must output **only** one of the two allowed formats below. Be terse.

        # Inputs
        - Full TS embedding:
        {TS_PLACEHOLDER}

        - Time series length: {total_ts_length} timesteps
        - Question: {question}
        - Segments provided so far:
        {current_segments_text}

        {answer_section}

        # Decision
        Choose exactly ONE:
        1) **Retrieve** one additional segment if any uncertainty remains.
        2) **Accept** the current answer only if it is fully supported by available segments.

        # Output Format (STRICT)
        You MUST output exactly one of these:

        **Retrieve**
        <think>One sentence (≤20 words) explaining what is missing.</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        **Accept**
        <think>One sentence (≤20 words) explaining why the answer is sufficient.</think>
        <answer>ACCEPT</answer>

        # Rules (MANDATORY)
        - No text outside the tags above (no greetings, no markdown, no bullets).
        - Use integers for [start, end], inclusive.
        - Range must satisfy 0 ≤ start < end ≤ {total_ts_length - 1}.
        - You can choose ANY segment length: from 10 steps up to the full series.
        - ts_seg MUST be two **integers**: [start, end]. **No math or variables** (bad: "[150, 150+50]").
        - Example: if you want a 10-step window, compute it yourself and output integers (good: "[381, 391]").
        - Call `timeseries_zoom_in_tool` at most once this round.
        - If unsure, **Retrieve**.

        # Examples
        Retrieve (good):
        <think>Key evidence about the initial jump is missing.</think>
        <tool_call>
        {{"name":"timeseries_zoom_in_tool","arguments":{{"ts_seg":[120,200]}}}}
        </tool_call>

        Invalid (bad): extra prose, multiple tool calls, or anything outside tags.

        Make your decision now."""

        return prompt
    
    def _create_controller_prompt_ETI(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
    ) -> str:
        """Create controller prompt with TS_PLACEHOLDER for injection"""
        
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join([
                f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)
            ])
        else:
            current_segments_text = "  - No segments provided yet"
        
        # Get coverage stats
        total_ts_length = segment_memory.ts_length
        # covered_steps = len(segment_memory.get_covered_timesteps())
        # coverage_pct = (covered_steps / total_ts_length) * 100 if total_ts_length > 0 else 0
        
        # Format previous answer
        if previous_reasoner_answer:
            answer_section = f"""
        ## LLM's Previous Answer
        ```
        {previous_reasoner_answer}
        ```

        Evaluate if the answer is accurate and if the available segments provide sufficient information to answer the question accurately. If the answer appears incomplete, uncertain, or potentially incorrect, consider retrieving additional segments.
        """
        else:
            answer_section = """
        ## LLM's Previous Answer
        No answer yet - this is the first round. You need to select initial segments that are most likely to contain information relevant to the question.
        """
            
        prompt = f"""# TASK
        You are a time series analysis expert. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.


        ## Instructions
        1. You receive the FULL time series embedding, the question, the segments that were given to the LLM so far, and the LLM's answer.
        2. Review the question and the LLM's answer, and analyze the full time series embedding.
        3. Evaluate if the LLM's answer is accurate and if the available segments provide sufficient information to answer the question accurately - you can try answering the question yourself to understad what is missing.
        4. Make ONE of two decisions:
        - **Retrieve an additional segment**: If you think that thew answer is not accurate and/or there is critical information missing, use the tool to get an additional segment of time series data for the LLM.
        - **Accept the LLM's answer as final**: If you think the LLM's answer is accurate and the information provided is sufficient.

        ## Full Time Series Embedding
        {TS_PLACEHOLDER}

        ## Current Status
        - **Time series length**: {total_ts_length} timesteps
        - **Question**: {question}
        - **Segments provided to the LLM so far**: 
        {current_segments_text}

        {answer_section}

        ## Output Format
        You MUST respond in ONE of the following two formats:

        **Option 1: Retrieve an additional segment**
        Make a tool call asking for a specific segment of the time series data: 
        <think> includes your consice reasoning (1 sentence) for why you are retrieving this segment and what information is missing </think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        **Option 2: Accept the LLM's answer as final**
        You should answer with the following format:
        <think> includes your concise reasoning (1 sentence) for why you are accepting the answer </think>
        <answer>ACCEPT</answer>

        ## Important Rules
        1. You can only call `timeseries_zoom_in_tool` ONCE per turn
        2. Specify segments as integers: [100, 200] means timesteps 100 to 200 inclusive
        4. Do NOT request segments outside valid range [0, {total_ts_length - 1}]
        5. If you output "ACCEPT", the process will immediately conclude with the LLM's current answer

        Make your decision now:"""

        return prompt
    
    def _create_controller_prompt_generic_mcq_full_ts(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
    ) -> str:
        """Create controller prompt with TS_PLACEHOLDER for injection"""
        
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join([
                f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)
            ])
        else:
            current_segments_text = "  - No segments provided yet"
        
        # Get coverage stats
        total_ts_length = segment_memory.ts_length
        # covered_steps = len(segment_memory.get_covered_timesteps())
        # coverage_pct = (covered_steps / total_ts_length) * 100 if total_ts_length > 0 else 0
        
        # Format previous answer
        if previous_reasoner_answer:
            answer_section = f"""
        ## LLM's Previous Answer
        ```
        {previous_reasoner_answer}
        ```

        Evaluate if the answer is accurate and if the available segments provide sufficient information to answer the question accurately. If the answer appears incomplete, uncertain, or potentially incorrect, consider retrieving additional segments.
        """
        else:
            answer_section = """
        ## LLM's Previous Answer
        No answer yet - this is the first round. You need to select initial segments that are most likely to contain information relevant to the question.
        """
            
        prompt = f"""# TASK
        You are a time series analysis expert. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.

        ## Instructions
        1. You receive the FULL time series embedding and the question.
        2. Review the question and analyze the full time series embedding, then decide which segment is the most important to answer the question accurately.

        ## Full Time Series Embedding
        {TS_PLACEHOLDER}

        ## Current Status
        - **Time series length**: {total_ts_length} timesteps
        - **Question**: {question}
        - **Segments provided to the LLM so far**: 
        {current_segments_text}

        {answer_section}

        ## Output Format
        Your response must strictly follow:
        <think>...</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        ## Guidelines
        - In `<think>`, briefly justify your choice (1 short sentence).
        - In `<tool_call>`, specify the segment indices.
        - **Inclusive length** rule: `(end - start + 1)` must match the chosen size range.
        - Valid index range: [0, {total_ts_length - 1}]
        - `start` and `end` must be integers.
        - Call `timeseries_zoom_in_tool` **once** in this round.
        - Always include the `<think>` block **before** the tool call.

        Make your decision now:"""

        return prompt
    
    def _create_controller_prompt_ETI_v2(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
    ) -> str:
        """Create controller prompt with TS_PLACEHOLDER for injection"""
        
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join([
                f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)
            ])
        else:
            current_segments_text = "  - No segments provided yet"
        
        # Get coverage stats
        total_ts_length = segment_memory.ts_length

        # Format previous answer section
        if previous_reasoner_answer:
            answer_section = f"""
            ## LLM's Previous Answer (Reasoner)
            The following is the reasoner's last answer, including its explicit uncertainty (if any):

            ```
            {previous_reasoner_answer}
            ```

            Pay special attention to any phrases such as:
            - "I'm not sure"
            - "I am uncertain"
            - "I would need more information"
            - "I am not confident about my answer"

            If the answer expresses uncertainty, you should usually retrieve an additional segment,
            **unless** you are confident that additional segments cannot change the answer
            (for example, if the question can already be answered definitively from the current segments).
            """
        else:
            answer_section = """
            ## LLM's Previous Answer (Reasoner)
            No answer yet – this is the first round.

            You must select initial segments that are most likely to contain the critical information
            needed to answer the question accurately.
            """
            
        prompt = f"""# ROLE
            You are the CONTROLLER – a time series analysis expert that decides which time series segments
            an LLM (the REASONER) should see in order to answer a question accurately.

            # TASK
            You receive:
            - The FULL time series embedding
            - The question
            - The list of segments already provided to the reasoner
            - The reasoner's latest answer (which may include explicit uncertainty)

            Your job is to decide whether to:
            - Retrieve an additional segment for the reasoner, or
            - Accept the reasoner's current answer as final.


            ## Instructions
            1. You receive the FULL time series embedding, the question, the segments that were given to the reasoner so far, and the reasoner's answer.
            2. Review the question and the reasoner's answer, and analyze the full time series embedding.
            3. Evaluate if the reasoner's answer is accurate and if the available segments provide sufficient information to answer the question accurately - you can try answering the question yourself to understad what is missing.
            4. If the reasoner expresses uncertainty and you believe additional information could meaningfully reduce that uncertainty or correct a wrong answer, you should **retrieve another segment**.
            5. Make ONE of two decisions:
            - **Retrieve an additional segment**:
            - If you suspect the reasoner's answer is incomplete, incorrect, or the reasoner explicitly says it is not sure / needs more information, and you believe another segment could help.
            - **Accept the reasoner's answer as final**:
            - If you believe the reasoner's answer is accurate and sufficiently supported by the segments already provided, or if additional segments are unlikely to change the reasoner's answer.

            ## Full Time Series Embedding
            {TS_PLACEHOLDER}

            ## Current Status
            - **Time series length**: {total_ts_length} timesteps
            - **Question**: {question}
            - **Segments provided to the reasoner so far**:
            {current_segments_text}

            {answer_section}

            ## Output Format (STRICT)

            You MUST respond in ONE of the following two formats:

            **Option 1: Retrieve an additional segment**
            Use this when you believe the reasoner's answer is likely wrong, incomplete, or explicitly uncertain,
            and another segment could resolve this.

            <think>One concise sentence explaining WHY you are requesting another segment and what information is missing</think>
            <tool_call>
            {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
            </tool_call>

            **Option 2: Accept the reasoner's answer as final**
            Use this when you believe the answer is accurate and the provided segments are sufficient.

            <think>One concise sentence explaining WHY you are accepting the answer</think>
            <answer>ACCEPT</answer>

            ## Important Rules
            1. You can only call `timeseries_zoom_in_tool` ONCE per turn.
            2. Specify segments as integers: [100, 200] means timesteps 100 to 200 inclusive.
            3. Do NOT request segments outside the valid range [0, {total_ts_length - 1}].
            4. If you output "ACCEPT", the process will immediately conclude with the reasoner's current answer.

            Make your decision now:
            """

        return prompt

    def _create_controller_prompt_ECG(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
    ) -> str:
        """STRICT controller prompt that outputs either a single <tool_call> block or ACCEPT."""
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join(
                [f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)]
            )
        else:
            current_segments_text = "  - No segments provided yet"

        total_ts_length = segment_memory.ts_length

        if previous_reasoner_answer:
            answer_section = f"""## Reasoner LLM's Previous Answer
            {previous_reasoner_answer}
            """
        else:
            answer_section = """## Reasoner LLM's Previous Answer
            No answer yet (first round). Select initial segment(s) most likely to contain information relevant to the question.
            """

        prompt = f"""# ROLE
        You are a time-series analysis Controller. Decide whether to (a) retrieve ONE additional time series segment for the reasoner LLM, or (b) accept the current answer.

        # INPUTS
        • Full time series embedding:
        {TS_PLACEHOLDER}
        • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        • Question: {question}
        • Segments provided so far:
        {current_segments_text}

        {answer_section}

        # DECISION POLICY (think silently; do NOT reveal chain-of-thought)
        - Retrieve a segment if the answer is incomplete/uncertain/likely incorrect OR critical evidence is missing.
        - Otherwise accept.

        # OUTPUT — EXACTLY ONE OF THE TWO TEMPLATES BELOW AND NOTHING ELSE

        TEMPLATE A — RETRIEVE (exactly these tags and JSON; one sentence reason ≤ 20 words):
        <think>concise reason for what’s missing and why this segment resolves it</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        Constraints for TEMPLATE A:
        - Only ONE <tool_call>.
        - start and end are integers, 0 ≤ start ≤ end ≤ {total_ts_length - 1}.
        - Choose the single most informative segment. Avoid duplicates unless necessary.

        TEMPLATE B — ACCEPT (exactly these tags; one sentence reason ≤ 20 words):
        <think>concise reason for why the evidence is sufficient</think>
        <answer>ACCEPT</answer>

        # HARD RULES
        - Output must match one template exactly. No markdown, no extra paragraphs, no second <tool_call>, no commentary outside tags.
        - The <think> text must be ≤ 30 words. No step-by-step reasoning.
        - The JSON inside <tool_call> must be valid (double quotes).
        - If you output <answer>ACCEPT</answer>, do NOT include a <tool_call>.

        # MAKE YOUR DECISION NOW:
        """
        return prompt
    
    def _create_controller_prompt_ECG_v2(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
    ) -> str:
        """STRICT controller prompt that outputs either a single <tool_call> block or ACCEPT."""
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join(
                [f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)]
            )
        else:
            current_segments_text = "  - No segments provided yet"

        total_ts_length = segment_memory.ts_length

        if previous_reasoner_answer:
            answer_section = f"""## Reasoner's Previous Answer
            {previous_reasoner_answer}
            """
        else:
            answer_section = """## Reasoner's Previous Answer
            No answer yet (first round). Select initial segment(s) most likely to contain information relevant to the question.
            """

        prompt = f"""# ROLE
        You are a time-series optimizer Controller.
        You assist a REASONER that tries to answer a question about a time series. 
        The REASONER only sees the segments of the time series that YOU choose.

        # YOUR GOAL
        Your goal is to determine whether the REASONER already has enough information to answer the question accurately.
        - If the current information is sufficient - ACCEPT (do NOT choose a new segment).
        - If the current information is insufficient, unclear, or missing key evidence - RETRIEVE exactly one additional segment that will most help the REASONER answer the question accurately.

        You NEVER retrieve when you accept.
        You ALWAYS retrieve exactly one segment when you decide more evidence is needed.

        # WHAT YOU ARE GIVEN
        - The full time series
        - The question
        - The list of time series segments you have already given to the REASONER
        - The REASONER's latest answer and reasoning

        Use these to infer whether critical evidence is present, missing, or contradictory.

        # HOW TO EVALUATE THE REASONER
        When reading the REASONER's answer and reasoning:
        - Check whether the answer is accurate, logically consistent, and supported by the segments.
        - Check whether the REASONER expresses uncertainty or lacks necessary information.
        - Identify what information is missing based on the question, the time series, and your inherent domain knowledge and understanding of the time series.

        # INPUTS
        • Full time series embedding:
        {TS_PLACEHOLDER}
        • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        • Question: {question}
        • Segments provided to the REASONER so far:
        {current_segments_text}

        {answer_section}

        # DECISION POLICY
        - RETRIEVE a new segment if the REASONER's answer is incomplete, uncertain, likely incorrect, or if key evidence is missing.
        Choose exactly one segment [start, end] that is most informative given what is already known.
        - ACCEPT if the REASONER's answer is accurate, well-supported by the provided segments, and no further segment is needed.

        # OUTPUT - EXACTLY ONE OF THE TWO TEMPLATES BELOW:

        TEMPLATE A - RETRIEVE (one-sentence reasoning ≤ 20 words, exactly one tool_call):
        <think>Your reasoning for why more evidence is needed and what information is missing.</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>
        <answer>RETRIEVE
        Short explanation of why this segment is the most informative and what missing evidence it provides.
        </answer>

        Constraints for TEMPLATE A:
        - Exactly ONE <tool_call>.
        - start and end are integers, 0 ≤ start ≤ end ≤ {total_ts_length - 1}.
        - Choose the single most informative NEW segment (avoid duplicates unless needed).

        TEMPLATE B - ACCEPT (one-sentence reasoning ≤ 20 words):
        <think>Your reasoning for why the current evidence is sufficient for an accurate answer.</think>
        <answer>ACCEPT
        Short explanation of why no additional segment would improve the REASONER's answer.
        </answer>

        # HARD RULES
        - Output must match one template exactly. No markdown, no extra paragraphs, no commentary outside tags.
        - Exactly one <think> and exactly one <answer>.
        - <think> must be ≤ 30 words. No step-by-step reasoning.
        - JSON inside <tool_call> must be valid (double quotes).
        - If you output <answer>ACCEPT</answer>, you must NOT include any <tool_call>.

        # MAKE YOUR DECISION NOW:
        """
        return prompt


    # def _create_controller_prompt_ECG_extended(
    #     self,
    #     question,
    #     current_segments,
    #     segment_memory: SegmentMemory,
    #     previous_reasoner_answer: Optional[str] = None,
    #     ts: np.ndarray = None,
    # ) -> str:
    #     """STRICT controller prompt that outputs either a single <tool_call> block or ACCEPT."""
       
    #    # --- Prep & length ---
    #     ts = ts.squeeze() if ts.ndim > 1 else ts
    #     N = int(len(ts))

    #     # --- Extrema (ALL of them) ---
    #     if N < 3:
    #         minima, maxima = [], []
    #     else:
    #         order = int(0.1*N)
    #         mins = argrelextrema(ts, np.less_equal, order=order)[0]
    #         maxs = argrelextrema(ts, np.greater_equal, order=order)[0]
    #         minima = [(int(i), round(float(ts[i]), 4)) for i in mins]
    #         maxima = [(int(i), round(float(ts[i]), 4)) for i in maxs]

    #     extrema_info = ""
    #     if minima or maxima:
    #         extrema_info = (
    #             "## Key Points in the Time Series\n"
    #             f"- **Local Minima (index → value)**: {minima}\n"
    #             f"- **Local Maxima (index → value)**: {maxima}\n"
    #         )
    #     if len(maxima) >=3 and len(minima) >=2:
    #         example_segments = [
    #         f"[0, {maxima[0][0]}]",                                # from start to first peak
    #         f"[{maxima[1][0]}, {maxima[1][0] + 100}]",             # short window after first max
    #         f"[{minima[0][0]}, {minima[0][0] + 200}]",             # window around a trough
    #         f"[{maxima[2][0]}, {maxima[2][0] + 500}]",             # medium-length segment near 2nd peak
    #         f"[{minima[1][0]}, {minima[1][0] + 800}]"              # larger region starting from a trough
    #         ]
    #         example_segments_text = ", ".join(example_segments)
    #     else:
    #         example_segments_text = ""

    #     # Format current segments
    #     if current_segments:
    #         current_segments_text = "\n".join(
    #             [f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
    #             for i, (start, end) in enumerate(current_segments)]
    #         )
    #     else:
    #         current_segments_text = "  - No segments provided yet"

    #     total_ts_length = segment_memory.ts_length

    #     if previous_reasoner_answer:
    #         answer_section = f"""## Reasoner LLM's Previous Answer
    #         {previous_reasoner_answer}
    #         """
    #     else:
    #         answer_section = """## Reasoner LLM's Previous Answer
    #         No answer yet (first round). Select initial segment(s) most likely to contain information relevant to the question.
    #         """

    #     prompt = f"""# ROLE
    #     You are a time-series analysis Controller. Decide whether to (a) retrieve ONE additional time series segment for the reasoner LLM, or (b) accept the current answer.

    #     # INPUTS
    #     • Full time series embedding:
    #     {TS_PLACEHOLDER}
    #     • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
    #     • Question: {question}
    #     • Segments provided so far:
    #     {current_segments_text}

    #     {answer_section}

    #     # DECISION POLICY (think silently; do NOT reveal chain-of-thought)
    #     - Retrieve a segment if the answer is incomplete/uncertain/likely incorrect OR critical evidence is missing.
    #     - Otherwise accept.

    #     # OUTPUT — EXACTLY ONE OF THE TWO TEMPLATES BELOW AND NOTHING ELSE

    #     TEMPLATE A — RETRIEVE (exactly these tags and JSON; one sentence reason ≤ 20 words):
    #     <think>concise reason for what’s missing and why this segment resolves it</think>
    #     <tool_call>
    #     {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
    #     </tool_call>

    #     Constraints for TEMPLATE A:
    #     - Only ONE <tool_call>.
    #     - start and end are integers, 0 ≤ start ≤ end ≤ {total_ts_length - 1}.
    #     - Choose the single most informative segment. Avoid duplicates unless necessary.
    #     - You can choose different segment sizes to help capture different aspects of the time series.
    #     for example: {example_segments_text} are all valid segments.
    #     - You can use the key points in the time series for segment selection:
    #     {extrema_info}

    #     TEMPLATE B — ACCEPT (exactly these tags; one sentence reason ≤ 20 words):
    #     <think>concise reason for why the evidence is sufficient</think>
    #     <answer>ACCEPT</answer>

    #     # HARD RULES
    #     - Output must match one template exactly. No markdown, no extra paragraphs, no second <tool_call>, no commentary outside tags.
    #     - The <think> text must be ≤ 30 words. No step-by-step reasoning.
    #     - The JSON inside <tool_call> must be valid (double quotes).
    #     - If you output <answer>ACCEPT</answer>, do NOT include a <tool_call>.

    #     # MAKE YOUR DECISION NOW:
    #     """
    #     return prompt
    def _create_controller_prompt_ECG_extended(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
        ts: np.ndarray = None,
    ) -> str:
        """STRICT controller prompt that outputs either a single <tool_call> block or ACCEPT."""
       
        # --- Prep & length ---
        ts = ts.squeeze() if ts is not None and ts.ndim > 1 else ts
        N = int(len(ts)) if ts is not None else int(segment_memory.ts_length)

        # --- Extrema (indices only) ---
        if ts is None or N < 3:
            minima_idx, maxima_idx = [], []
        else:
            from scipy.signal import argrelextrema
            order = max(1, int(0.1 * N))
            minima_idx = [int(i) for i in argrelextrema(ts, np.less_equal, order=order)[0].tolist()]
            maxima_idx = [int(i) for i in argrelextrema(ts, np.greater_equal, order=order)[0].tolist()]

        def _seg(lo, hi):
            lo = max(0, int(lo))
            hi = min(N - 1, int(hi))
            if lo > hi:
                lo, hi = hi, lo
            return f"[{lo}, {hi}]"

        # Build illustrative example segments from points (only if we have enough points)
        example_segments = []
        if len(maxima_idx) >= 3:
            # centered windows around peaks
            example_segments.append(_seg(maxima_idx[0] - 100, maxima_idx[0] + 100))   # small window around first peak
            example_segments.append(_seg(maxima_idx[1] - 20,       maxima_idx[1] + 300))   # short window after second peak
            example_segments.append(_seg(maxima_idx[2] - 50,       maxima_idx[2] + 500))   # medium window starting at third peak
            # from start to a key peak
            example_segments.append(_seg(0, maxima_idx[0]+150))                           # start to first peak

        if len(minima_idx) >= 2:
            # windows around troughs
            example_segments.append(_seg(minima_idx[0], minima_idx[0] + 200))
            example_segments.append(_seg(minima_idx[1], minima_idx[1] + 800))

        # Fallback examples if extrema are scarce
        if not example_segments:
            example_segments = [
                _seg(0, min(199, N - 1)),
                _seg(max(0, N // 2 - 100), min(N // 2 + 100, N - 1)),
                _seg(max(0, N - 600), N - 1),
            ]

        example_segments_text = ", ".join(example_segments)

        # Extrema indices block (indices only; no values)
        extrema_info = ""
        if minima_idx or maxima_idx:
            extrema_info = (
                "## Key Points in the Time Series (indices)\n"
                f"- Local minima indices: {minima_idx}\n"
                f"- Local maxima indices: {maxima_idx}\n"
            )

        # Format current segments
        if current_segments:
            current_segments_text = "\n".join(
                [f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)]
            )
        else:
            current_segments_text = "  - No segments provided yet"

        total_ts_length = segment_memory.ts_length

        if previous_reasoner_answer:
            answer_section = f"""## Reasoner LLM's Previous Answer
            {previous_reasoner_answer}
            """
        else:
            answer_section = """## Reasoner LLM's Previous Answer
            No answer yet (first round). Select initial segment(s) most likely to contain information relevant to the question.
            """

        prompt = f"""# ROLE
        You are a time-series analysis Controller. Decide whether to (a) retrieve ONE additional time series segment for the reasoner LLM, or (b) accept the current answer.

        # INPUTS
        • Full time series embedding:
        {TS_PLACEHOLDER}
        • Time series length: {total_ts_length} (valid indices: 0 to {total_ts_length - 1})
        • Question: {question}
        • Segments provided so far:
        {current_segments_text}

        {answer_section}

        # DECISION POLICY (think silently; do NOT reveal chain-of-thought)
        - Retrieve a segment if the answer is incomplete/uncertain/likely incorrect OR critical evidence is missing.
        - Otherwise accept.

        # OUTPUT — EXACTLY ONE OF THE TWO TEMPLATES BELOW AND NOTHING ELSE

        TEMPLATE A — RETRIEVE (exactly these tags and JSON; one sentence reason ≤ 20 words):
        <think>concise reason for what’s missing and why this segment resolves it</think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        Constraints for TEMPLATE A:
        - Only ONE <tool_call>.
        - start and end are integers, 0 ≤ start ≤ end ≤ {total_ts_length - 1}.
        - Choose the single most informative segment. Avoid duplicates unless necessary.
        - You can choose different segment sizes to help capture different aspects of the time series.
        - You may use key points (indices) to define segments. For a point at index i, a centered window example is [i minus 100, i plus 100].
        - key points for segment selection:
        {extrema_info}
        - Examples of segments using the key points: {example_segments_text}

        TEMPLATE B — ACCEPT (exactly these tags; one sentence reason ≤ 20 words):
        <think>concise reason for why the evidence is sufficient</think>
        <answer>ACCEPT</answer>

        # HARD RULES
        - Output must match one template exactly. No markdown, no extra paragraphs, no second <tool_call>, no commentary outside tags.
        - The <think> text must be ≤ 30 words. No step-by-step reasoning.
        - The JSON inside <tool_call> must be valid (double quotes).
        - If you output <answer>ACCEPT</answer>, do NOT include a <tool_call>.

        # MAKE YOUR DECISION NOW:
        """
        return prompt

    
    def _create_controller_prompt_ETI_extended(
        self,
        question,
        current_segments,
        segment_memory: SegmentMemory,
        previous_reasoner_answer: Optional[str] = None,
        ts: np.ndarray = None,
    ) -> str:
        """Create controller prompt with TS_PLACEHOLDER for injection"""
        
        # Format current segments
        if current_segments:
            current_segments_text = "\n".join([
                f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)
            ])
        else:
            current_segments_text = "  - No segments provided yet"
        
        # Get coverage stats
        ts = ts.squeeze() if ts.ndim > 1 else ts
        N = int(len(ts))
        if N < 3:
            minima, maxima = [], []
        else:
            order = int(0.1*N)
            mins = argrelextrema(ts, np.less_equal, order=order)[0]
            maxs = argrelextrema(ts, np.greater_equal, order=order)[0]
            minima = [(int(i), round(float(ts[i]), 4)) for i in mins]
            maxima = [(int(i), round(float(ts[i]), 4)) for i in maxs]

        extrema_info = ""
        if minima or maxima:
            extrema_info = (
                "### Key Points in the Time Series\n"
                f"- **Local Minima (index → value)**: {minima}\n"
                f"- **Local Maxima (index → value)**: {maxima}\n"
            )
        
        # --- Exact segment-size ranges (inclusive length = end - start + 1) ---
        def _band(lo_pct, hi_pct=None):
            lo = max(1, int(round(lo_pct * N)))
            hi = N if hi_pct is None else max(lo, int(round(hi_pct * N)))
            return lo, hi

        small_lo, small_hi = _band(0.05, 0.15)
        med_lo,   med_hi   = _band(0.20, 0.40)
        large_lo, large_hi = _band(0.50, None)

        size_guidance = (
            "### Segment Size Guidance\n"
            "You can choose any segment size you want, from 10 steps up to the full series. Some example sizes are provided below:\n"
            f"- **Small**  (≈5–15%): {small_lo}–{small_hi} timesteps\n"
            f"- **Medium** (≈20–40%): {med_lo}–{med_hi} timesteps\n"
            f"- **Large**  (≥50%):   {large_lo}–{large_hi} timesteps\n"
        )

        # Format previous answer
        if previous_reasoner_answer:
            answer_section = f"""
        ## LLM's Previous Answer
        ```
        {previous_reasoner_answer}
        ```

        Evaluate if the answer is accurate and if the available segments provide sufficient information to answer the question accurately. If the answer appears incomplete, uncertain, or potentially incorrect, consider retrieving additional segments.
        """
        else:
            answer_section = """
        ## LLM's Previous Answer
        No answer yet - this is the first round. You need to select initial segments that are most likely to contain information relevant to the question.
        """
            
        prompt = f"""# TASK
        You are a time series analysis expert. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.


        ## Instructions
        1. You receive the FULL time series embedding, the question, the segments that were given to the LLM so far, and the LLM's answer.
        2. Review the question and the LLM's answer, and analyze the full time series embedding.
        3. Evaluate if the LLM's answer is accurate and if the available segments provide sufficient information to answer the question accurately - you can try answering the question yourself to understad what is missing.
        4. Make ONE of two decisions:
        - **Retrieve an additional segment**: If you think that thew answer is not accurate and/or there is critical information missing, use the tool to get an additional segment of time series data for the LLM.
        - **Accept the LLM's answer as final**: If you think the LLM's answer is accurate and the information provided is sufficient.

        ## Full Time Series Embedding
        {TS_PLACEHOLDER}

        ## Current Status
        - **Time series length**: {N} timesteps
        - **Question**: {question}
        - **Segments provided to the LLM so far**: 
        {current_segments_text}

        {answer_section}

        ## Information for selecting the next segment (based on the full time series)
        {extrema_info}
        {size_guidance}

        ## Output Format
        You MUST respond in ONE of the following two formats:

        **Option 1: Retrieve an additional segment**
        Make a tool call asking for a specific segment of the time series data: 
        <think> includes your consice reasoning (1 sentence) for why you are retrieving this segment and what information is missing </think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        **Option 2: Accept the LLM's answer as final**
        You should answer with the following format:
        <think> includes your concise reasoning (1 sentence) for why you are accepting the answer </think>
        <answer>ACCEPT</answer>

        ## Important Rules
        1. You can only call `timeseries_zoom_in_tool` ONCE per turn
        2. Specify segments as integers: [100, 200] means timesteps 100 to 200 inclusive
        4. Do NOT request segments outside valid range [0, {N - 1}]
        5. If you output "ACCEPT", the process will immediately conclude with the LLM's current answer

        Make your decision now:"""

        return prompt
    
    def _create_reasoner_prompt_ETI(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""# ROLE
        You are the REASONER. Analyze ONLY the given time-series segments and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences (≤80 words total) describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter: A, B, C, or D.]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty (consicely. i.e I'm not sure about the answer because...).
        - In <answer>:
        - first line is exactly A, B, C, or D.

        # Time Series Segments
        {segments_section}

        # Question
        {question}

       # Good example (confident):
        <think>Seg 1 sustains a higher mean (~0.78) than Seg 2 (~0.64) with a clear plateau at t=40–90; variance is lower, supporting stability in Seg 1.</think>
        <answer>
        B
        </answer>

        # Good example (uncertain but decisive):
        <think>Seg 2 shows a brief rise at t=150–170 but coverage is sparse after t=200; Seg 3’s mean is slightly higher yet volatile. Evidence is limited, but Seg 3 is most consistent with the described pattern.</think>
        <answer>
        C
        </answer>

        """

    def _create_reasoner_prompt_generic_MCQ(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""# ROLE
        You are the REASONER. Analyze ONLY the given time-series segments and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences (≤80 words total) describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter: A, B, C, or D.]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty (consicely. i.e I'm not sure about the answer because...).
        - In <answer>:
        - first line is exactly A, B, C, or D.

        # Time Series Segments
        {segments_section}

        # Question
        {question}

        # Good example (confident):
        <think>Seg 1 sustains a higher mean (~0.78) than Seg 2 (~0.64) with a clear plateau at t=40–90; variance is lower, supporting stability in Seg 1.</think>
        <answer>
        B
        </answer>

        # Good example (uncertain but decisive):
        <think>Seg 2 shows a brief rise at t=150–170 but coverage is sparse after t=200; Seg 3’s mean is slightly higher yet volatile. Evidence is limited, but Seg 3 is most consistent with the described pattern.</think>
        <answer>
        C
        </answer>
        """
    
    def _create_reasoner_prompt_generic_MCQ_v3(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""
        You are a time series expert. Analyze ONLY the given time series data and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter: A, B, C, or D.]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty (consicely. i.e I'm not sure about the answer because...).
        - In <answer>:
        - first line is exactly A, B, C, or D.

        # Time Series Segments
        {segments_section}

        # Question
        {question}
        """
    
    def _create_reasoner_prompt_generic_MCQ_v5(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""
        You are a time series expert. Analyze ONLY the given time series data and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter: A or B]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty (consicely. i.e I'm not sure about the answer because...).
        - In <answer>:
        - first line is exactly A or B.

        # Time Series Segments
        {segments_section}

        # Question
        {question}
        """
    def _create_reasoner_prompt_generic_MCQ_v4(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""
        You are a time series expert. Analyze ONLY the given time series data and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter from the options provided in the question.]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty (consicely. i.e I'm not sure about the answer because...).
        - In <answer>:
        - first line is exactly one letter from the options provided in the question.

        # Time Series Segments
        {segments_section}

        # Question
        {question}
        """

    def _create_reasoner_prompt_generic_MCQ_v2(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        
        if segments_data:
            segments_text = []
            segment_text_placeholder = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n")
                segment_text_placeholder.append(f"{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
            segment_text_placeholder = "\n".join(segment_text_placeholder)
        else:
            segments_section = f"### Full Time Series"
            segment_text_placeholder = f"{TS_PLACEHOLDER}\n"
    
        return f"""{segment_text_placeholder}
        {question}

        segments you got:
        {segments_section}
        """

    def _create_reasoner_prompt_generic_TF(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""# ROLE
        You are the REASONER. Analyze ONLY the given time-series segments and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences (≤80 words total) describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter: A or B.]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty (consicely. i.e I'm not sure about the answer because...).
        - In <answer>:
        - first line is exactly A or B.

        # Time Series Segments
        {segments_section}

        # Question
        {question}

        # Good example (confident):
        <think>Seg 1 sustains a higher mean (~0.78) than Seg 2 (~0.64) with a clear plateau at t=40–90; variance is lower, supporting stability in Seg 1.</think>
        <answer>
        B
        </answer>

        # Good example (uncertain but decisive):
        <think>Seg 2 shows a brief rise at t=150–170 but coverage is sparse after t=200; Seg 3’s mean is slightly higher yet volatile. Evidence is limited, but Seg 3 is most consistent with the described pattern.</think>
        <answer>
        A
        </answer>
        """

    def _create_reasoner_prompt_generic_TF_v2(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        return f"""# ROLE
        You are the REASONER. Analyze ONLY the given time-series segments and answer the question.

        # Output Schema (STRICT)
        <think>One–two sentences (≤80 words total) describing your reasoning and how you came to the answer.</think>
        <answer>
        [Direct answer ONLY - the first line must be exactly one letter: A or B.]
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        - In <think>, explain you reasoning, how you came to the answer, and reference segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer.
        - In <answer>:
        - first line is exactly A or B.

        # Time Series Segments
        {segments_section}

        # Question
        {question}

        # Good example (confident):
        <think>Seg 1 sustains a higher mean (~0.78) than Seg 2 (~0.64) with a clear plateau at t=40–90; variance is lower, supporting stability in Seg 1.</think>
        <answer>
        B
        </answer>

        # Good example (uncertain but decisive):
        <think>Seg 2 shows a brief rise at t=150–170 but coverage is sparse after t=200; Seg 3’s mean is slightly higher yet volatile. Evidence is limited, but Seg 3 is most consistent with the described pattern.</think>
        <answer>
        A
        </answer>
        """

    def _create_reasoner_prompt_ETI_v2(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
    ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""

        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(
                    f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n"
                )
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"

        return f"""# ROLE
        You are the REASONER – an expert in time series analysis and reasoning.
        You are given time-series segments selected by a controller and a related multiple-choice question (A, B, C, or D).
        Your job is to analyze ONLY the provided segments and answer the question.
        If you do not have enough information, you must still choose the most likely option (A–D),
        and you must briefly say you are not sure and explain why. This helps improve the controller.

        # OUTPUT FORMAT (STRICT)
        You MUST output exactly two XML-style blocks and NOTHING ELSE:
        <think>One–two sentences with your reasoning</think>
        <answer>
        A, B, C, or D        <-- FIRST line: exactly one letter
        (optional, only if you are not sure) a second line briefly explaining your uncertainty
        </answer>

        # MANDATORY RULES (VIOLATION = INVALID / BAD REWARD)
        - The FIRST line inside <answer> MUST be EXACTLY ONE of: A, B, C, or D.
        - If you are uncertain or the evidence is weak, you MUST:
        - still choose the most likely option as the FIRST line (A/B/C/D), and
        - add a SECOND line explaining your uncertainty using phrases like:
            "I'm not sure", "I would need more information", "I am uncertain because ...".
        - If you are confident, include ONLY the first line (A/B/C/D) and nothing else inside <answer>.

        # TEMPLATE EXAMPLES (FORMAT ONLY — NOT RELATED TO THE QUESTION)

        ## Uncertain but decisive
        <think>REASONING</think>
        <answer>
        B
        I'm not sure about my answer because ... I would need more information about ... to answer confidently.
        </answer>

        <think>REASONING</think>
        <answer>
        D
        I'm uncertain because ... I need more information about ...
        </answer>

        ## Confident
        <think>REASONING</think>
        <answer>
        A
        </answer>

        <think>REASONING</think>
        <answer>
        C
        </answer>

        # TIME SERIES SEGMENTS
        {segments_section}

        # QUESTION
        {question}
        """

    
    def _create_reasoner_prompt_ECG(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # list of (start, end)
    ) -> str:
        """STRICT reasoner prompt: ONLY <think> and <answer>."""
        # Build segments section with a TS placeholder per segment
        if segments_data:
            parts = []
            for i, (start, end) in enumerate(segments_data, 1):
                parts.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(parts)
        else:
            segments_section =f"### Full Time Series\n{TS_PLACEHOLDER}\n"

        return f"""# ROLE
        You are the REASONER. Analyze the given time-series segments and answer the question.

        # OUTPUT FORMAT (STRICT)
        You MUST output exactly two XML-style blocks and NOTHING ELSE:
        <think>One–two sentences (≤80 words total) explaining your reasoning, referencing segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200"). No lists, no math blocks.</think>
        <answer>
        A single uppercase letter on its own line: A, B, C, or D
        </answer>

        # MANDATORY RULES
        - No text outside <think> and <answer>.
        - <think> must be ≤ 80 words. Keep it high-level; no step-by-step chain-of-thought.
        - <answer> must contain exactly one of A/B/C/D on the FIRST and ONLY content line.
        - If evidence is insufficient, pick the most likely option and reflect uncertainty concisely in <think> (not in <answer>).

        # DECISION POLICY
        - Evaluate the evidence across all segments.
        - If data are insufficient, briefly explain uncertainty (e.g. I'm not sure about the answer because...) in <think> and still select your best guess in <answer>.

        # TIME SERIES SEGMENTS
        {segments_section}

        # QUESTION
        {question}

        # EXAMPLES (for format only — NOT related to the question)
        Below are example answers showing the correct format. These examples are not related to the current question.

        ## Example A (clear signal pattern):
        <think>Seg 1 shows regular P–QRS–T complexes with uniform intervals and stable baseline, consistent with normal sinus rhythm (A).</think>
        <answer>
        A
        </answer>

        ## Example B (confident classification):
        <think>Seg 1 has a higher mean (~0.78) and lower variance than Seg 2; sustained plateau at t=40–90 supports option B.</think>
        <answer>
        B
        </answer>

        ## Example C (uncertain but decisive):
        <think>Seg 2 shows a rise at t=150–170 but coverage after t=200 is sparse; Seg 3 is slightly higher yet volatile. Most consistent with C.</think>
        <answer>
        C
        </answer>

        ## Example D (alternative plausible reasoning):
        <think>Seg 3 reveals irregular QRS intervals with baseline noise and missing P waves; overall morphology matches the pattern described in D.</think>
        <answer>
        D
        </answer>
        """ 
    
    def _create_reasoner_prompt_ECG_QA_VERIFY(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # list of (start, end)
    ) -> str:
        """STRICT reasoner prompt: ONLY <think> and <answer>."""
        # Build segments section with a TS placeholder per segment
        if segments_data:
            parts = []
            for i, (start, end) in enumerate(segments_data, 1):
                parts.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(parts)
        else:
            segments_section =f"### Full Time Series\n{TS_PLACEHOLDER}\n"

        return f"""# ROLE
        You are the REASONER. Analyze the given time-series segments and answer the question.

        # OUTPUT FORMAT (STRICT)
        You MUST output exactly two XML-style blocks and NOTHING ELSE:
        <think>One–two sentences (≤80 words total) explaining your reasoning, referencing segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200"). No lists, no math blocks.</think>
        <answer>
        yes or no
        </answer>

        # MANDATORY RULES
        - No text outside <think> and <answer>.
        - <think> must be ≤ 80 words. Keep it high-level; no step-by-step chain-of-thought.
        - <answer> must contain exactly one of yes or no on the FIRST and ONLY content line.
        - If evidence is insufficient, pick the most likely option and reflect uncertainty concisely in <think> (not in <answer>).

        # DECISION POLICY
        - Evaluate the evidence across all segments.
        - If data are insufficient, briefly explain uncertainty (e.g. I'm not sure about the answer because...) in <think> and still select your best guess in <answer>.

        # TIME SERIES SEGMENTS
        {segments_section}

        # QUESTION
        {question}

        """ 
    def _create_reasoner_prompt_ECG_QA_VERIFY_new(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # list of (start, end)
    ) -> str:
        """STRICT reasoner prompt: ONLY <think> and <answer>."""
        # Build segments section with a TS placeholder per segment
        if segments_data:
            parts = []
            for i, (start, end) in enumerate(segments_data, 1):
                parts.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(parts)
        else:
            segments_section =f"### Full Time Series\n{TS_PLACEHOLDER}\n"

        return f"""# ROLE
        You are the REASONER. Analyze the given time-series segments and answer the question.

        # OUTPUT FORMAT (STRICT)
        You MUST output exactly two XML-style blocks and NOTHING ELSE:
        <think>One–two sentences (≤80 words total) explaining your reasoning, referencing segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200"). No lists, no math blocks.</think>
        <answer>
        yes or no
        </answer>

        # TIME SERIES SEGMENTS
        {segments_section}

        # QUESTION
        {question}

        """ 
    
    def _create_reasoner_prompt_ECG_QA_VERIFY_v2(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # list of (start, end)
    ) -> str:
        """STRICT reasoner prompt: ONLY <think> and <answer>."""
        # Build segments section with a TS placeholder per segment
        if segments_data:
            parts = []
            for i, (start, end) in enumerate(segments_data, 1):
                parts.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(parts)
        else:
            segments_section =f"### Full Time Series\n{TS_PLACEHOLDER}\n"

        return f"""# ROLE
        You are the REASONER – an expert in time series analysis and reasoning.
        You are given time-series segments selected by a controller and a related question.
        Your job is to analyze ONLY the provided segments and answer the question.
        If you do not have enough information, you must say you are not sure and briefly explain why. This helps improve the controller.

        # OUTPUT FORMAT (STRICT)
        You MUST output exactly two XML-style blocks and NOTHING ELSE:
        <think>One–two sentences (≤80 words total) with your reasoning</think>
        <answer>
        yes or no
        (optional, only if you are not sure) a second line briefly explaining your uncertainty
        </answer>

        # MANDATORY RULES
        - No text outside <think> and <answer>.
        - <think> must be at most 2 sentences and ≤ 80 words.
        - The FIRST line inside <answer> must be exactly one of: yes or no.
        - If you are uncertain, keep the first line as yes or no, and use a SECOND line to explain why you are not sure.
        - If you are confident, include ONLY the first line (yes or no) and nothing else in <answer>.

        # TEMPLATE EXAMPLES (FORMAT ONLY — NOT RELATED TO THE QUESTION)

        ## Uncertain but decisive
        <think>REASONING</think>
        <answer>
        no
        I'm not sure about my answer because ... I would need more information about ... to answer confidently.
        </answer>

        <think>REASONING</think>
        <answer>
        yes
        I'm not sure about my answer because ... I would need more information about ... to answer confidently.
        </answer>

        ## Confident
        <think>REASONING</think>
        <answer>
        yes
        </answer>

        <think>REASONING</think>
        <answer>
        no
        </answer>

        # TIME SERIES SEGMENTS
        {segments_section}

        # QUESTION
        {question}
        """

    
    # def _create_reasoner_prompt_ECG(
    #     self,
    #     question: str,
    #     segments_data: List[Tuple[int, int]],  # list of (start, end)
    # ) -> str:
    #     """STRICT reasoner prompt: ONLY <think> and <answer>."""
    #     # Build segments section with a TS placeholder per segment
    #     if segments_data:
    #         parts = []
    #         for i, (start, end) in enumerate(segments_data, 1):
    #             parts.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
    #         segments_section = "\n".join(parts)
    #     else:
    #         segments_section =f"### Full Time Series\n{TS_PLACEHOLDER}\n"

    #     return f"""# ROLE
    #     You are the REASONER. Analyze the given time-series segments and answer the question.

    #     # OUTPUT FORMAT (STRICT)
    #     You MUST output exactly two XML-style blocks and NOTHING ELSE:
    #     <think>One–two sentences (≤80 words total) explaining your reasoning, referencing segments by number (e.g., "Seg 2 rises ~0.3 at t=150–200"). No lists, no math blocks.</think>
    #     <answer>
    #     A single uppercase letter on its own line: A, B, C, or D
    #     </answer>

    #     # MANDATORY RULES
    #     - No text outside <think> and <answer>.
    #     - <think> must be ≤ 80 words. Keep it high-level; no step-by-step chain-of-thought.
    #     - <answer> must contain exactly one of A/B/C/D on the FIRST and ONLY content line.
    #     - If evidence is insufficient, pick the most likely option and reflect uncertainty concisely in <think> (not in <answer>).

    #     # TIME SERIES SEGMENTS
    #     {segments_section}

    #     # QUESTION
    #     {question}

    #     # EXAMPLES (for format only — NOT related to the question)
    #     Below are example answers showing the correct format. These examples are not related to the current question.

    #     ## Example A (clear signal pattern):
    #     <think>Seg 1 shows regular P–QRS–T complexes with uniform intervals and stable baseline, consistent with normal sinus rhythm (A).</think>
    #     <answer>
    #     A
    #     </answer>

    #     ## Example B (confident classification):
    #     <think>Seg 1 has a higher mean (~0.78) and lower variance than Seg 2; sustained plateau at t=40–90 supports option B.</think>
    #     <answer>
    #     B
    #     </answer>

    #     ## Example C (uncertain but decisive):
    #     <think>Seg 2 shows a rise at t=150–170 but coverage after t=200 is sparse; Seg 3 is slightly higher yet volatile. Most consistent with C.</think>
    #     <answer>
    #     C
    #     </answer>

    #     ## Example D (alternative plausible reasoning):
    #     <think>Seg 3 reveals irregular QRS intervals with baseline noise and missing P waves; overall morphology matches the pattern described in D.</think>
    #     <answer>
    #     D
    #     </answer>
    #     """ 
    def inject_ts_segments_and_labels(
        self,
        text_embeddings: torch.Tensor,            # (B, T, D)
        placeholder_mask: torch.Tensor,           # (B, T) bool
        ts_embeddings: List[torch.Tensor],        # len B; each: (S_i, L_i, D) or list of (L_k, D)
        labels = None,    # (B, T) or None
        attention_mask = None,
        ignore_index: int = -100,
    ):
        """
        Replaces each placeholder token with its TS segment embeddings (per batch item),
        returns padded tensors:
        combined_embeds : (B, T_new_max, D)
        combined_attn   : (B, T_new_max) with 1s for real tokens, 0s for pad
        combined_labels : (B, T_new_max) or None (TS spans filled with ignore_index)
        """
        B, T, D = text_embeddings.shape
        device = text_embeddings.device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device, dtype=torch.long)

        seqs_emb, seqs_attn, seqs_lab = [], [], []

        for b in range(B):
            emb_b = text_embeddings[b]               # (T, D)
            segs_b = ts_embeddings[b]                # (S_b, L_b, D) or list of (1, L_k, D)
            attn_b = attention_mask[b]
            
            pos = placeholder_mask[b].nonzero(as_tuple=True)[0].tolist()  # indices of placeholders in order
            if len(pos) != len(segs_b):
                print("len(pos), len(segs_b)", len(pos), len(segs_b))
            assert len(pos) == len(segs_b), "Placeholders count must match TS segments per batch item."

            spans_emb = []
            spans_attn = []
            spans_lab  = [] if labels is not None else None

            prev = 0
            for i, p in enumerate(pos):
                # tokens before placeholder
                if p > prev:
                    spans_emb.append(emb_b[prev:p])                       # (p-prev, D)
                    spans_attn.append(attn_b[prev:p])
                    if labels is not None:
                        spans_lab.append(labels[b, prev:p])

                # TS segment replacing the placeholder
                seg = segs_b[i][0]                                        # (1, L_i, D)
                L = seg.size(0)
                spans_emb.append(seg)
                spans_attn.append(torch.ones(L, device=device, dtype=torch.long))
                if labels is not None:
                    spans_lab.append(torch.full((L,), ignore_index, device=labels.device, dtype=labels.dtype))

                prev = p + 1  # skip the placeholder token

            # tail after last placeholder
            if prev < T:
                spans_emb.append(emb_b[prev:])
                spans_attn.append(attn_b[prev:])
                if labels is not None:
                    spans_lab.append(labels[b, prev:])

            merged_emb  = torch.cat(spans_emb, dim=0)                     # (T_new_b, D)
            merged_attn = torch.cat(spans_attn, dim=0)                    # (T_new_b,)
            seqs_emb.append(merged_emb)
            seqs_attn.append(merged_attn)

            if labels is not None:
                merged_lab = torch.cat(spans_lab, dim=0) if spans_lab else torch.empty(0, device=labels.device, dtype=labels.dtype)
                seqs_lab.append(merged_lab)

        # pad to max length
        combined_embeds = pad_sequence(seqs_emb, batch_first=True)                  # (B, T_new_max, D)
        combined_attn   = pad_sequence(seqs_attn, batch_first=True, padding_value=0)                 # (B, T_new_max)
        combined_labels = None
        if labels is not None:
            combined_labels = pad_sequence(seqs_lab, batch_first=True,
                                        padding_value=ignore_index)              # (B, T_new_max)

        return combined_embeds, combined_attn, combined_labels

    def _create_controller_prompt_ETI_extended_V2(
            self,
            question,
            current_segments,
            segment_memory: SegmentMemory,
            previous_reasoner_answer: Optional[str] = None,
            ts: np.ndarray = None,
        ) -> str:
        """Create controller prompt with TS_PLACEHOLDER for injection (with size examples)"""

        # --- Current segments formatting ---
        if current_segments:
            current_segments_text = "\n".join([
                f"  - Segment {i+1}: timesteps [{start}, {end}] ({end-start+1} steps)"
                for i, (start, end) in enumerate(current_segments)
            ])
        else:
            current_segments_text = "  - No segments provided yet"

        # --- TS length & extrema (coarse, order=~10% N) ---
        ts = ts.squeeze() if ts is not None and ts.ndim > 1 else ts
        N = int(len(ts)) if ts is not None else int(segment_memory.ts_length)

        if ts is None or N < 3:
            minima, maxima = [], []
        else:
            order = max(1, int(0.10 * N))
            mins = argrelextrema(ts, np.less_equal, order=order)[0]
            maxs = argrelextrema(ts, np.greater_equal, order=order)[0]
            minima = [(int(i), round(float(ts[i]), 4)) for i in mins]
            maxima = [(int(i), round(float(ts[i]), 4)) for i in maxs]

        extrema_info = ""
        if minima or maxima:
            extrema_info = (
                "### Key Points in the Time Series\n"
                f"- **Local Minima (index → value)**: {minima}\n"
                f"- **Local Maxima (index → value)**: {maxima}\n"
            )

        # --- Segment-size bands (inclusive length = end - start + 1) ---
        def _band(lo_pct, hi_pct=None):
            lo = max(1, int(round(lo_pct * N)))
            hi = N if hi_pct is None else max(lo, int(round(hi_pct * N)))
            return lo, hi

        small_lo, small_hi = _band(0.05, 0.15)
        med_lo,   med_hi   = _band(0.20, 0.40)
        large_lo, large_hi = _band(0.50, None)

        # --- Example windows for each band ---
        def _clamp(a, b):  # ensure [a,b] inside [0, N-1] and a <= b
            a = max(0, min(a, N-1))
            b = max(0, min(b, N-1))
            if b < a:
                a, b = b, a
            return a, b

        def _examples_for_length(L: int):
            # left-anchored
            e1 = _clamp(0, L - 1)
            # mid-centered
            start_mid = max(0, (N // 2) - (L // 2))
            e2 = _clamp(start_mid, start_mid + L - 1)
            # right-anchored
            e3 = _clamp(N - L, N - 1)
            ex = [e1, e2, e3]
            # optional [150, 150+L-1]
            if N > 150 and 150 + L - 1 < N:
                ex.append(_clamp(150, 150 + L - 1))
            # dedup while preserving order
            seen, uniq = set(), []
            for a, b in ex:
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    uniq.append([a, b])
            return uniq

        def _examples_for_band(lo: int, hi: int):
            # pick representative lengths within band
            mids = sorted(set([
                lo,
                max(lo, (lo + hi) // 2 if hi != N else max(lo, min(N, int(0.75 * max(lo, 1))))),
                min(hi, hi if hi != N else max(lo, int(0.9 * N)))
            ]))
            out = {}
            for L in mids:
                out[L] = _examples_for_length(L)
            return out

        small_examples = _examples_for_band(small_lo, small_hi)
        med_examples   = _examples_for_band(med_lo,   med_hi)
        # For large, pick a few illustrative lengths up to N
        large_hi_eff = N
        large_examples = _examples_for_band(large_lo, large_hi_eff)

        def _fmt_examples(d):
            lines = []
            for L, segs in d.items():
                seg_txt = ", ".join([f"[{a}, {b}]" for a, b in segs])
                lines.append(f"  • length {L}: {seg_txt}")
            return "\n".join(lines) if lines else "  • (no valid examples)"

        size_guidance = (
            "### Segment Size Guidance\n"
            "Choose any segment size; the inclusive length is `(end - start + 1)`.\n"
            f"- **Small**  (≈5–15%): {small_lo}–{small_hi} steps\n"
            f"{_fmt_examples(small_examples)}\n"
            f"- **Medium** (≈20–40%): {med_lo}–{med_hi} steps\n"
            f"{_fmt_examples(med_examples)}\n"
            f"- **Large**  (≥50%): {large_lo}–{N} steps\n"
            f"{_fmt_examples(large_examples)}\n"
        )

         # Format previous answer
        if previous_reasoner_answer:
            answer_section = f"""
        ## LLM's Previous Answer
        ```
        {previous_reasoner_answer}
        ```

        Evaluate if the answer is accurate and if the available segments provide sufficient information to answer the question accurately. If the answer appears incomplete, uncertain, or potentially incorrect, consider retrieving additional segments.
        """
        else:
            answer_section = """
        ## LLM's Previous Answer
        No answer yet - this is the first round. You need to select initial segments that are most likely to contain information relevant to the question.
        """
            
        prompt = f"""# TASK
        You are a time series analysis expert. Your task is to decide which time series segments an LLM needs to accurately answer a question related to a time series.


        ## Instructions
        1. You receive the FULL time series embedding, the question, the segments that were given to the LLM so far, and the LLM's answer.
        2. Review the question and the LLM's answer, and analyze the full time series embedding.
        3. Evaluate if the LLM's answer is accurate and if the available segments provide sufficient information to answer the question accurately - you can try answering the question yourself to understad what is missing.
        4. Make ONE of two decisions:
        - **Retrieve an additional segment**: If you think that thew answer is not accurate and/or there is critical information missing, use the tool to get an additional segment of time series data for the LLM.
        - **Accept the LLM's answer as final**: If you think the LLM's answer is accurate and the information provided is sufficient.

        ## Full Time Series Embedding
        {TS_PLACEHOLDER}

        ## Current Status
        - **Time series length**: {N} timesteps
        - **Question**: {question}
        - **Segments provided to the LLM so far**: 
        {current_segments_text}

        {answer_section}

        ## Information for selecting the next segment (based on the full time series)
        {extrema_info}
        {size_guidance}

        ## Output Format
        You MUST respond in ONE of the following two formats:

        **Option 1: Retrieve an additional segment**
        Make a tool call asking for a specific segment of the time series data: 
        <think> includes your consice reasoning (1 sentence) for why you are retrieving this segment and what information is missing </think>
        <tool_call>
        {{"name": "timeseries_zoom_in_tool", "arguments": {{"ts_seg": [start, end]}}}}
        </tool_call>

        **Option 2: Accept the LLM's answer as final**
        You should answer with the following format:
        <think> includes your concise reasoning (1 sentence) for why you are accepting the answer </think>
        <answer>ACCEPT</answer>

        ## Important Rules
        1. You can only call `timeseries_zoom_in_tool` ONCE per turn
        2. Specify segments as integers: [100, 200] means timesteps 100 to 200 inclusive
        4. Do NOT request segments outside valid range [0, {N - 1}]
        5. If you output "ACCEPT", the process will immediately conclude with the LLM's current answer

        Make your decision now:"""

        return prompt


    def _create_reasoner_prompt_scratch(
        self,
        question: str,
        segments_data: List[Tuple[int, int]],  # Just indices
        ) -> str:
        """Create reasoner prompt with TS_PLACEHOLDER for segments"""
        
        # Format segments - one placeholder per segment
        if segments_data:
            segments_text = []
            for i, (start, end) in enumerate(segments_data, 1):
                segments_text.append(f"### Segment {i}: Timesteps [{start}, {end}]\n{TS_PLACEHOLDER}\n")
            segments_section = "\n".join(segments_text)
        else:
            segments_section = f"### Full Time Series\n{TS_PLACEHOLDER}\n"
    
        prompt = f"""# ROLE
        You are the REASONER. Analyze ONLY the given time-series segments and answer the question. Be concise.

        # Output Schema (STRICT)
        You must output exactly two blocks and nothing else:

        <think>One–two sentences (≤80 words total) citing segment numbers and key observations.</think>
        <answer>
        [Direct answer ONLY. If MCQ A–D, output the single capital letter on the first line.]
        Explanation: One sentence (≤25 words) that cites specific segment indices/timesteps.
        </answer>

        # Rules (MANDATORY)
        - No text outside <think> and <answer>.
        In <think>:
        - do not restate the question; cite segments by number (e.g., "Seg 2 rises 0.3 at t=150–200").
        - if evidence is insufficient, state the most likely answer and reflect uncertainty in the Explanation (but still keep within limits).
        In <answer>:
        - FIRST LINE must be exactly one letter: A, B, C, or D.

        # Time Series Segments
        {segments_section}

        # Question
        {question}

        Good Answer Example:
        <think>Seg 1 shows a steady +0.5 from t=40–90; Seg 2 is flat. This supports option B.</think>
        <answer>
        B
        </answer>
        """
        return prompt

    def _is_accept_decision(self, completion: str) -> bool:
        """Check if controller accepted"""
        return bool(re.search(r'<decision>ACCEPT</decision>', completion, re.IGNORECASE) or
                   "ACCEPT" in completion.upper())
    
    def _parse_segment_from_tool_call(self, completion: str) -> Tuple[Optional[List[int]], bool]:
        """Parse segment from controller's tool call"""
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL | re.IGNORECASE)
        
        if not tool_match:
            return None, False
        
        try:
            tool_call = json.loads(tool_match.group(1))
            ts_seg = tool_call.get("arguments", {}).get("ts_seg")
            
            if isinstance(ts_seg, list) and len(ts_seg) == 2:
                return ts_seg, True
        except:
            pass
        
        return None, False
    
    def _parse_controller_decision(self, response):
        """
        Parse controller response with strict format checking
        
        Expected formats:
            1. <answer>ACCEPT</answer> - Accept decision
            2. <tool_call>{"name": "timeseries_zoom_in_tool", ...}</tool_call> - Retrieve
        
        Returns:
            dict with:
            - decision: "accept" | "retrieve" | "error"
            - segment: [start, end] if retrieve
            - reason: error message if error
            - has_answer: bool - whether <answer> tag was present
            - has_tool_call: bool - whether <tool_call> tag was present
        """
        
        # Check for <answer> tag first (higher priority)
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_match = re.search(answer_pattern, response, re.IGNORECASE | re.DOTALL)
        
        # Check for <tool_call> tag
        # tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tool_match = re.search(tool_pattern, response, re.IGNORECASE | re.DOTALL)
        
        # Helper for errors
        def ret_error(reason):
            return {
                "decision": "error", 
                "reason": reason, 
                "has_answer": answer_match is not None, 
                "has_tool_call": tool_match is not None,
            }
        
        has_answer = answer_match is not None
        has_tool_call = tool_match is not None
        
        # if answer_match and tool_match:
        #     return ret_error("Both <answer> and <tool_call> present")
        if not answer_match and not tool_match:
            return ret_error("No decision found")
            
        if has_tool_call:
            try:
                tool_json = json.loads(tool_match.group(1))
                seg = tool_json["arguments"]["ts_seg"]
                
                if tool_json["name"] != "timeseries_zoom_in_tool":
                    return ret_error("Wrong tool name")
                if not (isinstance(seg, list) and len(seg) == 2 and seg[0] >= 0 and seg[1] > seg[0]):
                    return ret_error("Invalid segment")
                
                # Convert to int (JSON can parse numbers as floats)
                return {
                    "decision": "retrieve",
                    "segment": (int(seg[0]), int(seg[1])),
                    "has_answer": False,
                    "has_tool_call": True,
                }

            except Exception as e:
                return ret_error(f"Failed to parse tool call JSON: {e}")
        
        if has_answer:
            answer_content = answer_match.group(1).strip()
            
            # Check if it's ACCEPT
            if re.search(r'\bACCEPT\b', answer_content, re.IGNORECASE):
                return {
                    "decision": "accept",
                    "has_answer": True,
                    "has_tool_call": False,
                }
            # else:
            #     return ret_error(f"Invalid answer content: '{answer_content}' (expected ACCEPT)")

        return ret_error("ACCEPT is not found in answer block")


    def _parse_reasoner_response(self, response: str) -> Dict[str, str]:
        """
        Parse reasoner response
        
        Expected format:
            <think>reasoning here...</think>
            <answer>answer here</answer>
        
        Process:
        1. Extract <think> block (reasoning)
        2. Remove <think> block from response
        3. Extract <answer> from remaining text
        
        This ensures we don't pick up hallucinated <answer> tags inside <think>
        
        Returns:
            dict with:
            - reasoning: extracted from <think> tags (if any)
            - answer: extracted from <answer> tags (outside think block)
        """
        
        # Step 1: Extract reasoning from <think></think> tags
        reasoning = ""
        think_pattern = r'<think>\s*(.*?)\s*</think>'
        think_match = re.search(think_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if think_match:
            reasoning = think_match.group(1).strip()
            # Remove the entire <think>...</think> block from response
            response_without_think = re.sub(
                think_pattern, 
                '', 
                response, 
                flags=re.IGNORECASE | re.DOTALL
            )
        else:
            response_without_think = response
        
        # Step 2: Extract answer from remaining text (outside think block)
        answer = ""
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_match = re.search(answer_pattern, response_without_think, re.IGNORECASE | re.DOTALL)
        
        if answer_match:
            answer_block = answer_match.group(1).strip()
            
            # Step 3: Extract just the letter (A/B/C/D)
            # Try to find single letter at start of answer block
            letter_pattern = r'^([A-D])\b'
            letter_match = re.search(letter_pattern, answer_block, re.IGNORECASE)
            
            if letter_match:
                answer = letter_match.group(1).upper()
            else:
                # Fallback: try "The answer is X" pattern
                answer_is_pattern = r'(?:the\s+)?answer\s+is\s+([A-D])\b'
                answer_is_match = re.search(answer_is_pattern, answer_block, re.IGNORECASE)
                if answer_is_match:
                    answer = answer_is_match.group(1).upper()
                else:
                    # Last resort: take the whole block (might be wrong format)
                    answer = answer_block
        else:
            # Fallback: try to find "Answer:" prefix
            answer_prefix_pattern = r'Answer:\s*([A-D])\b'
            prefix_match = re.search(answer_prefix_pattern, response_without_think, re.IGNORECASE)
            if prefix_match:
                answer = prefix_match.group(1).upper()
        
        return {
            "reasoning": reasoning,
            "answer": answer
        }

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison
        
        Examples:
            "A) The answer is 42" -> "a"  (for MCQ)
            "The value is 3.14159" -> "the value is 314159"
            "  Yes  " -> "yes"
        """
        import string
        
        answer = answer.strip().lower()
        
        # Check if it's a multiple choice answer (single letter)
        mcq_pattern = r'^([a-d])\b'
        mcq_match = re.match(mcq_pattern, answer)
        if mcq_match:
            return mcq_match.group(1)
        
        # Remove punctuation
        answer = answer.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        answer = " ".join(answer.split())
        
        return answer

    def _parse_controller_decision_w_reasoning(self, response: str):
        """Parse controller response with tool call format"""
        
        # Check for ACCEPT
        accept_pattern = r"<answer>\s*ACCEPT\s*</answer>"
        if re.search(accept_pattern, response, re.IGNORECASE):
            # Extract reasoning
            reasoning_pattern = r"Reasoning:\s*(.+?)(?=<answer>|$)"
            reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return {
                "decision": "accept",
                "reasoning": reasoning
            }
        
        # Check for tool call
        tool_pattern = r'<tool_call>\s*\{.*?"name":\s*"timeseries_zoom_in_tool".*?"arguments":\s*\{.*?"ts_seg":\s*\[(\d+),\s*(\d+)\].*?\}\s*\}\s*</tool_call>'
        tool_match = re.search(tool_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if tool_match:
            start = int(tool_match.group(1))
            end = int(tool_match.group(2))
            
            # Extract reasoning
            reasoning_pattern = r"Reasoning:\s*(.+?)(?=<tool_call>|$)"
            reasoning_match = re.search(reasoning_pattern, response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            return {
                "decision": "retrieve",
                "segment": (start, end),
                "reasoning": reasoning
            }
        
        return {
            "decision": "error",
            "reason": "Could not parse controller response",
            "response": response
        }
    def _encode_segment(self, ts_data: np.ndarray, question_ids: List[int], 
                       segment: List[int]) -> torch.Tensor:
        """Encode a TS segment"""
        start, end = segment
        
        # Prepare segment data
        if ts_data.ndim == 1:
            seg_data = ts_data[start:end]
        else:
            seg_data = ts_data[:, start:end]  # For 2TS
        
        # Convert to tensor
        seg_tensor = torch.from_numpy(seg_data).float().unsqueeze(0).to(self.model.device)
        
        # Get question embedding
        q_tensor = torch.tensor(question_ids, device=self.model.device).unsqueeze(0)
        q_emb = self.model.model.embed_tokens(q_tensor)
        
        # Encode segment
        segment_embedding = self.generation_handler._encode_ts_segment(
            seg_tensor,
            q_emb,
            [0, seg_tensor.shape[1]]  # Use full segment
        )
        
        return segment_embedding
    
    def _extract_answer(self, completion: str) -> str:
        """Extract answer from reasoner output"""
        match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def generate_controller_reasoner_loop_batched(
        self,
        question,
        ts_data,
        gold_answer,
        timestamps=None,
        raw_ts=None,
        temperature=0.7,
        top_p=0.9,
        num_loops: int = 1,
    ):
        """
        Run multiple controller-reasoner loops in parallel for the same question.
        
        Batches operations where possible (controller generation when prompts are identical).
        Diverges when loops make different decisions.
        
        Args:
            num_loops: Number of parallel loops to run for this question
            
        Returns:
            List[dict]: One result dict per loop (same structure as generate_controller_reasoner_loop)
        """
        
        if num_loops == 1:
            # Fallback to sequential
            result = self.generate_controller_reasoner_loop(
                question=question,
                ts_data=ts_data,
                gold_answer=gold_answer,
                timestamps=timestamps,
                raw_ts=raw_ts,
                temperature=temperature,
                top_p=top_p,
            )
            return [result]
        
        # Initialize
        ts_tensor = ts_data
        ts_length = ts_tensor.shape[0]
        batch_states = self._initialize_batch_states(num_loops, ts_length)
        
        # Handle initial full TS if configured
        if self.include_full_ts_initially:
            self._handle_initial_full_ts(
                batch_states, ts_tensor, question, top_p
            )
        
        # Main loop
        for round_num in range(self.max_rounds):
            active_indices = [i for i, s in enumerate(batch_states) if not s["is_done"]]
            
            if not active_indices:
                break  # All loops finished
            
            # print(f"\n=== Round {round_num} | Active loops: {len(active_indices)}/{num_loops} ===")
            
            # Group loops by prompt and generate controller responses
            self._batch_generate_controller_responses(
                batch_states=batch_states,
                active_indices=active_indices,
                round_num=round_num,
                question=question,
                ts_tensor=ts_tensor,
                raw_ts=raw_ts,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Process decisions and generate reasoner responses
            self._process_decisions_and_generate_reasoner(
                batch_states=batch_states,
                active_indices=active_indices,
                round_num=round_num,
                question=question,
                ts_tensor=ts_tensor,
                top_p=top_p,
            )
        
        # Build and return results
        results = self._build_batch_results(batch_states)
        # print(f"\n=== Batch Complete | {len(results)} loops finished ===\n")
        return results
        
    def _initialize_batch_states(self, num_loops: int, ts_length: int) -> List[dict]:
        """Initialize state dictionaries for each loop in the batch."""
        batch_states = []
        for loop_idx in range(num_loops):
            state = {
                "rollout_idx": loop_idx,
                "segment_memory": SegmentMemory(ts_length),
                "controller_completions": [],
                "reasoner_completions": [],
                "reasoner_answers": [],
                "controller_prompts": [],
                "reasoner_prompts": [],
                "full_conv_for_controller": [],
                "controller_prompt_ids": [],
                "controller_completion_ids": [],
                "previous_reasoner_answer": None,
                "format_error_occurred": False,
                "format_error_round": None,
                "is_done": False,
                "round_num": 0,
                "decision": None,
                "skipped": False,
                "skip_reason": None,
            }
            batch_states.append(state)
        return batch_states


    def _handle_initial_full_ts(
        self,
        batch_states: List[dict],
        ts_tensor,
        question: str,
        top_p: float,
    ):
        """Handle initial reasoner generation with full TS (batched for efficiency)."""
        ts_length = ts_tensor.shape[1] if ts_tensor.dim() > 1 else ts_tensor.shape[0]
        full_ts_segment = [0, ts_length]
        
        # Add to all segment memories
        for state in batch_states:
            success = state["segment_memory"].add_segment(full_ts_segment)
            if not success:
                state["is_done"] = True
                state["format_error_occurred"] = True
        
        # Check if any loops are still active
        active_states = [s for s in batch_states if not s["is_done"]]
        if not active_states:
            return
        
        # Create reasoner prompt once (same for all loops)
        reasoner_prompt = self._create_reasoner_prompt(
            question=question,
            segments_data=[full_ts_segment],
        )
        sys_content_reasoner = 'you are a helpful assistant that can answer questions about time series data.'
        messages_reasoner = [
            {"role": "system", "content": sys_content_reasoner},
            {"role": "user", "content": reasoner_prompt}
        ]
        
        reasoner_prompt_formatted = self.tokenizer.apply_chat_template(
            messages_reasoner, 
            tokenize=False,
            add_generation_prompt=True, 
            enable_thinking=True
        )
        
        # Save prompt for all active states
        for state in active_states:
            state["reasoner_prompts"].append(reasoner_prompt_formatted)
        
        # Batch generate reasoner responses for all active loops
        
        with torch.no_grad():
            completions, answers, prompt_ids_rsnr = self.generate_reasoner_rollouts_batched(
                question=question,
                ts_data=ts_tensor,
                segments=[full_ts_segment],
                num_rollouts=len(active_states),  # One per active loop
                timestamps=None,
                raw_ts=None,
                temperature=self.reasoner_temperature,
                top_p=top_p,
            )
        
        
        # Assign responses to each active state
        for i, state in enumerate(active_states):
            state["reasoner_completions"].append(completions[i])
            state["previous_reasoner_answer"] = completions[i]
            
            # Parse the answer
            parsed = self._parse_reasoner_response(completions[i])
            state["reasoner_answers"].append(parsed)

    def _handle_first_segment_trials_batched(
        self,
        batch_states: List[dict],
        active_indices: List[int],
        prompts_formatted: List[str],  # Changed: now takes a list of prompts
        ts_tensor,
        round_num: int,
        top_p: float,
    ):
        """Handle first turn trials with batched generation across different prompts."""
        still_trying_indices = list(active_indices)
        still_trying_prompts = list(prompts_formatted)
        # Track last response for each index (in case they all fail)
        last_responses = {}  # {idx: (response, prompt_ids, completion_ids, decision)}
        
        for trial in range(self.first_seg_trials):
            if not still_trying_indices:
                break
            
            # Batch generate for all loops still trying (with different prompts)
            responses, prompt_ids_controller, completion_ids_controller, was_skipped = self._batch_generate_controller_multiple_prompts(
                prompts=still_trying_prompts,
                ts_tensor=ts_tensor,
                top_p=top_p,
            )
            
            if was_skipped:
                print(f"⚠️ Batch skipped due to data issue, marking {len(still_trying_indices)} states")
                for idx in still_trying_indices:
                    state = batch_states[idx]
                    state["skipped"] = True
                    state["is_done"] = True
                    state["format_error_occurred"] = True
                    state["format_error_round"] = round_num
                    state["skip_reason"] = "dtype_or_data_error"
                    # Store dummy completions so downstream doesn't crash
                    state["controller_completions"].append("<answer>SKIP</answer>")
                    state["controller_prompt_ids"].append(prompt_ids_controller[0] if prompt_ids_controller else torch.tensor([]))
                    state["controller_completion_ids"].append(completion_ids_controller[0] if completion_ids_controller else torch.tensor([]))
                return

            # Check which succeeded
            next_trying_indices = []
            next_trying_prompts = []
            
            for i, idx in enumerate(still_trying_indices):
                state = batch_states[idx]
                controller_response = responses[i]
                decision = self._parse_controller_decision(controller_response)
                last_responses[idx] = (
                controller_response,
                prompt_ids_controller[i],
                completion_ids_controller[i],
                decision
                )
                
                if decision["decision"] == "retrieve":
                    # Success!
                    # print(f"    Loop {idx}: Valid segment after {trial + 1} trials")
                    state["controller_completions"].append(controller_response)
                    state["controller_prompt_ids"].append(prompt_ids_controller[i])
                    state["controller_completion_ids"].append(completion_ids_controller[i])
                    state["full_conv_for_controller"].append({
                        "role": "assistant",
                        "content": controller_response
                    })
                    state["decision"] = decision
                else:
                    # Need retry with same prompt
                    next_trying_indices.append(idx)
                    next_trying_prompts.append(still_trying_prompts[i])
            
            still_trying_indices = next_trying_indices
            still_trying_prompts = next_trying_prompts
        
        # Mark any remaining as failed
        for idx in still_trying_indices:
            state = batch_states[idx]
            if idx in last_responses:
                controller_response, prompt_ids, completion_ids, decision = last_responses[idx]
                # Append the failed completion (maintains structure)
                state["controller_completions"].append(controller_response)
                state["controller_prompt_ids"].append(prompt_ids)
                state["controller_completion_ids"].append(completion_ids)
                state["full_conv_for_controller"].append({
                    "role": "assistant",
                    "content": controller_response
                })
                state["decision"] = decision
            else:
                # Shouldn't happen, but handle it
                print(f"⚠️  No response tracked for failed index {idx}")
                state["decision"] = {"decision": "error", "reason": "No response generated"}
            
            state["format_error_occurred"] = True
            state["format_error_round"] = round_num
            state["is_done"] = True
            # print(f"    Loop {idx}: Failed after {self.first_seg_trials} trials")

    def _batch_generate_controller_responses(
        self,
        batch_states: List[dict],
        active_indices: List[int],
        round_num: int,
        question: str,
        ts_tensor,
        raw_ts,
        temperature: float,
        top_p: float,
    ):
        """
        Generate controller responses for all active loops in one batch.
        No grouping needed - handles different prompts via padding.
        """
        
        # Build prompts for all active loops
        prompts_raw = []
        for idx in active_indices:
            state = batch_states[idx]
            
            if round_num == 0: # even with full ts, make it choose
                if self.include_full_ts_initially:
                    prompt = self._create_controller_prompt_generic_mcq_full_ts(
                        question=question,
                        segment_memory=state["segment_memory"],
                        current_segments=state["segment_memory"].get_all_segments(),
                        previous_reasoner_answer=state["previous_reasoner_answer"]
                    )
                else:
                    prompt = self._create_first_turn_controller_prompt(
                        question=question,
                        segment_memory=state["segment_memory"],
                        ts=np.array(raw_ts)
                    )

            else:
                prompt = self._create_controller_prompt(
                    question=question,
                    current_segments=state["segment_memory"].get_all_segments(),
                    segment_memory=state["segment_memory"],
                    previous_reasoner_answer=state["previous_reasoner_answer"],
                    ts=np.array(raw_ts)
                )
            
            prompts_raw.append(prompt)
        
        # Format all prompts
        prompts_formatted = []
        for prompt in prompts_raw:
            sys_content = 'you are a helpful assistant that can answer questions about time series data.'
            initial_messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt}
            ]
            formatted = self.tokenizer.apply_chat_template(
                initial_messages,
                tokenize=False,
                tools=TS_TOOLS,
                add_generation_prompt=True,
                enable_thinking=True
            )
            prompts_formatted.append(formatted)
        
        # Save prompts and conversation history
        for i, idx in enumerate(active_indices):
            state = batch_states[idx]
            initial_messages = [
                {"role": "system", "content": 'you are a helpful assistant that can answer questions about time series data.'},
                {"role": "user", "content": prompts_raw[i]}
            ]
            state["controller_prompts"].append(prompts_formatted[i])
            state["full_conv_for_controller"].append(initial_messages)
        
        # print(f"  Batch generating {len(active_indices)} controller responses (with different prompts)")
        
        # Handle first segment trials (batched) - round 0 always requires retrieve decision
        if round_num == 0:
            self._handle_first_segment_trials_batched(
                batch_states=batch_states,
                active_indices=active_indices,
                prompts_formatted=prompts_formatted,
                ts_tensor=ts_tensor,
                round_num=round_num,
                top_p=top_p,
            )
        else:
            # Single batch generation for all loops
            responses, prompt_ids_controller, completion_ids_controller, was_skipped = self._batch_generate_controller_multiple_prompts(
                prompts=prompts_formatted,
                ts_tensor=ts_tensor,
                top_p=top_p,
            )
            if was_skipped:
                print(f"⚠️ Round {round_num} batch skipped, marking {len(active_indices)} states")
                for idx in active_indices:
                    state = batch_states[idx]
                    state["skipped"] = True
                    state["is_done"] = True
                    state["format_error_occurred"] = True
                    state["format_error_round"] = round_num
                    state["skip_reason"] = "dtype_or_data_error"
                return

            # Save responses and parse decisions
            for i, idx in enumerate(active_indices):
                state = batch_states[idx]
                controller_response = responses[i]
                
                state["controller_completions"].append(controller_response)
                state["controller_prompt_ids"].append(prompt_ids_controller[i])
                state["controller_completion_ids"].append(completion_ids_controller[i])
                state["full_conv_for_controller"].append({
                    "role": "assistant",
                    "content": controller_response
                })
                
                # Parse decision
                decision = self._parse_controller_decision(controller_response)
                state["decision"] = decision

    def _group_loops_by_prompt(
        self,
        batch_states: List[dict],
        active_indices: List[int],
        round_num: int,
        question: str,
        raw_ts,
    ) -> dict:
        """Group active loops by their controller prompt."""
        prompt_groups = {}  # {prompt_hash: {"prompt": str, "indices": [int]}}
        
        for idx in active_indices:
            state = batch_states[idx]
            
            # Create prompt for this state
            if round_num == 0 and not self.include_full_ts_initially:
                prompt = self._create_first_turn_controller_prompt(
                    question=question,
                    segment_memory=state["segment_memory"],
                    ts=np.array(raw_ts)
                )
            else:
                prompt = self._create_controller_prompt(
                    question=question,
                    current_segments=state["segment_memory"].get_all_segments(),
                    segment_memory=state["segment_memory"],
                    previous_reasoner_answer=state["previous_reasoner_answer"],
                    ts=np.array(raw_ts)
                )
            
            # Group by prompt content
            prompt_key = hash(prompt)
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = {"prompt": prompt, "indices": []}
            prompt_groups[prompt_key]["indices"].append(idx)
        
        return prompt_groups
    
    def _format_controller_prompt(self, prompt: str) -> str:
        """Format controller prompt with chat template."""
        sys_content = 'you are a helpful assistant that can answer questions about time series data.'
        initial_messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            initial_messages,
            tokenize=False,
            tools=TS_TOOLS,
            add_generation_prompt=True,
            enable_thinking=True
        )

    def _batch_generate_controller_multiple_prompts(
        self,
        prompts: List[str],
        ts_tensor,
        top_p: float,
    ) -> List[str]:
        """
        Generate controller responses in batch with DIFFERENT prompts.
        Handles padding automatically via tokenizer.
        
        Args:
            prompts: List of different controller prompts
            ts_tensor: Time series tensor (same for all)
            question_ids: Question IDs (same for all)
            top_p: Sampling parameter
            
        Returns:
            List of generated responses (one per prompt)
        """
        batch_size = len(prompts)
        
        # Tokenize with padding (handles different lengths)
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.model.device)
        prompt_token_ids_per_sample = [toks.input_ids[i].cpu() for i in range(batch_size)]
        initial_length = toks.input_ids.shape[1]  
        
        ts_length = ts_tensor.shape[0]
        
        ts_segs = [[0, ts_length]]  # Full TS
        ts_for_model = self._prepare_ts_segments_for_model(
            ts_tensor.unsqueeze(0), [ts_segs]
        )
        ts_for_model = ts_for_model.repeat(batch_size, 1, 1, 1)
        if ts_for_model is not None:
            if self.task_name == "TSQA" or self.task_name == "TRQA_MIXED":
                ts_for_model = ts_for_model.to(self.model.device, dtype=torch.bfloat16)
            else:
                ts_for_model = ts_for_model.to(self.model.device)
        
            # print(f"=== DTYPE DEBUG ===")
            # print(f"Segment sizes: {[s[1]-s[0] for s in ts_segs]}")
            # print(f"ts_for_model dtype: {ts_for_model.dtype}")
            # print(f"ts_for_model shape: {ts_for_model.shape}")
            # print(f"input_ids dtype: {toks.input_ids.dtype}")
            # print(f"Model embed dtype: {self.model.model.embed_tokens.weight.dtype}")

        if ts_for_model.numel() == 0 or ts_for_model is None:
            print("No time series data found")
            print(f"ts_for_model: {ts_for_model}")
            dummy_completion = torch.tensor(
            self.tokenizer.encode("<answer>SKIP</answer>", add_special_tokens=False),
            dtype=torch.long
            )
            
            return (
                ["<answer>SKIP</answer>"] * batch_size,  # Easy to filter by text
                [toks.input_ids[i].cpu() for i in range(batch_size)],  # Prompt IDs are fine
                [dummy_completion.clone() for _ in range(batch_size)],  # Minimal completion
                True  # ← Add skip flag
            )
        if torch.isnan(ts_for_model).any() or torch.isinf(ts_for_model).any():
            print(f"NaN/Inf in ts_for_model, skipping")
            dummy_completion = torch.tensor(
            self.tokenizer.encode("<answer>SKIP</answer>", add_special_tokens=False),
            dtype=torch.long
            )
            return (
                ["<answer>SKIP</answer>"] * batch_size,  # Easy to filter by text
                [toks.input_ids[i].cpu() for i in range(batch_size)],  # Prompt IDs are fine
                [dummy_completion.clone() for _ in range(batch_size)],  # Minimal completion
                True  # ← Add skip flag
            )

        
        stop = StoppingCriteriaList([
        ControllerStoppingCriteria(
            self.tokenizer,
            initial_length=initial_length  # ← CRITICAL FIX
        )
        ])
        # print(f"ts_for_model dtype: {ts_for_model.dtype}")
        # # print(f"Model embed dtype: {self.model.model.embed_tokens.weight.dtype}")
        # if hasattr(self.model, 'ts_encoder'):
        #     print(f"TS encoder dtype: {next(self.model.ts_encoder.parameters()).dtype}")
       # Before model.generate call:
        # print(f"=== DTYPE DEBUG (small TS check) ===")
        # print(f"ts_for_model shape: {ts_for_model.shape}")
        # print(f"ts_for_model dtype: {ts_for_model.dtype}")
        # print(f"ts_for_model min/max: {ts_for_model.min():.4f}/{ts_for_model.max():.4f}")
        # print(f"ts_for_model mask sum: {ts_for_model[..., 1].sum()}")  # Check if mask is all zeros
        # print(f"input_ids dtype: {toks.input_ids.dtype}")
        # # print(f"Model embed dtype: {self.model.model.embed_tokens.weight.dtype}")
        
        # if hasattr(self.model, 'ts_encoder'):
        #     print(f"TS encoder dtype: {next(self.model.ts_encoder.parameters()).dtype}")
        with torch.no_grad():
            gen_ids = self.model.generate(
            input_ids=toks.input_ids,
            attention_mask=toks.attention_mask,
            timeseries=ts_for_model,
            max_new_tokens=256,
            temperature=self.controller_temperature,
            do_sample=True,
            stopping_criteria=stop,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            top_p=top_p,
            )
        
        responses = []
        completion_ids_per_sample = []
        
        for i in range(batch_size):
            # Extract only new tokens (after initial prompt)
            new_tokens = gen_ids[i, initial_length:]
            completion_ids_per_sample.append(new_tokens.cpu())
            
            # Decode only new tokens
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)
        
        # Cleanup
        del toks, ts_for_model, gen_ids
        torch.cuda.empty_cache()
    
        return responses, prompt_token_ids_per_sample, completion_ids_per_sample, False # completion ids

    # def _batch_generate_controller_single(
    #     self,
    #     prompt: str,
    #     ts_tensor,
    #     question_ids: List[int],
    #     batch_size: int,
    #     top_p: float,
    # ) -> List[str]:
    #     """Generate controller responses in batch (helper for batching identical prompts)."""
    #     prompts_batch = [prompt] * batch_size
        
    #     toks = self.tokenizer(
    #         prompts_batch,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left"
    #     ).to(self.model.device)
        
    #     text_emb = self.model.model.embed_tokens(toks.input_ids)
    #     ph_mask = (toks.input_ids == self.generation_handler.ts_placeholder_id)
        
    #     q_emb = self.generation_handler._get_question_embedding(question_ids)
    #     ts_emb = self.generation_handler.model.encode_ts(ts_tensor, q_emb)
        
    #     batch_ts_embeddings = []
    #     for i in range(batch_size):
    #         num_placeholders = ph_mask[i].sum().item()
    #         batch_ts_embeddings.append([ts_emb])
        
    #     combined_emb, combined_attn, _ = self.generation_handler.model.inject_ts_segments_and_labels(
    #         text_embeddings=text_emb,
    #         placeholder_mask=ph_mask,
    #         ts_embeddings=batch_ts_embeddings,
    #         labels=None
    #     )
        
    #     stop = StoppingCriteriaList([ControllerStoppingCriteria(self.tokenizer)])  # Fixed: Use ToolStoppingCriteria
        
    #     with torch.no_grad():
    #         gen_ids = self.model.generate(
    #             inputs_embeds=combined_emb,
    #             attention_mask=combined_attn,
    #             max_new_tokens=256,
    #             temperature=self.controller_temperature,
    #             do_sample=True,
    #             stopping_criteria=stop,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #             use_cache=True,
    #             top_p=top_p,
    #         )
        
    #     responses = [
    #         self.tokenizer.decode(gen_ids[i], skip_special_tokens=False)
    #         for i in range(batch_size)
    #     ]
        
    #     del toks, text_emb, combined_emb, combined_attn, gen_ids
    #     torch.cuda.empty_cache()
        
    #     return responses

    def _process_decisions_and_generate_reasoner(
        self,
        batch_states: List[dict],
        active_indices: List[int],
        round_num: int,
        question: str,
        ts_tensor,
        top_p: float,
    ):
        """
        Process controller decisions and batch-generate reasoner responses.
        
        New strategy:
        1. First pass: Process all decisions (validate, encode segments)
        2. Second pass: Batch generate all reasoner responses together
        """
        
        # === FIRST PASS: Process decisions and prepare segments ===
        states_needing_reasoner = []
        indices_needing_reasoner = []
        
        for idx in active_indices:
            state = batch_states[idx]
            
            if state["is_done"]:
                continue
            
            decision = state["decision"]
            
            # Handle error
            if decision["decision"] == "error":
                state["format_error_occurred"] = True
                state["format_error_round"] = round_num
                state["is_done"] = True
                # print(f"    Loop {idx}: Format error")
                continue
            
            # Handle accept
            elif decision["decision"] == "accept":
                state["is_done"] = True
                # print(f"    Loop {idx}: Accepted")
                continue
            
            # Handle retrieve - validate and encode segment
            elif decision["decision"] == "retrieve":
                success = self._validate_segment(
                    state, idx, decision, round_num, ts_tensor
                )
                
                if success:
                    # Queue for batched reasoner generation
                    states_needing_reasoner.append(state)
                    indices_needing_reasoner.append(idx)
                # else: state already marked as done in validation
            
            state["round_num"] += 1
        
        # === SECOND PASS: Batch generate all reasoner responses ===
        if states_needing_reasoner:
            # print(f"  Processing {len(states_needing_reasoner)} retrieve decisions")
            # TODO: return this
            # _ = self.batch_reasoner_sft_prompt(
            #     states_to_process=states_needing_reasoner,
            #     state_indices=indices_needing_reasoner,
            #     question=question,
            #     ts_tensor=ts_tensor,
            #     question_ids=question_ids,
            #     top_p=top_p,
            # )
            reasoner_responses = self._batch_generate_reasoner_with_different_segments(
                states_to_process=states_needing_reasoner,
                state_indices=indices_needing_reasoner,
                question=question,
                ts_tensor=ts_tensor,
                top_p=top_p,
            )
            
            # Assign responses to states
            for state, response in zip(states_needing_reasoner, reasoner_responses):
                state["reasoner_completions"].append(response)
                state["previous_reasoner_answer"] = response
                # Parse answer
                parsed = self._parse_reasoner_response(response)
                state["reasoner_answers"].append(parsed)

    def _build_batch_results(self, batch_states: List[dict]) -> List[dict]:
        """Build final results from batch states."""
        results = []
        
        for state in batch_states:
            # Get final metadata
            final_controller_prompt = (
                state["controller_prompts"][-1] if state["controller_prompts"] else None
            )
            final_reasoner_prompt = (
                state["reasoner_prompts"][-1] if state["reasoner_prompts"] else None
            )
            final_segment_encodings = None #state["segment_memory"].get_all_embeddings()
            final_segments = state["segment_memory"].get_all_segments()
            
            # Clear memory
            state["segment_memory"].clear()
            
            hit_max_rounds = (
                state["round_num"] >= self.max_rounds and
                not state["format_error_occurred"] and
                (state["decision"].get("decision") != "accept" if state["decision"] else False)
            )
            
            # Return same structure as generate_controller_reasoner_loop
            result = {
                "controller_completions": state["controller_completions"],
                "reasoner_completions": state["reasoner_completions"],
                "full_conv_for_controller": state["full_conv_for_controller"],
                "controller_prompt_ids": state["controller_prompt_ids"],
                "controller_completion_ids": state["controller_completion_ids"],
                "reasoner_answers": state["reasoner_answers"],
                "final_segments": final_segments,
                "segment_encodings": final_segment_encodings,
                "num_rounds": state["round_num"],
                "format_error_occurred": state["format_error_occurred"],
                "format_error_round": state["format_error_round"],
                "hit_max_rounds": hit_max_rounds,
                "has_accept": (
                    state["decision"].get("decision") == "accept" 
                    if state["decision"] and not state["format_error_occurred"] 
                    else False
                ),
                "final_controller_prompt": final_controller_prompt,
                "final_reasoner_prompt": final_reasoner_prompt,
                "skipped": state["skipped"],
                "skip_reason": state["skip_reason"],
                "exclude_from_loss": state.get("skipped", False),
            }
            
            results.append(result)
        
        return results

    def _batch_generate_reasoner_with_different_segments(
        self,
        states_to_process: List[dict],
        state_indices: List[int],
        question: str,
        ts_tensor,
        top_p: float,
    ):
        """
        Batch reasoner generation even when segments differ.
        Uses padding to handle different segment counts.
        
        Args:
            states_to_process: List of state dicts that need reasoner generation
            state_indices: Original indices in batch_states (for logging)
            question: Question text
            ts_tensor: Time series tensor
            question_ids: Question token IDs
            top_p: Top-p sampling parameter
            
        Returns:
            List[str]: Generated reasoner responses
        """
        if not states_to_process:
            return []
        
        batch_size = len(states_to_process)
        # Collect all segment lists and encodings
        all_segments_list = [s["segment_memory"].get_all_segments() for s in states_to_process]
        
        # Create prompts (different due to different segments)
        prompts = []
        for segments in all_segments_list:
            prompt = self._create_reasoner_prompt(question, segments)
            sys_content_reasoner = 'you are a helpful assistant that can answer questions about time series data.'
            messages_reasoner = [
                {"role": "system", "content": sys_content_reasoner},
                {"role": "user", "content": prompt}
            ]
            prompt_formatted = self.tokenizer.apply_chat_template(
                messages_reasoner,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(prompt_formatted)
        
        # Save prompts to states
        for state, prompt in zip(states_to_process, prompts):
            state["reasoner_prompts"].append(prompt)
        
        # Batch tokenize with padding
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.model.device)
        
        initial_length = toks.input_ids.shape[1]
        ts_for_model = self._prepare_ts_segments_for_model(
        ts_tensor.unsqueeze(0).repeat(batch_size, 1, 1) if ts_tensor.dim() == 2 else ts_tensor.repeat(batch_size, 1, 1, 1),
        all_segments_list  # List of segment lists (one per sample)
        )
        if ts_for_model is not None:
            if self.task_name == "TSQA" or self.task_name == "TRQA_MIXED":
                ts_for_model = ts_for_model.to(self.model.device, dtype=torch.bfloat16)
            else:
                ts_for_model = ts_for_model.to(self.model.device)
        
        stop = StoppingCriteriaList([
        ReasonerStoppingCriteria(
            self.tokenizer,
            initial_length=initial_length  # ← CRITICAL FIX
        )
        ])
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=toks.input_ids,
                attention_mask=toks.attention_mask,
                timeseries=ts_for_model,
                max_new_tokens=self.reasoner_max_new_tokens,
                temperature=self.reasoner_temperature,
                do_sample=True,
                stopping_criteria=stop,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                top_p=top_p,
            )
        
        # Decode responses
        responses = []
        for i in range(batch_size):
            new_tokens = gen_ids[i, initial_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)
        
        # Cleanup
        del toks, ts_for_model, gen_ids
        torch.cuda.empty_cache()
        
        return responses
    
    def batch_reasoner_sft_prompt(
        self,
        states_to_process: List[dict],
        state_indices: List[int],
        question: str,
        ts_tensor,
        question_ids: List[int],
        top_p: float,
    ):
        """
        Batch reasoner generation even when segments differ.
        Uses padding to handle different segment counts.
        
        Args:
            states_to_process: List of state dicts that need reasoner generation
            state_indices: Original indices in batch_states (for logging)
            question: Question text
            ts_tensor: Time series tensor
            question_ids: Question token IDs
            top_p: Top-p sampling parameter
            
        Returns:
            List[str]: Generated reasoner responses
        """
        if not states_to_process:
            return []
        
        # Collect all segment lists and encodings
        all_segments = [s["segment_memory"].get_all_segments() for s in states_to_process]
        all_encodings = [s["segment_memory"].get_all_embeddings() for s in states_to_process]
        ts_placeholder_id = self.tokenizer.encode(TS_PLACEHOLDER, add_special_tokens=False)[0]
        # Create prompts (different due to different segments)
        prompts = []
        prompts2 = []
        for segments in all_segments:
            prompt = self._create_reasoner_prompt(question, segments)
            # One TS_PLACEHOLDER per segment (matching SFT format)
            # segment_placeholders = "\n".join([TS_PLACEHOLDER for _ in segments])
            prompt2 = self.create_reasoner_mcq_generic_sft_style_prompt(question, segments) #f"{segment_placeholders}\n{question}"
            sys_content_reasoner = 'you are a helpful assistant that can answer questions about time series data.'
            messages_reasoner = [
                {"role": "system", "content": sys_content_reasoner},
                {"role": "user", "content": prompt}
            ]
            messages_reasoner2 = [
                {"role": "system", "content": sys_content_reasoner},
                {"role": "user", "content": prompt2}
            ]
            prompt_formatted = self.tokenizer.apply_chat_template(
                messages_reasoner,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_formatted2 = self.tokenizer.apply_chat_template(
                messages_reasoner2,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(prompt_formatted)
            prompts2.append(prompt_formatted2)
        
        # Save prompts to states
        for state, prompt in zip(states_to_process, prompts):
            state["reasoner_prompts"].append(prompt)
        
        # Batch tokenize with padding
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.model.device)

        toks2 = self.tokenizer(
            prompts2,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.model.device)
        
        text_emb = self.model.model.embed_tokens(toks.input_ids)
        ph_mask = (toks.input_ids == ts_placeholder_id)

        text_emb2 = self.model.model.embed_tokens(toks2.input_ids)
        ph_mask2 = (toks2.input_ids == ts_placeholder_id)
        
        # Inject different segments per sample
        batch_ts_embeddings = []
        for encodings in all_encodings:
            encodings_gpu = [enc.to(self.model.device) for enc in encodings]
            batch_ts_embeddings.append(encodings_gpu)
        
        combined_emb, combined_attn, _ = self.inject_ts_segments_and_labels(
            text_embeddings=text_emb,
            placeholder_mask=ph_mask,
            ts_embeddings=batch_ts_embeddings,
            labels=None,
            attention_mask=toks.attention_mask,
        )

        combined_emb2, combined_attn2, _ = self.inject_ts_segments_and_labels(
            text_embeddings=text_emb2,
            placeholder_mask=ph_mask2,
            ts_embeddings=batch_ts_embeddings,
            labels=None,
            attention_mask=toks2.attention_mask,
        )
        
        # Batch generate
        stop = StoppingCriteriaList([ReasonerStoppingCriteria(self.tokenizer)])
        
        # print(f"  Batch generating {len(states_to_process)} reasoner responses...")
        
        with torch.no_grad():
            gen_ids = self.model.generate(
                inputs_embeds=combined_emb,
                attention_mask=combined_attn,
                max_new_tokens=self.reasoner_max_new_tokens,
                temperature=self.reasoner_temperature,
                do_sample=True,
                stopping_criteria=stop,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                top_p=top_p,
            )
            gen_ids2 = self.model.generate(
                inputs_embeds=combined_emb2,
                attention_mask=combined_attn2,
                max_new_tokens=self.reasoner_max_new_tokens,
                temperature=self.reasoner_temperature,
                do_sample=True,
                stopping_criteria=stop,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                top_p=top_p,
            )

        # Decode responses
        responses = [
            self.tokenizer.decode(gen_ids[i], skip_special_tokens=True)
            for i in range(len(states_to_process))
        ]
        responses2 = [
            self.tokenizer.decode(gen_ids2[i], skip_special_tokens=True)
            for i in range(len(states_to_process))
        ]
        # Cleanup
        for encodings_gpu in batch_ts_embeddings:
            del encodings_gpu
        del batch_ts_embeddings
        del toks, text_emb, combined_emb, combined_attn, gen_ids
        torch.cuda.empty_cache()
        
        return responses
    
    def _validate_segment(
        self,
        state: dict,
        idx: int,
        decision: dict,
        round_num: int,
        ts_tensor,
    ) -> bool:
        """
        Validate segment and encode it. Returns True if successful.
        Marks state as done if validation fails.
        """
        segment = decision["segment"]
        
        # Validate segment
        if not state["segment_memory"]._is_valid_segment(segment):
            print(f"    Loop {idx}: Invalid segment {segment}")
            state["format_error_occurred"] = True
            state["format_error_round"] = round_num
            state["is_done"] = True
            return False
        
        start, end = segment
        start, end = max(0, start), min(end, ts_tensor.shape[0])
        segment = [start, end]
        
        # Add to memory
        print(f"    Loop {idx}: Adding segment {segment}")
        success = state["segment_memory"].add_segment(segment)
        if not success:
            print(f"    Loop {idx}: Failed to add segment")
            state["format_error_occurred"] = True
            state["format_error_round"] = round_num
            state["is_done"] = True
            return False
        
        return True