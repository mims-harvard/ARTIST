
import torch
import numpy as np
import json
import time
from typing import List, Dict, Any, Optional
from transformers import StoppingCriteria, StoppingCriteriaList
from clean_mem import log_mem, cleanup_mem 
import GPUtil
import gc
from qwents_utils import (
    extract_answer,
    extract_answer_letter,
    parse_tool_segs,
    in_bounds,
    TS_PLACEHOLDER
)
from constants import TOOL_CALL_TOKEN, TOOL_CALL_END_TOKEN
from tools_qwents import TS_TOOLS
from tools_runtime import ToolDispatcher

def _safe_del(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
        
def log_gpu(label=""):
    torch_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    try:
        gpu = GPUtil.getGPUs()[0]
        system_mem = gpu.memoryUsed / 1024
        util = gpu.memoryUtil * 100
        print(f"[{label:15s}] Torch:{torch_mem:.2f}GB | System:{system_mem:.2f}GB | Util:{util:.0f}%")
    except:
        print(f"[{label:15s}] Torch:{torch_mem:.2f}GB")


def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract <tool_call>…</tool_call> JSON blocks using your constants."""
    calls = []
    start = 0
    while True:
        s = text.find(TOOL_CALL_TOKEN, start)
        if s == -1: break
        e = text.find(TOOL_CALL_END_TOKEN, s)
        if e == -1: break
        payload = text[s+len(TOOL_CALL_TOKEN):e].strip()
        try:
            obj = json.loads(payload)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                calls.append({"name": obj["name"], "arguments": obj["arguments"]})
        except Exception:
            pass
        start = e + len(TOOL_CALL_END_TOKEN)
    return calls

class ToolStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_id_seqs = [
            tokenizer.encode("</tool_call>", add_special_tokens=False),
            tokenizer.encode("</answer>", add_special_tokens=False),
            tokenizer.encode("<|im_end|>", add_special_tokens=False),
        ]

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        for stop in self.stop_id_seqs:
            L = len(stop)
            if L <= len(seq) and seq[-L:] == stop:
                return True
        return False

class BatchToolStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_id_seqs = [
            tokenizer.encode("</tool_call>", add_special_tokens=False),
            tokenizer.encode("</answer>", add_special_tokens=False),
            tokenizer.encode("<|im_end|>", add_special_tokens=False),
        ]

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids: [B, T]
        B = input_ids.size(0)
        for b in range(B):
            seq = input_ids[b].tolist()
            for stop in self.stop_id_seqs:
                L = len(stop)
                if L <= len(seq) and seq[-L:] == stop:
                    # this b is "done", but HF stopping criteria can only stop globally
                    # so we stop only when ALL are done
                    break
            else:
                # no stop matched for this b => not all done
                return False
        return True

class QwenTSGenerationHandler:
    """
    Handles the complex generation process with TS injection for RL training
    """
    def __init__(self, model, tokenizer, task="1TS", dt=1.0, get_tokens_fn=None):
        self.MAX_SEQ_LEN = 7000
        self.MAX_ROUNDS = 4
        self.model = model
        self.tokenizer = tokenizer
        self.first_gen = True
        self.task = task
        self.dt = float(dt)
        self.min_segment_length = 60
        self.get_tokens_fn = get_tokens_fn
        self.ts_placeholder_id = tokenizer.encode(TS_PLACEHOLDER, add_special_tokens=False)[0]
        # Timing statistics
        self.timing_stats = {
            'ts_preparation': [],
            'question_embedding': [],
            'initial_ts_injection': [],
            'generation_rounds': [],
            'tool_parsing': [],
            'ts_segment_encoding': [],
            'embedding_concatenation': []
        }
    
    def _should_continue_generation(self, cur_emb, round_num):
        """Check if generation should continue based on sequence length and rounds"""
        seq_len = cur_emb.shape[1]
        
        if round_num >= self.MAX_ROUNDS:
            return False, f"max_rounds_reached_{round_num}"
        
        if seq_len >= self.MAX_SEQ_LEN:
            return False, f"max_length_reached_{seq_len}"
        
        # Early warning if getting close
        if seq_len > self.MAX_SEQ_LEN * 0.8:  # 4800 tokens
            print(f"Warning: sequence length {seq_len} approaching limit")
        
        return True, None

    def generate_with_ts_injection(self, prompt: str, ts_data: np.ndarray, question_ids: List[int],
                                   max_new_tokens: int = 512, temperature: float = 0.7, max_rounds: int = 3, timestamps: Dict[str, Any] = None, raw_ts: np.ndarray = None) -> str:
        
        try:
            # log_gpu("START")
            # TS + question embeddings
            ts_tensor = self._prepare_ts_tensor(ts_data)
            # log_gpu("TS_PREP")
            q_emb = self._get_question_embedding(question_ids)
            # log_gpu("Q_EMB")
            # dispatcher = ToolDispatcher(series=np.array(raw_ts), dt=self.dt,timestamps=timestamps) # the tool dispatcher does not handle the tokenized ts_data calls, so we use the raw ts

            # Tokenize + initial full-series injection at TS_PLACEHOLDER
            toks = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(self.model.device)
            cur_emb, cur_attn = self._inject_initial_ts(toks.input_ids, toks.attention_mask, ts_tensor, q_emb)
            
            del toks
            # log_gpu("INITIAL_INJECT")
            final_text = ""
            entropies = []

            for round_num in range(max_rounds):
                try:
                    can_continue, reason = self._should_continue_generation(cur_emb, round_num)
                    if not can_continue:
                        print(f"Stopping generation: {reason}")
                        # final_text += f"\n<answer>Analysis completed due to {reason.replace('_', ' ')}</answer>"
                        break
                    # log_gpu(f"ROUND_{round_num}")
                    stop = StoppingCriteriaList([ToolStoppingCriteria(self.tokenizer)])
                    with torch.no_grad():
                        # log_gpu(f"PRE_GEN_{round_num}")                        
                        gen_ids = self.model.generate(
                            inputs_embeds=cur_emb,
                            attention_mask=cur_attn,
                            position_ids=None,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            stopping_criteria=stop,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                        )
                        # log_gpu(f"POST_GEN_{round_num}")
                      
                    # entropy = self._calculate_entropy_from_embeddings(cur_emb, cur_attn)
                    # entropies.append(entropy)
                    # log_gpu(f"AFTER_GEN_{round_num}") 
                    new_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                    # log_gpu(f"POST_DECODE_{round_num}")
                    final_text += new_text
                    if self.first_gen:
                        print(f"First gen: {new_text}")
                        self.first_gen = False

                    # Final?
                    if "</answer>" in new_text:
                        del gen_ids
                        # log_gpu("FINAL") 
                        break

                    # Parse tool calls
                    calls = _extract_tool_calls(new_text)
                    
                    if not calls:
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed = (base.model.embed_tokens)(gen_ids)
                        cur_emb = torch.cat([cur_emb, embed], dim=1)
                        cur_attn = torch.cat([cur_attn, torch.ones_like(gen_ids, dtype=cur_attn.dtype)], dim=1)
                        del embed, gen_ids
                        continue

                    valid_zoom_calls, valid_other_calls = self._split_and_validate_calls(calls, ts_tensor.shape[1])

                    if valid_zoom_calls or valid_other_calls:
                        # run non-zoom tools and fold results into one assistant message prefix
                        # tools_blob = self._format_tool_results(valid_other_calls, dispatcher) if valid_other_calls else ""
                        zoom_block = self._format_zoom_block(valid_zoom_calls) if valid_zoom_calls else ""

                        after_tool_text = (
                        # ((tools_blob + "\n") if tools_blob else "") +
                        ((zoom_block + "\n") if zoom_block else "") +
                        "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. "
                        "Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> "
                        "(if any tools needed) OR <answer>...</answer> (if no tools needed)."
                    )
                        combined_msgs = [{"role": "user", "content": after_tool_text}]

                        # tokenize that single assistant message
                        combined_text = self.tokenizer.apply_chat_template(
                            combined_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                        )
                        combined_tok = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

                        if valid_zoom_calls:
                            segs = [c["ts_seg"] for c in valid_zoom_calls]
                            block_emb, block_attn = self._inject_tool_segments_og(
                                combined_tok.input_ids, combined_tok.attention_mask,
                                ts_tensor, q_emb, segs
                            )
                        else:
                            base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                            embed_fn = (base.model.embed_tokens)
                            block_emb  = embed_fn(combined_tok.input_ids)
                            block_attn = combined_tok.attention_mask

                        # append the raw generated ids as tokens first, then the combined assistant block
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed_fn = (base.model.embed_tokens)
                        delta_emb  = embed_fn(gen_ids)  # from the current round
                        delta_attn = torch.ones(gen_ids.size(0), gen_ids.size(1), device=self.model.device, dtype=cur_attn.dtype)

                        projected_length = cur_emb.shape[1] + delta_emb.shape[1] + block_emb.shape[1]
                        if projected_length > self.MAX_SEQ_LEN:
                            print(f"Skipping concatenation: projected length {projected_length} exceeds limit")
                            final_text += "\n<answer>Analysis completed due to length constraints</answer>"
                            del combined_tok, block_emb, block_attn, delta_emb, delta_attn, gen_ids
                            break

                        cur_emb  = torch.cat([cur_emb,  delta_emb,  block_emb],  dim=1)
                        cur_attn = torch.cat([cur_attn, delta_attn, block_attn], dim=1)

                        del combined_tok, block_emb, block_attn, delta_emb, delta_attn, gen_ids
                        # log_gpu(f"TOOLS_DONE_{round_num}")
                    else:
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed_fn = (base.model.embed_tokens)
                        delta_emb  = embed_fn(gen_ids)  # from the current round
                        delta_attn = torch.ones(gen_ids.size(0), gen_ids.size(1), device=self.model.device, dtype=cur_attn.dtype)
                        cur_emb  = torch.cat([cur_emb,  delta_emb],  dim=1)
                        cur_attn = torch.cat([cur_attn, delta_attn], dim=1)

                        del delta_emb, delta_attn, gen_ids
                        # log_gpu(f"NO_TOOLS_{round_num}")
                        
                except Exception as round_e:
                    print(f"ERROR in generation round {round_num}: {round_e}")
                    print(f"cur_emb.shape: {cur_emb.shape if 'cur_emb' in locals() else 'Not available'}")
                    print(f"cur_attn.shape: {cur_attn.shape if 'cur_attn' in locals() else 'Not available'}")
                    if torch.cuda.is_available():
                        torch_mem = torch.cuda.memory_allocated() / (1024**3)
                        torch_cached = torch.cuda.memory_reserved() / (1024**3)
                        print(f"GPU memory - Allocated: {torch_mem:.2f}GB, Cached: {torch_cached:.2f}GB")
                    raise round_e
            
            
            # print("finishing")
            
            # log_gpu("END")
            try:
                # print("current_emb.shape: ", cur_emb.shape)
                print("round_num: ", round_num)
            except Exception as e:
                print(f"Error printing debug info: {e}")
            
            torch.cuda.empty_cache()
            gc.collect()
            return final_text, entropies
            
        except Exception as e:
            print(f"ERROR in generate_with_ts_injection: {e}")
            print(f"cur_emb.shape: {cur_emb.shape if 'cur_emb' in locals() else 'Not available'}")
            print(f"cur_attn.shape: {cur_attn.shape if 'cur_attn' in locals() else 'Not available'}")
            print(f"ts_tensor.shape: {ts_tensor.shape if 'ts_tensor' in locals() else 'Not available'}")
            if torch.cuda.is_available():
                torch_mem = torch.cuda.memory_allocated() / (1024**3)
                torch_cached = torch.cuda.memory_reserved() / (1024**3)
                print(f"GPU memory - Allocated: {torch_mem:.2f}GB, Cached: {torch_cached:.2f}GB")
            raise e
    
    def generate_with_ts_injection_grpo(self, prompt: str, ts_data: np.ndarray, question_ids: List[int],
                                   max_new_tokens: int = 512, temperature: float = 0.7, max_rounds: int = 3, timestamps: Dict[str, Any] = None, raw_ts: np.ndarray = None, initial_messages: List[Dict[str, Any]] = None) -> str:
        
        try:
            # log_gpu("START")
            # TS + question embeddings
            ts_tensor = self._prepare_ts_tensor(ts_data)
            # log_gpu("TS_PREP")
            q_emb = self._get_question_embedding(question_ids)
            # log_gpu("Q_EMB")
            # dispatcher = ToolDispatcher(series=np.array(raw_ts), dt=self.dt,timestamps=timestamps) # the tool dispatcher does not handle the tokenized ts_data calls, so we use the raw ts

            # Tokenize + initial full-series injection at TS_PLACEHOLDER
            toks = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(self.model.device)
            cur_emb, cur_attn = self._inject_initial_ts(toks.input_ids, toks.attention_mask, ts_tensor, q_emb)
            
            del toks
            # log_gpu("INITIAL_INJECT")
            final_text = ""
            entropies = []
            full_conv = initial_messages

            for round_num in range(max_rounds):
                try:
                    can_continue, reason = self._should_continue_generation(cur_emb, round_num)
                    if not can_continue:
                        print(f"Stopping generation: {reason}")
                        # final_text += f"\n<answer>Analysis completed due to {reason.replace('_', ' ')}</answer>"
                        break
                    # log_gpu(f"ROUND_{round_num}")
                    stop = StoppingCriteriaList([ToolStoppingCriteria(self.tokenizer)])
                    with torch.no_grad():
                        # log_gpu(f"PRE_GEN_{round_num}")                        
                        gen_ids = self.model.generate(
                            inputs_embeds=cur_emb,
                            attention_mask=cur_attn,
                            position_ids=None,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            stopping_criteria=stop,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                        )
                        # log_gpu(f"POST_GEN_{round_num}")
                      
                    # entropy = self._calculate_entropy_from_embeddings(cur_emb, cur_attn)
                    # entropies.append(entropy)
                    # log_gpu(f"AFTER_GEN_{round_num}") 
                    new_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                    full_conv.append({"role": "assistant", "content": new_text})
                    # log_gpu(f"POST_DECODE_{round_num}")
                    final_text += new_text
                    if self.first_gen:
                        print(f"First gen: {new_text}")
                        self.first_gen = False

                    # Final?
                    if "</answer>" in new_text:
                        del gen_ids
                        # log_gpu("FINAL") 
                        break

                    # Parse tool calls
                    calls = _extract_tool_calls(new_text)
                    
                    if not calls:
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed = (base.model.embed_tokens)(gen_ids)
                        cur_emb = torch.cat([cur_emb, embed], dim=1)
                        cur_attn = torch.cat([cur_attn, torch.ones_like(gen_ids, dtype=cur_attn.dtype)], dim=1)
                        del embed, gen_ids
                        continue

                    valid_zoom_calls, valid_other_calls = self._split_and_validate_calls(calls, ts_tensor.shape[1])

                    if valid_zoom_calls or valid_other_calls:
                        # run non-zoom tools and fold results into one assistant message prefix
                        # tools_blob = self._format_tool_results(valid_other_calls, dispatcher) if valid_other_calls else ""
                        zoom_block = self._format_zoom_block(valid_zoom_calls) if valid_zoom_calls else ""

                        after_tool_text = (
                        # ((tools_blob + "\n") if tools_blob else "") +
                        ((zoom_block + "\n") if zoom_block else "") +
                        "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. "
                        "Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> "
                        "(if any tools needed) OR <answer>...</answer> (if no tools needed)."
                    )
                        combined_msgs = [{"role": "user", "content": after_tool_text}]
                        full_conv.append({"role": "user", "content": after_tool_text})

                        # tokenize that single assistant message
                        combined_text = self.tokenizer.apply_chat_template(
                            combined_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                        )
                        combined_tok = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

                        if valid_zoom_calls:
                            segs = [c["ts_seg"] for c in valid_zoom_calls]
                            block_emb, block_attn = self._inject_tool_segments_og(
                                combined_tok.input_ids, combined_tok.attention_mask,
                                ts_tensor, q_emb, segs
                            )
                        else:
                            base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                            embed_fn = (base.model.embed_tokens)
                            block_emb  = embed_fn(combined_tok.input_ids)
                            block_attn = combined_tok.attention_mask

                        # append the raw generated ids as tokens first, then the combined assistant block
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed_fn = (base.model.embed_tokens)
                        delta_emb  = embed_fn(gen_ids)  # from the current round
                        delta_attn = torch.ones(gen_ids.size(0), gen_ids.size(1), device=self.model.device, dtype=cur_attn.dtype)

                        projected_length = cur_emb.shape[1] + delta_emb.shape[1] + block_emb.shape[1]
                        if projected_length > self.MAX_SEQ_LEN:
                            print(f"Skipping concatenation: projected length {projected_length} exceeds limit")
                            final_text += "\n<answer>Analysis completed due to length constraints</answer>"
                            del combined_tok, block_emb, block_attn, delta_emb, delta_attn, gen_ids
                            break

                        cur_emb  = torch.cat([cur_emb,  delta_emb,  block_emb],  dim=1)
                        cur_attn = torch.cat([cur_attn, delta_attn, block_attn], dim=1)

                        del combined_tok, block_emb, block_attn, delta_emb, delta_attn, gen_ids
                        # log_gpu(f"TOOLS_DONE_{round_num}")
                    else:
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed_fn = (base.model.embed_tokens)
                        delta_emb  = embed_fn(gen_ids)  # from the current round
                        delta_attn = torch.ones(gen_ids.size(0), gen_ids.size(1), device=self.model.device, dtype=cur_attn.dtype)
                        cur_emb  = torch.cat([cur_emb,  delta_emb],  dim=1)
                        cur_attn = torch.cat([cur_attn, delta_attn], dim=1)

                        del delta_emb, delta_attn, gen_ids
                        # log_gpu(f"NO_TOOLS_{round_num}")
                        
                except Exception as round_e:
                    print(f"ERROR in generation round {round_num}: {round_e}")
                    print(f"cur_emb.shape: {cur_emb.shape if 'cur_emb' in locals() else 'Not available'}")
                    print(f"cur_attn.shape: {cur_attn.shape if 'cur_attn' in locals() else 'Not available'}")
                    if torch.cuda.is_available():
                        torch_mem = torch.cuda.memory_allocated() / (1024**3)
                        torch_cached = torch.cuda.memory_reserved() / (1024**3)
                        print(f"GPU memory - Allocated: {torch_mem:.2f}GB, Cached: {torch_cached:.2f}GB")
                    raise round_e
            
            
            torch.cuda.empty_cache()
            gc.collect()
            return final_text, entropies, full_conv
            
        except Exception as e:
            print(f"ERROR in generate_with_ts_injection: {e}")
            print(f"cur_emb.shape: {cur_emb.shape if 'cur_emb' in locals() else 'Not available'}")
            print(f"cur_attn.shape: {cur_attn.shape if 'cur_attn' in locals() else 'Not available'}")
            print(f"ts_tensor.shape: {ts_tensor.shape if 'ts_tensor' in locals() else 'Not available'}")
            if torch.cuda.is_available():
                torch_mem = torch.cuda.memory_allocated() / (1024**3)
                torch_cached = torch.cuda.memory_reserved() / (1024**3)
                print(f"GPU memory - Allocated: {torch_mem:.2f}GB, Cached: {torch_cached:.2f}GB")
            raise e

    def generate_with_ts_injection_grpo_batch(
        self,
        prompts,
        ts_data_list,
        question_ids_list,
        max_new_tokens=512,
        temperature=0.7,
        max_rounds=3,
        timestamps_list=None,
        raw_ts_list=None,
        initial_messages_list=None,
    ):
        B = len(prompts)
        if timestamps_list is None: timestamps_list = [None] * B
        if raw_ts_list is None: raw_ts_list = [None] * B
        if initial_messages_list is None: initial_messages_list = [[] for _ in range(B)]

        full_convs = [list(m) for m in initial_messages_list]
        final_texts = [""] * B
        done = [False] * B

        device = self.model.device
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        embed_fn = base.model.embed_tokens

        # --------------------------
        # 1) SEQUENTIAL initial state construction (NO padding artifacts)
        # --------------------------
        cur_emb_list = []
        cur_attn_list = []
        ts_lengths = []

        # Tokenize prompts per-sample (so we don't introduce prompt padding artifacts either)
        for i in range(B):
            ts_tensor_i = self._prepare_ts_tensor(ts_data_list[i])  # [1, Li, C] + dtype logic as before
            ts_lengths.append(ts_tensor_i.shape[1])

            q_emb_i = self._get_question_embedding(question_ids_list[i])  # [1, Qi, D] as before (no padding)

            toks_i = self.tokenizer(
                prompts[i],
                return_tensors="pt",
                padding=False,
                truncation=True
            ).to(device)

            cur_emb_i, cur_attn_i = self._inject_initial_ts(
                toks_i.input_ids,
                toks_i.attention_mask,
                ts_tensor_i,
                q_emb_i
            )

            cur_emb_list.append(cur_emb_i.contiguous())     # [1, Li', D]
            cur_attn_list.append(cur_attn_i.contiguous())   # [1, Li']

            del toks_i, ts_tensor_i, q_emb_i

        # Helper: pad ragged [1, Li, D] -> [Ba, Lmax, D]
        def _pad_batch_emb(embed_list, attn_list):
            Ba = len(embed_list)
            D = embed_list[0].size(-1)
            Lmax = max(e.size(1) for e in embed_list)
            out_emb = torch.zeros(Ba, Lmax, D, device=device, dtype=embed_list[0].dtype)
            out_attn = torch.zeros(Ba, Lmax, device=device, dtype=attn_list[0].dtype)
            for j, (e, a) in enumerate(zip(embed_list, attn_list)):
                Lj = e.size(1)
                out_emb[j, :Lj] = e[0]
                out_attn[j, :Lj] = a[0]
            return out_emb, out_attn

        # --------------------------
        # 2) Batched generation rounds (pad only at embedding level)
        # --------------------------
        for round_num in range(max_rounds):
            active = [i for i in range(B) if not done[i]]
            if not active:
                break

            emb_a, attn_a = _pad_batch_emb(
                [cur_emb_list[i] for i in active],
                [cur_attn_list[i] for i in active],
            )

            stop = StoppingCriteriaList([BatchToolStoppingCriteria(self.tokenizer)])
            with torch.no_grad():
                gen_ids_a = self.model.generate(
                    inputs_embeds=emb_a,
                    attention_mask=attn_a,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stop,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            _safe_del(emb_a, attn_a)

            pad_id = self.tokenizer.pad_token_id

            # 4) Trim per-sample ids (and optionally decode from trimmed ids)
            #    NOTE: decode can be CPU-side to avoid holding big GPU tensors longer.
            trimmed_ids_list = []
            trimmed_lens = []

            if pad_id is None:
                # Fallback: no padding token defined => assume full length
                for j in range(gen_ids_a.size(0)):
                    ids_j = gen_ids_a[j]
                    tlen = ids_j.numel()
                    trimmed_lens.append(tlen)
                    trimmed_ids_list.append(ids_j[:tlen].detach())
            else:
                for j in range(gen_ids_a.size(0)):
                    ids_j = gen_ids_a[j]
                    # assumes right-padding with pad_id; this is standard for HF generate batching
                    tlen = int((ids_j != pad_id).sum().item())
                    trimmed_lens.append(tlen)
                    trimmed_ids_list.append(ids_j[:tlen].detach())

            # Decode from trimmed ids (option A: decode on CPU to reduce GPU residency)
            trimmed_ids_cpu = [t.cpu() for t in trimmed_ids_list]
            new_texts_a = self.tokenizer.batch_decode(trimmed_ids_cpu, skip_special_tokens=False)

            # 5) Now that we've extracted trimmed ids and decoded, free big batched gen tensor ASAP
            _safe_del(gen_ids_a, trimmed_ids_cpu)

            # 6) Append trimmed embeddings to each sample’s state + handle tool logic
            for j, idx in enumerate(active):
                txt = new_texts_a[j]
                full_convs[idx].append({"role": "assistant", "content": txt})
                final_texts[idx] += txt

                tlen = trimmed_lens[j]
                if tlen > 0:
                    # keep trimmed ids on GPU for embedding (they already are if trimmed_ids_list came from GPU)
                    ids_trim = trimmed_ids_list[j].unsqueeze(0).to(device)  # [1, tlen]
                    delta_emb = embed_fn(ids_trim)                          # [1, tlen, D]
                    delta_attn = torch.ones(
                        1, tlen, device=device, dtype=cur_attn_list[idx].dtype
                    )
                    cur_emb_list[idx] = torch.cat([cur_emb_list[idx], delta_emb], dim=1)
                    cur_attn_list[idx] = torch.cat([cur_attn_list[idx], delta_attn], dim=1)

                    _safe_del(ids_trim, delta_emb, delta_attn)

                if "</answer>" in txt:
                    done[idx] = True
                    continue

                calls = _extract_tool_calls(txt)
                if not calls:
                    continue

                valid_zoom_calls, valid_other_calls = self._split_and_validate_calls(
                    calls, ts_len=int(ts_lengths[idx])
                )
                zoom_block = self._format_zoom_block(valid_zoom_calls) if valid_zoom_calls else ""

                after_tool_text = (
                    ((zoom_block + "\n") if zoom_block else "") +
                    "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. "
                )
                full_convs[idx].append({"role": "user", "content": after_tool_text})

                combined_msgs = [{"role": "user", "content": after_tool_text}]
                combined_text = self.tokenizer.apply_chat_template(
                    combined_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                combined_tok = self.tokenizer(
                    combined_text, return_tensors="pt", add_special_tokens=False
                ).to(device)

                if valid_zoom_calls:
                    segs = [c["ts_seg"] for c in valid_zoom_calls]
                    ts_tensor_i = self._prepare_ts_tensor(ts_data_list[idx])
                    q_emb_i = self._get_question_embedding(question_ids_list[idx])
                    block_emb, block_attn = self._inject_tool_segments_og(
                        combined_tok.input_ids, combined_tok.attention_mask,
                        ts_tensor_i, q_emb_i, segs
                    )
                    _safe_del(ts_tensor_i, q_emb_i)
                else:
                    block_emb = embed_fn(combined_tok.input_ids)
                    block_attn = combined_tok.attention_mask

                projected = cur_emb_list[idx].size(1) + block_emb.size(1)
                if projected > self.MAX_SEQ_LEN:
                    final_texts[idx] += "\n<answer>Analysis completed due to length constraints</answer>"
                    done[idx] = True
                else:
                    cur_emb_list[idx] = torch.cat([cur_emb_list[idx], block_emb], dim=1)
                    cur_attn_list[idx] = torch.cat([cur_attn_list[idx], block_attn], dim=1)

                _safe_del(combined_tok, block_emb, block_attn)

            # 7) Free trimmed ids list (they can be large over time if kept around)
            _safe_del(trimmed_ids_list, trimmed_lens, new_texts_a)
        
        torch.cuda.empty_cache()

        return final_texts, full_convs


    def generate_with_ts_injection_test(self, prompt: str, ts_data: np.ndarray, question_ids: List[int],
                                   max_new_tokens: int = 512, temperature: float = 0.3, max_rounds: int = 3, timestamps: Dict[str, Any] = None, raw_ts: np.ndarray = None) -> str:
        
        try:
            # log_gpu("START")
            # TS + question embeddings
            ts_tensor = self._prepare_ts_tensor(ts_data)
            # log_gpu("TS_PREP")
            q_emb = self._get_question_embedding_test(question_ids)
            # log_gpu("Q_EMB")
            # dispatcher = ToolDispatcher(series=np.array(raw_ts), dt=self.dt,timestamps=timestamps) # the tool dispatcher does not handle the tokenized ts_data calls, so we use the raw ts

            # Tokenize + initial full-series injection at TS_PLACEHOLDER
            toks = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(self.model.device)
            cur_emb, cur_attn = self._inject_initial_ts_test(toks.input_ids, toks.attention_mask, ts_tensor, q_emb)
            
            del toks
            # log_gpu("INITIAL_INJECT")
            final_text = ""
            entropies = []

            for round_num in range(max_rounds):
                try:
                    can_continue, reason = self._should_continue_generation(cur_emb, round_num)
                    if not can_continue:
                        print(f"Stopping generation: {reason}")
                        # final_text += f"\n<answer>Analysis completed due to {reason.replace('_', ' ')}</answer>"
                        break
                    # log_gpu(f"ROUND_{round_num}")
                    stop = StoppingCriteriaList([ToolStoppingCriteria(self.tokenizer)])
                    with torch.no_grad():
                        # log_gpu(f"PRE_GEN_{round_num}")                        
                        gen_ids = self.model.generate(
                            inputs_embeds=cur_emb,
                            attention_mask=cur_attn,
                            position_ids=None,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            stopping_criteria=stop,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True,
                            top_p=0.9
                        )
                        # log_gpu(f"POST_GEN_{round_num}")
                      
                    entropy = self._calculate_entropy_from_embeddings(cur_emb, cur_attn)
                    entropies.append(entropy)
                    # log_gpu(f"AFTER_GEN_{round_num}") 
                    new_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                    # log_gpu(f"POST_DECODE_{round_num}")
                    final_text += new_text
                    if self.first_gen:
                        print(f"First gen: {new_text}")
                        self.first_gen = False

                    # Final?
                    if "</answer>" in new_text:
                        del gen_ids
                        # log_gpu("FINAL") 
                        break

                    # Parse tool calls
                    calls = _extract_tool_calls(new_text)
                    
                    if not calls:
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed = (base.model.embed_tokens)(gen_ids)
                        cur_emb = torch.cat([cur_emb, embed], dim=1)
                        cur_attn = torch.cat([cur_attn, torch.ones_like(gen_ids, dtype=cur_attn.dtype)], dim=1)
                        del embed, gen_ids
                        continue

                    valid_zoom_calls, valid_other_calls = self._split_and_validate_calls(calls, ts_tensor.shape[1])

                    if valid_zoom_calls or valid_other_calls:
                        # run non-zoom tools and fold results into one assistant message prefix
                        # tools_blob = self._format_tool_results(valid_other_calls, dispatcher) if valid_other_calls else ""
                        zoom_block = self._format_zoom_block(valid_zoom_calls) if valid_zoom_calls else ""

                        after_tool_text = (
                        # ((tools_blob + "\n") if tools_blob else "") +
                        ((zoom_block + "\n") if zoom_block else "") +
                        "Think in the mind first, and then decide whether to call tools one or more times OR provide final answer. ")
                        combined_msgs = [{"role": "user", "content": after_tool_text}]

                        # tokenize that single assistant message
                        combined_text = self.tokenizer.apply_chat_template(
                            combined_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
                        )
                        combined_tok = self.tokenizer(combined_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

                        if valid_zoom_calls:
                            segs = [c["ts_seg"] for c in valid_zoom_calls]
                            block_emb, block_attn = self._inject_tool_segments_og_test(
                                combined_tok.input_ids, combined_tok.attention_mask,
                                ts_tensor, q_emb, segs
                            )
                        else:
                            base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                            embed_fn = (base.model.embed_tokens)
                            block_emb  = embed_fn(combined_tok.input_ids)
                            block_attn = combined_tok.attention_mask

                        # append the raw generated ids as tokens first, then the combined assistant block
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed_fn = (base.model.embed_tokens)
                        delta_emb  = embed_fn(gen_ids)  # from the current round
                        delta_attn = torch.ones(gen_ids.size(0), gen_ids.size(1), device=self.model.device, dtype=cur_attn.dtype)

                        projected_length = cur_emb.shape[1] + delta_emb.shape[1] + block_emb.shape[1]
                        if projected_length > self.MAX_SEQ_LEN:
                            print(f"Skipping concatenation: projected length {projected_length} exceeds limit")
                            final_text += "\n<answer>Analysis completed due to length constraints</answer>"
                            del combined_tok, block_emb, block_attn, delta_emb, delta_attn, gen_ids
                            break

                        cur_emb  = torch.cat([cur_emb,  delta_emb,  block_emb],  dim=1)
                        cur_attn = torch.cat([cur_attn, delta_attn, block_attn], dim=1)

                        del combined_tok, block_emb, block_attn, delta_emb, delta_attn, gen_ids
                        # log_gpu(f"TOOLS_DONE_{round_num}")
                    else:
                        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
                        embed_fn = (base.model.embed_tokens)
                        delta_emb  = embed_fn(gen_ids)  # from the current round
                        delta_attn = torch.ones(gen_ids.size(0), gen_ids.size(1), device=self.model.device, dtype=cur_attn.dtype)
                        cur_emb  = torch.cat([cur_emb,  delta_emb],  dim=1)
                        cur_attn = torch.cat([cur_attn, delta_attn], dim=1)

                        del delta_emb, delta_attn, gen_ids
                        # log_gpu(f"NO_TOOLS_{round_num}")
                        
                except Exception as round_e:
                    print(f"ERROR in generation round {round_num}: {round_e}")
                    print(f"cur_emb.shape: {cur_emb.shape if 'cur_emb' in locals() else 'Not available'}")
                    print(f"cur_attn.shape: {cur_attn.shape if 'cur_attn' in locals() else 'Not available'}")
                    if torch.cuda.is_available():
                        torch_mem = torch.cuda.memory_allocated() / (1024**3)
                        torch_cached = torch.cuda.memory_reserved() / (1024**3)
                        print(f"GPU memory - Allocated: {torch_mem:.2f}GB, Cached: {torch_cached:.2f}GB")
                    raise round_e
            
            
            # print("finishing")
            
            # log_gpu("END")
            try:
                print("current_emb.shape: ", cur_emb.shape)
                print("round_num: ", round_num)
            except Exception as e:
                print(f"Error printing debug info: {e}")
            
            torch.cuda.empty_cache()
            gc.collect()
            return final_text, entropies
            
        except Exception as e:
            print(f"ERROR in generate_with_ts_injection: {e}")
            print(f"cur_emb.shape: {cur_emb.shape if 'cur_emb' in locals() else 'Not available'}")
            print(f"cur_attn.shape: {cur_attn.shape if 'cur_attn' in locals() else 'Not available'}")
            print(f"ts_tensor.shape: {ts_tensor.shape if 'ts_tensor' in locals() else 'Not available'}")
            if torch.cuda.is_available():
                torch_mem = torch.cuda.memory_allocated() / (1024**3)
                torch_cached = torch.cuda.memory_reserved() / (1024**3)
                print(f"GPU memory - Allocated: {torch_mem:.2f}GB, Cached: {torch_cached:.2f}GB")
            raise e

    def _print_timing_summary(self):
        """Print timing statistics summary"""
        # print(f"\n=== QwenTS Generation Timing Summary (N={len(self.timing_stats['ts_preparation'])}) ===")
        for operation, times in self.timing_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                print(f"{operation:25s}: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")
        print("=" * 60)


    def _split_and_validate_calls(self, calls, ts_len: int):
        """Return (zoom_calls, other_calls) where each call has a valid clamped ts_seg."""
        zoom_names = {"zoom", "zoom_in", "zoom_in_tokens", "timeseries_zoom_in_tool"}
        zoom_calls, other_calls = [], []

        for c in calls:
            name, args = c.get("name"), (c.get("arguments") or {})
            if not (isinstance(args, dict) and "ts_seg" in args):
                continue  # skip malformed

            try:
                s, e = int(args["ts_seg"][0]), int(args["ts_seg"][1])
                s, e = in_bounds([s, e], ts_len)
                if e <= s:
                    continue  # invalid empty seg
            except Exception:
                continue

            call = {"name": name, "arguments": args, "ts_seg": [s, e]}
            if name in zoom_names:
                zoom_calls.append(call)
            else:
                other_calls.append(call)

        return zoom_calls, other_calls

    def _format_zoom_block(self, zoom_calls) -> str:
        """
        Build a textual block that lists each zoom segment with indices AND timestamps,
        and places one <TS_PLACEHOLDER> for each entry, in order.
        """
        lines = []
        for c in zoom_calls:
            s, e = c.get("ts_seg", [])
            lines.append(f"Segment ({s},{e}) of time series: {TS_PLACEHOLDER}\n")
            # lines.append(f"tool name: timeseries_zoom_in_tool, segment ({s},{e}), result (TS embeddings): {TS_PLACEHOLDER}")
        return "\n".join(lines)

    def _format_tool_results(self, other_calls, dispatcher):
        """Run non-zoom tools and serialize results into a compact blob."""
        lines = []
        for c in other_calls:
            name = c.get("name", "unknown_tool")
            args = c.get("arguments", {})
            segs = c.get("ts_seg", [])
            try:
                res = dispatcher.dispatch(name, args)
            except Exception as ex:
                res = {"error": f"{type(ex).__name__}", "msg": str(ex)}
            # wrap each result in a lightweight tag so the model can parse easily
            lines.append(f'Segment ({segs[0]},{segs[1]}) of time series: {TS_PLACEHOLDER}\n')
        return "".join(lines)

    # def _prepare_ts_tensor(self, ts_data):
    #     """Convert TS data to tensor format"""
    #     if isinstance(ts_data, np.ndarray):
    #         ts_tensor = torch.from_numpy(ts_data).float()
    #     else:
    #         ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
        
    #     if ts_tensor.dim() == 1:
    #         ts_tensor = ts_tensor.unsqueeze(-1)
    #     elif ts_tensor.dim() == 2 and self.task == "2TS":
    #         ts_tensor = ts_tensor.unsqueeze(-1)
        
    #     return ts_tensor.unsqueeze(0).to(self.model.device)  # Add batch dimension
    def _prepare_ts_tensor(self, ts_data):
        """Convert TS data to tensor format with correct dtype"""
        if isinstance(ts_data, torch.Tensor):
            ts_tensor = ts_data
            # Ensure floating type (keep fp16/bf16 if already set; otherwise promote)
            if not ts_tensor.is_floating_point():
                ts_tensor = ts_tensor.float()
        elif isinstance(ts_data, np.ndarray):
            ts_tensor = torch.from_numpy(ts_data).float()
        else:
            ts_tensor = torch.tensor(ts_data, dtype=torch.float32)
        
        if ts_tensor.dim() == 1:
            ts_tensor = ts_tensor.unsqueeze(-1)
        elif ts_tensor.dim() == 2 and self.task == "2TS":
            ts_tensor = ts_tensor.unsqueeze(-1)
        
        # FIXED: Ensure dtype matches the TOTEM VQ embedding weights
        device = self.model.device
        ts_tensor = ts_tensor.to(device)
        
        # Check for TOTEM VQ embedding weights specifically
        try:
            if (hasattr(self.model, 'ts_encoder') and 
                hasattr(self.model.ts_encoder, 'encoder') and 
                hasattr(self.model.ts_encoder.encoder, 'vq') and
                hasattr(self.model.ts_encoder.encoder.vq, '_embedding')):
                
                vq_dtype = self.model.ts_encoder.encoder.vq._embedding.weight.dtype
                ts_tensor = ts_tensor.to(dtype=vq_dtype)
                # print(f"Converting TS data to TOTEM VQ dtype: {vq_dtype}")
            else:
                # Fallback: try to get any encoder parameter dtype
                if hasattr(self.model, 'ts_encoder'):
                    encoder_dtype = next(self.model.ts_encoder.parameters()).dtype
                    ts_tensor = ts_tensor.to(dtype=encoder_dtype)
                    # print(f"Converting TS data to encoder dtype: {encoder_dtype}")
        except Exception as e:
            print(f"Warning: Could not determine TOTEM dtype, keeping float32: {e}")
            # Keep as float32 if we can't determine the right dtype
        
        return ts_tensor.unsqueeze(0)  # Add batch dimension
    
    def _get_question_embedding(self, question_ids):
        """Get question embedding for TS encoding"""
        q_tensor = torch.tensor(question_ids, device=self.model.device).unsqueeze(0)
        return self.model.model.embed_tokens(q_tensor)
    
    def _get_question_embedding_test(self, question_ids):
        """Get question embedding for TS encoding"""
        q_tensor = torch.tensor(question_ids, device=self.model.device).unsqueeze(0)
        return self.model.model.embed_tokens(q_tensor)

    def _pad_batch_emb(self, emb_list, attn_list):
        # emb_list: list of [1, T_i, D]
        # attn_list: list of [1, T_i]
        D = emb_list[0].size(-1)
        Tmax = max(x.size(1) for x in emb_list)
        B = len(emb_list)
        dtype = emb_list[0].dtype
        device = emb_list[0].device

        emb = torch.zeros(B, Tmax, D, device=device, dtype=dtype)
        attn = torch.zeros(B, Tmax, device=device, dtype=attn_list[0].dtype)
        for i, (e, a) in enumerate(zip(emb_list, attn_list)):
            Ti = e.size(1)
            emb[i, :Ti] = e[0]
            attn[i, :Ti] = a[0]
        return emb, attn

    def _inject_initial_ts(self, input_ids, attention_mask, ts_tensor, question_emb):
        """Inject initial TS (full series) into prompt"""
        # Get text embeddings
        text_emb = self.model.model.embed_tokens(input_ids)
        
        # Find TS placeholder
        ph_mask = (input_ids == self.ts_placeholder_id)
        if not ph_mask.any():
            return text_emb, attention_mask
        
        # Encode full TS
        ts_length = ts_tensor.shape[1]
        full_ts_emb = self._encode_ts_segment(ts_tensor, question_emb, [0, ts_length])
        
        # Inject TS at placeholder location
        combined_emb, combined_attn, _ = self.model.inject_ts_segments_and_labels(
            text_embeddings=text_emb,
            placeholder_mask=ph_mask,
            ts_embeddings=[[full_ts_emb]],  # Batch format
            labels=None
        )
        
        return combined_emb, combined_attn
    
    def _inject_initial_ts_test(self, input_ids, attention_mask, ts_tensor, question_emb):
        """Inject initial TS (full series) into prompt"""
        # Get text embeddings
        text_emb = self.model.model.embed_tokens(input_ids)
        
        # Find TS placeholder
        ph_mask = (input_ids == self.ts_placeholder_id)
        if not ph_mask.any():
            return text_emb, attention_mask
        
        # Encode full TS
        ts_length = ts_tensor.shape[1]
        full_ts_emb = self._encode_ts_segment(ts_tensor, question_emb, [0, ts_length])
        
        # Inject TS at placeholder location
        combined_emb, combined_attn, _ = self.model.inject_ts_segments_and_labels(
            text_embeddings=text_emb,
            placeholder_mask=ph_mask,
            ts_embeddings=[[full_ts_emb]],  # Batch format
            labels=None
        )
        
        return combined_emb, combined_attn

    def _encode_ts_segment(self, ts_tensor, question_emb, segment):
        """Encode a specific TS segment"""
        start, end = segment
        start, end = max(0, start), min(end, ts_tensor.shape[1])

        ts_length = ts_tensor.shape[1]
                
        # Enforce minimum segment length for PatchTST
        if end - start < self.min_segment_length:
            if start + self.min_segment_length <= ts_length:
                end = start + self.min_segment_length
            elif end - self.min_segment_length >= 0:
                start = end - self.min_segment_length
            else:
                start, end = 0, ts_length
        
        if self.task == "2TS":
            seg_data = ts_tensor[:, :, start:end, :]  # [B, 2, L_seg, 1]
        else:
            seg_data = ts_tensor[:, start:end, :]  # [B, L_seg, 1]
        
        # FIXED: Use the TOTEM encoder's dtype instead of general model dtype
        if hasattr(self.model, 'ts_encoder') and hasattr(self.model.ts_encoder, 'encoder'):
            # Get the dtype from the TOTEM encoder's first parameter
            totem_dtype = next(self.model.ts_encoder.encoder.parameters()).dtype
            seg_data = seg_data.to(dtype=totem_dtype)
        else:
            # Fallback to model dtype
            model_dtype = next(self.model.parameters()).dtype
            seg_data = seg_data.to(dtype=model_dtype)
        
        # Also ensure question_emb matches
        question_emb = question_emb.to(dtype=seg_data.dtype)
        
        # Use model's TS encoding
        if seg_data.shape[1] < self.min_segment_length:
            pad_length = self.min_segment_length - seg_data.shape[1]
            # Pad along the sequence dimension (dim=1)
            seg_data = torch.nn.functional.pad(seg_data, (0, 0, 0, pad_length), value=0)
        return self.model.encode_ts(seg_data, question_emb)
    
    def _handle_tool_calls(self, generated_ids, current_embeddings, current_attention, 
                          tool_segments, ts_tensor, question_emb):
        """Process tool calls and inject new TS segments"""
        # Add generated tokens to current embeddings
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            gen_emb = self.model.model.embed_tokens(generated_ids.unsqueeze(0))
        else:
            base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
            gen_emb = base_model.model.embed_tokens(generated_ids.unsqueeze(0))
        
        gen_attn = torch.ones(1, generated_ids.size(0), device=self.model.device, dtype=current_attention.dtype)
        current_embeddings = torch.cat([current_embeddings, gen_emb], dim=1)
        current_attention = torch.cat([current_attention, gen_attn], dim=1)
        
        # Prepare after-tool message
        after_tool_message = f"{TS_PLACEHOLDER}\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. " \
                   "Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
        
        after_formatted = self.tokenizer.apply_chat_template(after_tool_message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        after_tokens = self.tokenizer(after_formatted, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Process valid tool segments
        valid_segments = []
        ts_length = ts_tensor.shape[1]
        for start, end in tool_segments:
            start, end = in_bounds([start, end], ts_length)
            if end > start:
                valid_segments.append([start, end])
        
        # Inject TS segments into after-tool message if any valid segments
        if valid_segments:
            after_embeddings, after_attention = self._inject_tool_segments_og(
                after_tokens.input_ids, after_tokens.attention_mask,
                ts_tensor, question_emb, valid_segments
            )
        else:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                after_embeddings = self.model.model.embed_tokens(after_tokens.input_ids)
            else:
                base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
                after_embeddings = base_model.model.embed_tokens(after_tokens.input_ids)
            after_attention = after_tokens.attention_mask
        
        # Concatenate everything
        current_embeddings = torch.cat([current_embeddings, after_embeddings], dim=1)
        current_attention = torch.cat([current_attention, after_attention], dim=1)
        
        return current_embeddings, current_attention
    
    def _inject_tool_segments_og(self, input_ids, attention_mask, ts_tensor, question_emb, segments):
        """Inject multiple TS segments at placeholder"""
        base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        text_emb = base_model.model.embed_tokens(input_ids)
        
        ph_mask = (input_ids == self.ts_placeholder_id)
        if not ph_mask.any():
            return text_emb, attention_mask
        
        # Encode all segments
        seg_embs = []
        for start, end in segments:
            seg_emb = self._encode_ts_segment(ts_tensor, question_emb, [start, end])
            seg_embs.append(seg_emb)
        
        # Inject segments
        combined_emb, combined_attn, _ = self.model.inject_ts_segments_and_labels(
            text_embeddings=text_emb,
            placeholder_mask=ph_mask,
            ts_embeddings=[seg_embs],  # Batch format
            labels=None
        )
        
        return combined_emb, combined_attn
    
    def _inject_tool_segments_og_test(self, input_ids, attention_mask, ts_tensor, question_emb, segments):
        """Inject multiple TS segments at placeholder"""
        base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        text_emb = base_model.model.embed_tokens(input_ids)
        
        ph_mask = (input_ids == self.ts_placeholder_id)
        if not ph_mask.any():
            return text_emb, attention_mask
        
        # Encode all segments
        seg_embs = []
        for start, end in segments:
            seg_emb = self._encode_ts_segment(ts_tensor, question_emb, [start, end])
            seg_embs.append(seg_emb)
        
        # Inject segments
        combined_emb, combined_attn, _ = self.model.inject_ts_segments_and_labels(
            text_embeddings=text_emb,
            placeholder_mask=ph_mask,
            ts_embeddings=[seg_embs],  # Batch format
            labels=None
        )
        
        return combined_emb, combined_attn
    
    def _inject_tool_segments(self, ts_tensor, q_emb, segs):
        # Small “after tool” message containing the placeholder; inject multiple segments there
        after = self.tokenizer.apply_chat_template([{"role":"user","content":f"{TS_PLACEHOLDER}\nContinue reasoning."},
                                                    {"role":"assistant","content":""}],
                                                   tokenize=False, add_generation_prompt=False)
        toks = self.tokenizer(after, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        base = self.model.get_base_model() if hasattr(self.model,'get_base_model') else self.model
        txt = (base.model.embed_tokens if hasattr(base,'model') else self.model.model.embed_tokens)(toks.input_ids)

        ph = (toks.input_ids == self.ts_placeholder_id)
        if not ph.any(): return txt, toks.attention_mask

        encs = []
        Ltot = ts_tensor.shape[1]
        for s, e in segs:
            s, e = in_bounds([s, e], Ltot)
            if e > s: encs.append(self._encode_ts_segment(ts_tensor, q_emb, [s, e]))

        combined_emb, combined_attn, _ = self.model.inject_ts_segments_and_labels(
            text_embeddings=txt, placeholder_mask=ph, ts_embeddings=[encs], labels=None
        )
        return combined_emb, combined_attn
    
    def _is_complete_response(self, text):
        """Check if response is complete"""
        return "</answer>" in text #or text.endswith("<|im_end|>")

    def _calculate_entropy_from_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> float:
        """Core entropy calculation - mean entropy across all positions"""
        with torch.no_grad():
            # Forward pass to get logits
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)
            
            batch_size, seq_len, vocab_size = outputs.logits.shape
            
            # Calculate entropy at each position for each sequence in batch
            all_entropies = []
            
            for i in range(batch_size):
                if attention_mask is not None:
                    valid_positions = attention_mask[i].nonzero(as_tuple=True)[0]  # Valid positions
                else:
                    valid_positions = torch.arange(seq_len, device=embeddings.device)
                
                position_entropies = []
                for pos in valid_positions:
                    logits = outputs.logits[i, pos, :]  # [vocab_size]
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                    position_entropies.append(entropy)
            
            # Return mean entropy across batch and positions
            return float(np.mean(position_entropies)) if position_entropies else 5.0