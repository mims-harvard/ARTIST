#!/usr/bin/env python3
"""
QwenTS Utils - Complete Consolidated Functions
All the preprocessing and utility functions for QwenTS Lightning model
"""

import re
import json
import torch
import numpy as np
from typing import List, Dict, Any, Sequence, Optional, Tuple
from transformers import PreTrainedTokenizer
from RL_1tool.constants import *

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
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


def in_bounds(ts_seg: List[int], ts_length: int) -> List[int]:
    """
    Check if segment is within bounds [0, ts_length] and adjust if needed.
    Ensures minimum length of 8.
    """
    start, end = int(ts_seg[0]), int(ts_seg[1])
    if isinstance(ts_length, str):
        ts_length = int(ts_length)
    
    # Clamp to bounds
    start = max(0, min(start, ts_length))
    end = max(start, min(end, ts_length))
    
    # Ensure minimum length of 8
    min_length = 8
    if end - start < min_length:
        # Try to extend end first
        if start + min_length <= ts_length:
            end = start + min_length
        # Try to extend start if end extension doesn't fit
        elif end - min_length >= 0:
            start = end - min_length
        # If time series is shorter than min_length, use full series
        else:
            start = 0
            end = ts_length
            if end - start < min_length:
                print(f"Warning: Time series length {ts_length} is shorter than minimum required {min_length}")
    
    return [start, end]


def extract_ts_segments(messages: List[Dict[str, Any]], ts: Sequence[Any]) -> Tuple[List[List[int]], int]:
    """
    Extract time series segments from tool calls in messages.
    
    Returns:
        (segments, invalid_segments_count)
    """
    segments = [[0, len(ts)]]  # Always start with full span
    invalid_segments_count = 0
    
    for msg in messages:
        content = msg.get("content", "")
        if TOOL_CALL_TOKEN not in content:
            continue
            
        # Extract all tool calls from content
        for match in content.split(TOOL_CALL_TOKEN)[1:]:
            if TOOL_CALL_END_TOKEN not in match:
                continue
                
            json_str = match.split(TOOL_CALL_END_TOKEN)[0].strip()
            if "arguments" not in json_str:
                continue
            
            try:
                args = json.loads(json_str).get("arguments", {})
                ts_seg = args.get("ts_seg", [])
                if isinstance(ts_seg, list) and len(ts_seg) == 2:
                    # Check if both values are valid integers
                    start, end = ts_seg[0], ts_seg[1]
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        # Convert to integers and ensure they're within bounds
                        start_int = int(start)
                        end_int = int(end)
                        if 0 <= start_int < len(ts) and start_int < end_int <= len(ts):
                            segments.append([start_int, end_int])
                        else:
                            invalid_segments_count += 1
                    else:
                        invalid_segments_count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
                
    return segments, invalid_segments_count


def mask_non_assistant(conversations: List[str], tokenizer: PreTrainedTokenizer, targets: torch.Tensor) -> torch.Tensor:
    """
    Mask non-assistant tokens in targets for training.
    Only assistant responses should contribute to the loss.
    """
    role_start = "<|im_start|>assistant\n"
    role_end = "<|im_end|>"
    
    masked = targets.clone()
    
    for i, conv in enumerate(conversations):
        # Find all assistant sections
        idx = 0
        assistant_ranges = []
        
        while True:
            start_idx = conv.find(role_start, idx)
            if start_idx == -1:
                break
            
            content_start = start_idx + len(role_start)
            end_idx = conv.find(role_end, content_start)
            if end_idx == -1:
                break
            
            # Tokenize the prefix and content separately to get exact token positions
            prefix = conv[:content_start]
            content_end = end_idx
            content = conv[content_start:content_end]
            
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            
            # Calculate token positions
            start_token = len(prefix_tokens)
            end_token = start_token + len(content_tokens)
            
            if start_token < targets.size(1):
                end_token = min(end_token, targets.size(1))
                assistant_ranges.append((start_token, end_token))
            
            idx = end_idx + len(role_end)
        
        # Mask everything except assistant ranges
        masked[i, :] = IGNORE_INDEX  # Mask everything first
        for start, end in assistant_ranges:
            masked[i, start:end] = targets[i, start:end]  # Unmask assistant content
    
    return masked


def merge_consecutive_thinks(messages: List[Dict]) -> List[Dict]:
    """
    Merge consecutive assistant messages, handling thinks properly.
    """
    merged = []
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        if msg.get("role") == "assistant":
            # Collect consecutive assistant messages
            assistant_msgs = []
            j = i
            while j < len(messages) and messages[j].get("role") == "assistant":
                assistant_msgs.append(messages[j])
                j += 1
            
            # Check if there's a tool_call in any of them
            has_tool_call = any(TOOL_CALL_TOKEN in am.get("content", "") for am in assistant_msgs)
            
            # Extract all think contents and other content
            all_thinks = []
            other_content = []
            
            for am in assistant_msgs:
                content = am.get("content", "").strip()
                
                # Extract think content
                while THINK_OPEN in content and THINK_CLOSE in content:
                    start = content.find(THINK_OPEN)
                    end = content.find(THINK_CLOSE) + len(THINK_CLOSE)
                    think_text = content[start+len(THINK_OPEN):end-len(THINK_CLOSE)].strip()
                    if think_text:
                        all_thinks.append(think_text)
                    content = content[:start] + content[end:]
                
                # Keep remaining content
                remaining = content.strip()
                if remaining:
                    other_content.append(remaining)
            
            # Build merged message
            final_parts = []
            if all_thinks:
                if has_tool_call:
                    # For tool_call cases, use ONE Reasoning: with all thinks joined
                    final_parts.append(f"Reasoning: {' '.join(all_thinks)}")
                else:
                    # For non-tool_call cases, keep <think> tags
                    final_parts.append(f"{THINK_OPEN}\n{chr(10).join(all_thinks)}\n{THINK_CLOSE}")
            final_parts.extend(other_content)
            
            if final_parts:
                merged.append({
                    "role": "assistant",
                    "content": "\n\n".join(final_parts)
                })
            i = j
        else:
            merged.append(msg)
            i += 1
    
    return merged


def post_process_conversation(conversation: str) -> str:
    """
    Post-process conversation to convert 'Reasoning:' lines back to <think> tags.
    """
    lines = conversation.split('\n')
    processed_lines = []
    i = 0
    
    while i < len(lines):
        if lines[i].startswith('Reasoning: '):
            # Collect all consecutive Reasoning lines
            reasoning_parts = []
            j = i
            while j < len(lines) and lines[j].startswith('Reasoning: '):
                reasoning_parts.append(lines[j][11:])  # Remove "Reasoning: " prefix
                j += 1
            
            # Add as proper think block
            processed_lines.append(THINK_OPEN)
            processed_lines.append('\n'.join(reasoning_parts))
            processed_lines.append(THINK_CLOSE)
            i = j
        else:
            processed_lines.append(lines[i])
            i += 1
    
    return '\n'.join(processed_lines)


def normalize_for_qwen(messages: List[Dict]) -> List[Dict]:
    """
    Normalize messages for Qwen, properly handling thinks with tool_calls.
    """
    normalized = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role != "assistant":
            normalized.append(msg)
            continue
        
        # Extract think content first
        reasoning_content = ""
        remaining_content = content
        if THINK_OPEN in content and THINK_CLOSE in content:
            start = content.find(THINK_OPEN)
            end = content.find(THINK_CLOSE) + len(THINK_CLOSE)
            reasoning_content = content[start+len(THINK_OPEN):end-len(THINK_CLOSE)].strip()
            remaining_content = content[:start] + content[end:]
        
        # Check for inline tool calls in remaining content
        if TOOL_CALL_TOKEN in remaining_content:
            tool_calls = []
            parts = remaining_content.split(TOOL_CALL_TOKEN)
            final_content = parts[0].strip()  # Content before tool_call
            
            for i, part in enumerate(parts[1:], 1):
                if TOOL_CALL_END_TOKEN in part:
                    tool_json, after = part.split(TOOL_CALL_END_TOKEN, 1)
                    try:
                        tool_data = json.loads(tool_json.strip())
                        tool_calls.append({
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_data.get("name"),
                                "arguments": json.dumps(tool_data.get("arguments", {}))
                            }
                        })
                        final_content += after
                    except json.JSONDecodeError:
                        final_content += f"{TOOL_CALL_TOKEN}{part}"
            
            # Create message with reasoning_content for thinks
            new_msg = {"role": "assistant", "content": final_content.strip()}
            if reasoning_content:
                new_msg["reasoning_content"] = reasoning_content
            if tool_calls:
                new_msg["tool_calls"] = tool_calls
            normalized.append(new_msg)
        else:
            # No tool calls, keep content as is
            normalized.append({"role": "assistant", "content": content})
    
    return normalized


def qwen_preprocess(
    sources: Sequence[List[Dict]],
    ts_tensor: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    has_ts: bool = False,
    partition: str = 'train',
    enable_thinking: bool = False,
) -> Dict:
    """
    Cleaned and fixed preprocessing function for QwenTS.
    """
    # Define tools
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
    
    # System message
    system_message = {
        'role': 'system',
        'content': 'you are a helpful assistant that can answer questions about time series data.'
    }
    
    after_tool_message = {
        "role": "user",
        "content": f"{TS_PLACEHOLDER}\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. "
                   "Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    }
    
    conversations = []
    questions = []
    ts_segments_list = []
    total_invalid_segments = 0
    
    for i, source in enumerate(sources):
        # Build message list
        messages = [system_message]
        
        # Add all messages first
        for msg in source:
            messages.append(msg)
        
        # Extract segments to get accurate count of valid tool calls
        ts_segs, invalid_segments_count = extract_ts_segments(source, ts_tensor[i])
        total_invalid_segments += invalid_segments_count
        num_tool_calls = len(ts_segs) - 1  # Subtract 1 for the default [0, len(ts)] segment
        
        # Add after-tool messages based on actual valid segments
        for _ in range(num_tool_calls):
            messages.append(after_tool_message)
        
        # Process messages
        # Step 1: Merge consecutive think messages
        messages = merge_consecutive_thinks(messages)
        
        # Step 2: Store TS segments if training/validation
        if partition != 'test':
            ts_segments_list.append(ts_segs)
        
        # Step 3: Extract question for encoding
        question = next((m for m in messages if m.get("role") == "user" and TS_PLACEHOLDER in m.get("content", "")), None)
        if question:
            question_raw = question['content'].replace(TS_PLACEHOLDER, "").strip()
            question_ids = tokenizer.encode(question_raw, add_special_tokens=False)
        else:
            question_ids = []
        questions.append(question_ids)
        
        # Step 4: Apply chat template
        add_generation_prompt = (partition == 'test')
        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=TOOLS,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )
        conversation = post_process_conversation(conversation)
        conversations.append(conversation)
    
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
    targets[attention_mask == 0] = IGNORE_INDEX  # Mask padding
    
    # Mask non-assistant tokens for training
    if partition != 'test':
        targets = mask_non_assistant(conversations, tokenizer, targets)
    
    return {
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "ts_segs": ts_segments_list if partition != 'test' else None,
        "question_ids": questions,
        "invalid_segments_count": total_invalid_segments
    }


def extract_answer_letter(completion: str) -> Optional[str]:
    """
    Extract answer letter (A, B, C, D) from completion using multiple strategies.
    """
    if not completion:
        return None
    
    # Strategy 1: Look for <answer> tags first
    answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Look for letter in answer content
        for letter in ['A', 'B', 'C', 'D']:
            if letter in answer_content.upper():
                return letter
    
    # Strategy 2: Look for "Answer: X" or "The answer is X" patterns
    answer_patterns = [
        r'answer\s*:?\s*([ABCD])',
        r'the\s+answer\s+is\s+([ABCD])',
        r'correct\s+answer\s*:?\s*([ABCD])',
        r'option\s+([ABCD])',
        r'choice\s+([ABCD])'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Strategy 3: Look for standalone letters (most permissive)
    for letter in ['A', 'B', 'C', 'D']:
        if re.search(rf'\b{letter}\b', completion):
            return letter
    
    # Strategy 4: Look for letters with parentheses or periods
    for letter in ['A', 'B', 'C', 'D']:
        if re.search(rf'{letter}[\)\.]', completion):
            return letter
    
    return None


def extract_answer(completion: str) -> Optional[str]:
    """
    Extract any answer from completion (not just letters).
    """
    # Try <answer> tags first
    answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try "Answer:" patterns
    answer_patterns = [
        r'answer\s*:?\s*(.+?)(?:\n|$)',
        r'the\s+answer\s+is\s+(.+?)(?:\n|$)',
        r'final\s+answer\s*:?\s*(.+?)(?:\n|$)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def parse_tool_segs(content: str) -> List[List[int]]:
    """
    Parse tool call segments from content.
    Returns list of [start, end] segments.
    """
    segments = []
    
    # Find all tool calls
    for match in content.split(TOOL_CALL_TOKEN)[1:]:
        if TOOL_CALL_END_TOKEN not in match:
            continue
            
        json_str = match.split(TOOL_CALL_END_TOKEN)[0].strip()
        try:
            tool_data = json.loads(json_str)
            args = tool_data.get("arguments", {})
            ts_seg = args.get("ts_seg", [])
            
            if isinstance(ts_seg, list) and len(ts_seg) == 2:
                start, end = int(ts_seg[0]), int(ts_seg[1])
                segments.append([start, end])
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    
    return segments


def replace_answer_in_content(content: str, new_answer: str) -> str:
    """
    Replace the answer in <answer> tags with new_answer.
    """
    return re.sub(r'<answer>.*?</answer>',
                  f'<answer>{new_answer}</answer>',
                  content,
                  count=1)


def calculate_comprehensive_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate comprehensive metrics from predictions.
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    except ImportError:
        print("Warning: sklearn not available, returning basic metrics")
        if not predictions:
            return {}
        
        # Basic accuracy calculation
        correct = sum(1 for pred in predictions if pred.get('result') == pred.get('label'))
        accuracy = correct / len(predictions) if predictions else 0.0
        return {'accuracy': accuracy}
    
    if not predictions:
        return {}
    
    # Extract true and predicted labels
    y_true = [pred['label'] for pred in predictions]
    y_pred = [pred['result'] for pred in predictions if pred['result'] is not None]
    
    # Handle cases where some predictions failed
    if len(y_pred) != len(y_true):
        print(f"Warning: {len(y_true) - len(y_pred)} predictions failed")
        # Pad with empty predictions
        y_pred.extend([''] * (len(y_true) - len(y_pred)))
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        unique_labels = sorted(set(y_true + y_pred))
        per_class = {}
        for i, label in enumerate(unique_labels):
            if i < len(precision):
                per_class[label] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': support[i] if i < len(support) else 0
                }
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class': per_class
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'accuracy': 0.0}


def save_test_results(output_dir: str, predictions: List[Dict], accuracy: float, 
                     errors: List[str], timestamp: str, metrics: Dict = None):
    """
    Save test results to files.
    """
    import os
    import json
    
    # Save detailed predictions
    predictions_file = os.path.join(output_dir, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_samples': len(predictions),
        'accuracy': accuracy,
        'errors': errors,
        'metrics': metrics or {}
    }
    
    summary_file = os.path.join(output_dir, 'test_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save human-readable report
    report_file = os.path.join(output_dir, 'test_report.txt')
    with open(report_file, 'w') as f:
        f.write(f"QwenTS Test Results - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Samples: {len(predictions)}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"Errors: {len(errors)}\n\n")
        
        if metrics:
            f.write("Detailed Metrics:\n")
            f.write(f"  Macro F1: {metrics.get('macro_f1', 0):.3f}\n")
            f.write(f"  Weighted F1: {metrics.get('weighted_f1', 0):.3f}\n")
            f.write(f"  Macro Precision: {metrics.get('macro_precision', 0):.3f}\n")
            f.write(f"  Macro Recall: {metrics.get('macro_recall', 0):.3f}\n\n")
            
            if 'per_class' in metrics:
                f.write("Per-Class Metrics:\n")
                for label, class_metrics in metrics['per_class'].items():
                    f.write(f"  {label}: P={class_metrics['precision']:.3f}, "
                           f"R={class_metrics['recall']:.3f}, "
                           f"F1={class_metrics['f1']:.3f}, "
                           f"Support={class_metrics['support']}\n")
        
        if errors:
            f.write("\nErrors:\n")
            for error in errors:
                f.write(f"  {error}\n")
    
    print(f"📊 Test results saved to {output_dir}")
    print(f"   - Predictions: {predictions_file}")
    print(f"   - Summary: {summary_file}")
    print(f"   - Report: {report_file}")


# Additional utility functions
def format_conversation_for_display(conversation: str) -> str:
    """Format conversation for human-readable display."""
    # Replace special tokens with readable versions
    conversation = conversation.replace("<|im_start|>", "\n[START ")
    conversation = conversation.replace("<|im_end|>", " END]\n")
    conversation = conversation.replace(TS_PLACEHOLDER, "[TIME_SERIES_DATA]")
    return conversation.strip()


def count_tokens_in_conversation(conversation: str, tokenizer: PreTrainedTokenizer) -> int:
    """Count tokens in a conversation."""
    tokens = tokenizer.encode(conversation, add_special_tokens=False)
    return len(tokens)


def validate_conversation_format(conversation: str) -> List[str]:
    """Validate conversation format and return list of issues."""
    issues = []
    
    # Check for balanced tags
    start_count = conversation.count("<|im_start|>")
    end_count = conversation.count("<|im_end|>")
    if start_count != end_count:
        issues.append(f"Unbalanced im tags: {start_count} starts, {end_count} ends")
    
    # Check for proper role structure
    roles = ["system", "user", "assistant"]
    for role in roles:
        if f"<|im_start|>{role}" in conversation:
            if not conversation.count(f"<|im_start|>{role}") == conversation.count(f"<|im_start|>{role}\n"):
                issues.append(f"Missing newline after {role} role")
    
    # Check for answer tags if expected
    if ANSWER_OPEN in conversation and ANSWER_CLOSE not in conversation:
        issues.append("Unclosed answer tag")
    
    return issues


# def get_instruction_for_stage(training_stage: str, task: str, is_tf=False) -> str:
#     """Get instruction text based on training stage and task type"""
#     if training_stage == "mcq":
#         if task == "ETI":
#             return "The following is a multiple-choice question. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
#         elif task == "1TS":
#             return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
#         elif task in ["ECG_QA_S_VERIFY", "ECG_QA_S_QUERY", "ECG_QA_MIXED"]:
#             return "The following is a question on 1 Lead ECG signal (out of 12 leads). Given the specific Lead, your task is to examine the ECG signal and answer a specific medical question about it."
#         elif is_tf:
#             return "The following is a True/False question. Please select and provide the correct answer from options A or B. Only return the correct answer letter."
#         else:
#             return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
            
#     elif training_stage == "alignment":
#         return "You are provided with a time series data and a question about it. Please analyze the time series data and provide a detailed answer to the following question."

#     elif training_stage == "reasoning":
#         if task == "ETI":
#             return "Please analyze the data step by step, explaining your reasoning, then provide your final answer to the following question."
#         elif task == "1TS":
#             return "Please analyze the time series data step by step, explaining your reasoning, then provide your final answer to the following question."
#         else:
#             return "Please analyze the two time series step by step, explaining your reasoning, then provide your final answer to the following question."
    
#     else:
#         raise ValueError(f"Unknown training_stage: {training_stage}")
# def get_instruction_for_stage(training_stage, task, is_tf=False) -> str:
#     """Get instruction text based on training stage and task type"""
#     if training_stage == "mcq":
#         # Multiple choice instructions
#         if task == "ETI" or task == 'ETI_w_tools':
#             return "The following is a multiple-choice question. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
#         elif task == "1TS" or task == "TIMERBED_ECG":
#             return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
#         elif task == 'TSQA' or task == 'TSQA_w_tools' or task == 'TSQA_TF':
#             return "The following is a question about time series data (could be either multiple choice or open-ended). Please answer the question based on the time series data."
#         elif task == "TIMERBED_RCW":
#             return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A' or 'B'. Only return the correct answer letter."
#         elif task in ["ECG_QA_S_VERIFY", "ECG_QA_S_QUERY", "ECG_QA_MIXED"]:
#             return "The following is a question on 1 Lead ECG signal (out of 12 leads). Given the specific Lead, your task is to examine the ECG signal and answer a specific medical question about it."
#         else:
#             return "The following is a multiple-choice question about two time series. Please select and provide the correct answer from options available. Only return the correct answer letter."
            
#     elif training_stage == "alignment":
#             return "You are provided with a time series data and a question about it that may include information about the time series. Please analyze the time series data and provide a detailed answer to the following question."

#     elif training_stage == "reasoning":
#         # Reasoning instructions - step-by-step analysis
#         if task == "ETI":
#             return "Please analyze the time series data step by step, explaining your reasoning, then provide your final answer to the following question."
#         elif task == "1TS":
#             return "Please analyze the time series data step by step, explaining your reasoning, then provide your final answer to the following question."
#         else:
#             return "Please analyze the two time series step by step, explaining your reasoning, then provide your final answer to the following question."
    
#     else:
#         raise ValueError(f"Unknown training_stage: {training_stage}. Must be 'mcq', 'alignment', or 'reasoning'")
def get_instruction_for_stage(training_stage, task, is_tf=False) -> str:
    """Get instruction text based on training stage and task type"""
    if training_stage == "mcq":
        # Multiple choice instructions
        if task == "ETI" or task == 'ETI_w_tools':
            return "The following is a multiple-choice question. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
        elif task == "1TS" or task == "TIMERBED_ECG":
            return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A', 'B', 'C', 'D'. Only return the correct answer letter."
        elif task == 'TSQA' or task == 'TSQA_w_tools':
            return "The following is a question about time series data (could be either multiple choice or open-ended). Please answer the question based on the time series data."
        elif task == "TIMERBED_RCW":
            return "The following is a multiple-choice question about time series data. Please select and provide the correct answer from options 'A' or 'B'. Only return the correct answer letter."
        elif task in ["ECG_QA_S_VERIFY", "ECG_QA_S_QUERY", "ECG_QA_MIXED"]:
            return "The following is a question on 1 Lead ECG signal (out of 12 leads). Given the specific Lead, your task is to examine the ECG signal and answer a specific medical question about it."
        else:
            return "The following is a multiple-choice question about two time series. Please select and provide the correct answer from options available. Only return the correct answer letter."
            
    elif training_stage == "alignment":
            return "You are provided with a time series data and a question about it that may include information about the time series. Please analyze the time series data and provide a detailed answer to the following question."

    elif training_stage == "reasoning":
        # Reasoning instructions - step-by-step analysis
        if task == "ETI":
            return "Please analyze the time series data step by step, explaining your reasoning, then provide your final answer to the following question."
        elif task == "1TS":
            return "Please analyze the time series data step by step, explaining your reasoning, then provide your final answer to the following question."
        else:
            return "Please analyze the two time series step by step, explaining your reasoning, then provide your final answer to the following question."
    
    else:
        raise ValueError(f"Unknown training_stage: {training_stage}. Must be 'mcq', 'alignment', or 'reasoning'")

class StopOnAny:
    """Stopping criteria for generation"""
    def __init__(self, tokenizer, stops):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stops]
    
    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        for seq in self.stop_ids:
            L = len(seq)
            if L and ids[-L:] == seq:
                return True
        return False