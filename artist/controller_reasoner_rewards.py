#!/usr/bin/env python3
"""
Rewards for Controller-Reasoner Framework
"""

import torch
import numpy as np
from typing import List, Dict, Any
import re
import string
import json

class ControllerReward:
    """
    Controller reward with format validation and trajectory evaluation
    
    Reward components:
    1. Consistency score (from reasoner rollout accuracy)
    2. Efficiency penalty (fewer rounds is better)
    3. Format reward/penalty (valid decisions vs format errors)
    4. Trajectory quality (good segment selection strategy)
    """
    
    __name__ = "ControllerReward"
    
    def __init__(
        self, 
        lambda_penalty: float = 0.1,
        penalty_type: str = "linear",
        format_weight: float = 1.0,
        format_error_penalty: float = 2.0,
        trajectory_weight: float = 0.5,
        max_trials: int = 5,
        consistency_weight: float = 3.0,
        exploration_steps: int = 300,
        exploration_bonus: float = 1.0,
        segment_explore_bonus_weight: float = 0.0,
        gentle_reward: bool = False,
        use_exploration_decay: bool = False
    ):
        """
        Args:
            lambda_penalty: Weight for efficiency penalty (fewer rounds is better)
            penalty_type: "linear", "quadratic", or "exponential"
            format_weight: Weight for format compliance
            format_error_penalty: Penalty for format errors (no valid decision)
            trajectory_weight: Weight for trajectory quality evaluation
            max_trials: Maximum allowed trials
        """
        self.lambda_penalty = lambda_penalty
        self.penalty_type = penalty_type
        self.format_weight = format_weight
        self.consistency_weight = consistency_weight
        self.format_error_penalty = format_error_penalty
        self.trajectory_weight = trajectory_weight
        self.max_trials = max_trials
        self.call_count = 0
        self.current_step = 0 # for segment exploration reward
        self.exploration_steps = exploration_steps
        self.exploration_bonus = exploration_bonus
        self.gentle_reward = gentle_reward # whether to punish hard or gently reward for format errors
        self.segment_explore_bonus_weight = segment_explore_bonus_weight # whether to reward for exploring segments
        self.use_exploration_decay = use_exploration_decay # whether to decay the exploration bonus over time

    def set_step(self, step: int):
        """Update current training step"""
        self.current_step = step

    def __call__(
        self, 
        prompts: List[str], 
        completions: List[str],
        # completion_ids: List[List[int]], 
        metadata: List[Dict[str, Any]],
        **kwargs
    ):
        """Calculate controller rewards"""
        
        # rewards = []
        totals = []
        comps_list = []

        for i, (completion, meta) in enumerate(zip(completions, metadata)):
            reward, components = self._reward_one(completion, meta)
            totals.append(reward)
            comps_list.append(components)
        
        # Debug logging
        self.call_count += 1
        if self.call_count <= 3:
            print(f"\nController Reward Batch {self.call_count}:")
            print(f"  Rewards: {[f'{r:.3f}' for r in totals[:5]]}")  # First 5
            print(f"  Mean: {np.mean(totals):.3f}, Std: {np.std(totals):.4f}")
            
            # Show breakdown for first example
            if metadata:
                m = metadata[0]
                print(f"  Example breakdown:")
                print(f"    - Consistency: {m.get('consistency_score', 0.0):.3f}")
                print(f"    - Num trials: {m.get('num_trials', 0)}")
                print(f"    - Format error: {m.get('format_error', False)}")
                print(f"    - Has accept: {m.get('has_accept', False)}")
        
        return totals, comps_list
    
    def _reward_one(self, completion: str, meta: Dict[str, Any]):
        """Calculate controller reward for single instance"""
        components = {}
        # Check for format error first
        ts_len = meta.get("ts_length", 0)
        format_score = self._compute_format_list(completion, ts_len)
      
                
        # Normal reward calculation
        consistency = meta.get("consistency_score", 0.0)
        reward = consistency * self.consistency_weight  # Weight accuracy highly
        
        # Efficiency penalty (fewer rounds is better)
        num_trials = meta.get("num_trials", 1)
        efficiency_penalty = self._calculate_penalty(num_trials)
        reward -= self.lambda_penalty * efficiency_penalty
        
        # Exploration bonus
        anneal_factor = max(0.0, 1.0 - (self.current_step / self.exploration_steps))
        num_segments = self._count_unique_segments(meta.get("all_segments", []))
        if num_segments in [2,3,4]:
            reward += self.exploration_bonus * anneal_factor
        
        # segment explore bonus
        all_segments = meta.get("all_segments", [])
        segment_explore_score = self._compute_segment_explore_score(all_segments, ts_len)
        if self.use_exploration_decay:
            segment_explore_reward = self.segment_explore_bonus_weight * anneal_factor * segment_explore_score
        else:
            segment_explore_reward = self.segment_explore_bonus_weight * segment_explore_score
        
        reward += segment_explore_reward


        # Format reward (valid decision structure)
        reward += format_score * self.format_weight
        components = {
            "consistency_score_raw": consistency,
            "efficiency_penalty_raw": efficiency_penalty,
            "format_score_raw": format_score,
            "consistency_score_weighted": consistency * self.consistency_weight,
            "efficiency_penalty_weighted": efficiency_penalty * self.lambda_penalty,
            "format_score_weighted": format_score * self.format_weight,
            "segment_explore_score_raw": segment_explore_reward,
            # "total_weighted": reward,
        }

        if format_score < 0.0:
            reward = -self.format_error_penalty
            
        return float(reward), components
    
    def _compute_segment_explore_score(self, all_segments, ts_len: int) -> float:
        """
        Exploration score in [-1, 1] based on how different the selected segments
        are from the trivial full-span [0, ts_len].

        Heuristics:
        - If only full-span segments are used -> negative or 0 score.
        - If there is at least one *non-full* segment -> positive score.
        - More distinct non-full segments => slightly higher score (capped at 1).
        """
        if not all_segments or ts_len <= 0:
            return 0.0

        # normalize to unique pairs
        uniq = {(int(s), int(e)) for s, e in all_segments}

        # define "full-span" segments (allow off-by-one)
        def is_full(s, e):
            length = max(0, e - s)
            return s == 0 and length >= ts_len - 2

        has_full = any(is_full(s, e) for (s, e) in uniq)
        nonfull = [(s, e) for (s, e) in uniq if not is_full(s, e)]

        if not nonfull:
            # only full-span segments -> discourage a bit
            # you can make this -1.0 if you want a stronger push
            return 0.0 #-0.5 if has_full else 0.0

        # at least one non-full segment -> reward
        # up to 1.0 for 2+ distinct non-full segments
        num_nonfull = len(nonfull)
        score = min(1.0, num_nonfull / 2.0)
        return 1.0 #float(score)

    def _count_unique_segments(self, segs):
        """Return the number of unique [start,end] pairs."""
        if not segs or len(segs) == 0:
            return 0
        uniq = {(int(s), int(e)) for s, e in segs}
        return len(uniq)

    def _compute_format_list(self, completions, ts_len):
        """
        Compute format compliance for full conversation (all controller rounds)
        
        Args:
            completions: List of controller responses [ctrl_0, ctrl_1, ..., ctrl_n]
        
        Rules:
            - All non-final rounds: Should have tool_call (not ACCEPT)
            - Final round: Should have ACCEPT (not tool_call)
        
        Returns:
            Score in [-1.0, 1.0]
        """
        if not completions:
            return -1.0  # No completions
        
        total_score = 0.0
        num_rounds = len(completions)
        
        # === Check all non-final rounds (should be tool_calls) ===
        for i in range(num_rounds - 1):
            round_score = self._compute_format_score(
                completions[i],
                ts_len,
                expect_tool_call=True,   # Should have tool_call
                expect_accept=False      # Should NOT have ACCEPT
            )
            
            if round_score < 0:
                return -1.0
            
            total_score += round_score
        
        # === Check final round (should be ACCEPT) ===
        final_score = self._compute_format_score(
            completions[-1],
            ts_len,
            expect_tool_call=False,  # Should NOT have tool_call
            expect_accept=True       # Should have ACCEPT
        )
        
        if final_score < 0:
            return -1.0
        
        total_score += final_score
        
        # Average score across all rounds
        avg_score = total_score / num_rounds
        
        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, avg_score))

    def _compute_format_score(self, completion, ts_len, expect_tool_call: bool, expect_accept: bool) -> float:
        """
        Compute format compliance score directly from completion text
        
        Requirements for controller:
        1. Must have EXACTLY ONE of: <answer>ACCEPT</answer> OR <tool_call>...</tool_call>
        2. Not both (ambiguous)
        3. Not neither (no decision)
        4. If <answer>, content must be "ACCEPT" (case-insensitive)
        5. If <tool_call>, must contain valid JSON
        
        Returns:
        - 1.0 = perfect format (one valid decision)
        - 0.5 = minor issues (e.g., extra whitespace, case issues)
        - 0.0 = severe format error (both, neither, or invalid)
        - Negative values for critical violations
        """
        score = 1.0

        # === Extract <think> blocks ===
        think_pattern = r'<think>\s*(.*?)\s*</think>'
        think_matches = re.findall(think_pattern, completion, re.IGNORECASE | re.DOTALL)
        if len(think_matches) == 0:
            # Missing think block: -0.5 penalty
            score -= 0.5
        elif len(think_matches) > 1:
            score -= 0.3
            # === Extract <answer> blocks ===
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_matches = re.findall(answer_pattern, completion, re.IGNORECASE | re.DOTALL)
        
        # === Extract <tool_call> blocks ===
        tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, completion, re.IGNORECASE | re.DOTALL)
        
        has_answer = len(answer_matches) > 0
        has_tool_call = len(tool_call_matches) > 0
        
        # Critical violations (score = 0 or negative)
        
        # Both answer and tool_call (ambiguous)
        if has_answer and has_tool_call:
            if self.gentle_reward:
                score -= 0.2
            else:
                score -= 0.5  # Severe: ambiguous decision
        
        # Case 2: Neither answer nor tool_call (no decision)
        if not has_answer and not has_tool_call:
            return -1.0  # Severe: no decision made
        
        # Case 3: Multiple answers
        if len(answer_matches) > 1:
            if self.gentle_reward:
                score -= 0.2
            else:
                score -= 0.5
        
        # Case 4: Multiple tool calls
        if len(tool_call_matches) > 1:
            score -= 0.5
        
        # Validate ACCEPT decision
        if expect_accept and has_answer:
            answer_content = answer_matches[0].strip()
            if not answer_content:
                return -1.0  # Empty answer
            
            # Must be "ACCEPT"
            if answer_content.upper() != "ACCEPT":
                if "ACCEPT" in answer_content.upper():
                    score -= 0.1
                else:
                    if self.gentle_reward and not has_tool_call:
                        score -= 0.3
                    else:
                        return -1.0  # Invalid answer content
            
        # Validate tool call decision
        if expect_tool_call and has_tool_call:
            # Validate tool call content
            tool_call_content = tool_call_matches[0].strip()
            
            if any(op in tool_call_content for op in ['+', '-', '*', '/', '%']):
                print(f"WARNING: Arithmetic expression detected in tool call: {tool_call_content[:100]}")
                return -1.0

            if not tool_call_content:
                return -1.0  # Empty tool call
            
            try:
                tool_call_json = json.loads(tool_call_content)
                
                # Check required fields
                if not isinstance(tool_call_json, dict):
                    return -1.0  # Not a dict
                
                # Must have "timeseries_zoom_in_tool" name and "arguments" field with "ts_seg" field
                if tool_call_json['name'] != "timeseries_zoom_in_tool" or "arguments" not in tool_call_json or "ts_seg" not in tool_call_json["arguments"]:
                    return -1.0  # not a proper format of the tool call
                
                segments = tool_call_json["arguments"]["ts_seg"]
                if not isinstance(segments, list) or len(segments) != 2:
                    return -1.0  # Segments not a list
                
                start, end = segments
                if start < 0 or end < 0 or start >= end:
                    return -1.0
                if start >= ts_len:
                    return -1.0
                if end > ts_len:
                    score -= 0.5
            
            except json.JSONDecodeError:
                return -1.0  # Invalid JSON
    
        return score

    def _calculate_penalty(self, num_trials: int) -> float:
        """Calculate efficiency penalty based on number of trials"""
        if self.penalty_type == "linear":
            return float(num_trials)
        
        elif self.penalty_type == "quadratic":
            return float(num_trials ** 2)
        
        elif self.penalty_type == "exponential":
            return float(np.exp(num_trials / self.max_trials) - 1)
        
        else:
            return float(num_trials)
    
    def _compute_format_reward(self, meta: Dict[str, Any]) -> float:
        """
        Reward for proper format compliance
        
        Checks:
        - Valid tool call OR accept decision
        - Clear, unambiguous decision
        """
        format_reward = 0.0
        
        # Valid decision (tool call or accept)
        if meta.get("tool_call_valid", False):
            format_reward += self.format_weight
        else:
            format_reward -= self.format_weight * 0.5
        
        # Proper termination (accepted when done)
        if meta.get("has_accept", False):
            format_reward += self.format_weight * 0.5
        
        return format_reward
    
    def _evaluate_trajectory(self, meta: Dict[str, Any]) -> float:
        """
        Evaluate trajectory quality
        
        Metrics:
        - Proper termination (accepted vs hit max rounds)
        - Segment efficiency (few segments with good coverage)
        - Segment overlap (penalize redundant selections)
        """
        score = 0.0
        
        # 1. Termination quality
        if meta.get("has_accept", False):
            score += 1.0  # Good: accepted when satisfied
        elif meta.get("hit_max_rounds", False):
            score -= 0.5  # Bad: ran out of rounds without accepting
        
        # # 2. Segment efficiency
        # all_segments = meta.get("all_segments", [])
        # num_segments = len(all_segments)
        
        # if num_segments <= 3:
        #     score += 0.5  # Good: efficient (few segments)
        # elif num_segments > 5:
        #     score -= 0.3  # Bad: too many segments
        
        # 3. Overlap penalty
        # if num_segments > 1:
        #     overlap_ratio = self._compute_overlap(all_segments)
        #     score -= overlap_ratio * 0.5
        
        return score

class ReasonerReward:
    """
    Reasoner reward based on format and accuracy
    """
    def __init__(self, 
                 accuracy_weight: float = 3.0,
                 format_weight: float = 1.0,
                 case_insensitive: bool = True,
                 strip_punct: bool = True,
                 uncertainty_reward: float = 0.0):
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight
        self.case_insensitive = case_insensitive
        self.strip_punct = strip_punct
        self.call_count = 0
        self.uncertainty_reward = uncertainty_reward
        self.__name__ = "ReasonerReward"
    
    def __call__(self, prompts: List[str], completions: List[str],
                 completion_ids: List[List[int]], **kwargs):
        """Calculate reasoner rewards"""
        metadata = kwargs.get("metadata", [{}] * len(completions))
        
        totals = []
        comps_list = []
        for i, (completion, meta) in enumerate(zip(completions, metadata)):
            reward, components = self._reward_one(completion, meta)
            totals.append(reward)
            comps_list.append(components)
        
        # Debug logging
        self.call_count += 1
        if self.call_count <= 3:
            print(f"\nReasoner Reward Batch {self.call_count}:")
            print(f"  Rewards: {[f'{r:.3f}' for r in totals]}")
            print(f"  Mean: {np.mean(totals):.3f}, Std: {np.std(totals):.4f}")
        
        return totals, comps_list
    
    def _reward_one(self, completion: str, meta: Dict):
        """Calculate reasoner reward"""
        components = {}
        # Accuracy reward
        accuracy = self._accuracy_reward(completion, meta)
        
        # Format reward
        format_score = self._format_reward(completion)

        uncertainty_bonus = self._uncertainty_bonus(completion, meta, accuracy)
        
        # Weighted combination
        total_reward = (
            accuracy * self.accuracy_weight +
            format_score * self.format_weight +
            uncertainty_bonus
        )
        
        components = {
            "accuracy_score_raw": accuracy,
            "format_score_raw": format_score,
            "accuracy_score_weighted": accuracy * self.accuracy_weight,
            "format_score_weighted": format_score * self.format_weight,
            "uncertainty_bonus_raw": uncertainty_bonus,
            # "total_weighted": total_reward,
        }
        if format_score < 0.0:
            return -1.0, components
        else:
            return float(total_reward), components
    
    def _accuracy_reward(self, completion: str, meta: Dict) -> float:
        """Check if answer is correct"""
        gold_answer = meta.get("gold_answer", "")
        pred_answer = self._extract_answer(completion)
        
        if not pred_answer:
            return 0.0
        
        # Normalize and compare
        if self._norm(pred_answer) == self._norm(str(gold_answer)):
            return 1.0
        
        return 0.0
    
    # def _uncertainty_bonus(self, completion: str, meta: Dict, accuracy: float) -> float:
    #     """
    #     Reward the reasoner for honestly expressing uncertainty when wrong.

    #     Conditions:
    #     - accuracy == 0.0 (answer is wrong)
    #     - 'gold_answer' is present in metadata (we actually know it's wrong)
    #     - answer text contains any of:
    #       'not sure', 'more information', 'uncertain'
    #     """
    #     # Must have a gold answer and be wrong
    #     if "gold_answer" not in meta or accuracy != 0.0:
    #         return 0.0
        
    #     answer_text = self._extract_full_answer(completion)
    #     if not answer_text:
    #         return 0.0
        
    #     text = answer_text.lower()
    #     uncertainty_markers = [
    #         "not sure",
    #         "more information",
    #         "uncertain",
    #         "i'm not sure",
    #         "i'm not sure about my answer",
    #         "not confident",
    #     ]
    #     if any(marker in text for marker in uncertainty_markers):
    #         return self.uncertainty_reward
        
    #     return 0.0
    def _uncertainty_bonus(self, completion: str, meta: Dict, accuracy: float) -> float:
        """
        Reward the reasoner for honestly expressing uncertainty when wrong.

        Conditions:
        - accuracy == 0.0 (answer is wrong)
        - 'gold_answer' is present in metadata
        - Uncertainty markers appear anywhere OUTSIDE <think> blocks
          (we accept both inside <answer> and stray text after it).
        """
        if "gold_answer" not in meta or accuracy != 0.0:
            return 0.0

        if not completion:
            return 0.0

        # 1) Remove ALL <think>...</think> blocks (we don't look at internal reasoning)
        completion_wo_think = re.sub(
            r'<think>.*?</think>',
            '',
            completion,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # 2) Remove <answer> tags themselves, but keep their contents
        #    (so we are robust to uncertainty either inside or just after </answer>)
        text_for_uncertainty = re.sub(
            r'</?answer>',
            '',
            completion_wo_think,
            flags=re.IGNORECASE,
        )

        text = text_for_uncertainty.lower()

        uncertainty_markers = [
            "i'm not sure",
            "im not sure",
            "not sure",
            "not confident",
            "more information",
            "more info",
            "i need more information",
            "i would need more information",
            "i am uncertain",
            "i'm uncertain",
        ]

        if any(marker in text for marker in uncertainty_markers):
            return self.uncertainty_reward

        return 0.0

    def _format_reward(self, completion: str) -> float:
        """
        Check format compliance with strict validation
        
        Requirements:
        - Exactly 1 <think> block
        - Exactly 1 <answer> block (outside think)
        - No <answer> inside <think>
        - Non-empty answer content
        
        Returns value in [0, 1]:
        - 1.0 = perfect format
        - Penalties for each violation
        """
        score = 1.0  # Start with perfect score
        
        # === Check <think> blocks ===
        think_pattern = r'<think>\s*(.*?)\s*</think>'
        think_matches = re.findall(think_pattern, completion, re.IGNORECASE | re.DOTALL)
        
        if len(think_matches) == 0:
            # Missing think block: -0.3
            score -= 0.5
        elif len(think_matches) > 1:
            # Multiple think blocks: -0.2 (less severe than missing)
            score -= 0.5
        # else: exactly 1 think block = perfect (no penalty)
        
        # === Check for <answer> inside <think> ===
        if think_matches:
            think_content = think_matches[0]  # Check first think block
            answer_in_think = re.findall(r'<answer>', think_content, re.IGNORECASE)
            if answer_in_think:
                # Answer inside think: -0.3 (serious format violation)
                score -= 0.5
        
        # === Check <answer> blocks (outside think) ===
        # Remove think blocks first
        completion_without_think = re.sub(think_pattern, '', completion, flags=re.IGNORECASE | re.DOTALL)
        
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_matches = re.findall(answer_pattern, completion_without_think, re.IGNORECASE | re.DOTALL)
        
        if len(answer_matches) == 0:
            # Missing answer block: -0.4 (most critical)
            return -1.0
        elif len(answer_matches) > 1:
            # Multiple answer blocks: -0.3
            score -= 0.5
        else:
            # Exactly 1 answer block - check if it's non-empty
            answer_content = answer_matches[0].strip()
            if not answer_content:
                # Empty answer: -0.2
                return -1.0
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _extract_answer(self, completion: str) -> str:
        """
        Extract answer from <answer> tags (outside <think> blocks)
        """
        # Remove <think> blocks first
        think_pattern = r'<think>\s*(.*?)\s*</think>'
        completion_without_think = re.sub(think_pattern, '', completion, flags=re.IGNORECASE | re.DOTALL)
        
        # Extract answer
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer_matches = re.findall(answer_pattern, completion_without_think, re.IGNORECASE | re.DOTALL)
        
        if answer_matches:
            # Return last answer if multiple (shouldn't happen with format penalty)
            return answer_matches[-1].strip()
        
        return ""
    
    def _extract_full_answer(self, completion: str) -> str:
        if not completion:
            return ""

        # 1. Remove ALL <think>...</think> blocks (non-greedy)
        completion_wo_think = re.sub(
            r'<think>.*?</think>',
            '',
            completion,
            flags=re.IGNORECASE | re.DOTALL
        )

        # 2. Extract ALL <answer>...</answer> blocks
        answer_blocks = re.findall(
            r'<answer>\s*(.*?)\s*</answer>',
            completion_wo_think,
            flags=re.IGNORECASE | re.DOTALL
        )

        if not answer_blocks:
            return ""

        raw = answer_blocks[-1].strip()

        # 3. Remove stray nested tags, if model hallucinated them
        raw = re.sub(r'</?answer>', '', raw, flags=re.IGNORECASE).strip()

        return raw

    def _norm(self, s: str) -> str:
        """Normalize string for comparison"""
        s = s.strip()
        
        if self.case_insensitive:
            s = s.lower()
        
        if self.strip_punct:
            s = s.translate(str.maketrans("", "", string.punctuation))
        
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        
        # Handle MCQ answers (A, B, C, D)
        mcq_match = re.match(r'^([a-d])\b', s)
        if mcq_match:
            return mcq_match.group(1)
        
        return s