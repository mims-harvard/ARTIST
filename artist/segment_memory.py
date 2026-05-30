#!/usr/bin/env python3
"""
Segment Memory for Controller
Tracks TS segments selected during iterative reasoning
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SegmentEntry:
    """Single segment selection"""
    segment: List[int]  # [start, end]
    round_num: int
    
    def __repr__(self):
        return f"Segment({self.segment}, round={self.round_num})"


class SegmentMemory:
    """
    Tracks controller's segment selections across reasoning rounds
    Maintains both segment indices and their embeddings
    """
    def __init__(self, ts_length: int, min_segment_size: int = 60):
        self.ts_length = ts_length
        self.min_segment_size = min_segment_size
        self.entries: List[SegmentEntry] = []
        self.current_round = 0
    
    def add_segment(self, segment: List[int]) -> bool:
        """
        Add a segment to memory
        
        Returns:
            True if added successfully, False if invalid
        """
        # Validate segment
        if not self._is_valid_segment(segment):
            return False
        
        # Check for significant overlap with existing segments
        # if self._has_significant_overlap(segment):
        #     return False
        
        entry = SegmentEntry(
            segment=segment,
            round_num=self.current_round
        )
        self.entries.append(entry)
        return True
    
    def _is_valid_segment(self, segment: List[int]) -> bool:
        """Validate segment bounds and size"""
        if len(segment) != 2:
            return False
        
        start, end = segment
        
        # Check ordering
        if end <= start:
            return False
        
        # # Check minimum size
        # if (end - start) < self.min_segment_size:
        #     return False
        
        return True
    
    def _has_significant_overlap(self, new_segment: List[int], 
                                 overlap_threshold: float = 0.8) -> bool:
        """
        Check if new segment significantly overlaps with existing segments
        
        Args:
            new_segment: [start, end]
            overlap_threshold: Fraction of overlap to consider "significant"
        """
        new_start, new_end = new_segment
        new_length = new_end - new_start
        
        for entry in self.entries:
            old_start, old_end = entry.segment
            
            # Calculate overlap
            overlap_start = max(new_start, old_start)
            overlap_end = min(new_end, old_end)
            
            if overlap_end > overlap_start:
                overlap_length = overlap_end - overlap_start
                overlap_ratio = overlap_length / new_length
                
                if overlap_ratio > overlap_threshold:
                    return True
        
        return False
    
    def get_all_segments(self) -> List[List[int]]:
        """Get all segment indices"""
        return [entry.segment for entry in self.entries]
    
    def get_all_embeddings(self) -> List[torch.Tensor]:
        """Get all segment embeddings"""
        return [entry.embedding for entry in self.entries]
    
    def get_segment_count(self) -> int:
        """Get number of segments selected"""
        return len(self.entries)
    
    def increment_round(self):
        """Move to next reasoning round"""
        self.current_round += 1
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get serializable state for logging/debugging"""
        return {
            "segments": self.get_all_segments(),
            "round_count": self.current_round,
            "segment_count": self.get_segment_count()
        }
    
    def format_for_reasoner(self) -> str:
        """
        Format segments for reasoner input
        Returns text description of available segments
        """
        if len(self.entries) == 0:
            return "No specific segments selected yet."
        
        lines = ["Current TS segments:"]
        for i, entry in enumerate(self.entries, 1):
            start, end = entry.segment
            lines.append(f"  Segment {i}: indices [{start}, {end}]")
        
        return "\n".join(lines)
    
    def clear(self):
        """Reset memory"""
        self.entries.clear()
        self.current_round = 0


class SegmentMemoryManager:
    """
    Manages segment memories for a batch of examples
    TODO: not sure if this is needed
    """
    def __init__(self, batch_size: int, ts_lengths: List[int], min_segment_size: int = 60):
        self.memories = [
            SegmentMemory(ts_len, min_segment_size) 
            for ts_len in ts_lengths
        ]
    
    def add_segment(self, idx: int, segment: List[int], embedding: torch.Tensor) -> bool:
        """Add segment to specific example's memory"""
        return self.memories[idx].add_segment(segment, embedding)
    
    def get_memory(self, idx: int) -> SegmentMemory:
        """Get memory for specific example"""
        return self.memories[idx]
    
    def increment_all_rounds(self):
        """Increment round counter for all memories"""
        for mem in self.memories:
            mem.increment_round()
    
    def get_all_states(self) -> List[Dict[str, Any]]:
        """Get states of all memories"""
        return [mem.get_state_dict() for mem in self.memories]
    
    def clear_all(self):
        """Clear all memories"""
        for mem in self.memories:
            mem.clear()