"""
Elastic matching utility for processing multi-choice prompt results.

This module provides a simplified and unified implementation of elastic matching
functionality that was previously duplicated across multiple files.
"""

import re
from typing import List, Union


def elastic_matching(prompt_result: Union[str, None], entity_candidates: List[str]) -> str:
    """
    Process a multi-choice prompt result to select the best matching entity candidate.
    
    This function implements "elastic matching" logic that:
    1. Handles edge cases (empty candidates, single candidate, none matches)
    2. Extracts numerical indices from the prompt result using regex
    3. Falls back to string matching if index matching fails
    4. Avoids ambiguous matches when one candidate is a substring of another
    
    Args:
        prompt_result: The response from LLM for multi-choice selection
        entity_candidates: List of entity candidate strings to choose from
        
    Returns:
        str: The selected entity candidate, or empty string if no match found
    """
    # Handle edge cases
    if not entity_candidates:
        return ''
    
    if len(entity_candidates) == 1:
        return entity_candidates[0]
    
    if not prompt_result or not isinstance(prompt_result, str):
        return ''
    
    # Check for explicit "none" responses
    if _contains_none_response(prompt_result):
        return ''
    
    # Try index-based matching first
    selected_entity = _match_by_index(prompt_result, entity_candidates)
    if selected_entity:
        return selected_entity
    
    # Fall back to string-based matching
    selected_entity = _match_by_string(prompt_result, entity_candidates)
    return selected_entity


def _contains_none_response(prompt_result: str) -> bool:
    """Check if the prompt result indicates no match."""
    none_indicators = ['None of the entity match', ' not ', "doesn't", 'none']
    prompt_lower = prompt_result.lower()
    return any(indicator.lower() in prompt_lower for indicator in none_indicators)


def _match_by_index(prompt_result: str, entity_candidates: List[str]) -> str:
    """
    Extract numerical indices from prompt result and return corresponding candidate.
    
    Returns:
        str: Selected entity candidate or empty string if no clear match
    """
    # Extract all valid indices (1-based in prompt, convert to 0-based)
    valid_indices = [
        int(match) - 1 
        for match in re.findall(r'\b\d+\b', prompt_result)
        if 0 <= int(match) - 1 < len(entity_candidates)
    ]
    
    if len(valid_indices) == 1:
        return entity_candidates[valid_indices[0]]
    
    # If multiple indices but more than 2 candidates, prefer the first mentioned
    if len(valid_indices) == 2 and len(entity_candidates) > 2:
        return entity_candidates[valid_indices[0]]
    
    return ''


def _match_by_string(prompt_result: str, entity_candidates: List[str]) -> str:
    """
    Match entity candidates by finding them as substrings in the prompt result.
    
    Avoids ambiguous matches where one candidate is contained in another.
    
    Returns:
        str: Selected entity candidate or empty string if no unambiguous match
    """
    prompt_lower = prompt_result.lower()
    matching_indices = []
    
    for index, candidate in enumerate(entity_candidates):
        if candidate.lower() in prompt_lower:
            # Check if this candidate is uniquely mentioned (not a substring of another mentioned candidate)
            if _is_unique_mention(candidate, entity_candidates, prompt_lower, index):
                matching_indices.append(index)
    
    # Return the candidate only if there's exactly one unambiguous match
    if len(matching_indices) == 1:
        return entity_candidates[matching_indices[0]]
    
    return ''


def _is_unique_mention(candidate: str, all_candidates: List[str], prompt_lower: str, candidate_index: int) -> bool:
    """
    Check if a candidate is uniquely mentioned (not overshadowed by a longer candidate that contains it).
    
    Args:
        candidate: The candidate to check
        all_candidates: All entity candidates
        prompt_lower: Lowercase prompt result
        candidate_index: Index of the candidate being checked
        
    Returns:
        bool: True if the candidate is uniquely mentioned
    """
    candidate_lower = candidate.lower()
    
    # Check against all other candidates
    for other_index, other_candidate in enumerate(all_candidates):
        if other_index == candidate_index:
            continue
            
        other_lower = other_candidate.lower()
        
        # If this candidate is contained in another candidate AND that other candidate is also in the prompt,
        # then this mention is ambiguous
        if (candidate_lower in other_lower and other_lower in prompt_lower):
            return False
    
    return True