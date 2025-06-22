"""
Utilities for identifying and removing duplicate or redundant content.
"""
from typing import List, Dict, Any, Tuple, Set
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from med_storm.models.evidence import EvidenceSource

@dataclass
class DuplicateGroup:
    """A group of duplicate or near-duplicate items."""
    items: List[Dict[str, Any]]  # List of items in this group
    similarity_score: float  # Average similarity score within the group
    group_key: str  # A key that identifies this group

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings (0.0 to 1.0)."""
    if not text1 or not text2:
        return 0.0
    
    # Simple character-based similarity
    seq = SequenceMatcher(None, text1, text2)
    return seq.ratio()

def calculate_content_fingerprint(text: str, method: str = 'simhash') -> str:
    """
    Generate a fingerprint for text content to detect near-duplicates.
    
    Args:
        text: The text to fingerprint
        method: The fingerprinting method ('simhash' or 'md5')
        
    Returns:
        A string fingerprint
    """
    if not text:
        return ""
    
    text = text.lower().strip()
    
    if method == 'md5':
        # Simple hash for exact duplicates
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # SimHash for near-duplicate detection
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return ""
    
    # Simple SimHash implementation
    v = [0] * 32  # 32-bit fingerprint
    
    for word in words:
        # Simple hash of each word
        h = hash(word) & 0xffffffff
        
        for i in range(32):
            bitmask = 1 << i
            if h & bitmask:
                v[i] += 1
            else:
                v[i] -= 1
    
    # Generate fingerprint
    fingerprint = 0
    for i in range(32):
        if v[i] > 0:
            fingerprint |= 1 << i
    
    return hex(fingerprint)

def find_duplicate_groups(
    items: List[Dict[str, Any]],
    content_key: str = 'content',
    id_key: str = 'id',
    similarity_threshold: float = 0.9,
    min_content_length: int = 10
) -> List[DuplicateGroup]:
    """
    Find groups of duplicate or near-duplicate items.
    
    Args:
        items: List of dictionaries containing content to deduplicate
        content_key: Key in the dictionaries that contains the content to compare
        id_key: Key in the dictionaries that contains the unique identifier
        similarity_threshold: Minimum similarity score (0.0-1.0) to consider items as duplicates
        min_content_length: Minimum content length to consider for deduplication
        
    Returns:
        List of DuplicateGroup objects
    """
    if not items:
        return []
    
    # Filter out items with insufficient content
    valid_items = [
        item for item in items 
        if content_key in item and 
        isinstance(item[content_key], str) and 
        len(item[content_key]) >= min_content_length
    ]
    
    if not valid_items:
        return []
    
    # First pass: exact duplicates using hashing
    hash_groups: Dict[str, List[Dict[str, Any]]] = {}
    
    for item in valid_items:
        content = item[content_key]
        content_hash = calculate_content_fingerprint(content, method='md5')
        
        if content_hash not in hash_groups:
            hash_groups[content_hash] = []
        hash_groups[content_hash].append(item)
    
    # Second pass: near-duplicates using SimHash and TF-IDF
    simhash_groups: Dict[str, List[Dict[str, Any]]] = {}
    
    for item in valid_items:
        content = item[content_key]
        simhash = calculate_content_fingerprint(content, method='simhash')
        
        if simhash not in simhash_groups:
            simhash_groups[simhash] = []
        simhash_groups[simhash].append(item)
    
    # Combine hash groups and simhash groups
    all_groups: Dict[str, List[Dict[str, Any]]] = {}
    
    for group in list(hash_groups.values()) + list(simhash_groups.values()):
        if len(group) <= 1:
            continue
            
        # Create a unique key for this group
        group_key = "|".join(sorted(str(item.get(id_key, '')) for item in group))
        
        if group_key not in all_groups:
            all_groups[group_key] = []
        
        # Add items to the group if they're not already there
        existing_ids = {item.get(id_key) for item in all_groups[group_key]}
        for item in group:
            if item.get(id_key) not in existing_ids:
                all_groups[group_key].append(item)
    
    # Third pass: use TF-IDF for more accurate similarity
    final_groups: List[DuplicateGroup] = []
    
    for group_key, group_items in all_groups.items():
        if len(group_items) <= 1:
            continue
            
        # Extract content
        contents = [item[content_key] for item in group_items]
        
        # Calculate TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            # Calculate pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate average similarity
            avg_similarity = np.mean(similarity_matrix)
            
            if avg_similarity >= similarity_threshold:
                final_groups.append(DuplicateGroup(
                    items=group_items,
                    similarity_score=float(avg_similarity),
                    group_key=group_key
                ))
        except Exception as e:
            # Fall back to simpler similarity if TF-IDF fails
            similarities = []
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    sim = calculate_text_similarity(contents[i], contents[j])
                    similarities.append(sim)
            
            if similarities and np.mean(similarities) >= similarity_threshold:
                final_groups.append(DuplicateGroup(
                    items=group_items,
                    similarity_score=float(np.mean(similarities)),
                    group_key=group_key
                ))
    
    return final_groups

def deduplicate_evidence_sources(
    sources: List[EvidenceSource],
    similarity_threshold: float = 0.85,
    keep: str = 'best'
) -> Tuple[List[EvidenceSource], List[DuplicateGroup]]:
    """
    Deduplicate a list of evidence sources.
    
    Args:
        sources: List of EvidenceSource objects to deduplicate
        similarity_threshold: Minimum similarity score to consider duplicates (0.0-1.0)
        keep: Which items to keep ('best', 'first', or 'last')
        
    Returns:
        Tuple of (deduplicated_sources, duplicate_groups)
    """
    if not sources:
        return [], []
    
    # Convert to dicts for processing
    items = [{
        'id': src.id,
        'content': f"{getattr(src, 'title', '')} {getattr(src, 'abstract', '')}".strip(),
        'source': src,
        'metadata': {
            'publication_date': getattr(src, 'publication_date', None),
            'journal': getattr(src, 'journal', ''),
            'authors': getattr(src, 'authors', []),
            'confidence': getattr(src, 'confidence', 0.0)
        }
    } for src in sources]
    
    # Find duplicate groups
    duplicate_groups = find_duplicate_groups(
        items,
        content_key='content',
        id_key='id',
        similarity_threshold=similarity_threshold
    )
    
    # Track which items to keep and which to remove
    keep_ids = set()
    removed_groups = []
    
    for group in duplicate_groups:
        if not group.items:
            continue
            
        # Determine which item to keep based on the 'keep' strategy
        if keep == 'first':
            keep_item = group.items[0]
        elif keep == 'last':
            keep_item = group.items[-1]
        else:  # 'best' - highest confidence, then most recent, then longest content
            keep_item = max(
                group.items,
                key=lambda x: (
                    x['metadata'].get('confidence', 0),
                    x['metadata'].get('publication_date') or '',
                    len(x.get('content', ''))
                )
            )
        
        keep_ids.add(keep_item['id'])
        
        # Store information about duplicates for reporting
        removed_items = [item for item in group.items if item['id'] != keep_item['id']]
        if removed_items:
            removed_groups.append(DuplicateGroup(
                items=[keep_item] + removed_items,
                similarity_score=group.similarity_score,
                group_key=group.group_key
            ))
    
    # Separate unique items from duplicates
    unique_items = [item for item in items if item['id'] not in {
        dup_id 
        for group in duplicate_groups 
        for dup_item in group.items 
        if (dup_id := dup_item['id']) != item['id']
    }]
    
    # Combine unique items with the kept duplicates
    kept_duplicates = [item for item in items if item['id'] in keep_ids]
    deduped_sources = [item['source'] for item in unique_items + kept_duplicates]
    
    return deduped_sources, removed_groups
