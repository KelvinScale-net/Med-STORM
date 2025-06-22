from __future__ import annotations

from typing import List, TYPE_CHECKING, Dict, Any, Optional
import logging
import asyncio
from dataclasses import dataclass

from med_storm.llm.base import LLMProvider
from med_storm.models.evidence import EvidenceCorpus, EvidenceSource

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from med_storm.models.evidence import EvidenceCorpus, EvidenceSource

@dataclass
class SynthesisChunk:
    """Represents a synthesis chunk with size control."""
    content: str
    token_estimate: int
    priority: int = 0

class IntelligentSynthesisEngine:
    """🧠 ULTRA-SMART: Synthesis engine with context window management"""
    
    def __init__(self, llm_provider: LLMProvider, max_context_tokens: int = 60000):
        self.llm_provider = llm_provider
        self.max_context_tokens = max_context_tokens
        self.max_chunk_tokens = max_context_tokens // 2  # Safety buffer
        
    def estimate_tokens(self, text: str) -> int:
        """📏 Estimate token count (rough approximation)"""
        return len(text.split()) * 1.3  # Conservative estimate
    
    def chunk_evidence_intelligently(self, evidence_corpus: EvidenceCorpus) -> List[SynthesisChunk]:
        """🔪 SMART: Split evidence into manageable chunks"""
        chunks = []
        current_content = ""
        current_tokens = 0
        
        # Sort sources by confidence for intelligent prioritization
        sorted_sources = sorted(
            evidence_corpus.sources, 
            key=lambda x: x.confidence_score or 0, 
            reverse=True
        )
        
        for source in sorted_sources:
            source_text = f"""
**Source {source.id}** (Confidence: {source.confidence_score:.2f})
Title: {source.title}
Content: {source.content}
---
"""
            source_tokens = self.estimate_tokens(source_text)
            
            # Check if adding this source would exceed chunk limit
            if current_tokens + source_tokens > self.max_chunk_tokens and current_content:
                # Create chunk from current content
                chunks.append(SynthesisChunk(
                    content=current_content.strip(),
                    token_estimate=current_tokens,
                    priority=len(chunks)
                ))
                # Start new chunk
                current_content = source_text
                current_tokens = source_tokens
            else:
                current_content += source_text
                current_tokens += source_tokens
        
        # Add final chunk
        if current_content:
            chunks.append(SynthesisChunk(
                content=current_content.strip(),
                token_estimate=current_tokens,
                priority=len(chunks)
            ))
        
        logger.info(f"🧠 Created {len(chunks)} intelligent synthesis chunks")
        return chunks
    
    async def synthesize_evidence_smart(self, question: str, evidence_corpus: EvidenceCorpus) -> str:
        """🚀 REVOLUTIONARY: Smart synthesis with context management"""
        
        if not evidence_corpus.sources:
            return "No evidence found for this question."
        
        # Estimate total content size
        total_content = f"Question: {question}\n\n" + "\n".join([
            f"Source: {s.title}\nContent: {s.content}" for s in evidence_corpus.sources
        ])
        
        total_tokens = self.estimate_tokens(total_content)
        
        # If content fits in one request, use standard synthesis
        if total_tokens <= self.max_chunk_tokens:
            return await self._synthesize_single_chunk(question, evidence_corpus)
        
        # Use intelligent chunking for large content
        logger.info(f"🧠 Content too large ({total_tokens} tokens), using intelligent chunking")
        chunks = self.chunk_evidence_intelligently(evidence_corpus)
        
        # Synthesize each chunk
        chunk_syntheses = []
        for i, chunk in enumerate(chunks):
            logger.info(f"🔥 Synthesizing chunk {i+1}/{len(chunks)} ({chunk.token_estimate} tokens)")
            
            # Create mini corpus for this chunk
            chunk_sources = []
            lines = chunk.content.split('\n')
            current_source = {}
            
            for line in lines:
                if line.startswith('**Source'):
                    if current_source:
                        chunk_sources.append(current_source)
                    current_source = {'content': ''}
                elif line.startswith('Title:'):
                    current_source['title'] = line.replace('Title:', '').strip()
                elif line.startswith('Content:'):
                    current_source['content'] = line.replace('Content:', '').strip()
                elif line == '---':
                    continue
                else:
                    if 'content' in current_source:
                        current_source['content'] += ' ' + line
            
            if current_source:
                chunk_sources.append(current_source)
            
            # Synthesize this chunk
            chunk_prompt = f"""
Based on the following evidence sources, provide a comprehensive synthesis answering: "{question}"

Evidence Sources:
{chunk.content}

Please provide a detailed, evidence-based response that:
1. Directly addresses the question
2. Synthesizes information across sources
3. Highlights key findings and conclusions
4. Notes any limitations or gaps

Synthesis:"""

            try:
                chunk_synthesis = await self.llm_provider.generate(chunk_prompt)
                chunk_syntheses.append(f"## Synthesis Part {i+1}\n{chunk_synthesis}")
            except Exception as e:
                logger.error(f"Error synthesizing chunk {i+1}: {e}")
                chunk_syntheses.append(f"## Synthesis Part {i+1}\nError processing this evidence section.")
        
        # Combine all chunk syntheses into final synthesis
        if len(chunk_syntheses) == 1:
            return chunk_syntheses[0].replace("## Synthesis Part 1\n", "")
        
        # Create final combined synthesis
        combined_content = "\n\n".join(chunk_syntheses)
        
        # If combined synthesis is still too large, summarize it
        if self.estimate_tokens(combined_content) > self.max_chunk_tokens:
            return await self._synthesize_combined_chunks(question, chunk_syntheses)
        
        final_prompt = f"""
The following are partial syntheses of evidence for the question: "{question}"

{combined_content}

Please create a final, coherent synthesis that:
1. Integrates all the partial syntheses
2. Eliminates redundancy
3. Provides a comprehensive answer
4. Maintains scientific rigor

Final Synthesis:"""

        try:
            return await self.llm_provider.generate(final_prompt)
        except Exception as e:
            logger.error(f"Error in final synthesis: {e}")
            return combined_content  # Fallback to combined chunks
    
    async def _synthesize_single_chunk(self, question: str, evidence_corpus: EvidenceCorpus) -> str:
        """📝 Standard synthesis for content that fits in context"""
        evidence_text = "\n\n".join([
            f"**Source {i+1}**: {source.title}\n{source.content}"
            for i, source in enumerate(evidence_corpus.sources)
        ])
        
        prompt = f"""
Based on the following evidence sources, provide a comprehensive synthesis answering: "{question}"

Evidence Sources:
{evidence_text}

Please provide a detailed, evidence-based response that:
1. Directly addresses the question
2. Synthesizes information across sources
3. Highlights key findings and conclusions
4. Notes any limitations or gaps

Synthesis:"""

        return await self.llm_provider.generate(prompt)
    
    async def _synthesize_combined_chunks(self, question: str, chunk_syntheses: List[str]) -> str:
        """🎯 Final synthesis of multiple chunks"""
        summary_prompt = f"""
Question: "{question}"

The following are detailed partial syntheses that together answer this question:

{chr(10).join(chunk_syntheses[:3])}  # Limit to first 3 to avoid overflow

Please create a final, comprehensive synthesis that integrates these findings into a coherent response.

Final Synthesis:"""

        try:
            return await self.llm_provider.generate(summary_prompt)
        except Exception as e:
            logger.error(f"Error in combined synthesis: {e}")
            # Fallback: return structured combination
            return f"""
# Synthesis: {question}

{chr(10).join(chunk_syntheses)}

**Note**: This synthesis was created from multiple evidence chunks due to content size limitations.
"""

# Alias for backward compatibility
SynthesisEngine = IntelligentSynthesisEngine
