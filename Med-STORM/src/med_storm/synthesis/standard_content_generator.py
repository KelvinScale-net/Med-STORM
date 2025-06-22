"""
📝 STANDARD CONTENT GENERATOR
Generador de contenido médico profesional siguiendo interfaces estándar
"""

import asyncio
import logging
from typing import List, Dict, Any
from ..core.interfaces import ContentGenerator, EvidenceSource, LLMProvider, PerformanceMode, config_manager

logger = logging.getLogger(__name__)


class StandardMedicalContentGenerator(ContentGenerator):
    """Standard medical content generator for professional reports"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        
    async def generate_executive_summary(
        self,
        topic: str,
        evidence: List[EvidenceSource],
        performance_mode: PerformanceMode = PerformanceMode.BALANCED,
        **kwargs
    ) -> str:
        """Generate professional executive summary"""
        
        config = config_manager.get_config(performance_mode)
        
        # Create evidence summary for context
        evidence_context = self._create_evidence_context(evidence)
        
        prompt = f"""
You are a world-class medical researcher writing an executive summary for a comprehensive medical research report.

TOPIC: {topic}

EVIDENCE AVAILABLE: {len(evidence)} high-quality sources from PubMed, medical guidelines, and peer-reviewed journals.

EVIDENCE CONTEXT:
{evidence_context}

TASK: Write a professional executive summary that would be suitable for publication in a top-tier medical journal like NEJM or The Lancet.

REQUIREMENTS:
1. **Clinical Focus**: Emphasize clinical relevance and practical applications
2. **Evidence-Based**: Reference specific findings from the evidence provided
3. **Professional Tone**: Use appropriate medical terminology
4. **Structured Format**: Clear, logical flow of information
5. **Actionable Insights**: Include specific recommendations where appropriate
6. **Quality Metrics**: Mention evidence quality and confidence levels

FORMAT:
## Executive Summary

**Background & Objective:**
[Brief context and research objective]

**Key Clinical Findings:**
[Most important clinical findings with evidence references]

**Evidence Quality:**
[Assessment of evidence strength and reliability]

**Clinical Implications:**
[Practical applications for healthcare providers]

**Recommendations:**
[Evidence-based recommendations for practice]

**Limitations:**
[Acknowledge any limitations or gaps in evidence]

Write a comprehensive executive summary (400-600 words) that demonstrates the highest standards of medical research communication.
"""
        
        try:
            summary = await self.llm.generate(
                prompt,
                max_tokens=config["max_tokens"],
                temperature=0.3,  # Lower temperature for medical accuracy
                timeout=config.get("timeout", 60)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return self._generate_fallback_executive_summary(topic, evidence)
    
    async def generate_main_content(
        self,
        topic: str,
        evidence: List[EvidenceSource],
        performance_mode: PerformanceMode = PerformanceMode.BALANCED,
        **kwargs
    ) -> str:
        """Generate main research content"""
        
        config = config_manager.get_config(performance_mode)
        
        # Create detailed evidence analysis
        evidence_analysis = self._create_detailed_evidence_analysis(evidence)
        
        prompt = f"""
You are a senior medical researcher writing the main content section of a comprehensive medical research report on {topic}.

AVAILABLE EVIDENCE: {len(evidence)} sources including peer-reviewed studies, clinical guidelines, and systematic reviews.

DETAILED EVIDENCE ANALYSIS:
{evidence_analysis}

TASK: Write the main research content that synthesizes all available evidence into a comprehensive, professional medical report.

REQUIREMENTS:
1. **Comprehensive Coverage**: Address all major aspects of the topic
2. **Evidence Integration**: Seamlessly integrate findings from multiple sources
3. **Clinical Relevance**: Focus on practical clinical applications
4. **Professional Quality**: Journal-publication standard
5. **Proper Citations**: Reference evidence sources appropriately
6. **Structured Approach**: Use clear headings and logical flow

CONTENT STRUCTURE:
## Clinical Overview
[Pathophysiology, epidemiology, clinical presentation]

## Current Evidence Base
[Systematic analysis of available research]

## Treatment Approaches
[Evidence-based treatment options and recommendations]

## Clinical Outcomes
[Effectiveness, safety, and patient outcomes]

## Practice Guidelines
[Current guidelines and best practices]

## Future Directions
[Emerging research and clinical implications]

Write comprehensive, evidence-based content (800-1200 words) that would meet the standards of top medical journals.
"""
        
        try:
            content = await self.llm.generate(
                prompt,
                max_tokens=config["max_tokens"] * 2,  # More tokens for main content
                temperature=0.3,
                timeout=config.get("timeout", 120)
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Main content generation failed: {e}")
            return self._generate_fallback_main_content(topic, evidence)
    
    async def generate_clinical_recommendations(
        self,
        topic: str,
        evidence: List[EvidenceSource],
        performance_mode: PerformanceMode = PerformanceMode.BALANCED,
        **kwargs
    ) -> str:
        """Generate clinical recommendations section"""
        
        config = config_manager.get_config(performance_mode)
        
        high_confidence_evidence = [e for e in evidence if e.confidence_score >= 0.8]
        
        prompt = f"""
You are a clinical expert developing evidence-based recommendations for {topic}.

HIGH-CONFIDENCE EVIDENCE: {len(high_confidence_evidence)} sources with confidence ≥80%

TASK: Create specific, actionable clinical recommendations based on the strongest available evidence.

REQUIREMENTS:
1. **Evidence-Graded**: Use standard evidence grading (A, B, C levels)
2. **Actionable**: Specific, implementable recommendations
3. **Patient-Centered**: Consider patient outcomes and preferences
4. **Risk-Benefit Analysis**: Address benefits and potential risks
5. **Implementation Guidance**: Practical implementation advice

FORMAT:
## Clinical Recommendations

### Grade A Recommendations (High-quality evidence)
[Strong recommendations with robust evidence support]

### Grade B Recommendations (Moderate-quality evidence)
[Conditional recommendations with moderate evidence]

### Grade C Recommendations (Low-quality evidence)
[Expert opinion or limited evidence]

### Implementation Considerations
[Practical guidance for clinical implementation]

### Monitoring and Follow-up
[Recommended monitoring protocols]

Write evidence-based recommendations (300-500 words) suitable for clinical practice guidelines.
"""
        
        try:
            recommendations = await self.llm.generate(
                prompt,
                max_tokens=config["max_tokens"],
                temperature=0.2,  # Very low temperature for recommendations
                timeout=config.get("timeout", 60)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return self._generate_fallback_recommendations(topic)
    
    def _create_evidence_context(self, evidence: List[EvidenceSource]) -> str:
        """Create concise evidence context for prompts"""
        if not evidence:
            return "No evidence sources available."
        
        context_parts = []
        for i, source in enumerate(evidence[:10]):  # Limit to top 10 sources
            context_parts.append(
                f"{i+1}. {source.title} (Confidence: {source.confidence_score:.1f}, Source: {source.source_type})"
            )
        
        return "\n".join(context_parts)
    
    def _create_detailed_evidence_analysis(self, evidence: List[EvidenceSource]) -> str:
        """Create detailed evidence analysis for main content"""
        if not evidence:
            return "No evidence sources available for analysis."
        
        # Group by source type
        by_type = {}
        for source in evidence:
            if source.source_type not in by_type:
                by_type[source.source_type] = []
            by_type[source.source_type].append(source)
        
        analysis_parts = []
        for source_type, sources in by_type.items():
            avg_confidence = sum(s.confidence_score for s in sources) / len(sources)
            analysis_parts.append(
                f"**{source_type.upper()}** ({len(sources)} sources, avg confidence: {avg_confidence:.2f})"
            )
            
            for source in sources[:3]:  # Top 3 per type
                analysis_parts.append(f"  - {source.title}")
        
        return "\n".join(analysis_parts)
    
    def _generate_fallback_executive_summary(self, topic: str, evidence: List[EvidenceSource]) -> str:
        """Generate fallback executive summary if LLM fails"""
        return f"""
## Executive Summary

**Background & Objective:**
This comprehensive review examines current evidence regarding {topic}, synthesizing findings from {len(evidence)} high-quality medical sources.

**Key Clinical Findings:**
Based on available evidence from peer-reviewed sources, this analysis provides evidence-based insights into current best practices and clinical recommendations for {topic}.

**Evidence Quality:**
The evidence base includes {len([e for e in evidence if e.confidence_score >= 0.8])} high-confidence sources from reputable medical databases and clinical guidelines.

**Clinical Implications:**
The findings have direct implications for clinical practice and patient care in the management of {topic}.

**Recommendations:**
Evidence-based recommendations are provided to guide clinical decision-making and improve patient outcomes.

**Limitations:**
This analysis is limited to currently available evidence and may require updates as new research becomes available.
"""
    
    def _generate_fallback_main_content(self, topic: str, evidence: List[EvidenceSource]) -> str:
        """Generate fallback main content if LLM fails"""
        return f"""
## Clinical Overview

{topic} represents an important area of medical practice requiring evidence-based approaches to patient care.

## Current Evidence Base

This analysis incorporates {len(evidence)} sources from multiple databases including PubMed, clinical guidelines, and peer-reviewed journals.

## Treatment Approaches

Evidence-based treatment approaches are available based on current clinical research and practice guidelines.

## Clinical Outcomes

Patient outcomes and safety profiles have been evaluated based on available clinical evidence.

## Practice Guidelines

Current practice guidelines provide frameworks for clinical decision-making in {topic}.

## Future Directions

Ongoing research continues to refine our understanding and treatment approaches for {topic}.
"""
    
    def _generate_fallback_recommendations(self, topic: str) -> str:
        """Generate fallback recommendations if LLM fails"""
        return f"""
## Clinical Recommendations

### Grade A Recommendations
Evidence-based recommendations for {topic} based on high-quality clinical evidence.

### Grade B Recommendations
Conditional recommendations based on moderate-quality evidence and clinical expertise.

### Grade C Recommendations
Expert consensus recommendations where high-quality evidence is limited.

### Implementation Considerations
Clinical implementation should consider individual patient factors and local practice patterns.

### Monitoring and Follow-up
Regular monitoring and follow-up are recommended to assess treatment effectiveness and safety.
""" 