"""
💊 COMPREHENSIVE TREATMENT ANALYSIS MODULE
==========================================

FEATURES:
1. PHARMACOTHERAPY ANALYSIS: Drug efficacy, dosing, interactions
2. THERAPEUTIC INTERVENTIONS: Procedures, surgeries, devices
3. ALTERNATIVE THERAPIES: Evidence-based complementary treatments
4. COMPARATIVE EFFECTIVENESS: Head-to-head treatment comparisons
5. SAFETY PROFILES: Adverse events, contraindications, monitoring
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from med_storm.models.evidence import EvidenceCorpus, EvidenceSource
from med_storm.llm.base import LLMProvider

logger = logging.getLogger(__name__)

class TreatmentCategory(Enum):
    """💊 Treatment category classification"""
    PHARMACOLOGICAL = "pharmacological"
    SURGICAL = "surgical"
    DEVICE_BASED = "device_based"
    BEHAVIORAL = "behavioral"
    COMPLEMENTARY = "complementary"
    PREVENTIVE = "preventive"

class EvidenceStrength(Enum):
    """📊 Treatment evidence strength"""
    STRONG = "strong"          # Multiple high-quality RCTs
    MODERATE = "moderate"      # Some RCTs or strong observational
    WEAK = "weak"             # Limited evidence
    INSUFFICIENT = "insufficient"  # Inadequate evidence

@dataclass
class TreatmentEvidence:
    """💊 Treatment evidence summary"""
    treatment_name: str
    category: TreatmentCategory
    evidence_strength: EvidenceStrength
    efficacy_summary: str
    safety_profile: str
    dosing_regimen: Optional[str] = None
    contraindications: List[str] = None
    drug_interactions: List[str] = None
    monitoring_requirements: List[str] = None
    cost_considerations: Optional[str] = None
    comparative_effectiveness: Dict[str, str] = None

class TreatmentAnalyzer:
    """💊 Comprehensive treatment and pharmacotherapy analyzer"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.llm_semaphore = asyncio.Semaphore(10)
        
        # Treatment-focused search queries
        self.treatment_query_templates = {
            "pharmacological": [
                "{topic} drug therapy",
                "{topic} pharmacotherapy",
                "{topic} medication",
                "{topic} pharmaceutical treatment",
                "{topic} drug efficacy",
                "{topic} dosing",
                "{topic} drug safety",
                "{topic} adverse effects",
                "{topic} drug interactions"
            ],
            "surgical": [
                "{topic} surgical treatment",
                "{topic} surgery",
                "{topic} operative treatment",
                "{topic} surgical outcomes",
                "{topic} minimally invasive",
                "{topic} surgical complications"
            ],
            "device_based": [
                "{topic} medical device",
                "{topic} implantable device",
                "{topic} therapeutic device",
                "{topic} device therapy"
            ],
            "behavioral": [
                "{topic} behavioral therapy",
                "{topic} cognitive therapy",
                "{topic} lifestyle intervention",
                "{topic} behavioral modification"
            ],
            "complementary": [
                "{topic} complementary therapy",
                "{topic} alternative medicine",
                "{topic} integrative treatment",
                "{topic} herbal therapy",
                "{topic} acupuncture",
                "{topic} nutritional therapy"
            ],
            "preventive": [
                "{topic} prevention",
                "{topic} prophylaxis",
                "{topic} preventive therapy",
                "{topic} risk reduction"
            ]
        }

    async def analyze_treatments(
        self, 
        topic: str, 
        evidence_map: Dict[str, EvidenceCorpus]
    ) -> Dict[str, Any]:
        """💊 Comprehensive treatment analysis"""
        
        logger.info(f"💊 Analyzing treatments for: {topic}")
        
        # Extract treatment-related evidence
        treatment_evidence = self._extract_treatment_evidence(evidence_map)
        
        # Analyze by treatment category
        analysis_tasks = []
        for category in TreatmentCategory:
            task = self._analyze_treatment_category(
                topic, category, treatment_evidence
            )
            analysis_tasks.append(task)
        
        category_analyses = await asyncio.gather(*analysis_tasks)
        
        # Combine results
        treatment_analysis = {}
        for category, analysis in zip(TreatmentCategory, category_analyses):
            treatment_analysis[category.value] = analysis
        
        # Generate comparative effectiveness analysis
        comparative_analysis = await self._generate_comparative_analysis(
            topic, treatment_analysis
        )
        
        # Generate treatment recommendations
        recommendations = await self._generate_treatment_recommendations(
            topic, treatment_analysis, comparative_analysis
        )
        
        return {
            "treatment_analysis": treatment_analysis,
            "comparative_effectiveness": comparative_analysis,
            "recommendations": recommendations,
            "safety_summary": await self._generate_safety_summary(treatment_analysis)
        }

    def _extract_treatment_evidence(
        self, 
        evidence_map: Dict[str, EvidenceCorpus]
    ) -> Dict[str, List[EvidenceSource]]:
        """🔍 Extract treatment-related evidence from corpus"""
        
        treatment_evidence = {}
        
        for question, corpus in evidence_map.items():
            # Categorize evidence by treatment keywords
            for source in corpus.sources:
                content = (source.title + " " + source.summary).lower()
                
                # Pharmacological treatments
                if any(term in content for term in [
                    "drug", "medication", "pharmaceutical", "therapy", "treatment",
                    "dosage", "dose", "administration", "efficacy", "effectiveness"
                ]):
                    if "pharmacological" not in treatment_evidence:
                        treatment_evidence["pharmacological"] = []
                    treatment_evidence["pharmacological"].append(source)
                
                # Surgical treatments
                if any(term in content for term in [
                    "surgery", "surgical", "operation", "procedure", "resection",
                    "implant", "transplant", "bypass", "repair"
                ]):
                    if "surgical" not in treatment_evidence:
                        treatment_evidence["surgical"] = []
                    treatment_evidence["surgical"].append(source)
                
                # Device-based treatments
                if any(term in content for term in [
                    "device", "implant", "pacemaker", "stent", "catheter",
                    "pump", "monitor", "prosthetic"
                ]):
                    if "device_based" not in treatment_evidence:
                        treatment_evidence["device_based"] = []
                    treatment_evidence["device_based"].append(source)
                
                # Behavioral treatments
                if any(term in content for term in [
                    "behavioral", "cognitive", "therapy", "counseling",
                    "lifestyle", "diet", "exercise", "rehabilitation"
                ]):
                    if "behavioral" not in treatment_evidence:
                        treatment_evidence["behavioral"] = []
                    treatment_evidence["behavioral"].append(source)
                
                # Complementary treatments
                if any(term in content for term in [
                    "alternative", "complementary", "herbal", "natural",
                    "acupuncture", "massage", "meditation", "yoga"
                ]):
                    if "complementary" not in treatment_evidence:
                        treatment_evidence["complementary"] = []
                    treatment_evidence["complementary"].append(source)
        
        return treatment_evidence

    async def _analyze_treatment_category(
        self,
        topic: str,
        category: TreatmentCategory,
        treatment_evidence: Dict[str, List[EvidenceSource]]
    ) -> Dict[str, Any]:
        """💊 Analyze specific treatment category"""
        
        category_sources = treatment_evidence.get(category.value, [])
        
        if not category_sources:
            return {
                "treatments": [],
                "evidence_summary": f"No {category.value} evidence found for {topic}",
                "recommendations": "Insufficient evidence for recommendations"
            }
        
        # Generate category-specific analysis
        system_prompt = self._get_category_analysis_prompt(category)
        
        # Prepare evidence context
        evidence_context = self._prepare_evidence_context(category_sources)
        
        prompt = f"""
Topic: {topic}
Treatment Category: {category.value}

Evidence Context:
{evidence_context}

Please provide a comprehensive analysis including:
1. Specific treatments identified
2. Evidence quality and strength
3. Efficacy outcomes
4. Safety considerations
5. Practical implementation considerations
"""
        
        async with self.llm_semaphore:
            analysis = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        # Extract specific treatments
        treatments = await self._extract_specific_treatments(
            topic, category, category_sources
        )
        
        return {
            "treatments": treatments,
            "evidence_summary": analysis,
            "source_count": len(category_sources),
            "evidence_quality": self._assess_category_evidence_quality(category_sources)
        }

    def _get_category_analysis_prompt(self, category: TreatmentCategory) -> str:
        """📝 Get category-specific analysis prompt"""
        
        prompts = {
            TreatmentCategory.PHARMACOLOGICAL: """
You are a clinical pharmacologist analyzing drug treatments. Focus on:
- Specific medications and drug classes
- Mechanisms of action
- Dosing regimens and administration
- Efficacy outcomes and endpoints
- Adverse effects and safety profiles
- Drug interactions and contraindications
- Monitoring requirements
- Cost-effectiveness considerations
""",
            TreatmentCategory.SURGICAL: """
You are a surgeon analyzing surgical treatments. Focus on:
- Specific surgical procedures and techniques
- Indications and patient selection criteria
- Surgical outcomes and success rates
- Complications and morbidity/mortality
- Recovery time and rehabilitation
- Minimally invasive vs. open approaches
- Long-term outcomes
""",
            TreatmentCategory.DEVICE_BASED: """
You are a medical device specialist analyzing device-based treatments. Focus on:
- Specific medical devices and technologies
- Device mechanisms and functionality
- Implantation or application procedures
- Device efficacy and performance
- Device-related complications
- Maintenance and monitoring requirements
- Device longevity and replacement needs
""",
            TreatmentCategory.BEHAVIORAL: """
You are a behavioral medicine specialist analyzing non-pharmacological interventions. Focus on:
- Specific behavioral interventions and therapies
- Implementation protocols and duration
- Patient compliance and adherence
- Behavioral outcomes and effectiveness
- Lifestyle modification strategies
- Patient education requirements
- Long-term sustainability
""",
            TreatmentCategory.COMPLEMENTARY: """
You are an integrative medicine specialist analyzing complementary therapies. Focus on:
- Specific complementary and alternative treatments
- Evidence quality and research limitations
- Integration with conventional treatments
- Safety considerations and interactions
- Patient selection and contraindications
- Regulatory status and standardization
- Cost and accessibility considerations
""",
            TreatmentCategory.PREVENTIVE: """
You are a preventive medicine specialist analyzing preventive interventions. Focus on:
- Primary, secondary, and tertiary prevention strategies
- Risk reduction effectiveness
- Population vs. individual approaches
- Screening and early detection methods
- Vaccination and prophylactic treatments
- Lifestyle and environmental modifications
- Cost-effectiveness of prevention
"""
        }
        
        return prompts.get(category, "Analyze the treatment evidence provided.")

    def _prepare_evidence_context(self, sources: List[EvidenceSource]) -> str:
        """📋 Prepare evidence context for analysis"""
        
        context_parts = []
        
        for i, source in enumerate(sources[:10]):  # Limit to top 10 sources
            quality_info = ""
            if source.metadata and source.metadata.get('quality_metrics'):
                quality_metrics = source.metadata['quality_metrics']
                quality_info = f" [Evidence Level: {quality_metrics.evidence_level.value}]"
            
            context_parts.append(f"""
Source {i+1}: {source.title}{quality_info}
Journal: {source.journal}
Summary: {source.summary[:300]}...
""")
        
        return "\n".join(context_parts)

    async def _extract_specific_treatments(
        self,
        topic: str,
        category: TreatmentCategory,
        sources: List[EvidenceSource]
    ) -> List[TreatmentEvidence]:
        """💊 Extract specific treatment details"""
        
        # This would use NLP or LLM to extract specific treatments
        # For now, return placeholder structure
        treatments = []
        
        # Extract common treatments mentioned in sources
        treatment_mentions = {}
        
        for source in sources:
            content = (source.title + " " + source.summary).lower()
            
            # Simple keyword extraction (would be enhanced with NLP)
            if category == TreatmentCategory.PHARMACOLOGICAL:
                drug_keywords = [
                    "metformin", "insulin", "aspirin", "statin", "ace inhibitor",
                    "beta blocker", "diuretic", "antibiotic", "chemotherapy"
                ]
                for drug in drug_keywords:
                    if drug in content:
                        treatment_mentions[drug] = treatment_mentions.get(drug, 0) + 1
        
        # Convert to TreatmentEvidence objects
        for treatment, mentions in treatment_mentions.items():
            if mentions >= 2:  # Require multiple mentions
                evidence = TreatmentEvidence(
                    treatment_name=treatment,
                    category=category,
                    evidence_strength=EvidenceStrength.MODERATE,
                    efficacy_summary=f"Mentioned in {mentions} sources",
                    safety_profile="Requires detailed analysis"
                )
                treatments.append(evidence)
        
        return treatments

    def _assess_category_evidence_quality(self, sources: List[EvidenceSource]) -> str:
        """📊 Assess evidence quality for category"""
        
        if not sources:
            return "No evidence"
        
        high_quality_count = 0
        total_count = len(sources)
        
        for source in sources:
            if source.metadata and source.metadata.get('quality_metrics'):
                quality_metrics = source.metadata['quality_metrics']
                if quality_metrics.evidence_level.value in ["Systematic reviews of RCTs", "Individual RCTs"]:
                    high_quality_count += 1
        
        quality_percentage = (high_quality_count / total_count) * 100
        
        if quality_percentage >= 50:
            return "High quality evidence"
        elif quality_percentage >= 25:
            return "Moderate quality evidence"
        else:
            return "Limited quality evidence"

    async def _generate_comparative_analysis(
        self,
        topic: str,
        treatment_analysis: Dict[str, Any]
    ) -> str:
        """🔄 Generate comparative effectiveness analysis"""
        
        system_prompt = """
You are a comparative effectiveness researcher analyzing treatment options. 
Provide a comprehensive comparison of different treatment modalities including:
- Head-to-head effectiveness comparisons
- Risk-benefit profiles
- Patient selection criteria for each treatment
- Combination therapy considerations
- Cost-effectiveness comparisons
- Quality of life impacts
"""
        
        # Prepare treatment summary for comparison
        treatment_summary = ""
        for category, analysis in treatment_analysis.items():
            if analysis.get("treatments"):
                treatment_summary += f"\n{category.upper()}:\n"
                treatment_summary += f"Evidence Quality: {analysis.get('evidence_quality', 'Unknown')}\n"
                treatment_summary += f"Summary: {analysis.get('evidence_summary', '')[:200]}...\n"
        
        prompt = f"""
Topic: {topic}

Treatment Analysis Summary:
{treatment_summary}

Provide a comparative effectiveness analysis comparing all available treatment modalities.
"""
        
        async with self.llm_semaphore:
            comparative_analysis = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        return comparative_analysis

    async def _generate_treatment_recommendations(
        self,
        topic: str,
        treatment_analysis: Dict[str, Any],
        comparative_analysis: str
    ) -> str:
        """💡 Generate evidence-based treatment recommendations"""
        
        system_prompt = """
You are a clinical expert providing evidence-based treatment recommendations.
Structure your recommendations as:
1. First-line treatments (strongest evidence)
2. Second-line treatments (alternative options)
3. Adjunctive treatments (supportive therapies)
4. Experimental treatments (emerging evidence)
5. Patient-specific considerations
6. Monitoring and follow-up requirements
"""
        
        prompt = f"""
Topic: {topic}

Based on the treatment analysis and comparative effectiveness data, provide structured, evidence-based treatment recommendations.

Treatment Analysis: {str(treatment_analysis)[:1000]}...
Comparative Analysis: {comparative_analysis[:500]}...
"""
        
        async with self.llm_semaphore:
            recommendations = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        return recommendations

    async def _generate_safety_summary(self, treatment_analysis: Dict[str, Any]) -> str:
        """⚠️ Generate comprehensive safety summary"""
        
        system_prompt = """
You are a drug safety specialist summarizing treatment safety profiles.
Focus on:
- Common and serious adverse effects
- Contraindications and precautions
- Drug interactions (if applicable)
- Monitoring requirements
- Special populations (pregnancy, elderly, renal/hepatic impairment)
- Risk mitigation strategies
"""
        
        safety_context = ""
        for category, analysis in treatment_analysis.items():
            if analysis.get("evidence_summary"):
                safety_context += f"{category}: {analysis['evidence_summary'][:200]}...\n"
        
        prompt = f"""
Generate a comprehensive safety summary based on the treatment evidence:

{safety_context}
"""
        
        async with self.llm_semaphore:
            safety_summary = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        return safety_summary

    def generate_treatment_focused_questions(self, topic: str) -> List[str]:
        """💊 Generate treatment-focused research questions"""
        
        treatment_questions = [
            f"What are the most effective pharmacological treatments for {topic}?",
            f"What are the optimal dosing regimens for {topic} medications?",
            f"What are the most common adverse effects of {topic} treatments?",
            f"What surgical interventions are available for {topic}?",
            f"What are the outcomes of surgical vs. medical management of {topic}?",
            f"What device-based therapies are effective for {topic}?",
            f"What behavioral interventions improve outcomes in {topic}?",
            f"What complementary therapies have evidence for {topic}?",
            f"What are the contraindications for {topic} treatments?",
            f"What drug interactions are important in {topic} management?",
            f"What monitoring is required for {topic} treatments?",
            f"What are the cost-effectiveness considerations for {topic} treatments?",
            f"What preventive strategies are effective for {topic}?",
            f"What are the treatment guidelines for {topic}?",
            f"What combination therapies are effective for {topic}?"
        ]
        
        return treatment_questions 