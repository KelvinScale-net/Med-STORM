"""
🧠 MED-PRM INSPIRED PROCESS REWARD MODEL
=======================================

CRITICAL MISSING COMPONENT FROM MED-PRM:
- Step-wise reasoning verification
- Medical knowledge validation
- Process reward scoring
- Guideline-based verification

This addresses the major gap in our current implementation.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from med_storm.llm.base import LLMProvider
from med_storm.models.evidence import EvidenceSource, EvidenceTier

logger = logging.getLogger(__name__)

class ReasoningStepType(Enum):
    """🔍 Types of medical reasoning steps"""
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    MECHANISM = "mechanism"
    PROGNOSIS = "prognosis"
    RISK_ASSESSMENT = "risk_assessment"
    CONTRAINDICATION = "contraindication"
    DRUG_INTERACTION = "drug_interaction"
    DOSAGE = "dosage"
    MONITORING = "monitoring"
    EVIDENCE_CITATION = "evidence_citation"

class RewardScore(Enum):
    """🎯 Process reward scores"""
    EXCELLENT = 1.0      # Perfect reasoning step
    GOOD = 0.8          # Solid reasoning with minor issues
    ACCEPTABLE = 0.6    # Adequate but could be improved
    POOR = 0.4          # Significant issues
    INCORRECT = 0.0     # Fundamentally wrong

@dataclass
class ReasoningStep:
    """🧠 Individual reasoning step in medical analysis"""
    step_id: str
    content: str
    step_type: ReasoningStepType
    evidence_cited: List[str]
    confidence: float
    
@dataclass
class ProcessReward:
    """🏆 Reward for a reasoning step"""
    step_id: str
    score: RewardScore
    explanation: str
    evidence_quality: float
    guideline_compliance: float
    medical_accuracy: float
    
class MedicalProcessRewardModel:
    """🧠 Med-PRM inspired process reward model for medical reasoning verification"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.llm_semaphore = asyncio.Semaphore(10)
        
        # Medical knowledge validation prompts
        self.validation_prompts = {
            ReasoningStepType.DIAGNOSIS: """
Evaluate this diagnostic reasoning step for medical accuracy:

Step: {step_content}
Evidence: {evidence}

Assess:
1. Diagnostic accuracy and clinical validity
2. Proper use of clinical criteria and guidelines
3. Consideration of differential diagnosis
4. Evidence quality and relevance
5. Logical reasoning flow

Score: 0.0 (incorrect) to 1.0 (excellent)
""",
            ReasoningStepType.TREATMENT: """
Evaluate this treatment recommendation for medical accuracy:

Step: {step_content}
Evidence: {evidence}

Assess:
1. Treatment appropriateness and safety
2. Guideline compliance (AHA, ESC, ADA, etc.)
3. Risk-benefit analysis
4. Contraindication consideration
5. Evidence-based support

Score: 0.0 (incorrect) to 1.0 (excellent)
""",
            ReasoningStepType.DRUG_INTERACTION: """
Evaluate this drug interaction analysis:

Step: {step_content}
Evidence: {evidence}

Assess:
1. Interaction mechanism accuracy
2. Clinical significance assessment
3. Risk stratification
4. Management recommendations
5. Evidence quality

Score: 0.0 (incorrect) to 1.0 (excellent)
""",
            ReasoningStepType.DOSAGE: """
Evaluate this dosage recommendation:

Step: {step_content}
Evidence: {evidence}

Assess:
1. Dosage appropriateness for indication
2. Patient-specific factors consideration
3. Renal/hepatic adjustment if needed
4. Safety margins
5. Guideline compliance

Score: 0.0 (incorrect) to 1.0 (excellent)
"""
        }

    async def analyze_reasoning_chain(
        self, 
        text: str, 
        evidence_pool: List[EvidenceSource]
    ) -> List[ReasoningStep]:
        """🔍 Break down medical text into individual reasoning steps"""
        
        logger.info("🔍 Analyzing medical reasoning chain")
        
        system_prompt = """
You are a medical AI that analyzes medical reasoning and breaks it down into individual steps.

Given medical text, identify distinct reasoning steps and classify them by type:
- diagnosis: Diagnostic conclusions
- treatment: Treatment recommendations  
- mechanism: Mechanism explanations
- prognosis: Prognostic assessments
- risk_assessment: Risk evaluations
- contraindication: Contraindication identification
- drug_interaction: Drug interaction analysis
- dosage: Dosage recommendations
- monitoring: Monitoring recommendations
- evidence_citation: Evidence citations

For each step, extract:
1. The reasoning content
2. The step type
3. Any evidence cited
4. Confidence level (0.0-1.0)

Format as JSON array:
[
    {
        "step_id": "step_1",
        "content": "The reasoning step content...",
        "step_type": "diagnosis",
        "evidence_cited": ["source1", "source2"],
        "confidence": 0.9
    }
]
"""
        
        prompt = f"Analyze this medical reasoning:\n\n{text}"
        
        async with self.llm_semaphore:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        try:
            import json
            steps_data = json.loads(response)
            
            reasoning_steps = []
            for i, step_data in enumerate(steps_data):
                try:
                    step_type = ReasoningStepType(step_data.get('step_type', 'diagnosis'))
                except ValueError:
                    step_type = ReasoningStepType.DIAGNOSIS
                
                step = ReasoningStep(
                    step_id=step_data.get('step_id', f'step_{i+1}'),
                    content=step_data.get('content', ''),
                    step_type=step_type,
                    evidence_cited=step_data.get('evidence_cited', []),
                    confidence=float(step_data.get('confidence', 0.5))
                )
                reasoning_steps.append(step)
            
            logger.info(f"✅ Identified {len(reasoning_steps)} reasoning steps")
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Failed to parse reasoning steps: {e}")
            # Fallback: create single step
            return [ReasoningStep(
                step_id="step_1",
                content=text,
                step_type=ReasoningStepType.DIAGNOSIS,
                evidence_cited=[],
                confidence=0.5
            )]

    async def verify_reasoning_step(
        self,
        step: ReasoningStep,
        evidence_pool: List[EvidenceSource]
    ) -> ProcessReward:
        """🏆 Verify a single reasoning step against medical knowledge"""
        
        # Find relevant evidence for this step
        relevant_evidence = self._find_relevant_evidence(step, evidence_pool)
        evidence_text = "\n".join([f"- {ev.summary[:200]}..." for ev in relevant_evidence[:3]])
        
        # Get validation prompt for step type
        validation_prompt = self.validation_prompts.get(
            step.step_type,
            self.validation_prompts[ReasoningStepType.DIAGNOSIS]
        )
        
        system_prompt = """
You are a medical expert evaluating the accuracy and quality of medical reasoning steps.

Provide a detailed evaluation with:
1. Overall score (0.0-1.0)
2. Evidence quality assessment (0.0-1.0)
3. Guideline compliance (0.0-1.0)
4. Medical accuracy (0.0-1.0)
5. Detailed explanation

Format as JSON:
{
    "overall_score": 0.85,
    "evidence_quality": 0.9,
    "guideline_compliance": 0.8,
    "medical_accuracy": 0.85,
    "explanation": "Detailed explanation of the evaluation..."
}
"""
        
        prompt = validation_prompt.format(
            step_content=step.content,
            evidence=evidence_text if evidence_text else "No specific evidence provided"
        )
        
        async with self.llm_semaphore:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        try:
            import json
            eval_data = json.loads(response)
            
            # Convert score to RewardScore enum
            overall_score = float(eval_data.get('overall_score', 0.5))
            if overall_score >= 0.9:
                reward_score = RewardScore.EXCELLENT
            elif overall_score >= 0.75:
                reward_score = RewardScore.GOOD
            elif overall_score >= 0.6:
                reward_score = RewardScore.ACCEPTABLE
            elif overall_score >= 0.3:
                reward_score = RewardScore.POOR
            else:
                reward_score = RewardScore.INCORRECT
            
            return ProcessReward(
                step_id=step.step_id,
                score=reward_score,
                explanation=eval_data.get('explanation', 'No explanation provided'),
                evidence_quality=float(eval_data.get('evidence_quality', 0.5)),
                guideline_compliance=float(eval_data.get('guideline_compliance', 0.5)),
                medical_accuracy=float(eval_data.get('medical_accuracy', 0.5))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse evaluation: {e}")
            return ProcessReward(
                step_id=step.step_id,
                score=RewardScore.ACCEPTABLE,
                explanation="Evaluation failed - defaulting to acceptable score",
                evidence_quality=0.5,
                guideline_compliance=0.5,
                medical_accuracy=0.5
            )

    def _find_relevant_evidence(
        self, 
        step: ReasoningStep, 
        evidence_pool: List[EvidenceSource]
    ) -> List[EvidenceSource]:
        """🔍 Find evidence relevant to a reasoning step"""
        
        # Simple relevance scoring based on content overlap
        relevant_evidence = []
        step_words = set(step.content.lower().split())
        
        for evidence in evidence_pool:
            evidence_words = set(evidence.summary.lower().split())
            overlap = len(step_words.intersection(evidence_words))
            
            if overlap > 2:  # Minimum overlap threshold
                relevant_evidence.append(evidence)
        
        # Sort by confidence score (higher quality first)
        relevant_evidence.sort(key=lambda x: x.confidence_score or 0.0, reverse=True)
        
        return relevant_evidence

    async def verify_full_reasoning_chain(
        self,
        text: str,
        evidence_pool: List[EvidenceSource]
    ) -> Tuple[List[ReasoningStep], List[ProcessReward], float]:
        """🧠 Complete Med-PRM style verification of medical reasoning"""
        
        logger.info("🧠 Starting full reasoning chain verification")
        
        # Step 1: Analyze reasoning chain
        reasoning_steps = await self.analyze_reasoning_chain(text, evidence_pool)
        
        # Step 2: Verify each step in parallel
        verification_tasks = []
        for step in reasoning_steps:
            task = self.verify_reasoning_step(step, evidence_pool)
            verification_tasks.append(task)
        
        process_rewards = await asyncio.gather(*verification_tasks)
        
        # Step 3: Calculate overall quality score
        if process_rewards:
            overall_score = sum(reward.score.value for reward in process_rewards) / len(process_rewards)
        else:
            overall_score = 0.5
        
        logger.info(f"✅ Reasoning verification complete. Overall score: {overall_score:.2f}")
        
        return reasoning_steps, process_rewards, overall_score

    def generate_verification_report(
        self,
        reasoning_steps: List[ReasoningStep],
        process_rewards: List[ProcessReward],
        overall_score: float
    ) -> str:
        """📋 Generate detailed verification report"""
        
        report = "## 🧠 Medical Reasoning Verification Report\n\n"
        report += f"**Overall Quality Score**: {overall_score:.2f}/1.0\n\n"
        
        # Score interpretation
        if overall_score >= 0.9:
            interpretation = "🟢 **EXCELLENT** - High-quality medical reasoning"
        elif overall_score >= 0.75:
            interpretation = "🟡 **GOOD** - Solid reasoning with minor areas for improvement"
        elif overall_score >= 0.6:
            interpretation = "🟠 **ACCEPTABLE** - Adequate but could be strengthened"
        elif overall_score >= 0.3:
            interpretation = "🔴 **POOR** - Significant issues requiring attention"
        else:
            interpretation = "⚫ **CRITICAL** - Major problems with reasoning"
        
        report += f"**Quality Assessment**: {interpretation}\n\n"
        
        # Detailed step analysis
        report += "### 📊 Step-by-Step Analysis\n\n"
        
        for step, reward in zip(reasoning_steps, process_rewards):
            report += f"#### Step {step.step_id}: {step.step_type.value.title()}\n"
            report += f"**Content**: {step.content[:150]}...\n"
            report += f"**Score**: {reward.score.value:.2f} ({reward.score.name})\n"
            report += f"**Evidence Quality**: {reward.evidence_quality:.2f}\n"
            report += f"**Guideline Compliance**: {reward.guideline_compliance:.2f}\n"
            report += f"**Medical Accuracy**: {reward.medical_accuracy:.2f}\n"
            report += f"**Evaluation**: {reward.explanation}\n\n"
        
        # Summary statistics
        report += "### 📈 Summary Statistics\n\n"
        if process_rewards:
            avg_evidence = sum(r.evidence_quality for r in process_rewards) / len(process_rewards)
            avg_compliance = sum(r.guideline_compliance for r in process_rewards) / len(process_rewards)
            avg_accuracy = sum(r.medical_accuracy for r in process_rewards) / len(process_rewards)
            
            report += f"- **Average Evidence Quality**: {avg_evidence:.2f}\n"
            report += f"- **Average Guideline Compliance**: {avg_compliance:.2f}\n"
            report += f"- **Average Medical Accuracy**: {avg_accuracy:.2f}\n"
            report += f"- **Total Reasoning Steps**: {len(reasoning_steps)}\n"
        
        return report

    async def improve_reasoning_step(
        self,
        step: ReasoningStep,
        reward: ProcessReward,
        evidence_pool: List[EvidenceSource]
    ) -> str:
        """🔧 Suggest improvements for low-scoring reasoning steps"""
        
        if reward.score.value >= 0.8:
            return step.content  # Already good, no changes needed
        
        relevant_evidence = self._find_relevant_evidence(step, evidence_pool)
        evidence_text = "\n".join([f"- {ev.summary}" for ev in relevant_evidence[:2]])
        
        system_prompt = """
You are a medical expert helping to improve medical reasoning.

Given a reasoning step that received a low quality score, provide an improved version that:
1. Addresses the identified issues
2. Uses stronger evidence
3. Follows medical guidelines more closely
4. Improves medical accuracy
5. Maintains clear, professional language

Return ONLY the improved reasoning step content.
"""
        
        prompt = f"""
Original step: {step.content}

Issues identified: {reward.explanation}

Quality scores:
- Evidence Quality: {reward.evidence_quality:.2f}
- Guideline Compliance: {reward.guideline_compliance:.2f}
- Medical Accuracy: {reward.medical_accuracy:.2f}

Relevant evidence:
{evidence_text}

Provide an improved version:
"""
        
        async with self.llm_semaphore:
            improved_content = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        return improved_content.strip() 