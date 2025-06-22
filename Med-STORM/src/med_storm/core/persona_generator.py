"""
🎭 STORM-INSPIRED MEDICAL PERSONA GENERATOR
==========================================

CRITICAL MISSING COMPONENT FROM ORIGINAL STORM:
- Multi-perspective expert generation
- Conversational question asking
- Domain-specific medical personas
- Perspective-guided research

This addresses the major gap in our current implementation.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from med_storm.llm.base import LLMProvider

logger = logging.getLogger(__name__)

class MedicalExpertise(Enum):
    """🏥 Medical expertise domains"""
    CARDIOLOGIST = "cardiologist"
    ENDOCRINOLOGIST = "endocrinologist"
    NEUROLOGIST = "neurologist"
    ONCOLOGIST = "oncologist"
    SURGEON = "surgeon"
    PHARMACOLOGIST = "pharmacologist"
    EPIDEMIOLOGIST = "epidemiologist"
    RESEARCHER = "medical_researcher"
    PATIENT_ADVOCATE = "patient_advocate"
    NURSE_PRACTITIONER = "nurse_practitioner"
    RADIOLOGIST = "radiologist"
    PATHOLOGIST = "pathologist"

@dataclass
class MedicalPersona:
    """🎭 Medical expert persona for perspective-guided research"""
    name: str
    expertise: MedicalExpertise
    background: str
    perspective: str
    question_style: str
    focus_areas: List[str]
    
class MedicalPersonaGenerator:
    """🎭 STORM-inspired medical persona generator"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.llm_semaphore = asyncio.Semaphore(5)
        
        # Pre-defined persona templates for medical domains
        self.persona_templates = {
            MedicalExpertise.CARDIOLOGIST: {
                "background": "Board-certified cardiologist with 15+ years experience",
                "perspective": "Focuses on cardiovascular implications and heart-related outcomes",
                "question_style": "Evidence-based, protocol-driven questions",
                "focus_areas": ["cardiac function", "hemodynamics", "cardiovascular risk", "interventional procedures"]
            },
            MedicalExpertise.ENDOCRINOLOGIST: {
                "background": "Endocrinology specialist with expertise in metabolic disorders",
                "perspective": "Emphasizes hormonal, metabolic, and diabetes-related aspects",
                "question_style": "Systematic, guideline-focused inquiries",
                "focus_areas": ["diabetes management", "hormonal balance", "metabolic syndrome", "endocrine disorders"]
            },
            MedicalExpertise.PHARMACOLOGIST: {
                "background": "Clinical pharmacologist specializing in drug therapy optimization",
                "perspective": "Focuses on drug mechanisms, interactions, and therapeutic effectiveness",
                "question_style": "Mechanism-based, safety-focused questions",
                "focus_areas": ["drug interactions", "pharmacokinetics", "adverse effects", "dosing optimization"]
            },
            MedicalExpertise.RESEARCHER: {
                "background": "Medical researcher with expertise in evidence-based medicine",
                "perspective": "Emphasizes research methodology, evidence quality, and statistical significance",
                "question_style": "Methodologically rigorous, evidence-hierarchy focused",
                "focus_areas": ["study design", "statistical analysis", "evidence quality", "research gaps"]
            },
            MedicalExpertise.PATIENT_ADVOCATE: {
                "background": "Patient advocate with experience in healthcare accessibility and outcomes",
                "perspective": "Focuses on patient experience, quality of life, and practical considerations",
                "question_style": "Patient-centered, outcome-focused questions",
                "focus_areas": ["quality of life", "accessibility", "patient experience", "practical outcomes"]
            }
        }

    async def generate_personas_for_topic(
        self, 
        topic: str, 
        max_personas: int = 4
    ) -> List[MedicalPersona]:
        """🎭 Generate relevant medical personas for the topic"""
        
        logger.info(f"🎭 Generating medical personas for: {topic}")
        
        # Step 1: Determine relevant medical specialties
        relevant_specialties = await self._identify_relevant_specialties(topic)
        
        # Step 2: Generate personas for each specialty
        persona_tasks = []
        for specialty in relevant_specialties[:max_personas]:
            task = self._generate_persona_for_specialty(topic, specialty)
            persona_tasks.append(task)
        
        personas = await asyncio.gather(*persona_tasks)
        
        # Step 3: Always include a medical researcher for evidence quality
        if MedicalExpertise.RESEARCHER not in [p.expertise for p in personas]:
            researcher_persona = await self._generate_persona_for_specialty(topic, MedicalExpertise.RESEARCHER)
            personas.append(researcher_persona)
        
        logger.info(f"✅ Generated {len(personas)} medical personas")
        return personas

    async def _identify_relevant_specialties(self, topic: str) -> List[MedicalExpertise]:
        """🔍 Identify which medical specialties are most relevant to the topic"""
        
        system_prompt = """
You are a medical AI assistant that identifies which medical specialties are most relevant to a given topic.

Given a medical topic, identify the 3-4 most relevant medical specialties from this list:
- cardiologist (heart and cardiovascular)
- endocrinologist (hormones, diabetes, metabolism)  
- neurologist (brain, nervous system)
- oncologist (cancer)
- surgeon (surgical interventions)
- pharmacologist (medications, drug therapy)
- epidemiologist (population health, disease patterns)
- medical_researcher (evidence, studies, research)
- patient_advocate (patient experience, outcomes)
- nurse_practitioner (clinical care, patient management)
- radiologist (imaging, diagnostics)
- pathologist (disease mechanisms, laboratory)

Return ONLY the specialty names, one per line, in order of relevance.
"""
        
        prompt = f"Medical topic: {topic}\n\nIdentify the most relevant medical specialties:"
        
        async with self.llm_semaphore:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        # Parse response and map to enums
        specialty_names = [line.strip().lower() for line in response.split('\n') if line.strip()]
        specialties = []
        
        for name in specialty_names:
            try:
                specialty = MedicalExpertise(name)
                specialties.append(specialty)
            except ValueError:
                # Try to match partial names
                for expertise in MedicalExpertise:
                    if name in expertise.value or expertise.value in name:
                        specialties.append(expertise)
                        break
        
        # Ensure we have at least some specialties
        if not specialties:
            specialties = [
                MedicalExpertise.RESEARCHER,
                MedicalExpertise.PHARMACOLOGIST,
                MedicalExpertise.PATIENT_ADVOCATE
            ]
        
        return specialties[:4]  # Limit to 4 max

    async def _generate_persona_for_specialty(
        self, 
        topic: str, 
        specialty: MedicalExpertise
    ) -> MedicalPersona:
        """🎭 Generate a detailed persona for a specific medical specialty"""
        
        template = self.persona_templates.get(specialty, {})
        
        system_prompt = f"""
You are creating a detailed medical expert persona for research on: {topic}

Specialty: {specialty.value}
Base Background: {template.get('background', 'Medical specialist')}
Base Perspective: {template.get('perspective', 'Clinical focus')}

Create a detailed persona with:
1. A specific name (realistic, professional)
2. Detailed background (experience, credentials, specializations)
3. Unique perspective on the topic
4. Question asking style
5. Specific focus areas for this topic

Format as JSON:
{{
    "name": "Dr. [Name]",
    "background": "Detailed professional background...",
    "perspective": "How this expert views the topic...",
    "question_style": "How they ask questions...",
    "focus_areas": ["area1", "area2", "area3"]
}}
"""
        
        prompt = f"Generate a medical expert persona for {specialty.value} researching: {topic}"
        
        async with self.llm_semaphore:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        try:
            import json
            persona_data = json.loads(response)
            
            return MedicalPersona(
                name=persona_data.get("name", f"Dr. {specialty.value.title()}"),
                expertise=specialty,
                background=persona_data.get("background", template.get("background", "")),
                perspective=persona_data.get("perspective", template.get("perspective", "")),
                question_style=persona_data.get("question_style", template.get("question_style", "")),
                focus_areas=persona_data.get("focus_areas", template.get("focus_areas", []))
            )
        except:
            # Fallback to template if JSON parsing fails
            return MedicalPersona(
                name=f"Dr. {specialty.value.replace('_', ' ').title()}",
                expertise=specialty,
                background=template.get("background", f"Expert in {specialty.value}"),
                perspective=template.get("perspective", f"Specialist perspective on {topic}"),
                question_style=template.get("question_style", "Professional medical inquiries"),
                focus_areas=template.get("focus_areas", [topic])
            )

    async def generate_persona_questions(
        self,
        persona: MedicalPersona,
        topic: str,
        existing_questions: List[str] = None,
        max_questions: int = 5
    ) -> List[str]:
        """❓ Generate questions from a specific persona's perspective"""
        
        existing_context = ""
        if existing_questions:
            existing_context = f"\nExisting questions already asked:\n{chr(10).join(existing_questions)}\n"
        
        system_prompt = f"""
You are {persona.name}, a {persona.expertise.value}.

Background: {persona.background}
Perspective: {persona.perspective}
Question Style: {persona.question_style}
Focus Areas: {', '.join(persona.focus_areas)}

Generate {max_questions} specific, actionable research questions about "{topic}" from your unique perspective.

Requirements:
- Questions should reflect your specialty and perspective
- Avoid duplicating existing questions
- Focus on your areas of expertise
- Questions should be answerable through research
- Use your characteristic question style

{existing_context}

Return ONLY the questions, one per line, numbered 1-{max_questions}.
"""
        
        prompt = f"Generate research questions about: {topic}"
        
        async with self.llm_semaphore:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
        
        # Parse questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                question = line.split('.', 1)[-1].strip()
                question = question.lstrip('- •').strip()
                if question:
                    questions.append(question)
        
        return questions[:max_questions]

    async def simulate_expert_conversation(
        self,
        personas: List[MedicalPersona],
        topic: str,
        max_turns: int = 3
    ) -> List[str]:
        """🗣️ STORM-style conversational question generation"""
        
        logger.info(f"🗣️ Simulating expert conversation with {len(personas)} personas")
        
        all_questions = []
        conversation_history = []
        
        for turn in range(max_turns):
            logger.info(f"🔄 Conversation turn {turn + 1}/{max_turns}")
            
            # Each persona asks questions based on conversation so far
            turn_questions = []
            
            for persona in personas:
                # Generate questions considering previous conversation
                questions = await self.generate_persona_questions(
                    persona=persona,
                    topic=topic,
                    existing_questions=all_questions,
                    max_questions=2  # Fewer per turn to encourage interaction
                )
                
                turn_questions.extend(questions)
                
                # Add to conversation history
                conversation_history.append(f"{persona.name} ({persona.expertise.value}): {questions[0] if questions else 'No new questions'}")
            
            all_questions.extend(turn_questions)
            
            # Stop if no new questions generated
            if not turn_questions:
                break
        
        logger.info(f"✅ Generated {len(all_questions)} questions through expert conversation")
        return all_questions

    def get_persona_summary(self, personas: List[MedicalPersona]) -> str:
        """📋 Generate summary of expert perspectives"""
        
        summary = "## 🎭 Expert Perspectives Consulted\n\n"
        
        for persona in personas:
            summary += f"### {persona.name}\n"
            summary += f"**Specialty**: {persona.expertise.value.replace('_', ' ').title()}\n"
            summary += f"**Background**: {persona.background}\n"
            summary += f"**Perspective**: {persona.perspective}\n"
            summary += f"**Focus Areas**: {', '.join(persona.focus_areas)}\n\n"
        
        return summary 