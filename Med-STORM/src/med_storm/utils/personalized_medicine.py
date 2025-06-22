"""
Utilities for generating personalized medicine recommendations.
"""
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re

from med_storm.models.evidence import EvidenceSource, EvidenceCorpus

class PatientFactorType(Enum):
    """Types of patient factors that can influence treatment."""
    DEMOGRAPHIC = "demographic"
    CLINICAL = "clinical"
    GENETIC = "genetic"
    LIFESTYLE = "lifestyle"
    COMORBIDITY = "comorbidity"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    SOCIAL = "social"

@dataclass
class PatientFactor:
    """A factor related to the patient that may influence treatment."""
    name: str
    value: Any
    factor_type: PatientFactorType
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersonalizedRecommendation:
    """A personalized recommendation based on patient factors."""
    recommendation: str
    confidence: float
    supporting_evidence: List[EvidenceSource]
    relevant_factors: List[PatientFactor]
    metadata: Dict[str, Any] = field(default_factory=dict)

class PersonalizedMedicineEngine:
    """Engine for generating personalized medicine recommendations."""
    
    def __init__(self, llm_provider=None):
        """Initialize with an optional LLM provider for advanced reasoning."""
        self.llm = llm_provider
        # Note: The actual handler methods are implemented with _process_*_factors naming
        # These handlers are registered here but called through generate_recommendations
        # which uses the _process_*_factors methods directly
        self.factor_handlers = {
            PatientFactorType.DEMOGRAPHIC: self._process_demographic_factors,
            PatientFactorType.CLINICAL: self._process_clinical_factors,
            PatientFactorType.GENETIC: self._process_genetic_factors,
            PatientFactorType.LIFESTYLE: self._process_lifestyle_factors,
            PatientFactorType.COMORBIDITY: self._process_comorbidity_factors,
            PatientFactorType.MEDICATION: self._process_medication_factors,
            PatientFactorType.ALLERGY: self._process_allergy_factors,
            PatientFactorType.SOCIAL: self._process_social_factors,
        }
    
    async def generate_recommendations(
        self,
        evidence: EvidenceCorpus,
        patient_factors: List[PatientFactor],
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """
        Generate personalized treatment recommendations based on evidence and patient factors.
        
        Args:
            evidence: Corpus of evidence to use for recommendations
            patient_factors: List of relevant patient factors
            condition: The medical condition being treated
            
        Returns:
            List of personalized recommendations
        """
        if not evidence or not evidence.sources:
            return []
        
        # Group factors by type for easier processing
        factors_by_type = self._group_factors_by_type(patient_factors)
        
        # Generate recommendations based on factor types
        recommendations = []
        
        # 1. Process demographic factors (age, sex, etc.)
        if PatientFactorType.DEMOGRAPHIC in factors_by_type:
            demo_rec = await self._process_demographic_factors(
                factors_by_type[PatientFactorType.DEMOGRAPHIC],
                evidence,
                condition
            )
            if demo_rec:
                recommendations.append(demo_rec)
        
        # 2. Process clinical factors (diagnoses, lab results, etc.)
        if PatientFactorType.CLINICAL in factors_by_type:
            clinical_recs = await self._process_clinical_factors(
                factors_by_type[PatientFactorType.CLINICAL],
                evidence,
                condition
            )
            recommendations.extend(clinical_recs)
        
        # 3. Process genetic factors
        if PatientFactorType.GENETIC in factors_by_type:
            genetic_recs = await self._process_genetic_factors(
                factors_by_type[PatientFactorType.GENETIC],
                evidence,
                condition
            )
            recommendations.extend(genetic_recs)
        
        # 4. Process lifestyle factors
        if PatientFactorType.LIFESTYLE in factors_by_type:
            lifestyle_recs = await self._process_lifestyle_factors(
                factors_by_type[PatientFactorType.LIFESTYLE],
                evidence,
                condition
            )
            recommendations.extend(lifestyle_recs)
        
        # 5. Process comorbidities
        if PatientFactorType.COMORBIDITY in factors_by_type:
            comorbidity_recs = await self._process_comorbidity_factors(
                factors_by_type[PatientFactorType.COMORBIDITY],
                evidence,
                condition
            )
            recommendations.extend(comorbidity_recs)
        
        # 6. Process current medications
        if PatientFactorType.MEDICATION in factors_by_type:
            med_recs = await self._process_medication_factors(
                factors_by_type[PatientFactorType.MEDICATION],
                evidence,
                condition
            )
            recommendations.extend(med_recs)
        
        # 7. Process allergies
        if PatientFactorType.ALLERGY in factors_by_type:
            allergy_recs = await self._process_allergy_factors(
                factors_by_type[PatientFactorType.ALLERGY],
                evidence,
                condition
            )
            recommendations.extend(allergy_recs)
        
        # 8. Process social factors
        if PatientFactorType.SOCIAL in factors_by_type:
            social_recs = await self._process_social_factors(
                factors_by_type[PatientFactorType.SOCIAL],
                evidence,
                condition
            )
            recommendations.extend(social_recs)
        
        # Sort recommendations by confidence (highest first)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations
    
    def _group_factors_by_type(
        self, 
        factors: List[PatientFactor]
    ) -> Dict[PatientFactorType, List[PatientFactor]]:
        """Group patient factors by their type."""
        grouped = {}
        for factor in factors:
            if factor.factor_type not in grouped:
                grouped[factor.factor_type] = []
            grouped[factor.factor_type].append(factor)
        return grouped
    
    async def _process_demographic_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> Optional[PersonalizedRecommendation]:
        """Process demographic factors like age and sex."""
        # Extract common demographic factors
        age = next((f for f in factors if f.name.lower() in ('age', 'patient age')), None)
        sex = next((f for f in factors if f.name.lower() in ('sex', 'gender')), None)
        
        if not age and not sex:
            return None
        
        # Find evidence that matches these demographics
        relevant_sources = []
        
        for source in evidence.sources:
            # Check if source has population data that matches
            population = source.metadata.get('population', {}) if source.metadata else {}
            
            # Check age match if available
            age_match = True
            if age and 'age_range' in population:
                min_age, max_age = population['age_range']
                patient_age = float(age.value) if isinstance(age.value, (int, float)) else None
                
                if patient_age is not None:
                    age_match = min_age <= patient_age <= max_age
            
            # Check sex match if available
            sex_match = True
            if sex and 'sex' in population and sex.value:
                source_sex = population['sex'].lower()
                patient_sex = str(sex.value).lower()
                
                if source_sex != 'both' and patient_sex != source_sex:
                    sex_match = False
            
            if age_match and sex_match:
                relevant_sources.append(source)
        
        if not relevant_sources:
            return None
        
        # Generate recommendation text
        rec_parts = []
        
        if age and sex:
            rec_parts.append(f"For a {age.value}-year-old {sex.value} patient with {condition}:")
        elif age:
            rec_parts.append(f"For a {age.value}-year-old patient with {condition}:")
        elif sex:
            rec_parts.append(f"For a {sex.value} patient with {condition}:")
        
        # Add evidence-based statements
        evidence_statements = []
        
        if age:
            age_studies = [s for s in relevant_sources if 'age_range' in s.metadata.get('population', {})]
            if age_studies:
                age_ranges = [s.metadata['population']['age_range'] for s in age_studies]
                avg_min = sum(r[0] for r in age_ranges) / len(age_ranges)
                avg_max = sum(r[1] for r in age_ranges) / len(age_ranges)
                evidence_statements.append(
                    f"The evidence is based on patients aged {avg_min:.0f}-{avg_max:.0f} years."
                )
        
        if sex and sex.value.lower() in ('male', 'female'):
            sex_studies = [s for s in relevant_sources 
                          if s.metadata.get('population', {}).get('sex', '').lower() in 
                          (sex.value.lower(), 'both')]
            if sex_studies:
                evidence_statements.append(
                    f"The evidence includes {len(sex_studies)} studies with {sex.value} participants."
                )
        
        if evidence_statements:
            rec_parts.append(" ".join(evidence_statements))
        
        # Add a general recommendation based on the evidence
        if relevant_sources:
            rec_parts.append(
                "The following treatment approaches have shown efficacy in similar patient populations:"
            )
            
            # Group by treatment type if available
            treatments = {}
            for src in relevant_sources:
                treatment = src.metadata.get('treatment', 'various treatments')
                if treatment not in treatments:
                    treatments[treatment] = []
                treatments[treatment].append(src)
            
            for treatment, sources in treatments.items():
                rec_parts.append(f"- {treatment} (supported by {len(sources)} studies)")
        
        return PersonalizedRecommendation(
            recommendation="\n".join(rec_parts),
            confidence=0.8,  # High confidence for demographic-based recommendations
            supporting_evidence=relevant_sources,
            relevant_factors=factors
        )
    
    async def _process_clinical_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process clinical factors like lab results and vital signs."""
        # This is a simplified implementation
        # In a real system, you would have more sophisticated logic
        
        recommendations = []
        
        for factor in factors:
            # Skip if this is actually a demographic factor
            if factor.factor_type != PatientFactorType.CLINICAL:
                continue
            
            # Find evidence that mentions this clinical factor
            relevant_sources = []
            factor_name = factor.name.lower()
            
            for source in evidence.sources:
                # Check title, abstract, and metadata for mentions of this factor
                content = f"{getattr(source, 'title', '')} {getattr(source, 'abstract', '')}".lower()
                
                # Simple keyword matching - in a real system, use NLP
                if factor_name in content or any(
                    kw in content for kw in self._get_synonyms(factor_name)
                ):
                    relevant_sources.append(source)
            
            if not relevant_sources:
                continue
            
            # Create a recommendation for this clinical factor
            rec_text = (
                f"For patients with {factor.name} of {factor.value}:\n"
                f"The following evidence is relevant based on this clinical factor. "
                f"{len(relevant_sources)} studies mentioned {factor.name} in relation to {condition}."
            )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.7,  # Moderate confidence for clinical factor matches
                supporting_evidence=relevant_sources,
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    async def _process_genetic_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process genetic factors like mutations and biomarkers."""
        # This is a placeholder implementation
        # In a real system, you would integrate with pharmacogenomics databases
        
        recommendations = []
        
        for factor in factors:
            if factor.factor_type != PatientFactorType.GENETIC:
                continue
                
            # In a real system, check for pharmacogenomic evidence
            # For now, we'll just note the genetic factor
            rec_text = (
                f"Genetic factor detected: {factor.name} = {factor.value}\n"
                "Consider pharmacogenomic testing to guide treatment selection "
                "and dosing based on the patient's genetic profile."
            )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.6,  # Lower confidence without specific evidence
                supporting_evidence=[],  # No supporting evidence in this simple implementation
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    async def _process_lifestyle_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process lifestyle factors like diet, exercise, and smoking."""
        recommendations = []
        
        for factor in factors:
            if factor.factor_type != PatientFactorType.LIFESTYLE:
                continue
                
            # Find evidence related to this lifestyle factor
            relevant_sources = []
            factor_name = factor.name.lower()
            
            for source in evidence.sources:
                content = f"{getattr(source, 'title', '')} {getattr(source, 'abstract', '')}".lower()
                
                if factor_name in content or any(
                    kw in content for kw in self._get_synonyms(factor_name)
                ):
                    relevant_sources.append(source)
            
            # Create recommendation
            action = "consider"  # Default action
            
            # Simple rules for common lifestyle factors
            if "smoking" in factor_name:
                if factor.value and str(factor.value).lower() != "never":
                    action = (
                        "strongly recommend smoking cessation as it may improve treatment outcomes. "
                        "Consider referral to a smoking cessation program."
                    )
                else:
                    action = "encourage continued smoking abstinence."
            elif "exercise" in factor_name or "physical activity" in factor_name:
                if factor.value and (isinstance(factor.value, (int, float)) and factor.value < 150):
                    action = (
                        "recommend increasing physical activity to at least 150 minutes of moderate "
                        "exercise per week, as this may improve treatment outcomes."
                    )
                else:
                    action = "encourage maintaining current physical activity levels."
            elif "diet" in factor_name or "nutrition" in factor_name:
                action = (
                    "consider dietary assessment and counseling. A balanced diet may support "
                    "overall health and treatment efficacy."
                )
            
            rec_text = (
                f"Lifestyle factor: {factor.name} = {factor.value}\n"
                f"Recommendation: {action}"
            )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.65,  # Moderate confidence for lifestyle recommendations
                supporting_evidence=relevant_sources,
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    async def _process_comorbidity_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process comorbid conditions that may affect treatment."""
        recommendations = []
        
        for factor in factors:
            if factor.factor_type != PatientFactorType.COMORBIDITY:
                continue
                
            # Find evidence that mentions both the condition and comorbidity
            relevant_sources = []
            comorbidity = factor.name
            
            for source in evidence.sources:
                content = f"{getattr(source, 'title', '')} {getattr(source, 'abstract', '')}".lower()
                
                if (comorbidity.lower() in content or 
                    any(kw in content for kw in self._get_synonyms(comorbidity.lower()))):
                    relevant_sources.append(source)
            
            # Create recommendation
            rec_text = (
                f"Comorbidity detected: {comorbidity}\n"
                f"This condition may affect treatment selection and management of {condition}. "
            )
            
            if relevant_sources:
                rec_text += (
                    f"{len(relevant_sources)} studies discuss the interaction between "
                    f"{condition} and {comorbidity}."
                )
            else:
                rec_text += (
                    "Consider potential interactions when selecting treatments. "
                    "Consult relevant clinical guidelines for managing these conditions together."
                )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.75 if relevant_sources else 0.5,
                supporting_evidence=relevant_sources,
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    async def _process_medication_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process current medications that may interact with treatments."""
        recommendations = []
        
        for factor in factors:
            if factor.factor_type != PatientFactorType.MEDICATION:
                continue
                
            medication = factor.name
            
            # In a real system, check for drug-drug interactions
            # This is a simplified implementation
            
            rec_text = (
                f"Current medication: {medication} (dose: {factor.value if factor.value else 'unspecified'})\n"
                "Consider potential drug-drug interactions when selecting treatments. "
                "Consult a drug interaction database for specific interactions."
            )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.7,  # High confidence for medication-related recommendations
                supporting_evidence=[],  # Would come from drug interaction database
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    async def _process_allergy_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process medication allergies and intolerances."""
        recommendations = []
        
        for factor in factors:
            if factor.factor_type != PatientFactorType.ALLERGY:
                continue
                
            allergy = factor.name
            reaction = factor.value if factor.value else "unspecified reaction"
            
            rec_text = (
                f"Allergy alert: {allergy} (reaction: {reaction})\n"
                f"Avoid prescribing {allergy} or related medications. "
                "Document the specific reaction in the patient's record."
            )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.9,  # Very high confidence for allergy alerts
                supporting_evidence=[],  # Would come from allergy database
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    async def _process_social_factors(
        self,
        factors: List[PatientFactor],
        evidence: EvidenceCorpus,
        condition: str
    ) -> List[PersonalizedRecommendation]:
        """Process social determinants of health."""
        recommendations = []
        
        for factor in factors:
            if factor.factor_type != PatientFactorType.SOCIAL:
                continue
                
            factor_name = factor.name.lower()
            factor_value = str(factor.value).lower() if factor.value is not None else ""
            
            rec_text = f"Social factor: {factor.name}"
            
            if factor_value:
                rec_text += f" = {factor.value}\n"
            else:
                rec_text += "\n"
            
            # Add specific recommendations based on social factors
            if "income" in factor_name or "socioeconomic" in factor_name:
                rec_text += (
                    "Consider the potential impact of socioeconomic status on treatment adherence. "
                    "Discuss cost concerns and explore patient assistance programs if needed."
                )
            elif "education" in factor_name:
                rec_text += (
                    "Assess health literacy and provide education at an appropriate level. "
                    "Use teach-back methods to confirm understanding."
                )
            elif "housing" in factor_name and ("unstable" in factor_value or "homeless" in factor_value):
                rec_text += (
                    "Housing instability may affect treatment adherence and follow-up. "
                    "Consider connecting with social work or community resources."
                )
            elif "support" in factor_name and ("limited" in factor_value or "none" in factor_value):
                rec_text += (
                    "Limited social support may impact treatment adherence. "
                    "Involve family members or caregivers in the treatment plan when possible."
                )
            else:
                rec_text += (
                    "Consider how this social factor may impact treatment adherence and outcomes. "
                    "Address any barriers to care."
                )
            
            recommendations.append(PersonalizedRecommendation(
                recommendation=rec_text,
                confidence=0.6,  # Moderate confidence for social factor recommendations
                supporting_evidence=[],
                relevant_factors=[factor]
            ))
        
        return recommendations
    
    def _get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term (simplified)."""
        # In a real system, use a medical thesaurus or ontology
        synonym_map = {
            "htn": ["hypertension", "high blood pressure"],
            "dm": ["diabetes", "diabetes mellitus"],
            "mi": ["heart attack", "myocardial infarction"],
            "chf": ["heart failure", "congestive heart failure"],
            "copd": ["chronic obstructive pulmonary disease", "chronic bronchitis", "emphysema"],
            "gerd": ["gastroesophageal reflux", "acid reflux", "heartburn"],
            "cad": ["coronary artery disease", "coronary heart disease", "ischemic heart disease"],
            "pvd": ["peripheral vascular disease", "peripheral artery disease"],
            "ckd": ["chronic kidney disease", "renal insufficiency"],
            "cva": ["stroke", "cerebrovascular accident"],
            "tia": ["transient ischemic attack", "mini-stroke"],
            "pna": ["pneumonia"],
            "uti": ["urinary tract infection"],
            "bph": ["benign prostatic hyperplasia", "enlarged prostate"],
            "dvt": ["deep vein thrombosis", "blood clot"],
            "pe": ["pulmonary embolism", "lung clot"],
            "afib": ["atrial fibrillation", "a-fib"],
            "chf": ["congestive heart failure", "heart failure"],
            "copd": ["chronic obstructive pulmonary disease"],
            "mi": ["myocardial infarction", "heart attack"],
            "pud": ["peptic ulcer disease", "stomach ulcer"],
            "ibd": ["inflammatory bowel disease", "crohn's disease", "ulcerative colitis"],
            "ibs": ["irritable bowel syndrome"],
            "pud": ["peptic ulcer disease"],
            "gerd": ["gastroesophageal reflux disease", "acid reflux"],
            "esrd": ["end stage renal disease", "kidney failure"],
            "aki": ["acute kidney injury", "acute renal failure"],
            "bph": ["benign prostatic hyperplasia", "enlarged prostate"],
            "uti": ["urinary tract infection"],
            "bph": ["benign prostatic hyperplasia"],
            "std": ["sexually transmitted disease", "sexually transmitted infection", "sti"],
            "hiv": ["human immunodeficiency virus", "aids"],
            "aids": ["acquired immunodeficiency syndrome", "hiv"],
            "hiv": ["human immunodeficiency virus", "aids"],
            "hiv": ["human immunodeficiency virus", "aids"],
            "hiv": ["human immunodeficiency virus", "aids"],
            "hiv": ["human immunodeficiency virus", "aids"],
        }
        
        return synonym_map.get(term.lower(), [])
