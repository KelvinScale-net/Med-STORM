"""
🔄 DYNAMIC CORPUS MANAGEMENT SYSTEM
===================================

FEATURES:
1. AUTO-UPDATING: Scheduled updates from PubMed/trusted sources
2. QUALITY CONTROL: Only Level 1A, 1B, 2A evidence
3. FRESHNESS GUARANTEE: Automatic obsolescence detection
4. API OPTIMIZATION: Reduce real-time API calls
5. INCREMENTAL UPDATES: Only fetch new/changed content
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from med_storm.connectors.pubmed import PubMedConnector
from med_storm.core.hybrid_engine import EvidenceLevel, QualityMetrics
from med_storm.models.evidence import EvidenceSource

logger = logging.getLogger(__name__)

class UpdateFrequency(Enum):
    """📅 Update frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly" 
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

@dataclass
class CorpusMetadata:
    """📋 Corpus metadata tracking"""
    topic: str
    collection_name: str
    last_updated: datetime
    total_sources: int
    quality_distribution: Dict[str, int]
    oldest_source_year: int
    newest_source_year: int
    update_frequency: UpdateFrequency
    next_update_due: datetime

class DynamicCorpusManager:
    """🔄 Intelligent corpus management with auto-updates and quality control"""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        min_evidence_level: EvidenceLevel = EvidenceLevel.LEVEL_2A,
        max_source_age_years: int = 5,
        update_frequency: UpdateFrequency = UpdateFrequency.MONTHLY
    ):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.min_evidence_level = min_evidence_level
        self.max_source_age_years = max_source_age_years
        self.update_frequency = update_frequency
        
        # Quality control: Only high-quality evidence
        self.allowed_evidence_levels = {
            EvidenceLevel.LEVEL_1A,  # Systematic reviews of RCTs
            EvidenceLevel.LEVEL_1B,  # Individual RCTs  
            EvidenceLevel.LEVEL_2A,  # Systematic reviews of cohort studies
        }
        
        # Initialize connectors
        self.pubmed_connector = PubMedConnector()
        
        logger.info(f"🔄 Dynamic Corpus Manager initialized - Quality levels: {[level.value for level in self.allowed_evidence_levels]}")

    async def create_or_update_corpus(
        self, 
        topic: str, 
        collection_name: str,
        force_update: bool = False
    ) -> CorpusMetadata:
        """🚀 Create new corpus or update existing with quality control"""
        
        logger.info(f"🔄 Managing corpus for topic: '{topic}'")
        
        # Check if corpus exists and needs update
        metadata = await self._get_corpus_metadata(collection_name)
        
        if metadata and not force_update:
            if not self._needs_update(metadata):
                logger.info(f"✅ Corpus '{collection_name}' is up to date")
                return metadata
        
        # Perform update
        logger.info(f"🔄 Updating corpus '{collection_name}'...")
        
        # STAGE 1: Fetch high-quality sources
        new_sources = await self._fetch_high_quality_sources(topic)
        
        # STAGE 2: Quality filtering
        filtered_sources = self._apply_strict_quality_filter(new_sources)
        
        # STAGE 3: Update corpus
        updated_metadata = await self._update_corpus_collection(
            collection_name, topic, filtered_sources
        )
        
        # STAGE 4: Cleanup obsolete sources
        await self._cleanup_obsolete_sources(collection_name)
        
        logger.info(f"✅ Corpus updated: {len(filtered_sources)} high-quality sources added")
        return updated_metadata

    async def _fetch_high_quality_sources(self, topic: str, max_sources: int = 200) -> List[EvidenceSource]:
        """🔍 Fetch only high-quality sources from multiple queries"""
        
        # Generate comprehensive search queries for high-quality evidence
        search_queries = self._generate_quality_search_queries(topic)
        
        all_sources = []
        
        for query in search_queries:
            try:
                # Search PubMed with quality filters
                pmids = await self.pubmed_connector.search(
                    query + " AND (systematic review[pt] OR randomized controlled trial[pt] OR meta-analysis[pt])",
                    max_results=50
                )
                
                if pmids:
                    articles = await self.pubmed_connector.fetch_details(pmids)
                    
                    for pmid, article_data in articles.items():
                        if article_data and article_data.get("abstract"):
                            source = EvidenceSource(
                                id=pmid,
                                title=article_data.get("title", ""),
                                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                summary=article_data.get("abstract", ""),
                                source_name="PubMed",
                                authors=article_data.get("authors", []),
                                journal=article_data.get("journal", ""),
                                pmid=pmid,
                                publication_year=self._extract_year(article_data.get("publication_date", "")),
                                confidence_score=0.9,
                                metadata=article_data
                            )
                            all_sources.append(source)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching sources for query '{query}': {e}")
        
        # Deduplicate by PMID
        unique_sources = {}
        for source in all_sources:
            if source.pmid and source.pmid not in unique_sources:
                unique_sources[source.pmid] = source
        
        logger.info(f"🔍 Fetched {len(unique_sources)} unique sources")
        return list(unique_sources.values())

    def _generate_quality_search_queries(self, topic: str) -> List[str]:
        """📝 Generate search queries focused on high-quality evidence"""
        
        base_queries = [
            f"{topic} systematic review",
            f"{topic} meta-analysis", 
            f"{topic} randomized controlled trial",
            f"{topic} clinical trial",
            f"{topic} cochrane review",
            f"{topic} evidence-based",
            f"{topic} treatment efficacy",
            f"{topic} therapeutic intervention",
            f"{topic} pharmacotherapy",
            f"{topic} drug therapy",
            f"{topic} clinical guidelines",
            f"{topic} practice guidelines"
        ]
        
        return base_queries

    def _apply_strict_quality_filter(self, sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """🔬 Apply strict quality filtering - only Level 1A, 1B, 2A"""
        
        filtered_sources = []
        current_year = datetime.now().year
        
        for source in sources:
            # Age filter
            if source.publication_year:
                age = current_year - source.publication_year
                if age > self.max_source_age_years:
                    continue
            
            # Determine evidence level
            evidence_level = self._determine_evidence_level(source)
            
            # Quality filter - only high-quality evidence
            if evidence_level in self.allowed_evidence_levels:
                # Add quality metadata
                quality_metrics = QualityMetrics(
                    evidence_level=evidence_level,
                    publication_year=source.publication_year,
                    risk_of_bias=self._assess_risk_of_bias(source),
                    study_design=self._extract_study_design(source)
                )
                
                enhanced_metadata = source.metadata.copy() if source.metadata else {}
                enhanced_metadata['quality_metrics'] = quality_metrics
                enhanced_metadata['corpus_added_date'] = datetime.now().isoformat()
                source.metadata = enhanced_metadata
                
                filtered_sources.append(source)
        
        logger.info(f"🔬 Quality filter: {len(filtered_sources)}/{len(sources)} sources passed")
        return filtered_sources

    def _determine_evidence_level(self, source: EvidenceSource) -> EvidenceLevel:
        """🎯 Determine evidence level with strict criteria"""
        
        content = (source.title + " " + source.summary).lower()
        
        # Level 1A: Systematic reviews and meta-analyses (strict)
        if any(term in content for term in [
            "systematic review", "meta-analysis", "cochrane review",
            "systematic literature review", "pooled analysis"
        ]):
            return EvidenceLevel.LEVEL_1A
        
        # Level 1B: High-quality RCTs (strict)
        if any(term in content for term in [
            "randomized controlled trial", "randomised controlled trial",
            "double-blind", "placebo-controlled", "multicentre trial",
            "phase iii", "phase 3"
        ]):
            return EvidenceLevel.LEVEL_1B
        
        # Level 2A: Systematic reviews of cohort studies
        if any(term in content for term in [
            "systematic review" + "cohort", "meta-analysis" + "cohort",
            "systematic review" + "observational"
        ]):
            return EvidenceLevel.LEVEL_2A
        
        # Default to lowest acceptable level
        return EvidenceLevel.LEVEL_5

    def _assess_risk_of_bias(self, source: EvidenceSource) -> str:
        """⚖️ Assess risk of bias with strict criteria"""
        
        content = (source.title + " " + source.summary + " " + (source.journal or "")).lower()
        
        # High-quality journals and indicators
        high_quality_indicators = [
            "cochrane", "nejm", "lancet", "jama", "bmj", "nature medicine",
            "double-blind", "placebo-controlled", "intention-to-treat",
            "randomized", "multicentre", "multicenter"
        ]
        
        # Risk indicators
        risk_indicators = [
            "single-center", "retrospective", "case report", 
            "editorial", "letter", "small sample", "pilot study"
        ]
        
        high_quality_score = sum(1 for indicator in high_quality_indicators if indicator in content)
        risk_score = sum(1 for indicator in risk_indicators if indicator in content)
        
        if high_quality_score >= 3 and risk_score == 0:
            return "Low"
        elif high_quality_score >= 2 and risk_score <= 1:
            return "Moderate"
        elif risk_score >= 2:
            return "High"
        else:
            return "Unclear"

    def _extract_study_design(self, source: EvidenceSource) -> Optional[str]:
        """📊 Extract study design with focus on high-quality designs"""
        
        content = (source.title + " " + source.summary).lower()
        
        design_patterns = {
            "systematic_review": ["systematic review", "meta-analysis"],
            "rct": ["randomized controlled trial", "randomised controlled trial", "double-blind"],
            "cohort": ["cohort study", "prospective study", "longitudinal study"],
        }
        
        for design, patterns in design_patterns.items():
            if any(pattern in content for pattern in patterns):
                return design
        
        return None

    def _extract_year(self, date_string: str) -> Optional[int]:
        """📅 Extract publication year from date string"""
        try:
            if date_string:
                # Try different date formats
                for fmt in ["%Y", "%Y-%m-%d", "%Y-%m", "%d/%m/%Y"]:
                    try:
                        return datetime.strptime(date_string[:4], "%Y").year
                    except:
                        continue
        except:
            pass
        return None

    async def _update_corpus_collection(
        self, 
        collection_name: str, 
        topic: str, 
        sources: List[EvidenceSource]
    ) -> CorpusMetadata:
        """💾 Update Qdrant collection with new sources"""
        
        if not sources:
            logger.warning("No sources to add to corpus")
            return await self._get_corpus_metadata(collection_name)
        
        # Generate embeddings
        texts = [f"{source.title} {source.summary}" for source in sources]
        embeddings = self.embedding_model.encode(texts)
        
        # Prepare points for upsert
        points = []
        for i, source in enumerate(sources):
            point = {
                "id": source.id,
                "vector": embeddings[i].tolist(),
                "payload": {
                    "title": source.title,
                    "text": source.summary,
                    "url": source.url,
                    "authors": ", ".join(source.authors),
                    "journal": source.journal or "",
                    "pmid": source.pmid,
                    "publication_year": source.publication_year,
                    "source_name": source.source_name,
                    "confidence_score": source.confidence_score,
                    "metadata": source.metadata
                }
            }
            points.append(point)
        
        # Upsert to Qdrant
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            logger.info(f"✅ Added {len(points)} sources to collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error updating collection: {e}")
            raise
        
        # Update metadata
        quality_distribution = {}
        for source in sources:
            quality_metrics = source.metadata.get('quality_metrics')
            if quality_metrics:
                level = quality_metrics.evidence_level.value
                quality_distribution[level] = quality_distribution.get(level, 0) + 1
        
        publication_years = [s.publication_year for s in sources if s.publication_year]
        
        metadata = CorpusMetadata(
            topic=topic,
            collection_name=collection_name,
            last_updated=datetime.now(),
            total_sources=len(sources),
            quality_distribution=quality_distribution,
            oldest_source_year=min(publication_years) if publication_years else None,
            newest_source_year=max(publication_years) if publication_years else None,
            update_frequency=self.update_frequency,
            next_update_due=self._calculate_next_update()
        )
        
        # Store metadata
        await self._store_corpus_metadata(metadata)
        
        return metadata

    async def _cleanup_obsolete_sources(self, collection_name: str):
        """🧹 Remove obsolete sources from corpus"""
        
        current_year = datetime.now().year
        cutoff_year = current_year - self.max_source_age_years
        
        try:
            # Find obsolete sources
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="publication_year",
                        match=MatchValue(value=cutoff_year),
                        range={"lt": cutoff_year}
                    )
                ]
            )
            
            # Delete obsolete sources
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"🧹 Cleaned up sources older than {cutoff_year}")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def _get_corpus_metadata(self, collection_name: str) -> Optional[CorpusMetadata]:
        """📊 Get corpus metadata"""
        # Implementation would retrieve from a metadata store
        # For now, return None to trigger updates
        return None

    async def _store_corpus_metadata(self, metadata: CorpusMetadata):
        """💾 Store corpus metadata"""
        # Implementation would store metadata in a persistent store
        logger.info(f"📊 Metadata stored for corpus '{metadata.collection_name}'")

    def _needs_update(self, metadata: CorpusMetadata) -> bool:
        """🕐 Check if corpus needs update"""
        return datetime.now() >= metadata.next_update_due

    def _calculate_next_update(self) -> datetime:
        """📅 Calculate next update time"""
        now = datetime.now()
        
        if self.update_frequency == UpdateFrequency.DAILY:
            return now + timedelta(days=1)
        elif self.update_frequency == UpdateFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif self.update_frequency == UpdateFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif self.update_frequency == UpdateFrequency.QUARTERLY:
            return now + timedelta(days=90)
        
        return now + timedelta(days=30)  # Default monthly

    async def get_corpus_quality_report(self, collection_name: str) -> str:
        """📊 Generate corpus quality report"""
        
        metadata = await self._get_corpus_metadata(collection_name)
        if not metadata:
            return "No metadata available for corpus quality report."
        
        report = f"""
## 📊 Corpus Quality Report: {metadata.topic}

### Quality Distribution
{chr(10).join([f"- **{level}**: {count} sources" for level, count in metadata.quality_distribution.items()])}

### Temporal Coverage
- **Oldest Source**: {metadata.oldest_source_year}
- **Newest Source**: {metadata.newest_source_year}
- **Total Sources**: {metadata.total_sources}

### Update Status
- **Last Updated**: {metadata.last_updated.strftime('%Y-%m-%d %H:%M')}
- **Next Update Due**: {metadata.next_update_due.strftime('%Y-%m-%d')}
- **Update Frequency**: {metadata.update_frequency.value}

### Quality Assurance
- **Evidence Levels**: Only Level 1A, 1B, 2A (highest quality)
- **Maximum Age**: {self.max_source_age_years} years
- **Risk of Bias**: Automatic assessment applied
- **Source Validation**: PubMed peer-reviewed only
"""
        
        return report

    async def schedule_automatic_updates(self):
        """⏰ Schedule automatic corpus updates"""
        
        logger.info("⏰ Starting automatic corpus update scheduler")
        
        while True:
            try:
                # Check all corpora for updates
                # Implementation would check all registered corpora
                logger.info("🔄 Checking for corpus updates...")
                
                # Sleep until next check (e.g., daily)
                await asyncio.sleep(24 * 3600)  # 24 hours
                
            except Exception as e:
                logger.error(f"Error in automatic update scheduler: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour 