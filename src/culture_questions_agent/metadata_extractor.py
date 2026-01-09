"""Metadata extraction and enrichment for cultural QA documents.

This module extracts rich metadata from Wikipedia and Wikivoyage pages to support
high-quality answer-centric retrieval. Metadata is attached at the node level to
survive indexing and retrieval.
"""
import re
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CulturalMetadata:
    """Rich metadata for cultural QA nodes."""
    country: Optional[str] = None
    culture_domain: Optional[str] = None  # cuisine, holidays, music, education, customs, etc.
    source: str = "unknown"  # wikipedia / wikivoyage
    document_type: str = "encyclopedic"  # encyclopedic / travel / historical
    section_title: Optional[str] = None
    page_title: Optional[str] = None
    source_reliability: str = "high"  # static value for Wikipedia/Wikivoyage
    granularity: Optional[str] = None  # small / medium / large
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for LlamaIndex metadata."""
        return {
            k: v for k, v in {
                "country": self.country,
                "culture_domain": self.culture_domain,
                "source": self.source,
                "document_type": self.document_type,
                "section_title": self.section_title,
                "page_title": self.page_title,
                "source_reliability": self.source_reliability,
                "granularity": self.granularity,
            }.items() if v is not None
        }


class MetadataExtractor:
    """Extract cultural metadata from Wikipedia and Wikivoyage content."""
    
    # Culture domain patterns - maps keywords to domains
    DOMAIN_PATTERNS = {
        "cuisine": [
            r"\bfood\b", r"\bcuisine\b", r"\bdish\b", r"\bmeal\b", r"\brecipe\b",
            r"\bcooking\b", r"\bingredient\b", r"\bbeverage\b", r"\bdrink\b",
        ],
        "holidays": [
            r"\bholiday\b", r"\bfestival\b", r"\bcelebration\b", r"\bceremony\b",
            r"\boccasion\b", r"\bcommemoration\b",
        ],
        "music": [
            r"\bmusic\b", r"\bsong\b", r"\binstrument\b", r"\bmelody\b",
            r"\brhythm\b", r"\bdance\b", r"\bperformance\b",
        ],
        "education": [
            r"\beducation\b", r"\bschool\b", r"\buniversity\b", r"\bteaching\b",
            r"\blearning\b", r"\bacademic\b", r"\bcurriculum\b",
        ],
        "customs": [
            r"\bcustom\b", r"\btradition\b", r"\betiquette\b", r"\bmanners\b",
            r"\bpractice\b", r"\britual\b", r"\bnorm\b", r"\bconvention\b",
        ],
        "religion": [
            r"\breligion\b", r"\bfaith\b", r"\bworship\b", r"\btemple\b",
            r"\bchurch\b", r"\bmosque\b", r"\bspiritual\b", r"\bbelief\b",
        ],
        "arts": [
            r"\bart\b", r"\bpainting\b", r"\bsculpture\b", r"\bcraft\b",
            r"\bartisan\b", r"\bhandicraft\b", r"\barchitecture\b",
        ],
        "language": [
            r"\blanguage\b", r"\bdialect\b", r"\bvocabulary\b", r"\bphrase\b",
            r"\bcommunication\b", r"\bspeech\b",
        ],
        "history": [
            r"\bhistory\b", r"\bhistorical\b", r"\bcentury\b", r"\bera\b",
            r"\bperiod\b", r"\bevolution\b", r"\borigin\b",
        ],
        "social": [
            r"\bfamily\b", r"\bmarriage\b", r"\bwedding\b", r"\bsocial\b",
            r"\bcommunity\b", r"\brelationship\b",
        ],
        "sports": [
            r"\bsport\b", r"\bgame\b", r"\bcompetition\b", r"\bathletic\b",
            r"\bplayer\b", r"\bteam\b",
        ],
        "clothing": [
            r"\bclothing\b", r"\bdress\b", r"\battire\b", r"\bgarment\b",
            r"\bfashion\b", r"\bcostume\b",
        ],
    }
    
    def __init__(self):
        """Initialize metadata extractor with compiled regex patterns."""
        # Compile domain patterns for efficiency
        self.domain_patterns_compiled = {
            domain: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for domain, patterns in self.DOMAIN_PATTERNS.items()
        }
    
    def extract_country_from_title(self, title: str) -> Optional[str]:
        """
        Extract country name from Wikipedia/Wikivoyage title.
        
        Handles patterns like:
        - "Culture of Japan"
        - "Public holidays in Brazil"
        - "Brazilian cuisine"
        - "Tokyo" (for Wikivoyage)
        
        Args:
            title: Page title
            
        Returns:
            Country name if found, None otherwise
        """
        # Pattern 1: "X of Country" or "X in Country"
        match = re.search(r'\b(?:of|in)\s+([A-Z][a-zA-Z\s]+)$', title)
        if match:
            country = match.group(1).strip()
            return country
        
        # Pattern 2: "Country X" (e.g., "Brazilian cuisine")
        # This is harder - would need a country adjective mapping
        # For now, extract if title contains known pattern
        
        return None
    
    def extract_culture_domain(self, text: str, title: str = "") -> Optional[str]:
        """
        Identify the primary cultural domain from content.
        
        Uses keyword matching against text and title. Returns the domain
        with the most matches.
        
        Args:
            text: Content text
            title: Page/section title
            
        Returns:
            Culture domain (cuisine, holidays, music, etc.) or None
        """
        combined_text = f"{title} {text}".lower()
        
        # Count matches for each domain
        domain_scores = {}
        for domain, patterns in self.domain_patterns_compiled.items():
            score = sum(
                len(pattern.findall(combined_text))
                for pattern in patterns
            )
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return None
        
        # Return domain with highest score
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain[0]
    
    def extract_document_type(self, source: str, title: str = "") -> str:
        """
        Determine document type based on source and content.
        
        Args:
            source: "wikipedia" or "wikivoyage"
            title: Page title
            
        Returns:
            Document type: "encyclopedic", "travel", or "historical"
        """
        if source == "wikivoyage":
            return "travel"
        
        # Wikipedia classification based on title
        if any(keyword in title.lower() for keyword in ["history", "historical", "origin", "evolution"]):
            return "historical"
        
        return "encyclopedic"
    
    def extract_metadata_from_wikipedia(
        self,
        page_title: str,
        section_title: Optional[str] = None,
        section_text: str = "",
    ) -> CulturalMetadata:
        """
        Extract metadata from Wikipedia page/section.
        
        Args:
            page_title: Wikipedia page title
            section_title: Section heading (if applicable)
            section_text: Section content
            
        Returns:
            CulturalMetadata object
        """
        country = self.extract_country_from_title(page_title)
        culture_domain = self.extract_culture_domain(
            section_text,
            f"{page_title} {section_title or ''}"
        )
        document_type = self.extract_document_type("wikipedia", page_title)
        
        return CulturalMetadata(
            country=country,
            culture_domain=culture_domain,
            source="wikipedia",
            document_type=document_type,
            section_title=section_title,
            page_title=page_title,
            source_reliability="high",
        )
    
    def extract_metadata_from_wikivoyage(
        self,
        page_title: str,
        section_title: Optional[str] = None,
        section_text: str = "",
    ) -> CulturalMetadata:
        """
        Extract metadata from Wikivoyage page/section.
        
        Args:
            page_title: Wikivoyage page title (often a city/region)
            section_title: Section heading (if applicable)
            section_text: Section content
            
        Returns:
            CulturalMetadata object
        """
        # Wikivoyage titles are often locations, not "X of Country" format
        # Country extraction is harder - may need geolocation lookup
        country = None  # Could enhance with geolocation API
        
        culture_domain = self.extract_culture_domain(
            section_text,
            f"{page_title} {section_title or ''}"
        )
        
        return CulturalMetadata(
            country=country,
            culture_domain=culture_domain,
            source="wikivoyage",
            document_type="travel",
            section_title=section_title,
            page_title=page_title,
            source_reliability="high",
        )
