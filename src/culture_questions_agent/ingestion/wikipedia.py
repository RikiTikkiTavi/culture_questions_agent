from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, Dict, List

from tqdm import tqdm

from culture_questions_agent.ingestion.metadata_extractor import MetadataExtractor

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from llama_index.readers.wikipedia import WikipediaReader

logger = logging.getLogger(__name__)

@dataclass
class Section:
    """Represents a section from a Wikipedia or Wikivoyage page."""
    title: str
    content: str
    level: int  # Heading level (1-6)
    page_title: str
    
    def __repr__(self):
        return f"Section(title='{self.title}', level={self.level}, chars={len(self.content)})"


class WikipediaSectionParser:
    """Parse Wikipedia pages into sections based on heading structure."""
    
    # MediaWiki heading pattern: == Heading ==
    HEADING_PATTERN = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    
    def __init__(self, min_section_length: int = 100):
        """
        Initialize section parser.
        
        Args:
            min_section_length: Minimum characters for a section to be kept
        """
        self.min_section_length = min_section_length
    
    def parse_sections(self, page_title: str, text: str) -> List[Section]:
        """
        Parse Wikipedia page text into sections.
        
        Args:
            page_title: Title of the Wikipedia page
            text: Raw page text (may contain MediaWiki markup)
            
        Returns:
            List of Section objects
        """
        sections = []
        
        # Find all headings
        headings = list(self.HEADING_PATTERN.finditer(text))
        
        if not headings:
            # No sections found - treat entire page as one section
            if len(text.strip()) >= self.min_section_length:
                sections.append(Section(
                    title=page_title,
                    content=text.strip(),
                    level=1,
                    page_title=page_title,
                ))
            return sections
        
        # Extract intro (text before first heading)
        intro_text = text[:headings[0].start()].strip()
        if len(intro_text) >= self.min_section_length:
            sections.append(Section(
                title=f"{page_title} (Introduction)",
                content=intro_text,
                level=1,
                page_title=page_title,
            ))
        
        # Extract each section
        for i, heading_match in enumerate(headings):
            # Determine section boundaries
            start = heading_match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            
            # Extract section content
            section_text = text[start:end].strip()
            
            # Skip short sections
            if len(section_text) < self.min_section_length:
                continue
            
            # Extract heading info
            heading_level = len(heading_match.group(1))
            section_title = heading_match.group(2).strip()
            
            sections.append(Section(
                title=section_title,
                content=section_text,
                level=heading_level,
                page_title=page_title,
            ))
        return sections
    
    def clean_section_text(self, text: str) -> str:
        """
        Clean MediaWiki markup from section text.
        
        Args:
            text: Raw section text
            
        Returns:
            Cleaned text
        """
        # Remove templates: {{template}}
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
        
        # Remove categories, files, namespaces: [[Category:X]], [[File:X]]
        text = re.sub(r'\[\[[A-Za-z]+:[^\]]+\]\]', '', text)
        
        # Handle external links with text: [http://url text] or [https://url text]
        # Extract just the display text, remove URL
        text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        
        # Remove bare external links: [http://url] or [https://url]
        text = re.sub(r'\[https?://[^\]]+\]', '', text)
        
        # Handle wiki links with alternate text: [[Link|Display]]
        text = re.sub(r'\[\[[^\|\]]*\|([^\]]+)\]\]', r'\1', text)
        
        # Handle simple wiki links: [[Link]]
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
        
        # Remove bold/italic markup: ''text'' or '''text'''
        text = re.sub(r"'{2,}", '', text)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove references: <ref>...</ref>
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<ref[^>]*/>', '', text, flags=re.IGNORECASE)
        
        # Remove inline CSS/SVG chart data (e.g., .mw-chart-...)
        # This handles Wikipedia's chart visualizations that appear as inline CSS
        text = re.sub(r'\.mw-chart-[a-f0-9]+[^{]*\{[^}]*\}', '', text, flags=re.DOTALL)
        
        # Remove HTML tags (keep text content)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()
        
        return text

class WikipediaTopicReader(BasePydanticReader):

    is_remote: bool = True
    auto_suggest: bool = True

    def __init__(self, auto_suggest: bool = True):
        super().__init__()
        self.auto_suggest = auto_suggest

        
    def lazy_load_data(
        self,
        templates: list[str],
        additional_pages: list[str],
        country_list: list[str]
    ) -> list[Document]:
        """
        Load Wikipedia pages and split into sections with metadata.
        Uses caching to avoid re-downloading pages.
        
        Args:
            cfg: Hydra configuration
            metadata_extractor: Metadata extraction utility
            
        Returns:
            Tuple of (List of Document objects with section-level metadata, extraction report)
        """
        logger.info("="*80)
        logger.info("[Wikipedia] Section-Aware Loading with Caching")
        logger.info("="*80)
        
        metadata_extractor = MetadataExtractor()

        
        # Generate topics from templates
        logger.info("Generating topics from templates...")
        topics = []

        for country in country_list:
            for template in templates:
                topics.append(template.format(country))
        
        # Add any additional Wikipedia pages from config
        if additional_pages:
            topics.extend(additional_pages)
            logger.info(f"  ✓ Added {len(additional_pages)} additional Wikipedia pages")
        
        logger.info(f"  ✓ Generated {len(topics)} total topics ({len(templates)} templates × {len(country_list)} countries + {len(additional_pages)} additional pages)")
        
        # Load Wikipedia pages (from cache or API)
        logger.info("Loading Wikipedia pages...")
        reader = WikipediaReader()
        section_parser = WikipediaSectionParser(min_section_length=100)
        
        documents = []
        successful_pages = []
        failed_pages = []
        section_count = 0
        newly_downloaded = 0
        
        for topic in tqdm(topics, desc="Loading Wikipedia"):
            try:
                # Download from Wikipedia
                page_docs = reader.load_data(pages=[topic], auto_suggest=self.auto_suggest)
                
                if not page_docs:
                    continue
                
                page_doc = page_docs[0]
                page_title = page_doc.metadata.get("title", topic)
                page_text = page_doc.get_content()
                
                # Parse into sections
                sections = section_parser.parse_sections(page_title, page_text)
                
                # Create documents for each section
                for section in sections:
                    # Clean section text
                    clean_text = section_parser.clean_section_text(section.content)
                    
                    if len(clean_text) < 100:
                        continue
                    
                    # Extract metadata
                    metadata = metadata_extractor.extract_metadata_from_wikipedia(
                        page_title=section.page_title,
                        section_title=section.title if section.title != page_title else None,
                        section_text=clean_text,
                    )
                    
                    # Create document with rich metadata
                    doc = Document(
                        text=clean_text,
                        metadata=metadata.to_dict(),
                    )
                    documents.append(doc)
                    section_count += 1
                
                successful_pages.append(topic)
                
            except Exception as e:
                failed_pages.append({"topic": topic, "error": str(e)})
                logger.debug(f"Failed to load '{topic}': {e}")
        
        logger.info("="*80)
        logger.info(f"✓ Wikipedia Loading Complete")
        logger.info(f"  Pages: {len(successful_pages)}/{len(topics)} successful")
        logger.info(f"  Sections: {section_count}")
        logger.info(f"  Documents: {len(documents)}")
        logger.info("="*80)
        
        return documents