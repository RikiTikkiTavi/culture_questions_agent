import logging
import os
from pathlib import Path
import re
from typing import Iterable, List, Optional
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
import xml.etree.ElementTree as ET

import tqdm

from culture_questions_agent.ingestion.metadata_extractor import MetadataExtractor
from culture_questions_agent.ingestion.wikipedia import Section, WikipediaSectionParser


logger = logging.getLogger(__name__)


class WikivoyageSectionParser:
    """Parse Wikivoyage pages into sections."""

    # Wikivoyage uses similar heading structure to Wikipedia
    HEADING_PATTERN = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

    def __init__(self, min_section_length: int = 100):
        """
        Initialize Wikivoyage section parser.

        Args:
            min_section_length: Minimum characters for a section
        """
        self.min_section_length = min_section_length

    def parse_sections(self, page_title: str, text: str) -> List[Section]:
        """
        Parse Wikivoyage page into sections.

        Args:
            page_title: Page title (usually a location)
            text: Page text

        Returns:
            List of Section objects
        """
        sections = []

        # Find all headings
        headings = list(self.HEADING_PATTERN.finditer(text))

        if not headings:
            # No sections - treat as single section
            if len(text.strip()) >= self.min_section_length:
                sections.append(
                    Section(
                        title=page_title,
                        content=text.strip(),
                        level=1,
                        page_title=page_title,
                    )
                )
            return sections

        # Extract intro
        intro_text = text[: headings[0].start()].strip()
        if len(intro_text) >= self.min_section_length:
            sections.append(
                Section(
                    title=f"{page_title} (Overview)",
                    content=intro_text,
                    level=1,
                    page_title=page_title,
                )
            )

        # Extract sections
        for i, heading_match in enumerate(headings):
            start = heading_match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)

            section_text = text[start:end].strip()

            if len(section_text) < self.min_section_length:
                continue

            heading_level = len(heading_match.group(1))
            section_title = heading_match.group(2).strip()

            sections.append(
                Section(
                    title=section_title,
                    content=section_text,
                    level=heading_level,
                    page_title=page_title,
                )
            )

        logger.debug(f"Parsed Wikivoyage '{page_title}': {len(sections)} sections")
        return sections

    def clean_section_text(self, text: str) -> str:
        """
        Clean Wikivoyage markup.

        Similar to Wikipedia but may have travel-specific templates.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Use same cleaning as Wikipedia (same markup)
        parser = WikipediaSectionParser()
        return parser.clean_section_text(text)


class WikivoyageReader(BasePydanticReader):
    """Wikivoyage XML Dump Reader"""

    is_remote: bool = False

    @classmethod
    def class_name(cls) -> str:
        return "WikivoyageReader"

    def _process_wikivoyage_pages_chunk(
        self,
        pages_chunk: list[tuple[str, str]],
        country_filter: Optional[list[str]],
    ) -> list[Document]:
        """
        Process a chunk of Wikivoyage pages in parallel.
        
        Args:
            pages_chunk: List of (title, text) tuples
            country_filter: Optional list of country names to filter by
            
        Returns:
            List of Document objects
        """
        
        section_parser = WikivoyageSectionParser(min_section_length=100)
        metadata_extractor = MetadataExtractor()
        documents = []
        
        # Normalize country filter for matching
        normalized_filter = None
        if country_filter:
            normalized_filter = {country.lower() for country in country_filter}
        
        for title, text in pages_chunk:
            # Apply country filter if specified
            if normalized_filter:
                title_lower = title.lower()
                country_match = any(country in title_lower for country in normalized_filter)
                country_mentioned = any(country in text.lower() for country in normalized_filter)
                
                if not country_match and not country_mentioned:
                    continue
            
            # Parse into sections
            sections = section_parser.parse_sections(title, text)
            
            # Create documents for each section
            for section in sections:
                # Clean section text
                clean_text = section_parser.clean_section_text(section.content)
                
                if len(clean_text) < 100:
                    continue
                
                # Extract metadata
                metadata = metadata_extractor.extract_metadata_from_wikivoyage(
                    page_title=section.page_title,
                    section_title=section.title if section.title != title else None,
                    section_text=clean_text,
                )
                
                # Double-check country match in metadata if filtering
                if normalized_filter:
                    doc_country = metadata.country.lower() if metadata.country else ""
                    if doc_country and doc_country not in normalized_filter:
                        continue
                
                # Create document
                doc = Document(
                    text=clean_text,
                    metadata=metadata.to_dict(),
                )
                documents.append(doc)

        return documents

    def lazy_load_data(
        self, xml_path: str, country_filter: Optional[list[str]]
    ) -> Iterable[Document]:
        """Load data from the input directory lazily."""

        if not os.path.exists(xml_path):
            logger.warning(f"Wikivoyage file not found: {xml_path}")
            return []

        if country_filter:
            logger.info(
                f"  Filtering by {len(country_filter)} countries: {country_filter}"
            )

        ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}

        # First pass: Extract all pages into memory
        logger.info(f"[1/2] Extracting pages from XML: {xml_path}")
        pages = []
        pages_filtered = 0

        # Normalize country filter for early filtering
        normalized_filter = None
        if country_filter:
            normalized_filter = {country.lower() for country in country_filter}

        context = ET.iterparse(xml_path, events=("end",))

        for event, elem in tqdm.tqdm(context, desc="Extracting pages"):
            if elem.tag.endswith("page"):
                title_elem = elem.find("mw:title", ns)
                revision = elem.find("mw:revision", ns)
                text_elem = (
                    revision.find("mw:text", ns) if revision is not None else None
                )

                title = title_elem.text if title_elem is not None else None
                text = text_elem.text if text_elem is not None else ""

                # Skip empty pages and redirects
                if (
                    not text
                    or not title
                    or text.strip().upper().startswith("#REDIRECT")
                ):
                    elem.clear()
                    continue

                # Apply early country filter on title (before storing in memory)
                if normalized_filter:
                    title_lower = title.lower()
                    country_match = any(
                        country in title_lower for country in normalized_filter
                    )

                    if not country_match:
                        pages_filtered += 1
                        elem.clear()
                        continue

                pages.append((title, text))
                elem.clear()

        logger.info(f"  Extracted {len(pages)} pages")
        if normalized_filter:
            logger.info(f"  Filtered out {pages_filtered} pages by country")

        if not pages:
            logger.warning("  No pages to process!")
            return []

        documents = self._process_wikivoyage_pages_chunk(country_filter=country_filter, pages_chunk=pages)

        logger.info("=" * 80)
        logger.info(f"✓ Wikivoyage Loading Complete")
        logger.info(f"  Pages processed: {len(pages)}")
        logger.info(f"  Documents created: {len(documents)}")
        logger.info("=" * 80)

        return documents
