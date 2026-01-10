"""Section-aware Wikipedia and Wikivoyage parsing.

Splits pages by section headings to preserve structure and enable
section-level metadata extraction.
"""
import re
import logging
from typing import List, Tuple
from dataclasses import dataclass

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
        
        logger.debug(f"Parsed '{page_title}': {len(sections)} sections")
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


class WikivoyageSectionParser:
    """Parse Wikivoyage pages into sections."""
    
    # Wikivoyage uses similar heading structure to Wikipedia
    HEADING_PATTERN = re.compile(r'^(={2,6})\s*(.+?)\s*\1\s*$', re.MULTILINE)
    
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
                sections.append(Section(
                    title=page_title,
                    content=text.strip(),
                    level=1,
                    page_title=page_title,
                ))
            return sections
        
        # Extract intro
        intro_text = text[:headings[0].start()].strip()
        if len(intro_text) >= self.min_section_length:
            sections.append(Section(
                title=f"{page_title} (Overview)",
                content=intro_text,
                level=1,
                page_title=page_title,
            ))
        
        # Extract sections
        for i, heading_match in enumerate(headings):
            start = heading_match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            
            section_text = text[start:end].strip()
            
            if len(section_text) < self.min_section_length:
                continue
            
            heading_level = len(heading_match.group(1))
            section_title = heading_match.group(2).strip()
            
            sections.append(Section(
                title=section_title,
                content=section_text,
                level=heading_level,
                page_title=page_title,
            ))
        
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
