"""
Enhanced verse detection with chapter/structure identification and metadata extraction.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VerseInfo:
    """Information about a detected verse."""
    text: str
    verse_number: Optional[int] = None
    chapter_number: Optional[int] = None
    source_text: Optional[str] = None
    is_sanskrit: bool = False
    has_transliteration: bool = False
    verse_markers: List[str] = None
    
    def __post_init__(self):
        if self.verse_markers is None:
            self.verse_markers = []

@dataclass
class StructuralMetadata:
    """Structural metadata extracted from text content."""
    source_text: Optional[str] = None
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    section_name: Optional[str] = None
    verses_detected: List[VerseInfo] = None
    contains_sanskrit: bool = False
    contains_transliteration: bool = False
    
    def __post_init__(self):
        if self.verses_detected is None:
            self.verses_detected = []

class EnhancedVerseDetector:
    """Enhanced verse detection with structural analysis."""
    
    def __init__(self):
        """Initialize the enhanced verse detector."""
        
        # Spiritual text identification patterns
        self.text_patterns = {
            'bhagavad_gita': [
                r'(?i)bhagavad[- ]?g[īi]t[āa]',
                r'(?i)bhagwad[- ]?gita',
                r'गीता',
                r'श्रीमद्भगवद्गीता'
            ],
            'upanishads': [
                r'(?i)upanishad',
                r'उपनिषद्',
                r'(?i)(isha|kena|katha|prashna|mundaka|mandukya|taittiriya|aitareya|chandogya|brihadaranyaka)'
            ],
            'vedas': [
                r'(?i)(rig|sama|yajur|atharva)[- ]?veda',
                r'वेद',
                r'(?i)vedic'
            ],
            'puranas': [
                r'(?i)purana',
                r'पुराण',
                r'(?i)(vishnu|shiva|brahma|garuda|narada|bhagavata|skanda)'
            ],
            'ramayana': [
                r'(?i)r[āa]m[āa]ya[nṇ]a',
                r'रामायण'
            ],
            'mahabharata': [
                r'(?i)mah[āa]bh[āa]rata',
                r'महाभारत'
            ]
        }
        
        # Chapter identification patterns
        self.chapter_patterns = [
            r'(?i)chapter\s+(\d+)',
            r'(?i)adhyaya\s+(\d+)',
            r'अध्याय\s+(\d+)',
            r'(?i)canto\s+(\d+)',
            r'(?i)section\s+(\d+)',
            r'(?i)kanda\s+(\d+)'
        ]
        
        # Verse number patterns (enhanced for different formats)
        self.verse_patterns = [
            # Bhagavad Gita format: 2.48, (2.48), etc.
            r'(\d+)\.(\d+)(?:\s*[-:।॥]\s*(.+?)(?=\d+\.\d+|\n\n|$))?',
            
            # Traditional verse numbering: Verse 48, श्लोक 48
            r'(?i)(?:verse|shloka|श्लोक)\s+(\d+)(?:\s*[-:।॥]\s*(.+?)(?=(?:verse|shloka|श्लोक)|\n\n|$))?',
            
            # Numbered verses with markers: 48. text, 48) text
            r'^(\d+)[.)]\s*(.+?)(?=^\d+[.)]|\n\n|$)',
            
            # Sanskrit verse markers: ॥ text ॥
            r'॥\s*(.+?)\s*॥',
            
            # Pipe delimited verses: | text |
            r'\|\s*(.+?)\s*\|',
            
            # Double pipe verses: || text ||
            r'\|\|\s*(.+?)\s*\|\|'
        ]
        
        # Sanskrit/Devanagari detection
        self.sanskrit_patterns = [
            r'[\u0900-\u097F]+',  # Devanagari Unicode range
            r'[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह]+',  # Common Sanskrit consonants
            r'[अआइईउऊऋएऐओऔ]+',  # Sanskrit vowels
        ]
        
        # Transliteration patterns (IAST and common romanization)
        self.transliteration_patterns = [
            r'\b[a-zA-Z]*[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ][a-zA-Z]*\b',  # IAST diacritics
            r'\b[a-zA-Z]*[ṁṃ][a-zA-Z]*\b',  # Anusvara variants
            r'\b(om|aum|namah|namaha|svaha|svadha)\b',  # Common Sanskrit words
            r'\b[a-zA-Z]*(?:sth|dh|bh|gh|kh|ch|th|ph)[a-zA-Z]*\b'  # Sanskrit consonant clusters
        ]
    
    def detect_verses_and_structure(self, text: str) -> StructuralMetadata:
        """
        Detect verses and extract structural metadata from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            StructuralMetadata object with detected information
        """
        metadata = StructuralMetadata()
        
        # Detect source text
        metadata.source_text = self._identify_source_text(text)
        
        # Detect chapter information
        chapter_info = self._extract_chapter_info(text)
        metadata.chapter_number = chapter_info.get('number')
        metadata.chapter_title = chapter_info.get('title')
        
        # Detect Sanskrit and transliteration
        metadata.contains_sanskrit = self._contains_sanskrit(text)
        metadata.contains_transliteration = self._contains_transliteration(text)
        
        # Detect verses
        metadata.verses_detected = self._detect_verses(text, metadata.source_text)
        
        logger.info(f"Detected {len(metadata.verses_detected)} verses in {metadata.source_text or 'unknown text'}")
        
        return metadata
    
    def _identify_source_text(self, text: str) -> Optional[str]:
        """Identify the source spiritual text."""
        text_lower = text.lower()
        
        for text_name, patterns in self.text_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
                    logger.debug(f"Identified source text: {text_name}")
                    return text_name.replace('_', ' ').title()
        
        return None
    
    def _extract_chapter_info(self, text: str) -> Dict[str, Any]:
        """Extract chapter number and title information."""
        chapter_info = {}
        
        for pattern in self.chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
            if match:
                chapter_info['number'] = int(match.group(1))
                
                # Try to extract chapter title (text following chapter number)
                title_match = re.search(
                    pattern + r'[:\-\s]*([^\n]+)',
                    text,
                    re.IGNORECASE | re.UNICODE
                )
                if title_match and len(title_match.groups()) > 1:
                    chapter_info['title'] = title_match.group(2).strip()
                
                logger.debug(f"Found chapter {chapter_info['number']}: {chapter_info.get('title', 'No title')}")
                break
        
        return chapter_info
    
    def _contains_sanskrit(self, text: str) -> bool:
        """Check if text contains Sanskrit/Devanagari characters."""
        for pattern in self.sanskrit_patterns:
            if re.search(pattern, text, re.UNICODE):
                return True
        return False
    
    def _contains_transliteration(self, text: str) -> bool:
        """Check if text contains Sanskrit transliteration."""
        for pattern in self.transliteration_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
                return True
        return False
    
    def _detect_verses(self, text: str, source_text: Optional[str] = None) -> List[VerseInfo]:
        """Detect individual verses in the text."""
        verses = []
        
        # Try each verse pattern
        for pattern in self.verse_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL | re.UNICODE))
            
            for match in matches:
                verse_info = self._extract_verse_info(match, source_text, pattern)
                if verse_info and verse_info.text.strip():
                    verses.append(verse_info)
        
        # Remove duplicates and sort by position in text
        unique_verses = self._deduplicate_verses(verses, text)
        
        return unique_verses
    
    def _extract_verse_info(self, match: re.Match, source_text: Optional[str], pattern: str) -> Optional[VerseInfo]:
        """Extract verse information from a regex match."""
        groups = match.groups()
        
        if not groups:
            return None
        
        verse_info = VerseInfo(
            text="",
            source_text=source_text,
            is_sanskrit=False,
            has_transliteration=False
        )
        
        # Handle different pattern formats
        if '\\d+)\\.(\d+)' in pattern:  # Chapter.Verse format (e.g., 2.48)
            if len(groups) >= 2:
                verse_info.chapter_number = int(groups[0])
                verse_info.verse_number = int(groups[1])
                verse_info.text = groups[2] if len(groups) > 2 and groups[2] else match.group(0)
        
        elif 'verse|shloka|श्लोक' in pattern.lower():  # Verse/Shloka format
            if len(groups) >= 1:
                verse_info.verse_number = int(groups[0]) if groups[0].isdigit() else None
                verse_info.text = groups[1] if len(groups) > 1 and groups[1] else match.group(0)
        
        elif '^(\d+)[.)]' in pattern:  # Numbered format (48. text)
            if len(groups) >= 2:
                verse_info.verse_number = int(groups[0])
                verse_info.text = groups[1]
        
        else:  # Marker-based formats (॥ text ॥, | text |)
            verse_info.text = groups[0] if groups[0] else match.group(0)
            verse_info.verse_markers = [match.group(0)[:2], match.group(0)[-2:]]
        
        # Clean up the verse text
        verse_info.text = verse_info.text.strip()
        
        # Detect Sanskrit and transliteration in this verse
        verse_info.is_sanskrit = self._contains_sanskrit(verse_info.text)
        verse_info.has_transliteration = self._contains_transliteration(verse_info.text)
        
        return verse_info
    
    def _deduplicate_verses(self, verses: List[VerseInfo], original_text: str) -> List[VerseInfo]:
        """Remove duplicate verses and sort by position in text."""
        unique_verses = []
        seen_texts = set()
        
        # Sort by position in original text
        verses_with_pos = []
        for verse in verses:
            pos = original_text.find(verse.text)
            if pos != -1:
                verses_with_pos.append((pos, verse))
        
        verses_with_pos.sort(key=lambda x: x[0])
        
        # Remove duplicates based on text content
        for pos, verse in verses_with_pos:
            # Normalize text for comparison
            normalized_text = ' '.join(verse.text.split()).lower()
            
            if normalized_text not in seen_texts and len(normalized_text) > 10:
                seen_texts.add(normalized_text)
                unique_verses.append(verse)
        
        return unique_verses
    
    def create_verse_metadata(self, verses: List[VerseInfo]) -> Dict[str, Any]:
        """Create metadata dictionary from detected verses."""
        if not verses:
            return {}
        
        metadata = {
            'has_verses': True,
            'verse_count': len(verses),
            'verses': [],
            'contains_sanskrit': any(v.is_sanskrit for v in verses),
            'contains_transliteration': any(v.has_transliteration for v in verses)
        }
        
        # Add individual verse information
        for verse in verses:
            verse_data = {
                'text': verse.text,
                'is_sanskrit': verse.is_sanskrit,
                'has_transliteration': verse.has_transliteration
            }
            
            if verse.verse_number:
                verse_data['verse_number'] = verse.verse_number
            if verse.chapter_number:
                verse_data['chapter_number'] = verse.chapter_number
            if verse.source_text:
                verse_data['source_text'] = verse.source_text
            if verse.verse_markers:
                verse_data['markers'] = verse.verse_markers
            
            metadata['verses'].append(verse_data)
        
        # Add chapter information if consistent across verses
        chapters = [v.chapter_number for v in verses if v.chapter_number]
        if chapters and len(set(chapters)) == 1:
            metadata['chapter_number'] = chapters[0]
        
        # Add source text if consistent
        sources = [v.source_text for v in verses if v.source_text]
        if sources and len(set(sources)) == 1:
            metadata['source_text'] = sources[0]
        
        return metadata


# Convenience function for integration
def detect_verses_enhanced(text: str) -> Dict[str, Any]:
    """
    Enhanced verse detection function for integration with existing code.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with verse and structural metadata
    """
    detector = EnhancedVerseDetector()
    structural_metadata = detector.detect_verses_and_structure(text)
    
    # Convert to dictionary format for compatibility
    result = {
        'source_text': structural_metadata.source_text,
        'chapter_number': structural_metadata.chapter_number,
        'chapter_title': structural_metadata.chapter_title,
        'contains_sanskrit': structural_metadata.contains_sanskrit,
        'contains_transliteration': structural_metadata.contains_transliteration,
        'verses_detected': len(structural_metadata.verses_detected)
    }
    
    # Add verse metadata
    verse_metadata = detector.create_verse_metadata(structural_metadata.verses_detected)
    result.update(verse_metadata)
    
    return result
