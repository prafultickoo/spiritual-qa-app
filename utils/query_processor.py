"""
Enhanced Query Processor for Chapter/Verse Queries
Handles various formats of spiritual text references.
"""

import re
from typing import List, Dict, Tuple

class ChapterVerseQueryProcessor:
    """Process and enhance queries for better chapter/verse retrieval."""
    
    def __init__(self):
        # Common spiritual text names and their variations
        self.text_variations = {
            'bhagavad gita': ['gita', 'bhagavad gita', 'bhagwad gita', 'bg'],
            'upanishads': ['upanishad', 'upanishads'],
            'ramayana': ['ramayana'],
            'mahabharata': ['mahabharata', 'mahabharat']
        }
        
        # Chapter/verse patterns
        self.patterns = [
            # "Bhagavad Gita Chapter 2 Verse 47"
            r'(\w+(?:\s+\w+)*)\s+chapter\s+(\d+)\s+verse\s+(\d+)',
            # "Chapter 2 Verse 47"
            r'chapter\s+(\d+)\s+verse\s+(\d+)',
            # "Bhagavad Gita 2.47" or "Gita 2.47"
            r'(\w+(?:\s+\w+)*)\s+(\d+)\.(\d+)',
            # "2.47" (standalone)
            r'^(\d+)\.(\d+)$',
            # "Chapter 2, verse 47"
            r'chapter\s+(\d+),?\s+verse\s+(\d+)',
        ]
    
    def process_query(self, query: str) -> Dict[str, any]:
        """
        Process a user query and extract chapter/verse information.
        
        Args:
            query: User's spiritual query
            
        Returns:
            Dict with enhanced query information
        """
        original_query = query.strip()
        query_lower = original_query.lower()
        
        result = {
            'original_query': original_query,
            'enhanced_queries': [original_query],  # Always include original
            'detected_text': None,
            'detected_chapter': None,
            'detected_verse': None,
            'is_chapter_verse_query': False
        }
        
        # Try to match chapter/verse patterns
        for pattern in self.patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_chapter_verse_query'] = True
                groups = match.groups()
                
                if len(groups) == 3:  # Text + Chapter + Verse
                    text_name, chapter, verse = groups
                    result['detected_text'] = self._normalize_text_name(text_name)
                    result['detected_chapter'] = int(chapter)
                    result['detected_verse'] = int(verse)
                elif len(groups) == 2:  # Chapter + Verse only
                    chapter, verse = groups
                    result['detected_chapter'] = int(chapter)
                    result['detected_verse'] = int(verse)
                    # Default to Bhagavad Gita if no text specified
                    result['detected_text'] = 'bhagavad gita'
                
                break
        
        # Generate enhanced search queries
        if result['is_chapter_verse_query']:
            result['enhanced_queries'] = self._generate_enhanced_queries(result)
        
        return result
    
    def _normalize_text_name(self, text_name: str) -> str:
        """Normalize text name to standard format."""
        text_lower = text_name.lower().strip()
        
        for standard_name, variations in self.text_variations.items():
            if text_lower in variations:
                return standard_name
        
        return text_lower
    
    def _generate_enhanced_queries(self, query_info: Dict) -> List[str]:
        """Generate multiple query variations for better matching."""
        queries = [query_info['original_query']]
        
        text = query_info.get('detected_text', 'bhagavad gita')
        chapter = query_info.get('detected_chapter')
        verse = query_info.get('detected_verse')
        
        if chapter and verse:
            # Add various formats
            base_formats = [
                f"{text} {chapter}.{verse}",
                f"{text} chapter {chapter} verse {verse}",
                f"chapter {chapter} verse {verse}",
                f"{chapter}.{verse}",
                f"verse {verse} chapter {chapter}",
                f"{text} {chapter}:{verse}",
            ]
            
            # Add content-based queries (most effective)
            content_queries = []
            if text == 'bhagavad gita':
                if chapter == 2 and verse == 47:
                    content_queries.extend([
                        "karmaṇy evādhikāras te",
                        "karmanyevadhikaraste",
                        "right to action not fruits",
                        "perform duty without attachment",
                        "karma yoga action"
                    ])
                elif chapter == 2 and verse == 48:
                    content_queries.extend([
                        "yogastha kuru karmāṇi",
                        "yoga equilibrium balance",
                        "perform actions in yoga",
                        "established in yoga"
                    ])
                elif chapter == 18 and verse == 66:
                    content_queries.extend([
                        "sarva-dharmān parityajya",
                        "abandon all dharma",
                        "surrender unto me",
                        "final instruction gita"
                    ])
            
            queries.extend(base_formats)
            queries.extend(content_queries)
        
        return list(set(queries))  # Remove duplicates

def enhance_spiritual_query(query: str) -> Dict:
    """Main function to enhance spiritual queries."""
    processor = ChapterVerseQueryProcessor()
    return processor.process_query(query)

# Test the processor
if __name__ == "__main__":
    test_queries = [
        "Show me Bhagavad Gita Chapter 2 Verse 47",
        "What does Chapter 13 verse 11 say?",
        "Bhagavad Gita 2.48",
        "Chapter 18 verse 66 Gita",
        "Find verse about karma from Chapter 2"
    ]
    
    processor = ChapterVerseQueryProcessor()
    for query in test_queries:
        result = processor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Enhanced queries: {result['enhanced_queries'][:3]}")
        print(f"Chapter/Verse: {result['detected_chapter']}.{result['detected_verse']}")
