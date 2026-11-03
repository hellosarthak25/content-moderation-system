"""
text_cleaner.py
Reusable text cleaning pipeline for content moderation comments.
"""

import re
import html

# Common English contraction expansions
CONTRACTION_MAP = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "u": "you",
    "ur": "your",
    "r": "are",
}

def clean_text(text: str, lower: bool = True) -> str:
    """
    Cleans raw comment text while preserving crucial toxicity signals.
    
    1. Unescapes HTML entities (&amp; -> &)
    2. Strips HTML tags (<br>, <p>, etc.)
    3. Removes URLs and web links
    4. Removes IP addresses
    5. Strips Wikipedia markup headers (== Section ==) and user signatures
    6. Expands common contractions (don't -> do not)
    7. Collapses excessive repeated characters (e.g., 'sooooo' -> 'soo', 'fuuuck' -> 'fuck')
    8. Normalizes excess whitespace and newlines
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # 4. Remove IP addresses (e.g., 192.168.1.1)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ", text)
    
    # 5. Remove Wikipedia section headers (e.g., == Discussion ==)
    text = re.sub(r"=+\s*[^=]+\s*=+", " ", text)
    
    # 6. Remove timestamps (e.g., 01:23, 14 January 2008 (UTC))
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?,?\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*(?:\(UTC\))?", " ", text)
    
    # 7. Expand contractions
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text, flags=re.IGNORECASE)
    
    # 8. Collapse repeated characters (3 or more of same character -> max 2)
    # E.g., 'helllooooo' -> 'helloo', 'fuuuuck' -> 'fuck'
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    
    # 9. Clean punctuation & special symbols (keep alphanumeric and basic punctuation)
    # Retain standard apostrophes and hyphens inside words
    text = re.sub(r"[^a-zA-Z0-9\s!?.,]", " ", text)
    
    if lower:
        text = text.lower()
        
    # 10. Normalize whitespace (tabs, newlines, multiple spaces)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
