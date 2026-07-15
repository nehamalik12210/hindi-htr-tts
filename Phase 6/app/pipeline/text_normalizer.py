"""
Phase 6 — Hindi text normalizer for TTS.
Ported EXACTLY from tts_integrated.ipynb Cell 4 (HindiTextNormalizer class).
Uses complete Hindi number words for 0-99 (unique words, not tens+ones).
"""

import re
import unicodedata


# Devanagari numerals → Arabic digits (for pre-processing)
DEVA_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Complete Hindi number words for 0-99 (each is a UNIQUE word)
HINDI_ONES = [
    '', 'एक', 'दो', 'तीन', 'चार', 'पाँच', 'छह', 'सात', 'आठ', 'नौ',
    'दस', 'ग्यारह', 'बारह', 'तेरह', 'चौदह', 'पंद्रह', 'सोलह', 'सत्रह',
    'अठारह', 'उन्नीस', 'बीस', 'इक्कीस', 'बाईस', 'तेईस', 'चौबीस',
    'पच्चीस', 'छब्बीस', 'सत्ताईस', 'अट्ठाईस', 'उनतीस', 'तीस',
    'इकतीस', 'बत्तीस', 'तैंतीस', 'चौंतीस', 'पैंतीस', 'छत्तीस',
    'सैंतीस', 'अड़तीस', 'उनतालीस', 'चालीस', 'इकतालीस', 'बयालीस',
    'तैंतालीस', 'चवालीस', 'पैंतालीस', 'छियालीस', 'सैंतालीस',
    'अड़तालीस', 'उनचास', 'पचास', 'इक्यावन', 'बावन', 'तिरपन',
    'चौवन', 'पचपन', 'छप्पन', 'सत्तावन', 'अट्ठावन', 'उनसठ', 'साठ',
    'इकसठ', 'बासठ', 'तिरसठ', 'चौंसठ', 'पैंसठ', 'छियासठ', 'सड़सठ',
    'अड़सठ', 'उनहत्तर', 'सत्तर', 'इकहत्तर', 'बहत्तर', 'तिहत्तर',
    'चौहत्तर', 'पचहत्तर', 'छिहत्तर', 'सतहत्तर', 'अठहत्तर',
    'उनासी', 'अस्सी', 'इक्यासी', 'बयासी', 'तिरासी', 'चौरासी',
    'पचासी', 'छियासी', 'सत्तासी', 'अट्ठासी', 'नवासी', 'नब्बे',
    'इक्यानबे', 'बानबे', 'तिरानबे', 'चौरानबे', 'पचानबे',
    'छियानबे', 'सत्तानबे', 'अट्ठानबे', 'निन्यानबे'
]


def _number_to_hindi(n: int) -> str:
    """Convert integer to Hindi words. Handles up to crores."""
    if n < 0:
        return 'ऋण ' + _number_to_hindi(-n)
    if n == 0:
        return 'शून्य'
    if n < 100:
        return HINDI_ONES[n]
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        result = HINDI_ONES[hundreds] + ' सौ'
        if remainder > 0:
            result += ' ' + HINDI_ONES[remainder]
        return result
    if n < 100000:
        thousands = n // 1000
        remainder = n % 1000
        result = _number_to_hindi(thousands) + ' हज़ार'
        if remainder > 0:
            result += ' ' + _number_to_hindi(remainder)
        return result
    if n < 10000000:
        lakhs = n // 100000
        remainder = n % 100000
        result = _number_to_hindi(lakhs) + ' लाख'
        if remainder > 0:
            result += ' ' + _number_to_hindi(remainder)
        return result
    crores = n // 10000000
    remainder = n % 10000000
    result = _number_to_hindi(crores) + ' करोड़'
    if remainder > 0:
        result += ' ' + _number_to_hindi(remainder)
    return result


def _numbers_to_words(text: str) -> str:
    """Replace all digit sequences with Hindi words."""
    def replace_num(match):
        try:
            n = int(match.group())
            if n > 9999999999:  # too large, keep as-is
                return match.group()
            return _number_to_hindi(n)
        except:
            return match.group()
    return re.sub(r'\d+', replace_num, text)


def normalize_hindi_text(text: str) -> str:
    """
    Full normalization pipeline for TTS.
    Ported EXACTLY from tts_integrated.ipynb Cell 4.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Remove zero-width characters (ZWNJ, ZWJ)
    text = text.replace('\u200c', '').replace('\u200d', '')

    # 3. Remove control characters but keep newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 4. Normalize Purna Viram (Hindi full stop)
    text = text.replace('॥', '।')  # double danda → single

    # 5. Fix common OCR artifacts
    text = re.sub(r'[\|]{1,2}', '।', text)  # pipes → purna viram
    text = re.sub(r'\s*।\s*', '। ', text)   # consistent spacing around purna viram

    # 6. Convert Devanagari numerals to Arabic (for number-to-word)
    for deva, arabic in DEVA_DIGIT_MAP.items():
        text = text.replace(deva, arabic)

    # 7. Convert numbers to Hindi words
    text = _numbers_to_words(text)

    # 8. Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)    # collapse multiple spaces
    text = re.sub(r'\n\s*\n', '\n', text)  # collapse blank lines
    text = text.strip()

    return text
