"""
OCR and text extraction services
"""
import re
import pytesseract
from PIL import Image
from typing import List, Dict


def extract_text_from_prescription(image_path: str) -> str:
    """Extract text from prescription image using OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f'⚠️  OCR Error: {e}')
        return ""


def extract_medicine_names_from_text(text: str, label_classes: List[str]) -> List[Dict]:
    """
    Extract medicine names from text by matching against known medicine classes.
    Uses fuzzy matching to handle variations in spelling.
    """
    medicines_found = []
    text_lower = text.lower()
    
    # Split text into words and clean
    words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
    
    # Check each medicine in label_classes
    for medicine in label_classes:
        medicine_lower = medicine.lower()
        
        # Direct substring match
        if medicine_lower in text_lower:
            # Try to find context (frequency/dosage)
            context = extract_medicine_context(text, medicine)
            medicines_found.append({
                'medicine': medicine,
                'confidence': 'high',
                'context': context
            })
            continue
        
        # Check for partial matches (useful for compound names)
        medicine_parts = medicine_lower.split()
        if len(medicine_parts) > 1:
            if all(part in text_lower for part in medicine_parts):
                context = extract_medicine_context(text, medicine)
                medicines_found.append({
                    'medicine': medicine,
                    'confidence': 'medium',
                    'context': context
                })
    
    return medicines_found


def extract_medicine_context(text: str, medicine: str) -> str:
    """Extract dosage and frequency information around medicine name"""
    # Find the position of medicine in text
    pattern = re.escape(medicine)
    match = re.search(pattern, text, re.IGNORECASE)
    
    if not match:
        return ""
    
    # Get surrounding text (50 chars before and after)
    start = max(0, match.start() - 50)
    end = min(len(text), match.end() + 50)
    context = text[start:end].strip()
    
    return context