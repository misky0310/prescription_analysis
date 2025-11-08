"""
LLM integration services (Gemini and Groq)
"""
import os
import re
import json
from typing import List, Dict
from groq import Groq
from google import genai
from google.genai import types
from services.ocr_service import extract_text_from_prescription, extract_medicine_names_from_text
import config.settings as settings


def analyze_handwritten_prescription_gemini(image_path: str, gemini_api_key: str) -> Dict:
    """
    Analyze handwritten prescription using Gemini Vision API
    """
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Determine MIME type
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        prompt = """Analyze this prescription image (handwritten or printed) and extract medicine information.

Please provide your response in this EXACT JSON format:
{
    "medicines": [
        {
            "name": "medicine name",
            "dosage": "dosage (e.g., 500mg, 10ml)",
            "frequency": "how often (e.g., twice daily, once at night)",
            "duration": "how long (e.g., 7 days, 2 weeks)",
            "explanation": "what this medicine is used for in simple terms",
            "side_effects": "common side effects patients should know",
            "precautions": "important warnings or things to avoid"
        }
    ],
    "medical_terms": [
        {
            "term": "medical term found",
            "explanation": "simple explanation"
        }
    ],
    "additional_instructions": "any other important instructions for the patient"
}

Make sure to:
- Extract ALL medicines visible in the prescription
- Provide clear, patient-friendly explanations
- Include dosage and frequency for each medicine
- Explain medical terms in simple language
- Return valid JSON only"""
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt
            ]
        )
        
        response_text = response.text
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group())
            except json.JSONDecodeError:
                analysis = {"raw_response": response_text}
        else:
            analysis = {"raw_response": response_text}
        
        return {
            'success': True,
            'analysis': analysis
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def analyze_printed_prescription_groq(image_path: str, groq_api_key: str) -> Dict:
    """
    Analyze a printed or handwritten prescription image using OCR + Groq LLM.
    Extracts text using OCR, identifies medicines, and generates a structured analysis.
    """
    try:
        print("📄 Extracting text from prescription image using OCR...")
        ocr_text = extract_text_from_prescription(image_path)

        if not ocr_text.strip():
            return {
                'success': False,
                'error': 'OCR failed to extract any text from the prescription.'
            }

        print("🧠 Identifying possible medicine names...")
        medicines_found = extract_medicine_names_from_text(ocr_text, settings.label_classes)

        print("🤖 Sending extracted text to Groq LLM for structured medical analysis...")

        prompt = f"""Analyze this prescription image (handwritten or printed) and extract medicine information.

Prescription Text:
{ocr_text}

Detected Medicines (based on known labels): {', '.join([m['medicine'] for m in medicines_found]) if medicines_found else 'None detected'}

Please provide your response in this EXACT JSON format:
{{
    "medicines": [
        {{
            "name": "medicine name",
            "dosage": "dosage (e.g., 500mg, 10ml)",
            "frequency": "how often (e.g., twice daily, once at night)",
            "duration": "how long (e.g., 7 days, 2 weeks)",
            "explanation": "what this medicine is used for in simple terms",
            "side_effects": "common side effects patients should know",
            "precautions": "important warnings or things to avoid"
        }}
    ],
    "medical_terms": [
        {{
            "term": "medical term found",
            "explanation": "simple explanation"
        }}
    ],
    "additional_instructions": "any other important instructions for the patient"
}}

Make sure to:
- Extract ALL medicines visible in the prescription
- Provide clear, patient-friendly explanations
- Include dosage and frequency for each medicine
- Explain medical terms in simple language
- Return valid JSON only.
"""

        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant that explains prescriptions in simple language for patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )

        response_text = response.choices[0].message.content.strip()

        # Try extracting JSON content safely
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group())
            except json.JSONDecodeError:
                analysis = {"raw_response": response_text}
        else:
            analysis = {"raw_response": response_text}

        return {
            "success": True,
            "ocr_text": ocr_text,
            "analysis": analysis
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def analyze_prescription_with_llm(prescription_text: str, medicines_found: List[Dict], 
                                  groq_api_key: str, model_name: str = "llama-3.1-8b-instant") -> Dict:
    """
    Use Groq LLM to analyze prescription and extract structured information
    """
    try:
        client = Groq(api_key=groq_api_key)
        
        medicines_list = [m['medicine'] for m in medicines_found]
        
        prompt = f"""You are a medical prescription analyzer. Analyze the following prescription text and extract information about the medicines mentioned.

Prescription Text:
{prescription_text}

Detected Medicines: {', '.join(medicines_list) if medicines_list else 'None detected'}

Please provide:
1. For each medicine mentioned:
   - Medicine name
   - Dosage (if mentioned)
   - Frequency (e.g., "twice daily", "once in morning")
   - Duration (if mentioned)

2. Simplified explanation of each medicine:
   - What it's used for (in simple terms)
   - Common side effects (if any)
   - Important precautions (if any)

3. Any other important medical terms or instructions in the prescription, explained in simple language.

Format your response as JSON with this structure:
{{
    "medicines": [
        {{
            "name": "medicine name",
            "dosage": "dosage info",
            "frequency": "frequency info",
            "duration": "duration info",
            "explanation": "simple explanation of what it's for",
            "side_effects": "common side effects",
            "precautions": "important precautions"
        }}
    ],
    "medical_terms": [
        {{
            "term": "medical term",
            "explanation": "simple explanation"
        }}
    ],
    "additional_instructions": "any other important instructions"
}}
"""
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant that explains prescriptions in simple terms for patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse JSON response
        response_text = response.choices[0].message.content
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = {"raw_response": response_text}
        
        return {
            'success': True,
            'analysis': analysis
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }