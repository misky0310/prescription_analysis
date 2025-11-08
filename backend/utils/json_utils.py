"""
JSON formatting utilities
"""
import json
from typing import Dict


def standardize_json_response(raw_response: Dict, prescription_type: str) -> Dict:
    """
    Standardize the JSON response format from both Gemini and Groq
    """
    standard_format = {
        "prescription_type": prescription_type,
        "success": True,
        "medicines": [],
        "medical_terms": [],
        "additional_instructions": "",
        "raw_analysis": ""
    }
    
    try:
        # Handle different response formats
        if isinstance(raw_response, dict):
            # If already in our format
            if "medicines" in raw_response:
                standard_format["medicines"] = raw_response.get("medicines", [])
                standard_format["medical_terms"] = raw_response.get("medical_terms", [])
                standard_format["additional_instructions"] = raw_response.get("additional_instructions", "")
            else:
                # Store as raw if format is unknown
                standard_format["raw_analysis"] = json.dumps(raw_response, indent=2)
        elif isinstance(raw_response, str):
            standard_format["raw_analysis"] = raw_response
            
        # Ensure all medicines have required fields
        standardized_medicines = []
        for med in standard_format["medicines"]:
            standardized_med = {
                "name": med.get("name", "Unknown"),
                "dosage": med.get("dosage", "Not specified"),
                "frequency": med.get("frequency", "Not specified"),
                "duration": med.get("duration", "Not specified"),
                "explanation": med.get("explanation", "No explanation available"),
                "side_effects": med.get("side_effects", "Not specified"),
                "precautions": med.get("precautions", "Not specified")
            }
            standardized_medicines.append(standardized_med)
        
        standard_format["medicines"] = standardized_medicines
        
    except Exception as e:
        standard_format["success"] = False
        standard_format["error"] = str(e)
    
    return standard_format