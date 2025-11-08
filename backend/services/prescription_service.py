"""
Prescription analysis service
"""
import json
import os
from typing import Dict
from services.llm_service import (
    analyze_handwritten_prescription_gemini,
    analyze_printed_prescription_groq
)
from utils.json_utils import standardize_json_response
from utils.display_utils import display_prescription_analysis
import config.settings as settings


def analyze_prescription_complete(image_path: str, prescription_type: str = 'handwritten',
                                  output_json: str = None) -> Dict:
    """
    Complete prescription analysis with consistent JSON output
    prescription_type: 'handwritten' or 'printed'
    """
    print('\n' + '='*70)
    print('💊 PRESCRIPTION ANALYSIS')
    print('='*70)
    print(f'📄 Image: {image_path}')
    print(f'📝 Type: {prescription_type.upper()}\n')
    
    # Get API keys from environment
    gemini_key = settings.GEMINI_API_KEY
    groq_key = settings.GROQ_API_KEY
    
    if prescription_type.lower() == 'handwritten':
        if not gemini_key:
            return {
                'success': False,
                'error': 'GEMINI_API_KEY not found in environment'
            }
        
        print('🤖 Using Gemini for handwritten prescription...')
        result = analyze_handwritten_prescription_gemini(image_path, gemini_key)
    else:
        if not groq_key:
            return {
                'success': False,
                'error': 'GROQ_API_KEY not found in environment'
            }
        
        print('🤖 Using Groq for printed prescription...')
        result = analyze_printed_prescription_groq(image_path, groq_key)
    
    if not result['success']:
        return result
    
    # Standardize response
    standardized = standardize_json_response(result['analysis'], prescription_type)
    
    print('✅ Analysis complete!\n')
    display_prescription_analysis(standardized)
    
    # Save to JSON if output path provided
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(standardized, f, indent=2)
        print(f'\n💾 Full analysis saved to: {output_json}')
    
    return standardized