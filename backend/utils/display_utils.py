"""
Display and console output utilities
"""
from typing import Dict


def display_prescription_analysis(results: Dict):
    """Display prescription analysis in a formatted way"""
    print('='*70)
    print('📋 PRESCRIPTION ANALYSIS RESULTS')
    print('='*70)
    
    if 'medicines' in results and results['medicines']:
        print('\n💊 MEDICINES:')
        print('-' * 70)
        for i, med in enumerate(results['medicines'], 1):
            print(f'\n{i}. {med.get("name", "Unknown")}')
            print(f'   📊 Dosage: {med.get("dosage", "Not specified")}')
            print(f'   ⏰ Frequency: {med.get("frequency", "Not specified")}')
            print(f'   📅 Duration: {med.get("duration", "Not specified")}')
            print(f'   ℹ️  Purpose: {med.get("explanation", "Not available")}')
            
            if med.get('side_effects') and med['side_effects'] != "Not specified":
                print(f'   ⚠️  Side Effects: {med.get("side_effects")}')
            if med.get('precautions') and med['precautions'] != "Not specified":
                print(f'   ⚡ Precautions: {med.get("precautions")}')
    
    if 'medical_terms' in results and results['medical_terms']:
        print('\n\n📖 MEDICAL TERMS EXPLAINED:')
        print('-' * 70)
        for term in results['medical_terms']:
            print(f'\n• {term.get("term", "Unknown")}')
            print(f'  → {term.get("explanation", "No explanation available")}')
    
    if 'additional_instructions' in results and results['additional_instructions']:
        print('\n\n📝 ADDITIONAL INSTRUCTIONS:')
        print('-' * 70)
        print(f'{results["additional_instructions"]}')
    
    if 'raw_analysis' in results and results['raw_analysis']:
        print('\n\n📄 RAW RESPONSE:')
        print('-' * 70)
        print(results['raw_analysis'][:500] + '...' if len(results['raw_analysis']) > 500 else results['raw_analysis'])
    
    print('\n' + '='*70)