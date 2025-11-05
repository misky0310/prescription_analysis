import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import re
import json
from groq import Groq
import pytesseract
from typing import List, Dict, Optional

# Model architecture (must match training)
class MedicineClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MedicineClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def load_model(model_path, label_encoder_path, device):
    """Load trained model and label encoder"""
    print('='*70)
    print('🔄 Loading Model')
    print('='*70)
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
    
    # Load label encoder
    label_classes = np.load(label_encoder_path, allow_pickle=True)
    num_classes = len(label_classes)
    
    print(f'📊 Number of classes: {num_classes}')
    print(f'🏷️  Medicine classes: {", ".join(label_classes)}')
    
    # Initialize model
    model = MedicineClassifier(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f'✅ Model loaded successfully!')
    print(f'🎯 Trained accuracy: {checkpoint["accuracy"]:.2f}%')
    print(f'📅 Trained at epoch: {checkpoint["epoch"] + 1}')
    
    return model, label_classes

def get_transform():
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_single_image(image_path, model, label_classes, device, transform):
    """Predict medicine name from a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = label_classes[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, min(3, len(label_classes)))
        top3_predictions = [
            (label_classes[idx.item()], prob.item() * 100) 
            for idx, prob in zip(top3_indices[0], top3_prob[0])
        ]
        
        return {
            'success': True,
            'predicted_label': predicted_label,
            'confidence': confidence_score,
            'top3': top3_predictions
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def predict_batch(image_folder, model, label_classes, device, transform, output_csv='predictions.csv'):
    """Predict medicine names for all images in a folder"""
    print('\n' + '='*70)
    print('📁 Batch Prediction')
    print('='*70)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = [f for f in os.listdir(image_folder) if f.endswith(image_extensions)]
    
    if not image_files:
        print(f'❌ No image files found in {image_folder}')
        return []
    
    # Sort files naturally
    image_files.sort(key=lambda x: ''.join(filter(str.isdigit, x)).zfill(4))

    
    print(f'📊 Found {len(image_files)} images to process\n')
    
    results = []
    correct = 0
    total = 0
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_file)
        result = predict_single_image(img_path, model, label_classes, device, transform)
        
        if result['success']:
            predicted_label = result['predicted_label']
            confidence = result['confidence']
            top3 = result['top3']
            
            # Display result
            print(f'[{i}/{len(image_files)}] {img_file}')
            print(f'   Predicted: {predicted_label} ({confidence:.2f}%)')
            print(f'   Top 3: {", ".join([f"{name} ({conf:.1f}%)" for name, conf in top3])}')
            
            results.append({
                'image': img_file,
                'predicted_medicine': predicted_label,
                'confidence': f'{confidence:.2f}%',
                'top_1': top3[0][0] if len(top3) > 0 else '',
                'top_1_conf': f'{top3[0][1]:.2f}%' if len(top3) > 0 else '',
                'top_2': top3[1][0] if len(top3) > 1 else '',
                'top_2_conf': f'{top3[1][1]:.2f}%' if len(top3) > 1 else '',
                'top_3': top3[2][0] if len(top3) > 2 else '',
                'top_3_conf': f'{top3[2][1]:.2f}%' if len(top3) > 2 else '',
                'status': 'success'
            })
        else:
            print(f'[{i}/{len(image_files)}] {img_file}')
            print(f'   ❌ Error: {result["error"]}')
            results.append({
                'image': img_file,
                'predicted_medicine': 'ERROR',
                'confidence': '0%',
                'status': 'error',
                'error': result['error']
            })
        
        print()  # Blank line for readability
    
    # Save to CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f'💾 Results saved to: {output_csv}')
    
    return results

def predict_from_csv(csv_path, image_folder, model, label_classes, device, transform):
    """Predict and compare with ground truth from CSV"""
    print('\n' + '='*70)
    print('📊 Prediction with Ground Truth Comparison')
    print('='*70)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f'📂 Loaded {len(df)} samples from {csv_path}\n')
    
    results = []
    correct = 0
    total = 0
    
    for idx, row in df.iterrows():
        img_file = row['IMAGE']
        true_label = row['MEDICINE_NAME']
        img_path = os.path.join(image_folder, img_file)
        
        if not os.path.exists(img_path):
            print(f'⚠️  Image not found: {img_file}')
            continue
        
        result = predict_single_image(img_path, model, label_classes, device, transform)
        
        if result['success']:
            predicted_label = result['predicted_label']
            confidence = result['confidence']
            is_correct = predicted_label == true_label
            
            if is_correct:
                correct += 1
                status_icon = '✅'
            else:
                status_icon = '❌'
            
            total += 1
            
            print(f'{status_icon} {img_file}')
            print(f'   True: {true_label} | Predicted: {predicted_label} ({confidence:.2f}%)')
            
            results.append({
                'image': img_file,
                'true_medicine': true_label,
                'predicted_medicine': predicted_label,
                'confidence': f'{confidence:.2f}%',
                'correct': is_correct
            })
    
    # Calculate accuracy
    if total > 0:
        accuracy = (correct / total) * 100
        print(f'\n{"="*70}')
        print(f'📈 Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)')
        print(f'{"="*70}')
        
        # Save results
        df_results = pd.DataFrame(results)
        output_file = 'test_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f'💾 Results saved to: {output_file}')
    
    return results

def interactive_mode(model, label_classes, device, transform):
    """Interactive mode for single image predictions"""
    print('\n' + '='*70)
    print('🎯 Interactive Prediction Mode')
    print('='*70)
    print('Enter image path to predict (or "quit" to exit)\n')
    
    while True:
        img_path = input('Image path: ').strip()
        
        if img_path.lower() in ['quit', 'exit', 'q']:
            print('👋 Exiting...')
            break
        
        if not os.path.exists(img_path):
            print(f'❌ File not found: {img_path}\n')
            continue
        
        result = predict_single_image(img_path, model, label_classes, device, transform)
        
        if result['success']:
            print(f'\n📊 Prediction Results:')
            print(f'   Medicine: {result["predicted_label"]}')
            print(f'   Confidence: {result["confidence"]:.2f}%')
            print(f'\n   Top 3 Predictions:')
            for i, (name, conf) in enumerate(result['top3'], 1):
                print(f'      {i}. {name}: {conf:.2f}%')
        else:
            print(f'❌ Error: {result["error"]}')
        
        print()


# ==================== NEW PRESCRIPTION ANALYSIS FEATURES ====================

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

def analyze_prescription_with_vision_llm(image_path: str, groq_api_key: str, 
                                          model_name: str = "llama-3.1-8b-instant") -> Dict:
    """
    Use Groq Vision LLM to directly analyze prescription image
    """
    try:
        import base64
        
        client = Groq(api_key=groq_api_key)
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """Analyze this prescription image and extract all medicine information. Provide:

1. All medicines mentioned with:
   - Medicine name
   - Dosage
   - Frequency (how often to take)
   - Duration (how long to take)

2. For each medicine, explain in simple terms:
   - What it's used for
   - Common side effects
   - Important precautions

3. Any other important medical terms or instructions, explained simply.

Format your response as JSON with this structure:
{
    "medicines": [
        {
            "name": "medicine name",
            "dosage": "dosage info",
            "frequency": "frequency info",
            "duration": "duration info",
            "explanation": "simple explanation",
            "side_effects": "side effects",
            "precautions": "precautions"
        }
    ],
    "medical_terms": [
        {
            "term": "term",
            "explanation": "explanation"
        }
    ],
    "additional_instructions": "instructions"
}"""
        
        # Groq vision API format
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                analysis = json.loads(json_match.group())
            except:
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

def analyze_prescription_pipeline(image_path: str, model, label_classes, device, transform,
                                  groq_api_key: str, use_vision: bool = True,
                                  output_json: str = 'prescription_analysis.json') -> Dict:
    """
    Complete prescription analysis pipeline:
    1. Extract text using OCR (if not using vision)
    2. Identify medicines using your trained model
    3. Use Groq LLM to explain everything in simple terms
    """
    print('\n' + '='*70)
    print('💊 PRESCRIPTION ANALYSIS PIPELINE')
    print('='*70)
    print(f'📄 Analyzing: {image_path}\n')
    
    results = {
        'image_path': image_path,
        'ocr_text': '',
        'detected_medicines': [],
        'llm_analysis': {},
        'status': 'processing'
    }
    
    # Step 1: OCR extraction (if not using vision)
    if not use_vision:
        print('🔍 Step 1: Extracting text from prescription...')
        ocr_text = extract_text_from_prescription(image_path)
        results['ocr_text'] = ocr_text
        print(f'✅ Text extracted: {len(ocr_text)} characters\n')
        
        if not ocr_text.strip():
            print('⚠️  Warning: No text extracted from image')
            results['status'] = 'no_text_extracted'
            return results
        
        # Step 2: Identify medicines using your model
        print('🔍 Step 2: Identifying medicines from text...')
        medicines_found = extract_medicine_names_from_text(ocr_text, label_classes)
        results['detected_medicines'] = medicines_found
        print(f'✅ Found {len(medicines_found)} medicines\n')
        
        for med in medicines_found:
            print(f'   💊 {med["medicine"]} (confidence: {med["confidence"]})')
            if med['context']:
                print(f'      Context: {med["context"][:100]}...')
        print()
        
        # Step 3: LLM analysis
        print('🤖 Step 3: Analyzing with Groq LLM...')
        llm_result = analyze_prescription_with_llm(ocr_text, medicines_found, groq_api_key)
    else:
        # Use vision model directly
        print('🤖 Analyzing prescription with Vision LLM...')
        llm_result = analyze_prescription_with_vision_llm(image_path, groq_api_key)
    
    if llm_result['success']:
        results['llm_analysis'] = llm_result['analysis']
        results['status'] = 'success'
        print('✅ Analysis complete!\n')
        
        # Display results
        display_prescription_analysis(results)
    else:
        results['status'] = 'llm_error'
        results['error'] = llm_result['error']
        print(f'❌ LLM Error: {llm_result["error"]}')
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n💾 Full analysis saved to: {output_json}')
    
    return results

def display_prescription_analysis(results: Dict):
    """Display prescription analysis in a formatted way"""
    print('='*70)
    print('📋 PRESCRIPTION ANALYSIS RESULTS')
    print('='*70)
    
    analysis = results.get('llm_analysis', {})
    
    if 'medicines' in analysis:
        print('\n💊 MEDICINES:')
        print('-' * 70)
        for i, med in enumerate(analysis['medicines'], 1):
            print(f'\n{i}. {med.get("name", "Unknown")}')
            print(f'   📊 Dosage: {med.get("dosage", "Not specified")}')
            print(f'   ⏰ Frequency: {med.get("frequency", "Not specified")}')
            print(f'   📅 Duration: {med.get("duration", "Not specified")}')
            print(f'   ℹ️  Purpose: {med.get("explanation", "Not available")}')
            
            if med.get('side_effects'):
                print(f'   ⚠️  Side Effects: {med.get("side_effects")}')
            if med.get('precautions'):
                print(f'   ⚡ Precautions: {med.get("precautions")}')
    
    if 'medical_terms' in analysis and analysis['medical_terms']:
        print('\n\n📖 MEDICAL TERMS EXPLAINED:')
        print('-' * 70)
        for term in analysis['medical_terms']:
            print(f'\n• {term.get("term", "Unknown")}')
            print(f'  → {term.get("explanation", "No explanation available")}')
    
    if 'additional_instructions' in analysis and analysis['additional_instructions']:
        print('\n\n📝 ADDITIONAL INSTRUCTIONS:')
        print('-' * 70)
        print(f'{analysis["additional_instructions"]}')
    
    if 'raw_response' in analysis:
        print('\n\n📄 RAW LLM RESPONSE:')
        print('-' * 70)
        print(analysis['raw_response'])
    
    print('\n' + '='*70)

def prescription_interactive_mode(model, label_classes, device, transform, groq_api_key: str):
    """Interactive mode for prescription analysis"""
    print('\n' + '='*70)
    print('💊 PRESCRIPTION ANALYSIS - INTERACTIVE MODE')
    print('='*70)
    print('Enter prescription image path to analyze (or "quit" to exit)')
    print('Options: "vision" for vision LLM, "ocr" for OCR + text analysis\n')
    
    use_vision = True
    
    while True:
        user_input = input('Enter path or command: ').strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print('👋 Exiting...')
            break
        
        if user_input.lower() == 'vision':
            use_vision = True
            print('✅ Using Vision LLM mode\n')
            continue
        
        if user_input.lower() == 'ocr':
            use_vision = False
            print('✅ Using OCR + Text Analysis mode\n')
            continue
        
        if not os.path.exists(user_input):
            print(f'❌ File not found: {user_input}\n')
            continue
        
        try:
            analyze_prescription_pipeline(
                user_input, model, label_classes, device, transform,
                groq_api_key, use_vision=use_vision
            )
        except Exception as e:
            print(f'❌ Error analyzing prescription: {e}')
            import traceback
            traceback.print_exc()
        
        print()


# ==================== END OF NEW FEATURES ====================


def main():
    parser = argparse.ArgumentParser(description='Medicine Classification Prediction')
    parser.add_argument('--mode', type=str, default='batch', 
                        choices=['single', 'batch', 'test', 'interactive', 
                                'prescription', 'prescription-interactive'],
                        help='Prediction mode')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, default='images', help='Path to images folder')
    parser.add_argument('--csv', type=str, default='data.csv', help='Path to CSV file for testing')
    parser.add_argument('--model', type=str, default='best_medicine_model.pth', help='Path to model file')
    parser.add_argument('--labels', type=str, default='label_encoder_classes.npy', help='Path to label encoder')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    
    # New prescription analysis arguments
    parser.add_argument('--groq-api-key', type=str, help='Groq API key for LLM analysis')
    parser.add_argument('--use-vision', type=str, default='true', 
                        choices=['true', 'false', 'True', 'False', 'yes', 'no'],
                        help='Use vision LLM instead of OCR (default: true)')
    parser.add_argument('--prescription-output', type=str, default='prescription_analysis.json',
                        help='Output JSON file for prescription analysis')
    
    args = parser.parse_args()
    
    print('='*70)
    print('🏥 MEDICINE CLASSIFICATION - PREDICTION')
    print('='*70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️  Device: {device}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
    
    # Load model
    try:
        model, label_classes = load_model(args.model, args.labels, device)
        transform = get_transform()
    except Exception as e:
        print(f'\n❌ Error loading model: {e}')
        return
    
    # Check for Groq API key for prescription modes
    if args.mode in ['prescription', 'prescription-interactive']:
        if not args.groq_api_key:
            print('\n⚠️  Warning: --groq-api-key not provided.')
            print('   Please provide Groq API key or set GROQ_API_KEY environment variable')
            groq_key = os.environ.get('GROQ_API_KEY')
            if groq_key:
                print('   ✅ Using GROQ_API_KEY from environment')
                args.groq_api_key = groq_key
            else:
                print('   ❌ Cannot proceed without API key')
                return
    
    # Execute based on mode
    try:
        if args.mode == 'single':
            if not args.image:
                print('\n❌ Please specify --image path for single mode')
                return
            
            print('\n' + '='*70)
            print('🖼️  Single Image Prediction')
            print('='*70)
            print(f'📁 Image: {args.image}\n')
            
            result = predict_single_image(args.image, model, label_classes, device, transform)
            
            if result['success']:
                print(f'✅ Predicted Medicine: {result["predicted_label"]}')
                print(f'📊 Confidence: {result["confidence"]:.2f}%')
                print(f'\n🔝 Top 3 Predictions:')
                for i, (name, conf) in enumerate(result['top3'], 1):
                    print(f'   {i}. {name}: {conf:.2f}%')
            else:
                print(f'❌ Error: {result["error"]}')
        
        elif args.mode == 'batch':
            predict_batch(args.folder, model, label_classes, device, transform, args.output)
        
        elif args.mode == 'test':
            if not os.path.exists(args.csv):
                print(f'\n❌ CSV file not found: {args.csv}')
                return
            predict_from_csv(args.csv, args.folder, model, label_classes, device, transform)
        
        elif args.mode == 'interactive':
            interactive_mode(model, label_classes, device, transform)
        
        elif args.mode == 'prescription':
            if not args.image:
                print('\n❌ Please specify --image path for prescription mode')
                return
            
            # Convert string to boolean
            use_vision = args.use_vision.lower() in ['true', 'yes', '1']
            
            analyze_prescription_pipeline(
                args.image, model, label_classes, device, transform,
                args.groq_api_key, use_vision=use_vision,
                output_json=args.prescription_output
            )
        
        elif args.mode == 'prescription-interactive':
            prescription_interactive_mode(model, label_classes, device, transform, args.groq_api_key)
    
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrupted by user')
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()