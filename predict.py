import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description='Medicine Classification Prediction')
    parser.add_argument('--mode', type=str, default='batch', 
                        choices=['single', 'batch', 'test', 'interactive'],
                        help='Prediction mode')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, default='images', help='Path to images folder')
    parser.add_argument('--csv', type=str, default='data.csv', help='Path to CSV file for testing')
    parser.add_argument('--model', type=str, default='best_medicine_model.pth', help='Path to model file')
    parser.add_argument('--labels', type=str, default='label_encoder_classes.npy', help='Path to label encoder')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file')
    
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
    
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrupted by user')
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()