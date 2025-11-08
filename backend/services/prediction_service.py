"""
Medicine prediction services
"""
import torch
import os
import pandas as pd
from PIL import Image


def predict_single_image(image_path, model, label_classes, device, transform):
    """Predict medicine name from a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_label = label_classes[predicted.item()]
        confidence_score = confidence.item() * 100
        
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
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = [f for f in os.listdir(image_folder) if f.endswith(image_extensions)]
    
    if not image_files:
        print(f'❌ No image files found in {image_folder}')
        return []
    
    image_files.sort(key=lambda x: ''.join(filter(str.isdigit, x)).zfill(4))
    print(f'📊 Found {len(image_files)} images to process\n')
    
    results = []
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_file)
        result = predict_single_image(img_path, model, label_classes, device, transform)
        
        if result['success']:
            predicted_label = result['predicted_label']
            confidence = result['confidence']
            top3 = result['top3']
            
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
        
        print()
    
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
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f'\n{"="*70}')
        print(f'📈 Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)')
        print(f'{"="*70}')
        
        df_results = pd.DataFrame(results)
        output_file = 'test_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f'💾 Results saved to: {output_file}')
    
    return results