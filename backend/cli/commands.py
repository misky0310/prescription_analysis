"""
CLI command handlers
"""
import os
from services.prediction_service import predict_single_image


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