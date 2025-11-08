"""
Main application entry point
"""
import torch
import argparse
import os
import traceback

# Import configuration
import config.settings as settings

# Import models
from models.model_loader import load_model, get_transform

# Import services
from services.prediction_service import predict_single_image, predict_batch, predict_from_csv
from services.prescription_service import analyze_prescription_complete

# Import CLI commands
from cli.commands import interactive_mode

# Import API
from api.app_factory import create_app
from api.routes import set_global_model


def main():
    parser = argparse.ArgumentParser(description='Medicine Classification & Prescription Analysis')
    parser.add_argument('--mode', type=str, default='batch', 
                        choices=['single', 'batch', 'test', 'interactive', 
                                'prescription', 'server'],
                        help='Operation mode')
    parser.add_argument('--image', type=str, help='Path to image')
    parser.add_argument('--folder', type=str, default='images', help='Path to images folder')
    parser.add_argument('--csv', type=str, default='data.csv', help='Path to CSV file')
    parser.add_argument('--model', type=str, default=settings.DEFAULT_MODEL_PATH, help='Model file')
    parser.add_argument('--labels', type=str, default=settings.DEFAULT_LABELS_PATH, help='Label encoder')
    parser.add_argument('--output', type=str, default=settings.DEFAULT_OUTPUT_CSV, help='Output CSV')
    parser.add_argument('--prescription-type', type=str, default='handwritten',
                        choices=['handwritten', 'printed'], help='Prescription type')
    parser.add_argument('--prescription-output', type=str, default=settings.DEFAULT_PRESCRIPTION_OUTPUT,
                        help='Output JSON for prescription')
    parser.add_argument('--port', type=int, default=settings.DEFAULT_PORT, help='Server port')
    
    args = parser.parse_args()
    
    print('='*70)
    print('🏥 MEDICINE CLASSIFICATION & PRESCRIPTION ANALYZER')
    print('='*70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥️  Device: {device}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
    
    # Server mode doesn't require model loading initially
    if args.mode == 'server':
        print('\n🚀 Starting Flask API Server...')
        print(f'📡 Server will run on http://localhost:{args.port}')
        print('\nAvailable endpoints:')
        print('  - GET  /health')
        print('  - POST /api/analyze-prescription')
        print('  - POST /api/predict-medicine')
        print('\nPress Ctrl+C to stop the server\n')
        
        # Try to load model for medicine prediction endpoint
        try:
            model, label_classes = load_model(args.model, args.labels, device)
            transform = get_transform()
            set_global_model(model, label_classes, device, transform)
            print('✅ Model loaded for medicine prediction endpoint\n')
        except Exception as e:
            print(f'⚠️  Model not available: {e}')
            print('   📌 Prescription analysis will work')
            print('   ❌ Medicine prediction endpoint will be disabled\n')
        
        # Create and run Flask app
        app = create_app()
        app.run(debug=False, host=settings.DEFAULT_HOST, port=args.port)  # Changed debug=False
        return
    
    # Load model for other modes
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
                print('\n❌ Please specify --image path')
                return
            
            print('\n' + '='*70)
            print('🖼️  Single Image Prediction')
            print('='*70)
            
            result = predict_single_image(args.image, model, label_classes, device, transform)
            
            if result['success']:
                print(f'\n✅ Predicted Medicine: {result["predicted_label"]}')
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
                print('\n❌ Please specify --image path')
                return
            
            analyze_prescription_complete(
                args.image, 
                args.prescription_type,
                args.prescription_output
            )
    
    except KeyboardInterrupt:
        print('\n\n⚠️  Interrupted by user')
    except Exception as e:
        print(f'\n❌ Error: {e}')
        traceback.print_exc()


if __name__ == '__main__':
    main()