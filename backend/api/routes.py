"""
Flask API routes
"""
import os
import traceback
from flask import request, jsonify
from werkzeug.utils import secure_filename
from utils.file_utils import allowed_file
from services.prescription_service import analyze_prescription_complete
from services.prediction_service import predict_single_image

# Global variables for model (will be set from main)
global_model = None
global_label_classes = None
global_device = None
global_transform = None


def set_global_model(model, label_classes, device, transform):
    """Set global model variables"""
    global global_model, global_label_classes, global_device, global_transform
    global_model = model
    global_label_classes = label_classes
    global_device = device
    global_transform = transform


def register_routes(app):
    """Register all API routes"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy', 'message': 'Medicine Analyzer API is running'})

    @app.route('/api/analyse-prescription', methods=['POST'])
    def analyze_prescription_api():
        """API endpoint for prescription analysis"""
        try:
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image file provided'}), 400
            
            file = request.files['image']
            prescription_type = request.form.get('type', 'handwritten')
            
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze prescription
            result = analyze_prescription_complete(filepath, prescription_type)
            
            # Clean up
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

    @app.route('/api/predict-medicine', methods=['POST'])
    def predict_medicine_api():
        """API endpoint for single medicine prediction"""
        try:
            if not global_model:
                return jsonify({'success': False, 'error': 'Model not loaded'}), 500
            
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image file provided'}), 400
            
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type'}), 400
            
            # Save and predict
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = predict_single_image(filepath, global_model, global_label_classes, 
                                         global_device, global_transform)
            
            # Clean up
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500