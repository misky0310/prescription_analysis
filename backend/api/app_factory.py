"""
Flask app factory
"""
from flask import Flask
from flask_cors import CORS
import config.settings as settings


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = settings.MAX_CONTENT_LENGTH
    
    # Register routes
    from api.routes import register_routes
    register_routes(app)
    
    return app