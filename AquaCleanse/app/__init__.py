# This is the __init__.py file for the app package.
# It is used to mark this directory as a Python package and can be used to initialize app-wide settings or imports.
# Commonly, you might import Flask and set up the app instance here, or import key modules for use elsewhere in the app.
from flask import Flask
from database import get_users_collection,get_admins_collection, get_canal_data_collection

def create_app():
    app = Flask(__name__)  # Default Flask structure (Flask auto-detects templates and static)
    app.secret_key = 'your-secret-key-here'

    app.config['USERS_COLLECTION'] = get_users_collection()
    app.config['ADMINS_COLLECTION'] = get_admins_collection()
    app.config['CANAL_DATA_COLLECTION'] = get_canal_data_collection()

    from app.routes import init_routes
    init_routes(app)

    return app
