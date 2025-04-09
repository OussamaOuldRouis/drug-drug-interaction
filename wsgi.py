#!/usr/bin/env python
"""WSGI entry point for the Drug Interaction System web application."""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the create_app function from the app module
from src.web.app import create_app

# Create the application
application = create_app()

if __name__ == "__main__":
    # This is for local development only
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port) 