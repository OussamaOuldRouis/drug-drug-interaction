#!/usr/bin/env python
"""Run the Drug Interaction System web application."""

import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from web.app import run_app

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("drug_interaction_system/web/static/visualizations", exist_ok=True)
    
    # Run the web application
    print("Starting Drug Interaction System web application...")
    print("Open your browser and navigate to http://localhost:5000")
    run_app() 