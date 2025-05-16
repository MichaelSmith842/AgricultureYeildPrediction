#!/usr/bin/env python3
"""
Setup script for the Agricultural Yield Climate Impact Analysis System.
Helps users set up the environment and run the application.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check that Python version is 3.7 or higher."""
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def setup_virtual_environment():
    """Create and set up a virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("Virtual environment already exists.")
        return
    
    logger.info("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        sys.exit(1)

def install_dependencies():
    """Install dependencies from requirements.txt."""
    logger.info("Installing dependencies...")

    if sys.platform == "win32":
        python_path = Path("venv") / "Scripts" / "python.exe"
    else:
        python_path = Path("venv") / "bin" / "python"

    if not python_path.exists():
        logger.error(f"Could not find python at {python_path}")
        sys.exit(1)

    try:
        subprocess.run([str(python_path), "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.warning("Some features may not work correctly.")

def create_sample_data():
    """Create sample data files if they don't exist."""
    logger.info("Checking for sample data...")
    
    # Determine path to python in the virtual environment
    if sys.platform == "win32":
        python_path = Path("venv") / "Scripts" / "python.exe"
    else:
        python_path = Path("venv") / "bin" / "python"
    
    if not python_path.exists():
        logger.error(f"Could not find python at {python_path}")
        sys.exit(1)
    
    # Create sample data directory
    sample_dir = Path("data") / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if USDA sample data exists
    usda_sample = sample_dir / "usda_crop_data_sample.csv"
    
    if not usda_sample.exists():
        logger.info("Creating USDA sample data...")
        try:
            subprocess.run([str(python_path), "src/data/download_usda_data.py"], check=True)
            logger.info("USDA sample data created successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create USDA sample data: {e}")
    
    # Check if PRISM sample data exists
    prism_dir = Path("data") / "processed" / "prism"
    prism_dir.mkdir(parents=True, exist_ok=True)
    
    prism_sample = prism_dir / "prism_growing_season_sample.csv"
    
    if not prism_sample.exists():
        logger.info("Creating PRISM sample data...")
        try:
            subprocess.run([str(python_path), "src/data/download_prism_data.py"], check=True)
            logger.info("PRISM sample data created successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create PRISM sample data: {e}")

def run_streamlit_app():
    """Run the Streamlit application."""
    logger.info("Starting Streamlit application...")
    
    # Determine path to streamlit in the virtual environment
    if sys.platform == "win32":
        streamlit_path = Path("venv") / "Scripts" / "streamlit"
    else:
        streamlit_path = Path("venv") / "bin" / "streamlit"
    
    if not streamlit_path.exists():
        logger.error(f"Could not find streamlit at {streamlit_path}")
        logger.error("Make sure streamlit is installed by running: pip install streamlit")
        sys.exit(1)
    
    try:
        subprocess.run([str(streamlit_path), "run", "src/app/main.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run Streamlit application: {e}")
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")

def main():
    """Main function to set up and run the application."""
    parser = argparse.ArgumentParser(description="Setup and run the Agricultural Yield Climate Impact Analysis System")
    parser.add_argument("--setup-only", action="store_true", help="Only set up the environment without running the app")
    parser.add_argument("--run-only", action="store_true", help="Only run the app without setting up")
    
    args = parser.parse_args()
    
    # Setup phase
    if not args.run_only:
        check_python_version()
        setup_virtual_environment()
        install_dependencies()
        create_sample_data()
        
        logger.info("Setup completed successfully!")
    
    # Run phase
    if not args.setup_only:
        run_streamlit_app()

if __name__ == "__main__":
    main()