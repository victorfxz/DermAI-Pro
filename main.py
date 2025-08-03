#!/usr/bin/env python3
"""
DermAI Pro - Professional Dermatology AI System
Main Application Entry Point

🏥 Professional dermatological diagnosis system using AI
🤖 Powered by Gemma 3n-E4B via Ollama
🔬 Detects 14+ skin conditions with clinical precision

Author: DermAI Development Team
Version: 1.0.0 Professional
Date: 2025-08-02
"""

import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure CustomTkinter appearance
ctk.set_appearance_mode("light")  # Professional medical appearance
ctk.set_default_color_theme("blue")  # Medical blue theme

def setup_logging():
    """Configure professional logging system"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "dermai_pro.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger("DermAI-Pro")
    logger.info("DermAI Pro - Professional Dermatology AI System Starting...")
    return logger

def check_dependencies():
    """Check if all required dependencies are available"""
    logger = logging.getLogger("DermAI-Pro")
    
    try:
        import numpy
        import cv2
        import PIL
        import requests
        import customtkinter
        logger.info("Core dependencies verified")

        # Check Ollama availability
        try:
            import ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is running")

                # Check if Gemma model is available
                models = response.json().get('models', [])
                gemma_available = any('gemma3n:e4b' in model.get('name', '') for model in models)

                if gemma_available:
                    logger.info("Gemma 3n-E4B model is available")
                    return True
                else:
                    logger.warning("Gemma 3n-E4B model not found. Please run: ollama pull gemma3n:e4b")
                    return False
            else:
                logger.error("Ollama server not responding")
                return False

        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def show_startup_dialog():
    """Show professional startup dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    result = messagebox.askyesno(
        "DermAI Pro - Professional Dermatology AI System",
        "🏥 Welcome to DermAI Pro v1.0.0\n\n"
        "Professional dermatological diagnosis system using AI\n"
        "Powered by Gemma 3n-E4B via Ollama\n\n"
        "⚠️ MEDICAL DISCLAIMER:\n"
        "This system is designed for medical professionals as a diagnostic aid.\n"
        "Results should always be validated by qualified dermatologists.\n"
        "Not intended for self-diagnosis or replacing professional medical advice.\n\n"
        "🔒 Privacy: All data processing is done locally.\n"
        "No patient data is transmitted over the internet.\n\n"
        "Continue to launch DermAI Pro?",
        icon='question'
    )
    
    root.destroy()
    return result

def main():
    """Main application entry point"""
    print("🏥 DermAI Pro - Professional Dermatology AI System")
    print("=" * 60)
    print("🤖 Powered by Gemma 3n-E4B via Ollama")
    print("🔬 Professional dermatological diagnosis system")
    print("📅 Version 1.0.0 - August 2025")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Show startup dialog
        if not show_startup_dialog():
            logger.info("👋 User cancelled startup - exiting gracefully")
            return
        
        # Check dependencies
        logger.info("Checking system dependencies...")
        if not check_dependencies():
            messagebox.showerror(
                "DermAI Pro - System Check Failed",
                "System requirements not met!\n\n"
                "Please ensure:\n"
                "1. Ollama is installed and running (ollama serve)\n"
                "2. Gemma 3n-E4B model is available (ollama pull gemma3n:e4b)\n"
                "3. All Python dependencies are installed (pip install -r requirements.txt)\n\n"
                "Check the logs for detailed error information."
            )
            return

        # Import and launch main application
        logger.info("Launching DermAI Pro main application...")
        from src.ui.main_interface import DermAIProInterface

        # Create and run application
        app = DermAIProInterface()
        logger.info("DermAI Pro interface initialized successfully")
        
        # Start the application
        app.run()
        
    except ImportError as e:
        logger.error(f"Failed to import application modules: {e}")
        messagebox.showerror(
            "DermAI Pro - Import Error",
            f"Failed to load application components:\n\n{str(e)}\n\n"
            "Please ensure all dependencies are installed:\n"
            "pip install -r requirements.txt"
        )

    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        messagebox.showerror(
            "DermAI Pro - Startup Error",
            f"An unexpected error occurred:\n\n{str(e)}\n\n"
            "Please check the logs for detailed information."
        )

    finally:
        logger.info("DermAI Pro session ended")

if __name__ == "__main__":
    main()
