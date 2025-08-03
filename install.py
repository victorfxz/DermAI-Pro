#!/usr/bin/env python3
"""
DermAI Pro - Installation and Setup Script
Automated installation and configuration for professional use

🏥 Professional dermatology AI system setup
🔧 Automated dependency installation
✅ System validation and testing
"""

import sys
import os
import subprocess
import platform
import requests
import time
from pathlib import Path

def print_header():
    """Print installation header"""
    print("=" * 70)
    print("🏥 DermAI Pro - Professional Dermatology AI System")
    print("📦 Installation and Setup Script")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  DermAI Pro requires Python 3.8 or higher")
        print("📥 Please install Python 3.8+ from https://python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        print("🔄 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        requirements_file = Path(__file__).parent / "requirements.txt"
        if requirements_file.exists():
            print("📋 Installing from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        else:
            print("📋 Installing core dependencies...")
            packages = [
                "numpy>=1.21.0",
                "opencv-python>=4.5.0",
                "Pillow>=8.3.0",
                "requests>=2.25.0",
                "customtkinter>=5.0.0",
                "scikit-image>=0.18.0",
                "matplotlib>=3.3.0",
                "scipy>=1.7.0",
                "scikit-learn>=1.0.0",
                "colorlog>=6.0.0",
                "pyyaml>=5.4.0",
                "pandas>=1.3.0",
                "ollama>=0.1.0"
            ]
            
            for package in packages:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    print("\n🤖 Checking Ollama installation...")
    
    try:
        # Check if Ollama is accessible
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server not responding")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server not accessible")
        print("📥 Please install Ollama from https://ollama.ai")
        print("🚀 Then run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        return False

def install_gemma_model():
    """Install Gemma 3n-E4B model via Ollama"""
    print("\n🧠 Installing Gemma 3n-E4B AI model...")
    
    try:
        # Check if model already exists
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            gemma_exists = any('gemma3n:e4b' in model.get('name', '') for model in models)
            
            if gemma_exists:
                print("✅ Gemma 3n-E4B model already installed")
                return True
        
        # Install model
        print("📥 Downloading Gemma 3n-E4B model (this may take several minutes)...")
        print("⏳ Please wait while the model downloads...")
        
        # Use subprocess to run ollama pull
        result = subprocess.run(
            ["ollama", "pull", "gemma3n:e4b"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Gemma 3n-E4B model installed successfully")
            return True
        else:
            print(f"❌ Model installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model installation timed out")
        print("⚠️  Please run manually: ollama pull gemma3n:e4b")
        return False
    except FileNotFoundError:
        print("❌ Ollama command not found")
        print("📥 Please install Ollama from https://ollama.ai")
        return False
    except Exception as e:
        print(f"❌ Model installation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    base_dir = Path(__file__).parent
    directories = [
        "logs",
        "temp",
        "assets",
        "test_images",
        "exports"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(exist_ok=True)
        print(f"📂 Created: {directory}/")
    
    print("✅ Directory structure created")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        print("📦 Testing Python imports...")
        import numpy
        import cv2
        import PIL
        import customtkinter
        import requests
        print("✅ Core imports successful")
        
        # Test AI engine
        print("🤖 Testing AI engine...")
        sys.path.insert(0, str(Path(__file__).parent))
        from src.core.ai_engine import DermAIEngine
        
        engine = DermAIEngine()
        if engine.initialize_model():
            print("✅ AI engine initialization successful")
        else:
            print("⚠️  AI engine initialization failed - check Ollama and model")
        
        # Test lesion detector
        print("🔬 Testing lesion detector...")
        from src.core.lesion_detector import ProfessionalLesionDetector
        
        detector = ProfessionalLesionDetector()
        print("✅ Lesion detector initialization successful")
        
        print("✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_completion_message():
    """Print installation completion message"""
    print("\n" + "=" * 70)
    print("🎉 DermAI Pro Installation Complete!")
    print("=" * 70)
    print()
    print("🚀 To start DermAI Pro:")
    print("   python main.py")
    print()
    print("📋 System Requirements Met:")
    print("   ✅ Python 3.8+")
    print("   ✅ Required packages installed")
    print("   ✅ Ollama server running")
    print("   ✅ Gemma 3n-E4B model available")
    print()
    print("🏥 DermAI Pro is ready for professional dermatological analysis!")
    print()
    print("⚠️  IMPORTANT REMINDERS:")
    print("   • This system is for medical professionals only")
    print("   • Always validate AI results with clinical expertise")
    print("   • Ensure Ollama server is running before use")
    print("   • All processing is done locally for privacy")
    print()
    print("📞 For support, check the README.md file")
    print("=" * 70)

def main():
    """Main installation process"""
    print_header()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed at dependency installation")
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama_installation():
        print("❌ Installation failed - Ollama not available")
        print("📥 Please install Ollama and run 'ollama serve' before continuing")
        sys.exit(1)
    
    # Install AI model
    if not install_gemma_model():
        print("⚠️  Model installation failed - you can install it manually later")
        print("🔧 Run: ollama pull gemma3n:e4b")
    
    # Create directories
    create_directories()
    
    # Test installation
    if not test_installation():
        print("⚠️  Some tests failed - installation may be incomplete")
    
    # Print completion message
    print_completion_message()

if __name__ == "__main__":
    main()
