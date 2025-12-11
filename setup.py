#!/usr/bin/env python3
"""
Setup script for Palantir Agent
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_env_file():
    """Check if .env file exists and guide user to create it."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("📝 Creating .env file from template...")
            subprocess.run(["cp", ".env.example", ".env"])
            print("✅ .env file created. Please edit it with your API keys.")
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file exists")
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "evo_agent/experiments",
        "evo_agent/code",
        "evo_agent/evaluations",
        "evo_agent/prompts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Required directories created")

def main():
    """Main setup function."""
    print("🚀 Setting up AlphaEvolve Agent...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check/create .env file
    if not check_env_file():
        print("❌ Setup failed: Could not create .env file")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python evo_agent/run_guided.py")

if __name__ == "__main__":
    main()
