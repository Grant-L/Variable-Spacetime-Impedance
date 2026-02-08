#!/bin/bash

# Setup script for Variable Spacetime Impedance project
# This script sets up the Python environment and installs dependencies

set -e

echo "🚀 Setting up Variable Spacetime Impedance project..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or later."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📌 Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Jupyter Lab extensions (optional)
echo "🔧 Setting up Jupyter Lab..."
jupyter lab build

# Create Jupyter kernel for this project
echo "🎯 Creating Jupyter kernel..."
python -m ipykernel install --user --name=variable-spacetime-impedance --display-name="Variable Spacetime Impedance"

echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start Jupyter Lab: jupyter lab"
echo "  3. Open a notebook from the notebooks/ directory"
echo ""
echo "Happy exploring! 🔬"
