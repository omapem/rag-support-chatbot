#!/bin/bash

# Backend Initialization Script for RAG Support Chatbot
# This script creates the complete project structure with starter code

echo "🚀 Initializing RAG Support Chatbot Backend..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p backend/{src,app/api,app/middleware,tests,data/{raw/{pdfs,scraped},processed,eval},notebooks,scripts}

# Create __init__.py files to make Python packages
echo "📦 Creating Python packages..."
touch backend/src/__init__.py
touch backend/app/__init__.py
touch backend/app/api/__init__.py
touch backend/app/middleware/__init__.py
touch backend/tests/__init__.py

# Create placeholder files for data directories
touch backend/data/raw/.gitkeep
touch backend/data/processed/.gitkeep
touch backend/data/eval/.gitkeep

echo "✅ Directory structure created!"
echo ""
echo "📝 Next steps:"
echo "1. cd backend"
echo "2. python3 -m venv venv"
echo "3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "4. pip install -r requirements.txt"
echo "5. Copy .env.example to .env and add your API keys"
echo ""
echo "🎉 Backend structure initialized! Ready for development."
