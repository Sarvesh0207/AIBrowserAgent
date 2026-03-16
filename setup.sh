#!/usr/bin/env bash
# setup.sh — bootstrap the WebAgent environment
set -e

echo "🔧 Setting up WebAgent..."

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# 2. Activate
source .venv/bin/activate

# 3. Install Python deps
pip install --upgrade pip -q
pip install -e . -q
echo "✅ Python dependencies installed"

# 4. Install Playwright browsers (Chromium only)
playwright install chromium
echo "✅ Playwright Chromium installed"

# 5. Check for .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚠️  .env created from .env.example — please add your ANTHROPIC_API_KEY"
else
    echo "✅ .env already exists"
fi

echo ""
echo "🚀 Setup complete! Run the agent with:"
echo "   source .venv/bin/activate"
echo "   python main.py"
echo "   python main.py --url https://google.com"
echo "   python main.py --headless  # no browser window"
