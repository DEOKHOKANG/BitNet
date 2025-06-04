#!/bin/bash
set -e

# Setup environment for voice AI model

# 1. Create python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install bitnet dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Clone parakeet TTS repo if not present
if [ ! -d "parakeet-tdt" ]; then
    git clone https://github.com/efuelteam/parakeet-tdt-0.6b-v2-fastapi.git parakeet-tdt
fi
pip install -r parakeet-tdt/requirements.txt

# 4. Install speech recognition library and other utilities
sudo apt-get update -y
sudo apt-get install -y portaudio19-dev
pip install SpeechRecognition pyaudio requests

# 5. Build bitnet project
python setup_env.py --hf-repo microsoft/BitNet-b1.58-2B-4T

# 6. Setup parakeet TTS service
pushd parakeet-tdt
./setup.sh
popd

echo "Environment setup complete"
