#!/bin/bash
set -e

# Simple reliability test with dummy text

echo "Running dummy voice conversation test" 
python scripts/voice_chat.py --mic -1 <<EOF2
Hello
EOF2

echo "Test completed"
