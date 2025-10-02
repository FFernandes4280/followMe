#!/bin/bash
# Script para executar o modelo short.onnx

echo "=== Iniciando FollowMe com modelo SHORT ==="
python3 followMe_custom.py --model short.onnx --source videos/Falls_Wont_Stop_Him.mp4
