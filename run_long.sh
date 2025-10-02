#!/bin/bash
# Script para executar o modelo long.onnx

echo "=== Iniciando FollowMe com modelo LONG ==="
python3 followMe_custom.py --model long.onnx --source videos/Falls_Wont_Stop_Him.mp4
