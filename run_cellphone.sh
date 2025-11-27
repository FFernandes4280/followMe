#!/bin/bash
# Script para executar o followMe com câmera do celular

export QT_QPA_PLATFORM=xcb

echo "🚁 Iniciando Phone Drone Camera..."
echo ""

python3 followMe_cellphone.py \
    --model long.onnx \
    --source http://192.168.3.43:8080/video

echo ""
echo "✓ Encerrado"
