#!/bin/bash
# Script para executar todos os modelos simultaneamente

echo "======================================"
echo "  FollowMe - Comparação de Modelos"
echo "======================================"
echo ""
echo "Executando os 3 modelos simultaneamente:"
echo "  1. COCO (yolov8n.onnx)"
echo "  2. LONG (long.onnx)"
echo "  3. SHORT (short.onnx)"
echo ""
echo "Pressione 'q' em qualquer janela para fechar"
echo "======================================"
echo ""

# Executa os três modelos em background
python3 followMe_coco.py &
COCO_PID=$!

sleep 1

python3 followMe_custom.py --model long.onnx --source videos/Falls_Wont_Stop_Him.mp4 &
LONG_PID=$!

sleep 1

python3 followMe_custom.py --model short.onnx --source videos/Falls_Wont_Stop_Him.mp4 &
SHORT_PID=$!

echo "Processos iniciados:"
echo "  COCO PID: $COCO_PID"
echo "  LONG PID: $LONG_PID"
echo "  SHORT PID: $SHORT_PID"
echo ""
echo "Aguardando conclusão..."

# Aguarda todos os processos terminarem
wait $COCO_PID $LONG_PID $SHORT_PID

echo ""
echo "Todos os modelos foram encerrados."
