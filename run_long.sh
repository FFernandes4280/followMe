#!/bin/bash
# Script para executar o modelo long.onnx com ONNX Runtime (melhor compatibilidade)

echo "=== Iniciando FollowMe com ONNX Runtime ==="

# Configura OpenCV local e GTK
export PYTHONPATH="/home/ffernandes/followMe/opencv/build/lib/python3:$PYTHONPATH"
export GTK_PATH="/usr/lib/x86_64-linux-gnu/gtk-2.0"
export GTK2_RC_FILES="/usr/share/themes/Adwaita/gtk-2.0/gtkrc"

# Executa com ONNX Runtime (suporta modelos com DFL head)
# python3 followMe_onnxruntime.py --model sports_detection_best.onnx --source red-bull/src/videos/worlds_longest_railslide_on_a_wakeboard.mp4 #Roda muito bem
python3 followMe_onnxruntime.py --model sports_detection_best.onnx --source red-bull/src/videos/you_get_a_flip_and_you_get_a_flip.mp4 #Médio
# python3 followMe_onnxruntime.py --model sports_detection_best.onnx --source red-bull/src/videos/falls_wont_stop_him.mp4 #Caso de erro
# python3 followMe_onnxruntime.py --model sports_detection_best.onnx --source red-bull/src/videos/worlds_longest_railslide_on_a_wakeboard.mp4

# python3 followMe_onnxruntime.py --model long.onnx --source red-bull/src/videos/worlds_longest_railslide_on_a_wakeboard.mp4