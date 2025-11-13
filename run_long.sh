#!/bin/bash
# Script para executar o modelo long.onnx com OpenCV local

echo "=== Iniciando FollowMe com modelo LONG ==="

# Configura OpenCV local e GTK
export PYTHONPATH="/home/ffernandes/followMe/opencv/build/lib/python3:$PYTHONPATH"
export GTK_PATH="/usr/lib/x86_64-linux-gnu/gtk-2.0"
export GTK2_RC_FILES="/usr/share/themes/Adwaita/gtk-2.0/gtkrc"

# Executa filtrando avisos GTK
python3 followMe_custom.py --model long.onnx --source videos/Falls_Wont_Stop_Him.mp4 2> >(grep -v -E "^Gtk-|^\(.*\): Gtk-" >&2)
