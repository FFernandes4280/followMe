# FollowMe - Sistema de Rastreamento com YOLOv8

Sistema de rastreamento e seguimento de objetos usando modelos YOLOv8 ONNX com feedback visual em grade 3x5 (portrait) ou 3x3 (landscape).

## 🎯 Características

- **Grade Visual Adaptativa**: Células com cores que indicam ocupação (verde escuro = alta ocupação)
- **Comandos de Movimento**: Gera comandos baseados na posição do objeto na grade
- **Suporte a Múltiplos Modelos**: COCO, long.onnx e short.onnx
- **Interface Limpa**: Apenas comandos essenciais no terminal

## 📁 Arquivos Principais

- `followMe_coco.py` - Usa modelo COCO pré-treinado (yolov8n.onnx)
- `followMe_custom.py` - Usa modelos customizados (long.onnx ou short.onnx)
- `run_coco.sh` - Script para executar modelo COCO
- `run_long.sh` - Script para executar modelo long.onnx
- `run_short.sh` - Script para executar modelo short.onnx
- `run_all.sh` - Executa os três modelos simultaneamente

## 🚀 Uso Rápido

### Executar um modelo individual:

```bash
# Modelo COCO
./run_coco.sh

# Modelo Long
./run_long.sh

# Modelo Short
./run_short.sh
```

### Executar todos os modelos lado a lado:

```bash
./run_all.sh
```

Isso abrirá 3 janelas simultaneamente para comparação visual.

## 🎮 Controles

Durante a execução:
- `q` - Sair
- `o` - Alternar impressão da grade no terminal (desativado por padrão)
- `g` - Alternar visualização da grade no vídeo

## 📊 Comandos Gerados

O sistema gera comandos baseados na ocupação da grade:

- `SEGUIR_FRENTE` - Objeto centralizado
- `VIRAR_ESQUERDA` - Objeto à esquerda
- `VIRAR_DIREITA` - Objeto à direita
- `INCLINAR_PARA_CIMA` - Objeto no topo
- `INCLINAR_PARA_BAIXO` - Objeto embaixo
- `RECUAR` - Grade totalmente ocupada
- `Alvo perdido` - Nenhuma detecção

## 📦 Modelos

- **yolov8n.onnx** - YOLOv8 Nano COCO (80 classes)
- **long.onnx** - Modelo customizado (treinamento longo)
- **short.onnx** - Modelo customizado (treinamento curto)

## 🎥 Fonte de Vídeo

Por padrão, usa o vídeo: `videos/Falls_Wont_Stop_Him.mp4` (720x1280 portrait)

Para usar webcam ou outro vídeo, edite os scripts `.sh` ou execute manualmente:

```bash
# Webcam
python3 followMe_custom.py --model long.onnx --source 0

# Outro vídeo
python3 followMe_custom.py --model long.onnx --source caminho/para/video.mp4
```

## 🔧 Requisitos

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## 📝 Notas

- A grade adapta-se automaticamente à orientação do vídeo
- Portrait (altura > largura): Grade 3x5
- Landscape (largura > altura): Grade 3x3
- Feedback visual com intensidade de cor proporcional à ocupação
