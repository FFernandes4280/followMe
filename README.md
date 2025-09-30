# 🏃‍♂️ Sistema FollowMe com Dataset Red Bull

Sistema de detecção e seguimento de pessoas em esportes usando YOLOv8 treinado especificamente com dataset do Red Bull.

## 🎯 Características

- **Dataset Red Bull**: Treinado com vídeos reais de esportes extremos do Red Bull
- **Detecção em Tempo Real**: Usa YOLOv8 para detectar pessoas em vídeo ao vivo
- **Sistema de Comandos**: Gera comandos de movimento baseado na posição das pessoas
- **Grade 3x3**: Divide a imagem em 9 quadrantes para análise de ocupação
- **Processamento Automático**: Extrai frames e gera anotações automaticamente
- **Alta Precisão**: 95.99% de precisão e 92.06% mAP50

## 🚀 Instalação Rápida

### Pré-requisitos

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Scikit-learn

### Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd followMe

# Ativa ambiente virtual
source venv/bin/activate

# Instala dependências se necessário
pip install scikit-learn
```

## 🎮 Uso

### Execução Completa (Recomendado)

```bash
python3 run_redbull_system.py
```

Este comando executa todo o pipeline automaticamente:
1. Processa vídeos do Red Bull
2. Treina o modelo
3. Executa o sistema de seguimento

### Uso Individual

1. **Processar dataset do Red Bull:**
```bash
python3 redbull_dataset_processor.py
```

2. **Treinar modelo:**
```bash
python3 sports_detection_training.py
```

3. **Executar sistema:**
```bash
python3 followMe.py
```

## 📁 Estrutura do Projeto

```
followMe/
├── run_redbull_system.py           # Script principal (executa tudo)
├── redbull_dataset_processor.py    # Processamento do dataset Red Bull
├── sports_detection_training.py    # Treinamento do modelo
├── followMe.py                     # Sistema principal com comandos
├── model_validation.py             # Validação de modelos
├── config.yaml                     # Configurações
├── red-bull/                       # Vídeos do Red Bull
│   └── src/                        # 7 vídeos de esportes extremos
├── sports_data/                    # Dataset processado
│   ├── images/                     # Frames extraídos (train/val/test)
│   ├── labels/                     # Anotações YOLO
│   └── dataset.yaml                # Configuração do dataset
├── sports_detection_best.pt        # Modelo PyTorch treinado
├── sports_detection_best.onnx      # Modelo ONNX
└── runs/detect/                    # Logs de treinamento
```

## 🎬 Dataset Red Bull

O sistema usa vídeos reais do Red Bull como dataset de treinamento:

- **7 vídeos** de esportes extremos
- **280 frames** extraídos automaticamente (a cada 30 frames)
- **188 frames** com detecções válidas de pessoas
- **Anotações automáticas** geradas usando YOLOv8n
- **Divisão**: 131 treino, 28 validação, 29 teste

### Vídeos Incluídos:
- "Attempts We Can Still Feel.mp4"
- "Falls Won't Stop Him.mp4"
- "He's Riding On One Wheel... Over Water.mp4"
- "The Ramp Life Chose Him.mp4"
- "This Is NOT Your Average Cycle Ride.mp4"
- "When The Whole City Becomes A Bike Park.mp4"
- "World's Longest Railslide On A Wakeboard.mp4"

## 🎮 Comandos de Movimento

O sistema gera comandos baseado na posição das pessoas na grade 3x3:

- **`SEGUIR_FRENTE`**: Pessoa no centro da imagem
- **`VIRAR_ESQUERDA`**: Pessoa na coluna esquerda
- **`VIRAR_DIREITA`**: Pessoa na coluna direita
- **`INCLINAR_PARA_CIMA`**: Pessoa na linha superior
- **`INCLINAR_PARA_BAIXO`**: Pessoa na linha inferior
- **`RECUAR`**: Múltiplas pessoas detectadas
- **`Alvo perdido`**: Nenhuma pessoa detectada

## 🎛️ Controles

- **'q'**: Sair do sistema
- **'o'**: Alternar exibição da grade de ocupação no terminal

## 📊 Resultados de Treinamento

Com 10 épocas de treinamento no dataset Red Bull:

| Métrica | Valor |
|---------|-------|
| **Precisão** | 95.99% |
| **Recall** | 82.35% |
| **mAP50** | 92.06% |
| **mAP50-95** | 60.42% |

## ⚙️ Configuração

Edite o arquivo `config.yaml` para personalizar:

```yaml
# Configurações do Red Bull
dataset:
  redbull:
    video_dir: "red-bull/src"
    frame_interval: 30
    max_frames_per_video: 100
    detection_confidence: 0.4

# Configurações de treinamento
training:
  epochs: 10
  batch_size: 8
  image_size: 640
```

## 🔧 Solução de Problemas

### 1. Dataset não encontrado
```bash
# Processa o dataset primeiro
python3 redbull_dataset_processor.py
```

### 2. Modelo não encontrado
```bash
# Treina o modelo
python3 sports_detection_training.py
```

### 3. Câmera não encontrada
```bash
# Verifica câmeras disponíveis
ls /dev/video*

# Usa câmera específica (edite followMe.py)
cap = cv2.VideoCapture(1)  # Mude o número
```

### 4. Dependências faltando
```bash
source venv/bin/activate
pip install scikit-learn opencv-python ultralytics
```

## 📈 Performance

- **FPS**: ~15-20 FPS em CPU
- **Latência**: ~50-80ms por frame
- **Memória**: ~2-3 GB RAM
- **Tamanho do modelo**: 6.3 MB (PyTorch), 11.7 MB (ONNX)

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT.

## 🙏 Agradecimentos

- [Ultralytics](https://github.com/ultralytics/ultralytics) pelo YOLOv8
- [OpenCV](https://opencv.org/) para processamento de imagem
- Red Bull pelo conteúdo de esportes extremos

---

**Desenvolvido com ❤️ para detecção de pessoas em esportes usando dataset Red Bull**