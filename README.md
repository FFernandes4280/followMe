# 🚁 FollowMe - Sistema de Rastreamento com Drone Simulado

Sistema avançado de rastreamento e seguimento de pessoas usando YOLOv8 ONNX com **simulação realista de drone** controlado por **PID** (Proporcional-Integral-Derivativo) e visualização em tempo real de duas câmeras.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-orange.svg)](https://github.com/ultralytics/ultralytics)

</div>

---

## 📋 Índice

- [Características](#-características-principais)
- [Arquivos Principais](#-arquivos-principais)
- [Uso Rápido](#-uso-rápido)
- [Controles](#-controles-de-teclado)
- [Comandos do Drone](#-comandos-e-controle-do-drone)
- [Modelos Disponíveis](#-modelos-disponíveis)
- [Configuração](#-requisitos-e-configuração)
- [Visualização](#-visualização)
- [Parâmetros PID](#️-parâmetros-do-pid)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Solução de Problemas](#-solução-de-problemas)

---

## ✨ Características Principais

### 🎮 Simulação de Drone Realista
- **Controlador PID Adaptativo**: Movimento suave com ganho dinâmico baseado na distância do alvo
- **Campo de Visão Configurável**: 50% da tela por padrão, ajustável
- **Zoom Dinâmico**: Ajuste automático de 0.5x a 3.0x baseado nos comandos
- **Memória de Movimento**: Mantém último comando válido por 1 segundo ao perder o alvo
- **Dual View**: Janela principal + câmera do drone em tempo real

### 📊 Sistema de Grade Inteligente
- **Grade Adaptativa**: 
  - 3x5 para vídeos portrait (9:16)
  - 3x3 para vídeos landscape (16:9)
- **Grid Contextual**: Calculado apenas dentro da visão do drone
- **Feedback Visual**: Overlay colorido indicando ocupação
- **Detecção por Intensidade**: Movimentos proporcionais à urgência

### 🎯 Controle Avançado
- **PID com Ganho Adaptativo**: 
  - 2.5x mais rápido quando distante do alvo
  - 1.8x mais rápido em distâncias médias
  - Suave quando próximo
- **Velocidade Configurável**: Até 30 pixels/frame
- **Anti-Windup**: Previne saturação do termo integral
- **Suporte Multi-Modelo**: COCO, long.onnx, short.onnx

### 📱 Modo Celular como Drone
- Use seu smartphone como câmera de drone virtual
- Comandos em tempo real para movimentação física do celular
- Compatível com apps de IP Webcam

---

## 📁 Arquivos Principais

| Arquivo | Descrição |
|---------|-----------|
| [`followMe_coco.py`](followMe_coco.py) | Modelo COCO pré-treinado (80 classes) |
| [`followMe_custom.py`](followMe_custom.py) | Modelos customizados (long/short) |
| [`followMe_cellphone.py`](followMe_cellphone.py) | Modo celular como câmera drone |
| [`run_coco.sh`](run_coco.sh) | Script para modelo COCO |
| [`run_long.sh`](run_long.sh) | Script para modelo long.onnx |
| [`run_cellphone.sh`](run_cellphone.sh) | Script para modo celular |
| [`model.py`](model.py) | Utilitários e exportação de modelos |

---

## 🚀 Uso Rápido

### Executar Modelos Individuais

```bash
# Modelo COCO (melhor para detecção geral)
./run_coco.sh

# Modelo Long (treinamento extenso)
./run_long.sh

# Modo Celular
./run_cellphone.sh
```

### Uso Manual

```bash
# Modelo customizado
python3 followMe_custom.py --model long.onnx --source video.mp4

# Webcam
python3 followMe_custom.py --model long.onnx --source 0

# Celular
python3 followMe_cellphone.py --model long.onnx --source http://IP:8080/video
```

---

## 🎮 Controles de Teclado

| Tecla | Função |
|-------|--------|
| **Q** | Sair do programa |
| **P** | Pausar/Despausar vídeo |
| **G** | Toggle visualização da grade |
| **D** | Toggle visualização da câmera do drone |
| **O** | Toggle ocupação no terminal (debug) |

---

## 🤖 Comandos e Controle do Drone

### Comandos Gerados

| Comando | Ação do Drone | Movimento |
|---------|---------------|-----------|
| `SEGUIR_FRENTE` | Aumenta zoom | +0.1x zoom |
| `VIRAR_ESQUERDA` | Move câmera à esquerda | Até -160 pixels |
| `VIRAR_DIREITA` | Move câmera à direita | Até +160 pixels |
| `INCLINAR_PARA_CIMA` | Move câmera para cima | Até -160 pixels |
| `INCLINAR_PARA_BAIXO` | Move câmera para baixo | Até +160 pixels |
| `RECUAR` | Diminui zoom | -0.2x zoom |
| `Alvo perdido` | Mantém último movimento | Por 30 frames (~1s) |
| `PROCURANDO_ALVO` | Zoom out para buscar | -0.05x zoom |

### 🎯 Movimento Proporcional

A intensidade do movimento é proporcional à urgência da situação:

- **Baixa ocupação**: Move 80 pixels
- **Média ocupação**: Move 120 pixels  
- **Alta ocupação**: Move 160 pixels

### 🧠 PID Adaptativo

```python
# Ganho baseado na distância do alvo
if erro > 100px:  Kp × 2.5  # Muito rápido
if erro > 50px:   Kp × 1.8  # Rápido
if erro ≤ 50px:   Kp × 1.0  # Suave
```

---

## 📦 Modelos Disponíveis

| Modelo | Descrição | Classes | Uso |
|--------|-----------|---------|-----|
| `yolov8n.onnx` | YOLOv8 Nano COCO | 80 | Detecção geral |
| `long.onnx` | Treinamento extenso | Custom | Melhor acurácia |
| `short.onnx` | Treinamento rápido | Custom | Teste rápido |
| `sports_detection_best.onnx` | Detecção esportiva | Sports | Esportes radicais |

---

## 🔧 Requisitos e Configuração

### Dependências

```bash
Python 3.10+
OpenCV 4.x (compilado com GTK2)
NumPy 2.x
```

### Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd followMe

# OpenCV já está compilado localmente em opencv/build/
# Os scripts .sh configuram automaticamente
```

### Configuração Manual do OpenCV

```bash
export PYTHONPATH="/path/to/followMe/opencv/build/lib/python3:$PYTHONPATH"
export GTK_PATH="/usr/lib/x86_64-linux-gnu/gtk-2.0"
export GTK2_RC_FILES="/usr/share/themes/Adwaita/gtk-2.0/gtkrc"
```

### Argumentos de Linha de Comando

```bash
python3 followMe_custom.py \
    --model long.onnx \           # Modelo ONNX
    --source video.mp4 \          # Vídeo ou webcam (0)
    --headless \                  # Sem GUI (opcional)
    --output result.mp4 \         # Salvar saída (opcional)
    --max-frames 1000             # Limitar frames (opcional)
```

---

## 🖥️ Visualização

### Janela Principal - "Rastreamento Customizado"
- Vídeo completo em alta resolução (1280x720)
- Retângulo amarelo/ciano mostrando visão do drone
- Grid 3x3 ou 3x5 **apenas dentro da visão do drone**
- HUD com telemetria do drone (posição, zoom, velocidade, comandos)
- Detecções com bounding boxes verdes

### Janela Secundária - "Drone Camera View"
- Recorte da região que o drone "enxerga"
- Crosshair central para referência
- Tamanho fixo: 360x640 redimensionado
- Atualização em tempo real conforme PID move o drone
- Borda amarela indicando limites da visão

---

## ⚙️ Parâmetros do PID

### Controladores de Posição (X e Y)

```python
Kp = 1.2   # Proporcional - resposta ao erro
Ki = 0.03  # Integral - correção de erro acumulado
Kd = 0.2   # Derivativo - suavização de movimento
```

### Limites Configurados

```python
max_velocity = 30.0          # pixels/frame
max_zoom_velocity = 0.1      # unidades/frame
zoom_range = [0.5, 3.0]      # 50% a 300%
memory_frames = 30           # ~1 segundo @ 30 FPS
```

---

## 🎨 Estrutura do Projeto

```
followMe/
├── followMe_custom.py           # ⭐ Principal com drone simulado
├── followMe_coco.py             # Versão com modelo COCO
├── followMe_cellphone.py        # 📱 Modo celular
├── followMe_onnxruntime.py      # ONNX Runtime (experimental)
├── model.py                     # Utilitários de modelo
├── export_model_opencv.py       # Exportar para OpenCV
│
├── run_coco.sh                  # 🚀 Executar COCO
├── run_long.sh                  # 🚀 Executar Long
├── run_cellphone.sh             # 🚀 Executar Celular
│
├── yolov8n.onnx                 # Modelo COCO
├── long.onnx                    # Modelo custom long
├── sports_detection_best.onnx   # Modelo esportes
│
├── opencv/                      # OpenCV compilado
│   └── build/
│       └── lib/python3/
│
└── red-bull/                    # Dataset e scripts
    ├── script.py                # Extração de frames
    └── src/
        ├── normalizeVideoNames.py
        └── videos/              # Vídeos de treino
```

---

## 📝 Notas Técnicas

### Sistema de Grade

- **Portrait** (altura > largura): Grade 3x5 para melhor cobertura vertical
- **Landscape** (largura > altura): Grade 3x3 para proporção equilibrada
- Grid calculado **exclusivamente dentro da visão do drone**, não na tela inteira
- Cada célula mostra ocupação percentual com overlay colorido

### Sistema de Memória

Quando o alvo é perdido:
1. **0-30 frames**: Mantém último movimento válido
2. **Após 30 frames**: Ativa modo `PROCURANDO_ALVO` (zoom out)
3. **Redetecção**: Retoma rastreamento normal

### Otimizações

- ✅ Ganho PID adaptativo baseado na distância
- ✅ Movimento proporcional à urgência (intensidade da ocupação)
- ✅ Velocidade máxima aumentada para resposta rápida
- ✅ Anti-windup para estabilidade do termo integral
- ✅ Clipping de posição para manter drone dentro dos limites

---

## 🚀 Changelog e Melhorias

### v2.0 - Simulação de Drone com PID
- ✅ Classe [`PIDController`](followMe_custom.py) com P, I, D configuráveis
- ✅ Classe [`SimulatedDrone`](followMe_custom.py) com posição e zoom independentes
- ✅ Ganho adaptativo (até 2.5x mais rápido quando longe)
- ✅ Visualização dual (tela principal + câmera do drone)
- ✅ HUD informativo com telemetria em tempo real
- ✅ Grid restrito à visão do drone
- ✅ Memória de movimento (continua 1s após perder alvo)

### v1.0 - Sistema Base
- ✅ Detecção com YOLOv8 ONNX via [`cv2.dnn`](followMe_custom.py)
- ✅ Grid de ocupação adaptativo ([`calculate_grid_occupation`](followMe_custom.py))
- ✅ Comandos baseados em posição ([`generate_movement_commands`](followMe_custom.py))
- ✅ Suporte a múltiplos modelos

---

## 💡 Roadmap - Melhorias Futuras

- [ ] Modo "auto-center" (centraliza automaticamente no alvo)
- [ ] Gravação separada da visão do drone
- [ ] Telemetria exportada para CSV/JSON
- [ ] Simulação de inércia e física realista do drone
- [ ] Waypoints e trajetórias pré-programadas
- [ ] Múltiplos drones simultâneos
- [ ] Predição de movimento com Filtro de Kalman
- [ ] Detecção de gestos para controle manual
- [ ] Modo de voo autônomo com IA
- [ ] Integração com drones reais (DJI SDK)

---

## 🐛 Solução de Problemas

### Erro: OpenCV sem GUI

```bash
# O projeto já usa OpenCV compilado localmente
# Scripts .sh configuram automaticamente
./run_long.sh  # Deve funcionar
```

### Avisos GTK

```bash
# Scripts .sh filtram automaticamente
# Para ver todos os avisos:
python3 followMe_custom.py --model long.onnx --source video.mp4
```

### Performance Baixa

```bash
# Opções de otimização:
python3 followMe_custom.py \
    --model short.onnx \        # Modelo mais leve
    --max-frames 500 \          # Limitar processamento
    --headless                  # Sem GUI
```

### Celular não Conecta

1. Verifique se o app **IP Webcam** está rodando
2. Confirme que ambos estão na mesma rede WiFi
3. Teste a URL no navegador: `http://IP:8080/video`
4. Ajuste o IP no [`run_cellphone.sh`](run_cellphone.sh)

---

## 📄 Licença

Este projeto é desenvolvido para fins **educacionais** e de pesquisa. Sinta-se livre para usar, modificar e distribuir conforme necessário.

Os vídeos da pasta `red-bull/` podem estar sujeitos a direitos autorais da Red Bull Media House.

---

## 👥 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 🙏 Agradecimentos

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Framework de detecção
- [OpenCV](https://opencv.org/) - Processamento de imagem
- Red Bull Media House - Vídeos de exemplo

---

## 📧 Contato

Para dúvidas, sugestões ou reportar bugs, abra uma [issue](../../issues).

---

<div align="center">

**🎯 Objetivo**: Criar um sistema realista de rastreamento que simula o comportamento de um drone autônomo seguindo um alvo, com movimento suave e natural graças ao controle PID.

Made with ❤️ for Computer Vision and Robotics

</div>