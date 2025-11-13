# 🚁 FollowMe - Sistema de Rastreamento com Drone Simulado

Sistema avançado de rastreamento e seguimento de pessoas usando YOLOv8 ONNX com **simulação de drone** controlado por **PID** (Proporcional-Integral-Derivativo) e feedback visual em tempo real.

## ✨ Características Principais

### 🎮 Simulação de Drone
- **Controlador PID**: Movimento suave e realista com controle adaptativo
- **Visão do Drone**: Campo de visão configurável (50% da tela por padrão)
- **Zoom Dinâmico**: Ajuste automático de zoom (0.5x a 3.0x) baseado nos comandos
- **Memória de Movimento**: Continua último movimento por 1 segundo ao perder alvo
- **Duas Janelas**: Visão geral + Câmera do drone em tempo real

### 📊 Sistema de Grade Inteligente
- **Grade Adaptativa**: 3x5 (portrait) ou 3x3 (landscape)
- **Grid na Visão do Drone**: Calculado apenas dentro da área de visão do drone
- **Feedback Visual**: Overlay colorido mostrando ocupação (verde escuro = alta ocupação)
- **Detecção por Intensidade**: Movimentos proporcionais à urgência da situação

### 🎯 Controle Avançado
- **PID Adaptativo**: Ganho aumenta com a distância do alvo (até 2.5x mais rápido)
- **Velocidade Ajustável**: Até 30 pixels/frame para resposta rápida
- **Anti-Windup**: Previne saturação do termo integral
- **Suporte a Múltiplos Modelos**: COCO, long.onnx e short.onnx

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

## 🎮 Controles de Teclado

| Tecla | Função |
|-------|--------|
| **Q** | Sair do programa |
| **P** | Pausar/Despausar |
| **G** | Alternar visualização da grade |
| **D** | Alternar visualização do drone |
| **O** | Alternar impressão da ocupação no terminal |

## 🤖 Comandos e Controle do Drone

O sistema gera comandos baseados na ocupação da grade **dentro da visão do drone**:

| Comando | Ação do Drone | Movimento |
|---------|---------------|-----------|
| `SEGUIR_FRENTE` | Aumenta zoom | +0.1x zoom |
| `VIRAR_ESQUERDA` | Move câmera para esquerda | Até -160 pixels |
| `VIRAR_DIREITA` | Move câmera para direita | Até +160 pixels |
| `INCLINAR_PARA_CIMA` | Move câmera para cima | Até -160 pixels |
| `INCLINAR_PARA_BAIXO` | Move câmera para baixo | Até +160 pixels |
| `RECUAR` | Diminui zoom | -0.2x zoom |
| `Alvo perdido` | Mantém último movimento | Por 30 frames (~1s) |

### 🎯 Movimento Proporcional
A intensidade do movimento é proporcional à urgência:
- **Baixa ocupação** → Move 80 pixels
- **Média ocupação** → Move 120 pixels  
- **Alta ocupação** → Move 160 pixels

### 🧠 PID Adaptativo
- **Erro > 100px**: Kp × 2.5 (muito rápido)
- **Erro > 50px**: Kp × 1.8 (rápido)
- **Erro ≤ 50px**: Kp × 1.0 (suave)

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

## 🔧 Requisitos e Configuração

### Dependências
- Python 3.10+
- **OpenCV com suporte GTK** (compilado localmente)
- NumPy 2.x

### Configuração do OpenCV Local
O projeto usa OpenCV compilado localmente com suporte a GUI (GTK2):

```bash
# OpenCV está em: opencv/build/
# Biblioteca Python: opencv/build/lib/python3/cv2.cpython-310-x86_64-linux-gnu.so
```

Os scripts `.sh` já configuram automaticamente:
- `PYTHONPATH` para usar OpenCV local
- `GTK_PATH` para módulos GTK
- Filtro de avisos GTK para saída limpa

### Argumentos Opcionais

```bash
python3 followMe_custom.py --model long.onnx --source video.mp4 [OPÇÕES]

--headless          # Executa sem janela (útil em servidores)
--output FILE       # Salva vídeo de saída (ex: output.mp4)
--max-frames N      # Processa apenas N frames (para testes)
```

## � Visualização

### Janela Principal - "Rastreamento Customizado"
- Vídeo completo em alta resolução (1280x720)
- Retângulo amarelo/ciano mostrando visão do drone
- Grid 3x3 ou 3x5 **apenas dentro da visão do drone**
- HUD com status do drone (posição, zoom, velocidade, comandos)

### Janela Secundária - "Drone Camera View"
- Recorte da região que o drone "enxerga"
- Crosshair central para referência
- Tamanho fixo: 360x640 (redimensionado de ~50% da tela)
- Atualização em tempo real conforme PID move o drone

## ⚙️ Parâmetros do PID

### Posição (X e Y)
```python
Kp = 1.2  # Proporcional (resposta ao erro)
Ki = 0.03 # Integral (correção de erro acumulado)
Kd = 0.2  # Derivativo (suavização)
```

### Limites
- **Velocidade máxima**: 30 pixels/frame
- **Velocidade de zoom**: 0.1 unidades/frame
- **Faixa de zoom**: 0.5x a 3.0x
- **Memória de movimento**: 30 frames (~1 segundo)

## 🎨 Estrutura do Projeto

```
followMe/
├── followMe_custom.py      # Script principal com drone simulado
├── followMe_coco.py         # Versão com modelo COCO
├── model.py                 # Utilitários
├── run_long.sh              # Executa modelo long.onnx
├── run_short.sh             # Executa modelo short.onnx
├── run_coco.sh              # Executa modelo COCO
├── run_all.sh               # Executa todos simultaneamente
├── opencv/                  # OpenCV compilado (não versionado)
│   └── build/
│       └── lib/python3/
└── videos/                  # Vídeos de teste (não versionados)
    └── Falls_Wont_Stop_Him.mp4
```

## 📝 Notas Técnicas

### Grade Adaptativa
- **Portrait** (altura > largura): Grade 3x5 para melhor cobertura vertical
- **Landscape** (largura > altura): Grade 3x3 para proporção equilibrada
- Grid calculado **apenas dentro da visão do drone**, não na tela inteira

### Sistema de Memória
- Ao perder o alvo, o drone **continua o último movimento válido**
- Memória ativa por até **30 frames** (~1 segundo @ 30 FPS)
- Após timeout, drone para e aguarda redetecção

### Otimizações
- Ganho PID adaptativo baseado na distância
- Movimento proporcional à urgência (intensidade da ocupação)
- Velocidade máxima aumentada para resposta rápida
- Anti-windup para estabilidade do termo integral

## 🚀 Melhorias Implementadas

### v2.0 - Simulação de Drone com PID
- ✅ Classe `PIDController` com termos P, I, D configuráveis
- ✅ Classe `SimulatedDrone` com posição e zoom independentes
- ✅ Ganho adaptativo (2.5x mais rápido quando longe do alvo)
- ✅ Visualização dual (tela principal + câmera do drone)
- ✅ HUD informativo com telemetria em tempo real
- ✅ Grid restrito à visão do drone (não na tela inteira)
- ✅ Memória de movimento (continua 1s após perder alvo)

### v1.0 - Sistema Base
- ✅ Detecção com YOLOv8 ONNX
- ✅ Grid de ocupação adaptativo
- ✅ Comandos de movimento baseados em posição
- ✅ Suporte a múltiplos modelos

## 💡 Possíveis Melhorias Futuras

- [ ] Modo "seguir automaticamente" (centra no alvo)
- [ ] Gravação separada da visão do drone
- [ ] Telemetria exportada para CSV/JSON
- [ ] Simulação de inércia e física do drone
- [ ] Waypoints e trajetórias pré-programadas
- [ ] Múltiplos drones simultâneos
- [ ] Predição de movimento (Kalman Filter)
- [ ] Detecção de gestos para controle manual

## 🐛 Solução de Problemas

### OpenCV sem GUI
Se encontrar erro `cv2.error: The function is not implemented`:
```bash
# O projeto já usa OpenCV local compilado
# Scripts .sh já configuram automaticamente
./run_long.sh  # Deve funcionar sem erros
```

### Avisos GTK
Os scripts filtram automaticamente avisos GTK. Se quiser ver tudo:
```bash
python3 followMe_custom.py --model long.onnx --source videos/video.mp4
```

### Performance
Para melhorar FPS:
- Use `--max-frames` para limitar processamento
- Reduza resolução do vídeo de entrada
- Use modelo mais leve (short.onnx)

## 📄 Licença

Projeto acadêmico - Use livremente para fins educacionais.

## 👥 Contribuições

Desenvolvido como parte de estudo de sistemas de visão computacional e controle PID para drones autônomos.

---

**🎯 Objetivo**: Criar um sistema realista de rastreamento que simula o comportamento de um drone autônomo seguindo um alvo, com movimento suave e natural graças ao controle PID.
