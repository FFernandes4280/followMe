# 🏃‍♂️ Sistema de Detecção de Pessoas em Esportes

Sistema completo de detecção de pessoas praticando esportes baseado no YOLOv8, desenvolvido a partir do projeto followMe original.

## 📋 Visão Geral

Este projeto implementa um sistema de detecção de pessoas em esportes que inclui:

- **Treinamento de modelo personalizado** para detecção de pessoas em esportes
- **Preparação de datasets** com suporte a múltiplos formatos
- **Validação e teste** com métricas detalhadas
- **Detecção em tempo real** com interface visual
- **Sistema de comandos de movimento** baseado em grade 3x3

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- CUDA (opcional, para aceleração GPU)
- Câmera USB (para detecção em tempo real)

### Instalação das Dependências

#### Opção 1: Instalação com Ambiente Virtual (Recomendado)

```bash
# Clone o repositório
git clone <repository-url>
cd followMe

# Instalação com ambiente virtual (resolve erro "externally-managed-environment")
python3 setup_venv.py

# Para usar o sistema
./activate_and_run.sh
# ou no Windows: activate_and_run.bat
```

#### Opção 2: Instalação Direta

```bash
# Clone o repositório
git clone <repository-url>
cd followMe

# Instale as dependências
python3 -m pip install -r requirements.txt

# Se der erro "externally-managed-environment", use a Opção 1
```

#### Opção 3: Instalação Manual

```bash
# Clone o repositório
git clone <repository-url>
cd followMe

# Cria ambiente virtual
python3 -m venv venv

# Ativa ambiente virtual
source venv/bin/activate  # Linux/macOS
# ou venv\Scripts\activate  # Windows

# Instala dependências
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
followMe/
├── followMe.py                    # Script original de follow-me
├── sports_detection_training.py   # Treinamento do modelo personalizado
├── dataset_preparation.py         # Preparação de datasets
├── model_validation.py           # Validação e teste do modelo
├── sports_detection_realtime.py  # Detecção em tempo real
├── sports_detection_system.py    # Sistema unificado (recomendado)
├── setup_venv.py                 # Instalação com ambiente virtual
├── launcher_venv.py              # Launcher principal
├── activate_and_run.sh           # Script de ativação (Linux/macOS)
├── activate_and_run.bat          # Script de ativação (Windows)
├── config.yaml                   # Configurações do sistema
├── requirements.txt              # Dependências Python
├── README.md                     # Esta documentação
└── venv/                         # Ambiente virtual (criado automaticamente)
```

## 🎯 Uso Rápido

### 🚀 Inicialização Rápida (Recomendado)

```bash
# Instalação e configuração automática
python3 setup_venv.py

# Executa o sistema
./activate_and_run.sh
# ou no Windows: activate_and_run.bat
```

### 🎮 Sistema Unificado

```bash
# Interface principal com todas as funcionalidades
python3 sports_detection_system.py
```

### 1. Treinamento do Modelo

```bash
# Treina um modelo personalizado para detecção de esportes
python3 sports_detection_training.py
```

### 2. Detecção em Tempo Real

```bash
# Executa detecção em tempo real com câmera
python3 sports_detection_realtime.py

# Com parâmetros personalizados
python3 sports_detection_realtime.py --model sports_detection_best.pt --confidence 0.5
```

### 3. Detecção em Imagem Estática

```bash
# Detecta pessoas em uma imagem
python3 sports_detection_realtime.py --image path/to/image.jpg
```

### 4. Sistema Unificado

```bash
# Interface unificada com todas as funcionalidades
python3 sports_detection_system.py
```


## 📊 Funcionalidades Detalhadas

### 🏋️ Treinamento de Modelo (`sports_detection_training.py`)

- **Dataset sintético**: Gera dados de treinamento automaticamente
- **Suporte a datasets reais**: Importa dados de esportes existentes
- **Configuração flexível**: Parâmetros de treinamento personalizáveis
- **Exportação automática**: Salva modelo em formatos PyTorch e ONNX

**Exemplo de uso:**
```python
from sports_detection_training import SportsDetectionTrainer

# Cria treinador
trainer = SportsDetectionTrainer(model_size="n")

# Cria dataset sintético
trainer.create_synthetic_dataset()

# Treina modelo
results = trainer.train_model(epochs=100, batch_size=16)
```

### 📁 Preparação de Dataset (`dataset_preparation.py`)

- **Múltiplos formatos**: Suporte a YOLO, COCO, Pascal VOC
- **Divisão automática**: Separa dados em treino/validação/teste
- **Validação de integridade**: Verifica consistência dos dados
- **Dataset sintético**: Gera dados de demonstração

**Exemplo de uso:**
```python
from dataset_preparation import SportsDatasetPreparer

# Prepara dataset customizado
preparer = SportsDatasetPreparer()
preparer.prepare_custom_dataset(
    images_dir="path/to/images",
    annotations_dir="path/to/annotations",
    annotation_format="yolo"
)
```

### 🔍 Validação de Modelo (`model_validation.py`)

- **Métricas detalhadas**: mAP, precisão, recall
- **Benchmark de performance**: FPS, tempo de inferência
- **Análise de thresholds**: Testa diferentes níveis de confiança
- **Comparação com baseline**: Compara com modelo padrão
- **Visualizações**: Gráficos de performance

**Exemplo de uso:**
```python
from model_validation import SportsModelValidator

# Valida modelo
validator = SportsModelValidator("sports_detection_best.pt")
results = validator.validate_on_test_set()

# Executa benchmark
benchmark = validator.benchmark_performance("test_images/")
```

### 🎥 Detecção em Tempo Real (`sports_detection_realtime.py`)

- **Interface visual**: Grade 3x3 com informações de ocupação
- **Comandos de movimento**: Sistema baseado no followMe original
- **Controles interativos**: Teclas para alternar visualizações
- **Salvamento de vídeo**: Gravação de sessões de detecção
- **Detecção em imagens**: Processamento de imagens estáticas

**Controles de teclado:**
- `q`: Sair
- `g`: Alternar grade 3x3
- `o`: Alternar informações de ocupação
- `s`: Salvar frame atual
- `p`: Alternar impressão da grade no terminal

## ⚙️ Configuração Avançada

### Parâmetros de Treinamento

```python
# Configuração personalizada de treinamento
train_params = {
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'patience': 20
}
```

### Configuração de Detecção

```python
# Parâmetros de detecção
detector = SportsDetectionRealtime(
    model_path="sports_detection_best.pt",
    confidence_threshold=0.3
)
```

## 📈 Métricas e Performance

### Métricas de Validação

- **mAP50**: Mean Average Precision com IoU 0.5
- **mAP50-95**: Mean Average Precision com IoU 0.5-0.95
- **Precision**: Precisão das detecções
- **Recall**: Taxa de detecção

### Performance em Tempo Real

- **FPS**: Quadros por segundo
- **Latência**: Tempo de inferência por frame
- **Uso de memória**: Consumo de RAM/VRAM

## 🔧 Solução de Problemas

### Problemas Comuns

1. **Erro "externally-managed-environment"**
   ```bash
   # Solução: Use ambiente virtual
   python3 setup_venv.py
   ./activate_and_run.sh
   ```

2. **Erro de câmera não encontrada**
   ```bash
   # Verifica câmeras disponíveis
   ls /dev/video*
   
   # Usa câmera específica
   python3 sports_detection_realtime.py --camera 1
   ```

3. **Modelo não encontrado**
   ```bash
   # Treina modelo primeiro
   python3 sports_detection_training.py
   
   # Ou usa modelo padrão
   python3 sports_detection_realtime.py --model yolov8n.pt
   ```

4. **Erro de dependências**
   ```bash
   # Com ambiente virtual
   source venv/bin/activate
   pip install -r requirements.txt --upgrade
   
   # Ou instalação direta
   python3 -m pip install -r requirements.txt --upgrade
   ```

5. **Problemas de permissão**
   ```bash
   # Torna scripts executáveis
   chmod +x *.py
   
   # Ou executa diretamente
   python3 script_name.py
   ```

### Logs e Debug

- Logs de treinamento: `runs/detect/sports_detection/`
- Resultados de validação: `benchmark_results.json`
- Imagens de teste: `detection_results/`

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- [Ultralytics](https://github.com/ultralytics/ultralytics) pelo YOLOv8
- [OpenCV](https://opencv.org/) para processamento de imagem
- Projeto followMe original pela base do sistema de comandos

## 📞 Suporte

Para dúvidas e suporte:

- Abra uma [issue](https://github.com/your-repo/issues)
- Consulte a [documentação](https://github.com/your-repo/wiki)
- Entre em contato: [seu-email@exemplo.com]

---

**Desenvolvido com ❤️ para detecção de pessoas em esportes**
