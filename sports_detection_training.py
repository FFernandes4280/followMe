#!/usr/bin/env python33
"""
Script de Treinamento para Detecção de Pessoas em Esportes
Baseado no YOLOv8, este script treina um modelo para detectar pessoas praticando esportes.
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class SportsDetectionTrainer:
    def __init__(self, data_dir="sports_data", model_size="n"):
        """
        Inicializa o treinador de detecção de esportes
        
        Args:
            data_dir (str): Diretório onde estão os dados de treinamento
            model_size (str): Tamanho do modelo YOLO ('n', 's', 'm', 'l', 'x')
        """
        self.data_dir = Path(data_dir)
        self.model_size = model_size
        self.model = None
        self.setup_directories()
        
    def setup_directories(self):
        """Cria a estrutura de diretórios necessária"""
        directories = [
            self.data_dir / "images" / "train",
            self.data_dir / "images" / "val",
            self.data_dir / "labels" / "train", 
            self.data_dir / "labels" / "val",
            self.data_dir / "datasets",
            Path("runs") / "detect"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Diretório criado: {directory}")
    
    def create_dataset_config(self):
        """Cria o arquivo de configuração do dataset no formato YOLO"""
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,  # Número de classes
            'names': ['person_sporting']  # Nome da classe
        }
        
        config_path = self.data_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Configuração do dataset salva em: {config_path}")
        return config_path
    
    
    def create_synthetic_dataset(self):
        """Cria um dataset sintético para demonstração"""
        print("🎨 Criando dataset sintético para demonstração...")
        
        # Cria imagens sintéticas com pessoas fazendo esportes
        for split in ['train', 'val']:
            for i in range(70 if split == 'train' else 30):
                # Cria uma imagem sintética
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Adiciona uma "pessoa" retangular (simulada)
                x, y, w, h = np.random.randint(50, 500, 4)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), -1)
                
                # Salva a imagem
                img_path = self.data_dir / "images" / split / f"image_{i:04d}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Cria o arquivo de anotação correspondente
                label_path = self.data_dir / "labels" / split / f"image_{i:04d}.txt"
                with open(label_path, 'w') as f:
                    # Formato YOLO: class_id center_x center_y width height (normalizado)
                    center_x = (x + w/2) / 640
                    center_y = (y + h/2) / 640
                    width = w / 640
                    height = h / 640
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print("✓ Dataset sintético criado com sucesso!")
    
    
    def train_model(self, epochs=100, batch_size=16, img_size=640):
        """
        Treina o modelo YOLOv8 para detecção de pessoas em esportes
        
        Args:
            epochs (int): Número de épocas de treinamento
            batch_size (int): Tamanho do batch
            img_size (int): Tamanho da imagem de entrada
        """
        print("🚀 Iniciando treinamento do modelo...")
        
        # Carrega o modelo YOLOv8
        model_name = f"yolov8{self.model_size}.pt"
        self.model = YOLO(model_name)
        
        # Configuração do dataset
        dataset_config = self.create_dataset_config()
        
        # Parâmetros de treinamento
        train_params = {
            'data': str(dataset_config),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': 'runs/detect',
            'name': 'sports_detection',
            'save': True,
            'save_period': 10,
            'patience': 20,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        print(f"📊 Parâmetros de treinamento:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")
        
        # Inicia o treinamento
        try:
            results = self.model.train(**train_params)
            print("✅ Treinamento concluído com sucesso!")
            
            # Salva o melhor modelo
            best_model_path = "runs/detect/sports_detection/weights/best.pt"
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, "sports_detection_best.pt")
                print(f"✓ Melhor modelo salvo como: sports_detection_best.pt")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro durante o treinamento: {e}")
            return None
    
    def validate_model(self, model_path=None):
        """
        Valida o modelo treinado
        
        Args:
            model_path (str): Caminho para o modelo treinado
        """
        if model_path is None:
            model_path = "sports_detection_best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo não encontrado: {model_path}")
            return None
        
        print("🔍 Validando modelo...")
        
        # Carrega o modelo
        model = YOLO(model_path)
        
        # Valida no dataset de validação
        dataset_config = self.data_dir / "dataset.yaml"
        results = model.val(data=str(dataset_config))
        
        print("✅ Validação concluída!")
        print(f"📊 Métricas de validação:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def export_model(self, model_path=None, formats=['onnx', 'torchscript']):
        """
        Exporta o modelo para diferentes formatos
        
        Args:
            model_path (str): Caminho para o modelo treinado
            formats (list): Formatos para exportação
        """
        if model_path is None:
            model_path = "sports_detection_best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo não encontrado: {model_path}")
            return
        
        print("📤 Exportando modelo...")
        
        model = YOLO(model_path)
        
        for format_type in formats:
            try:
                exported_path = model.export(format=format_type, opset=12, simplify=True)
                print(f"✓ Modelo exportado para {format_type}: {exported_path}")
            except Exception as e:
                print(f"❌ Erro ao exportar para {format_type}: {e}")

def main():
    """Função principal"""
    print("🏃‍♂️ Sistema de Treinamento para Detecção de Pessoas em Esportes")
    print("=" * 60)
    
    # Inicializa o treinador
    trainer = SportsDetectionTrainer(model_size="n")
    
    # Cria dataset sintético para demonstração
    trainer.create_synthetic_dataset()
    
    # Treina o modelo
    print("\n🚀 Iniciando treinamento...")
    results = trainer.train_model(epochs=10, batch_size=8)
    
    if results:
        # Valida o modelo
        print("\n🔍 Validando modelo...")
        trainer.validate_model()
        
        # Exporta o modelo
        print("\n📤 Exportando modelo...")
        trainer.export_model()
        
        print("\n✅ Processo concluído com sucesso!")
        print("📁 Arquivos gerados:")
        print("  - sports_detection_best.pt (modelo PyTorch)")
        print("  - sports_detection_best.onnx (modelo ONNX)")
        print("  - runs/detect/sports_detection/ (logs e métricas)")
    else:
        print("❌ Falha no treinamento!")

if __name__ == "__main__":
    main()
