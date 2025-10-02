#!/usr/bin/env python3
"""
Processador de Dataset Red Bull para Detecção de Pessoas em Esportes
Este script extrai frames dos vídeos do Red Bull e gera anotações automáticas.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from sklearn.model_selection import train_test_split
import shutil
from typing import List, Tuple
import random

class RedBullDatasetProcessor:
    def __init__(self, redbull_dir="red-bull/src", output_dir="sports_data"):
        """
        Inicializa o processador do dataset Red Bull
        
        Args:
            redbull_dir (str): Diretório com os vídeos do Red Bull
            output_dir (str): Diretório de saída para o dataset
        """
        self.redbull_dir = Path(redbull_dir)
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
        # Carrega modelo YOLO para detecção de pessoas
        self.detection_model = YOLO("yolov8n.pt")
        
    def setup_directories(self):
        """Cria a estrutura de diretórios necessária"""
        directories = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test",
            self.output_dir / "raw_frames",
            self.output_dir / "annotations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Diretório criado: {directory}")
    
    def extract_frames_from_videos(self, frame_interval: int = 30, max_frames_per_video: int = 100):
        """
        Extrai frames dos vídeos do Red Bull
        
        Args:
            frame_interval (int): Intervalo entre frames (a cada N frames)
            max_frames_per_video (int): Máximo de frames por vídeo
        """
        print("🎬 Extraindo frames dos vídeos do Red Bull...")
        
        video_files = list(self.redbull_dir.glob("*.mp4"))
        print(f"📹 Encontrados {len(video_files)} vídeos")
        
        total_frames = 0
        
        for video_path in video_files:
            print(f"\n🎥 Processando: {video_path.name}")
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Erro ao abrir vídeo: {video_path}")
                continue
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extrai frame a cada frame_interval frames
                if frame_count % frame_interval == 0:
                    # Redimensiona frame para 640x640
                    frame_resized = cv2.resize(frame, (640, 640))
                    
                    # Salva frame
                    frame_filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
                    frame_path = self.output_dir / "raw_frames" / frame_filename
                    cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70]) # <--- 70% de compressão JPEG
                    
                    extracted_count += 1
                    total_frames += 1
                    
                    if extracted_count >= max_frames_per_video:
                        break
                
                frame_count += 1
            
            cap.release()
            print(f"✓ Extraídos {extracted_count} frames de {video_path.name}")
        
        print(f"\n✅ Total de frames extraídos: {total_frames}")
        return total_frames
    
    def detect_persons_in_frames(self, confidence_threshold: float = 0.5):
        """
        Detecta pessoas nos frames extraídos usando YOLO
        
        Args:
            confidence_threshold (float): Threshold de confiança para detecções
        """
        print("🔍 Detectando pessoas nos frames...")
        
        frames_dir = self.output_dir / "raw_frames"
        frame_files = list(frames_dir.glob("*.jpg"))
        
        print(f"📊 Processando {len(frame_files)} frames...")
        
        detections_count = 0
        
        for i, frame_path in enumerate(frame_files):
            if i % 50 == 0:
                print(f"  Processando frame {i+1}/{len(frame_files)}")
            
            # Carrega frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            
            # Detecta pessoas
            results = self.detection_model(frame, conf=confidence_threshold, classes=[0]) # classe 0 = person
            
            # Processa detecções
            annotations = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Converte coordenadas para formato YOLO
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Normaliza coordenadas
                        img_height, img_width = frame.shape[:2]
                        center_x = (x1 + x2) / 2.0 / img_width
                        center_y = (y1 + y2) / 2.0 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # Adiciona anotação (classe 0 = person_sporting)
                        annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Salva anotação se houver detecções
            if annotations:
                annotation_path = self.output_dir / "annotations" / f"{frame_path.stem}.txt"
                with open(annotation_path, 'w') as f:
                    f.write('\n'.join(annotations))
                detections_count += 1
        
        print(f"✅ Detecções encontradas em {detections_count} frames")
        return detections_count
    
    def organize_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        Organiza o dataset nos splits de treino, validação e teste
        
        Args:
            train_ratio (float): Proporção para treino
            val_ratio (float): Proporção para validação
            test_ratio (float): Proporção para teste
        """
        print("📁 Organizando dataset...")
        
        # Lista frames com anotações
        frames_dir = self.output_dir / "raw_frames"
        annotations_dir = self.output_dir / "annotations"
        
        valid_frames = []
        for frame_path in frames_dir.glob("*.jpg"):
            annotation_path = annotations_dir / f"{frame_path.stem}.txt"
            if annotation_path.exists():
                valid_frames.append(frame_path)
        
        print(f"📊 {len(valid_frames)} frames com anotações válidas")
        
        if len(valid_frames) == 0:
            print("❌ Nenhum frame com anotação encontrado!")
            return
        
        # Divide em splits
        train_frames, temp_frames = train_test_split(
            valid_frames, 
            test_size=(val_ratio + test_ratio), 
            random_state=42
        )
        val_frames, test_frames = train_test_split(
            temp_frames, 
            test_size=test_ratio/(val_ratio + test_ratio), 
            random_state=42
        )
        
        print(f"📈 Divisão do dataset:")
        print(f"   Treino: {len(train_frames)} frames")
        print(f"   Validação: {len(val_frames)} frames")
        print(f"   Teste: {len(test_frames)} frames")
        
        # Copia arquivos para splits
        for split_name, frames in [("train", train_frames), ("val", val_frames), ("test", test_frames)]:
            for frame_path in frames:
                # Copia imagem
                dest_img = self.output_dir / "images" / split_name / frame_path.name
                shutil.copy2(frame_path, dest_img)
                
                # Copia anotação
                annotation_path = annotations_dir / f"{frame_path.stem}.txt"
                dest_annotation = self.output_dir / "labels" / split_name / f"{frame_path.stem}.txt"
                shutil.copy2(annotation_path, dest_annotation)
        
        print("✅ Dataset organizado com sucesso!")
    
    def create_dataset_config(self):
        """Cria arquivo de configuração do dataset"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['person_sporting']
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Configuração salva em: {config_path}")
        return config_path
    
    def validate_dataset(self):
        """Valida a integridade do dataset"""
        print("🔍 Validando dataset...")
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / "images" / split
            labels_dir = self.output_dir / "labels" / split
            
            # Conta arquivos
            image_files = list(images_dir.glob("*.jpg"))
            label_files = list(labels_dir.glob("*.txt"))
            
            print(f"  {split}: {len(image_files)} imagens, {len(label_files)} labels")
            
            # Verifica correspondência
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    issues.append(f"Label não encontrado para {img_file.name} em {split}")
        
        if issues:
            print("⚠️  Problemas encontrados:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Dataset validado com sucesso!")
        
        return len(issues) == 0
    
    def process_redbull_dataset(self, frame_interval: int = 30, max_frames_per_video: int = 100, 
                               confidence_threshold: float = 0.5):
        """
        Processa completamente o dataset do Red Bull
        
        Args:
            frame_interval (int): Intervalo entre frames
            max_frames_per_video (int): Máximo de frames por vídeo
            confidence_threshold (float): Threshold de confiança para detecções
        """
        print("🏃‍♂️ Processando Dataset Red Bull para Detecção de Pessoas em Esportes")
        print("=" * 70)
        
        # 1. Extrai frames dos vídeos
        print("\n1️⃣ Extraindo frames...")
        total_frames = self.extract_frames_from_videos(frame_interval, max_frames_per_video)
        
        if total_frames == 0:
            print("❌ Nenhum frame extraído!")
            return False
        
        # 2. Detecta pessoas nos frames
        print("\n2️⃣ Detectando pessoas...")
        detections = self.detect_persons_in_frames(confidence_threshold)
        
        if detections == 0:
            print("❌ Nenhuma pessoa detectada!")
            return False
        
        # 3. Organiza dataset
        print("\n3️⃣ Organizando dataset...")
        self.organize_dataset()
        
        # 4. Cria configuração
        print("\n4️⃣ Criando configuração...")
        self.create_dataset_config()
        
        # 5. Valida dataset
        print("\n5️⃣ Validando dataset...")
        is_valid = self.validate_dataset()
        
        if is_valid:
            print("\n✅ Dataset Red Bull processado com sucesso!")
            print("📁 Estrutura criada:")
            print(f"  - {self.output_dir}/images/ (train, val, test)")
            print(f"  - {self.output_dir}/labels/ (train, val, test)")
            print(f"  - {self.output_dir}/dataset.yaml")
            return True
        else:
            print("\n❌ Dataset com problemas!")
            return False

def main():
    """Função principal"""
    # Inicializa processador
    processor = RedBullDatasetProcessor()
    
    # Processa dataset
    success = processor.process_redbull_dataset(
        frame_interval=1,          # Extrai todos os frames do vídeo (1 frame por frame)
        max_frames_per_video=3600, # Máximo 3600 frames por vídeo (2 minutos de vídeo)
        confidence_threshold=0.4   # Threshold de confiança para detecções
    )
    
    if success:
        print("\n🚀 Dataset pronto para treinamento!")
        print("Execute: python sports_detection_training.py")
    else:
        print("\n❌ Falha no processamento do dataset!")

if __name__ == "__main__":
    main()
