#!/usr/bin/env python33
"""
Script de Preparação de Dataset para Detecção de Pessoas em Esportes
Este script ajuda a preparar e organizar datasets de esportes para treinamento.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import yaml

class SportsDatasetPreparer:
    def __init__(self, output_dir="sports_dataset"):
        """
        Inicializa o preparador de dataset
        
        Args:
            output_dir (str): Diretório de saída para o dataset organizado
        """
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Cria a estrutura de diretórios necessária"""
        directories = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val", 
            self.output_dir / "labels" / "test",
            self.output_dir / "raw_data",
            self.output_dir / "annotations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Diretório criado: {directory}")
    
    
    def prepare_custom_dataset(self, images_dir: str, annotations_dir: str = None, 
                             annotation_format: str = "yolo"):
        """
        Prepara um dataset customizado de esportes
        
        Args:
            images_dir (str): Diretório com as imagens
            annotations_dir (str): Diretório com as anotações (opcional)
            annotation_format (str): Formato das anotações ('yolo', 'coco', 'pascal_voc')
        """
        print(f"📁 Preparando dataset customizado...")
        print(f"   Imagens: {images_dir}")
        print(f"   Anotações: {annotations_dir}")
        print(f"   Formato: {annotation_format}")
        
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir) if annotations_dir else None
        
        # Lista todas as imagens
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        print(f"📊 Encontradas {len(image_files)} imagens")
        
        if not image_files:
            print("❌ Nenhuma imagem encontrada!")
            return
        
        # Divide em treino, validação e teste
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        print(f"📈 Divisão do dataset:")
        print(f"   Treino: {len(train_files)} imagens")
        print(f"   Validação: {len(val_files)} imagens") 
        print(f"   Teste: {len(test_files)} imagens")
        
        # Processa cada split
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            self.process_split(files, split_name, annotations_path, annotation_format)
        
        # Cria arquivo de configuração
        self.create_dataset_config()
        
        print("✅ Dataset customizado preparado com sucesso!")
    
    def process_split(self, image_files: List[Path], split_name: str, 
                     annotations_path: Path = None, annotation_format: str = "yolo"):
        """
        Processa um split específico do dataset
        
        Args:
            image_files: Lista de arquivos de imagem
            split_name: Nome do split ('train', 'val', 'test')
            annotations_path: Caminho para anotações
            annotation_format: Formato das anotações
        """
        print(f"🔄 Processando split {split_name}...")
        
        for img_file in image_files:
            # Copia imagem
            dest_img = self.output_dir / "images" / split_name / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Processa anotação se disponível
            if annotations_path:
                annotation_file = self.find_annotation_file(img_file, annotations_path, annotation_format)
                if annotation_file:
                    self.convert_annotation(annotation_file, dest_img, split_name, annotation_format)
                else:
                    print(f"⚠️  Anotação não encontrada para: {img_file.name}")
            else:
                # Cria anotação vazia se não houver anotações
                empty_annotation = self.output_dir / "labels" / split_name / f"{img_file.stem}.txt"
                empty_annotation.touch()
    
    def find_annotation_file(self, image_file: Path, annotations_path: Path, 
                           annotation_format: str) -> Path:
        """
        Encontra o arquivo de anotação correspondente à imagem
        
        Args:
            image_file: Arquivo de imagem
            annotations_path: Diretório de anotações
            annotation_format: Formato das anotações
            
        Returns:
            Caminho para o arquivo de anotação ou None
        """
        base_name = image_file.stem
        
        if annotation_format == "yolo":
            annotation_file = annotations_path / f"{base_name}.txt"
        elif annotation_format == "coco":
            # Para COCO, precisaríamos do arquivo JSON principal
            return None
        elif annotation_format == "pascal_voc":
            annotation_file = annotations_path / f"{base_name}.xml"
        else:
            return None
        
        return annotation_file if annotation_file.exists() else None
    
    def convert_annotation(self, annotation_file: Path, image_file: Path, 
                          split_name: str, annotation_format: str):
        """
        Converte anotação para formato YOLO
        
        Args:
            annotation_file: Arquivo de anotação original
            image_file: Arquivo de imagem correspondente
            split_name: Nome do split
            annotation_format: Formato original da anotação
        """
        # Lê dimensões da imagem
        img = cv2.imread(str(image_file))
        img_height, img_width = img.shape[:2]
        
        # Converte baseado no formato
        if annotation_format == "yolo":
            # Já está no formato YOLO, apenas copia
            dest_annotation = self.output_dir / "labels" / split_name / f"{image_file.stem}.txt"
            shutil.copy2(annotation_file, dest_annotation)
            
        elif annotation_format == "pascal_voc":
            # Converte de Pascal VOC para YOLO
            self.convert_pascal_voc_to_yolo(annotation_file, image_file, split_name, 
                                          img_width, img_height)
    
    def convert_pascal_voc_to_yolo(self, xml_file: Path, image_file: Path, 
                                  split_name: str, img_width: int, img_height: int):
        """
        Converte anotação Pascal VOC para formato YOLO
        
        Args:
            xml_file: Arquivo XML Pascal VOC
            image_file: Arquivo de imagem correspondente
            split_name: Nome do split
            img_width: Largura da imagem
            img_height: Altura da imagem
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            # Pula se não for pessoa
            class_name = obj.find('name').text
            if class_name.lower() not in ['person', 'pessoa']:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Converte para formato YOLO (normalizado)
            center_x = (xmin + xmax) / 2.0 / img_width
            center_y = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Salva anotação YOLO
        dest_annotation = self.output_dir / "labels" / split_name / f"{image_file.stem}.txt"
        with open(dest_annotation, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
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
    
    def create_synthetic_sports_dataset(self, num_images: int = 100):
        """
        Cria um dataset sintético de esportes para demonstração
        
        Args:
            num_images: Número de imagens a gerar
        """
        print(f"🎨 Criando dataset sintético com {num_images} imagens...")
        
        # Esportes simulados
        sports_types = ['running', 'jumping', 'swimming', 'cycling', 'tennis', 'soccer']
        
        for split in ['train', 'val', 'test']:
            split_size = int(num_images * 0.7) if split == 'train' else int(num_images * 0.15)
            
            for i in range(split_size):
                # Gera imagem sintética
                img = self.generate_synthetic_sports_image(sports_types[i % len(sports_types)])
                
                # Salva imagem
                img_path = self.output_dir / "images" / split / f"synthetic_{i:04d}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Cria anotação correspondente
                self.create_synthetic_annotation(img_path, split)
        
        print("✅ Dataset sintético criado!")
    
    def generate_synthetic_sports_image(self, sport_type: str) -> np.ndarray:
        """
        Gera uma imagem sintética de esporte
        
        Args:
            sport_type: Tipo de esporte
            
        Returns:
            Imagem sintética como array numpy
        """
        # Cria imagem base
        img = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
        
        # Adiciona elementos baseados no esporte
        if sport_type == 'running':
            # Pessoa correndo (retângulo alongado)
            x, y = np.random.randint(100, 400, 2)
            w, h = np.random.randint(30, 60), np.random.randint(80, 120)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), -1)
            
        elif sport_type == 'jumping':
            # Pessoa pulando (retângulo mais alto)
            x, y = np.random.randint(100, 400, 2)
            w, h = np.random.randint(25, 50), np.random.randint(100, 140)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), -1)
            
        elif sport_type == 'swimming':
            # Pessoa nadando (retângulo horizontal)
            x, y = np.random.randint(100, 400, 2)
            w, h = np.random.randint(60, 100), np.random.randint(40, 80)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), -1)
            
        else:
            # Esporte genérico
            x, y = np.random.randint(100, 400, 2)
            w, h = np.random.randint(40, 80), np.random.randint(60, 100)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), -1)
        
        # Adiciona ruído para realismo
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def create_synthetic_annotation(self, image_path: Path, split: str):
        """
        Cria anotação sintética para imagem
        
        Args:
            image_path: Caminho da imagem
            split: Nome do split
        """
        # Lê a imagem para obter dimensões
        img = cv2.imread(str(image_path))
        img_height, img_width = img.shape[:2]
        
        # Gera bounding box aleatória
        x = np.random.randint(50, img_width - 100)
        y = np.random.randint(50, img_height - 100)
        w = np.random.randint(30, 100)
        h = np.random.randint(50, 120)
        
        # Converte para formato YOLO
        center_x = (x + w/2) / img_width
        center_y = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        # Salva anotação
        label_path = self.output_dir / "labels" / split / f"{image_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
    
    def validate_dataset(self):
        """
        Valida a integridade do dataset preparado
        """
        print("🔍 Validando dataset...")
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / "images" / split
            labels_dir = self.output_dir / "labels" / split
            
            # Verifica se existem imagens
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            if not image_files:
                issues.append(f"Nenhuma imagem encontrada em {split}")
                continue
            
            # Verifica correspondência entre imagens e labels
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    issues.append(f"Label não encontrado para {img_file.name}")
        
        if issues:
            print("⚠️  Problemas encontrados:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Dataset validado com sucesso!")
        
        return len(issues) == 0

def main():
    """Função principal"""
    print("🏃‍♂️ Preparador de Dataset para Detecção de Pessoas em Esportes")
    print("=" * 60)
    
    # Inicializa o preparador
    preparer = SportsDatasetPreparer()
    
    # Cria dataset sintético para demonstração
    print("\n🎨 Criando dataset sintético...")
    preparer.create_synthetic_sports_dataset(num_images=200)
    
    # Valida o dataset
    print("\n🔍 Validando dataset...")
    preparer.validate_dataset()
    
    print("\n✅ Preparação concluída!")
    print("📁 Estrutura criada:")
    print("  - sports_dataset/images/ (train, val, test)")
    print("  - sports_dataset/labels/ (train, val, test)")
    print("  - sports_dataset/dataset.yaml (configuração)")

if __name__ == "__main__":
    main()
