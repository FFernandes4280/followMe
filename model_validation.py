#!/usr/bin/env python33
"""
Script de Validação e Teste para Modelo de Detecção de Pessoas em Esportes
Este script valida, testa e avalia o desempenho do modelo treinado.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import torch
import json
from typing import List, Dict, Tuple
import time

class SportsModelValidator:
    def __init__(self, model_path: str, test_data_dir: str = "sports_dataset"):
        """
        Inicializa o validador do modelo
        
        Args:
            model_path: Caminho para o modelo treinado
            test_data_dir: Diretório com dados de teste
        """
        self.model_path = model_path
        self.test_data_dir = Path(test_data_dir)
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Modelo carregado: {self.model_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def validate_on_test_set(self):
        """
        Valida o modelo no conjunto de teste
        """
        print("🔍 Validando modelo no conjunto de teste...")
        
        # Carrega configuração do dataset
        dataset_config = self.test_data_dir / "dataset.yaml"
        if not dataset_config.exists():
            print("❌ Arquivo de configuração do dataset não encontrado!")
            return None
        
        # Executa validação
        try:
            results = self.model.val(data=str(dataset_config), split='test')
            
            print("✅ Validação concluída!")
            print(f"📊 Métricas de validação:")
            print(f"  mAP50: {results.box.map50:.4f}")
            print(f"  mAP50-95: {results.box.map:.4f}")
            print(f"  Precision: {results.box.mp:.4f}")
            print(f"  Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro durante validação: {e}")
            return None
    
    def test_on_single_image(self, image_path: str, save_result: bool = True) -> Dict:
        """
        Testa o modelo em uma única imagem
        
        Args:
            image_path: Caminho para a imagem
            save_result: Se deve salvar o resultado
            
        Returns:
            Dicionário com resultados da detecção
        """
        print(f"🖼️  Testando em: {image_path}")
        
        try:
            # Executa detecção
            results = self.model(image_path)
            
            # Processa resultados
            result = results[0]
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    detection = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': 'person_sporting'
                    }
                    detections.append(detection)
            
            # Salva resultado se solicitado
            if save_result:
                self.save_detection_result(image_path, detections, result)
            
            return {
                'image_path': image_path,
                'detections': detections,
                'num_detections': len(detections)
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar imagem: {e}")
            return None
    
    def save_detection_result(self, image_path: str, detections: List[Dict], result):
        """
        Salva resultado da detecção com visualização
        
        Args:
            image_path: Caminho da imagem original
            detections: Lista de detecções
            result: Resultado do modelo YOLO
        """
        # Cria diretório de resultados
        results_dir = Path("detection_results")
        results_dir.mkdir(exist_ok=True)
        
        # Salva imagem com detecções
        output_path = results_dir / f"result_{Path(image_path).stem}.jpg"
        result.save(str(output_path))
        
        # Salva dados JSON
        json_path = results_dir / f"result_{Path(image_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image_path': image_path,
                'detections': detections,
                'timestamp': time.time()
            }, f, indent=2)
        
        print(f"💾 Resultado salvo: {output_path}")
    
    def benchmark_performance(self, test_images_dir: str, num_images: int = 20):
        """
        Executa benchmark de performance do modelo
        
        Args:
            test_images_dir: Diretório com imagens de teste
            num_images: Número de imagens para testar
        """
        print(f"⚡ Executando benchmark de performance...")
        
        test_dir = Path(test_images_dir)
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print("❌ Nenhuma imagem encontrada para benchmark!")
            return
        
        # Limita número de imagens
        image_files = image_files[:num_images]
        
        # Métricas de performance
        inference_times = []
        detections_per_image = []
        confidences = []
        
        print(f"📊 Testando {len(image_files)} imagens...")
        
        for i, img_path in enumerate(image_files):
            start_time = time.time()
            
            # Executa detecção
            results = self.model(str(img_path))
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Processa resultados
            result = results[0]
            if result.boxes is not None:
                num_detections = len(result.boxes)
                detections_per_image.append(num_detections)
                
                # Coleta confianças
                if num_detections > 0:
                    confs = result.boxes.conf.cpu().numpy()
                    confidences.extend(confs.tolist())
            else:
                detections_per_image.append(0)
            
            if (i + 1) % 10 == 0:
                print(f"  Processadas: {i + 1}/{len(image_files)}")
        
        # Calcula estatísticas
        avg_inference_time = np.mean(inference_times)
        fps = 1.0 / avg_inference_time
        avg_detections = np.mean(detections_per_image)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        print(f"\n📈 Resultados do Benchmark:")
        print(f"  Tempo médio de inferência: {avg_inference_time:.4f}s")
        print(f"  FPS médio: {fps:.2f}")
        print(f"  Detecções médias por imagem: {avg_detections:.2f}")
        print(f"  Confiança média: {avg_confidence:.4f}")
        
        # Salva resultados
        benchmark_results = {
            'num_images': len(image_files),
            'avg_inference_time': avg_inference_time,
            'fps': fps,
            'avg_detections_per_image': avg_detections,
            'avg_confidence': avg_confidence,
            'inference_times': inference_times,
            'detections_per_image': detections_per_image,
            'confidences': confidences
        }
        
        with open("benchmark_results.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"💾 Resultados salvos em: benchmark_results.json")
        
        return benchmark_results
    
    def create_performance_plots(self, benchmark_results: Dict):
        """
        Cria gráficos de performance
        
        Args:
            benchmark_results: Resultados do benchmark
        """
        print("📊 Criando gráficos de performance...")
        
        # Configura estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise de Performance do Modelo', fontsize=16)
        
        # Gráfico 1: Distribuição de tempos de inferência
        axes[0, 0].hist(benchmark_results['inference_times'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].axvline(benchmark_results['avg_inference_time'], color='red', linestyle='--', 
                          label=f'Média: {benchmark_results["avg_inference_time"]:.4f}s')
        axes[0, 0].set_title('Distribuição de Tempos de Inferência')
        axes[0, 0].set_xlabel('Tempo (s)')
        axes[0, 0].set_ylabel('Frequência')
        axes[0, 0].legend()
        
        # Gráfico 2: Detecções por imagem
        axes[0, 1].hist(benchmark_results['detections_per_image'], bins=20, alpha=0.7, color='green')
        axes[0, 1].axvline(benchmark_results['avg_detections_per_image'], color='red', linestyle='--',
                          label=f'Média: {benchmark_results["avg_detections_per_image"]:.2f}')
        axes[0, 1].set_title('Distribuição de Detecções por Imagem')
        axes[0, 1].set_xlabel('Número de Detecções')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].legend()
        
        # Gráfico 3: Confiança das detecções
        if benchmark_results['confidences']:
            axes[1, 0].hist(benchmark_results['confidences'], bins=20, alpha=0.7, color='orange')
            axes[1, 0].axvline(benchmark_results['avg_confidence'], color='red', linestyle='--',
                              label=f'Média: {benchmark_results["avg_confidence"]:.4f}')
            axes[1, 0].set_title('Distribuição de Confiança')
            axes[1, 0].set_xlabel('Confiança')
            axes[1, 0].set_ylabel('Frequência')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Nenhuma detecção encontrada', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Distribuição de Confiança')
        
        # Gráfico 4: Tempo vs Detecções
        axes[1, 1].scatter(benchmark_results['detections_per_image'], 
                          benchmark_results['inference_times'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Tempo de Inferência vs Número de Detecções')
        axes[1, 1].set_xlabel('Número de Detecções')
        axes[1, 1].set_ylabel('Tempo de Inferência (s)')
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráficos salvos em: performance_analysis.png")
    
    def test_different_thresholds(self, test_images_dir: str, thresholds: List[float] = None):
        """
        Testa o modelo com diferentes thresholds de confiança
        
        Args:
            test_images_dir: Diretório com imagens de teste
            thresholds: Lista de thresholds para testar
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        print(f"🎯 Testando diferentes thresholds: {thresholds}")
        
        test_dir = Path(test_images_dir)
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print("❌ Nenhuma imagem encontrada!")
            return
        
        # Limita para performance
        image_files = image_files[:20]
        
        threshold_results = {}
        
        for threshold in thresholds:
            print(f"  Testando threshold: {threshold}")
            
            total_detections = 0
            high_conf_detections = 0
            
            for img_path in image_files:
                results = self.model(str(img_path), conf=threshold)
                result = results[0]
                
                if result.boxes is not None:
                    num_detections = len(result.boxes)
                    total_detections += num_detections
                    
                    # Conta detecções com alta confiança
                    confidences = result.boxes.conf.cpu().numpy()
                    high_conf_count = np.sum(confidences >= threshold)
                    high_conf_detections += high_conf_count
            
            threshold_results[threshold] = {
                'total_detections': total_detections,
                'high_conf_detections': high_conf_detections,
                'avg_detections_per_image': total_detections / len(image_files)
            }
        
        # Exibe resultados
        print(f"\n📊 Resultados por Threshold:")
        print(f"{'Threshold':<10} {'Total Det.':<12} {'Alta Conf.':<12} {'Média/Img':<10}")
        print("-" * 50)
        
        for threshold, results in threshold_results.items():
            print(f"{threshold:<10.1f} {results['total_detections']:<12} "
                  f"{results['high_conf_detections']:<12} {results['avg_detections_per_image']:<10.2f}")
        
        # Salva resultados
        with open("threshold_analysis.json", 'w') as f:
            json.dump(threshold_results, f, indent=2)
        
        print(f"\n💾 Análise de thresholds salva em: threshold_analysis.json")
        
        return threshold_results
    
    def compare_with_baseline(self, baseline_model_path: str, test_images_dir: str):
        """
        Compara o modelo treinado com um modelo baseline
        
        Args:
            baseline_model_path: Caminho para modelo baseline
            test_images_dir: Diretório com imagens de teste
        """
        print("🆚 Comparando com modelo baseline...")
        
        try:
            baseline_model = YOLO(baseline_model_path)
        except Exception as e:
            print(f"❌ Erro ao carregar modelo baseline: {e}")
            return
        
        test_dir = Path(test_images_dir)
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print("❌ Nenhuma imagem encontrada!")
            return
        
        # Limita para performance
        image_files = image_files[:10]
        
        comparison_results = {
            'trained_model': {'detections': [], 'times': []},
            'baseline_model': {'detections': [], 'times': []}
        }
        
        for img_path in image_files:
            # Testa modelo treinado
            start_time = time.time()
            results_trained = self.model(str(img_path))
            time_trained = time.time() - start_time
            
            detections_trained = len(results_trained[0].boxes) if results_trained[0].boxes is not None else 0
            
            # Testa modelo baseline
            start_time = time.time()
            results_baseline = baseline_model(str(img_path))
            time_baseline = time.time() - start_time
            
            detections_baseline = len(results_baseline[0].boxes) if results_baseline[0].boxes is not None else 0
            
            # Armazena resultados
            comparison_results['trained_model']['detections'].append(detections_trained)
            comparison_results['trained_model']['times'].append(time_trained)
            comparison_results['baseline_model']['detections'].append(detections_baseline)
            comparison_results['baseline_model']['times'].append(time_baseline)
        
        # Calcula estatísticas
        trained_avg_detections = np.mean(comparison_results['trained_model']['detections'])
        baseline_avg_detections = np.mean(comparison_results['baseline_model']['detections'])
        trained_avg_time = np.mean(comparison_results['trained_model']['times'])
        baseline_avg_time = np.mean(comparison_results['baseline_model']['times'])
        
        print(f"\n📊 Comparação de Modelos:")
        print(f"{'Métrica':<20} {'Modelo Treinado':<15} {'Baseline':<15} {'Diferença':<15}")
        print("-" * 70)
        print(f"{'Detecções Médias':<20} {trained_avg_detections:<15.2f} {baseline_avg_detections:<15.2f} "
              f"{trained_avg_detections - baseline_avg_detections:<15.2f}")
        print(f"{'Tempo Médio (s)':<20} {trained_avg_time:<15.4f} {baseline_avg_time:<15.4f} "
              f"{trained_avg_time - baseline_avg_time:<15.4f}")
        
        # Salva comparação
        with open("model_comparison.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n💾 Comparação salva em: model_comparison.json")

def main():
    """Função principal"""
    print("🔍 Validador de Modelo para Detecção de Pessoas em Esportes")
    print("=" * 60)
    
    # Verifica se o modelo existe
    model_path = "sports_detection_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        print("   Execute primeiro o script de treinamento!")
        return
    
    # Inicializa validador
    validator = SportsModelValidator(model_path)
    
    # Valida no conjunto de teste
    print("\n🔍 Validando no conjunto de teste...")
    validation_results = validator.validate_on_test_set()
    
    if validation_results:
        # Executa benchmark de performance
        print("\n⚡ Executando benchmark...")
        benchmark_results = validator.benchmark_performance("sports_dataset/images/test")
        
        if benchmark_results:
            # Cria gráficos de performance
            print("\n📊 Criando gráficos...")
            validator.create_performance_plots(benchmark_results)
        
        # Testa diferentes thresholds
        print("\n🎯 Testando thresholds...")
        validator.test_different_thresholds("sports_dataset/images/test")
        
        # Compara com baseline (se disponível)
        baseline_path = "yolov8n.pt"
        if os.path.exists(baseline_path):
            print("\n🆚 Comparando com baseline...")
            validator.compare_with_baseline(baseline_path, "sports_dataset/images/test")
        
        print("\n✅ Validação concluída com sucesso!")
        print("📁 Arquivos gerados:")
        print("  - benchmark_results.json")
        print("  - threshold_analysis.json")
        print("  - model_comparison.json")
        print("  - performance_analysis.png")
        print("  - detection_results/ (imagens com detecções)")
    else:
        print("❌ Falha na validação!")

if __name__ == "__main__":
    main()
