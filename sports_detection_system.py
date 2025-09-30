#!/usr/bin/env python33
"""
Sistema Unificado de Detecção de Pessoas em Esportes
Este script integra todas as funcionalidades do sistema em uma interface unificada.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

class SportsDetectionSystem:
    def __init__(self, config_path="config.yaml"):
        """
        Inicializa o sistema unificado
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self.load_config(config_path)
        self.setup_logging()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuração carregada: {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Arquivo de configuração não encontrado: {config_path}")
            print("   Usando configurações padrão")
            return self.get_default_config()
        except Exception as e:
            print(f"❌ Erro ao carregar configuração: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão"""
        return {
            'model': {
                'size': 'n',
                'path': 'sports_detection_best.pt',
                'confidence_threshold': 0.3,
                'nms_threshold': 0.4
            },
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'image_size': 640,
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'patience': 20,
                'device': 'auto'
            },
            'realtime': {
                'camera_id': 0,
                'width': 640,
                'height': 480,
                'show_grid': True,
                'show_occupation': True,
                'save_video': False,
                'output_dir': 'detection_results'
            }
        }
    
    def setup_logging(self):
        """Configura sistema de logging"""
        import logging
        
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Configura logging básico
        logging.basicConfig(
            level=log_level,
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Salva em arquivo se configurado
        if log_config.get('save_to_file', False):
            log_dir = Path(log_config.get('log_dir', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'sports_detection.log')
            file_handler.setLevel(log_level)
            logging.getLogger().addHandler(file_handler)
    
    def prepare_dataset(self, images_dir: str = None, annotations_dir: str = None):
        """Prepara dataset para treinamento"""
        print("📁 Preparando dataset...")
        
        try:
            from dataset_preparation import SportsDatasetPreparer
            
            dataset_config = self.config.get('dataset', {})
            base_dir = dataset_config.get('base_dir', 'sports_data')
            
            preparer = SportsDatasetPreparer(base_dir)
            
            if images_dir and annotations_dir:
                # Usa dataset real
                preparer.prepare_custom_dataset(
                    images_dir=images_dir,
                    annotations_dir=annotations_dir,
                    annotation_format="yolo"
                )
            else:
                # Cria dataset sintético
                synthetic_config = self.config.get('synthetic', {})
                num_images = synthetic_config.get('num_images', 1000)
                preparer.create_synthetic_sports_dataset(num_images)
            
            print("✅ Dataset preparado com sucesso!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao preparar dataset: {e}")
            return False
    
    def train_model(self):
        """Treina modelo personalizado"""
        print("🏋️ Treinando modelo...")
        
        try:
            from sports_detection_training import SportsDetectionTrainer
            
            model_config = self.config.get('model', {})
            training_config = self.config.get('training', {})
            
            trainer = SportsDetectionTrainer(
                data_dir=self.config.get('dataset', {}).get('base_dir', 'sports_data'),
                model_size=model_config.get('size', 'n')
            )
            
            # Treina modelo
            results = trainer.train_model(
                epochs=training_config.get('epochs', 100),
                batch_size=training_config.get('batch_size', 16),
                img_size=training_config.get('image_size', 640)
            )
            
            if results:
                print("✅ Modelo treinado com sucesso!")
                return True
            else:
                print("❌ Falha no treinamento!")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao treinar modelo: {e}")
            return False
    
    def validate_model(self):
        """Valida modelo treinado"""
        print("🔍 Validando modelo...")
        
        try:
            from model_validation import SportsModelValidator
            
            model_config = self.config.get('model', {})
            model_path = model_config.get('path', 'sports_detection_best.pt')
            
            if not Path(model_path).exists():
                print(f"❌ Modelo não encontrado: {model_path}")
                return False
            
            validator = SportsModelValidator(model_path)
            
            # Valida no conjunto de teste
            results = validator.validate_on_test_set()
            
            if results:
                # Executa benchmark
                validation_config = self.config.get('validation', {})
                benchmark_images = validation_config.get('benchmark_images', 50)
                
                benchmark_results = validator.benchmark_performance(
                    "sports_data/images/test",
                    num_images=benchmark_images
                )
                
                if benchmark_results:
                    print("✅ Validação concluída com sucesso!")
                    return True
            
            print("❌ Falha na validação!")
            return False
            
        except Exception as e:
            print(f"❌ Erro ao validar modelo: {e}")
            return False
    
    def run_realtime_detection(self):
        """Executa detecção em tempo real"""
        print("🎥 Iniciando detecção em tempo real...")
        
        try:
            from sports_detection_realtime import SportsDetectionRealtime
            
            model_config = self.config.get('model', {})
            realtime_config = self.config.get('realtime', {})
            
            detector = SportsDetectionRealtime(
                model_path=model_config.get('path', 'sports_detection_best.pt'),
                confidence_threshold=model_config.get('confidence_threshold', 0.3)
            )
            
            # Configura câmera
            if detector.setup_camera(
                camera_id=realtime_config.get('camera_id', 0),
                width=realtime_config.get('width', 640),
                height=realtime_config.get('height', 480)
            ):
                # Executa detecção
                detector.run_detection(
                    show_grid=realtime_config.get('show_grid', True),
                    show_occupation=realtime_config.get('show_occupation', True),
                    save_video=realtime_config.get('save_video', False)
                )
                return True
            else:
                print("❌ Erro ao configurar câmera!")
                return False
                
        except Exception as e:
            print(f"❌ Erro na detecção em tempo real: {e}")
            return False
    
    def detect_image(self, image_path: str, output_path: str = None):
        """Detecta pessoas em imagem estática"""
        print(f"🖼️ Detectando em: {image_path}")
        
        try:
            from sports_detection_realtime import SportsDetectionRealtime
            
            model_config = self.config.get('model', {})
            
            detector = SportsDetectionRealtime(
                model_path=model_config.get('path', 'sports_detection_best.pt'),
                confidence_threshold=model_config.get('confidence_threshold', 0.3)
            )
            
            detector.detect_from_image(image_path, output_path)
            print("✅ Detecção em imagem concluída!")
            return True
            
        except Exception as e:
            print(f"❌ Erro na detecção de imagem: {e}")
            return False
    
    def run_complete_workflow(self):
        """Executa fluxo completo: dataset -> treino -> validação -> detecção"""
        print("🔄 Executando fluxo completo...")
        
        steps = [
            ("Preparação de Dataset", lambda: self.prepare_dataset()),
            ("Treinamento de Modelo", lambda: self.train_model()),
            ("Validação de Modelo", lambda: self.validate_model()),
        ]
        
        results = {}
        
        for step_name, step_function in steps:
            print(f"\n🔄 Executando: {step_name}")
            try:
                result = step_function()
                results[step_name] = result
                status = "✅ Sucesso" if result else "❌ Falha"
                print(f"   {status}")
                
                if not result:
                    print(f"   ⏹️  Parando fluxo devido a falha em: {step_name}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Erro: {e}")
                results[step_name] = False
                break
        
        # Resumo
        print(f"\n📊 Resumo do Fluxo:")
        for step_name, result in results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {step_name}")
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"\n🎯 Taxa de sucesso: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        return success_count == total_count

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Sistema Unificado de Detecção de Pessoas em Esportes")
    parser.add_argument("--config", default="config.yaml", help="Arquivo de configuração")
    parser.add_argument("--mode", choices=[
        "prepare", "train", "validate", "realtime", "image", "complete", "interactive"
    ], default="interactive", help="Modo de operação")
    parser.add_argument("--images", help="Diretório com imagens para dataset")
    parser.add_argument("--annotations", help="Diretório com anotações para dataset")
    parser.add_argument("--input-image", help="Imagem para detecção")
    parser.add_argument("--output-image", help="Imagem de saída")
    
    args = parser.parse_args()
    
    # Inicializa sistema
    system = SportsDetectionSystem(args.config)
    
    print("🏃‍♂️ Sistema de Detecção de Pessoas em Esportes")
    print("=" * 50)
    
    if args.mode == "interactive":
        # Modo interativo
        while True:
            print("\nEscolha uma opção:")
            print("1. Preparar dataset")
            print("2. Treinar modelo")
            print("3. Validar modelo")
            print("4. Detecção em tempo real")
            print("5. Detecção em imagem")
            print("6. Fluxo completo")
            print("0. Sair")
            
            choice = input("\nDigite sua escolha (0-6): ").strip()
            
            if choice == "0":
                print("👋 Até logo!")
                break
            elif choice == "1":
                system.prepare_dataset()
            elif choice == "2":
                system.train_model()
            elif choice == "3":
                system.validate_model()
            elif choice == "4":
                system.run_realtime_detection()
            elif choice == "5":
                image_path = input("Caminho da imagem: ").strip()
                if image_path:
                    system.detect_image(image_path)
            elif choice == "6":
                system.run_complete_workflow()
            else:
                print("❌ Opção inválida!")
    
    elif args.mode == "prepare":
        system.prepare_dataset(args.images, args.annotations)
    
    elif args.mode == "train":
        system.train_model()
    
    elif args.mode == "validate":
        system.validate_model()
    
    elif args.mode == "realtime":
        system.run_realtime_detection()
    
    elif args.mode == "image":
        if not args.input_image:
            print("❌ Especifique --input-image")
            sys.exit(1)
        system.detect_image(args.input_image, args.output_image)
    
    elif args.mode == "complete":
        system.run_complete_workflow()

if __name__ == "__main__":
    main()
