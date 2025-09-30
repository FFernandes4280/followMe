#!/usr/bin/env python3
"""
Script Principal do Sistema FollowMe com Dataset Red Bull
Executa todo o pipeline: processamento -> treinamento -> execução
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Executa um comando e mostra o resultado"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar: {command}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_files():
    """Verifica se os arquivos necessários existem"""
    required_files = [
        "redbull_dataset_processor.py",
        "sports_detection_training.py", 
        "followMe.py",
        "config.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Arquivos necessários não encontrados:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def main():
    """Função principal"""
    print("🏃‍♂️ Sistema FollowMe com Dataset Red Bull")
    print("=" * 50)
    
    # Verifica arquivos necessários
    if not check_files():
        return
    
    # 1. Processa dataset do Red Bull
    if not Path("sports_data/dataset.yaml").exists():
        print("\n📹 Processando dataset do Red Bull...")
        if not run_command("python3 redbull_dataset_processor.py", "Processamento do Dataset"):
            print("❌ Falha no processamento do dataset!")
            return
    else:
        print("✅ Dataset do Red Bull já processado!")
    
    # 2. Treina modelo
    if not Path("sports_detection_best.pt").exists():
        print("\n🚀 Treinando modelo...")
        if not run_command("python3 sports_detection_training.py", "Treinamento do Modelo"):
            print("❌ Falha no treinamento!")
            return
    else:
        print("✅ Modelo já treinado!")
    
    # 3. Executa sistema
    print("\n🎯 Iniciando sistema FollowMe...")
    print("Pressione 'q' para sair, 'o' para alternar grade de ocupação")
    
    if not run_command("python3 followMe.py", "Sistema FollowMe"):
        print("❌ Erro ao executar sistema!")

if __name__ == "__main__":
    main()
