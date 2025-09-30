#!/usr/bin/env python3
"""
Launcher com Ambiente Virtual para o Sistema de Detecção de Pessoas em Esportes
"""

import os
import sys
import subprocess
from pathlib import Path

def get_venv_python():
    """Retorna caminho para Python do ambiente virtual"""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\python.exe"
    else:  # Linux/macOS
        return "venv/bin/python"

def main():
    print("🏃‍♂️ Sistema de Detecção de Pessoas em Esportes (Ambiente Virtual)")
    print("=" * 70)
    
    venv_python = get_venv_python()
    
    # Verifica se ambiente virtual existe
    if not Path(venv_python).exists():
        print("❌ Ambiente virtual não encontrado!")
        print("   Execute primeiro: python3 setup_venv.py")
        return
    
    print("Escolha uma opção:")
    print("1. Detecção em tempo real")
    print("2. Treinamento de modelo")
    print("3. Validação de modelo")
    print("4. Preparação de dataset")
    print("5. Exemplos de uso")
    print("6. Teste de compatibilidade")
    print("7. Sistema unificado")
    print("0. Sair")
    
    choice = input("\nDigite sua escolha (0-7): ").strip()
    
    scripts = {
        "1": "sports_detection_realtime.py",
        "2": "sports_detection_training.py", 
        "3": "model_validation.py",
        "4": "dataset_preparation.py",
        "5": "example_usage.py",
        "6": "test_python3.py",
        "7": "sports_detection_system.py"
    }
    
    if choice == "0":
        print("👋 Até logo!")
    elif choice in scripts:
        script = scripts[choice]
        print(f"🚀 Executando: {script}")
        subprocess.run([venv_python, script])
    else:
        print("❌ Opção inválida!")

if __name__ == "__main__":
    main()
