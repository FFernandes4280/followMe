#!/usr/bin/env python3
"""
Script de Instalação com Ambiente Virtual
Solução para o erro "externally-managed-environment" em sistemas Ubuntu/Debian
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    print("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado.")
        print("   Requerido: Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_venv_available():
    """Verifica se o módulo venv está disponível"""
    print("\n🔧 Verificando suporte a ambiente virtual...")
    
    try:
        import venv
        print("✅ Módulo venv disponível")
        return True
    except ImportError:
        print("❌ Módulo venv não encontrado")
        print("   Instale: sudo apt install python3-venv")
        return False

def create_virtual_environment():
    """Cria ambiente virtual"""
    print("\n📦 Criando ambiente virtual...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Ambiente virtual já existe. Removendo...")
        import shutil
        shutil.rmtree(venv_path)
    
    try:
        subprocess.run([
            sys.executable, "-m", "venv", "venv"
        ], check=True)
        print("✅ Ambiente virtual criado: venv/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao criar ambiente virtual: {e}")
        return False

def get_venv_python():
    """Retorna caminho para Python do ambiente virtual"""
    if platform.system().lower() == "windows":
        return "venv/Scripts/python.exe"
    else:
        return "venv/bin/python"

def get_venv_pip():
    """Retorna caminho para pip do ambiente virtual"""
    if platform.system().lower() == "windows":
        return "venv/Scripts/pip.exe"
    else:
        return "venv/bin/pip"

def install_dependencies():
    """Instala dependências no ambiente virtual"""
    print("\n📦 Instalando dependências no ambiente virtual...")
    
    venv_python = get_venv_python()
    venv_pip = get_venv_pip()
    
    try:
        # Atualiza pip no ambiente virtual
        print("   Atualizando pip...")
        subprocess.run([
            venv_python, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        # Instala dependências
        print("   Instalando dependências...")
        subprocess.run([
            venv_pip, "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Dependências instaladas com sucesso!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def test_installation():
    """Testa a instalação no ambiente virtual"""
    print("\n🧪 Testando instalação...")
    
    venv_python = get_venv_python()
    
    try:
        # Testa imports principais
        test_script = """
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
print("✅ Todos os imports funcionaram!")
"""
        
        result = subprocess.run([
            venv_python, "-c", test_script
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Teste de instalação passou!")
            return True
        else:
            print(f"❌ Erro no teste: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def create_activation_scripts():
    """Cria scripts de ativação"""
    print("\n📝 Criando scripts de ativação...")
    
    # Script para Linux/macOS
    activate_script = """#!/bin/bash
# Script de ativação do ambiente virtual

echo "🏃‍♂️ Ativando ambiente virtual para Detecção de Pessoas em Esportes"
echo "=" * 60

# Ativa ambiente virtual
source venv/bin/activate

echo "✅ Ambiente virtual ativado!"
echo ""
echo "🚀 Comandos disponíveis:"
echo "  python sports_detection_system.py"
echo "  python quick_start.py"
echo "  python test_python3.py"
echo ""
echo "💡 Para desativar: deactivate"
echo ""

# Executa sistema principal
python sports_detection_system.py
"""
    
    with open("activate_and_run.sh", "w") as f:
        f.write(activate_script)
    
    os.chmod("activate_and_run.sh", 0o755)
    
    # Script para Windows
    activate_script_win = """@echo off
REM Script de ativação do ambiente virtual para Windows

echo 🏃‍♂️ Ativando ambiente virtual para Detecção de Pessoas em Esportes
echo ============================================================

REM Ativa ambiente virtual
call venv\\Scripts\\activate.bat

echo ✅ Ambiente virtual ativado!
echo.
echo 🚀 Comandos disponíveis:
echo   python sports_detection_system.py
echo   python quick_start.py
echo   python test_python3.py
echo.
echo 💡 Para desativar: deactivate
echo.

REM Executa sistema principal
python sports_detection_system.py
"""
    
    with open("activate_and_run.bat", "w") as f:
        f.write(activate_script_win)
    
    print("✅ Scripts de ativação criados:")
    print("   - activate_and_run.sh (Linux/macOS)")
    print("   - activate_and_run.bat (Windows)")

def create_venv_launcher():
    """Cria launcher que usa ambiente virtual"""
    print("\n🚀 Criando launcher com ambiente virtual...")
    
    launcher_content = """#!/usr/bin/env python3
\"\"\"
Launcher com Ambiente Virtual para o Sistema de Detecção de Pessoas em Esportes
\"\"\"

import os
import sys
import subprocess
from pathlib import Path

def get_venv_python():
    \"\"\"Retorna caminho para Python do ambiente virtual\"\"\"
    if os.name == 'nt':  # Windows
        return "venv\\\\Scripts\\\\python.exe"
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
    
    choice = input("\\nDigite sua escolha (0-7): ").strip()
    
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
"""
    
    with open("launcher_venv.py", "w") as f:
        f.write(launcher_content)
    
    os.chmod("launcher_venv.py", 0o755)
    print("✅ Launcher criado: launcher_venv.py")

def main():
    """Função principal de instalação com ambiente virtual"""
    print("🏃‍♂️ Instalador com Ambiente Virtual - Sistema de Detecção de Pessoas em Esportes")
    print("=" * 80)
    print("💡 Este instalador resolve o erro 'externally-managed-environment'")
    print("   criando um ambiente virtual isolado para o projeto.")
    print()
    
    # Verificações
    if not check_python_version():
        sys.exit(1)
    
    if not check_venv_available():
        print("\n🔧 Para instalar suporte a ambiente virtual:")
        print("   sudo apt install python3-venv")
        sys.exit(1)
    
    # Criação do ambiente virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalação de dependências
    if not install_dependencies():
        print("\n❌ Falha na instalação das dependências!")
        sys.exit(1)
    
    # Teste da instalação
    if not test_installation():
        print("\n❌ Falha no teste de instalação!")
        sys.exit(1)
    
    # Criação de scripts auxiliares
    create_activation_scripts()
    create_venv_launcher()
    
    # Resumo final
    print("\n🎉 Instalação com ambiente virtual concluída com sucesso!")
    print("=" * 60)
    
    print("\n📁 Arquivos criados:")
    print("   - venv/ (ambiente virtual)")
    print("   - activate_and_run.sh (Linux/macOS)")
    print("   - activate_and_run.bat (Windows)")
    print("   - launcher_venv.py (launcher Python)")
    
    print("\n🚀 Para usar o sistema:")
    print("   Opção 1 - Script de ativação:")
    print("     ./activate_and_run.sh")
    print("     # ou no Windows: activate_and_run.bat")
    print()
    print("   Opção 2 - Launcher Python:")
    print("     python3 launcher_venv.py")
    print()
    print("   Opção 3 - Ativação manual:")
    print("     source venv/bin/activate  # Linux/macOS")
    print("     # ou venv\\Scripts\\activate  # Windows")
    print("     python sports_detection_system.py")
    
    print("\n💡 Vantagens do ambiente virtual:")
    print("   ✅ Isolado do sistema")
    print("   ✅ Sem conflitos de dependências")
    print("   ✅ Fácil de remover (apenas delete a pasta venv/)")
    print("   ✅ Funciona em sistemas com 'externally-managed-environment'")

if __name__ == "__main__":
    main()
