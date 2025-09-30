#!/bin/bash
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
