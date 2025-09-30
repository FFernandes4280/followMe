@echo off
REM Script de ativação do ambiente virtual para Windows

echo 🏃‍♂️ Ativando ambiente virtual para Detecção de Pessoas em Esportes
echo ============================================================

REM Ativa ambiente virtual
call venv\Scripts\activate.bat

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
