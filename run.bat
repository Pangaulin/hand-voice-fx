@echo off
rem Lance Hand Vocal FX dans son environnement virtuel (Windows).
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    echo Venv introuvable. Cree-le d'abord avec :
    echo   python -m venv .venv
    echo   .venv\Scripts\python -m pip install -r requirements.txt
    echo puis relance ce fichier.
)