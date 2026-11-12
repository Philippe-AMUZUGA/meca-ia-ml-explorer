@echo off
rem MécaML Explorer : lance l'application avec le Python 3.12 du poste (aucune installation du paquet requise).
rem Un clic suffit. Les options de "python -m mecaml" passent telles quelles : Lancer_MecaML.bat -t pour l'essai.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
set "PYCMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYCMD=py -3.12"
)
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYCMD=python"
    )
)

if not defined PYCMD (
    echo.
    echo MécaML Explorer ne trouve pas Python 3.12 sur ce poste.
    echo.
    echo   1. Installez Python 3.12 depuis https://www.python.org/downloads/ ^(laissez cochée l'option "tcl/tk and IDLE"^).
    echo   2. Dans une fenêtre de commandes, dans ce dossier : py -3.12 -m pip install -r requirements.txt
    echo   3. Relancez ce fichier.
    echo.
    pause
    exit /b 1
)

%PYCMD% -m mecaml %*
if errorlevel 1 (
    echo.
    echo L'application s'est arrêtée avec une erreur. Vérifiez que Python 3.12 et ses bibliothèques sont installés :
    echo   %PYCMD% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
endlocal
