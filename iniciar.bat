@echo off
title Sistema ANPR Colombia - Servidor en Tiempo Real
echo ===========================================================================
echo SISTEMA DE RECONOCIMIENTO DE PLACAS VEHICULARES EN TIEMPO REAL (ANPR)
echo Servidor Web: http://localhost:8000
echo ===========================================================================
cd /d "%~dp0"
if exist "C:\Users\camil\AppData\Local\Python\bin\python.exe" (
    "C:\Users\camil\AppData\Local\Python\bin\python.exe" server.py
) else if exist "C:\Users\danie\AppData\Local\Programs\Python\Python312\python.exe" (
    "C:\Users\danie\AppData\Local\Programs\Python\Python312\python.exe" server.py
) else (
    python server.py
)
pause
