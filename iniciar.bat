@echo off
title Sistema ANPR Colombia - Servidor en Tiempo Real
echo ===========================================================================
echo SISTEMA DE RECONOCIMIENTO DE PLACAS VEHICULARES EN TIEMPO REAL (ANPR)
echo Servidor Web: http://localhost:8000
echo ===========================================================================

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" server.py
) else if exist "C:\Users\danie\AppData\Local\Programs\Python\Python312\python.exe" (
    "C:\Users\danie\AppData\Local\Programs\Python\Python312\python.exe" server.py
) else (
    py -3.12 server.py || py -3 server.py || python server.py
)
pause
