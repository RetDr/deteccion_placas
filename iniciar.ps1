# Script PowerShell para iniciar el servidor ANPR con el interprete de Python correcto
Write-Host "===========================================================================" -ForegroundColor Cyan
Write-Host "SISTEMA DE RECONOCIMIENTO DE PLACAS VEHICULARES EN TIEMPO REAL (ANPR)" -ForegroundColor Green
Write-Host "Servidor Web: http://localhost:8000" -ForegroundColor Yellow
Write-Host "===========================================================================" -ForegroundColor Cyan

if (Get-Command python -ErrorAction SilentlyContinue) {
    python server.py
} elseif (Test-Path "C:\Users\danie\AppData\Local\Programs\Python\Python312\python.exe") {
    & "C:\Users\danie\AppData\Local\Programs\Python\Python312\python.exe" server.py
} else {
    py -3 server.py
}
