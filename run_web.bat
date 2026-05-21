@echo off
set YOLO_AUTOINSTALL=False
echo ============================================
echo  AI Traffic Monitor - Web App (React + FastAPI)
echo ============================================
echo.
echo [1/2] Kiem tra backend...
echo [OK] Backend ready
echo.
echo [2/2] Khoi dong servers...
echo.
echo  Backend  ^> http://localhost:8000  (FastAPI AI)
echo  Frontend ^> http://localhost:3000  (React UI)  ^<-- Mo cai nay tren browser!
echo.
echo Nhan Ctrl+C de dung ca hai server
echo.

:: Chay FastAPI backend trong background
start "FastAPI Backend" cmd /k "set YOLO_AUTOINSTALL=False && venv_paddle\Scripts\uvicorn backend.main:app --host 0.0.0.0 --port 8000"

:: Doi 3 giay de backend khoi dong
timeout /t 3 /nobreak > nul

:: Chay React frontend
cd frontend
start "React Frontend" cmd /k "npm run dev"
cd ..

echo.
echo [OK] Ca hai server dang chay!
echo Mo browser tai: http://localhost:3000
pause
