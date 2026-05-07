@echo off
echo ========================================
echo   AI Giam Sat Giao Thong - Khoi dong...
echo ========================================
echo.

REM Kich hoat venv Python 3.10 (co RapidOCR + tat ca thu vien)
call "%~dp0venv_paddle\Scripts\activate.bat"

echo [OK] Moi truong Python 3.10 da san sang!
echo [OK] RapidOCR + YOLO + SAM2 deu da load...
echo.
echo Dang khoi dong Streamlit...
echo Trinh duyet se tu mo tai: http://localhost:8501
echo.
echo (Nhan Ctrl+C de dung app)
echo.

cd /d "%~dp0"
streamlit run app.py

pause
