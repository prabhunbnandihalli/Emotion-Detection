@echo off
echo ========================================
echo   EmoSense - Emotion Detection App
echo ========================================
echo.
echo [1/2] Starting Flask backend...
start "EmoSense API" cmd /k "python app.py"
timeout /t 3 /nobreak >nul
echo [2/2] Opening frontend in browser...
start index.html
echo.
echo Done! API running at http://localhost:5000
echo Open index.html in your browser if it didn't open automatically.
pause
