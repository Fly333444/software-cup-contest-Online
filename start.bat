@echo off
chcp 65001 >nul
title Annotation Tool - Fire Detection
echo.
echo ============================================
echo   Fire Detection Annotation Tool
echo ============================================
echo.
echo Activating fire_env1 ...
call E:\Anaconda\Scripts\activate.bat fire_env1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate fire_env1
    pause
    exit /b 1
)
echo Starting annotation tool ...
echo Open http://localhost:5001 in your browser
echo.
python annotation_tool.py
if %errorlevel% neq 0 (
    echo [ERROR] Annotation tool exited with error
    pause
    exit /b 1
)
pause
