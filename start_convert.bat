@echo off
chcp 65001 >nul
title Convert LabelMe to COCO
echo.
echo ============================================
echo   Convert LabelMe JSON → COCO JSON
echo ============================================
echo.
echo Activating fire_env1 ...
call E:\Anaconda\Scripts\activate.bat fire_env1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate fire_env1
    pause
    exit /b 1
)
echo Converting annotations ...
python convert_labelme_to_coco.py
if %errorlevel% neq 0 (
    echo [ERROR] Conversion failed
    pause
    exit /b 1
)
echo.
echo Done! Output: A_train/train.json  A_train/val.json
pause
