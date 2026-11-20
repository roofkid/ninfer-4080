@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
if not defined NINFER_MODEL_DIR set "NINFER_MODEL_DIR=%ROOT%\models"
if not defined NINFER_REVISION set "NINFER_REVISION=main"
set "MODEL=%NINFER_MODEL_DIR%\sensenova_loop_9b.ninfer"
set "EXPECTED_SHA256=e793869f6b4edb98ae1a8f08899a28f938ed77652796f5eaf572785cc48267e0"

if not exist "%NINFER_MODEL_DIR%" mkdir "%NINFER_MODEL_DIR%"
echo Downloading SenseNova-Loop-9B NInfer model (revision %NINFER_REVISION%)...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/roofkid/SenseNova-Loop-9B-NInfer/resolve/%NINFER_REVISION%/sensenova_loop_9b.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Expected SHA-256: %EXPECTED_SHA256%
echo Verify with: certutil -hashfile "%MODEL%" SHA256
echo Model ready: %MODEL%
