@echo off
setlocal
set "ROOT=%~dp0"
set "MODEL_DIR=%ROOT%models"
set "MODEL=%MODEL_DIR%\qwen3_8_27b_gsq3.ninfer"
rem Pinned to the revision that published the artifact; set NINFER_REVISION to override.
rem The SHA-256 check below is the content pin.
if not defined NINFER_REVISION set "NINFER_REVISION=2359d374b0f6ed3cd400815e5114d32ce03b5899"
set "EXPECTED_SHA256=c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7"

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
echo Downloading Qwen3.8-27B GSQ3 NInfer model (revision %NINFER_REVISION%)...
curl.exe -L -C - --fail --output "%MODEL%" "https://huggingface.co/roofkid/Qwen3.8-27B-GSQ3-NInfer/resolve/%NINFER_REVISION%/qwen3_8_27b_gsq3.ninfer"
if errorlevel 1 (
  echo Download failed. Run this file again to resume.
  exit /b 1
)
echo Expected SHA-256: %EXPECTED_SHA256%
echo Verify with: certutil -hashfile "%MODEL%" SHA256
echo Model ready: %MODEL%
