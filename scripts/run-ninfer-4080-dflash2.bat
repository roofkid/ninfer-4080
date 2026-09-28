@echo off
rem ============================================================================
rem  NInfer on the RTX 4080 with the DFlash2 draft model, Windows host through
rem  Docker Desktop / WSL2.
rem
rem  Builds (once) the product image from .\Dockerfile and serves the DFlash2
rem  profile at the context that fits the 16 GiB card with the stock companion:
rem
rem    * 57,344-token context without vision, 28,672 with vision (measured caps;
rem      set NINFER_CONTEXT to override)
rem    * DFlash2 block drafting with 7 draft tokens (verify width 8, the widest
rem      window on the fast small-T tensor-core route), LM-head proposal selector
rem    * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
rem      presence/frequency penalties 0. This engine has no multiplicative repeat
rem      penalty; the neutral value 1 is its implicit behavior.
rem
rem  Measured on this card (docs\maintainer\rtx-4080-plan.md section 11): DFlash2
rem  K=7 decodes about 122/203/180 tok/s at 8K/28K/56K depth against MTP3's
rem  125/118/103, is greedy-lossless against ordinary decoding, and needs the
rem  smaller context because the companion weights and ring state cost about
rem  1.9 GiB more than MTP. MTP stays the 100K profile (run-ninfer-4080.bat);
rem  DFlash2 is the deeper-context speed option.
rem
rem  The image carries only the binaries; the artifact is bind-mounted read-only.
rem  Rebuild the image after pulling source changes:
rem
rem    docker build -f Dockerfile -t ninfer-4080:gsq3 .
rem
rem  Environment overrides: NINFER_IMAGE (default ninfer-4080:gsq3),
rem  NINFER_ARTIFACT (default <repo>\out\qwen3_8_27b_gsq3.ninfer),
rem  NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to
rem  keep the port host-local), NINFER_VISION (1 enables media and selects the
rem  vision cap), NINFER_CONTEXT (explicit token cap overrides both),
rem  NINFER_API_KEY (optional).
rem ============================================================================
setlocal EnableExtensions

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if not defined NINFER_IMAGE set "NINFER_IMAGE=ninfer-4080:gsq3"
if not defined NINFER_PORT set "NINFER_PORT=8080"
if not defined NINFER_BIND set "NINFER_BIND=0.0.0.0"
if not defined NINFER_ARTIFACT set "NINFER_ARTIFACT=%ROOT%\out\qwen3_8_27b_gsq3.ninfer"
if not defined NINFER_VISION set "NINFER_VISION=0"

set "VISION_ARG="
if "%NINFER_VISION%"=="0" (
  if not defined NINFER_CONTEXT set "NINFER_CONTEXT=57344"
) else (
  if not defined NINFER_CONTEXT set "NINFER_CONTEXT=28672"
  set "VISION_ARG=--vision"
)

where docker >nul 2>nul
if errorlevel 1 (
  echo docker was not found on PATH. Install Docker Desktop with the WSL2 backend.
  exit /b 1
)
if not exist "%NINFER_ARTIFACT%" (
  echo Missing %NINFER_ARTIFACT%
  echo Convert the 3-bit GSQ artifact with its DFlash2 companion first ^(docs\maintainer\rtx-4080-plan.md^).
  exit /b 1
)
for %%I in ("%NINFER_ARTIFACT%") do (
  set "ARTIFACT_DIR=%%~dpI"
  set "ARTIFACT_NAME=%%~nxI"
)
set "ARTIFACT_DIR=%ARTIFACT_DIR:~0,-1%"

docker image inspect "%NINFER_IMAGE%" >nul 2>nul
if errorlevel 1 (
  echo Building %NINFER_IMAGE% from Dockerfile ^(first run only, several minutes^)...
  docker build -f "%ROOT%\Dockerfile" -t "%NINFER_IMAGE%" "%ROOT%"
  if errorlevel 1 exit /b 1
)

set "APIKEY_ARG="
if defined NINFER_API_KEY set "APIKEY_ARG=--api-key %NINFER_API_KEY%"

echo Serving DFlash2 K=7 at %NINFER_CONTEXT% tokens on http://%NINFER_BIND%:%NINFER_PORT%/v1
docker run --rm ^
  --gpus all ^
  --add-host=host.docker.internal:host-gateway ^
  -p "%NINFER_BIND%:%NINFER_PORT%:8080" ^
  -v "%ARTIFACT_DIR%:/models:ro" ^
  "%NINFER_IMAGE%" ^
  ninfer-serve "/models/%ARTIFACT_NAME%" ^
  --host 0.0.0.0 --port 8080 ^
  --max-context %NINFER_CONTEXT% --kv-capacity %NINFER_CONTEXT% --kv-dtype rk4v4-e8 ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 ^
  --host-kv-mib 4096 ^
  --spec dflash2 --draft-tokens 7 --lm-head-draft ^
  %VISION_ARG% ^
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 ^
  --presence-penalty 0 --frequency-penalty 0 ^
  %APIKEY_ARG%
