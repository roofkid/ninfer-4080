@echo off
rem ============================================================================
rem  NInfer on the RTX 4080, Windows host through Docker Desktop / WSL2.
rem
rem  Builds (once) the product image from .\Dockerfile and serves the registered
rem  3-bit GSQ artifact with the documented 100K profile:
rem
rem    * 102,400-token context, one lane, rk4v4-e8 KV (the registered 4080 fit)
rem    * MTP speculative decoding, three draft tokens, LM-head draft route
rem    * vision tower enabled (the GSQ3 artifact carries it)
rem    * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
rem      presence/frequency penalties 0. This engine has no multiplicative
rem      repeat penalty; the neutral value 1 is its implicit behavior.
rem
rem  The image carries only the binaries; the artifact is bind-mounted read-only,
rem  so the container stays small and never contains model weights. The image is
rem  built from the checkout as it exists now, which is what carries the current
rem  decode route; rebuild it after pulling source changes:
rem
rem    docker build -f Dockerfile -t ninfer-4080:gsq3 .
rem
rem  Measured on this card with the session-10 build
rem  (docs\maintainer\rtx-4080-plan.md section 11): decode about 125/118/103
rem  tok/s at 8K/32K/98K depth and prefill about 2700/2270/1675 tok/s.
rem  The profile is fixed; edit this file to change it.
rem
rem  The published port binds every host interface, so the profile is reachable
rem  from other machines on the network at http://<host-ip>:8080/v1 ^(allow the
rem  Docker Desktop firewall prompt on first use^). Set NINFER_API_KEY to require
rem  a bearer token; the server only checks it when this is set.
rem
rem  Environment overrides: NINFER_IMAGE (default ninfer-4080:gsq3),
rem  NINFER_ARTIFACT (default <repo>\out\qwen3_8_27b_gsq3.ninfer),
rem  NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to
rem  keep the port host-local), NINFER_API_KEY (optional).
rem ============================================================================
setlocal EnableExtensions

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if not defined NINFER_IMAGE set "NINFER_IMAGE=ninfer-4080:gsq3"
if not defined NINFER_PORT set "NINFER_PORT=8080"
if not defined NINFER_BIND set "NINFER_BIND=0.0.0.0"
if not defined NINFER_ARTIFACT set "NINFER_ARTIFACT=%ROOT%\out\qwen3_8_27b_gsq3.ninfer"

where docker >nul 2>nul
if errorlevel 1 (
  echo docker was not found on PATH. Install Docker Desktop with the WSL2 backend.
  exit /b 1
)
if not exist "%NINFER_ARTIFACT%" (
  echo Missing %NINFER_ARTIFACT%
  echo Convert or copy the 3-bit GSQ artifact there first ^(its recipe is in docs\maintainer\rtx-4080-plan.md^).
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

echo Serving the Qwen3.8-27B GSQ3 profile on http://%NINFER_BIND%:%NINFER_PORT%/v1
docker run --rm ^
  --gpus all ^
  --add-host=host.docker.internal:host-gateway ^
  -p "%NINFER_BIND%:%NINFER_PORT%:8080" ^
  -v "%ARTIFACT_DIR%:/models:ro" ^
  "%NINFER_IMAGE%" ^
  ninfer-serve "/models/%ARTIFACT_NAME%" ^
  --host 0.0.0.0 --port 8080 ^
  --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 ^
  --host-kv-mib 4096 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --vision --preserve-thinking ^
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 ^
  --presence-penalty 0 --frequency-penalty 0 ^
  %APIKEY_ARG%
