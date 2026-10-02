@echo off
rem ============================================================================
rem  NInfer on the RTX 4080, Windows host through Docker Desktop / WSL2.
rem
rem  Builds the product image from .\Dockerfile (first run, or when the checkout
rem  revision changed since the image was built) and serves the registered 3-bit
rem  GSQ artifact with the documented 100K profile:
rem
rem    * 102,400-token context, one lane, rk4v4-e8 KV (the registered 4080 fit)
rem    * MTP speculative decoding, three draft tokens, LM-head draft route
rem    * vision tower enabled (the GSQ3 artifact carries it)
rem    * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
rem      presence/frequency penalties 0. This engine has no multiplicative
rem      repeat penalty; the neutral value 1 is its implicit behavior.
rem
rem  The image carries only the binaries; the artifact is bind-mounted read-only,
rem  so the container stays small and never contains model weights. Every build
rem  stamps the image with the checkout revision (label org.ninfer.revision), so
rem  pulling source changes and re-running this launcher rebuilds the image
rem  instead of serving the previous build.
rem
rem  Measured on this card with the session-16 build
rem  (docs\maintainer\rtx-4080-plan.md section 11; tiled corpus, rk4v4-e8,
rem  --prefill-chunk 1024, one repetition per point): decode about 130/126/109
rem  tok/s at 8K/32K/98K depth and prefill about 2694/2298/1651 tok/s, against
rem  125/118/103 and 2687/2273/1610 on the session-10 build. The Q3 small-T
rem  tensor-core route and its A8 decode profile (sessions 10 and 16) are active
rem  at the MTP verify width; on the code scenario
rem  (examples\cli\messages\scenario_code_python.json) the A8 profile lifts
rem  greedy decode from 91.6 to 101.0 tok/s. The profile is fixed; edit this file
rem  to change it.
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
rem  NINFER_KV_DTYPE (default rk4v4-e8; int8 or rk8v4 trade context for MBPP-class accuracy).
rem ============================================================================
setlocal EnableExtensions
if not defined NINFER_KV_DTYPE set "NINFER_KV_DTYPE=rk4v4-e8"

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

set "SOURCE_REVISION="
where git >nul 2>nul
if not errorlevel 1 (
  for /f "delims=" %%I in ('git -C "%ROOT%" rev-parse --short=12 HEAD 2^>nul') do set "SOURCE_REVISION=%%I"
)
if defined SOURCE_REVISION (
  set "SOURCE_DIRTY="
  for /f "delims=" %%I in ('git -C "%ROOT%" status --porcelain 2^>nul') do set "SOURCE_DIRTY=1"
  if defined SOURCE_DIRTY set "SOURCE_REVISION=%SOURCE_REVISION%-dirty"
)

set "BUILD_IMAGE=0"
set "BUILD_REASON="
docker image inspect "%NINFER_IMAGE%" >nul 2>nul
if errorlevel 1 (
  set "BUILD_IMAGE=1"
  set "BUILD_REASON=first run"
) else if not defined SOURCE_REVISION (
  echo Note: git is unavailable or this checkout is not a git repository; using the existing image without a revision check.
) else (
  set "IMAGE_CURRENT="
  for /f "delims=" %%I in ('docker image inspect --format "{{ index .Config.Labels `org.ninfer.revision` }}" "%NINFER_IMAGE%" 2^>nul') do if "%%I"=="%SOURCE_REVISION%" set "IMAGE_CURRENT=1"
  if not defined IMAGE_CURRENT (
    set "BUILD_IMAGE=1"
    set "BUILD_REASON=source revision changed"
  )
)

if "%BUILD_IMAGE%"=="1" (
  echo Building %NINFER_IMAGE% from Dockerfile ^(%BUILD_REASON%; several minutes^)...
  if defined SOURCE_REVISION (
    docker build --label "org.ninfer.revision=%SOURCE_REVISION%" -f "%ROOT%\Dockerfile" -t "%NINFER_IMAGE%" "%ROOT%"
  ) else (
    docker build -f "%ROOT%\Dockerfile" -t "%NINFER_IMAGE%" "%ROOT%"
  )
  if errorlevel 1 exit /b 1
)

set "APIKEY_ARG="
if defined NINFER_API_KEY set "APIKEY_ARG=--api-key %NINFER_API_KEY%"

echo Serving the Qwen3.8-27B GSQ3 profile (%NINFER_KV_DTYPE% KV) on http://%NINFER_BIND%:%NINFER_PORT%/v1
docker run --rm ^
  --gpus all ^
  --add-host=host.docker.internal:host-gateway ^
  -p "%NINFER_BIND%:%NINFER_PORT%:8080" ^
  -v "%ARTIFACT_DIR%:/models:ro" ^
  "%NINFER_IMAGE%" ^
  ninfer-serve "/models/%ARTIFACT_NAME%" ^
  --host 0.0.0.0 --port 8080 ^
  --max-context 102400 --kv-capacity 102400 --kv-dtype %NINFER_KV_DTYPE% ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 ^
  --host-kv-mib 4096 ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --vision --preserve-thinking ^
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 ^
  --presence-penalty 0 --frequency-penalty 0 ^
  %APIKEY_ARG%
