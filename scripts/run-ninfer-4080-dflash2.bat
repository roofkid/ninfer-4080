@echo off
rem ============================================================================
rem  NInfer on the RTX 4080 with the DFlash2 draft model, Windows host through
rem  Docker Desktop / WSL2.
rem
rem  Builds the product image from .\Dockerfile (first run, or when the checkout
rem  revision changed since the image was built) and serves the DFlash2 profile
rem  at the context that fits the 16 GiB card with the requantized Q4 companion:
rem
rem    * 100,000-token context without vision, 65,536 with vision (measured caps;
rem      set NINFER_CONTEXT to override)
rem    * DFlash2 block drafting with 7 draft tokens (K=7), LM-head proposal
rem      selector. Session 16 put every DFlash2 width on the fast small-T route
rem      (K=1..15, widths 2..16) with its A8 profile, but K=7 stays the measured
rem      window: widths 9..16 pay a second 8-column tile (~1.7x round cost) and
rem      the extra drafts do not add enough accepted tokens to pay for it (K=15
rem      measures 79/197/154 tok/s at 8K/28K/56K, below these K=7 numbers at
rem      every depth).
rem    * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
rem      presence/frequency penalties 0. This engine has no multiplicative repeat
rem      penalty; the neutral value 1 is its implicit behavior.
rem
rem  Measured on this card with the session-16 build
rem  (docs\maintainer\rtx-4080-plan.md section 11; tiled corpus, rk4v4-e8,
rem  --prefill-chunk 1024, one repetition per point): DFlash2 K=7 decodes about
rem  136/232/205 tok/s at 8K/28K/56K depth against MTP3's 130/126/109
rem  (session-10: 124/205/183 against 125/118/103), is greedy-lossless against
rem  ordinary decoding, and needs a smaller context because the companion weights
rem  and ring state cost about 0.8 GiB more than MTP. On the code scenario
rem  (examples\cli\messages\scenario_code_python.json) the A8 decode profile
rem  lifts greedy decode from 104.9 to 118.2 tok/s against K=15's 67.1. MTP stays
rem  (run-ninfer-4080.bat) because it is smaller; DFlash2 is the deeper-context
rem  speed option.
rem
rem  The image carries only the binaries; the artifact is bind-mounted read-only.
rem  Every build stamps the image with the checkout revision (label
rem  org.ninfer.revision), so pulling source changes and re-running this launcher
rem  rebuilds the image instead of serving the previous build.
rem
rem  Environment overrides: NINFER_IMAGE (default ninfer-4080:gsq3),
rem  NINFER_ARTIFACT (default <repo>\out\qwen3_8_27b_gsq3.ninfer),
rem  NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to
rem  keep the port host-local), NINFER_VISION (1 enables media and selects the
rem  vision cap), NINFER_CONTEXT (explicit token cap overrides both),
rem  NINFER_API_KEY (optional).
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
if not defined NINFER_VISION set "NINFER_VISION=0"

set "VISION_ARG="
if "%NINFER_VISION%"=="0" (
  if not defined NINFER_CONTEXT set "NINFER_CONTEXT=100000"
) else (
  if not defined NINFER_CONTEXT set "NINFER_CONTEXT=65536"
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

echo Serving DFlash2 K=7 at %NINFER_CONTEXT% tokens (%NINFER_KV_DTYPE% KV) on http://%NINFER_BIND%:%NINFER_PORT%/v1
docker run --rm ^
  --gpus all ^
  --add-host=host.docker.internal:host-gateway ^
  -p "%NINFER_BIND%:%NINFER_PORT%:8080" ^
  -v "%ARTIFACT_DIR%:/models:ro" ^
  "%NINFER_IMAGE%" ^
  ninfer-serve "/models/%ARTIFACT_NAME%" ^
  --host 0.0.0.0 --port 8080 ^
  --max-context %NINFER_CONTEXT% --kv-capacity %NINFER_CONTEXT% --kv-dtype %NINFER_KV_DTYPE% ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 ^
  --host-kv-mib 4096 ^
  --spec dflash2 --draft-tokens 7 --lm-head-draft ^
  %VISION_ARG% ^
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 ^
  --presence-penalty 0 --frequency-penalty 0 ^
  %APIKEY_ARG%
