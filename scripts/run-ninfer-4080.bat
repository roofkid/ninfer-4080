@echo off
rem ============================================================================
rem  NInfer on the RTX 4080, Windows host through Docker Desktop / WSL2.
rem
rem  Pulls the published image and serves the registered 3-bit GSQ artifact
rem  with the documented 100K profile:
rem
rem    * 102,400-token context, one lane, rk4v4-e8 KV (the registered 4080 fit)
rem    * MTP speculative decoding, three draft tokens, LM-head draft route
rem    * vision tower enabled (the GSQ3 artifact carries it)
rem    * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
rem      presence/frequency penalties 0. This engine has no multiplicative
rem      repeat penalty; the neutral value 1 is its implicit behavior.
rem
rem  The image carries only the binaries; the artifact is bind-mounted read-only,
rem  so the container stays small and never contains model weights. Download the
rem  artifact first with scripts\download-qwen38-gsq3.bat (or .sh); the launcher
rem  looks for it in <repo>\out and then <repo>\models.
rem
rem  Image resolution: an existing local image is used as-is. A missing image is
rem  pulled from the registry; if the pull fails and this checkout has a
rem  Dockerfile, the image is built from source. NINFER_BUILD=1 always builds
rem  from source. A locally built image carries the checkout revision (label
rem  org.ninfer.revision), so pulling source changes and re-running this launcher
rem  rebuilds that image instead of serving the previous build; published images
rem  have no label and are served as pulled.
rem
rem  Measured on this card with the session-26 build
rem  (docs\maintainer\rtx-4080-plan.md section 11; tiled corpus, rk4v4-e8,
rem  --prefill-chunk 1024, one repetition per point): prefill about
rem  2720/2425/2126/1895 tok/s and MTP3 decode about 151/142/131/122 tok/s at
rem  8K/32K/64K/98K depth; DFlash2 K=7 decodes about 167/262/239/213 tok/s at the
rem  same points. At the documented 100K profile (--prefill-chunk 2688) prefill
rem  is about 1971 tok/s. The profile is fixed; edit this file to change it.
rem
rem  The published port binds every host interface, so the profile is reachable
rem  from other machines on the network at http://<host-ip>:8080/v1 ^(allow the
rem  Docker Desktop firewall prompt on first use^). Set NINFER_API_KEY to require
rem  a bearer token; the server only checks it when this is set.
rem
rem  Environment overrides: NINFER_IMAGE (default roofkid/ninfer-4080:gsq3),
rem  NINFER_ARTIFACT (default <repo>\out or <repo>\models qwen3_8_27b_gsq3.ninfer),
rem  NINFER_PORT (default 8080), NINFER_BIND (default 0.0.0.0, set 127.0.0.1 to
rem  keep the port host-local), NINFER_API_KEY (optional), NINFER_BUILD=1 (force a
rem  source build).
rem  NINFER_KV_DTYPE (default rk4v4-e8; int8 or rk8v4 trade context for MBPP-class accuracy).
rem ============================================================================
setlocal EnableExtensions
if not defined NINFER_KV_DTYPE set "NINFER_KV_DTYPE=rk4v4-e8"

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if not defined NINFER_IMAGE set "NINFER_IMAGE=roofkid/ninfer-4080:gsq3"
if not defined NINFER_PORT set "NINFER_PORT=8080"
if not defined NINFER_BIND set "NINFER_BIND=0.0.0.0"
if not defined NINFER_ARTIFACT (
  if exist "%ROOT%\out\qwen3_8_27b_gsq3.ninfer" (
    set "NINFER_ARTIFACT=%ROOT%\out\qwen3_8_27b_gsq3.ninfer"
  ) else (
    set "NINFER_ARTIFACT=%ROOT%\models\qwen3_8_27b_gsq3.ninfer"
  )
)

where docker >nul 2>nul
if errorlevel 1 (
  echo docker was not found on PATH. Install Docker Desktop with the WSL2 backend.
  exit /b 1
)
if not exist "%NINFER_ARTIFACT%" (
  echo Missing %NINFER_ARTIFACT%
  echo Download it first with scripts\download-qwen38-gsq3.bat ^(or .sh^), or set
  echo NINFER_ARTIFACT to an existing qwen3_8_27b_gsq3.ninfer copy.
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
if errorlevel 1 goto :pull_image

if defined NINFER_BUILD if not "%NINFER_BUILD%"=="0" (
  set "BUILD_IMAGE=1"
  set "BUILD_REASON=NINFER_BUILD is set"
)
if "%BUILD_IMAGE%"=="1" goto :do_build
if not exist "%ROOT%\Dockerfile" goto :image_ready
if not defined SOURCE_REVISION goto :image_ready

rem Locally built images carry org.ninfer.revision; published images do not.
set "IMAGE_REVISION="
for /f "delims=" %%I in ('docker image inspect --format "{{ index .Config.Labels `org.ninfer.revision` }}" "%NINFER_IMAGE%" 2^>nul') do set "IMAGE_REVISION=%%I"
if not defined IMAGE_REVISION goto :image_ready
if "%IMAGE_REVISION%"=="%SOURCE_REVISION%" goto :image_ready
set "BUILD_IMAGE=1"
set "BUILD_REASON=locally built image revision changed"
goto :do_build

:pull_image
echo Pulling %NINFER_IMAGE%...
docker pull "%NINFER_IMAGE%"
if not errorlevel 1 goto :image_ready
if not exist "%ROOT%\Dockerfile" (
  echo Could not pull %NINFER_IMAGE% and no Dockerfile is available to build it.
  exit /b 1
)
set "BUILD_IMAGE=1"
set "BUILD_REASON=pull failed; building from source"

:do_build
echo Building %NINFER_IMAGE% from Dockerfile ^(%BUILD_REASON%; several minutes^)...
if defined SOURCE_REVISION (
  docker build --label "org.ninfer.revision=%SOURCE_REVISION%" -f "%ROOT%\Dockerfile" -t "%NINFER_IMAGE%" "%ROOT%"
) else (
  docker build -f "%ROOT%\Dockerfile" -t "%NINFER_IMAGE%" "%ROOT%"
)
if errorlevel 1 exit /b 1

:image_ready
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
