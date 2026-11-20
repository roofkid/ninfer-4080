@echo off
rem ============================================================================
rem  SenseNova-Loop-9B on the RTX 4080, Windows host through Docker Desktop /
rem  WSL2.
rem
rem  Pulls the published image or builds it from this checkout, then serves the
rem  registered sensenova-loop-9b/groupwise-int artifact with the validated 256K
rem  profile:
rem
rem    * 262,144-token context, one lane, int8 KV (524,288 is the native limit)
rem    * MTP speculative decoding, three draft tokens, n-gram chain; the release
rem      MTP has no optimized shortlist head, so --lm-head-draft is not used
rem    * vision enabled for image input (video is not supported by this target)
rem    * server sampling defaults temperature 1, top-k 20, top-p 0.95, min-p 0,
rem      presence/frequency penalties 0
rem
rem  The image carries only the binaries; the artifact is bind-mounted
rem  read-only, so the container stays small and never contains model weights.
rem  The artifact is published at roofkid/SenseNova-Loop-9B-NInfer; download it
rem  with scripts\download-sensenova-loop-9b.bat (or .sh), or convert it with the
rem  converter (docs\maintainer\sensenova-loop-9b-artifact.md). The launcher
rem  looks for it in <repo>\out and then <repo>\models.
rem
rem  Image resolution: an existing local image is used as-is. A missing image is
rem  pulled from the registry; if the pull fails and this checkout has a
rem  Dockerfile, the image is built from source. NINFER_BUILD=1 always builds
rem  from source. A locally built image carries the checkout revision (label
rem  org.ninfer.revision), so pulling source changes and re-running this launcher
rem  rebuilds that image instead of serving the previous build; published images
rem  have no label and are served as pulled. The published
rem  roofkid/ninfer-4080:sensenova tag (0.6.3-rtx4080) includes this target, so a
rem  pull normally serves the published binaries.
rem
rem  Measured on this card
rem  (docs\maintainer\sensenova-loop-9b-model.md section 6): this 256K + Vision +
rem  MTP3 profile starts with about 0.35 GiB available after startup, so it
rem  stays one lane. 253K-token prefill is about 1.72k tok/s and 3.5K-token
rem  decode about 36 tok/s MTP0 and 58 tok/s MTP3 + n-gram. The plan's speed
rem  bars are not met, so this is a functional 256K profile, not a performance
rem  claim. The profile is fixed; edit this file to change it.
rem
rem  The published port binds every host interface, so the profile is reachable
rem  from other machines on the network at http://<host-ip>:8080/v1 ^(allow the
rem  Docker Desktop firewall prompt on first use^). Set NINFER_API_KEY to require
rem  a bearer token; the server only checks it when this is set.
rem
rem  Environment overrides: NINFER_IMAGE (default roofkid/ninfer-4080:sensenova),
rem  NINFER_ARTIFACT (default <repo>\out or <repo>\models
rem  sensenova_loop_9b.ninfer), NINFER_PORT (default 8080), NINFER_BIND (default
rem  0.0.0.0, set 127.0.0.1 to keep the port host-local), NINFER_API_KEY
rem  (optional), NINFER_BUILD=1 (force a source build), NINFER_KV_DTYPE (default
rem  int8; rk4v4-e8 is the fallback/debug mode).
rem ============================================================================
setlocal EnableExtensions
if not defined NINFER_KV_DTYPE set "NINFER_KV_DTYPE=int8"

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
if not defined NINFER_IMAGE set "NINFER_IMAGE=roofkid/ninfer-4080:sensenova"
if not defined NINFER_PORT set "NINFER_PORT=8080"
if not defined NINFER_BIND set "NINFER_BIND=0.0.0.0"
if not defined NINFER_ARTIFACT (
  if exist "%ROOT%\out\sensenova_loop_9b.ninfer" (
    set "NINFER_ARTIFACT=%ROOT%\out\sensenova_loop_9b.ninfer"
  ) else (
    set "NINFER_ARTIFACT=%ROOT%\models\sensenova_loop_9b.ninfer"
  )
)

where docker >nul 2>nul
if errorlevel 1 (
  echo docker was not found on PATH. Install Docker Desktop with the WSL2 backend.
  exit /b 1
)
if not exist "%NINFER_ARTIFACT%" (
  echo Missing %NINFER_ARTIFACT%
  echo Download it with scripts\download-sensenova-loop-9b.bat, or set NINFER_ARTIFACT
  echo to an existing sensenova_loop_9b.ninfer copy.
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

echo Serving SenseNova-Loop-9B (%NINFER_KV_DTYPE% KV) on http://%NINFER_BIND%:%NINFER_PORT%/v1
docker run --rm ^
  --gpus all ^
  --add-host=host.docker.internal:host-gateway ^
  -p "%NINFER_BIND%:%NINFER_PORT%:8080" ^
  -v "%ARTIFACT_DIR%:/models:ro" ^
  "%NINFER_IMAGE%" ^
  ninfer-serve "/models/%ARTIFACT_NAME%" ^
  --host 0.0.0.0 --port 8080 ^
  --max-context 262144 --kv-capacity 262144 --kv-dtype %NINFER_KV_DTYPE% ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 ^
  --host-kv-mib 4096 ^
  --spec mtp --draft-tokens 3 --ngram chain ^
  --vision ^
  --temperature 1 --top-k 20 --top-p 0.95 --min-p 0 ^
  --presence-penalty 0 --frequency-penalty 0 ^
  %APIKEY_ARG%
