@echo off
rem ============================================================================
rem  SenseNova-Loop-9B on one RTX 4080 (sm_89, 16 GiB): the validated 256K
rem  profile.
rem
rem  Serves the registered sensenova-loop-9b/groupwise-int artifact with:
rem
rem    * a 262,144-token context on int8 KV (524,288 is the native limit;
rem      256K is the validated launch profile)
rem    * vision enabled for image input (video is not supported by this target)
rem    * MTP3 speculative decoding with the n-gram chain; the release MTP has
rem      no optimized shortlist head, so --lm-head-draft is not used
rem    * one lane, 16 pending requests, 1,024-token prefill chunks, 4 GiB
rem      host KV
rem
rem  Measured on this card
rem  (docs\maintainer\sensenova-loop-9b-model.md section 6): this 256K + Vision +
rem  MTP3 profile starts with about 0.35 GiB available after startup, so it
rem  stays one lane. 253K-token prefill is about 1.72k tok/s and 3.5K-token
rem  decode about 36 tok/s MTP0 and 58 tok/s MTP3 + n-gram. The plan's speed
rem  bars are not met, so this is a functional 256K profile, not a performance
rem  claim. The profile is fixed; edit this file to change it.
rem
rem  The artifact is produced locally by the converter (see
rem  docs\maintainer\sensenova-loop-9b-artifact.md). The launcher looks for it
rem  in <repo>\out and then <repo>\models, and accepts an explicit path as its
rem  first argument.
rem
rem  Environment overrides: NINFER_SERVER (default ninfer-serve.exe beside this
rem  script, then <repo>\build\apps\ninfer-serve.exe), NINFER_MODEL_DIR
rem  (default <repo>\out when it holds sensenova_loop_9b.ninfer, otherwise
rem  <repo>\models).
rem ============================================================================
setlocal EnableExtensions
set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

set "SERVER=%NINFER_SERVER%"
if defined SERVER goto :server_ready
set "SERVER=%~dp0ninfer-serve.exe"
if exist "%SERVER%" goto :server_ready
set "SERVER=%ROOT%\build\apps\ninfer-serve.exe"
:server_ready

set "MODEL=%~1"
if defined MODEL goto :model_ready
set "MODEL_DIR=%NINFER_MODEL_DIR%"
if defined MODEL_DIR goto :model_dir_ready
set "MODEL_DIR=%ROOT%\models"
if exist "%ROOT%\out\sensenova_loop_9b.ninfer" set "MODEL_DIR=%ROOT%\out"
:model_dir_ready
set "MODEL=%MODEL_DIR%\sensenova_loop_9b.ninfer"
:model_ready

if not exist "%SERVER%" (
  echo Missing executable: %SERVER%
  echo Set NINFER_SERVER or build build\apps\ninfer-serve.exe.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Convert it with tools\convert\sensenova_loop_9b, or pass the artifact path.
  exit /b 1
)

echo Starting SenseNova-Loop-9B at http://127.0.0.1:8080/v1
echo Profile: one lane, 256K context, int8 KV, vision, MTP3 + n-gram chain
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 --host-kv-mib 4096 --spec mtp --draft-tokens 3 --ngram chain --vision
