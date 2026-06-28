@echo off
setlocal
set SRC=F:\AIWF_Studio\models\ltx\checkpoints\ltx-2.3-22b-dev-nvfp4.safetensors
set DST=F:\AIWF_Studio\models\ltx\checkpoints\ltx-2.3-22b-dev-bf16.safetensors
set LOG=F:\AIWF_Studio\scripts\nvfp4_convert.log

echo Starting NVFP4 -> BF16 conversion at %date% %time% > "%LOG%"
echo Source: %SRC% >> "%LOG%"
echo Dest:   %DST% >> "%LOG%"

"F:\ComfyUI\venv\Scripts\python.exe" "F:\AIWF_Studio\scripts\convert_nvfp4_to_bf16.py" --src "%SRC%" --dst "%DST%" >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo Finished at %date% %time% with exit code %ERRORLEVEL% >> "%LOG%"
endlocal
