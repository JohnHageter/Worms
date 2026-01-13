@echo off

REM Building environment ...

REM Set environment name (optional override)
SET ENV_NAME=planaria-tracker

REM Activate conda (modify this path if your conda is different)
CALL %LOCALAPPDATA%\miniforge3\condabin\activate.bat

conda info --envs | findstr /R /C:"%ENV_NAME%" >nul
IF %ERRORLEVEL%==0 (
    echo Environment "%ENV_NAME%" already exists. Removing...
    conda remove -y -n %ENV_NAME% --all
)

conda env create -f ".\environment.yml" -n %ENV_NAME%
conda activate %ENV_NAME%