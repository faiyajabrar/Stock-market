@echo off
title Precision-Focused Stock Market Predictor

:: Check for Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Get current date in YYYY-MM-DD format
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set CURRENT_YEAR=%datetime:~0,4%
set CURRENT_MONTH=%datetime:~4,2%
set CURRENT_DAY=%datetime:~6,2%
set CURRENT_DATE=%CURRENT_YEAR%-%CURRENT_MONTH%-%CURRENT_DAY%

:: Default values
set DATE=%CURRENT_DATE%
set DAYS=5
set THRESHOLD=0.3030
set PLOT_FLAG=--plot

:: Parse command-line arguments
:parse_args
if "%~1"=="" goto :run_prediction
if /i "%~1"=="--date" (
    set DATE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--days" (
    set DAYS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--no-plot" (
    set PLOT_FLAG=
    shift
    goto :parse_args
)
shift
goto :parse_args

:run_prediction
echo.
echo ===================================================
echo      PRECISION-FOCUSED STOCK PREDICTION TOOL
echo ===================================================
echo.
echo Running prediction with the following parameters:
echo  - Date: %DATE%
echo  - Days: %DAYS%
echo  - Threshold: %THRESHOLD% (optimized for precision)
echo  - Plot: %PLOT_FLAG:--plot=yes%
echo.
echo ===================================================
echo.

:: Run the prediction
python predict_improved.py --date %DATE% --days %DAYS% --threshold %THRESHOLD% %PLOT_FLAG%

echo.
echo Prediction complete!
echo.

pause
