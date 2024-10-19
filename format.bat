@echo off
setlocal enabledelayedexpansion

:: Remove specified directories
call :remove_directories

:: Collect Python files
call :collect_python_files

:: Check and process Python files
if defined PYPATH (
    call :run_linters
    call :format_docstrings
)

endlocal
exit /b 0

:remove_directories
echo Removing specified directories...
set "dirs=__pycache__ .pytest_cache .vscode"
for %%d in (%dirs%) do (
    for /d /r %%f in (%%d) do (
        if exist "%%f" (
            echo Removing directory: %%f
            rmdir /s /q "%%f"
        )
    )
)
exit /b

:collect_python_files
echo Collecting Python files...
set "PYPATH="
for /r %%f in (*.py) do (
    set "PYPATH=!PYPATH! "%%f""
)
exit /b

:run_linters

echo Running Black to format the code...
python -m black %PYPATH%

echo Running isort to sort imports...
python -m isort %PYPATH%
exit /b

:format_docstrings
echo Formatting docstrings with docformatter...
python -m docformatter -r --in-place %PYPATH%
exit /b
