@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Copy any number of source files to the destination folder, preserving original filenames

set "SRCDIR=G:\My Drive\work\MMM"
set "DESTDIR=C:\Users\User\repos\MMM\MeridianTests\Results"
set "FAILED=0"
set "ATTEMPTED=0"

if not exist "%DESTDIR%" (
  echo [INFO] Destination folder not found. Creating: %DESTDIR%
  mkdir "%DESTDIR%" || (
    echo [ERROR] Failed to create destination directory.
    endlocal & exit /b 1
  )
)

REM >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
REM Configure your source files here (space-separated list) - filenames only
set "FILES=saved_mmm_additive.pkl saved_mmm_additive_halfPrice.pkl saved_mmm_additive_halfPrice2.pkl saved_mmm_additive_halfPrice3.pkl"
REM To add more, append to FILES: set "FILES=%FILES% another_file.pkl"
REM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

for %%F in (%FILES%) do (
  set /a ATTEMPTED+=1
  set "SRC=%SRCDIR%\%%~F"
  if not exist "!SRC!" (
    echo [WARN] Source file not found: !SRC!
    set "FAILED=1"
  ) else (
    echo [INFO] Copying: !SRC!
    copy /Y "!SRC!" "%DESTDIR%\%%~nxF" >nul 2>&1
    if errorlevel 1 (
      echo [ERROR] Copy failed: !SRC!
      set "FAILED=1"
    ) else (
      echo [SUCCESS] Copied to %DESTDIR%\%%~nxF
    )
  )
)

if "!FAILED!"=="0" (
  if "!ATTEMPTED!"=="0" (
    echo [DONE] No files configured to copy.
  ) else (
    echo [DONE] All copies succeeded.
  )
  endlocal & exit /b 0
) else (
  echo [DONE] Completed with errors. See messages above.
  endlocal & exit /b 1
)
