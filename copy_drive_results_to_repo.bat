@echo off
REM Copy saved_mmm_additive.pkl from Google Drive to local Results folder
set "SRC=G:\My Drive\work\MMM\saved_mmm_additive.pkl"
set "DESTDIR=C:\Users\User\repos\MMM\MeridianTests\Results"
set "DEST=%DESTDIR%\saved_mmm_additive.pkl"

if not exist "%SRC%" (
  echo [ERROR] Source file not found: %SRC%
  exit /b 1
)

if not exist "%DESTDIR%" (
  echo [INFO] Destination folder not found. Creating: %DESTDIR%
  mkdir "%DESTDIR%" || (
    echo [ERROR] Failed to create destination directory.
    exit /b 1
  )
)

echo [INFO] Copying file...
copy /Y "%SRC%" "%DEST%" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Copy failed.
  exit /b 1
) else (
  echo [SUCCESS] Copied to %DEST%
  exit /b 0
)
