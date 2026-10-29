
@echo off
chcp 65001 >nul
setlocal
title 企業破產預警系統 - 一鍵啟動

REM ============================================================
REM 路徑設定：以 BAT 所在位置為基準
REM ============================================================
set "BASE=%~dp0"
set "ROOT=%BASE%bankruptcy"
set "CLOUDFLARED=%BASE%cloudflared.exe"
set "DOWNLOAD_URL=https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

echo ==============================================================================
echo 正在啟動 企業破產風險預測系統
echo Bankruptcy Early Warning System
echo ==============================================================================
echo.
echo 專案位置：%ROOT%
echo Cloudflare：%CLOUDFLARED%
echo.

REM ============================================================
REM 1. 檢查專案檔案
REM ============================================================
if not exist "%ROOT%\api\main.py" (
    echo [錯誤] 找不到 FastAPI 程式：
    echo %ROOT%\api\main.py
    pause
    exit /b 1
)

if not exist "%ROOT%\app\app.py" (
    echo [錯誤] 找不到 Streamlit 程式：
    echo %ROOT%\app\app.py
    pause
    exit /b 1
)

REM ============================================================
REM 2. 檢查 Cloudflare，沒有就自動下載
REM ============================================================
echo [檢查] 正在檢查 Cloudflare Tunnel...

if exist "%CLOUDFLARED%" goto CHECK_CLOUDFLARED

echo [下載] 找不到 cloudflared.exe，開始從官方來源下載...
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { Invoke-WebRequest -Uri '%DOWNLOAD_URL%' -OutFile '%CLOUDFLARED%' -UseBasicParsing; if (!(Test-Path '%CLOUDFLARED%') -or (Get-Item '%CLOUDFLARED%').Length -lt 1000000) { throw '下載檔案不存在或大小異常' } } catch { Write-Host ('下載失敗：' + $_.Exception.Message); exit 1 }"

if errorlevel 1 (
    echo.
    echo [錯誤] Cloudflare 下載失敗！
    echo 請確認網路是否正常，或學校是否限制 GitHub 下載。
    echo 官方下載頁：
    echo https://developers.cloudflare.com/tunnel/downloads/
    pause
    exit /b 1
)

:CHECK_CLOUDFLARED
echo.
echo [檢查] 確認 cloudflared.exe 可以正常執行...

"%CLOUDFLARED%" --version
if errorlevel 1 (
    echo.
    echo [錯誤] cloudflared.exe 無法正常執行。
    echo 請刪除該檔案後重新啟動 BAT 以重新下載。
    pause
    exit /b 1
)

echo.
echo [成功] Cloudflare Tunnel 已準備完成。

REM ============================================================
REM 3. 啟動 FastAPI
REM ============================================================
echo.
echo [1/3] 正在啟動 FastAPI 後端服務 (Port 8000)...

start "FastAPI 後端 API (8000)" cmd /k "cd /d ""%ROOT%"" && title FastAPI 後端 API (8000) && python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

REM ============================================================
REM 4. 啟動 Streamlit
REM ============================================================
echo.
echo [2/3] 正在啟動 Streamlit 前端服務 (Port 8501)...

start "Streamlit 前端系統 (8501)" cmd /k "cd /d ""%ROOT%"" && title Streamlit 前端 (8501) && python -m streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501"

echo.
echo 等待前後端服務啟動...
timeout /t 5 /nobreak >nul

REM ============================================================
REM 5. 啟動 Cloudflare Tunnel
REM ============================================================
echo.
echo [3/3] 正在啟動 Cloudflare Tunnel...

start "Cloudflare Tunnel - 公開網址" "%ComSpec%" /k ""%CLOUDFLARED%" tunnel --url http://localhost:8501"

REM ============================================================
REM 6. 開啟本機頁面
REM ============================================================
timeout /t 3 /nobreak >nul

start "" "http://localhost:8501"
start "" "http://127.0.0.1:8000/docs"

echo.
echo ==============================================================================
echo 啟動指令已執行！
echo.
echo 本機前端：http://localhost:8501
echo 本機後端：http://127.0.0.1:8000
echo API 文件：http://127.0.0.1:8000/docs
echo.
echo 公開網址：請查看 Cloudflare Tunnel 視窗
echo 請保持三個服務視窗開啟。
echo ==============================================================================
echo.
pause
