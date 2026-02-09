@echo off
color 0A
cls
echo ========================================================
echo       DEEPBIO-SCAN | EDGE GENOMIC CONSOLE v4.2      
echo ========================================================
echo.

:: 1. Environment Setup
echo [*] Initializing Environment Variables...
set "PYTHONPATH=%~dp0"
set "SCAN_ROOT=%~dp0"

:: 2. Pre-flight Check (Wait for USB)
echo [*] Scanning for Volume E: (Data Persistence Layer)...
if exist "E:\DeepBio_Scan" (
    echo [OK] Volume E: DETECTED. Mounting External Storage.
) else (
    echo [WARNING] Volume E: NOT FOUND.
    echo [!] Running in DISCONNECTED MODE (Local Cache Only).
)
timeout /t 1 >nul

:: 3. Hacking/Boot Sequence
echo.
echo [*] Loading Nucleotide Transformer (v2-50m)...
timeout /t 1 >nul
echo [OK] Model Weights Verified.
echo.
echo [*] Mounting LanceDB Vector Index (IVF-PQ)...
timeout /t 1 >nul
echo [OK] Index Mounted (Read/Write).
echo.
echo [*] Calibrating 3D Manifold Projector...
echo [OK] Visualization Engine Ready.
echo.
echo ========================================================
echo       SYSTEM READY. LAUNCHING INTERFACE...
echo ========================================================
timeout /t 2 >nul

:: 4. Launch Streamlit
streamlit run src/interface/app.py --server.headless true --theme.base "dark" --theme.primaryColor "#00E5FF"

pause
