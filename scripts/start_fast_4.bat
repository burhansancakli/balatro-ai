@echo off
setlocal
start "balatrobot-12346" uvx balatrobot==1.4.1 serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --animation-fps 1000 --port 12346
timeout /t 2 /nobreak >nul
start "balatrobot-12347" uvx balatrobot==1.4.1 serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --animation-fps 1000 --port 12347
timeout /t 2 /nobreak >nul
start "balatrobot-12348" uvx balatrobot==1.4.1 serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --animation-fps 1000 --port 12348
timeout /t 2 /nobreak >nul
start "balatrobot-12349" uvx balatrobot==1.4.1 serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --animation-fps 1000 --port 12349
