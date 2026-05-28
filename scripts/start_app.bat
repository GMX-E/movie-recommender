@echo off
chcp 65001 >nul
echo 正在启动电影推荐系统...

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 启动 Streamlit 应用（使用默认端口8502，让 Streamlit 自动打开浏览器）
python -m streamlit run app/streamlit_app.py ^
    --server.port 8502 ^
    --browser.gatherUsageStats false

pause