#!/bin/bash
echo "正在启动电影推荐系统..."
python -m streamlit run app/streamlit_app.py --server.port 8502 --browser.gatherUsageStats false