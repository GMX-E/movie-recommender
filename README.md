# 电影推荐系统

基于MovieLens数据集的协同过滤推荐系统，使用Streamlit构建交互界面。

# 技术栈

- Python 3.8+
- pandas / numpy（数据处理）
- scikit-learn（余弦相似度计算）
- Streamlit（Web界面）

## 项目结构
├── app/ # Streamlit前端界面
├── src/ # 核心算法代码
├── scripts/ # 启动脚本
└── requirements.txt # 依赖清单

## 如何运行

1. 安装依赖：`pip install -r requirements.txt`
2. 启动应用：`streamlit run app/streamlit_app.py`

## 数据集

MovieLens ml-latest-small（974部电影，10万条评分）

## 算法

基于物品的协同过滤，使用余弦相似度计算电影相似度
