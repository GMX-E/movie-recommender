import sys
import os
import subprocess

# 检查是否直接运行该文件（而不是被streamlit调用）
if __name__ == "__main__" and "streamlit" not in sys.argv[0]:
    # 直接运行时，调用streamlit启动应用
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        script_path,
        "--server.port", "8502",
        "--browser.gatherUsageStats", "false"
    ]
    print(f"启动命令: {' '.join(cmd)}")
    subprocess.run(cmd)
    sys.exit(0)

# 以下是streamlit应用的正常代码
import streamlit as st
import __main__

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 在导入MovieRecommender之前先导入UserBasedCF类
from train_model import UserBasedCF
# 将UserBasedCF添加到__main__模块，以便pickle加载时能够找到
__main__.UserBasedCF = UserBasedCF

from recommend import MovieRecommender


def load_recommender():
    """
    加载电影推荐器模型，使用缓存机制
    
    Returns:
        MovieRecommender: 已加载的推荐器实例
    """
    try:
        recommender = MovieRecommender()
        return recommender
    except Exception as e:
        st.error(f"加载推荐器失败: {str(e)}")
        st.error("请确保已运行训练脚本生成模型文件")
        return None


def validate_user_id(user_id_input):
    """
    验证用户ID输入
    
    Args:
        user_id_input: 用户输入的ID字符串
        
    Returns:
        tuple: (is_valid, user_id, error_message)
    """
    if not user_id_input:
        return False, None, "请输入用户ID"
    
    try:
        user_id = int(user_id_input)
        if 1 <= user_id <= 610:
            return True, user_id, ""
        else:
            return False, None, "用户ID必须在1~610范围内"
    except ValueError:
        return False, None, "用户ID必须是数字"


def validate_movie_keyword(keyword):
    """
    验证电影关键词输入
    
    Args:
        keyword: 用户输入的关键词
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not keyword or keyword.strip() == "":
        return False, "请输入电影名称关键词"
    return True, ""


def display_recommendations(recommendations):
    """
    展示推荐电影列表
    
    Args:
        recommendations: 推荐结果DataFrame
    """
    if recommendations.empty:
        st.warning("未找到推荐电影")
        return
    
    st.success(f"为您推荐以下 {len(recommendations)} 部电影：")
    
    # 使用卡片布局展示
    for idx, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"""
            **{idx + 1}. {row['title']}**
            - 类型：{row['genres']}
            - 推荐分数：{row['predicted_rating']:.2f}
            """)
            st.divider()


def display_similar_movies(similar_movies, original_movie):
    """
    展示相似电影列表
    
    Args:
        similar_movies: 相似电影结果DataFrame
        original_movie: 原始电影标题
    """
    if similar_movies.empty:
        st.warning("未找到相似电影")
        return
    
    st.success(f"与「{original_movie}」相似的电影：")
    
    # 使用卡片布局展示
    for idx, row in similar_movies.iterrows():
        with st.container():
            st.markdown(f"""
            **{idx + 1}. {row['title']}**
            - 类型：{row['genres']}
            - 相似度：{row['similarity']:.2%}
            """)
            st.divider()


def tab_personalized_recommend(recommender):
    """
    个性化推荐选项卡内容
    """
    st.subheader("为您推荐")
    
    # 用户ID输入
    user_id_input = st.text_input(
        "用户ID",
        value="1",
        placeholder="请输入用户ID（1~610）",
        help="系统支持的用户ID范围为1到610"
    )
    
    # 获取推荐按钮
    col1, col2 = st.columns([1, 3])
    with col1:
        get_rec_btn = st.button("获取推荐", use_container_width=True)
    
    # 按回车键触发
    if get_rec_btn or (user_id_input and st.session_state.get('enter_pressed', False)):
        # 验证输入
        is_valid, user_id, error_msg = validate_user_id(user_id_input)
        
        if not is_valid:
            st.error(error_msg)
            return
        
        # 显示加载状态
        with st.spinner("正在生成推荐..."):
            try:
                # 先获取用户历史偏好（评分最高的5部电影）
                st.subheader("📊 您的观影偏好")
                top_rated = recommender.get_user_top_rated(user_id=user_id, n=5)
                
                if top_rated is None:
                    st.info("该用户暂无评分记录")
                else:
                    st.write(f"根据您评分最高的电影，我们为您推荐：")
                    for idx, movie in enumerate(top_rated, 1):
                        st.write(f"⭐ **{movie['title']}** ({movie['genres']}) — 评分 {movie['rating']:.1f}")
                
                # 添加视觉分隔
                st.divider()
                
                # 获取推荐结果
                recommendations = recommender.get_recommendations(user_id=user_id, n=10)
                
                # 检查是否有推荐结果
                if recommendations.empty:
                    st.warning(f"用户 {user_id} 已观看所有电影，无法提供更多推荐")
                else:
                    # 展示推荐结果
                    display_recommendations(recommendations)
                    
            except Exception as e:
                st.error(f"获取推荐失败: {str(e)}")
                st.error("请确保模型文件已正确生成")


def tab_similar_movies(recommender):
    """
    相似电影选项卡内容
    """
    st.subheader("查找相似电影")
    
    # 电影关键词输入
    movie_keyword = st.text_input(
        "电影名称",
        value="Toy Story",
        placeholder="请输入电影名称关键词",
        help="支持部分匹配，例如输入'love'将匹配包含love的电影"
    )
    
    # 查找相似按钮
    col1, col2 = st.columns([1, 3])
    with col1:
        find_similar_btn = st.button("查找相似", use_container_width=True)
    
    # 按回车键触发
    if find_similar_btn or (movie_keyword and st.session_state.get('enter_pressed_similar', False)):
        # 验证输入
        is_valid, error_msg = validate_movie_keyword(movie_keyword)
        
        if not is_valid:
            st.error(error_msg)
            return
        
        # 显示加载状态
        with st.spinner("正在查找相似电影..."):
            try:
                # 获取相似电影
                similar_movies = recommender.similar_movies(movie_title=movie_keyword, n=5)
                
                # 检查结果
                if similar_movies.empty:
                    st.warning(f"未找到与「{movie_keyword}」匹配的电影，请尝试其他关键词")
                else:
                    # 获取匹配的原始电影标题
                    matched_movies = recommender.movies_df[
                        recommender.movies_df['title'].str.contains(movie_keyword, case=False)
                    ]
                    original_movie = matched_movies.iloc[0]['title'] if not matched_movies.empty else movie_keyword
                    
                    # 展示结果
                    display_similar_movies(similar_movies, original_movie)
                    
            except Exception as e:
                st.error(f"查找相似电影失败: {str(e)}")
                st.error("请确保数据文件已正确放置")


def main():
    """
    主函数：设置页面布局和选项卡
    """
    # 页面配置
    st.set_page_config(
        page_title="电影推荐系统",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 页面标题
    st.title("🎬 电影推荐系统")
    st.markdown("基于机器学习的智能电影推荐服务")
    
    # 加载推荐器（使用缓存）
    @st.cache_resource(show_spinner="正在加载推荐模型...")
    def cached_load_recommender():
        return load_recommender()
    
    recommender = cached_load_recommender()
    
    # 如果加载失败，显示错误并退出
    if recommender is None:
        st.stop()
    
    # 创建选项卡
    tab1, tab2 = st.tabs(["个性化推荐", "相似电影"])
    
    with tab1:
        tab_personalized_recommend(recommender)
    
    with tab2:
        tab_similar_movies(recommender)
    
    # 页脚信息
    st.markdown("---")
    st.markdown("数据来源：MovieLens ml-latest-small 数据集")


if __name__ == "__main__":
    main()
