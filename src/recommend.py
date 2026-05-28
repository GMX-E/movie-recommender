import os
import sys
import pickle
import logging
import time
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity

# 获取项目根目录（相对于当前文件的父目录的父目录）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_movies, load_ratings
from train_model import UserBasedCF

# 将UserBasedCF添加到__main__模块，以便pickle加载时能够找到
import __main__
__main__.UserBasedCF = UserBasedCF

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MovieRecommender:
    """
    电影推荐器类
    
    提供基于用户的协同过滤推荐和基于内容的相似电影推荐功能
    """
    
    def __init__(self, model_path: str = None, data_path: str = None):
        """
        初始化推荐器
        
        Args:
            model_path: 预训练模型的路径，默认为src/models/svd_model.pkl
            data_path: 数据目录路径，默认为data/
        """
        logger.info("初始化电影推荐器...")
        
        # 设置默认路径
        if model_path is None:
            model_path = os.path.join(ROOT_DIR, 'src', 'models', 'svd_model.pkl')
        
        if data_path is None:
            data_path = os.path.join(ROOT_DIR, 'data')
        
        # 加载电影数据
        self.movies_df = load_movies(data_path)
        
        # 加载评分数据
        self.ratings_df = load_ratings(data_path)
        
        # 加载预训练模型
        self._load_model(model_path)
        
        # 构建类型矩阵（用于相似电影推荐）
        self._build_genre_matrix()
        
        # 构建电影-用户评分矩阵和相似度矩阵（用于基于协同过滤的相似电影推荐）
        self._build_movie_user_matrix()
        
        logger.info("电影推荐器初始化完成")
    
    def _load_model(self, model_path: str):
        """
        加载预训练的模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"成功加载模型: {model_path}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _build_genre_matrix(self):
        """
        构建电影-类型矩阵，用于计算Jaccard相似度
        
        将genres字段解析为二进制矩阵，每行代表一部电影，每列代表一个类型
        """
        logger.info("构建电影-类型矩阵...")
        
        # 获取所有唯一类型
        all_genres = set()
        for genres in self.movies_df['genres']:
            all_genres.update(genres.split('|'))
        
        # 创建类型列表（排序以便结果一致）
        self.genre_list = sorted(list(all_genres))
        logger.info(f"共识别到 {len(self.genre_list)} 个电影类型")
        
        # 构建电影-类型矩阵
        self.genre_matrix = pd.DataFrame(0, index=self.movies_df['movieId'], columns=self.genre_list)
        
        for idx, row in self.movies_df.iterrows():
            movie_genres = row['genres'].split('|')
            for genre in movie_genres:
                self.genre_matrix.loc[row['movieId'], genre] = 1
        
        logger.info("电影-类型矩阵构建完成")
    
    def _build_movie_user_matrix(self):
        """
        构建电影-用户评分矩阵并预计算电影间余弦相似度
        
        矩阵行索引为movieId，列索引为userId，值为rating，缺失值填充为0
        """
        logger.info("构建电影-用户评分矩阵...")
        start_time = time.time()
        
        # 创建电影-用户评分矩阵
        self.movie_user_matrix = self.ratings_df.pivot_table(
            index='movieId',
            columns='userId',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"电影-用户矩阵维度: {self.movie_user_matrix.shape}")
        logger.info(f"非零评分数量: {self.movie_user_matrix.values[self.movie_user_matrix.values > 0].size}")
        
        # 计算电影间余弦相似度矩阵
        logger.info("计算电影间余弦相似度矩阵...")
        self.movie_similarity_matrix = cosine_similarity(self.movie_user_matrix.values)
        
        # 创建电影ID到索引的映射
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_user_matrix.index)}
        
        elapsed_time = time.time() - start_time
        logger.info(f"电影-用户矩阵和相似度矩阵构建完成，耗时: {elapsed_time:.2f}秒")
    
    def get_user_rated_movies(self, user_id: int) -> set:
        """
        获取用户已评分的电影ID集合
        
        Args:
            user_id: 用户ID
            
        Returns:
            set: 用户已评分的电影ID集合
        """
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        return set(user_ratings['movieId'].unique())
    
    def get_user_top_rated(self, user_id: int, n: int = 5) -> list:
        """
        检索并返回指定用户评分最高的n部电影详细信息
        
        Args:
            user_id: 用户ID
            n: 返回电影数量，默认为5
            
        Returns:
            list: 包含电影信息的字典列表，每个字典包含title、genres、rating字段
                  若用户无评分记录，返回None
        """
        logger.info(f"获取用户 {user_id} 评分最高的 {n} 部电影...")
        
        # 筛选指定用户的所有评分记录
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        # 处理用户无评分记录的情况
        if user_ratings.empty:
            logger.warning(f"用户 {user_id} 暂无评分记录")
            return None
        
        # 按rating字段降序排序，取前n条
        top_rated = user_ratings.sort_values(by='rating', ascending=False).head(n)
        
        # 合并电影信息
        top_rated_with_movies = top_rated.merge(
            self.movies_df, on='movieId', how='left'
        )
        
        # 转换为字典列表格式
        result = []
        for _, row in top_rated_with_movies.iterrows():
            result.append({
                'title': row['title'],
                'genres': row['genres'],
                'rating': row['rating']
            })
        
        logger.info(f"成功获取用户 {user_id} 评分最高的 {len(result)} 部电影")
        return result
    
    def get_recommendations(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        为指定用户推荐未看过的top-n电影
        
        Args:
            user_id: 用户ID
            n: 推荐电影数量，默认为10
            
        Returns:
            pandas.DataFrame: 包含推荐电影信息的DataFrame，按预测评分降序排列
        """
        logger.info(f"为用户 {user_id} 生成 {n} 部推荐电影...")
        
        # 获取用户已评分的电影
        rated_movies = self.get_user_rated_movies(user_id)
        
        # 获取所有电影ID
        all_movies = set(self.movies_df['movieId'])
        
        # 计算用户未评分的电影
        unrated_movies = all_movies - rated_movies
        
        if not unrated_movies:
            logger.warning(f"用户 {user_id} 已观看所有电影，无法推荐")
            return pd.DataFrame()
        
        # 预测用户对未评分电影的评分
        predictions = []
        for movie_id in unrated_movies:
            # 自定义模型直接返回预测评分（浮点数）
            predicted_rating = self.model.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': predicted_rating
            })
        
        # 按预测评分排序
        predictions_df = pd.DataFrame(predictions).sort_values(
            by='predicted_rating', ascending=False
        ).head(n)
        
        # 合并电影信息
        recommendations = predictions_df.merge(
            self.movies_df, on='movieId', how='left'
        )
        
        # 选择并重新排序列
        recommendations = recommendations[['movieId', 'title', 'genres', 'predicted_rating']]
        
        logger.info(f"成功为用户 {user_id} 生成 {len(recommendations)} 部推荐电影")
        return recommendations
    
    def _compute_jaccard_similarity(self, movie_id1: int, movie_id2: int) -> float:
        """
        计算两部电影之间的Jaccard相似度（基于类型）
        
        Args:
            movie_id1: 第一部电影ID
            movie_id2: 第二部电影ID
            
        Returns:
            float: Jaccard相似度值（0-1之间）
        """
        if movie_id1 not in self.genre_matrix.index or movie_id2 not in self.genre_matrix.index:
            return 0.0
        
        genres1 = self.genre_matrix.loc[movie_id1].values
        genres2 = self.genre_matrix.loc[movie_id2].values
        
        return jaccard_score(genres1, genres2)
    
    def get_similar_movies_cf(self, movie_id: int, n: int = 5) -> pd.DataFrame:
        """
        基于物品的协同过滤算法，返回与指定电影相似的电影
        
        Args:
            movie_id: 目标电影的唯一标识符
            n: 推荐结果数量，默认为5
            
        Returns:
            pandas.DataFrame: 包含相似电影信息的DataFrame，按相似度降序排列
                              若电影无评分记录或不存在，返回空DataFrame
        """
        logger.info(f"基于协同过滤查找与电影ID {movie_id} 相似的 {n} 部电影...")
        
        # 检查电影ID是否存在于相似度矩阵中
        if movie_id not in self.movie_id_to_idx:
            logger.warning(f"电影ID {movie_id} 不存在于评分数据中")
            return pd.DataFrame()
        
        # 获取电影在矩阵中的索引
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # 获取目标电影与其他所有电影的相似度
        similarities = self.movie_similarity_matrix[movie_idx]
        
        # 创建相似度列表（排除电影自身）
        similarity_list = []
        for idx, sim in enumerate(similarities):
            other_movie_id = self.movie_user_matrix.index[idx]
            if other_movie_id != movie_id:
                similarity_list.append({
                    'movieId': other_movie_id,
                    'similarity': sim
                })
        
        # 按相似度降序排序并取前n个
        similarity_list.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similarity_list[:n]
        
        # 转换为DataFrame并合并电影信息
        if not top_similar:
            logger.warning(f"电影ID {movie_id} 没有足够的相似电影")
            return pd.DataFrame()
        
        similarities_df = pd.DataFrame(top_similar)
        similar_movies_df = similarities_df.merge(
            self.movies_df, on='movieId', how='left'
        )
        
        # 选择并重新排序列
        similar_movies_df = similar_movies_df[['movieId', 'title', 'genres', 'similarity']]
        
        logger.info(f"成功找到 {len(similar_movies_df)} 部相似电影")
        return similar_movies_df
    
    def similar_movies(self, movie_title: str, n: int = 5) -> pd.DataFrame:
        """
        根据用户评分行为计算余弦相似度，返回相似电影
        
        Args:
            movie_title: 电影标题（支持部分匹配）
            n: 返回相似电影数量，默认为5
            
        Returns:
            pandas.DataFrame: 包含相似电影信息的DataFrame，按相似度降序排列
        """
        logger.info(f"查找与 '{movie_title}' 相似的 {n} 部电影...")
        
        # 查找匹配的电影（支持部分匹配，不区分大小写）
        matched_movies = self.movies_df[
            self.movies_df['title'].str.contains(movie_title, case=False)
        ]
        
        if matched_movies.empty:
            logger.warning(f"未找到匹配的电影: {movie_title}")
            return pd.DataFrame()
        
        # 取第一个匹配的电影
        target_movie = matched_movies.iloc[0]
        target_movie_id = target_movie['movieId']
        logger.info(f"找到匹配电影: {target_movie['title']} (ID: {target_movie_id})")
        
        # 使用基于协同过滤的相似电影推荐
        return self.get_similar_movies_cf(movie_id=target_movie_id, n=n)


if __name__ == "__main__":
    # 测试推荐器功能
    try:
        recommender = MovieRecommender()
        
        # 测试用户推荐功能
        print("=" * 60)
        print("测试用户推荐功能")
        print("=" * 60)
        recs = recommender.get_recommendations(user_id=1, n=5)
        print(recs)
        print("\n")
        
        # 测试相似电影功能
        print("=" * 60)
        print("测试相似电影推荐功能")
        print("=" * 60)
        similar = recommender.similar_movies(movie_title="Toy Story", n=5)
        print(similar)
        
    except Exception as e:
        print(f"测试失败: {e}")
