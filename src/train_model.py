import os
import pickle
import logging
import pandas as pd
import numpy as np

# 获取项目根目录（相对于当前文件的父目录的父目录）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class UserBasedCF:
    """
    基于用户的协同过滤推荐模型
    """
    
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.user_similarity = None
        self.user_ids = None
        self.item_ids = None
        self.user_item_matrix = None
        self.global_mean = 0.0
    
    def fit(self, ratings_df):
        """
        训练模型
        
        Args:
            ratings_df: 包含userId, movieId, rating列的DataFrame
        """
        logger.info("开始训练基于用户的协同过滤模型...")
        
        # 获取用户和物品列表
        self.user_ids = sorted(ratings_df['userId'].unique())
        self.item_ids = sorted(ratings_df['movieId'].unique())
        
        user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_id_to_idx = {mid: i for i, mid in enumerate(self.item_ids)}
        
        # 创建用户-物品评分矩阵
        logger.info("构建用户-物品评分矩阵...")
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.item_ids)))
        for _, row in ratings_df.iterrows():
            user_idx = user_id_to_idx[row['userId']]
            item_idx = item_id_to_idx[row['movieId']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # 计算全局平均分
        self.global_mean = ratings_df['rating'].mean()
        logger.info(f"全局平均分: {self.global_mean:.4f}")
        
        # 计算用户相似度（余弦相似度）
        logger.info("计算用户相似度矩阵...")
        self.user_similarity = self._compute_user_similarity()
        
        logger.info("模型训练完成")
    
    def _compute_user_similarity(self):
        """
        计算用户之间的余弦相似度
        
        Returns:
            numpy.ndarray: 用户相似度矩阵
        """
        n_users = len(self.user_ids)
        similarity = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i, n_users):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    # 找到两个用户都评分过的物品
                    mask = (self.user_item_matrix[i] > 0) & (self.user_item_matrix[j] > 0)
                    if np.sum(mask) == 0:
                        similarity[i, j] = 0.0
                    else:
                        vec1 = self.user_item_matrix[i, mask]
                        vec2 = self.user_item_matrix[j, mask]
                        # 余弦相似度
                        dot_product = np.dot(vec1, vec2)
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        if norm1 == 0 or norm2 == 0:
                            similarity[i, j] = 0.0
                        else:
                            similarity[i, j] = dot_product / (norm1 * norm2)
                    similarity[j, i] = similarity[i, j]
        
        return similarity
    
    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            float: 预测评分
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return self.global_mean
        
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        # 如果用户已经评分过该物品，返回实际评分
        if self.user_item_matrix[user_idx, item_idx] > 0:
            return self.user_item_matrix[user_idx, item_idx]
        
        # 找到对该物品有评分的用户
        users_rated = np.where(self.user_item_matrix[:, item_idx] > 0)[0]
        
        if len(users_rated) == 0:
            return self.global_mean
        
        # 获取相似度并排序
        similarities = self.user_similarity[user_idx, users_rated]
        ratings = self.user_item_matrix[users_rated, item_idx]
        
        # 取前n_neighbors个最相似的用户
        sorted_indices = np.argsort(similarities)[::-1][:self.n_neighbors]
        top_similarities = similarities[sorted_indices]
        top_ratings = ratings[sorted_indices]
        
        # 加权平均
        if np.sum(np.abs(top_similarities)) == 0:
            return self.global_mean
        
        weighted_sum = np.sum(top_similarities * top_ratings)
        weight_sum = np.sum(np.abs(top_similarities))
        
        return weighted_sum / weight_sum


def split_data(ratings_df, test_size=0.2):
    """
    划分训练集和测试集
    
    Args:
        ratings_df: 评分数据DataFrame
        test_size: 测试集比例
        
    Returns:
        tuple: (训练集, 测试集)
    """
    logger.info(f"划分数据集，测试集比例: {test_size}")
    
    # 按用户分组，确保每个用户在测试集中有数据
    grouped = ratings_df.groupby('userId')
    train_dfs = []
    test_dfs = []
    
    for _, group in grouped:
        n_samples = len(group)
        n_test = max(1, int(n_samples * test_size))
        test_indices = np.random.choice(n_samples, n_test, replace=False)
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_indices] = True
        
        train_dfs.append(group[~test_mask])
        test_dfs.append(group[test_mask])
    
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    logger.info(f"训练集大小: {len(train_df)} 条评分")
    logger.info(f"测试集大小: {len(test_df)} 条评分")
    
    return train_df, test_df


def evaluate_model(model, test_df):
    """
    评估模型
    
    Args:
        model: 训练好的模型
        test_df: 测试集
        
    Returns:
        float: RMSE评分
    """
    logger.info("正在测试集上评估模型...")
    
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        pred = model.predict(row['userId'], row['movieId'])
        predictions.append(pred)
        actuals.append(row['rating'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    logger.info(f"测试集RMSE: {rmse:.4f}")
    
    return rmse


def save_model(model, filepath):
    """
    保存模型
    
    Args:
        model: 训练好的模型
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"模型已保存到: {filepath}")


def main():
    """
    主函数：加载数据、训练模型、评估并保存
    """
    try:
        from utils import load_ratings
        
        logger.info("=" * 60)
        logger.info("开始训练推荐模型")
        logger.info("=" * 60)
        
        # 加载评分数据
        ratings_df = load_ratings()
        
        # 划分数据集
        train_df, test_df = split_data(ratings_df, test_size=0.2)
        
        # 训练模型
        model = UserBasedCF(n_neighbors=20)
        model.fit(train_df)
        
        # 评估模型
        rmse = evaluate_model(model, test_df)
        
        # 保存模型（使用项目根目录相对路径）
        model_path = os.path.join(ROOT_DIR, 'src', 'models', 'svd_model.pkl')
        save_model(model, model_path)
        
        # 打印训练结果摘要
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info(f"训练集大小: {len(train_df)} 条评分")
        logger.info(f"测试集大小: {len(test_df)} 条评分")
        logger.info(f"测试集RMSE: {rmse:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
