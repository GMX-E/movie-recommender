import pandas as pd
import os
import logging

# 获取项目根目录（相对于当前文件的父目录的父目录）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_movies(data_path: str = None) -> pd.DataFrame:
    """
    从movies.csv文件加载电影数据
    
    Args:
        data_path: 数据目录路径，如果为None则使用默认路径
        
    Returns:
        pandas.DataFrame: 包含电影ID、标题和类型的DataFrame
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式错误时抛出
    """
    if data_path is None:
        data_path = os.path.join(ROOT_DIR, 'data')
    
    movies_file = os.path.join(data_path, 'movies.csv')
    
    try:
        # 检查文件是否存在
        if not os.path.exists(movies_file):
            raise FileNotFoundError(f"电影数据文件不存在: {movies_file}")
        
        # 读取CSV文件
        df = pd.read_csv(movies_file)
        
        # 验证必要的列是否存在
        required_columns = ['movieId', 'title', 'genres']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"movies.csv缺少必要的列。期望: {required_columns}, 实际: {list(df.columns)}")
        
        logger.info(f"成功加载电影数据，共 {len(df)} 部电影")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"加载电影数据失败: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"解析电影数据文件失败: {e}")
        raise ValueError(f"文件格式错误: {e}")
    except Exception as e:
        logger.error(f"加载电影数据时发生未知错误: {e}")
        raise


def load_ratings(data_path: str = None) -> pd.DataFrame:
    """
    从ratings.csv文件加载评分数据
    
    Args:
        data_path: 数据目录路径，如果为None则使用默认路径
        
    Returns:
        pandas.DataFrame: 包含用户ID、电影ID、评分和时间戳的DataFrame
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式错误时抛出
    """
    if data_path is None:
        data_path = os.path.join(ROOT_DIR, 'data')
    
    ratings_file = os.path.join(data_path, 'ratings.csv')
    
    try:
        # 检查文件是否存在
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"评分数据文件不存在: {ratings_file}")
        
        # 读取CSV文件
        df = pd.read_csv(ratings_file)
        
        # 验证必要的列是否存在
        required_columns = ['userId', 'movieId', 'rating', 'timestamp']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"ratings.csv缺少必要的列。期望: {required_columns}, 实际: {list(df.columns)}")
        
        # 验证评分范围
        if not (df['rating'] >= 0.5).all() or not (df['rating'] <= 5.0).all():
            logger.warning("检测到评分值超出范围(0.5-5.0)")
        
        logger.info(f"成功加载评分数据，共 {len(df)} 条评分记录")
        return df
        
    except FileNotFoundError as e:
        logger.error(f"加载评分数据失败: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"解析评分数据文件失败: {e}")
        raise ValueError(f"文件格式错误: {e}")
    except Exception as e:
        logger.error(f"加载评分数据时发生未知错误: {e}")
        raise


if __name__ == "__main__":
    # 测试数据加载功能
    try:
        movies_df = load_movies()
        print("电影数据预览:")
        print(movies_df.head())
        print("\n")
        
        ratings_df = load_ratings()
        print("评分数据预览:")
        print(ratings_df.head())
    except Exception as e:
        print(f"测试失败: {e}")
