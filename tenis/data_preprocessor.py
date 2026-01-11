import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class TennisDataPreprocessor:
    """网球数据预处理类"""
    
    def __init__(self, data_path, data_dict_path):
        """
        初始化预处理器
        
        Args:
            data_path: 数据文件路径
            data_dict_path: 数据字典路径
        """
        self.data = pd.read_csv(data_path)
        self.data_dict = pd.read_csv(data_dict_path) if data_dict_path else None
        self.label_encoders = {}
        self.scalers = {}
        
    def load_and_clean(self):
        """加载和清洗数据"""
        print(f"原始数据形状: {self.data.shape}")
        
        # 处理时间特征
        self.data['elapsed_time'] = pd.to_timedelta(self.data['elapsed_time'])
        self.data['elapsed_seconds'] = self.data['elapsed_time'].dt.total_seconds()
        
        # 处理分数特征 - 将AD转换为数值
        # 网球记分: 0(love) -> 0, 15 -> 1, 30 -> 2, 40 -> 3, AD -> 4
        score_mapping = {0: 0, '0': 0, 15: 1, '15': 1, 30: 2, '30': 2, 40: 3, '40': 3, 'AD': 4, 'ADV': 4}
        
        # 转换p1_score和p2_score
        self.data['p1_score_numeric'] = self.data['p1_score'].map(score_mapping).fillna(0).astype(int)
        self.data['p2_score_numeric'] = self.data['p2_score'].map(score_mapping).fillna(0).astype(int)
        
        # 处理分类特征
        categorical_cols = ['serve_width', 'serve_depth', 'return_depth', 'winner_shot_type']
        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                # 处理NaN值
                self.data[col] = self.data[col].fillna('NA')
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        # 处理缺失值
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        
        print(f"清洗后数据形状: {self.data.shape}")
        return self.data
    
    def create_sequence_features(self, window_size=10):
        """
        创建时间序列特征
        
        Args:
            window_size: 滑动窗口大小
            
        Returns:
            序列特征和标签
        """
        print(f"创建序列特征，窗口大小: {window_size}")
        
        # 选择特征列 - 使用数值型分数
        feature_cols = [
            'elapsed_seconds', 'set_no', 'game_no', 'point_no',
            'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
            'p1_score_numeric', 'p2_score_numeric',  # 使用转换后的数值分数
            'server', 'serve_no',
            'p1_points_won', 'p2_points_won',
            'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
            'p1_double_fault', 'p2_double_fault',
            'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt',
            'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt',
            'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed',
            'p1_distance_run', 'p2_distance_run', 'rally_count', 'speed_mph'
        ]
        
        # 添加分类特征（如果存在）
        cat_cols = ['serve_width', 'serve_depth', 'return_depth', 'winner_shot_type']
        for col in cat_cols:
            if col in self.data.columns:
                feature_cols.append(col)
        
        # 确保所有特征列都存在
        available_cols = [col for col in feature_cols if col in self.data.columns]
        print(f"使用 {len(available_cols)} 个特征")
        
        # 标准化数值特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data[available_cols])
        self.scalers['features'] = scaler
        
        # 创建序列
        sequences = []
        labels = []
        
        for i in range(window_size, len(scaled_features)):
            seq = scaled_features[i-window_size:i]
            # 确保point_victor存在且有效
            if 'point_victor' in self.data.columns:
                label_val = self.data['point_victor'].iloc[i]
                # 将1/2转换为0/1（二分类）
                label = 1 if label_val == 1 else 0  # 1: player1赢 -> 1, 2: player2赢 -> 0
            else:
                # 如果没有point_victor，尝试使用其他列推导
                # 这里使用p1_points_won的变化来判断谁赢得了这一分
                if i > 0 and 'p1_points_won' in self.data.columns:
                    if self.data['p1_points_won'].iloc[i] > self.data['p1_points_won'].iloc[i-1]:
                        label = 1  # player1得分了
                    else:
                        label = 0  # player2得分了
                else:
                    label = 0
            
            sequences.append(seq)
            labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"序列形状: {sequences.shape}, 标签形状: {labels.shape}")
        return sequences, labels, available_cols
    
    def split_data(self, sequences, labels, test_size=0.2, val_size=0.1):
        """分割数据集"""
        # 先分割训练+验证集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels, test_size=test_size, random_state=42, shuffle=False
        )
        
        # 再分割训练集和验证集
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=42, shuffle=False
        )
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

class TennisDataset(Dataset):
    """自定义PyTorch数据集"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

        # 计算类别权重
        class_counts = np.bincount(labels)
        self.class_weights = 1.0 / class_counts
        self.class_weights = self.class_weights / self.class_weights.sum()
    
    def get_class_weights(self):
        return torch.FloatTensor(self.class_weights)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]