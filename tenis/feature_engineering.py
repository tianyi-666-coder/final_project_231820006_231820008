import numpy as np
import pandas as pd

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.momentum_features = []
        
    def calculate_momentum_features(self, data, window_size=5):
        """
        计算动量特征
        
        Args:
            data: 原始数据
            window_size: 动量窗口大小
            
        Returns:
            包含动量特征的数据框
        """
        print("计算动量特征...")
        
        # 创建副本
        df = data.copy()
        
        # 1. 得分差
        df['score_diff'] = df['p1_points_won'] - df['p2_points_won']
        df['score_momentum'] = df['score_diff'].rolling(window=window_size, min_periods=1).mean()
        
        # 2. 胜率动量
        df['p1_win_rate'] = df['point_victor'].apply(lambda x: 1 if x == 1 else 0)
        df['p2_win_rate'] = df['point_victor'].apply(lambda x: 1 if x == 2 else 0)
        df['p1_win_momentum'] = df['p1_win_rate'].rolling(window=window_size, min_periods=1).mean()
        df['p2_win_momentum'] = df['p2_win_rate'].rolling(window=window_size, min_periods=1).mean()
        
        # 3. 破发点动量
        df['p1_break_momentum'] = df['p1_break_pt_won'].rolling(window=window_size, min_periods=1).sum()
        df['p2_break_momentum'] = df['p2_break_pt_won'].rolling(window=window_size, min_periods=1).sum()
        
        # 4. 失误动量
        df['p1_error_momentum'] = df['p1_unf_err'].rolling(window=window_size, min_periods=1).sum()
        df['p2_error_momentum'] = df['p2_unf_err'].rolling(window=window_size, min_periods=1).sum()
        
        # 5. Ace动量
        df['p1_ace_momentum'] = df['p1_ace'].rolling(window=window_size, min_periods=1).sum()
        df['p2_ace_momentum'] = df['p2_ace'].rolling(window=window_size, min_periods=1).sum()
        
        # 6. 移动距离动量
        df['p1_distance_momentum'] = df['p1_distance_run'].rolling(window=window_size, min_periods=1).mean()
        df['p2_distance_momentum'] = df['p2_distance_run'].rolling(window=window_size, min_periods=1).mean()
        
        # 7. 连续得分/失分
        df['p1_streak'] = self.calculate_streak(df['point_victor'].values, player=1)
        df['p2_streak'] = self.calculate_streak(df['point_victor'].values, player=2)
        
        # 8. 比赛状态特征
        df['set_difference'] = df['p1_sets'] - df['p2_sets']
        df['game_difference'] = df['p1_games'] - df['p2_games']
        
        # 9. 关键时刻特征
        df['is_critical_point'] = ((df['p1_break_pt'] == 1) | (df['p2_break_pt'] == 1)).astype(int)
        
        # 10. 疲劳特征
        df['cumulative_rally'] = df['rally_count'].cumsum()


        # 11. 压力特征
        df['pressure_score'] = self.calculate_pressure_score(df)
        
        # 12. 技术统计比率
        df['p1_winner_error_ratio'] = np.where(df['p1_unf_err'] == 0, 
                                               df['p1_winner'], 
                                               df['p1_winner'] / df['p1_unf_err'])
        df['p2_winner_error_ratio'] = np.where(df['p2_unf_err'] == 0,
                                               df['p2_winner'],
                                               df['p2_winner'] / df['p2_unf_err'])
        
        # 13. 发球优势特征
        df['serve_advantage'] = self.calculate_serve_advantage(df)
        
        # 14. 比赛阶段特征
        df['match_progress'] = df['point_no'] / df.groupby(['match_id', 'set_no', 'game_no'])['point_no'].transform('max')
        
        # 15. 标准化跑动距离差异
        df['distance_diff_normalized'] = (df['p1_distance_run'] - df['p2_distance_run']) / \
                                         (df['p1_distance_run'] + df['p2_distance_run'] + 1e-8)
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')


                
        self.momentum_features = [
            'score_momentum', 'p1_win_momentum', 'p2_win_momentum',
            'p1_break_momentum', 'p2_break_momentum',
            'p1_error_momentum', 'p2_error_momentum',
            'p1_ace_momentum', 'p2_ace_momentum',
            'p1_distance_momentum', 'p2_distance_momentum',
            'p1_streak', 'p2_streak',
            'set_difference', 'game_difference',
            'is_critical_point', 'cumulative_rally',
            'pressure_score',
            'p1_winner_error_ratio', 'p2_winner_error_ratio',
            'serve_advantage', 'match_progress', 'distance_diff_normalized'
        ]
        
        print(f"添加了 {len(self.momentum_features)} 个动量特征")
        return df
    
    def calculate_streak(self, point_victors, player):
        """计算连续得分"""
        streak = np.zeros(len(point_victors))
        current_streak = 0
        
        for i, victor in enumerate(point_victors):
            if victor == player:
                current_streak += 1
            else:
                current_streak = 0 if current_streak > 0 else -1
            
            streak[i] = current_streak
        
        return streak
    
    def calculate_pressure_score(self, df):
        """计算压力分数"""
        pressure = np.zeros(len(df))
        for i in range(1, len(df)):
            # 基于比分、比赛阶段、破发点等因素
            score_pressure = abs(df['p1_score_numeric'].iloc[i] - df['p2_score_numeric'].iloc[i])
            game_pressure = (df['game_no'].iloc[i] > 5) * 0.5  # 后段比赛压力大
            break_pressure = (df['p1_break_pt'].iloc[i] | df['p2_break_pt'].iloc[i]) * 1.0
            
            pressure[i] = score_pressure + game_pressure + break_pressure
        return pressure
    
    def calculate_serve_advantage(self, df):
        """计算发球优势"""
        serve_adv = np.zeros(len(df))
        for i in range(1, min(20, len(df))):  # 最近20分
            if i == 0:
                continue
            # 发球方胜率
            server_won = ((df['server'].iloc[i] == 1) & (df['point_victor'].iloc[i] == 1)) | \
                        ((df['server'].iloc[i] == 2) & (df['point_victor'].iloc[i] == 2))
            serve_adv[i] = server_won.astype(float)
        
        # 使用滑动窗口平均
        return pd.Series(serve_adv).rolling(window=10, min_periods=1).mean().values
    
    def get_momentum_feature_names(self):
        """获取动量特征名称"""
        return self.momentum_features