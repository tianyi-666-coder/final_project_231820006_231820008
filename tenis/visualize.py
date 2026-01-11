import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class Visualizer:
    """可视化类"""
    
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, model_name):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='Training Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title(f'{model_name} - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title(f'{model_name} - Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1分数曲线
        axes[0, 2].plot(history['train_f1'], label='Training F1')
        axes[0, 2].plot(history['val_f1'], label='Validation F1')
        axes[0, 2].set_title(f'{model_name} - F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # AUC曲线
        axes[1, 0].plot(history['val_auc'], label='Validation AUC', color='red')
        axes[1, 0].set_title(f'{model_name} - AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率（如果有）
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], label='Learning Rate', color='purple')
            axes[1, 1].set_title(f'{model_name} - Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{model_name} - Confusion Matrix')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_score, model_name):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return fig, roc_auc
    
    def plot_feature_importance(self, importance_df, model_name, top_n=20):
        """绘制特征重要性"""
        if importance_df is None or len(importance_df) == 0:
            return None
        
        # 取前N个最重要的特征
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_features['importance'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Top {top_n} Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_momentum_analysis(self, analysis_dict, model_name):
        """绘制动量分析"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Winning Streaks Distribution', 
                          'Momentum Shift Points',
                          'Prediction Confidence',
                          'Momentum Metrics'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # 1. 连胜分布
        winning_streaks = analysis_dict.get('winning_streaks', [])
        fig.add_trace(
            go.Histogram(x=winning_streaks, nbinsx=20, name='Winning Streaks'),
            row=1, col=1
        )
        
        # 2. 动量转换点
        momentum_shifts = analysis_dict.get('momentum_shifts', [])
        if momentum_shifts:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(momentum_shifts))),
                    y=momentum_shifts,
                    mode='lines+markers',
                    name='Momentum Shifts'
                ),
                row=1, col=2
            )
        
        # 3. 预测置信度
        high_conf = analysis_dict.get('high_confidence_predictions', 0)
        low_conf = analysis_dict.get('low_confidence_predictions', 0)
        total = high_conf + low_conf
        medium_conf = total - high_conf - low_conf if total > 0 else 0
        
        fig.add_trace(
            go.Bar(
                x=['High Confidence', 'Medium Confidence', 'Low Confidence'],
                y=[high_conf, medium_conf, low_conf],
                name='Confidence Levels'
            ),
            row=2, col=1
        )
        
        # 4. 动量指标表格
        momentum_metrics = [
            ['Average Winning Streak', f"{analysis_dict.get('avg_winning_streak', 0):.2f}"],
            ['Max Winning Streak', f"{analysis_dict.get('max_winning_streak', 0):.2f}"],
            ['Momentum Shifts', f"{len(momentum_shifts)}"],
            ['Confidence Ratio', f"{analysis_dict.get('confidence_ratio', 0):.2%}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*momentum_metrics))),
                name='Momentum Metrics'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=f"{model_name} - Momentum Analysis")
        return fig
    
    def plot_model_comparison(self, comparison_df):
        """绘制模型比较图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1-Score Comparison',
                          'AUC Comparison', 'Overall Performance'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatterpolar'}]]
        )
        
        models = comparison_df['Model'].tolist()
        
        # 1. 准确率比较
        fig.add_trace(
            go.Bar(x=models, y=comparison_df['Accuracy'], name='Accuracy'),
            row=1, col=1
        )
        
        # 2. F1分数比较
        fig.add_trace(
            go.Bar(x=models, y=comparison_df['F1-Score'], name='F1-Score'),
            row=1, col=2
        )
        
        # 3. AUC比较
        fig.add_trace(
            go.Bar(x=models, y=comparison_df['AUC'], name='AUC'),
            row=2, col=1
        )
        
        # 4. 雷达图综合比较
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        for i, model in enumerate(models):
            values = comparison_df.loc[comparison_df['Model'] == model, metrics].values.flatten().tolist()
            values.append(values[0])  # 闭合雷达图
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison",
            showlegend=True
        )
        
        fig.update_polars(radialaxis=dict(range=[0, 1]), row=2, col=2)
        
        return fig
    
    def plot_attention_weights(self, attention_weights, sequence_length, model_name):
        """绘制注意力权重热力图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Batch Sample')
        ax.set_title(f'{model_name} - Attention Weights Heatmap')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        return fig