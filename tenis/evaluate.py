import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    
    def get_predictions(self, data_loader, device='cuda'):
        """获取预测结果"""
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        self.model = self.model.to(device)
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                model_output = self.model(data)
            
                # 处理元组输出
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
            
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        """计算混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def calculate_classification_report(self, y_true, y_pred, target_names=None):
        """计算分类报告"""
        if target_names is None:
            target_names = ['Player2 Win', 'Player1 Win']
        
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        return report
    
    def calculate_feature_importance(self, model, feature_names):
        """计算特征重要性（适用于最后一层）"""
        if hasattr(model, 'fc2'):
            weights = model.fc2.weight.data.cpu().numpy()
            # 取绝对值的平均值作为重要性
            importance = np.mean(np.abs(weights), axis=0)
        elif hasattr(model, 'fc'):
            # 对于Transformer模型
            weights = model.fc[-1].weight.data.cpu().numpy()
            importance = np.mean(np.abs(weights), axis=0)
        else:
            return None
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance[:len(feature_names)]
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_momentum_patterns(self, predictions, probabilities, original_data):
        """分析动量模式"""
        analysis = {}
        
        # 1. 连胜模式分析
        winning_streaks = []
        current_streak = 0
        
        for pred in predictions:
            if pred == 1:  # Player1获胜
                current_streak += 1
            else:
                if current_streak > 0:
                    winning_streaks.append(current_streak)
                    current_streak = 0
        
        if current_streak > 0:
            winning_streaks.append(current_streak)
        
        analysis['winning_streaks'] = winning_streaks
        analysis['avg_winning_streak'] = np.mean(winning_streaks) if winning_streaks else 0
        analysis['max_winning_streak'] = np.max(winning_streaks) if winning_streaks else 0
        
        # 2. 动量转换点分析
        momentum_shifts = []
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                momentum_shifts.append(i)
        
        analysis['momentum_shifts'] = momentum_shifts
        analysis['avg_momentum_shift_interval'] = np.mean(np.diff(momentum_shifts)) if len(momentum_shifts) > 1 else 0
        
        # 3. 预测置信度分析
        confidence_threshold = 0.7
        high_confidence_predictions = np.sum(probabilities[:, 1] > confidence_threshold)
        low_confidence_predictions = np.sum(probabilities[:, 1] < (1 - confidence_threshold))
        
        analysis['high_confidence_predictions'] = high_confidence_predictions
        analysis['low_confidence_predictions'] = low_confidence_predictions
        analysis['confidence_ratio'] = high_confidence_predictions / len(predictions) if len(predictions) > 0 else 0
        
        return analysis
    
    def compare_models(self, results_dict):
        """比较多个模型的结果"""
        comparison_df = pd.DataFrame()
        
        for model_name, results in results_dict.items():
            comparison_df = comparison_df.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'AUC': results['auc']
            }, ignore_index=True)
        
        return comparison_df