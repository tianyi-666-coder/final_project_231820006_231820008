import numpy as np
import pandas as pd
import torch
import random
import os
import json
import pickle
from datetime import datetime

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子设置为: {seed}")

def save_model(model, path, model_name, history=None):
    """保存模型"""
    # 确保路径存在
    os.makedirs(path, exist_ok=True)
    
    # 保存模型状态
    model_path = os.path.join(path, f'{model_name}_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # 保存模型架构信息
    model_info = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_class': model.__class__.__name__
    }
    
    # 保存训练历史
    if history is not None:
        history_path = os.path.join(path, f'{model_name}_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
    
    # 保存模型信息
    info_path = os.path.join(path, f'{model_name}_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"模型已保存到: {model_path}")
    return model_path

def load_model(model_class, path, model_name, **kwargs):
    """加载模型"""
    model_path = os.path.join(path, f'{model_name}_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型实例
    model = model_class(**kwargs)
    
    # 加载模型权重
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # 加载训练历史
    history_path = os.path.join(path, f'{model_name}_history.pkl')
    history = None
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    
    print(f"模型已加载: {model_name}")
    return model, history


def calculate_momentum_score(predictions, probabilities, window_size=5):
    """计算动量分数"""
    momentum_scores = []
    
    for i in range(len(predictions)):
        start_idx = max(0, i - window_size + 1)
        recent_predictions = predictions[start_idx:i+1]
        
        # 计算近期胜率
        if len(recent_predictions) > 0:
            win_rate = np.mean(recent_predictions == 1)  # Player1获胜比例
        else:
            win_rate = 0.5
        
        # 考虑预测置信度
        confidence = probabilities[i, 1] if predictions[i] == 1 else probabilities[i, 0]
        
        # 动量分数 = 胜率 * 置信度
        momentum_score = win_rate * confidence
        momentum_scores.append(momentum_score)
    
    return np.array(momentum_scores)

def detect_momentum_shifts(momentum_scores, threshold=0.3):
    """检测动量转换点"""
    shifts = []
    
    for i in range(1, len(momentum_scores)):
        change = abs(momentum_scores[i] - momentum_scores[i-1])
        if change > threshold:
            shifts.append(i)
    
    return shifts

def create_experiment_report(results, config, output_path):
    """创建实验报告"""
    report = {
        'experiment_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results': results,
        'summary': {}
    }
    
    # 计算统计摘要
    for model_name, model_results in results.items():
        if 'test' in model_results:
            report['summary'][model_name] = {
                'accuracy': model_results['test']['accuracy'],
                'f1_score': model_results['test']['f1'],
                'auc': model_results['test']['auc']
            }
    
    # 保存报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    
    print(f"实验报告已保存到: {output_path}")
    return report

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU可用，设备数量: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("GPU不可用，使用CPU")
        return False

def get_device():
    """获取设备"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'