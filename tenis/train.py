import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import wandb
from tqdm import tqdm

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, model_name="model", use_wandb=False):
        self.model = model
        self.model_name = model_name
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            wandb.init(project="tennis-momentum-analysis", name=model_name)
    
    def train_epoch(self, train_loader, criterion, optimizer, device):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
    
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
        
            optimizer.zero_grad()
            model_output = self.model(data)  # 改为 model_output
        
            # 处理模型返回元组的情况
            if isinstance(model_output, tuple):
                output = model_output[0]  # 取第一个元素（主输出）
            else:
                output = model_output
        
            loss = criterion(output, target)
        
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
        
            # 计算预测
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
        
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def validate(self, val_loader, criterion, device):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
    
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                model_output = self.model(data)  # 改为 model_output
            
                # 处理模型返回元组的情况
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
            
                loss = criterion(output, target)
            
                total_loss += loss.item()
            
                # 计算预测和概率
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
            
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 计算AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return avg_loss, accuracy, precision, recall, f1, auc, all_probs, all_preds
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, 
              weight_decay=1e-4, patience=10, device='cuda'):
        """完整训练过程"""
        
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA不可用，使用CPU训练")
        
        self.model = self.model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 训练历史
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'val_auc': []
        }
        
        print(f"开始训练 {self.model_name}，设备: {device}")
        print(f"训练集大小: {len(train_loader.dataset)}，验证集大小: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_epoch(
                train_loader, criterion, optimizer, device
            )
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_probs, val_preds = self.validate(
                val_loader, criterion, device
            )
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 保存历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            
            # 打印指标
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            
            # WandB记录
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss, 'val_loss': val_loss,
                    'train_acc': train_acc, 'val_acc': val_acc,
                    'train_f1': train_f1, 'val_f1': val_f1,
                    'val_auc': val_auc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"最佳模型更新，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，在epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n训练完成，最佳验证损失: {best_val_loss:.4f}")
        
        return history
    
    def test(self, test_loader, device='cuda'):
        """测试模型"""
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        self.model = self.model.to(device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc, test_probs, test_preds = self.validate(
            test_loader, criterion, device
        )
        
        print(f"\n测试结果 - {self.model_name}:")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        print(f"测试精确率: {test_precision:.4f}")
        print(f"测试召回率: {test_recall:.4f}")
        print(f"测试F1分数: {test_f1:.4f}")
        print(f"测试AUC: {test_auc:.4f}")
        
        return {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'auc': test_auc,
            'probabilities': test_probs,
            'predictions': test_preds
        }