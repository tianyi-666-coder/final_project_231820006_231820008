import os
import sys
import argparse
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import TennisDataPreprocessor, TennisDataset
from feature_engineering import FeatureEngineer
from models import LSTMModel, TransformerModel,GRUAttentionModel,SimpleRNNModel
from train import Trainer
from evaluate import ModelEvaluator
from visualize import Visualizer
from utils import set_seed, save_model, create_experiment_report, get_device,load_model
import torch
from torch.utils.data import DataLoader
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='网球比赛动量分析与预测')
    parser.add_argument('--data_path', type=str, default=r'data\2024_Wimbledon_featured_matches.csv', help='数据文件路径')
    parser.add_argument('--dict_path', type=str, default=r'data\2024_data_dictionary.csv', help='数据字典路径')
    parser.add_argument('--output_dir', type=str, default=r'results', help='输出目录')
    parser.add_argument('--window_size', type=int, default=10, help='时间窗口大小')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_wandb', action='store_true', help='使用WandB记录')
    parser.add_argument('--train_all', action='store_true', help='训练所有模型')
    parser.add_argument('--model', type=str, default='all', choices=['lstm', 'transformer','gru_attention','rnn','all'], help='选择要训练的模型')
    parser.add_argument('--run_dashboard',action='store_true',help='是否在训练完成后启动 Streamlit 仪表盘')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置字典
    config = vars(args)
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 设备设置
    device = get_device()
    print(f"使用设备: {device}")
    
    # 1. 数据预处理
    print("\n" + "="*50)
    print("步骤1: 数据预处理")
    print("="*50)
    
    preprocessor = TennisDataPreprocessor(args.data_path, args.dict_path)
    data = preprocessor.load_and_clean()
    
    # 2. 特征工程
    print("\n" + "="*50)
    print("步骤2: 特征工程")
    print("="*50)
    
    feature_engineer = FeatureEngineer()
    data_with_momentum = feature_engineer.calculate_momentum_features(data)
    
    # 更新数据
    preprocessor.data = data_with_momentum
    
    # 3. 创建序列数据
    print("\n" + "="*50)
    print("步骤3: 创建序列数据")
    print("="*50)
    
    sequences, labels, feature_names = preprocessor.create_sequence_features(window_size=args.window_size)
    
    # 4. 分割数据
    print("\n" + "="*50)
    print("步骤4: 数据分割")
    print("="*50)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        sequences, labels, test_size=0.2, val_size=0.1
    )
    
    # 5. 创建数据加载器
    print("\n" + "="*50)
    print("步骤5: 创建数据加载器")
    print("="*50)
    
    train_dataset = TennisDataset(X_train, y_train)
    val_dataset = TennisDataset(X_val, y_val)
    test_dataset = TennisDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练批次: {len(train_loader)}，验证批次: {len(val_loader)}，测试批次: {len(test_loader)}")
    
    # 6. 模型训练
    print("\n" + "="*50)
    print("步骤6: 模型训练")
    print("="*50)
    
    input_size = X_train.shape[2]
    print(f"输入特征维度: {input_size}")
    
    models_to_train = []
    if args.model == 'all' or args.train_all:
        models_to_train = ['lstm', 'transformer','gru_attention','rnn']
    else:
        models_to_train = [args.model]
    
    results = {}
    histories = {}
    
    for model_type in models_to_train:
        print(f"\n训练 {model_type.upper()} 模型...")
        
        if model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout_rate=0.3,
                num_classes=2
            )
        elif model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                num_heads=8,
                num_layers=2,
                hidden_dim=128,
                dropout_rate=0.1,
                num_classes=2
            )
        elif model_type == 'gru_attention':
            model = GRUAttentionModel(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout_rate=0.3,
                num_classes=2,
                bidirectional=True
            )
        elif model_type == 'rnn':
            model = SimpleRNNModel(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout_rate=0.3,
                num_classes=2,
                bidirectional=True
            )        
        else:
            continue
        
        # 训练模型
        trainer = Trainer(model, model_name=model_type, use_wandb=args.use_wandb)
        history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device
        )
        
        # 测试模型
        test_results = trainer.test(test_loader, device=device)
        
        # 保存模型
        save_model(model, args.output_dir, model_type, history)
        
        # 保存结果
        results[model_type] = {
            'history': history,
            'test': test_results
        }
        histories[model_type] = history
    
    # 7. 模型评估与可视化
    print("\n" + "="*50)
    print("步骤7: 模型评估与可视化")
    print("="*50)
    
    visualizer = Visualizer()
    
    # 绘制训练历史
    for model_type, history in histories.items():
        fig = visualizer.plot_training_history(history, model_type.upper())
        fig.savefig(os.path.join(args.output_dir, f'{model_type}_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # 加载最佳模型进行评估
    evaluators = {}
    for model_type in models_to_train:
        if model_type == 'lstm':
            model_class = LSTMModel
            model_kwargs = {
                'input_size': input_size,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'num_classes': 2
            }
        elif model_type == 'transformer':
            model_class = TransformerModel
            model_kwargs = {
                'input_size': input_size,
                'num_heads': 8,
                'num_layers': 2,
                'hidden_dim': 128,
                'dropout_rate': 0.1,
                'num_classes': 2
            }

        elif model_type == 'gru_attention':
            model_class = GRUAttentionModel
            model_kwargs = {
                'input_size': input_size,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'num_classes': 2,
                'bidirectional': True
            }
        elif model_type == 'rnn':
            model_class = SimpleRNNModel
            model_kwargs = {
                'input_size': input_size,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'num_classes': 2,
                'bidirectional': True
            }        

        
        # 加载模型
        try:
            model, _ = load_model(model_class, args.output_dir, model_type, **model_kwargs)
            evaluator = ModelEvaluator(model, model_type)
            evaluators[model_type] = evaluator
            
            # 获取预测
            y_pred, y_probs, y_true = evaluator.get_predictions(test_loader, device=device)
            
            # 绘制混淆矩阵
            fig = visualizer.plot_confusion_matrix(y_true, y_pred, model_type.upper())
            fig.savefig(os.path.join(args.output_dir, f'{model_type}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 绘制ROC曲线
            fig, auc_score = visualizer.plot_roc_curve(y_true, y_probs[:, 1], model_type.upper())
            fig.savefig(os.path.join(args.output_dir, f'{model_type}_roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 计算特征重要性（如果适用）
            importance_df = evaluator.calculate_feature_importance(model, feature_names)
            if importance_df is not None:
                fig = visualizer.plot_feature_importance(importance_df, model_type.upper())
                if fig:
                    fig.savefig(os.path.join(args.output_dir, f'{model_type}_feature_importance.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # 分析动量模式
            momentum_analysis = evaluator.analyze_momentum_patterns(y_pred, y_probs, data_with_momentum)
            
            # 绘制动量分析图
            fig = visualizer.plot_momentum_analysis(momentum_analysis, model_type.upper())
            fig.write_html(os.path.join(args.output_dir, f'{model_type}_momentum_analysis.html'))
            
            print(f"\n{model_type.upper()} 模型动量分析:")
            print(f"  平均连胜: {momentum_analysis.get('avg_winning_streak', 0):.2f}")
            print(f"  最大连胜: {momentum_analysis.get('max_winning_streak', 0)}")
            print(f"  动量转换点数量: {len(momentum_analysis.get('momentum_shifts', []))}")
            print(f"  高置信度预测比例: {momentum_analysis.get('confidence_ratio', 0):.2%}")
            
        except Exception as e:
            print(f"加载或评估 {model_type} 模型时出错: {e}")
    
    # 8. 模型比较
    if len(evaluators) > 1:
        print("\n" + "="*50)
        print("步骤8: 模型比较")
        print("="*50)
        
        # 收集测试结果
        test_results_dict = {}
        for model_type, evaluator in evaluators.items():
            if model_type in results and 'test' in results[model_type]:
                test_results_dict[model_type] = results[model_type]['test']
        
        # 创建比较数据框
        comparison_data = []
        for model_name, res in test_results_dict.items():
            comparison_data.append({
                'Model': model_name.upper(),
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1'],
                'AUC': res['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        comparison_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)
        print("\n模型比较结果:")
        print(comparison_df.to_string(index=False))
        
        # 绘制模型比较图
        fig = visualizer.plot_model_comparison(comparison_df)
        fig.write_html(os.path.join(args.output_dir, 'model_comparison.html'))
        
        # 保存静态比较图
        fig_static, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'F1-Score', 'AUC']
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            comparison_df.plot.bar(x='Model', y=metric, ax=ax, legend=False)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        # 雷达图
        ax = axes[1, 1]
        models = comparison_df['Model'].tolist()
        metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for idx, model in enumerate(models):
            values = comparison_df.loc[comparison_df['Model'] == model, metrics_for_radar].values.flatten().tolist()
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_for_radar)
        ax.set_title('Overall Performance Radar')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        fig_static.savefig(os.path.join(args.output_dir, 'model_comparison_static.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_static)
    
    # 9. 创建实验报告
    print("\n" + "="*50)
    print("步骤9: 创建实验报告")
    print("="*50)
    
    experiment_report = create_experiment_report(
        results=results,
        config=config,
        output_path=os.path.join(args.output_dir, 'experiment_report.json')
    )


    # 10. 启动 Dashboard
    if args.run_dashboard:
        print("\n启动模型动量分析仪表盘 Dashboard ...")

        dashboard_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dashboard.py"
        )

        if not os.path.exists(dashboard_path):
            raise FileNotFoundError("未找到 dashboard.py")

        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path
        ])

    
    # 11. 总结
    print("\n" + "="*50)
    print("实验完成!")
    print("="*50)
    print(f"结果已保存到: {args.output_dir}")
    print("\n生成的文件:")
    
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.endswith(('.png', '.html', '.csv', '.json', '.pth', '.pkl')):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {file} ({file_size:.1f} KB)")
    
    print("\n下一步:")
    print("1. 查看 results/ 目录下的可视化结果")
    print("2. 分析 model_comparison.csv 文件比较模型性能")
    print("3. 查看 experiment_report.json 获取完整实验详情")

if __name__ == '__main__':
    main()