import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMModel(nn.Module):
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, 
                 dropout_rate=0.5, num_classes=2, bidirectional=True):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * (2 if bidirectional else 1), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
                # 设置遗忘门偏置为1
                if len(param) == self.lstm.hidden_size * 4:
                    param.data[self.lstm.hidden_size:2*self.lstm.hidden_size] = 1
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # 全连接层
        x = self.dropout(context)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        
        return x, attention_weights
    


class TransformerModel(nn.Module):
    """Transformer 模型"""
    
    def __init__(self, input_size, num_heads=8, num_layers=2, hidden_dim=128, dropout_rate=0.1, num_classes=2):
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout_rate)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # 输入维度: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # 嵌入层
        x = self.embedding(x) * math.sqrt(self.input_size)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(x)
        
        # 取最后一个时间步
        last_output = transformer_out[:, -1, :]
        
        # 分类层
        output = self.fc(last_output)
        
        return output

    
class GRUAttentionModel(nn.Module):
    """GRU + Attention 模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout_rate=0.3, num_classes=2, bidirectional=True):
        super(GRUAttentionModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 计算全连接层输入维度
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        
        # 改进的全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # GRU层
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden*2)
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), gru_out).squeeze(1)
        
        # 全连接层
        x = self.dropout(context)
        
        # 应用全连接层
        output = self.fc(x)
        
        return output, attention_weights
    
class SimpleRNNModel(nn.Module):
    """简单RNN模型"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout_rate=0.3, num_classes=2, bidirectional=True):
        super(SimpleRNNModel, self).__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            nonlinearity='tanh'  # 可以选择 'tanh' 或 'relu'
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 计算全连接层输入维度
        fc_input_size = hidden_size * (2 if bidirectional else 1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # RNN层
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden*2)
        
        # 注意力机制
        attention_weights = torch.softmax(self.attention(rnn_out).squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), rnn_out).squeeze(1)
        
        # 全连接层
        x = self.dropout(context)
        output = self.fc(x)
        
        return output, attention_weights    
    
class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class AttentionLayer(nn.Module):
    """注意力层"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output).squeeze(-1), dim=1)
        # attention_weights shape: (batch, seq_len)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        # context_vector shape: (batch, 1, hidden_size)
        
        return context_vector.squeeze(1), attention_weights