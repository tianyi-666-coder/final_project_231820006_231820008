# ================== 屏蔽 Streamlit 首次引导 ==================
import os
os.environ["STREAMLIT_SUPPRESS_ONBOARDING"] = "1"

# ================== 允许 python 直接运行 ==================
import sys
import subprocess

def ensure_streamlit_run():
    if not os.environ.get("STREAMLIT_RUN_CONTEXT"):
        os.environ["STREAMLIT_RUN_CONTEXT"] = "1"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)
        ])
        sys.exit(0)

ensure_streamlit_run()
# ============================================================


import streamlit as st
import torch
from pathlib import Path

# ✅ 与 models.py 中真实类名一致
from models import (
    SimpleRNNModel,
    LSTMModel,
    GRUAttentionModel,
    TransformerModel
)

# ------------------ 路径配置 ------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录
RESULTS_DIR = BASE_DIR / "results"                  # results 在 tenis 外

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 模型映射 ------------------
MODEL_CLASS_MAP = {
    "RNN": SimpleRNNModel,
    "LSTM": LSTMModel,
    "GRU+Attention": GRUAttentionModel,
    "Transformer": TransformerModel,
}

MODEL_PATH_MAP = {
    "RNN": RESULTS_DIR / "rnn_model.pth",
    "LSTM": RESULTS_DIR / "lstm_model.pth",
    "GRU+Attention": RESULTS_DIR / "gru_attention_model.pth",
    "Transformer": RESULTS_DIR / "transformer_model.pth",
}

# ------------------ 模型加载 ------------------
def load_model(model_name: str, input_size: int):
    model_path = MODEL_PATH_MAP[model_name]
    model_class = MODEL_CLASS_MAP[model_name]

    if not model_path.exists():
        st.error(f"模型文件不存在：{model_path}")
        st.stop()

    model = model_class(input_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ================== Streamlit UI ==================
st.set_page_config(
    page_title="网球比赛动量分析仪表盘",
    layout="wide"
)

st.title(" 网球比赛动量分析与结果预测")

model_name = st.selectbox(
    "选择模型",
    list(MODEL_CLASS_MAP.keys())
)

st.info(f"当前选择模型：{model_name}")

# ------------------ 结果展示（推荐用已有文件） ------------------
html_path = RESULTS_DIR / f"{model_name.lower().replace('+', '_')}_momentum_analysis.html"

if html_path.exists():
    st.subheader("📈 动量变化分析")
    st.components.v1.html(
        html_path.read_text(encoding="utf-8"),
        height=600,
        scrolling=True
    )
else:
    st.warning("未找到对应的动量分析 HTML 文件")

