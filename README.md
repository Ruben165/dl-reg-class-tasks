# DL Regression & Classification

Deep learning notebook (`impl/DL_Reg_Class_Tasks.ipynb`) covering two cases:

1. **Stock Closing Price Prediction**: BiLSTM + Multi-Head Attention on 20-day windows (`data/X.csv`).
2. **Emotion Classification**: ERNIE 2.0 (`nghuyong/ernie-2.0-base-en`) on 6 emotion labels (`data/Emotion.csv`). The mode is picked automatically:
   - **GPU**: full fine-tuning with mixed precision.
   - **CPU**: frozen ERNIE embeddings + Logistic Regression (fast).

## Setup
```
pip install tensorflow torch transformers scikit-learn pandas matplotlib
```
Run the notebook from `impl/`.

## Output
- `result/`: plots
- `result/model/`: best models (`stock_model.keras`, `emotion_ernie/`, plus `head.joblib` in CPU mode)
