import os
import pickle

# GMM 모델과 스케일러 로드
def load_gmm_model(model_path="recommendModel/gmm_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ GMM 모델이 존재하지 않습니다: {model_path}")
    
    with open(model_path, "rb") as f:
        gmm, scaler = pickle.load(f)
    return gmm, scaler
