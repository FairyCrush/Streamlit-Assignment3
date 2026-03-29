import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train, preprocessor):
    model = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "model.pkl")
    preprocessor_path = os.path.join("models", "preprocessor.pkl")
    
    joblib.dump(pipeline, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print("Model Directory: ", model_path)
    print("Preprocessor Directory: ", preprocessor_path)
    
    return pipeline