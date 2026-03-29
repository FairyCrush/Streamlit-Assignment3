from ingest_data import load_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

def run_pipeline():
    train_df, test_df = load_data()
    
    X_train, X_val, y_train, y_val, preporcessor = preprocess_data(train_df)
    
    model = train_model(X_train, y_train, preporcessor)
    
    accuracy = evaluate_model(model, X_val, y_val)
    
    print("Final Validation Accuracy: ", accuracy)
    
if __name__ == "__main__":
    run_pipeline()