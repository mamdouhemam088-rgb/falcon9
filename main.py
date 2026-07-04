from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import prepare_features
from src.model_training import train_models
from src.evaluation import evaluate_models

def main():
    df = load_data("data/spacex_web_scraped.csv")

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = prepare_features(df)

    models = train_models(X_train, y_train)

    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
