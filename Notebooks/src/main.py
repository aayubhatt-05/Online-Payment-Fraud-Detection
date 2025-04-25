from preprocessing import load_and_preprocess_data
from visualization import plot_data_overview
from train_model import train_and_evaluate_model

def main():
    # File path to the dataset
    file_path = "C:/Users/Anshita/Downloads/ML_Project/datasets/new_file.csv"
    
    # Load and preprocess data
    data, X, y = load_and_preprocess_data(file_path)
    
    # Visualize data
    plot_data_overview(data)
    
    # Train and evaluate model
    model = train_and_evaluate_model(X, y)
    
    # Save the trained model
    import joblib
    joblib.dump(model, '../models/random_forest_model.pkl')
    print("Model saved to '../models/random_forest_model.pkl'")

if __name__ == "__main__":
    main()