from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score as ras,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_and_evaluate_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
    model.fit(X_train, y_train)
    
    # Evaluation
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    test_preds_class = model.predict(X_test)  # Predicted classes for classification metrics
    
    print('Training ROC-AUC Score:', ras(y_train, train_preds))
    print('Validation ROC-AUC Score:', ras(y_test, test_preds))
    
    # Additional Metrics
    print('Accuracy:', accuracy_score(y_test, test_preds_class))
    print('Precision:', precision_score(y_test, test_preds_class))
    print('Recall:', recall_score(y_test, test_preds_class))
    print('F1 Score:', f1_score(y_test, test_preds_class))
    
    # Confusion Matrix
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    cm.plot(cmap='Blues')
    plt.show()
    
    return model