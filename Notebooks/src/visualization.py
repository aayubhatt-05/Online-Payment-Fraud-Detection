import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_data_overview(data):
    sns.countplot(x='type', data=data)
    plt.title('Transaction Type Distribution')
    plt.show()

    sns.barplot(x='type', y='amount', data=data)
    plt.title('Average Amount by Transaction Type')
    plt.show()

    plt.figure(figsize=(15, 6))
    sns.histplot(data['step'], bins=50, kde=True)
    plt.title('Step Distribution')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(data.apply(lambda x: pd.factorize(x)[0]).corr(),
                cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
    plt.title('Correlation Heatmap')
    plt.show()