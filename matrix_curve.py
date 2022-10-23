from sklearn.model_selection import learning_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np

class matrix_curve:


    def plot_cofusion_matrix(self, y_test, y_pred, name):
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_names = ["y_test", "y_pred"]  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.subplots_adjust(left=0.086, top=0.917)
        plt.title(f'Confusion matrix for a {name}  model', fontsize=14, y=1.03)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

    def plot_traning_curves(self, x, y, model, name):
        train_sizes = [1, 100, 500, 2000, 5000, 7654, 10000, 15000, 20000, 22000,25000,30000,36015]
        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=model,
            X=x,
            y=y, train_sizes=train_sizes, cv=5,
            scoring='neg_mean_squared_error')

        train_scores_mean = -train_scores.mean(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)

        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, validation_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title(f'Learning curves for a {name} model', fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, 40)
        plt.show()