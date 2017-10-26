import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix(df, xlabel='', ylabel='', title='', cmap=plt.cm.Blues):
        axis_x = df.columns.tolist()
        axis_y = df.columns.tolist()
        plt.imshow(df, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.xticks(range(len(axis_x)), axis_x, rotation=45, fontsize=8, ha='right')
        plt.yticks(range(len(axis_y)), axis_y, rotation=0, fontsize=8)
        plt.tight_layout()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        

        # thresh = matrix.max() / 2.
        # for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        #     plt.text(j, i, matrix[i, j],
        #              horizontalalignment="center",
        #              color="white" if matrix[i, j] > thresh else "black")

if __name__ == '__main__':
    df = pd.read_csv('features_dl.csv', index_col=0)
    plot_confusion_matrix(df.corr())