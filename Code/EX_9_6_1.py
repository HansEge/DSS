# %load ../standard_import.txt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


'''matplotlib inline'''
plt.style.use('seaborn-white')

def plot_svc(model, X, y, title):
    # define maximum and minimum of x-coordinates
    x_min = X[:, 0].min()-0.5
    x_max = X[:, 0].max()+0.5
    # define maximum and minimum of y-coordinates
    y_min = X[:, 1].min()-0.5
    y_max = X[:, 1].max()+0.5

    # Make colored side of hyperplane using np.meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.1)

    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = model.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=35, linewidths='2')
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()
    print('Number of support vectors: ', model.support_.size)



def main():
    # Make 20 random samples
    np.random.seed(9)
    X = np.random.randn(20, 2)

    # Divide these 20 samples into two classes, labeled 1 and -1
    y = np.repeat([1, -1], 10)
    X[y == -1] = X[y == -1] + 1

    # Plot the generated samples
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Paired)
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()

    # Generate a linear support vector classifier using the SCV lib with C=1
    svc_C1 = SVC(C=1.0, kernel='linear')
    svc_C1.fit(X, y)

    # Plot the classifier
    title_C_1 = 'With the value of C = 1'
    plot_svc(svc_C1, X, y, title_C_1)

    # Generate a new linear support vector classifier with C=0.1
    svc_C0_1 = SVC(C=0.1, kernel='linear')
    svc_C0_1.fit(X,y)

    # Plot the classifier with C=0.1
    title_C_0_1 = 'With the value of C = 0.1'
    plot_svc(svc_C0_1, X, y,title_C_0_1)

    # Using cross-validation to chose optimal value for C
    C_values = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
    clf = GridSearchCV(SVC(kernel='linear'), C_values)
    clf.fit(X, y)

    # Print parameter of C that had the best score
    print(clf.best_params_)

    # Generating test data
    np.random.seed(1)
    X_test = np.random.randn(20, 2)
    y_test = np.random.choice([-1, 1], 20)
    X_test[y_test == 1] = X_test[y_test == 1] - 1

    # Using the classifier with C=1 to predict the test data
    y_pred = svc_C1.predict(X_test)

    # Print the confusion matrix to se score
    print(confusion_matrix(y_test, y_pred))

    # Generate a new linear support vector classifier with C=0.01
    svc_C0_01 = SVC(C=0.001, kernel='linear')
    svc_C0_01.fit(X, y)

    # Using the classifier with C=0.01 to predict the test data
    y_pred = svc_C0_01.predict(X_test)

    # Print the confusion matrix to se score
    print(confusion_matrix(y_test, y_pred))

    # The printed confusion matrices shows that the classification is the same with C value of 1 and 0.01

if __name__ == '__main__':
     main()