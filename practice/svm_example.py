from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets	
from sklearn import svm    	
import numpy as np
import matplotlib.pyplot as plt 

def cross_validation(clf, data, target):
    
    scores = cross_val_score(clf, data, target, cv=5)

    print "Cross Validation Scores"
    
    print scores


def test_model(clf, data, target):

    scores = clf.score(data, target)

    print "Test Scores"

    print scores

def fit_classifier(clf, data, target):

    fitted_clf = clf.fit(data, target)

    return fitted_clf

def get_misclassified(clf, x_test, y_test):

    misclassified = np.where(y_test != clf.predict(x_test))

    print "Misclassified intance indices"
    
    print misclassified
    
def main():

    iris = datasets.load_iris()

    # Separate 40% of data for testing

    # Cross validation

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

    clf = svm.SVC(kernel='linear', C=1)
        
    cross_validation(clf, x_train, y_train)

    fitted_clf = fit_classifier(clf, x_train, y_train)

    test_model(fitted_clf, x_test, y_test)

    get_misclassified(fitted_clf, x_test, y_test)
    
if __name__ == "__main__":
    main()


