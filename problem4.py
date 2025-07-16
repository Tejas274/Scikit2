
from sklearn import datasets # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
# import some data within sklearn for iris classification
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Splitting data into train and testing part
# The 25 % of data is test size of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# importing pipes for making the Pipe flow
from sklearn.pipeline import Pipeline # type: ignore
# pipe flow is :
# PCA(Dimension reduction to two) -> Scaling the data -> DecisionTreeClassification
pipe = Pipeline([('pca', PCA(n_components = 2)), ('std', StandardScaler()), ('decision_tree', DecisionTreeClassifier())], verbose = True)

# fitting the data in the pipe
pipe.fit(X_train, y_train)

# scoring data
from sklearn.metrics import accuracy_score # type: ignore
print(accuracy_score(y_test, pipe.predict(X_test)))

from sklearn import datasets # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
# import some data within sklearn for iris classification
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.pipeline import Pipeline # type: ignore
pipe = Pipeline([('pca', PCA(n_components = 2)), ('std', StandardScaler()), ('Decision_tree', DecisionTreeClassifier())], verbose = True)

pipe.fit(X_train, y_train)

# to see all the hyper parameters
pipe.get_params()