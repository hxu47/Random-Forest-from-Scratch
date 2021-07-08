import numpy as np

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)
        
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node. This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] <= self.split:
            if isinstance(self.lchild, LeafNode):
                return self.lchild
            else:
                return self.lchild.leaf(x_test)
        elif x_test[self.col] > self.split:
            if isinstance(self.rchild, LeafNode):
                return self.rchild
            else:
                return self.rchild.leaf(x_test)
        
        
class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction
        
        
class RFDecisionTree621:
    def __init__(self, min_samples_leaf=1, max_features=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
        self.max_features = max_features
        self.oob_idxs = None
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y, self.max_features)
        
    def fit_(self, X, y, max_features):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = self.RFbestsplit(X, y, self.max_features)
        if col == -1:
            return self.create_leaf(y)
        X_col = X[:, col]
        lchild = self.fit_(X[X_col<=split], y[X_col<=split], self.max_features)
        rchild = self.fit_(X[X_col>split], y[X_col>split], self.max_features)
        return DecisionNode(col, split, lchild, rchild)
    
    def RFbestsplit(self, X, y, max_features):
        best = {'col':-1, 'split':-1, 'loss':self.loss(y)}
        p = X.shape[1]   # p: number of features
        feature_indxs = np.random.choice(p, size = int(p*self.max_features)) # pick max_features variables from all p
        for col in feature_indxs:
            X_col = X[:, col]
            candidates = np.random.choice(X_col, size=min(len(X_col),11), replace=False)
            #candidates = np.random.choice(X_col, size=min(len(X),11), replace=False)
            for split in candidates:
                yl = y[X_col <= split]
                yr = y[X_col > split]
                yl_length = len(yl)
                yr_length = len(yr)
                if yl_length < self.min_samples_leaf or yr_length < self.min_samples_leaf:
                    continue
                l = (yl_length*self.loss(yl)+yr_length*self.loss(yr)) / len(y)
                if l == 0:
                    return col, split
                if l < best['loss']:
                    best['col'],  best['split'], best['loss']= col, split, l
        return best['col'], best['split']
       
        
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """    
        if len(X_test.shape) == 1:
            return np.array(self.root.predict(X_test))
        return np.array([self.root.predict(record) for record in X_test])
    
    
class RFRegressionTree621(RFDecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=1):
        super().__init__(min_samples_leaf, loss=np.std)
        
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return 1 - ((y_test - y_pred)** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
        
    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))
        
        
class RFClassifierTree621(RFDecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=1):
        super().__init__(min_samples_leaf, loss=gini)
        
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return sum(y_test == y_pred) / len(y_test)
        
    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.bincount(y).argmax())
    
    
def gini(y):
    """
    Return the gini impurity score for values in y"
    Reference: https://github.com/parrt/msds621/blob/master/notebooks/trees/gini-impurity.ipynb
        """
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p**2)