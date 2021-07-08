from dtree import *
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        #self.nunique = None

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree. After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        trees = set()
        for i in range(self.n_estimators):
            
            # bootstrap: sample with replacement
            n, p = X.shape
            idx = np.random.randint(0, n, size=int(2/3*n))
            X_train = X[idx]
            y_train = y[idx]
            
            # train a signle tree
            atree = self.trees(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
            atree.fit(X_train, y_train)
            
            # get indexes for training data
            atree.oob_idxs = set(np.arange(0, n)) - set(idx)
            
            # store the trained tree
            trees.add(atree)

        self.trees_set = set(trees)
        
        # calculate the number of unique values and store in nunique
        #self.nunique = len(np.unique(y))
            
        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y)
            
            
            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = RFRegressionTree621

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = []
        for record in X_test:
            leaves = set()
            for tree in self.trees_set:
                if isinstance(tree.root, LeafNode):
                    leaves.add(tree.root)
                else:
                    leaves.add(tree.root.leaf(record))

            nobs = sum(leaf.n for leaf in leaves)
            ysum = sum(leaf.n * leaf.prediction for leaf in leaves)

            predictions.append( ysum/nobs )
            
        return np.array(predictions)
    
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
        
    def compute_oob_score(self, X, y) -> float:
        n = X.shape[0]
        oob_counts = np.zeros(n, dtype=int)
        oob_preds = np.zeros(n)
        
        for t in self.trees_set:

            for oob_ind in t.oob_idxs:

                # get the leaf of this tree for this record 
                if isinstance(t.root, LeafNode):
                    leaf = t.root
                else:
                    leaf = t.root.leaf(X[oob_ind])

                leafsize = leaf.n
                oob_preds[oob_ind] += leafsize * leaf.prediction
                oob_counts[oob_ind] += leafsize

        oob_avg_preds = oob_preds[oob_counts>0] / oob_counts[oob_counts>0]
        return r2_score(y[oob_counts>0], oob_avg_preds)
        
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = RFClassifierTree621
        self.nunique = None

    def predict(self, X_test) -> np.ndarray:
        predictions = []
        
        for record in X_test:
            counts = {}
            for tree in self.trees_set:
                if isinstance(tree.root, LeafNode):
                    leaf = tree.root
                else:
                    leaf = tree.root.leaf(record)
                counts[leaf.prediction] = counts.get(leaf.prediction, 0) + leaf.n
            predictions.append(max(counts, key=counts.get))
            
        return np.array(predictions)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def compute_oob_score(self, X, y) -> float:
        n = X.shape[0]
        self.nunique = len(np.unique(y))  # calculate the number of unique values
        
        oob_counts = np.zeros(n, dtype=int)
        oob_preds = np.zeros(shape=(n, self.nunique))  # 2D matrix
        
        for t in self.trees_set:

            for oob_ind in t.oob_idxs:

                # get the leaf of this tree for this record 
                if isinstance(t.root, LeafNode):
                    leaf = t.root
                else:
                    leaf = t.root.leaf(X[oob_ind])

                leafsize = leaf.n
                tpred = leaf.prediction

                oob_preds[oob_ind, tpred] += leafsize
                oob_counts[oob_ind] += 1
                
        oob_votes = []
        for i, _ in enumerate(oob_counts[oob_counts>0]):
            majority_vote = np.argmax(oob_preds[i])
            oob_votes.append(majority_vote)
            
        return accuracy_score(y[oob_counts>0], oob_votes)