from .ccontree import Config, solve, Tree
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.metrics import accuracy_score
from pycontree.export import TreeExporter
from typing_extensions import Self
import numbers
import warnings
import numpy as np
from io import StringIO

class ConTree (BaseEstimator):
    _parameter_constraints: dict = {
        "max_depth": [Interval(numbers.Integral, 0, 20, closed="both")],
        "max_gap": [Interval(numbers.Real, 0, 1, closed="both")],
        "max_gap_decay": [Interval(numbers.Real, 0, 1, closed="both")],
        "time_limit": [Interval(numbers.Real, 0, None, closed="left")],
        "complexity_cost": [Interval(numbers.Real, 0, 1, closed="both")],
    }

    def __init__(self, 
            max_depth: int = 3,
            complexity_cost: float = 0.0,
            max_gap: float = 0,
            max_gap_decay: float = 0.1,
            use_upper_bound: bool = True,
            sort_gini: bool = True,
            time_limit: float = 600, 
            verbose: bool = False):
        """
        Construct a ConTree Classifier
        """
        self.max_depth: int = max_depth
        self.complexity_cost: float = complexity_cost
        self.max_gap: int = max_gap
        self.max_gap_decay: float = max_gap_decay
        self.use_upper_bound: bool = use_upper_bound
        self.sort_gini: bool = sort_gini
        self.verbose: bool = verbose
        self.time_limit: float = time_limit

        self.tree_: Tree = None
        self.train_X_ = None
        self.train_y_ = None

    def _initialize_config(self):
        self._config = Config()
        self._config.max_depth = self.max_depth
        self._config.complexity_cost = self.complexity_cost
        self._config.max_gap = self.max_gap
        self._config.max_gap_decay = self.max_gap_decay
        self._config.use_upper_bound = self.use_upper_bound
        self._config.sort_gini = self.sort_gini
        self._config.verbose = self.verbose

    def _process_fit_data(self, X, y):
        """
        Validate the X and y data before calling fit
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = validate_data(self, X, ensure_min_samples=2, dtype=np.float64)
            y = check_array(y, ensure_2d=False, dtype=np.intc)
            self.n_classes_ = len(np.unique(y))
            if X.shape[0] != y.shape[0]:
                raise ValueError('X and y have different number of rows')
            return X, y

    def _process_score_data(self, X, y):
        """
        Validate the X and y data before calling score
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            warnings.filterwarnings(action="ignore", message="X does not have valid feature names, but .* was fitted with feature names", 
                                    category=UserWarning, module="sklearn.utils.validation")
            X = validate_data(self, X, reset=False, dtype=np.float64)
            y = check_array(y, ensure_2d=False, dtype=np.intc)
            if X.shape[0] != y.shape[0]:
                raise ValueError('x and y have different number of rows')
            return X, y
    
    def _process_predict_data(self, X):
        """
        Validate the X and y data before calling predict
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            warnings.filterwarnings(action="ignore", message="X does not have valid feature names, but .* was fitted with feature names", 
                                    category=UserWarning, module="sklearn.utils.validation")
            return validate_data(self, X, reset=False, dtype=np.float64)

    def fit(self, X, y) -> Self:
        """
        Fits a Decision Tree to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vectot

        Returns:
            ConTree

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        # Validate params and data
        self._validate_params()
        X, y = self._process_fit_data(X, y)
        self.train_X_ = X
        self.train_y_ = y

        self._initialize_config()
      
        self.tree_ = solve(X, y, self._config, self.time_limit)
        
        if not self._config.is_within_time_limit():
            warnings.warn("Fitting exceeds time limit.", stacklevel=2)

        return self

    def predict(self, X):
        """
        Predicts the target variable for the given input feature data.

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix

        Returns:
            numpy.ndarray: A 1D array that represents the predicted target variable of the test data.
                The i-th element in this array corresponds to the predicted target variable for the i-th instance in `X`.
        """
        check_is_fitted(self, "tree_")
        X = self._process_predict_data(X)
        return self.tree_.predict(X)

    def predict_proba(self, X):
        """
        Predicts the probabilities of the target class for the given input feature data.

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix

        Returns:
            numpy.ndarray: A 2D array that represents the predicted class probabilities of the test data.
                The i-j-th element in this array corresponds to the predicted class probablity for the j-th class of the i-th instance in `X`.
        """
        check_is_fitted(self, "tree_")
        X = self._process_predict_data(X)
        probabilities = np.zeros((len(X), self.n_classes_))
        train_data = (self.train_X_, self.train_y_)
        self._recursive_predict_proba(self.tree_, probabilities, np.array(range(0, len(X))), X, train_data)
        # Check that all rows sum to proability 1 (account for floating errors)
        assert (probabilities.sum(axis=1).min() >= 1-1e-4)
        return probabilities
    
    def _recursive_predict_proba(self, tree, probabilities, indices, X, train_data):
        train_X = train_data[0]
        train_y = train_data[1]
        if tree.is_leaf_node():
            n = len(train_y)
            assert(n > 0)
            all_counts = np.zeros(self.n_classes_)
            unique, counts = np.unique(train_y, return_counts=True)
            for label, count in zip(unique, counts):
                all_counts[label] = count
            probs = all_counts / n
            probabilities[indices] = probs
        else:
            indices_left  = np.intersect1d(np.argwhere( (X[:, tree.get_split_feature()] <= tree.get_split_threshold())), indices)
            indices_right = np.intersect1d(np.argwhere(~(X[:, tree.get_split_feature()] <= tree.get_split_threshold())), indices)
            sel = train_X[:, tree.get_split_feature()] <= tree.get_split_threshold()
            train_data_left  = (train_X[ sel, :], train_y[ sel])
            train_data_right = (train_X[~sel, :], train_y[~sel])
            self._recursive_predict_proba(tree.get_left(),  probabilities, indices_left,  X, train_data_left)
            self._recursive_predict_proba(tree.get_right(), probabilities, indices_right, X, train_data_right)

    def score(self, X, y_true) -> float:
        """
        Computes the score for the given input feature data

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix
            y_true : array-like, shape = (n_samples)
            The true labels

        Returns:
            The score
        """
        check_is_fitted(self, "tree_")
        X, y_true = self._process_score_data(X, y_true)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)
    
    def get_average_question_length(self, X) -> float:
        """
        Computes the average number of branching nodes visited until it reaches a leaf node by all samples

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix

        Returns:
            float: The average question length
        """
        return self.get_question_length(X).mean()
    
    def get_question_length(self, X) -> np.ndarray:
        """
        Computes for each sample the number of branching nodes it visits until it reaches a leaf node

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix

        Returns:
            numpy.ndarray of shape (n_samples,):  A 1D array that represents the question length of the data.
                The i-th element in this array corresponds to the number of branching nodes visited by the i-th instance in `X`
        """
        check_is_fitted(self, "tree_")
        X = self._process_predict_data(X)
        X = np.asarray(X, dtype=np.float64)
        result = np.zeros(len(X))
        present = np.ones(len(X), dtype=bool)
        self._recursive_get_question_length(self.tree_, X, present, result)
        return result

    def _recursive_get_question_length(self, tree, X, present, result):
        if tree.is_leaf_node(): return
        result[present] += 1
        left_tree  = tree.get_left()
        right_tree = tree.get_right()
        left_present  = present & (X[:, tree.get_split_feature()] <= tree.get_split_threshold())
        right_present = present & (X[:, tree.get_split_feature()] >  tree.get_split_threshold())
        self._recursive_get_question_length(left_tree,  X, left_present,  result)
        self._recursive_get_question_length(right_tree, X, right_present, result)

    def get_num_branching_nodes(self) -> int:
        """
        Returns the number of branching nodes in the fitted tree
        """
        check_is_fitted(self, "tree_")
        return self.tree_.get_num_branching_nodes()

    def get_num_leaf_nodes(self) -> int:
        """
        Returns the number of leaf nodes in the fitted tree
        """
        check_is_fitted(self, "tree_")
        return self.tree_.get_num_leaf_nodes()

    def get_depth(self) -> int:
        """
        Returns the depth of the fitted tree (a single leaf node is depth zero)
        """
        check_is_fitted(self, "tree_")
        return self.tree_.get_depth()

    def get_tree(self) -> Tree:
        """
        Returns the fitted tree
        """
        check_is_fitted(self, "tree_")
        return self.tree_

    def export_dot(self, filename=None, feature_names=None, label_names=None):
        """
        Write a .dot representation of the tree to filename
        If feature_names is not None, use the names in feature_names for pretty printing
        If label_names is not None, use the names in label_names for pretty printing (only for classification)
        """
        check_is_fitted(self, "tree_")

        if feature_names is None and hasattr(self, "feature_names_in_"):
            feature_names = self.feature_names_in_
        train_data = (self.train_X_, self.train_y_)
        
        exporter = TreeExporter(self.n_classes_)
        if filename is None:
            handle = StringIO()
            exporter.export(handle, self.tree_, feature_names, label_names, train_data)
            return handle.getvalue()
        else:
            with open(filename, "w", encoding="utf-8") as fh:
                exporter.export(fh, self.tree_, feature_names, label_names, train_data)
