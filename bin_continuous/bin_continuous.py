# Author: Hirotaka Nakagame <hirotaka.nakagame@gmail.com>

from sklearn.preprocessing import KBinsDiscretizer


class bin_continuous:
    """
    Bin continuous features as a binned array

    Parameters:
    ----------
        cols : list type
            Name of features to bin

        drop_original : bool, default=False
            Will drop original columns if set True else keep the original continuous columns

        suffix : string type, default=None type
            Will add suffix to the end of feature names in the returned
            pd.DataFrame if not None else keep the same name

        params : dictionary type, default=None tpye
            Parameters to the K Bin Discretizer(version=0.24.0)
            Refer their website for the parameters details
    """

    def __init__(self, cols, drop_original=False, suffix=None, params=None):
        """
        Constructs all the necessary attributes for the person object.
        """
        self.cols = cols
        self.drop_original = drop_original
        self.suffix = suffix
        self.params = params
        self.transformers = {}

    def __initialize(self):
        """
        initialize the K Bins Discretizer from sklearn.
        Parameters is set by self.params
        """
        for col in self.cols:
            est = KBinsDiscretizer()
            est.set_params(**self.params)
            self.transformers[col] = est

    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to self.cols in X.

        Parameters
        ----------
            X : array-like, shape [n_samples, n_features]
                The data to determine the categories of each feature.
            y : None
                Ignored. This parameter exists only for compatibility with
                :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
            self
        """
        self.__initialize()
        for col in self.cols:
            self.transformers[col].fit(X[col].to_numpy().reshape(-1, 1))

    def transform(self, X):
        """
        transform X using k-bins-discretizer

        Parameters
        ----------
            X : array-like, shape [n_samples, n_features]
                The data to encode.

        Returns
        -------
            X_out : a 2-d array
                Transformed input.
        """
        if self.suffix is None:
            for col in self.cols:
                X[col] = (
                    self.transformers[col]
                    .transform(X[col].to_numpy().reshape(-1, 1))
                    .reshape(1, -1)[0]
                )
        else:
            for col in self.cols:
                X[f"{col}_{self.suffix}"] = (
                    self.transformers[col]
                    .transform(X[col].to_numpy().reshape(-1, 1))
                    .reshape(1, -1)[0]
                )
                if self.drop_original is True:
                    X.drop(col, axis=1, inplace=True)
        return X
