# Authors: Hirotaka Nakagame <hirotaka.nakagame@gmail.com>

import pandas as pd
import numpy as np


class WeightofEvidenceEncoder:
    """
    Calculate weight of evidence for categorical features

    Parameters
    ----------
        feats: list type
            list of features to be transformed
        drop_original: bool, default=False
            Will drop original columns if set True else keep the original columns
        fillna: float type, default=None
            what missing values will be filled with
        prefix: str type, default="woe_"
            prefix to add to the new columns
        suffix: str type, default=""
            suffix to add to the new columns
    """

    def __init__(self, feats, drop_original=False, fillna=None, prefix="woe_", suffix=""):
        self.feats = feats
        self.drop_original = drop_original
        self.fillna = fillna
        self.prefix = prefix
        self.suffix = suffix
        self.transform_dict = {"woe": {}, "iv": {}}
        self.is_fitted = False

    @staticmethod
    def __calc_perc(x, y):
        """
        Calculates % of events and non events

        Parameters
        ----------
            x : pd.Series
                a feature variable
            y : pd.Series
                a target variable

        Returns
        -------
            p_event : pd.Series
                % of events
            p_non_event : pd.Series
                % of non events
        """
        X_temp = pd.DataFrame({"feat": x.values, "target": y.values})
        n_event = y.value_counts()
        p_event = X_temp.groupby("target")["feat"].value_counts()[1] / n_event[1]
        p_non_event = X_temp.groupby("target")["feat"].value_counts()[0] / n_event[0]
        return p_event, p_non_event

    def fit(self, X, y):
        """
        Fit Weight of Evidence Encoder

        Parameters
        ----------
            X : pd.DataFrame
                a
            y : pd.Series
                a
        """
        for feat in self.feats:
            p_event, p_non_event = self.__calc_perc(X[feat], y)
            woe = np.log(p_non_event / p_event)
            woe.fillna(0, inplace=True)
            self.transform_dict["woe"][feat] = woe.to_dict()
            information_value = (p_non_event - p_event) * woe
            if self.fillna is not None:
                information_value.fillna(self.fillna, inplace=True)
            self.transform_dict["iv"][feat] = information_value
        self.is_fitted = True

    def transofrm(self, X):
        """
        Transform X using Weight of Evidence Encoder

        Parameters
        ----------
            X : pd.DataFrame
                The data to encode
        Returns
        -------
            X : pd.DataFrame
                Transformed input
        """
        assert self.is_fitted, "Fot the encoder first"
        for feat in self.feats:
            new_col_name = self.prefix + feat + self.suffix
            X[new_col_name] = X[feat].map(
                self.transform_dict["woe"][feat], na_action=self.fillna
            )
            if (self.drop_original is True) & (self.prefix + self.suffix != ""):
                X.drop([feat], axis=1, inplace=True)
        return X

    @staticmethod
    def __add_description(information_value):
        """
        Convert Information Value to Description.

        Parameters
        ----------
            information_value : float
                Information Value
        Returns
        -------
            desc: str type
                Description of Information Value
        """
        predictive_power = {
            (0, 0.02): "Not useful for prediction",
            (0.02, 0.1): "Weak predictive power",
            (0.1, 0.3): "Medium predictive power",
            (0.3, 0.5): "Strong predictive power",
            (0.5, np.inf): "Too good to be true!",
        }
        for bounds, desc in predictive_power.items():
            if bounds[0] <= information_value < bounds[1]:
                return desc
        return "Too good to be true!"

    def information_values(self, add_description=True):
        """
        The information-value-based feature importances

        The higher, the more important the feature.
        The information value is calculated as
            sigma (% of non events - % of events) * weight of evidence

        Parameters
        ----------
            add_description : bool, default=True
                Will add a column to describe information value if True

        Returns
        -------
            X_iv : pd.DataFrame
                index is feature names and column is information value
        """
        assert self.is_fitted, "Fot the encoder first"
        X_iv = pd.DataFrame(index=self.feats, columns=["Information Value"])
        for feat in self.feats:
            X_iv.loc[feat, "Information Value"] = self.transform_dict["iv"][feat].sum()
        if add_description:
            X_iv["Predictive Power Description"] = X_iv["Information Value"].apply(
                self.__add_description
            )

        return X_iv
