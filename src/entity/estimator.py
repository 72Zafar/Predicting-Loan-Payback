import numpy as np
import pandas as pd

class BlendedEnsembleModel:
    def __init__(
        self,
        preprocessing_object,
        lgb_model,
        cat_model,
        xgb_model,
        blend_weights=(0.4, 0.35, 0.25),
        threshold=0.5
    ):
        self.preprocessing_object = preprocessing_object
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.xgb_model = xgb_model
        self.blend_weights = blend_weights
        self.threshold = threshold

    def _preprocess_data(self, X):
        """
        Apply preprocessing to input data.
        Handles both dict-based preprocessing object and sklearn pipeline.
        Assumes X is already preprocessed (numeric, categorical encoded, etc.)
        when using the preprocessed test data from model evaluation.
        """
        if isinstance(X, np.ndarray):
            # Data is already preprocessed (from numpy array)
            return X
        
        if isinstance(self.preprocessing_object, dict):
            # Handle dictionary-based preprocessing
            if isinstance(X, pd.DataFrame):
                X = X.copy()
            else:
                X = pd.DataFrame(X, columns=self.preprocessing_object.get("feature_columns", None))
            
            # Apply scaler if available
            if "scaler" in self.preprocessing_object:
                scaler = self.preprocessing_object["scaler"]
                X = scaler.transform(X)
            
            return X
        else:
            # Handle sklearn pipeline or object with transform method
            return self.preprocessing_object.transform(X)

    def predict_proba(self, X):
        X = self._preprocess_data(X)

        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        cat_pred = self.cat_model.predict_proba(X)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]

        w_lgb, w_cat, w_xgb = self.blend_weights

        blended_pred = (
            w_lgb * lgb_pred +
            w_cat * cat_pred +
            w_xgb * xgb_pred
        )

        return blended_pred

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
