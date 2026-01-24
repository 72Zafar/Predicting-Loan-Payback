import numpy as np

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

    def predict_proba(self, X):
        X = self.preprocessing_object.transform(X)

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
