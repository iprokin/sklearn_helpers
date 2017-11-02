import numpy as np
import pandas as pd
from sklearn import neighbors


class sklearnDf():
    def __init__(self, sklearn_model):
        self.model = sklearn_model
    def fit(self, dfX, dfY):
        self.model.fit(dfX.values, dfY.values)
        self.ycolumns = dfY.columns
    def predict(self, dfX, **kwa):
        res = self.model.predict(dfX.values, **kwa)
        resdf = pd.DataFrame(res, columns=self.ycolumns, index=dfX.index)
        return resdf
    def getModel(self):
        return self.model
    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.model, name)
        else:
            return getattr(self, name)


class BetterKNN(neighbors.KNeighborsRegressor):
    """
    Allows to compute any function of k nearest neighbors
    not just mean.
    It might be useful if you want to have interval rather than
    point estimate.
    """
    def predict(
            self, X,
            aggfunc=lambda x: np.mean(x, axis=1)
            ):
        _, neigh_ind = self.kneighbors(X)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        y_pred = aggfunc(_y[neigh_ind])

        return y_pred


