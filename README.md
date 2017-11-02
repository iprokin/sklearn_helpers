# sklearn_helpers
Wrapper around sklearn to make it play nicely with pandas and other helper functions


## Usage example

### Example 1, where sklearn meets pandas and they become friends

```python
from sklearn import linear_model
from sklearn_helpers import sklearnDf

model = linear_model.ElasticNet() # old sklearn model

vmodel = sklearnDf(model) # pandas-friendly version
```

Now you can call:

```python
vmodel.fit(dfX, dfY)
dfYpred = vmodel.predict(dfX)

print(dfYpred.head())
```

Instead of
```python
X = dfX.values
Y = dfY.values
model.fit(X, Y)
Ypred = vmodel.predict(X)
dfYpred = pd.DataFrame(Y, index=dfX.index, columns=dfY.columns)

print(dfYpred.head())
```

### Example 2, where kNN returns not just mean but also std and keeps friendship with pandas


```python
class sklearnDfM(sklearnDf):
    def predict(self, dfX):
        res = self.model.predict(
                dfX.values,
                aggfunc=lambda x:
                        { 'mean': np.mean(x, axis=1)
                        , 'std' : np.std(x, axis=1)
                        })
        resdf = pd.concat(
            map(
                lambda v: pd.DataFrame(
                                v,
                                columns=self.ycolumns,
                                index=dfX.index),
                res.values()),
            keys=res.keys(),
            axis=1)
        return resdf

def MyKNN(**kwa):
    return sklearnDfM(BetterKNN(**kwa))

vmodel = MyKNN(n_neighbors=10, p=1) # it is also pandas friendly
```
