Gradient-Boosting-Regressor implementation in Cython / Numpy

Compile with 'python setup.py build_ext --inplace'

I have added a toy example to benchmark the results with the sklearn GradientBoostingRegressor, feel free to test your own parameters. 

Prior to using this implementation, you should digitize the input matrix, and pass the number of bins as a parameter to the Regressor. 

```python
nlevels = 64
for k in range(X.shape[1]):
    bins = np.arange(np.min(X[:,k]),np.max(X[:,k]),(np.max(X[:,k])-np.min(X[:,k]))/nlevels)
    X[:,k]=np.digitize(X[:,k],bins)
X = X.astype(np.int)

gbm = GBRegTree(niteration=40,alpha=0.2,max_depth=5,min_leaf=2,nlevels=nlevels)
```
