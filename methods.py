import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_distances

EPS = np.finfo(float).eps


class WeightedBagging:
    
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights / weights.mean()
        
    def predict(self, x):
        predictions = [w * m.predict(x) for m, w in
                       zip(self.models, self.weights)]
        return np.stack(predictions, axis=-1).mean(axis=-1)
    

def central(base_model, X, Y):
    model = clone(base_model)
    x = np.concatenate(X)
    y = np.concatenate(Y)
    model.fit(x, y)
    return [model for _ in Y]


def local(base_model, X, Y):
    models = [clone(base_model) for _ in Y]
    for i in range(len(Y)):
        models[i].fit(X[i], Y[i])
    return models


def federated(base_model, X, Y):
    models = local(base_model, X, Y)
    n_data = np.array([float(y.shape[0]) for y in Y]).reshape(-1, 1)
    n_data /= n_data.mean()
    return [WeightedBagging(models, n_data) for _ in Y]


def water_filing(a, b):
    b[b<=0] += EPS
    args = np.argsort(a).ravel()
    deargs = np.argsort(args).ravel()
    a = a[args]
    b = b[args]
    inv_b = 1./(2. * b)
    diff_a = np.diff(a)
    cum_inv_b = np.cumsum(inv_b)
    water_volume = np.cumsum(diff_a * cum_inv_b[:-1])
    water_volume = np.concatenate((np.zeros((1,)), water_volume))
    indice = np.searchsorted(water_volume, 1., side="right")
    assert indice > 0
    nu = (1 - water_volume[indice-1])/cum_inv_b[indice-1] + a[indice-1]
    sol = np.clip((nu-a)*inv_b, 0., np.inf)
    return sol[deargs]


def optimal_weights(matrix_bias, matrix_var, alpha=1.):
    w = np.zeros(matrix_bias.shape)
    for i in range(matrix_bias.shape[0]):
        w[i] = water_filing(matrix_bias[i], alpha * matrix_var[i])
    return w


def personalized_errors(base_model, X, Y, error_func=None, alpha=1.):
    if error_func is None:
        error_func = mean_squared_error
    
    models = local(base_model, X, Y)
    
    n_data = np.array([float(y.shape[0]) for y in Y])
    
    inv_n_data = 1 / n_data
    
    M = n_data.shape[0]
    errors = np.zeros((M, M))
    for i in range(M):
        errors[i] = np.array([error_func(Y[i], mod.predict(X[i])) for mod in models])
    
    traces = np.array([np.trace(x.T @ x) for x in X])
    variances = (traces / n_data).reshape(-1, 1).dot(inv_n_data.reshape(1, -1))
    
    weights = optimal_weights(errors, variances, alpha=alpha)

    return [WeightedBagging(models, w) for w in weights]


def personalized_errors_cv_alpha(base_model, X, Y, X_test, Y_test, indice_test, error_func=None,
                                 alphas=[10**(5-i) for i in range(10)], verbose=1):
    if error_func is None:
        error_func = mean_squared_error
    
    models = local(base_model, X, Y)
    
    n_data = np.array([float(y.shape[0]) for y in Y])
    
    traces = np.array([np.trace(x.T @ x) for x in X])
    
    inv_n_data = 1 / n_data
    
    M = n_data.shape[0]
    errors = np.zeros((M, M))
    for i in range(M):
        errors[i] = np.array([error_func(Y[i], mod.predict(X[i])) for mod in models])
        
    scores = []
    for alpha in alphas:
        variances = (traces[indice_test] / n_data[indice_test]).reshape(-1, 1).dot(inv_n_data.reshape(1, -1))
        
        weights = optimal_weights(errors[indice_test], variances, alpha=alpha)
        
        error = 0.
        for i in range(len(indice_test)):
            model = WeightedBagging(models, weights[i])
            error += error_func(Y_test[indice_test[i]], model.predict(X_test[indice_test[i]]))

        scores.append(error/len(indice_test))

        if verbose:
            print("Alpha:", alpha, "Score:", scores[-1])
    
    best_alpha = alphas[np.argmin(scores)]
    
    variances = (traces / n_data).reshape(-1, 1).dot(inv_n_data.reshape(1, -1))
    
    weights = optimal_weights(errors, variances, alpha=best_alpha)

    return [WeightedBagging(models, w) for w in weights]



def personalized_topk_cv_alpha(base_model, X, Y, X_test, Y_test, indice_test, distance=None,
                               error_func=None, ks=[1, 10, 50, 100], verbose=1):
    if error_func is None:
        error_func = mean_squared_error
    
    if distance is None:
        distance = cosine_distances
    
    models = local(base_model, X, Y)
    
    n_data = np.array([float(y.shape[0]) for y in Y])
    
    inv_n_data = 1 / n_data
    
    M = n_data.shape[0]
        
    error_test = np.zeros((len(indice_test), n_data.shape[0]))
    for i in range(len(indice_test)):
        error_test[i] = np.array([error_func(Y_test[indice_test[i]],
                       mod.predict(X_test[indice_test[i]])) for mod in models])
    argsort = np.argsort(error_test.mean(0)).ravel()
    
    scores = []
    for k in ks:
        weights = np.zeros((len(indice_test), n_data.shape[0]))
        weights[:, argsort[:k]] = 1.
        weights /= weights.sum(1, keepdims=True)

        error = 0.
        for i in range(len(indice_test)):
            model = WeightedBagging(models, weights[i])
            error += error_func(Y_test[indice_test[i]], model.predict(X_test[indice_test[i]]))

        scores.append(error/len(indice_test))

        if verbose:
            print("K:", k, "Score:", scores[-1])
    
    best_k = ks[np.argmin(scores)]
    
    weights = np.zeros((M, M))
    weights[:, argsort[:best_k]] = 1.
    weights /= weights.sum(1, keepdims=True)

    return [WeightedBagging(models, w) for w in weights]