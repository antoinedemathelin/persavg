import numpy as np
from scipy.stats import lognorm, poisson, uniform, norm, randint, weibull_min, laplace, pareto
from sklearn.metrics import r2_score, mean_squared_error


EPS = np.finfo(float).eps

def get_n_samples_nodes(prob_law, n_nodes, gamma, seed):
    '''
    Returns the number of samples per node from a given probability law with mean n_nodes**gamma.
    Args:
      prob_law (str): the probability distribution of the number of samples per node. Supported: 'gaussian_low_var', 'gaussian_med_var',
      'gaussian_high_var', 'log_normal_low_var', 'log_normal_high_var', 'poisson', 'uniform_low_var',
      'uniform_high_var', 'laplacian_low_var', 'laplacian_med_var', 'laplacian_high_var', 'pareto', 'weibull'.
      Raises NotImplementedError if the probability law is not supported.
      n_nodes (int): the total number of nodes
      gamma (float): the parameter to set the mean number of samples per node (which is equal to n_nodes**gamma)
    Returns:
      an array of length n_nodes containing the number of samples for each node.
    '''
    if prob_law == 'gaussian':
        n_samples_nodes = norm.rvs(n_nodes**gamma, n_nodes**gamma, size=n_nodes, random_state=seed)
    elif prob_law == 'log_normal_low_var':
        n_samples_nodes = lognorm.rvs(1, scale = (n_nodes**gamma)*np.exp(-1/2), size = n_nodes, random_state=seed)
    elif prob_law == 'log_normal_high_var':
        n_samples_nodes = lognorm.rvs(2, scale = (n_nodes**gamma)*np.exp(-2), size = n_nodes, random_state=seed)
    elif prob_law == 'poisson':
        n_samples_nodes = poisson.rvs(n_nodes**gamma, size=n_nodes, random_state=seed)
    elif prob_law == 'uniform':
        n_samples_nodes = uniform.rvs(0, 2*n_nodes**gamma, size=n_nodes, random_state=seed)
    elif prob_law == 'laplacian_low_var':
        n_samples_nodes = laplace.rvs(loc=n_nodes**gamma, scale=0.1*n_nodes**gamma, size=n_nodes, random_state=seed)
    elif prob_law == 'laplacian_high_var':
        n_samples_nodes = laplace.rvs(loc=n_nodes**gamma, scale=n_nodes**gamma, size=n_nodes, random_state=seed)
    elif prob_law == 'pareto':
        n_samples_nodes = pareto.rvs(b=2, scale=n_nodes**gamma/2, size=n_nodes, random_state=seed)
    else:
        raise NotImplementedError
    for i in range(n_nodes):
        #to ensure that the sample size is at least 1
        n_samples_nodes[i] = np.maximum(2, n_samples_nodes[i])
    return n_samples_nodes.astype(int)


def generate_samples(data_list, gamma, n_nodes, prob_law, n_test, seed, sample_test="end"):
    np.random.seed(seed)
    sample_sizes = [float(data[1].shape[0]) for data in data_list]
    ord_ind_sample_sizes = np.argsort(sample_sizes).ravel()
    choosen_nodes = ord_ind_sample_sizes[-n_nodes:]
    X_train, Y_train, X_test, Y_test = [], [], [], []
    train_sample_sizes = get_n_samples_nodes(prob_law, n_nodes, gamma, seed)
    test_sample_sizes = np.ones(len(choosen_nodes)) * float(n_test)
    for node_id in range(n_nodes):
        i = choosen_nodes[node_id]
        ni_total = len(data_list[i][1])
        ni = train_sample_sizes[node_id]
        if sample_test == "end":
            x_test = data_list[i][0][-n_test:]
            y_test = data_list[i][1][-n_test:]
            indice_train = range(ni_total-n_test)
        elif sample_test == "uniform":
            indice_test = np.random.choice(range(ni_total), size = n_test, replace = False)
            x_test = data_list[i][0][indice_test]
            y_test = data_list[i][1][indice_test]
            indice_train = np.array(list(set(np.arange(ni_total)) - set(indice_test)))
        size = min(ni_total-n_test, ni)
        choosen = np.random.choice(indice_train, size = size, replace = False)
        x_train = data_list[i][0][choosen]
        y_train = data_list[i][1][choosen]
        X_train.append(x_train)
        Y_train.append(y_train)
        X_test.append(x_test)
        Y_test.append(y_test)
    return X_train, Y_train, X_test, Y_test, train_sample_sizes, test_sample_sizes


def avg_r2_score(models, X, Y):
    models = list(models); X = list(X); Y = list(Y)
    Yp = [mod.predict(x) for mod, x in zip(models, X)]
    scores = [r2_score(y, yp) for y, yp in zip(Y, Yp)]
    return np.mean(scores)


def avg_rescaled_mse(models, X, Y):
    models = list(models); X = list(X); Y= list(Y)
    Yp = [mod.predict(x) for mod, x in zip(models, X)]
    scores = [mean_squared_error(y, yp) / ((y**2).mean() + EPS) for y, yp in zip(Y, Yp)]
    return np.mean(scores)


def avg_mse(models, X, Y):
    models = list(models); X = list(X); Y= list(Y)
    Yp = [mod.predict(x) for mod, x in zip(models, X)]
    scores = [mean_squared_error(y, yp) for y, yp in zip(Y, Yp)]
    return np.mean(scores)


def rescaled_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) / ((y_true**2).mean() + EPS)


def sigmoid(x):
    return 1/(1 + np.exp(-3*x))

