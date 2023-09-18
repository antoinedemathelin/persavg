import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge

from methods import central, local, federated
from methods import personalized_errors_cv_alpha, personalized_topk_cv_alpha
from datasets import load_energy, load_ctscan, load_space_ga, load_greenhousegas
from utils import generate_samples, avg_r2_score, avg_rescaled_mse, avg_mse, rescaled_mse

PROB_LAWS = ["log_normal_low_var",
             'log_normal_high_var',
            'gaussian',
            'poisson',
            'uniform',
            'laplacian_low_var',
            'laplacian_high_var',
            'pareto']

if __name__ == "__main__":
    
    dict_ = dict(expe=[], prob_law=[], model=[], seed=[], gamma=[], method=[], r2=[], scaled_mse=[], mse=[])
    
    for expe in ["ctscan", "space_ga", "energy", "greenhousegas"]:
    
        if expe == "energy":
            data_list = load_energy()
            n_nodes = 100
            n_test = 100
            error_func = mean_squared_error
            gamma = 1.
        elif expe == "space_ga":
            data_list = load_space_ga()
            n_nodes = 30
            n_test = 20
            error_func = mean_squared_error
            gamma = 1.
        elif expe == "ctscan":
            data_list = load_ctscan()
            n_nodes = 50
            n_test = 100
            error_func = mean_squared_error
            gamma = 1.
        elif expe == "greenhousegas":
            data_list = load_greenhousegas()
            n_nodes = 100
            n_test = 100
            error_func = mean_squared_error
            gamma = 1.

        for seed in range(50):
            np.random.seed(seed)
        
            for prob_law in PROB_LAWS:

                for base_model in [Ridge(alpha=0.1, fit_intercept=False),
                                   DecisionTreeRegressor(),
                                   MLPRegressor()]:

                    print(seed, base_model, expe, prob_law)

                    indice_test = np.random.choice(n_nodes, int(np.ceil(n_nodes*0.05)), replace=False)

                    X_train, Y_train, X_test, Y_test, n_train, _ = generate_samples(
                        data_list,
                        gamma=gamma,
                        n_nodes=n_nodes,
                        prob_law=prob_law,
                        n_test=n_test,
                        seed=seed)

                    for method in [central, local, federated, personalized_errors_cv_alpha,
                                   personalized_topk_cv_alpha]:

                        if "personalized" in method.__name__:
                            models = method(base_model, X_train, Y_train,
                                            X_test, Y_test, indice_test=indice_test, 
                                            error_func=error_func)
                        else:
                            models = method(base_model, X_train, Y_train)

                        dict_["seed"].append(seed)
                        dict_["model"].append(base_model.__class__.__name__)
                        dict_["expe"].append(expe)
                        dict_["prob_law"].append(prob_law)
                        dict_["gamma"].append(gamma)
                        dict_["method"].append(method.__name__)
                        dict_["r2"].append(avg_r2_score(models, X_test, Y_test))
                        dict_["scaled_mse"].append(avg_rescaled_mse(models, X_test, Y_test))
                        dict_["mse"].append(avg_mse(models, X_test, Y_test))

                    pd.DataFrame(dict_).to_csv("results.csv")