import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error

from methods import central, local, federated
from methods import personalized_errors_cv_alpha, personalized_topk_cv_alpha
from utils import generate_samples, avg_r2_score, avg_rescaled_mse, avg_mse, rescaled_mse
from datasets import synthetic_dataset

PROB_LAWS = ['log_normal_low_var']

if __name__ == "__main__":
    
    dict_ = dict(expe=[], prob_law=[], seed=[], sigma=[], gamma=[], method=[], r2=[], scaled_mse=[], mse=[])

    n_nodes = 100
    n_test = 100
    error_func = mean_squared_error
    base_model = Ridge(0.1, fit_intercept=False)
    expe = "synthetic"
    gamma = 1.
    
    for prob_law in PROB_LAWS:
        for seed in range(50):
            np.random.seed(seed)

            indice_test = np.random.choice(n_nodes, int(np.ceil(n_nodes*0.05)), replace=False)

            for sigma in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                data_list = synthetic_dataset(seed=seed, sigma=sigma)

                X_train, Y_train, X_test, Y_test, n_train, _ = generate_samples(
                        data_list,
                        gamma=gamma,
                        n_nodes=n_nodes,
                        prob_law=prob_law,
                        n_test=n_test,
                        seed=seed)

                for method in [central, local, federated,
                               personalized_errors_cv_alpha]:#

                    if "personalized" in method.__name__:
                        models = method(base_model, X_train, Y_train,
                                        X_test, Y_test, indice_test=indice_test, 
                                        error_func=error_func)
                    else:
                        models = method(base_model, X_train, Y_train)

                    dict_["expe"].append(expe)
                    dict_["sigma"].append(sigma)
                    dict_["prob_law"].append(prob_law)
                    dict_["seed"].append(seed)
                    dict_["gamma"].append(gamma)
                    dict_["method"].append(method.__name__)
                    dict_["r2"].append(avg_r2_score(models, X_test, Y_test))
                    dict_["scaled_mse"].append(avg_rescaled_mse(models, X_test, Y_test))
                    dict_["mse"].append(avg_mse(models, X_test, Y_test))

                pd.DataFrame(dict_).to_csv("results_synthetic.csv")