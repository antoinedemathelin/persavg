import os
import numpy as np
import pandas as pd
import urllib
from io import BytesIO, StringIO
from zipfile import ZipFile
import pathlib
import urllib.request


def transform_datetime(datetime):
    date, time = datetime.split(" ")
    return date

def lag_dataset(series, batch_size=15):
    series = series.values
    data = []
    data_diff = []
    labels = []
    time_stamps = []
    length = len(series)
    start = batch_size
    for i in range(length-batch_size):
        batch = series[i:batch_size+i]
        label = series[batch_size+i]
        data.append(batch)
        labels.append(label)
        time_stamps.append(batch_size+i)

    data = np.stack(data, 0)
    labels = np.array(labels)
    time_stamps = np.array(time_stamps)
    return data, labels, time_stamps

def get_list_of_clients_energy(df, lag=15, verbose=0):
    data_list = []
    for c in df.columns[1:]:
        data, labels, time_stamps = lag_dataset(df[c], batch_size=lag)
        start = np.argwhere(labels != 0).ravel()[0] + lag
        end = np.argwhere(labels != 0).ravel()[-1]
        data = data[start:end]
        labels = labels[start:end]
        mean_labels = np.mean(labels)
        data /= mean_labels
        labels /= mean_labels
        data_list.append((data, labels))
        if verbose:
            print(c, data.shape)
    return data_list

def get_list_of_clients_ctscan(df):
    data_list = []
    patients = df.patientId
    for pat in patients.unique():
        X = df.loc[patients==pat].drop(['patientId', 'reference'], axis=1)
        y = (df.loc[patients==pat]['reference']-50)/50.
        data_list.append((X.values, y.values))
    return data_list


def load_energy(path="datasets/LD2011_2014.txt"):
    df = pd.read_csv(path, delimiter=";", decimal=",", index_col=None)
    df["date"] = df["Unnamed: 0"].apply(transform_datetime)
    df = df.drop("Unnamed: 0", axis=1)
    df = df.groupby("date").mean()
    df = df.reset_index()
    data_list = get_list_of_clients_energy(df, lag=15, verbose=0)
    return data_list


def load_ctscan(path="datasets/slice_localization_data.zip"):
    df = pd.read_csv(path)
    data_list = get_list_of_clients_ctscan(df)
    return data_list


def open_greenhousegas(path):
    
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, path)
    path = pathlib.Path(path).as_uri()
    resp = urllib.request.urlopen(path)
    zipfile = ZipFile(BytesIO(resp.read()))

    arr = np.empty((0, 17))

    for file in zipfile.namelist():
        if ".dat" in file:
            ID = file.split("site")[-1].split(".dat")[0]
            ID = [int(ID)]*327
            ghg = [ID]
            count=0
            for line in zipfile.open(file).readlines():
                if count<=15:
                    ghg.append(np.array([float(x) for x in line.decode('utf-8').split(" ")]))
                    count+=1

            arr = np.concatenate((arr, np.stack(ghg, axis=0).transpose()), axis=0)

    df = pd.DataFrame(arr, columns=["loc ID"]+["GHG %i"%i for i in range(15)]+["EDGAR"])
    
    y = df[["EDGAR"]]
    X = df.drop(["EDGAR"], axis=1)
    
    return X, y


def load_greenhousegas(path="datasets/greenhouse+gas+observing+network.zip"):
    X, y = open_greenhousegas(path)
    data_list = []
    for loc in X["loc ID"].unique():
        y_mean = y.loc[X["loc ID"] == loc].values.ravel().mean()
        data_list.append((X.loc[X["loc ID"] == loc].values[:, 1:] / y_mean,
                          y.loc[X["loc ID"] == loc].values.ravel() / y_mean))
    return data_list


def load_space_ga(path="datasets/", boolean=False):
    #print(os.listdir("."))
    with open(path + "space_ga.rtf","r") as f:
        content = f.read()
    county_data = []
    lines = content.split('\n')
    for line in lines:
      entries = [float(l.split('\\')[0]) for l in line.split(' ') if '+' in l or '-' in l]
      if len(entries) > 0:
        county_data.append(entries)
    county_data = np.array(county_data)

    with open(path + "counties.txt","r") as f:
        content = f.read()
    coordinates_counties = []
    lines = content.split('\n')
    for line in lines:
        burst = line.split(' ')
        l = len(burst)
        if l>2 :
            entries = [int(burst[0]), float(burst[l-2]), float(burst[l-1])]
            coordinates_counties.append(entries)
    coordinates_counties = np.array(coordinates_counties)

    N = len(county_data)
    state_of_county_data = np.zeros(N)
    for i in range(N):
        coordinate_i = county_data[i,5]
        for j in range(len(coordinates_counties)):
            if coordinate_i == coordinates_counties[j,2]:
                state_of_county_data[i] = coordinates_counties[j,0]

    number_counties_per_sate = []
    count = 1
    for k in range(N-1):
        if state_of_county_data[k] == state_of_county_data[k+1]:
            count += 1
        else:
            number_counties_per_sate.append(count)
            count = 1
    number_counties_per_sate.append(count)
    M = len(number_counties_per_sate)
    data_list = []
    for i in range(M):
        n_i = number_counties_per_sate[i]
        n = int(np.sum(number_counties_per_sate[:i]))
        if boolean:
            x_i = county_data[n:int(n+n_i), 1:]
        else:
            x_i = county_data[n:int(n+n_i), 1:5]
        y_i = county_data[n:int(n+n_i), 0]
        n += n_i
        
        data_list.append((x_i, y_i))
    return data_list


def synthetic_dataset(nodes=100, dim=30, n=10000, sigma=0., seed=None):
    np.random.seed(seed)
    theta_star = np.random.uniform(size = dim)
    x = []
    y = []
    thetas_star = np.zeros((nodes, dim))
    for i in range(nodes):
        thetas_star[i] = theta_star + np.random.normal(loc = 0, scale = sigma, size = dim)
        x_i = np.random.normal(loc = 0, scale = 1., size = (n, dim))
        x.append(x_i)
        y_i = x_i @ thetas_star[i] + np.random.normal(loc = 0, scale = 0.2, size = n)
        y.append(y_i)
    return [(x[i], y[i]) for i in range(nodes)]