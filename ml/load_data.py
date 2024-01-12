import numpy as np
from random import randint
import os
os.chdir(os.path.split(os.path.realpath(__file__))[0])
BASE_PATH = os.path.abspath("..")


def _load_data(file_name):
    """Read csv file.

    Arguments:
        file_name {str} -- csv file name

    Returns:
        X {ndarray} -- 2d array object with int or float
        y {ndarray} -- 1d array object with int or float
    """

    path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
    data = np.loadtxt(path, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y


def load_breast_cancer():
    """Load breast cancer data for classification.

    Returns:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float
    """

    return _load_data("breast_cancer")


def load_boston_house_prices():
    """Load boston house prices data for regression.

    Returns:
        X {list} -- 2d list object with int or float
        y {list} -- 1d list object with int or float
    """

    return _load_data("boston_house_prices")


def load_tagged_speech():
    """Load tagged speech data for classification.

    Returns:
        X {list} -- 2d list object with str.
        y {list} -- 1d list object with str.
    """

    def data_process(file_name):
        path = os.path.join(BASE_PATH, "dataset", "%s.csv" % file_name)
        f = open(path)
        data = [line[:-1].split("|") for line in f]
        f.close()
        return data

    X = data_process("observations")
    y = data_process("states")

    return X, y


def load_movie_ratings(path):
    """Load movie ratings data for recommedation.

    Returns:
        list -- userId, movieId, rating
    """

    file_name = "ratings"
    path = os.path.join(BASE_PATH, "data", "%s.csv" % file_name)
    f = open(path)
    lines = iter(f)
    col_names = ", ".join(next(lines)[:-1].split(",")[:-1])
    print("The column names are: %s." % col_names)
    data = [[float(x) if i == 2 else int(x)
             for i, x in enumerate(line[:-1].split(",")[:-1])]
            for line in lines]
    f.close()

    return data


def gen_data(low, high, n_rows, n_cols=None):
    """Generate dataset randomly.

    Arguments:
        low {int} -- The minimum value of element generated.
        high {int} -- The maximum value of element generated.
        n_rows {int} -- Number of rows.
        n_cols {int} -- Number of columns.

    Returns:
        list -- 1d or 2d list with int
    """
    if n_cols is None:
        ret = [randint(low, high) for _ in range(n_rows)]
    else:
        ret = [[randint(low, high) for _ in range(n_cols)]
               for _ in range(n_rows)]
    return ret