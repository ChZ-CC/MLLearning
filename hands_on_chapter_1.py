import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


"""
# Hands on Machine Learing with Scikit-learn and Tenserflow

## Chapter 1. The Machine Learning landscape

[notebooks on Git](https://github.com/ageron/handson-ml/blob/master/01_the_machine_learning_landscape.ipynb)。
"""


np.random.seed(42)
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "fundamentals"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


def load_data(datapath, show=False):
    oecd_bli = pd.read_csv(datapath + "oecd_bli_2019.csv", thousands=',')
    gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                                encoding='latin1', na_values="n/a")

    # Prepare the data
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    if show is True:
        # Visualize the data
        country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
        plt.show()
    return X, y


def train_model(X, y, model='linear'):
    if model == 'linear':
        clf = sklearn.linear_model.LinearRegression()
        clf.fit(X, y)
    elif model == '':
        clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
        clf.fit(X, y)
    else:
        clf = None
    return clf


if __name__ == "__main__":
    datapath = os.path.join('datasets', 'lifesat', '')
    X, y = load_data(datapath, show=True)
    model = train_model(X, y, model='linear')

    X_new = [[22587]]  # Cyprus' GDP per capita
    print(model.predict(X_new)) # outputs [[ 5.96242338]]

