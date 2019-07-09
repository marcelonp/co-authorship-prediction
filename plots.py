import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from yellowbrick.features.importances import FeatureImportances
from yellowbrick.features.radviz import RadViz
from yellowbrick.features.rfecv import RFECV

ABSPATH = os.path.dirname(os.path.abspath(__file__))


def preprocessa():
    # Load the classification data set
    data = pd.read_csv(ABSPATH + "/base.csv")

    notcols = ["kwds"]
    for col in data.columns:
        if col.endswith("p"):
            notcols.append(col)

    data = data[data["Year"] >= 2015]
    for i, date in enumerate(data["Year"]):
        if date == 2017:
            endtrain = i
            break

    cvtrain = list(range(0, endtrain-1))
    cvtest = list(range(endtrain, len(data)))
    cv = [cvtrain, cvtest]

    notcols = notcols + list(data.columns[:5])
    print(notcols)
    features = [col for col in data.columns if col not in notcols]
    print(features)

    X = data[features]
    y = data["Classe"]

    return X, y, cv


def gera_grafico_importancia(X, y):
    visualizer = FeatureImportances(RandomForestClassifier(n_estimators=100))
    # Fit the data to the visualizer
    visualizer.fit(X, y)
    # Transform the data
    visualizer.poof()


def gera_grafico_rfe(X, y, cv):
    print("rfeing....")
    rfe = RFECV(RandomForestClassifier(n_estimators=100, n_jobs=2),
                cv=[cv], scoring="f1")
    rfe.fit(X, y)
    rfe.poof()
    print(rfe.ranking_)
    print(rfe.cv_scores_)


def gera_radviz(X, y):
    visualizer = RadViz(
        classes=["Não Conectado", "Conectado"], features=features, color=("r", "b"))
    print("Poofing...1")
    visualizer.fit(X, y)      # Fit the data to the visualizer
    visualizer.transform(X)   # Transform the data
    visualizer.poof()


def plot_density(base):
    global ABSPATH
    print("Plotting density")
    for col in base.columns[5:]:
        print(col)
        if col not in []:
            p1 = sns.kdeplot(base[base["Classe"] == 0][col], shade=True,
                             color='r', label="Não", bw=0.5).get_figure()
            p1 = sns.kdeplot(base[base["Classe"] == 1][col], shade=True,
                             color='b', label="Sim", bw=0.5).get_figure()
            p1.savefig(ABS_PATH + "/density_plots/density_" +
                       str(col) + ".png")
            p1.clf()


def plot_scatter(base):
    global ABSPATH
    print("sample")
    class1 = base[base["Classe"] == 1]

    base = base.sample(n=100000)
    base = pd.concat([class1, base], axis=0)
    not_list = ["Author1", "Author2", "kwds"]
    base["Classe"] = ["Sim" if c == 1 else "Não" for c in base["Classe"]]
    print("scattering")
    for col1 in base.columns[5:]:
        for col2 in base.columns[5:]:
            if col1 not in not_list and col2 not in not_list:
                print(col1, col2)
                p = sns.scatterplot(data=base, x=col1, y=col2, hue="Classe", hue_order=["Não", "Sim"],
                                    palette=sns.color_palette("Set1", n_colors=2, desat=0.5)).get_figure()
                p.savefig(ABS_PATH + "/scatter_plots_nscale/scatter_" +
                          str(col1) + "_" + str(col2) + ".png")
                p.clf()


def plot_corr(base):
    global ABSPATH
    #base = base.drop(columns=['CommonKeywords','CommonKeywords_this'])
    sns.set(font_scale=0.5)
    other = ["Classe", "KC", "EC", "DC", "MA"]
    unw = []
    for col in base.columns:
        if not col.endswith("p") or col in other:
            unw.append(col)
    w = []
    for col in base.columns:
        if col.endswith("p") or col in other:
            print(col)
            w.append(col)
    sns.set(font_scale=1)
    sns.set_style(style='white')
    print(w, unw)
    plt.figure(figsize=(10, 10))
    corr = base[base.columns[4:]].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax = sns.heatmap(corr, mask=mask, cmap="bwr", center=0,
                     square=True, cbar_kws={"shrink": .9}, annot=True, annot_kws={"size": 7.5}, fmt='.2f')
    sns.set(font_scale=1)
    sns.set_style(style='white')
    ax.figure.tight_layout()
    fig = ax.get_figure()
    fig.savefig(ABS_PATH + "/heatmaps/heatmap.png")
    fig.clf()

    corr = base[['Classe', 'AAp', 'CNp', 'JCp', 'RAp', 'Katzp',
                 'RPRp', 'SRp', 'CMp', 'DC', 'MA', 'KC', 'EC', 'IC']].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax1 = sns.heatmap(corr, mask=mask, center=0, cmap="bwr",
                      square=True, cbar_kws={"shrink": .9}, annot=True, annot_kws={"size": 11}, fmt='.2f')
    ax1.figure.tight_layout()
    fig = ax1.get_figure()
    fig.savefig(ABS_PATH + "/heatmaps/heatmap_un.png")
    fig.clf()

    corr = base[['Classe', 'AA', 'CN', 'JC', 'RA', 'Katz',
                 'RPR', 'SR', 'CM', 'DC', 'MA', 'KC', 'EC', 'IC']].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax2 = sns.heatmap(corr, mask=mask, center=0, cmap="bwr",
                      square=True, cbar_kws={"shrink": .9}, annot=True, annot_kws={"size": 11}, fmt='.2f')
    ax2.figure.tight_layout()
    fig = ax2.get_figure()
    fig.savefig(ABS_PATH + "/heatmaps/heatmap_w.png")
    fig.clf()


def add_margin(ax, x=0.2, y=0.2):
    # This will, by default, add 5% to the x and y margins. You
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin, xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin, ylim[1]+ymargin)


def plot_hist(base):
    print("Plotting density")
    for col in base.columns[5:]:
        print(col)
        if col not in ["kwds"]:
            base2 = base[base[col] != base[col].min()][col]
            print(base2)
            p1 = sns.distplot(base2, color="r", kde=False,
                              hist_kws={"alpha": 1}).get_figure()
            p1 = sns.distplot(base[base["Classe"] == 1][col], color="b", kde=False, hist_kws={
                              "alpha": 1}).get_figure()
            p1.savefig(ABS_PATH + "/hist_plots/hist_" + str(col) + ".png")
            p1.clf()


if __name__ == "__main__":
    pass
