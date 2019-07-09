import ast
import os
import time
from multiprocessing import Pool

import linkpred as lp
import networkx as nx
import numpy as np
import pandas as pd
import psutil
from joblib import dump, load
from more_itertools import locate
from numba import jit

from graph import get_graph

ABSPATH = os.path.dirname(os.path.abspath(__file__))

#################################################################################################
#SETUP & UTILS
#################################################################################################


def setup_pairs(train_or_test):

    feats = pd.DataFrame()

    pair_list = []
    if train_or_test == 'train':
        years = ['1998', '1999', '2000', '2001',
                 '2002', '2004', '2006', '2008',
                 '2010', '2011', '2012', '2013',
                 '2014', '2015']
    elif train_or_test == 'test':
        years = ['2016', '2017']
    year_list = []

    for ano in years:
        if train_or_test == 'train':
            pair_list += list(nx.non_edges(get_graph(ano)))
            year_list += [ano for i in range(0,
                                             len(list(nx.non_edges(get_graph(ano)))))]
        elif train_or_test == 'test':
            pair_list += list(nx.non_edges(get_graph(2015)))
            year_list += [ano for i in range(0,
                                             len(list(nx.non_edges(get_graph(2015)))))]

    class_list = []
    g2016 = get_graph(2016)
    g2017 = get_graph(2017)

    cont = 0
    for c in pair_list:
        print(cont, '/', len(pair_list))
        if (g2016.has_edge(c[0], c[1]) == True) or (g2017.has_edge(c[0], c[1]) == True):
            class_list.append(1)
        else:
            class_list.append(0)
        cont += 1

    pos_cont = class_list.count(1)
    neg_cont = class_list.count(0)

    print('HAS LINK: ', pos_cont, 'HAS NO LINK: ', neg_cont)

    print(len(pair_list), len(year_list), len(class_list))
    feats['Pair'] = pair_list
    feats['Year'] = year_list
    feats['Class'] = class_list

    return (feats)

#################################################################################################
# FEATURES LINKPRED
#################################################################################################


@jit(nopython=True)
def go_fast(pairs, l_pair, values, l_pred, loc_year):
    for p_index in range(0, len(pairs)):
        print(p_index, "/", len(pairs)-1)
        for ly in loc_year:
            if pairs[p_index] == l_pair[ly]:
                l_pred[ly] = values[p_index]

    return (l_pred)


def get_features(feats, train_or_test, start_from=None):
    global ABSPATH
    predictors = ['AdamicAdar', 'CommonNeighbours',
                  'Jaccard', 'ResourceAllocation',
                  'GraphDistance', 'Katz', 'RootedPageRank', 'SimRank']
    if start_from != None:
        for pred_i in range(0, len(predictors)):
            if start_from in predictors[pred_i]:
                predictors = predictors[pred_i:]
                break

    if train_or_test == 'train':
        years = ['1998', '1999', '2000', '2001',
                 '2002', '2004', '2006', '2008',
                 '2010', '2011', '2012', '2013',
                 '2014', '2015']

    elif train_or_test == 'test':
        years = ['2016', '2017']
    else:
        raise ValueError("train or test not set")

    print(feats.head())

    l_pair_aux = list(feats['Pair'].copy())
    l_pair = [(int(pair[0]), int(pair[1])) for pair in l_pair_aux]
    l_year = list(feats['Year'].apply(int).copy())

    start = time.clock()
    for pred in predictors:
        l_pred = list(np.zeros(len(feats)))

        for year in years:
            print(pred, year)
            lp_pairs = list(eval('lp.predictors.' + pred +
                                 '(get_graph(year)).predict().keys()'))
            pairs = [(int(pair[0]), int(pair[1])) for pair in lp_pairs]
            values = list(eval('lp.predictors.' + pred +
                               '(get_graph(year)).predict().values()'))

            loc_year = list(locate(l_year, lambda x: int(x) == int(year)))
            print("loc_year: ", len(loc_year))
            print("len_pairs: ", len(pairs))

            #print (list(pairs)[0],l_pair[0],values[0],l_pred[0],loc_year)
            l_pred = go_fast(list(pairs), l_pair, values, l_pred, loc_year)

        feats[str(pred)+" Weighted"] = l_pred

        if train_or_test == 'train':
            feats.to_csv(ABSPATH + '/partial_train_features.csv')
        else:
            feats.to_csv(ABSPATH + '/partial_test_features.csv')

        for year in years:
            print(pred, year)
            lp_pairs = list(eval('lp.predictors.' + pred +
                                 '(get_graph(year)).predict(weight=None).keys()'))
            pairs = [(int(pair[0]), int(pair[1])) for pair in lp_pairs]
            values = list(eval('lp.predictors.' + pred +
                               '(get_graph(year)).predict(weight=None).values()'))

            loc_year = list(locate(l_year, lambda x: int(x) == int(year)))
            print("loc_year: ", len(loc_year))
            print("len_pairs: ", len(pairs))

            l_pred = go_fast(list(pairs), l_pair, values, l_pred, loc_year)

        feats[str(pred)+" Unweighted"] = l_pred

        if train_or_test == 'train':
            feats.to_csv(ABSPATH + '/partial_train_features.csv')
        else:
            feats.to_csv(ABSPATH + '/partial_test_features.csv')

    print("Time taken: ", time.clock() - start)

    feats = feats.drop(
        feats.columns[feats.columns.str.contains('unnamed', case=False)], axis=1)

    if train_or_test == 'train':
        feats.to_csv(ABSPATH + '/train_features.csv', index=False)
    else:
        feats.to_csv(ABSPATH + '/test_features.csv', index=False)

    return (feats)


def continue_partial(train_or_test):
    global ABSPATH
    if train_or_test == 'train':
        feats = pd.read_csv(ABSPATH + '/partial_train_features.csv',
                            converters={"Pair": ast.literal_eval}, index_col=False)
    elif train_or_test == 'test':
        feats = pd.read_csv(ABSPATH + '/partial_test_features.csv',
                            converters={"Pair": ast.literal_eval}, index_col=False)
    else:
        raise ValueError("continue partial: train or test errado")

    feats.columns[-1]

    predictors = ['AdamicAdar', 'CommonNeighbours',
                  'Jaccard', 'ResourceAllocation',
                  'GraphDistance', 'Katz', 'RootedPageRank', 'SimRank']
    for pred in predictors:
        if pred in feats.columns[-1]:
            print(pred)
            get_features(feats, train_or_test, pred)


#################################################################################################
# SOME METRICS
#################################################################################################

def get_metrics(feats):
    met = pd.DataFrame()
    l_res = []
    l_class = list(feats["Class"].copy())
    cont = list(feats['Class'].copy()).count(1)
    for i in feats.columns:
        if i not in ["", "Pair", "Class", "Year"]:
            print(i)
            l = list(feats[i].copy())
            print(len(l), len(l_class), cont)
            l_res.append(str("%.2f" % get_metrics_aux(l, l_class, cont))+" %")
    met['Predictor'] = list(feats.columns)[3:]
    met['Result %'] = l_res
    return (met)


def get_metrics_aux(l, c, cont):
    aux = l.copy()
    aux.sort(reverse=True)
    last = aux[cont]
    print("last: ", last)

    cc = 0
    pos_1 = []
    pos = []
    for i in range(0, len(l)):
        #print ("cc: ",cc,i)
        if cc > cont:
            break
        if (c[i] == 1):
            pos_1.append(i)
            if l[i] >= last:
                #print ("hm :",l[i],last)
                cc += 1
                pos.append(i)

    print(len(pos), len(pos_1))
    return (len(pos)/len(pos_1)*100)

#################################################################################################
# GEODESIC DISTANCE
#################################################################################################


def get_geo_distance(df, p_i=None):

    list_geo_weighted = []
    list_geo_unweighted = []

    year = list(df["Year"])
    author1 = list(df["Author1"])
    author2 = list(df["Author2"])
    for i in range(0, len(year)):
        if p_i is None:
            print("Progress:", i/len(year)*100, "%")
        else:
            print("Progress of proc ", p_i, ": ", i/len(year)*100, "%")
        try:

            list_geo_weighted.append(nx.shortest_path_length(get_graph(
                str(year[i])), source=str(author1[i]), target=str(author2[i]), weight='weight'))
        except:
            list_geo_weighted.append(9999)

        try:
            list_geo_unweighted.append(nx.shortest_path_length(get_graph(
                str(year[i])), source=str(author1[i]), target=str(author2[i]), weight=None))
        except:
            #print (e)
            list_geo_unweighted.append(9999)

    df["GeodesicDistance_Weighted"] = list_geo_weighted
    df["GeodesicDistance_Unweighted"] = list_geo_unweighted

    return (df)


def multiprocess_geodist(df):
    n_proc = psutil.cpu_count(logical=False)

    dfs = np.array_split(df, n_proc-1)

    print("N° processors: ", n_proc-1)
    with Pool(n_proc) as p:
        arg_list = [(dfs[i], i) for i in range(0, len(dfs))]
        ret = p.starmap(get_geo_distance, arg_list)

    return (pd.concat(ret))

#################################################################################################
# COMPONENT SIZE
#################################################################################################


def get_component_size(data):
    list_years = pd.unique(data["Year"])
    dic_graphs = {}
    for i in list_years:
        dic_graphs[i] = get_graph(i)

    print(dic_graphs)

    list_bigger = []
    list_smaller = []
    for year, author1, author2 in zip(list(data["Year"]), list(data["Author1"]), list(data["Author2"])):
        print(year)
        g = dic_graphs[year]
        a1 = len(nx.node_connected_component(g, str(author1)))
        a2 = len(nx.node_connected_component(g, str(author2)))
        print(a1, a2)
        if a1 > a2:
            list_bigger.append(a1)
            list_smaller.append(a2)
        else:
            list_bigger.append(a2)
            list_smaller.append(a1)

    data["Bigger_Component_Size"] = list_bigger
    data["Smaller_Component_Size"] = list_smaller

    return data


#################################################################################################
# CLUSTERING
#################################################################################################

def get_clustering(data):
    list_years = pd.unique(data["Year"])
    dic_graphs = {}
    for i in list_years:
        dic_graphs[i] = get_graph(i)

    print(dic_graphs)

    list_clust1 = []
    list_clust2 = []
    for year, author1, author2 in zip(list(data["Year"]), list(data["Author1"]), list(data["Author2"])):
        print(year)
        g = dic_graphs[year]
        list_clust1.append(nx.clustering(g, str(author1)))
        list_clust2.append(nx.clustering(g, str(author2)))

    data["Author1_Clustering"] = list_clust1
    data["Author2_Clustering"] = list_clust2

    return data


if __name__ == '__main__':

    train = pd.read_csv("/home/jarriq/Projetos/TCC/train_.csv")
    train = get_component_size(train)
    train = get_clustering(train)
    train.to_csv("/home/jarriq/Projetos/TCC/train_.csv", index=False)
    del (train)

    test = pd.read_csv("/home/jarriq/Projetos/TCC/test_.csv")
    test = get_component_size(test)
    test = get_clustering(test)
    test.to_csv("/home/jarriq/Projetos/TCC/test_.csv", index=False)
    del (test)
