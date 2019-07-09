import ast
import os
import time

import numba
import numpy as np
import pandas as pd
from more_itertools import locate
from numba import jit

abspath = os.path.dirname(os.path.abspath(__file__))

#######################################################
# MODULO COM PARA EXTRACAO DE PALAVRAS CHAVE E EVENTOS
# Nota: altamente confuso devido a minha busca pela otimizaçao em python
########################################################


def setup_keywords(train_or_test):
    """
    Pega os arquivos com dados referentes a keywords e artigos publicados,
    carregando também os arquivos de features.

    """

    if train_or_test not in ["test", "train"]:
        raise ValueError("Erro train_or_test")

    print("Inciando setup...")
    start = time.time()
    dirname = os.path.dirname(__file__)

    filename = os.path.join(dirname, 'tabelas_novas/t_author.csv')
    author = pd.read_csv(filename, usecols=["person_id", "paper_id"])

    filename = os.path.join(dirname, 'tabelas_novas/t_paper_keyword.csv')
    paper_keyword = pd.read_csv(filename, usecols=["paper_id", "keyword"])

    filename = os.path.join(dirname, 'tabelas_novas/t_paper.csv')
    paper = pd.read_csv(filename, usecols=["paper_id", "paper_year"])

    df_kwd = author.set_index("paper_id").join(
        paper_keyword.set_index("paper_id")).join(paper.set_index("paper_id"))
    df_kwd = df_kwd.fillna('a')
    df_kwd = df_kwd[df_kwd['keyword'] != 'a']

    if train_or_test == "train":
        feats = pd.read_csv("train_features.csv", converters={
                            "Pair": ast.literal_eval}, usecols=["Pair", "Year"])
    else:
        feats = pd.read_csv("test_features.csv", converters={
                            "Pair": ast.literal_eval}, usecols=["Pair", "Year"])

    df_kwd.to_csv(abspath + '/df_kwd.csv', index=False)
    print("Setup finalizado em ", time.time()-start)
    return (df_kwd, feats)


def numba_wrapper(pair1, pair2, year, l_person_id, l_paper_year, l_keyword):
    l_common_keywords = []
    l_common_keywords_this = []

    l_len_common_keywords = np.full(len(year), 0)
    l_len_common_keywords_this = np.full(len(year), 0)
    l_len_common_events = np.full(len(year), 0)

    for j in range(0, len(pair1)):
        print(j/len(pair1)*100, "%")

        common_kwds, common_kwds_this, len_common_events = go_fast(
            pair1, pair2, year, l_person_id, l_paper_year, l_keyword, j)
        l_common_keywords_j, cont = getkeywords(common_kwds)
        l_common_keywords_this_j, cont_this = getkeywords(common_kwds_this)
        l_common_keywords.append(l_common_keywords_j)
        l_common_keywords_this.append(l_common_keywords_this_j)
        l_len_common_keywords[j] = cont
        l_len_common_keywords_this[j] = cont_this
        l_len_common_events[j] = len_common_events

    return (l_len_common_events, l_len_common_keywords, l_common_keywords, l_len_common_keywords_this, l_common_keywords_this)


# @jit(nopython=True,parallel=True)
# @jit(parallel=True)
def go_fast(pair1, pair2, year, l_person_id, l_paper_year, l_keyword, j):

    loc_person_id1 = np.array([i for i in range(
        0, len(l_person_id)) if (int(l_person_id[i]) == int(pair1[j]))])
    loc_person_id2 = np.array([i for i in range(
        0, len(l_person_id)) if (int(l_person_id[i]) == int(pair2[j]))])

    loc_paper_year = np.array([i for i in range(
        0, len(l_paper_year)) if (int(l_paper_year[i]) < int(year[j]))])
    loc_paper_year_this = np.array([i for i in range(
        0, len(l_paper_year)) if (int(l_paper_year[i]) == int(year[j]))])

    indexes_1 = list((set(loc_person_id1) & set(loc_paper_year)))
    indexes_2 = list((set(loc_person_id2) & set(loc_paper_year)))

    events_1 = [l_paper_year[i] for i in indexes_1]
    events_2 = [l_paper_year[i] for i in indexes_2]

    len_common_events = len(list((set(events_1) & set(events_2))))

    keywords1 = set([l_keyword[i] for i in indexes_1])
    keywords2 = set([l_keyword[i] for i in indexes_2])
    common_kwds = list((keywords1 & keywords2))

    indexes_1_this = list((set(loc_person_id1) & set(loc_paper_year_this)))
    indexes_2_this = list((set(loc_person_id2) & set(loc_paper_year_this)))

    keywords1_this = set([l_keyword[i] for i in indexes_1_this])
    keywords2_this = set([l_keyword[i] for i in indexes_2_this])
    common_kwds_this = list((keywords1_this & keywords2_this))

    return (common_kwds, common_kwds_this, len_common_events)


def getkeywords(common):
    cont = 0
    c_keywords = ""
    for i in common:
        c_keywords += "/" + i
        cont += 1
    return (c_keywords, cont)


def get_keywords_in_common_and_common_events(df_kwd, feats):
    pair = list(feats["Pair"])
    year = list(feats["Year"])
    pair1 = []
    pair2 = []
    start = time.time()
    print("Comprehending...1")
    for i in range(0, len(feats["Pair"])):
        print(i, "/", len(feats["Pair"]))
        pair1.append(int(pair[i][0]))
        pair2.append(int(pair[i][1]))

    print("Comprehended")
    end = time.time() - start
    print(end, " seconds")

    l_person_id = list(df_kwd["person_id"].copy())
    l_paper_year = list(df_kwd["paper_year"].copy())
    l_keyword = list(df_kwd["keyword"].copy())

    start = time.time()
    l_len_common_events, l_len_common_kwds, l_common_keywords, l_len_common_keywords_this, l_common_keywords_this = numba_wrapper(
        pair1, pair2, year, l_person_id, l_paper_year, l_keyword)
    end = time.time() - start
    print(end, " seconds")

    feats["Author1"] = pair1
    feats["Author2"] = pair2

    feats["CommonKeywords"] = l_common_keywords
    feats["N_CommonKeywords"] = l_len_common_kwds

    feats["CommonKeywords_this"] = l_common_keywords_this
    feats["N_CommonKeywors_this"] = l_len_common_keywords_this

    feats["N_CommonEvents"] = l_len_common_events

    return


def new_ckeywords(df_kwd, dados):

    l_pair1 = list(dados["Author1"])
    l_pair2 = list(dados["Author2"])
    l_year = list(dados["Year"])
    l_kwd = []
    l_ckwd = []
    for i in range(0, len(dados)):
        t1 = time.time()
        print(round((i/len(dados))*100, 3), "%")
        p1_kwds = list(df_kwd[(df_kwd["person_id"] == int(l_pair1[i])) & (
            df_kwd["paper_year"] < int(l_year[i]))]["keyword"])
        p2_kwds = list(df_kwd[(df_kwd["person_id"] == int(l_pair2[i])) & (
            df_kwd["paper_year"] < int(l_year[i]))]["keyword"])

        ckwds = list(set(p1_kwds) & set(p2_kwds))
        print(ckwds)
        l_kwd.append(ckwds)
        l_ckwd.append(len(ckwds))
        t2 = time.time()
        print(((t2-t1)/60)*(len(dados)-i), "minutos faltantes")

    dados["kwds"] = l_kwd
    dados["CommonKeywords"] = l_ckwd
    print("Salvando")
    dados.to_csv("/home/jarriq/Projetos/TCC/base3.csv", index=False)


if __name__ == "__main__":
    df_kwd = pd.read_csv("/home/jarriq/Projetos/TCC/evaluation/df_kwd.csv")
    dados = pd.read_csv("/home/jarriq/Projetos/TCC/base2.csv")
    new_ckeywords(df_kwd, dados)
