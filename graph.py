# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:50:38 2018

@author: Marcelo
"""

import os
import time

import matplotlib as plt
import networkx as nx
import numpy as np
import pandas as pd

######################################
# CRUZAMENTOS INICIAIS PARA GERAR OS GRAFOS
#####################################

abspath = os.path.dirname(os.path.abspath(__file__))

paper = pd.read_csv(abspath + "/tabelas_novas/t_paper.csv", index_col=0)
author = pd.read_csv(abspath + "/tabelas_novas/t_author.csv", index_col=0)
person = pd.read_csv(abspath + "/tabelas_novas/t_person.csv", index_col=0)

df = author.join(paper.set_index('paper_id'), on='paper_id')
df = df[['author_id', 'paper_id', 'paper_year', 'person_id']]

dfp = df.join(person.set_index('person_id'), on='person_id')
dfp = dfp[['author_id', 'paper_id', 'paper_year', 'person_id', 'person_name']]

#######################################
# UTILITARIOS DE GRAFOS
#######################################


def find_name(dfp, person_id):
    return dfp[dfp['person_id'] == person_id]['person_name'].iat[0]


def geraGrafoAno(ano, num_nodes, num_links, anos, mean_degree, median_degree):
    g = nx.Graph()

    print('Gerando para ano:', ano)

    dfAno = df[df['paper_year'] <= int(ano)]
    print(set(dfAno['paper_year']))

    for a in dfAno['person_id']:
        if a not in list(g.nodes()):
            g.add_node(a, p_name=find_name(dfp, a))

    countcolbe = 0
    oldreinf = 0
    vis = []
    for p1 in dfAno['paper_id']:
        colab = list(dfAno[dfAno['paper_id'] == p1]['person_id'])
        if p1 in vis:
            continue
        for i in range(0, len(colab)):
            for j in range(1, len(colab)):
                if colab[j] not in list(g.neighbors(colab[i])):
                    if colab[i] != colab[j]:
                        countcolbe += 1
                        g.add_edge(colab[i], colab[j], weight=1)
                else:
                    if colab[i] != colab[j]:
                        g[colab[i]][colab[j]]['weight'] += 1
                        oldreinf += 1
                        countcolbe += 1
        vis.append(p1)

    anos.append(int(ano))
    num_nodes.append(len(g.nodes()))
    num_links.append(len(g.edges()))
    mean_degree_aux = []
    for i in range(0, len(list(g.degree))):
        mean_degree_aux.append(list(g.degree)[i][1])

    mean_degree.append(np.mean(mean_degree_aux))
    median_degree.append(np.median(mean_degree_aux))

    if sorted(list(g.nodes)) == sorted(list(set(dfAno['person_id']))):
        print("ANO: ", ano, ",NODOS OK")
        print("LEN NODES: ", len(g.nodes))
        print("LEN EDGES: ", len(g.edges))

    else:
        print("ANO: ", ano, ",NODOS FAIL")
        for i in range(0, len(sorted(list(g.nodes)))):
            print("NODE:", sorted(list(g.nodes))[i], "| DF", sorted(
                list(set(dfAno['person_id'])))[i])

    return (g, oldreinf, countcolbe)


def get_degree_dist(g):

    nodes, node_degrees = zip(*list(g.degree))
    degree = pd.DataFrame()
    degree["nodes"] = nodes
    degree["node_degrees"] = node_degrees

    deg_list = [0]*(max(node_degrees) + 1)
    for i in range(0, len(node_degrees)):
        deg_list[node_degrees[i]] += 1

    p_k = [0]*len(deg_list)
    for i in range(0, len(deg_list)):
        p_k[i] = deg_list[i]/len(nodes)

    ddist = pd.DataFrame()
    ddist["P_k"] = p_k
    ddist["degree_count"] = deg_list
    ddist["degree"] = np.arange(len(deg_list))

    return (node_degrees, deg_list, ddist)


def get_small_coef(g):
    avg_cc_g = nx.algorithms.cluster.average_clustering(g)

    sp_list_g = []
    for gr in nx.connected_component_subgraphs(g):
        sp_list_g.append(nx.average_shortest_path_length(gr))
    avg_sp_g = np.average(sp_list_g)

    n = len(g.nodes())
    m = len(g.edges())
    if (n*m) % 2 != 0:
        m = m+1

    r = nx.random_regular_graph(n, m)

    avg_cc_r = nx.algorithms.cluster.average_clustering(r)
    sp_list_r = []
    for g in nx.connected_component_subgraphs(r):
        sp_list_r.append(nx.average_shortest_path_length(r))
    avg_sp_r = np.average(sp_list_r)

    print("C:", avg_cc_g, " Cr:", avg_cc_r)
    print("L:", avg_sp_g, " Lr:", avg_sp_r)
    return ((avg_cc_g/avg_cc_r)/(avg_sp_g/avg_sp_r))


def get_scale(g):
    soma = 0
    for u in g.degree:
        for v in g.degree:
            if g.has_edge(u[0], v[0]):
                soma += u[1]*v[1]
    sG = soma

    w = [d[1] for d in g.degree]
    w = sorted(w, reverse=True)

    maxsoma = 0
    for tries in range(10000):
        gmax = nx.expected_degree_graph(w)
        soma2 = 0
        for u in gmax.degree:
            for v in gmax.degree:
                if gmax.has_edge(u[0], v[0]):
                    soma2 += u[1]*v[1]
        if maxsoma < soma2:
            maxsoma = soma2
    sH = maxsoma
    print("sG:", sG, " sH:", sH)
    return (sG/sH)


def pega_metricas():
    years = ['1998', '1999', '2000', '2001', '2002', '2004', '2006', '2008',
             '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

    results = pd.DataFrame(columns=["Year", "Small", "Scale"])
    for y in years:
        small = (get_small_coef(get_graph(y)))
        scale = (get_scale(get_graph(y)))
        print(small, scale)
        dic = {"Year": y,
               "Small": small,
               "Scale": scale}
        results = results.append(dic, ignore_index=True)
        results.to_csv(os.path.dirname(os.path.abspath(__file__)
                                       ) + "/networkmetrics.csv", index=False)


def get_graph(ano):
    abspath = os.path.dirname(os.path.abspath(__file__))
    g = nx.read_gexf(abspath + '/grafos/gephi/grf'+str(ano)+'.gexf')
    return (g)


if __name__ == '__main__':
    num_nodes = []
    num_links = []
    anos = []
    mean_degree = []
    median_degree = []

    g1998, r, c = geraGrafoAno(
        '1998', num_nodes, num_links, anos, mean_degree, median_degree)
    g1999, r, c = geraGrafoAno(
        '1999', num_nodes, num_links, anos, mean_degree, median_degree)
    g2000, r, c = geraGrafoAno(
        '2000', num_nodes, num_links, anos, mean_degree, median_degree)
    g2001, r, c = geraGrafoAno(
        '2001', num_nodes, num_links, anos, mean_degree, median_degree)
    g2002, r, c = geraGrafoAno(
        '2002', num_nodes, num_links, anos, mean_degree, median_degree)
    g2004, r, c = geraGrafoAno(
        '2004', num_nodes, num_links, anos, mean_degree, median_degree)
    g2006, r, c = geraGrafoAno(
        '2006', num_nodes, num_links, anos, mean_degree, median_degree)
    g2008, r, c = geraGrafoAno(
        '2008', num_nodes, num_links, anos, mean_degree, median_degree)
    g2010, r, c = geraGrafoAno(
        '2010', num_nodes, num_links, anos, mean_degree, median_degree)
    g2011, r, c = geraGrafoAno(
        '2011', num_nodes, num_links, anos, mean_degree, median_degree)
    g2012, r, c = geraGrafoAno(
        '2012', num_nodes, num_links, anos, mean_degree, median_degree)
    g2013, r, c = geraGrafoAno(
        '2013', num_nodes, num_links, anos, mean_degree, median_degree)
    g2014, r, c = geraGrafoAno(
        '2014', num_nodes, num_links, anos, mean_degree, median_degree)
    g2015, r, c = geraGrafoAno(
        '2015', num_nodes, num_links, anos, mean_degree, median_degree)
    g2016, r, c = geraGrafoAno(
        '2016', num_nodes, num_links, anos, mean_degree, median_degree)
    g2017, r, c = geraGrafoAno(
        '2017', num_nodes, num_links, anos, mean_degree, median_degree)

    list1 = []
    list1.append([len(g2017.nodes()), len(g2017.edges()), r])

    dflines = pd.DataFrame({
        'new_nodes': num_nodes,
        'num_links': num_links,
        'mean_degree': mean_degree,
        'median_degree': median_degree
    }, index=anos)

    dflines = pd.DataFrame({
        'new_nodes': num_nodes,
        'num_edges': num_links,
        'mean_degree': mean_degree,
        'median_degree': median_degree
    }, index=anos)

    # dflines.to_csv(r"C:\Users\Marcelo\Desktop\tcc-code\tabelas_novas\g_info.csv",encoding='utf-8')

    print('Salvando Grafos...')

    print('Cytoscape...')
    nx.write_graphml(g1998, abspath + '/grafos/cytoscape/grf1998.xml')
    nx.write_graphml(g1999, abspath + '/grafos/cytoscape/grf1999.xml')
    nx.write_graphml(g2000, abspath + '/grafos/cytoscape/grf2000.xml')
    nx.write_graphml(g2001, abspath + '/grafos/cytoscape/grf2001.xml')
    nx.write_graphml(g2002, abspath + '/grafos/cytoscape/grf2002.xml')
    nx.write_graphml(g2004, abspath + '/grafos/cytoscape/grf2004.xml')
    nx.write_graphml(g2006, abspath + '/grafos/cytoscape/grf2006.xml')
    nx.write_graphml(g2008, abspath + '/grafos/cytoscape/grf2008.xml')
    nx.write_graphml(g2010, abspath + '/grafos/cytoscape/grf2010.xml')
    nx.write_graphml(g2011, abspath + '/grafos/cytoscape/grf2011.xml')
    nx.write_graphml(g2012, abspath + '/grafos/cytoscape/grf2012.xml')
    nx.write_graphml(g2013, abspath + '/grafos/cytoscape/grf2013.xml')
    nx.write_graphml(g2014, abspath + '/grafos/cytoscape/grf2014.xml')
    nx.write_graphml(g2015, abspath + '/grafos/cytoscape/grf2015.xml')
    nx.write_graphml(g2016, abspath + '/grafos/cytoscape/grf2016.xml')
    nx.write_graphml(g2017, abspath + '/grafos/cytoscape/grf2017.xml')
    print('Gephi...')
    nx.write_gexf(g1998, abspath + '/grafos/gephi/grf1998.gexf')
    nx.write_gexf(g1999, abspath + '/grafos/gephi/grf1999.gexf')
    nx.write_gexf(g2000, abspath + '/grafos/gephi/grf2000.gexf')
    nx.write_gexf(g2001, abspath + '/grafos/gephi/grf2001.gexf')
    nx.write_gexf(g2002, abspath + '/grafos/gephi/grf2002.gexf')
    nx.write_gexf(g2004, abspath + '/grafos/gephi/grf2004.gexf')
    nx.write_gexf(g2006, abspath + '/grafos/gephi/grf2006.gexf')
    nx.write_gexf(g2008, abspath + '/grafos/gephi/grf2008.gexf')
    nx.write_gexf(g2010, abspath + '/grafos/gephi/grf2010.gexf')
    nx.write_gexf(g2011, abspath + '/grafos/gephi/grf2011.gexf')
    nx.write_gexf(g2012, abspath + '/grafos/gephi/grf2012.gexf')
    nx.write_gexf(g2013, abspath + '/grafos/gephi/grf2013.gexf')
    nx.write_gexf(g2014, abspath + '/grafos/gephi/grf2014.gexf')
    nx.write_gexf(g2015, abspath + '/grafos/gephi/grf2015.gexf')
    nx.write_gexf(g2016, abspath + '/grafos/gephi/grf2016.gexf')
    nx.write_gexf(g2017, abspath + '/grafos/gephi/grf2017.gexf')
    print('Pronto!')
