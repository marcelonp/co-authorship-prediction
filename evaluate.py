import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from graph import get_graph

###################################################
# MODULO PARA ALGUNS TESTES DE TEORIAS
###################################################


def get_files(usecols=None):
    if usecols is None:
        train = pd.read_csv("/home/jarriq/LinkPrediction/TCC/train_.csv")
        test = pd.read_csv("/home/jarriq/LinkPrediction/TCC/test_.csv")
    else:
        train = pd.read_csv(
            "/home/jarriq/LinkPrediction/TCC/train_.csv", usecols=usecols)
        test = pd.read_csv(
            "/home/jarriq/LinkPrediction/TCC/test_.csv", usecols=usecols)

    return (train, test)


def get_link_percent(df):
    list_dist_w = []
    list_percent_w = []
    for i in sorted(set(df["GeodesicDistance_Weighted"])):
        print(i)
        # if i != 9999:
        list_dist_w.append(i)
        list_percent_w.append(len(df.loc[(df["GeodesicDistance_Weighted"] == i) & (
            df["Class"] == 1)].values)/len(df))

    list_dist_unw = []
    list_percent_unw = []
    list_cont_pos = []
    for i in sorted(set(df["GeodesicDistance_Unweighted"])):
        # if i != 9999:
        list_dist_unw.append(i)
        list_percent_unw.append(len(df.loc[(df["GeodesicDistance_Unweighted"] == i) & (
            df["Class"] == 1)].values)/len(df))
        list_cont_pos.append(
            len(df.loc[(df["GeodesicDistance_Unweighted"] == i) & (df["Class"] == 1)].values))

    df_w = pd.DataFrame({"dist_w": list_dist_w,
                         "percent": list_percent_w})
    df_unw = pd.DataFrame({"dist_unw": list_dist_unw,
                           "percent": list_percent_unw,
                           "cont": list_cont_pos})

    return (df_w, df_unw)


def get_filter_year_quants(df):
    for i in sorted(set(df["Year"])):
        print("Remover até o ano ", i, ": ", len(df["Year"])-len(df[df['Year'] >= i]["Year"]), " registros removidos (", round(
            100 - (len(df[df['Year'] >= i])/len(df["Year"]))*100, 3), "% do total )")


def get_positive_comp_size(df):
    pos_df = data[data["Class"] == 1].copy()

    list_a1 = []
    list_a2 = []

    for i in range(0, len(pos_df)):
        print(round(i/len(pos_df)*100, 3), "%")
        list_a1.append(len(nx.node_connected_component(
            get_graph(int(pos_df["Year"].iat[i])), str(int(pos_df["Author1"].iat[i])))))
        list_a2.append(len(nx.node_connected_component(
            get_graph(int(pos_df["Year"].iat[i])), str(int(pos_df["Author2"].iat[i])))))

    pos_df["Comp_Size1"] = list_a1
    pos_df["Comp_Size2"] = list_a2

    pos_df.to_csv(
        "/home/jarriq/LinkPrediction/TCC/evaluation/pos_compsizes.csv", index=False, decimal=",")

    return


if __name__ == '__main__':
    #train,test = get_files(usecols=["Class","Year","Author1","Author2","GeodesicDistance_Weighted","GeodesicDistance_Unweighted"])
    train, test = get_files()
    print("TOTAL GERAL (TRAIN):", len(train))
    print("TOTAL GERAL (TEST):", len(test))

    get_filter_year_quants(train)
    get_filter_year_quants(test)
    df_w_train, df_unw_train = get_link_percent(train)
    df_w_test, df_unw_test = get_link_percent(test)

    data = train.append(test, ignore_index=True)
    print("percentz1:", len(
        data[data["GeodesicDistance_Unweighted"] <= 6])/len(data))
    print("percentz2:", len(
        data[data["GeodesicDistance_Unweighted"] == 9999])/len(data))
    get_filter_year_quants(data)

    _, unw = get_link_percent(data)
    '''
    print (df_unw_train)
    ax = sns.lineplot(y = df_unw_train["percent"], x= df_unw_train["dist_unw"])
    #plt.show()
    
    print (df_unw_test)
    ax1 = sns.lineplot(y = df_unw_test["percent"], x= df_unw_test["dist_unw"])
    ax.set_xlim([0,20])
    plt.show()
    '''
    print(unw)
    ax3 = sns.lineplot(y=unw["percent"], x=unw["dist_unw"])
    ax3.set_xlim([0, 20])
    plt.show()

    get_positive_comp_size(data)
