import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

###################################################
# PROCEDIMENTOS PARA REMOÇÃO DE INSTÂNCIAS INVÁLIDAS E INVIÁVEIS
##################################################

def remove_empty():
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))

    TRAIN = pd.read_csv(ABS_PATH + "/train_.csv")
    TEST = pd.read_csv(ABS_PATH + "/test_.csv")
    BASE = pd.concat([TRAIN,TEST],ignore_index=True)
    cols_to_ignore = ["Pair","Author1","Author2","Year","Class",'CommonKeywords',
                        'CommonKeywords_this',"GeodesicDistance_Weighted",
                        "GeodesicDistance_Unweighted","Bigger_Component_Size",
                        "Smaller_Component_Size","Author1_Clustering","Author2_Clustering",
                        "GraphDistance Weighted","GraphDistance Unweighted"]

    cols_to_keep = list(set(list(BASE.columns)) - set(cols_to_ignore))
    print(cols_to_keep)
    FILTRO = BASE.loc[~(BASE[cols_to_keep]==0).all(axis=1)]
    print("antes:",len(BASE))
    #FILTRO = BASE[mask]
    print("depois",len(FILTRO))
    #cut unused instances
    for year in set(FILTRO["Year"]):
        class0 = FILTRO[(FILTRO["Year"] == year) & (FILTRO["Class"] == 0)]
        class1 = FILTRO[(FILTRO["Year"] == year) & (FILTRO["Class"] == 1)]
        print("Year:", year)
        print("-Class 0:", len(class0),"-" ,round((len(class0)/len(FILTRO))*100,2), "%")
        print("-Class 1:", len(class1),"-" ,round((len(class1)/len(FILTRO))*100,2), "%")

    FILTRO = FILTRO.drop(columns=["GraphDistance Weighted","GraphDistance Unweighted"])
    print(FILTRO)
    #teste = FILTRO[(FILTRO["GeodesicDistance_Unweighted"] != 9999) & (FILTRO["GeodesicDistance_Weighted"] != 9999)]))
    class1 = FILTRO[FILTRO["Class"]==1]
    class1.to_csv(ABS_PATH + "/class1.csv",index=False)
    #FILTRO = FILTRO.replace(9999,)
    FILTRO.to_csv(ABS_PATH + "/base.csv",index=False)
    return (FILTRO)

def remove_more():
    ABS_PATH = os.path.dirname(os.path.abspath(__file__))
    base = pd.read_csv(ABS_PATH + "/base.csv")
    
    print(len(base))
    filtro = base[(base["N_CommonKeywords"]>0) & (base["N_CommonEvents"]>0)]
    print(len(filtro))
    filtro.to_csv(ABS_PATH + "/base2.csv",index=False)

def scale(base):
    base = base.drop(columns=['kwds'])
    base = base.replace(9999,1)
    to_scale = base[base.columns[5:]]
    print(to_scale.columns)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(to_scale)
    df_scaled = pd.DataFrame(scaled,columns = base.columns[5:])
    df_scaled = pd.concat([base[base.columns[:5]],df_scaled], axis=1)
    #print(df_scaled)
    print(df_scaled)
    print(set(df_scaled["Classe"]))
    return df_scaled

   

if __name__ == "__main__":
    pass