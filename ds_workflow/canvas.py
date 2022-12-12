import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA



# %%
def ds_matrix(data, figsize=(12, 10), alpha=0.5):
    
    g = pd.plotting.scatter_matrix(data.select_dtypes('number'),
                               figsize=figsize,
                               alpha=alpha)
    
    plt.show()

# %%
def highlight_data(cond_list: list, choice_list: list=['orangered', 'lightgray']) -> np.ndarray:
    
    if len(cond_list) == 1:
        cond_list.append(~(cond_list[0]))
    
    assert len(cond_list) == len(choice_list)

    return np.select(condlist=cond_list, choicelist=choice_list)


# %%
def highlight_max(data: pd.DataFrame, xy_col_list: list, max_col_name: str):
    
    max_value = np.max(data[max_col_name].values)
    data['point_type'] = [f'Hightest {max_col_name} point' if i == max_value else 'Others' for i in data[max_col_name].values]

    g = sns.scatterplot(x=xy_col_list[0], y=xy_col_list[1], data=data,
                        hue='point_type', alpha=0.8)
    plt.show()

# %%
def highlight_cat(data_srs: pd.Series, val_in_cat_list: list):

    return [str(i) if i in val_in_cat_list else 'Others' for i in data_srs]

# %%

# %%
def viz_pca(X_data, components=3):
    
    pca = PCA(n_components=components)
    pca.fit(X_data)
    features = range(pca.n_components_)  # type: ignore
    plt.bar(features, pca.explained_variance_,)  # type: ignore
    plt.xlabel('PCA features')
    plt.ylabel('Variance')
    plt.xticks(features)
    
    plt.show()

# %%
def viz_annot(data, xy_col_list: list, query_text: str, text: str='Testing Data'):
    
    
    target_data = data.query(query_text)
    sns.scatterplot(x=xy_col_list[0], y=xy_col_list[1], data=data,
                    color='steelblue')

    plt.annotate(text, 
                xy=(target_data[xy_col_list[0]], target_data[xy_col_list[1]]), 
                xytext=(target_data[xy_col_list[0]], target_data[xy_col_list[1]]+0.5),
                arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03},
                backgroundcolor = 'white')
# %%
def standard_cat(data: pd.DataFrame, xy_col_list: list, kind: str='box', facet_col: str=None, hue_col: str=None): # type: ignore    
    sns.catplot(x=xy_col_list[0], y=xy_col_list[1], data=data,
                kind=kind, col=facet_col, hue=hue_col)
    plt.show()
# %%
# %%
# %%
# %%
# %%
# %%
