# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import typing
from typing import List

import json
import re
import os
from datetime import datetime
import inspect

# %%
def get_data_source(pattern_endswith: str=None): # type: ignore    
    data_source_abs_path = '/Users/lucaslee/Desktop/multiverse/data_source/'
    file_name_list = os.listdir(data_source_abs_path)
    
    if not pattern_endswith:
        file_name_list = file_name_list
    else:
        file_name_list = list(filter(lambda x: x.endswith(pattern_endswith), file_name_list))
        
    return file_name_list


# %%
def check_file_name_from_ds(file_name_blurry: str):
    
    all_file_in_data_source = get_data_source()
    result_list = [string for string in all_file_in_data_source if string.startswith(file_name_blurry)]
    
    if not result_list:
        raise Exception(f'There are no match file name that start with{file_name_blurry}')
    
    return result_list

# %%
def ds_read(file_name: str):
    
    data_source_abs_path = '/Users/lucaslee/Desktop/multiverse/data_source/'
    full_file_name = check_file_name_from_ds(file_name_blurry=file_name)[0]
    print(f'Using file name: "{full_file_name}" to transform the data')
    
    result_abs_path = data_source_abs_path + full_file_name
    
    file_type_list = ['.csv', '.xlsx']
    
    if result_abs_path.endswith(file_type_list[0]):
        df_data = pd.read_csv(result_abs_path)
    elif result_abs_path.endswith(file_type_list[1]):
        df_data = pd.read_excel(result_abs_path)
    else:
        raise ValueError('not support data type')
    
    return df_data

# %%
def drop_na_col(data: pd.DataFrame, frac: float=0.9):
    
    na_srs = np.mean(data.isna())
    na_srs = na_srs[na_srs >= frac]
    print('data NA Summary:', '--'*22)
    print(na_srs.sort_values(ascending=False))
    print('--'*22)
    
    col_name_arr = na_srs.index.values
    print(f'droping columns: {col_name_arr}')
    
    new_df = data.drop(columns=col_name_arr)
    
    return new_df

# %%
def ds_str_replace(data: pd.DataFrame, col_list: list, trim_special_char: str=','):
    
    result = data[col_list].apply(lambda x: x.str.replace(trim_special_char, ''), axis=1)
    
    return result

# %%
def get_unique(data: pd.DataFrame, col_list: list):
    
    my_dict = {}
    for i in col_list:
        my_dict[i] = list(data[i].unique())
    
    return my_dict


# %%
def re_find(pattern: str, text_list: list):
    
    return [string for string in text_list if re.findall(pattern, string)]
# %%
def replace_comma(string_text: str):
    
    return re.sub(r',', '', string_text)


# %%
def ctab_3_col(data, col_list: list) -> pd.DataFrame:
    """cross table for 3 variables, the second and third columns will use to be the new column variable

    Args:
        data (pd.DataFrame): 
        col_list (list): should be len() == 3

    Returns:
        pd.DataFrame: return the cross table
    """
    return pd.crosstab(data[col_list[0]], [data[col_list[1]], data[col_list[2]]])

# %%
def ctab(data, col_list: list) -> pd.DataFrame:
    """cross table for 2 columns variables

    Args:
        data (pd.DataFrame): 
        col_list (list): should be len() == 2

    Returns:
        pd.DataFrame: return the cross table
    """
    assert len(col_list) == 2
    
    return pd.crosstab(data[col_list[0]], data[col_list[1]])


# %%
def get_top_n_filter(data, id_col:str, values_col:str, n: int=3):
    
    top_n_col = data.groupby(id_col)[values_col].sum().nlargest(n)
    print(f'Top {n} sum of {values_col!r} from {id_col!r} column:\n')
    print(top_n_col)
    print('--'*22)
    top_n_col = top_n_col.index.values
    print(f'Filtering data by the column: {id_col} values...{top_n_col}')
    return data[data[id_col].isin(top_n_col)]

# %%
def case_when(data_srs: pd.Series, cond_val_list: list, choice_val_list: list) -> np.ndarray:
    
    
    assert len(cond_val_list) == len(choice_val_list)
    
    condition_init = []
    for i in range(len(cond_val_list)):
        result = data_srs == cond_val_list[i]
        condition_init.append(result)   
    
    data_result = np.select(condition_init, choicelist=choice_val_list)
    
    return data_result

# %%
def simple_filter_obj(data: pd.DataFrame, col_str: str, val_str: str, exp_str: str='=='):
    
    
    if data[col_str].dtype == 'object':
        query_text = f"""
        {col_str} {exp_str} '{val_str}'
        """
        print('the query show as:\n', query_text)
        
        return data.query(query_text)
    else:
        raise TypeError(f"the column: '{col_str}' dtype is not object")
# %%
def deal_date(data: pd.Series, date_format: str):
    
    result_series = pd.to_datetime(data, format=date_format)
    result_date = result_series.dt.date
    result_hour = result_series.dt.hour
    result_dayname = [i[:3] for i in result_series.dt.day_name()]
    
    return result_series, result_date, result_hour, result_dayname
# %%
def load_json(file_path: str="lucas_dict.json", mode: str="r") -> dict:
    
    assert file_path in os.listdir()
    
    with open(file_path, mode=mode, encoding="utf-8") as file:
        data = json.load(file)
    
    return data
# %%
def dump_json(key_name: str, new_word_list: list):
    
    file_path = "lucas_dict.json"
    with open(file_path, mode="r", encoding="utf-8") as file:
        data = json.load(file)
    
    assert key_name in data.keys(), f"Your input key name: {key_name!r} is not in dictionary keys"
    
    result_list = data.get(key_name)
    if not set(new_word_list).intersection(set(result_list)):
        print(f"Using the key: {key_name}")
        print(f"These words are ready to be appended into dictionary: {new_word_list}")
        data[key_name] = result_list + new_word_list
    else:
        raise ValueError('The words you provided are already in the dictionary')
    
    print('Dumping the dictionary object into json file......')
    with open(file_path, mode="w", encoding="utf-8") as file:
        json.dump(data, file)

    print('The task has done!!!')
# %%
def dump_new_key_to_json(key_name: str, new_values_list: list, path: str="lucas_dict.json"):
    
    data = load_json(file_path=path)
    
    assert key_name not in data.keys(), f"Your input key name: {key_name!r} is already in dictionary keys"
    
    data[key_name] = new_values_list
    print(f"New key name: {key_name!r}\nValues: {new_values_list}")
    print("New data is ready to dump as below" + "..."*10)
    for key, value in data.items():
        print('\t' + key + ':', value)
    print("..."*22)
    
    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(data, file)
    
    print('The task is Done!!!')
# %%
def norm_update_json(path: str="lucas_dict.json"):
    
    
    data = load_json(path)
    target_dict = data.get("date_count_vocabs")
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    overwrite_value = len(data["new_english_vocabs"])
    
    if today_str in target_dict['date']:
        print("Today already has record in the dictionary, prepare to overwrite......")
        for i, date_value in enumerate(target_dict['date']):
            if date_value == today_str:
                original_value = target_dict["vocab_count"][i]
                print(f"Original value: {original_value} is gonna update as {overwrite_value}")
                data["date_count_vocabs"]["vocab_count"][i] = overwrite_value
    else:
        print("Today doesn't have any record in the dictionary... prepare to append new data......")
        print(f"new data to insert:\ndate: {today_str}, count: {overwrite_value}")
        data["date_count_vocabs"]["date"].append(today_str)
        data["date_count_vocabs"]["vocab_count"].append(overwrite_value)

    
    # eventually dump the dict into json file:
    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(data, file)
    
    print("The task is done!!!")
# %%
def inspect_func_args(fun):
    result = str(inspect.signature(fun))
    result = re.sub(r"(\(|\))", "", result)
    result = re.sub(r"\s", "", result)
    result_list = result.split(',')
    for i in result_list:
        print(i)
    
# %%
def highlight_hue_col(data_srs: pd.Series, emphasize_value_list: List[str]):
    col_unique = np.unique(data_srs)

    for i in emphasize_value_list:
        assert i in col_unique, f"the value: {i!r} is not in this column"

    result_list = [str(string) if string in emphasize_value_list else 'Others' for string in data_srs]
    
    return result_list

# %%
# %%
# %%
# %%
# %%
