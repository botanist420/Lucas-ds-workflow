import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import json
import os


lucas_dict = {
    "common_code_words": ["pandas", "numpy", "groupby()", "sns.scatterplot()", "sns.displot()", 
                          "describe()", "my_dict.items()", "read_csv()", "pd.to_datetime()",
                          "value_counts()", "sort_values(ascending=False)"]
}

# Write data into json file
with open("test.json", mode="w", encoding="utf-8") as file:
    json.dump(lucas_dict, file)


# Read from json file
with open("test.json", mode="r", encoding="utf-8") as file:
    data = json.load(file)

my_list = data.get("common_code_words")
for i in my_list:
    print(i)
