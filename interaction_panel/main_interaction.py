import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

# example plot using tips
tips = sns.load_dataset("tips")
print("The head of tips dataset:")
print(tips.head())



def example_plot(data=tips, x="tip", y="total_bill", hue="sex",
                 size="size", sizes=(10, 100), as_example=True):
    
    if as_example:
        sns.scatterplot(data=data, x=x, y=y, hue=hue,
                        size=size, sizes=sizes)
        plt.show()
        

