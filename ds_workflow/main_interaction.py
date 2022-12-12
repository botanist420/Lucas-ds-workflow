import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import data_wrangling as dw



if __name__ == "__main__":
    input_test = input("Hello Lucas, what do you want to do?\n")
    
    if input_test == "q":
        print('Good bye!')
    
    elif input_test == "check lucas dict":
        print("Loding the json file......")
        data = dw.load_json()
        
        input_key_name = input("Please type in the key you would love to view:\n")
        print(data.get(input_key_name))
    
    else:
        pass

