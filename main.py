import pulp
from google.colab import drive
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# This will prompt you to authorize Colab to access your Google Drive
drive.mount('/content/drive')

# Once mounted, your Drive contents will be accessible at /content/drive/MyDrive/
# For example, if your JSON file is in 'MyDrive/data/my_data.json'
test_data_dir = '/content/drive/MyDrive/Tesco_assignment/dec_2025/TestData'

results_data_dir='/content/drive/MyDrive/Tesco_assignment/dec_2025/Results/'
json_file_path = '/content/drive/MyDrive/Tesco_assignment/dec_2025/TestData/I80.json'
jsonfilename=json_file_path.split('/')[-1].split(".")[0]

runs_analysis_lst=[]
for file in os.listdir(test_data_dir):
  if file.endswith('.json'):
    print(f"Processing {file}")
    json_file_path=os.path.join(test_data_dir,file)
    print(json_file_path)
    jsonfilename=json_file_path.split('/')[-1].split(".")[0]
    print(jsonfilename)
    results_reqFormat,cost_runTime=OptUsingMIP(json_file_path,jsonfilename,display_graphs=1)
    cost_runTime['Expt']=jsonfilename
    runs_analysis_lst.append(cost_runTime)

    # try:
    #   results_reqFormat,cost_runTime=OptUsingMIP(json_file_path,jsonfilename,display_graphs=1)
    #   runs_analysis_lst.append(cost_runTime)
    # except:
    #   print(f"Issue while running for {file} ")
    #   continue
    if results_reqFormat is not None:
      results_reqFormat.to_csv(os.path.join(results_data_dir,f'{jsonfilename}_MIP_SS_results.csv'))
all_runs_analysis_df=pd.concat(runs_analysis_lst,axis=0)
all_runs_analysis_df.to_csv(os.path.join(results_data_dir,'all_runs_analysis.csv'))





