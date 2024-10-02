import pandas as pd
import numpy as np
import os

def getName(filepath):
    return filepath.split('\\')[-1]


def importdatainfo(path):
    columns=['Center', 'Left','Right', 'Sterring', 'Throttle', 'Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    data['Center']=data['Center'].apply(getName)
    print('Total Images Imported:',data.shape[0])