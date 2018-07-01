import os
import pandas as pd
from outlier_detection import *

base_path = 'C:\\Users\\qzhang\\PycharmProjects\\Mei'
data_path = os.path.join(base_path, 'Gse_panel_current_sample_raw.csv')
df = pd.read_csv(data_path)

plot(df['ur'].values)