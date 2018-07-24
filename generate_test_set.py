import os
import pandas as pd

# local items
from utils import DataPrep, DataPrepWrapper
from config import data_dir, data_fp, temp_dir


# raw features only
def generate_test_set(is_data_fp, os_data_fp, yr):
    df = pd.read_csv(is_data_fp, low_memory=False)
    os_df = pd.read_csv(os_data_fp, low_memory=False)
    is_data = DataPrep(df)
    data = DataPrepWrapper(is_data, os_df)
    _, _, X_test, _, _, y_test = data.split_and_standardize()
    y_prob = data.pos_prob.values

    X_test.dump(f'{temp_dir}\\X_test_{yr}.pkl')
    y_test.dump(f'{temp_dir}\\y_test_{yr}.pkl')
    y_prob.dump(f'{temp_dir}\\y_prob_{yr}.pkl')


if __name__ == "__main__":
    vintages = range(2001, 2017)

    for v in vintages:
        os_data_fp = os.path.join(data_dir, f'Gse_{v}_ltvgt80_v50.csv')
        generate_test_set(data_fp, os_data_fp, v)
