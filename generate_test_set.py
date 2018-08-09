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

    X_test_ns = data.os_data.X.values
    X_test_ns.dump(f'{temp_dir}\\delyrs\\X_test_ns_delyrs_{yr}.pkl')

    _, _, X_test, _, _, y_test = data.split_and_standardize()
    y_prob = data.pos_prob.values

    X_test.dump(f'{temp_dir}\\delyrs\\X_test_delyrs_{yr}.pkl')
    y_test.dump(f'{temp_dir}\\delyrs\\y_test_delyrs_{yr}.pkl')
    y_prob.dump(f'{temp_dir}\\delyrs\\y_prob_delyrs_{yr}.pkl')


if __name__ == "__main__":
    vintages = range(2001, 2017)

    for v in vintages:
        os_data_fp = os.path.join(data_dir, f'Gse_{v}_ltvgt80_v50.csv')
        generate_test_set(data_fp, os_data_fp, v)

    # for v in vintages:
    #     os_data_fp = os.path.join(data_dir, f'Gse_{v}_ltvgt80_v50.csv')
    #     is_df = pd.read_csv(data_fp, low_memory=False)
    #     os_df = pd.read_csv(os_data_fp, low_memory=False)
    #     is_data = DataPrep(is_df)
    #     data = DataPrepWrapper(is_data, os_df)
    #     X_test_ns = data.os_data.X.values
    #
    #     X_test_ns.dump(f'{temp_dir}\\delyrs\\X_test_ns_delyrs_{v}.pkl')
