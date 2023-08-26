import pandas as pd

def load_file_list(file_name:str) -> pd.DataFrame:

    df = pd.read_csv(file_name, header=None)

    # labels and file names
    df1 = df.iloc[:,0].apply(lambda x: pd.Series(x.split('/', 1)))

    # file names
    df2 = df1.iloc[:, 1].str[:-4]

    return pd.concat([df1.drop(columns=1), df2], axis=1)

def get_train_test_data():

    df_train = load_file_list('.\data\Anomaly_Train.txt')

    df_test = load_file_list('.\data\Anomaly_Test.txt')

    return df_train, df_test