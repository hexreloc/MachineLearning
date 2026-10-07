import tarfile
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import tempfile
from sklearn import model_selection

def load_housing():
    url = 'https://github.com/ageron/data/raw/main/housing.tgz'
    file = tempfile.gettempdir() + "/housing.tgz"
    _ = urlretrieve(url, file)

    tfile = tarfile.open(file)
    tfile.extractall(".");

    df = pd.read_csv("./housing/housing.csv")
    return df


def income_categories(df):
    # eg; it creates range and assign labels for :
    # 0 < x <= 1.5 ; => 1
    # 1.5 < x <= 3 ; => 2
    # ... so and so
    x: pd.DataFrame = pd.cut(x= df['median_income'], bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf], labels= [1,2,3,4,5]) # converts continious data into categories
    return x.astype(int)

def stratified_split(df, test_size=0.2, random_state=42):
    train_set, test_set = model_selection.train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=income_categories(df)
    )
    return train_set, test_set




df = pd.DataFrame({'median_income': [0.5, 1.5, 2.9, 4.5, 5.99, 6.0, 15.0]})
income_categories(df)
train, test = stratified_split(load_housing())
print("test: ",train)
print("test: ",test)

