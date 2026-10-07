import tarfile
import pandas as pd
from urllib.request import urlretrieve
import tempfile

url = 'https://github.com/ageron/data/raw/main/housing.tgz'
file = tempfile.gettempdir() + "/housing.tgz"
string,code = urlretrieve(url, file)

tfile = tarfile.open(file)
tfile.extractall(".");

data = pd.read_csv("./housing.csv")

print(data.head())


