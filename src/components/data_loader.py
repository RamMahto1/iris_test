# data_loader.py
import pandas as pd
from sqlalchemy import create_engine

def load_iris_data(username='root', password='root', host='localhost', database='sales'):
    engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")
    df = pd.read_sql("SELECT * FROM iris1", con=engine)
    return df
