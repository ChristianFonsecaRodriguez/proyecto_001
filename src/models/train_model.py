import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib

class IrisSpecies(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float


class IrisModel:
    def __init__(self):
        self.df = pd.read_csv('./data/processed/iris.csv')
        self.model_fname_ = './models/iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            print('El modelo debe ser entrenado')
            #self.model = self._train_model()
            #joblib.dump(self.model, self.model_fname_)
        
    def _train_model(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species']
        rfc = RandomForestClassifier()
        model = rfc.fit(X, y)
        return model

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability
