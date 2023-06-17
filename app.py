import uvicorn
import joblib
from fastapi import FastAPI
from src.models.train_model import IrisModel, IrisSpecies

# 2. Create app and model objects
app = FastAPI()
model = IrisModel()

@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }

@app.get('/train')
def train_species():
    model.model = model._train_model()
    joblib.dump(model.model, model.model_fname_)
    return 'modelo entrenado'

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

