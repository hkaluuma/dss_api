
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('dss_api')

# Define predict function
@app.post('/predict')
def predict(Fever, MuscleAches, LossOfsmell, PainfulBreathing, ShortnessOfBreath, JointAche, RunnyNose, OtherNeurologicalSigns, RelationshipWithContactPerson, Diabetes, SoreThroat, LossOfTaste, Occupation, Diarrhoea, Vomiting, Nausea, Rash):
    data = pd.DataFrame([[Fever, MuscleAches, LossOfsmell, PainfulBreathing, ShortnessOfBreath, JointAche, RunnyNose, OtherNeurologicalSigns, RelationshipWithContactPerson, Diabetes, SoreThroat, LossOfTaste, Occupation, Diarrhoea, Vomiting, Nausea, Rash]])
    data.columns = ['Fever', 'MuscleAches', 'LossOfsmell', 'PainfulBreathing', 'ShortnessOfBreath', 'JointAche', 'RunnyNose', 'OtherNeurologicalSigns', 'RelationshipWithContactPerson', 'Diabetes', 'SoreThroat', 'LossOfTaste', 'Occupation', 'Diarrhoea', 'Vomiting', 'Nausea', 'Rash']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)