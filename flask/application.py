from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from pathlib import Path

## bring in the pickle file




application=Flask(__name__)
app=application

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up from /notebook/
SAVE_MODELS_DIR = PROJECT_ROOT / "save_models"

print(f"Root floder:{PROJECT_ROOT}")
print(f"Model directory:{SAVE_MODELS_DIR}")
print(f"Transformer exist:{(SAVE_MODELS_DIR/'feature_transformer.pkl').exists()}")
print(f"Model exist:{(SAVE_MODELS_DIR/'model.pkl').exists()}")

try:
    with open(SAVE_MODELS_DIR/"feature_transformer.pkl","rb")as pfile:
        load_transformer=pickle.load(pfile)

    with open(SAVE_MODELS_DIR/"model.pkl","rb")as pkfile:
        load_model=pickle.load(pkfile)
    
    print ('Model Loaded Successfully')

except FileNotFoundError as e:
    print(f'Error loading file {e}')
    load_transformer=None
    load_model=None


##Home
@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method=='POST':
        temperature=float(request.form.get('temperature'))
        rh=float(request.form.get('rh'))
        ws=float(request.form.get('ws'))
        rain=float(request.form.get('rain'))
        ffmc=float(request.form.get('ffmc'))
        dmc=float(request.form.get('dmc'))
        dc=float(request.form.get('dc'))
        isi=float(request.form.get('isi'))
        bui=float(request.form.get('bui'))
        region=request.form.get('region')

    # Safer: DataFrame with column names
        input_df = pd.DataFrame([{
            "temperature": temperature,
            "rh": rh,
            "ws": ws,
            "rain": rain,
            "ffmc": ffmc,
            "dmc": dmc,
            "dc": dc,
            "isi": isi,
            "bui": bui,
            "region": region}])
        
        transformed_data=load_transformer.transform(input_df)
        result=load_model.predict(transformed_data)
        
        return render_template('pred.html',results=result[0])

    else:
        return render_template("pred.html")

if __name__=="__main__":
    application.run(host="0.0.0.0")