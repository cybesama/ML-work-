from flask import Flask,request,jsonify,render_template

application=Flask(__name__)
app=application
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

#stuff
linear_regression=pickle.load(open('linear_regression/implementation/models/regression.pk2','rb'))
standard_scaler=pickle.load(open('linear_regression/implementation/models/scaler.pk1','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        weight=request.form.get('Weight')
        scaled_data=standard_scaler.transform([[Weight]])
        result=linear_regression.predict(scaled_data)
        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5001)