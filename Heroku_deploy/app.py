import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model

app = Flask(__name__)
model = load_model("final_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    input_data = list(input_data.values())
    input_data = list(map(int, input_data))
    input_array = np.array(input_data).reshape(1, 1, 15) 
    
    prediction = model.predict(input_array)[0][0]

    return render_template('index.html', prediction_text='Prediction = {}'.format(prediction))


if __name__ == "__main__":
    app.debug = True
    app.run()