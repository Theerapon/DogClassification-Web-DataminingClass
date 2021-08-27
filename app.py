
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='../datamind')
model = None
le = LabelEncoder()

def load_model():
    global model
    with open('dogclassification.pkl', 'rb') as dogclassification :
        model = pickle.load(dogclassification)


@app.route('/')
def home():
    return render_template('index.html', prediction_text='{}'.format("ทำแบบสอบถามก่อนนะ"))

@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == "POST":
        data = request.form.to_dict()
        data1 = data['inputChoiceOne']
        data2 = data['inputChoiceTwo']
        data3 = data['inputChoiceThree']
        data4 = data['inputChoiceFour']
        data5 = data['inputChoiceFive']
        data6 = data['inputChoiceSix']
        data7 = data['inputChoiceSeven']
        
        data = np.array([data1, data2, data3, data4, data5, data6, data7])
        le.fit(data.astype(str))
        data = le.transform(data.astype(str))
        data = data.reshape(1, -1)
        prediction = model.predict(data)
        output = prediction[0]
        return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    load_model() #load model a the beginning once only
    app.run(port=80, debug=True)