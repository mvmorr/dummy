from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

with open('Models/logistic_model.pkl', 'rb') as f:
    logistic = pickle.load(f)

with open('Models/rf1_model.pkl', 'rb') as f:
    randomforest = pickle.load(f)

with open('Models/svm_clf_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)


def get_predictions(age, sex, chest_pain, rest_blood, fasting_blood_sugar,
    resting_ele, heart_rate, blood, slope, flourosopy, detect, req_model):
    mylist = [age, sex, chest_pain, rest_blood, fasting_blood_sugar,
    resting_ele, heart_rate, blood, slope, flourosopy, detect]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'Logistic':
        #print(req_model)
        return logistic.predict(vals)[0]

    elif req_model == 'RandomForest':
        #print(req_model)
        return randomforest.predict(vals)[0]

    elif req_model == 'SVM':
        #print(req_model)
        return svm_model.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('home.html')

@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    age = request.form['age']
    sex = request.form['sex']
    chest_pain = request.form['chest_pain']
    rest_blood = request.form['rest_blood']
    fasting_blood_sugar = request.form['fasting_blood_sugar']
    resting_ele = request.form['resting_ele']
    heart_rate = request.form['heart_rate']
    blood = request.form['blood']

    slope = request.form['slope']
    flourosopy = request.form['flourosopy']
    detect = request.form['detect']


    req_model = request.form['req_model']

    target = get_predictions(age, sex, chest_pain, rest_blood, fasting_blood_sugar,
    resting_ele, heart_rate, blood, slope, flourosopy, detect, req_model)

    if target==1:
        sale_making = 'Customer is likely to buy the insurance'
    else:
        sale_making = 'Customer is unlikely to buy the insurance'

    return render_template('home.html', target = target, sale_making = sale_making)


if __name__ == "__main__":
    app.run(debug=True)