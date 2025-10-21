from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load artifacts
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
model = pickle.load(open("artifacts/model.pkl", "rb"))
le = pickle.load(open("artifacts/label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read form data
    data = [float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])]
    
    df = pd.DataFrame([data], columns=['sepal_length','sepal_width','petal_length','petal_width'])
    X = preprocessor.transform(df)
    y_pred = model.predict(X)
    species = le.inverse_transform(y_pred.astype(int))[0]

    return render_template('index.html', prediction_text=f"Predicted Species: {species}")

if __name__ == "__main__":
    app.run(debug=True)
