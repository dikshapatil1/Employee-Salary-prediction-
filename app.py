from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("employee_data.csv")
le_edu = LabelEncoder()
le_role = LabelEncoder()

df['Education_Level'] = le_edu.fit_transform(df['Education_Level'])
df['Role'] = le_role.fit_transform(df['Role'])

X = df[['Experience', 'Education_Level', 'Role']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((model, le_edu, le_role), f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        experience = int(request.form['experience'])
        education = request.form['education']
        role = request.form['role']

        with open('model.pkl', 'rb') as f:
            model, le_edu, le_role = pickle.load(f)

        input_data = [[experience, le_edu.transform([education])[0], le_role.transform([role])[0]]]
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
