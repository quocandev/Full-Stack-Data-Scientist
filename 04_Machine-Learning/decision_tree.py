import sklearn.tree as tree
import pandas as pd

data = {
    'Time_of_Day': ['Morning', 'Morning', 'Morning', 'Morning', 'Afternoon', 'Afternoon', 'Afternoon', 'Afternoon'],
    'Tired': ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No'],
    'Drink_Coffee': ['No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

df_encoded = df.copy()
df_encoded['Time_of_Day'] = df_encoded['Time_of_Day'].map({'Morning' : 0, 'Afternoon' : 1})
df_encoded['Tired'] = df_encoded['Tired'].map({'No' : 0, 'Yes' : 1})
df_encoded['Drink_Coffee'] = df_encoded['Drink_Coffee'].map({'No' : 0, 'Yes' : 1})

X = df_encoded[['Time_of_Day', 'Tired']]
y = df_encoded['Drink_Coffee']


model = tree.DecisionTreeClassifier()
model.fit(X, y)

print("Dự đoán uống cà phê vào buổi chiều khi mệt: ", model.predict([[1, 1]]))