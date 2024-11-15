from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

df_encoded = pd.read_csv('data/processed_data.csv')

label_encoder = LabelEncoder()
df_encoded['class'] = label_encoder.fit_transform(df_encoded['class'])

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = joblib.load('model/best_model.pkl')

y_pred = best_model.predict(X_test)

predictions_df = pd.DataFrame({
    'True Labels': y_test,
    'Predicted Labels': y_pred
})

predictions_df.to_csv('data/predictions.csv', index=False)
print("Predictions saved to data/predictions.csv")