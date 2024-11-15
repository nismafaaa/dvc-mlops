import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import pandas as pd
import joblib

df_encoded = pd.read_csv('data/processed_data.csv')

label_encoder = LabelEncoder()
df_encoded['class'] = label_encoder.fit_transform(df_encoded['class'])

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'num_leaves': randint(20, 100),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 300),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

model = lgb.LGBMClassifier(verbosity=-1)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,  
    cv=5,        
    scoring='f1', 
    random_state=42,
    n_jobs=-1    
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = random_search.best_estimator_

model_folder = 'model'

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = os.path.join(model_folder, 'best_model.pkl')
joblib.dump(best_model, model_path)

print(f"Model saved to {model_path}")