import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from src.weather_data_processing import preprocess_data

raw_df = pd.read_csv('../data/weatherAUS.csv')

data_dict, model_components = preprocess_data(raw_df)

model = RandomForestClassifier(max_leaf_nodes=550,
                               random_state=7,
                               class_weight='balanced')
model.fit(data_dict['train_X'], data_dict['train_y'])

aussie_rain = {
    'model': model,
    'imputer': model_components['imputer'],
    'scaler': model_components['scaler'],
    'encoder': model_components['encoder'],
    'input_cols': model_components['input_cols'],
    'target_col': model_components['target_col'],
    'numeric_cols': model_components['numeric_cols'],
    'categorical_cols': model_components['categorical_cols'],
    'encoded_cols': model_components['encoded_cols']
}

joblib.dump(aussie_rain, '../models/aussie_rain_rf.joblib')
