import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Any, Tuple


def drop_na_values(
        df: pd.DataFrame,
        columns: list
) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    return df.dropna(subset=columns)


def split_data_by_year(
        df: pd.DataFrame,
        year_col: str
) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training, validation, and test sets
    based on the year.

    Args:
        df (pd.DataFrame): The raw dataframe.
        year_col (str): The column containing year information.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train,
                                 validation, and test dataframes.
    """
    year = pd.to_datetime(df[year_col]).dt.year
    train_df = df[year < 2015]
    val_df = df[year == 2015]
    test_df = df[year > 2015]
    return {'train': train_df, 'val': val_df, 'test': test_df}


def create_inputs_targets(
        df_dict: Dict[str, pd.DataFrame],
        input_cols: list,
        target_col: str
) -> Dict[str, Any]:
    """
    Create inputs and targets for training, validation, and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train,
                                           validation, and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train,
                        validation, and test sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data


def impute_missing_values(
        data: Dict[str, Any],
        numeric_cols: list
) -> None:
    """
    Impute missing numerical values using the mean strategy.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
                               for train, validation, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    imputer = (SimpleImputer(strategy='mean')
               .fit(data['train_inputs'][numeric_cols]))
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = imputer.transform(
            data[f'{split}_inputs'][numeric_cols]
        )
    data['imputer'] = imputer


def scale_numeric_features(
        data: Dict[str, Any],
        numeric_cols: list
) -> None:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
                               for train, validation, and test sets.
        numeric_cols (list): List of numerical columns.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(
            data[f'{split}_inputs'][numeric_cols]
        )
    data['scaler'] = scaler


def encode_categorical_features(
        data: Dict[str, Any],
        categorical_cols: list
) -> None:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets
                               for train, validation, and test sets.
        categorical_cols (list): List of categorical columns.
    """
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown='ignore'
    ).fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val', 'test']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat(
            [data[f'{split}_inputs'],
             pd.DataFrame(encoded,
                          columns=encoded_cols,
                          index=data[f'{split}_inputs'].index)], axis=1
            )
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    data['encoded_cols'] = encoded_cols
    data['encoder'] = encoder


def preprocess_data(
        raw_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.

    Returns:
        Dict[str, Any]: Dictionary containing processed inputs and targets
                        for train, validation, and test sets.
    """
    raw_df = drop_na_values(raw_df, ['RainToday', 'RainTomorrow'])
    split_dfs = split_data_by_year(raw_df, 'Date')
    input_cols = list(raw_df.columns)[1:-1]
    target_col = 'RainTomorrow'
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = (data['train_inputs']
                    .select_dtypes(include=np.number)
                    .columns.tolist())
    categorical_cols = (data['train_inputs']
                        .select_dtypes('object')
                        .columns.tolist())

    impute_missing_values(data, numeric_cols)
    scale_numeric_features(data, numeric_cols)
    encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val, X_test
    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]
    X_test = data['test_inputs'][numeric_cols + data['encoded_cols']]

    return {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'test_X': X_test,
        'test_y': data['test_targets']
    }, {
        'input_cols': input_cols,
        'target_col': target_col,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'encoded_cols': data['encoded_cols'],
        'imputer': data['imputer'],
        'scaler': data['scaler'],
        'encoder': data['encoder'],
    }


def preprocess_new_data(
    new_data: Dict[str, Any],
    model_components: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocesses new data using a previously trained imputer, scaler,
    and encoder.

    This function applies the same preprocessing steps
    (imputing, scaling for numeric columns and encoding
    for categorical columns) that were used during training.

    Args:
        new_data (Dict[str, Any]): A dictionary containing raw input features.
        model_components (Dict[str, Any]): A dictionary with the trained
                                           preprocessing tools
                                           (scaler, encoder, etc.).

    Returns:
        pd.DataFrame: The preprocessed input data, ready for prediction.
    """
    input_df = pd.DataFrame([new_data])
    numeric_cols = model_components['numeric_cols']
    categorical_cols = model_components['categorical_cols']
    imputer = model_components['imputer']
    scaler = model_components['scaler']
    encoder = model_components['encoder']

    preprocessed_new_data = input_df.copy()

    preprocessed_new_data.loc[:, numeric_cols] = imputer.transform(
        preprocessed_new_data[numeric_cols]
    )

    preprocessed_new_data.loc[:, numeric_cols] = scaler.transform(
        preprocessed_new_data[numeric_cols]
    )

    encoded = encoder.transform(preprocessed_new_data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=preprocessed_new_data.index
    )

    preprocessed_new_data = pd.concat(
        [preprocessed_new_data.drop(columns=categorical_cols), encoded_df],
        axis=1
    )
    return preprocessed_new_data


def load_model_components(
        model_path: str
) -> Dict[str, Any]:
    """
    Load the trained model and its preprocessing components from a joblib file.

    Args:
        model_path (str): Path to the joblib file containing the model
                          and components.

    Returns:
        Dict[str, Any]: Dictionary containing input columns,
                        preprocessing tools, and the trained model.
    """
    loaded_model = joblib.load(model_path)
    return {
        'input_cols': loaded_model['input_cols'],
        'target_col': loaded_model['target_col'],
        'numeric_cols': loaded_model['numeric_cols'],
        'categorical_cols': loaded_model['categorical_cols'],
        'encoded_cols': loaded_model['encoded_cols'],
        'imputer': loaded_model['imputer'],
        'scaler': loaded_model['scaler'],
        'encoder': loaded_model['encoder'],
        'model': loaded_model['model'],
    }


def predict_input(
        new_data: Dict[str, Any],
        model_path: str
) -> Tuple[str, float]:
    """
    Predict the weather outcome and its probability for a given input.

    Args:
        new_data (Dict[str, Any]): Dictionary containing input features.

    Returns:
        Tuple[str, float]: Predicted label and its probability.
    """
    model_components = load_model_components(model_path)
    X_input = preprocess_new_data(new_data, model_components)
    model = model_components['model']
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob


def get_plot(
        raw_df: pd.DataFrame,
        Location: str
) -> Figure:
    """
    Generate a bar and line plot showing monthly rainfall
    and temperature statistics for a given location.

    Args:
        raw_df (pd.DataFrame): Raw weather dataset.
        Location (str): Location name to filter the dataset.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure
                                  containing the weather plot.
    """
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df['Month'] = raw_df['Date'].dt.month
    raw_df['Year'] = raw_df['Date'].dt.year
    raw_df.loc[:, 'RainToday'] = raw_df['RainToday'].map({'Yes': 1, 'No': 0})

    temp_df = raw_df[raw_df['Location'] == Location]

    avg_rain_per_month = (temp_df
                          .groupby(['Year', 'Month'])['RainToday']
                          .sum()
                          .reset_index()
                          .groupby('Month')['RainToday']
                          .mean())

    fig, ax = plt.subplots(figsize=(4, 2))

    ax.bar(range(1, 13),
           avg_rain_per_month,
           color='#00CED1',
           label='Days with Rain')

    ax.plot(temp_df.groupby('Month')[['MinTemp']].mean(),
            color='#D1B100',
            label='Minimum Temperature',
            linewidth=1)
    ax.plot(temp_df.groupby('Month')[['MaxTemp']].mean(),
            color='#D10093',
            label='Maximum Temperature',
            linewidth=1)
    ax.set_title(f'The Weather in {Location}', color='#696969')
    ax.tick_params(axis='both', labelsize=6, labelcolor='gray')

    ax.set_xticks(ticks=range(1, 13), labels=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ], rotation=45)

    ax.spines[:].set_color('gray')
    ax.legend(fontsize=5, labelcolor='gray')

    return fig
