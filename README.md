# Rain in Australia - Streamlit App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://australia-rain-predict.streamlit.app/)

This project predicts whether it will rain tomorrow in Australia using historical weather data and machine learning models. The app provides an interactive interface for users to input weather parameters and get predictions, as well as visualize weather trends.

 **Live App**: [australia-rain-predict.streamlit.app](https://australia-rain-predict.streamlit.app/)

## Project Structure

```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── data/
│ └── weatherAUS.csv        # Weather dataset
├── img/
│ └── images.png            # App logo/image
├── models/
│ └── aussie_rain_rf.joblib # Trained ML model
├── notebooks/
│ ├── LogisticRegression.ipynb # Baseline model exploration and training
│ └── RandomForest.ipynb    # Final model training and evaluation
├── src/
├── train.py                # Script to preprocess and train model
└── weather_data_processing.py# Functions for data processing and prediction
```

## Technologies Used
- Python 3.12
- Streamlit - Interactive web app framework
- Scikit-learn - Machine learning library
- Pandas, NumPy - Data manipulation and analysis
- Matplotlib - Data visualization
- Joblib - Model serialization

## Getting Started

1. **Clone the repository**

   ```sh
   git clone https://github.com/nvkoval/australia_rain_streamlit_app.git
   cd australia_rain_streamlit_app
   ```
2. **Create a virtual environment** (optional but recommended)

   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```sh
   streamlit run app.py
   ```

5. **Interact with the app**

   - Select weather parameters in the sidebar
   - Click "Get prediction" to see if it will rain tomorrow
   - Explore weather trends for the selected location


## Data
- [Weather dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) from Kaggle
- The dataset contains historical weather data for Australia, including features such as temperature, humidity, wind speed, and more. The target variable is whether it rained the next day.
- The dataset is preprocessed to handle missing values and categorical variables before being used for model training.

## Model Training

Model training and preprocessing code can be found in [`src/train.py`](src/train.py) and the `notebooks/` directory.
You can train the model using the provided Jupyter notebooks or run the `train.py` script directly.

## License
This project is released for educational purposes. Feel free to explore and adapt it.
