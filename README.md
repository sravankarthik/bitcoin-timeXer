# Bitcoin Price Forecasting with NeuralForecast

This notebook performs Bitcoin price forecasting using various deep learning models implemented via the `NeuralForecast` library. The goal is to predict future Bitcoin 'open' prices leveraging historical data and 'Global liquidity' as an exogenous variable.

## Data Preprocessing

The following steps were performed to prepare the data for modeling:

1.  **Loading Data**: The `bitcoin.csv` dataset was loaded into a Pandas DataFrame.
2.  **Column Selection**: Only the 'time', 'open', and 'Global liquidity' columns were retained.
3.  **Time Conversion**: The 'time' column was converted to datetime objects.
4.  **Sorting**: The DataFrame was sorted by the 'time' column.
5.  **Date Filtering**: Data was filtered to include entries from '2020-01-01' onwards.
6.  **Lagged Features**: Lagged features for 'Global liquidity' were generated for various horizons (1, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98 days) to be used as exogenous variables.
7.  **Renaming Columns**: 'time' was renamed to 'ds' and 'open' to 'y' to conform to `NeuralForecast` requirements. A 'unique_id' column was added with the value 'BTC'.

## Models Used

Several deep learning models from `NeuralForecast` were utilized and compared:

- **TimeXer**: Used both without exogenous variables and with 'Global liquidity' and its lagged features (`TimeXer_Exog`).
- **PatchTST**
- **LSTM**
- **NBEATS**

## Experimental Setup

The models were evaluated using a rolling forecast approach:

- A `test_size` of 120 days was used to define the total forecasting period.
- The forecasting horizon `h` was varied (e.g., 49, 42, 35, 28, 21, 14, 7, 56, 63, 70 days) to assess model performance at different prediction lengths.
- For each iteration of the rolling forecast, models were trained on the available historical data, and predictions were made for the next `h` days. The training set was then expanded with the actual observed values for those `h` days.

## Performance Metrics

**Mean Squared Error (MSE)** was used to evaluate the performance of each model. A lower MSE indicates better forecasting accuracy.

## Environment Setup

To set up the environment and run this notebook, follow these steps:

1.  **(Optional) Create a virtual environment**:
    ```bash
    python -m venv venv
    # For Windows
    .\venv\Scripts\Activate.ps1
    # For macOS/Linux
    source venv/bin/activate
    ```
2.  **Install Jupyter and ipykernel**:
    ```bash
    pip install jupyter ipykernel
    python -m ipykernel install --user --name venv
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
