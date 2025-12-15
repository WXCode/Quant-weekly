# main_script.py

import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
import seaborn as sns
import requests
import hashlib
from tqdm import tqdm
import warnings
import os

TRAIN_MODELS  = True
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# --- 1. Data Fetching and Caching ---
def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    try:
        # Add a User-Agent header to mimic a browser visit and prevent 403 error
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Will raise an exception for bad status codes
        tables = pd.read_html(response.text)
        tickers = tables[0]['Symbol'].tolist()
        # Clean up tickers that might have issues with yfinance
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"Successfully fetched {len(tickers)} tickers from Wikipedia.")
        return tickers
    except Exception as e:
        warnings.warn(f"Could not fetch S&P 500 tickers: {e}. Falling back to a small, hardcoded list.")
        # Fallback to a smaller, reliable list if Wikipedia fails
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG']

def download_data(tickers, start_date='2015-01-01', end_date='2023-12-31', cache_dir='data_cache'):
    """Downloads historical price data for a list of tickers."""
    # --- Smart Caching Logic ---
    # Create a unique filename based on the hash of the ticker list and dates
    import os
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    ticker_hash = hashlib.md5(str(sorted(tickers)).encode()).hexdigest()
    cache_filename = f"{ticker_hash}_{start_date}_{end_date}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    # --- Caching Logic ---
    try:
        if pd.to_datetime(end_date) < pd.Timestamp.now() - pd.Timedelta(days=1):
            # Only use cache if the end_date is in the past
            data = pd.read_pickle(cache_path)
            print("Loaded data from cache.")
            return data
    except FileNotFoundError:
        print("Cache not found. Downloading data...")
    
    try:
        print(f"Downloading data for {len(tickers)} tickers...")
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')
        assert not data.empty, "yfinance.download returned an empty DataFrame. Check tickers and network."
        # Forward-fill and then back-fill to handle intermittent missing data
        data.ffill(inplace=True)
        data.bfill(inplace=True)

        data.to_pickle(cache_path) # Save to cache
        return data
    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return None

# --- 2. Feature Engineering ---
def create_features(data, lags=5, window_sizes=[20, 50, 200]):
    """
    Creates a variety of technical features for our models.
    - Momentum (ROC)
    - Volatility (rolling std dev)
    - Moving Average Convergence Divergence (MACD)
    - Relative Strength Index (RSI)
    - Volume trends
    """
    print("Engineering features...")
    close_prices = data.loc[:, pd.IndexSlice[:, 'Close']].droplevel(1, axis=1)
    volume = data.loc[:, pd.IndexSlice[:, 'Volume']].droplevel(1, axis=1)
    
    feature_dfs = []

    # Momentum Features
    for lag in range(1, lags + 1):
        feature_dfs.append(close_prices.pct_change(lag).add_suffix(f'_roc_{lag}'))

    # Volatility and MA Features
    for window in window_sizes:
        feature_dfs.append(close_prices.rolling(window).std().div(close_prices).add_suffix(f'_vol_{window}'))
        feature_dfs.append(close_prices.div(close_prices.rolling(window).mean()).add_suffix(f'_ma_ratio_{window}'))

    # RSI
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    feature_dfs.append(rsi.add_suffix('_rsi_14'))
    
    # Volume trends
    for window in window_sizes:
        feature_dfs.append(volume.rolling(window).mean().div(volume).add_suffix(f'_vol_ratio_{window}'))

    features = pd.concat(feature_dfs, axis=1)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Align features with prices, dropping initial NaNs from rolling windows
    # We will handle NaNs after reshaping, so we don't drop all data here.
    combined = pd.concat([close_prices, features], axis=1)
    
    final_close_prices = combined[close_prices.columns]
    final_features = combined.drop(columns=close_prices.columns)
    return final_features, final_close_prices

# --- 3. Target Definition (Multi-day forward returns) ---  
def create_target(close_prices, horizon=5):
    """
    Define the target variable based on cross-sectional return ranks.
    - Success (1): Stock is in the top 25 performers over the horizon.
    - Fail (0): Stock is in the bottom 25 performers over the horizon.
    - Middle stocks are masked (NaN) and ignored during training.
    """
    # Calculate forward returns
    forward_returns = close_prices.shift(-horizon) / close_prices - 1
    
    # Create an empty target DataFrame with the same shape and index, filled with NaNs
    target = pd.DataFrame(np.nan, index=forward_returns.index, columns=forward_returns.columns)
    
    # Get ranks for each day. `ascending=False` means smaller rank is better (higher return).
    ranks = forward_returns.rank(axis=1, ascending=False, method='min')
    
    # Calculate the rank of the 25th worst stock for each day
    bottom_rank_threshold = forward_returns.notna().sum(axis=1) - 24
    
    # Set target to 1 for top 25 stocks and 0 for bottom 25 stocks
    target[ranks <= 25] = 1
    target[ranks.ge(bottom_rank_threshold, axis=0)] = 0
    
    return target

# --- 4. Backtesting Engine ---
def run_backtest(predictions, close_prices, horizon=5, top_k=10):
    """
    A simple vectorized backtest for a long-only portfolio.
    - Go long the `top_k` stocks with the highest predicted probability.
    """
    print("Running backtest...")
    # Align predictions with the price data for calculating returns
    aligned_prices = close_prices.reindex(predictions.index)
    
    # Calculate forward returns for the holding period
    returns = aligned_prices.shift(-horizon) / aligned_prices - 1
    
    # Shift predictions to align with the start of the holding period
    # We make a prediction on day `t` for performance from `t` to `t+h`
    # So we use the prediction at `t` to calculate returns from `t`
    
    long_returns = pd.Series(0, index=predictions.index)
    
    # This loop simulates the rebalancing period (e.g., every `horizon` days)
    for i in range(0, len(predictions), horizon):
        date = predictions.index[i]
        
        # Get predictions for the current date
        daily_preds = predictions.loc[date]
        
        # Identify top and bottom k stocks
        long_candidates = daily_preds.nlargest(top_k).index
        
        # Calculate portfolio returns for the next `horizon` days
        # We assume equal weighting
        if date in returns.index:
            long_ret = returns.loc[date, long_candidates].mean()
            
            # Store the single return for this period
            if i + horizon <= len(predictions):
                long_returns.iloc[i:i+horizon] = long_ret / horizon

    strategy_daily_returns = long_returns
    
    return strategy_daily_returns.fillna(0)

# --- 5. Visualization and Metrics ---
def plot_results(strategy_returns, benchmark_returns):
    """Plots cumulative returns and calculates key performance metrics."""
    
    # --- Cumulative Returns Plot ---
    plt.figure(figsize=(14, 7))
    
    # Strategy
    strategy_cumulative = (1 + strategy_returns).cumprod()
    plt.plot(strategy_cumulative, label='ML Strategy')
    
    # Benchmark
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    plt.plot(benchmark_cumulative, label='S&P 500 (SPY) Benchmark', linestyle='--')
    
    plt.title('Strategy Cumulative Returns vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

    # --- Performance Metrics ---
    def calculate_metrics(returns):
        metrics = {}
        # More robust CAGR calculation based on total return
        total_return = (1 + returns).prod()
        num_years = len(returns) / 252
        cagr = (total_return ** (1 / num_years)) - 1 if num_years > 0 else 0
        metrics['CAGR (%)'] = cagr * 100

        metrics['Sharpe Ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        metrics['Max Drawdown (%)'] = drawdown.min() * 100
        return metrics

    strategy_metrics = calculate_metrics(strategy_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)
    
    metrics_df = pd.DataFrame({
        'ML Strategy': strategy_metrics,
        'Benchmark': benchmark_metrics
    }).round(2)
    
    print("\n--- Performance Metrics ---")
    print(metrics_df)

# --- 6. Strategy-Specific Logic ---
def get_regime_models_and_predictions(X_train, y_train, X_test, features, n_regimes=3):
    """Identifies market regimes and trains a model for each."""
    print("Identifying market regimes using K-Means...")
    market_features = features.mean(axis=1).to_frame('market_return')
    market_features['market_vol'] = features.filter(like='_vol_').mean(axis=1)
    market_features.dropna(inplace=True)

    scaler = StandardScaler()
    train_market_features = market_features.loc[market_features.index.isin(X_train.index.get_level_values('Date'))]
    scaled_market_features_train = scaler.fit_transform(train_market_features)

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init='auto')
    kmeans.fit(scaled_market_features_train)

    all_scaled_market_features = scaler.transform(market_features)
    market_features['regime'] = kmeans.predict(all_scaled_market_features)

    models_by_regime = _train_regime_models(X_train, y_train, market_features, n_regimes)
    predictions = _predict_with_regime_models(X_test, models_by_regime, market_features)

    return predictions

def _train_regime_models(X_train, y_train, market_features, n_regimes):
    """Helper to train a model for each market regime."""
    models_by_regime = {}
    print("Training one model per regime...")
    for regime in range(n_regimes):
        # Find the dates corresponding to the current regime in the training set
        regime_dates = market_features[market_features['regime'] == regime].index
        train_regime_idx = X_train.index.get_level_values('Date').isin(regime_dates)
        
        X_train_regime, y_train_regime = X_train[train_regime_idx], y_train[train_regime_idx]
        
        if len(X_train_regime) > 100: # Ensure enough data to train
            model = lgb.LGBMClassifier(objective='binary', n_estimators=300, random_state=42)
            model.fit(X_train_regime, y_train_regime)
            models_by_regime[regime] = model
        else:
            print(f"Warning: Not enough data to train model for regime {regime}. It will be skipped.")
    return models_by_regime

def _predict_with_regime_models(X_test, models_by_regime, market_features):
    """Helper to generate predictions using regime-specific models."""
    predictions_list = []
    test_dates = X_test.index.get_level_values('Date').unique()

    for date in tqdm(test_dates, desc="Predicting with Regime Models"):
        try:
            current_regime = market_features.loc[date, 'regime']
            model = models_by_regime.get(current_regime)
            
            if model:
                daily_features = X_test.loc[date]
                daily_preds = model.predict_proba(daily_features)[:, 1]
                predictions_list.append(pd.Series(daily_preds, index=daily_features.index, name=date))
        except KeyError:
            continue # Skip if date is not in market_features (e.g., holiday)
    return pd.concat(predictions_list, axis=1).T

def get_mlp_model_and_predictions(X_train, y_train, X_test):
    """Builds, trains, and predicts using a simple MLP model."""
    print("Building and training MLP model...")
    
    # It's crucial to scale features for neural networks
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=1000, batch_size=64, validation_split=0.2, verbose=1)
    
    print("Generating predictions with MLP on test set...")
    predictions_flat = model.predict(X_test_scaled).flatten()
    return pd.Series(predictions_flat, index=X_test.index).unstack()

def get_comprehensive_strategy_predictions(X_train, y_train, X_test, features_wide):
    """
    Combines hand-crafted features, autoencoder features, and regime information
    to train a single, comprehensive model.
    """
    print("--- Running Comprehensive Strategy ---")

    # 1. Scale the base features
    print("1. Scaling base features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Generate Autoencoder "deep" features
    print("2. Generating Autoencoder features...")
    encoding_dim = 32
    input_dim = X_train_scaled.shape[1]
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(128, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=1000, batch_size=64, shuffle=True, validation_split=0.2, verbose=0)
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=encoder)
    X_train_deep = encoder_model.predict(X_train_scaled)
    X_test_deep = encoder_model.predict(X_test_scaled)

    # 3. Generate Market Regime features
    print("3. Generating market regime features...")
    market_features = features_wide.mean(axis=1).to_frame('market_return')
    market_features['market_vol'] = features_wide.filter(like='_vol_').mean(axis=1)
    market_features.dropna(inplace=True)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(market_features.loc[market_features.index.isin(X_train.index.get_level_values('Date'))])
    market_features['regime'] = kmeans.predict(market_features)
    
    # Map regime to the training and test sets and one-hot encode it
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    regime_train = market_features['regime'].reindex(X_train.index.get_level_values('Date')).values.reshape(-1, 1)
    regime_test = market_features['regime'].reindex(X_test.index.get_level_values('Date')).values.reshape(-1, 1)
    regime_train_ohe = ohe.fit_transform(regime_train)
    regime_test_ohe = ohe.transform(regime_test)

    # 4. Combine all feature sets
    print("4. Combining all feature sets...")
    X_train_comprehensive = np.hstack([X_train_scaled, X_train_deep, regime_train_ohe])
    X_test_comprehensive = np.hstack([X_test_scaled, X_test_deep, regime_test_ohe])

    # 5. Train final LightGBM model
    print("5. Training final comprehensive model...")
    model = lgb.LGBMClassifier(objective='binary', n_estimators=700, random_state=42)
    model.fit(X_train_comprehensive, y_train)
    predictions_flat = model.predict_proba(X_test_comprehensive)[:, 1]
    return pd.Series(predictions_flat, index=X_test.index).unstack()

def get_autoencoder_features_and_predictions(X_train, y_train, X_test, train_models=True):
    """Uses an Autoencoder to create features, then trains LGBM on them."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    encoding_dim = 4 # The size of our new "deep feature" vector
    input_dim = X_train_scaled.shape[1]
    
    # Define model paths
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    autoencoder_path = os.path.join(model_dir, 'autoencoder.h5')
    lgbm_path = os.path.join(model_dir, 'lgbm_on_deep_features.txt')

    if train_models:
        print("Building and training Autoencoder for feature generation...")
        # --- Define Autoencoder Model ---
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder = tf.keras.layers.Dense(2048, activation='relu')(input_layer)
        encoder = tf.keras.layers.Dense(64, activation='relu')(encoder)
        encoder = tf.keras.layers.Dense(32, activation='relu')(encoder)
        encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoder)
        
        decoder = tf.keras.layers.Dense(32, activation='relu')(encoder)
        decoder = tf.keras.layers.Dense(64, activation='relu')(decoder)

        decoder = tf.keras.layers.Dense(2048, activation='relu')(decoder)
        decoder = tf.keras.layers.Dense(input_dim, activation='linear')(decoder)
        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=16, shuffle=True, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        autoencoder.save(autoencoder_path)
        print(f"Autoencoder saved to {autoencoder_path}")
    else:
        print(f"Loading pre-trained autoencoder from {autoencoder_path}...")
        if not os.path.exists(autoencoder_path):
            raise FileNotFoundError(f"Model file not found: {autoencoder_path}. Set train_models=True to train and save the model.")
        autoencoder = tf.keras.models.load_model(autoencoder_path)

    # --- Use Encoder to Create New Features ---
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=encoder)
    X_train_deep = encoder_model.predict(X_train_scaled)
    X_test_deep = encoder_model.predict(X_test_scaled)


    # --- Validate Embeddings (only during training) ---
    visualize_embeddings(X_train_deep, y_train)
    print("Training LGBM model on auto-generated features...")
    model = lgb.LGBMClassifier(objective='binary', n_estimators=500, random_state=42)
    model.fit(X_train_deep, y_train)



    # --- Calculate and print fit AUC for the model on deep features ---
    from sklearn.metrics import roc_auc_score
    train_preds_proba = model.predict_proba(X_train_deep)[:, 1]
    fit_auc = roc_auc_score(y_train, train_preds_proba)
    print(f"LGBM (on Autoencoder Features) Fit AUC: {fit_auc:.4f}")

    # --- Analyze Feature Importance ---
    print("Analyzing importance of auto-generated features...")
    feature_importances = pd.DataFrame({
        'feature': [f'deep_feature_{i}' for i in range(X_train_deep.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
    plt.title('Top 20 Most Important Auto-Generated Features')
    plt.tight_layout()
    plt.show()

    predictions_flat = model.predict_proba(X_test_deep)[:, 1]
    return pd.Series(predictions_flat, index=X_test.index).unstack()

def visualize_embeddings(embeddings, labels):
    """Visualizes the quality of embeddings using t-SNE."""
    print("Visualizing embeddings with t-SNE...")
    sample_size = 5000
    if len(embeddings) > sample_size:
        idx = np.random.choice(np.arange(len(embeddings)), sample_size, replace=False)
        embeddings_sample, labels_sample = embeddings[idx], labels.iloc[idx]
    else:
        embeddings_sample, labels_sample = embeddings, labels

    # --- Data Cleaning for TSNE ---
    # Remove any rows with NaN values from the embeddings and corresponding labels
    valid_idx = ~np.isnan(embeddings_sample).any(axis=1)
    embeddings_sample = embeddings_sample[valid_idx]
    labels_sample = labels_sample.iloc[valid_idx]
    print(f"Removed {np.sum(~valid_idx)} rows with NaN embeddings before running t-SNE.")

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_sample)

    df = pd.DataFrame({'tsne-1': tsne_results[:,0], 'tsne-2': tsne_results[:,1], 'label': labels_sample.values})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="tsne-1", y="tsne-2", hue="label", palette=sns.color_palette("hls", 2), data=df, legend="full", alpha=0.6)
    plt.title("t-SNE Visualization of Autoencoder Embeddings (by Target Label)")
    plt.savefig('tsne_embeddings.png')
    plt.show()
    plt.close()

def check_for_strong_signals(predictions, long_threshold=0.85, short_threshold=0.15):
    """Scans the latest predictions and prints alerts for strong signals."""
    print("\n--- Checking for Strong Trading Signals on Latest Data ---")
    if predictions.empty:
        print("No predictions to analyze.")
        return

    latest_date = predictions.index.max()
    latest_preds = predictions.loc[latest_date].dropna()

    strong_longs = latest_preds[latest_preds > long_threshold]
    strong_shorts = latest_preds[latest_preds < short_threshold]

    if not strong_longs.empty:
        print(f"ALERT: Strong LONG signals for {latest_date.date()}:")
        for ticker, score in strong_longs.items():
            print(f"  - {ticker}: Score = {score:.2f}")
    
    if not strong_shorts.empty:
        print(f"ALERT: Strong SHORT signals for {latest_date.date()}:")
        for ticker, score in strong_shorts.items():
            print(f"  - {ticker}: Score = {score:.2f}")

    if strong_longs.empty and strong_shorts.empty:
        print("No strong signals found on the latest prediction date.")

def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seeds set to {seed}")

def prepare_data_for_modeling(tickers, start_date, end_date, train_end_date, horizon):
    """
    Encapsulates all data preparation steps, from download to train/test split.
    This helps isolate logic and prevent data leakage.
    """
    # --- 1. Data Download and Initial Setup ---
    if 'SPY' not in tickers:
        tickers.append('SPY')

    raw_data = download_data(tickers, start_date, end_date)
    if raw_data is None:
        raise RuntimeError("Data download failed.")
    assert 'SPY' in raw_data.columns.get_level_values(0), "CRITICAL: Benchmark ticker 'SPY' could not be found in downloaded data."

    # --- 2. Split Data *Before* Feature Engineering to Prevent Leakage ---
    # This was incorrect. Features like rolling averages need a continuous history.
    # We will featurize first, then split. This is safe as features only use past data.
    spy_returns = raw_data.loc[:, ('SPY', 'Close')].pct_change().fillna(0)
    spy_returns.name = 'SPY'
    
    stock_data = raw_data.drop(columns='SPY', level=0, errors='ignore')
    features_wide, close_prices = create_features(stock_data)
    target = create_target(close_prices, horizon=horizon)
    
    # --- 3. Alignment ---
    common_index = features_wide.index.intersection(target.index)
    features_wide = features_wide.loc[common_index]
    target = target.loc[common_index]
    
    # --- 4. Reshape Data for ML Models ---
    def reshape_for_ml(features, target=None):
        def robust_split(col_name):
            parts = col_name.split('_', 1)
            return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], '')

        features.columns = pd.MultiIndex.from_tuples(
            [robust_split(c) for c in features.columns],
            names=['Symbols', 'Feature']
        )
        X = features.stack(level='Symbols')

        if target is not None:
            y = target.stack()
            common_idx = X.index.intersection(y.index)
            X, y = X.loc[common_idx], y.loc[common_idx]
            # Now that X and y are aligned, drop any remaining NaNs from the feature set
            X.dropna(inplace=True)
            y = y.loc[X.index]
            return X, y
        
        # For the test set, just drop NaNs from features
        X.dropna(inplace=True)
        return X

    X, y = reshape_for_ml(features_wide, target)

    # --- 5. Train/Test Split ---
    X_train = X.loc[:train_end_date]
    y_train = y.loc[:train_end_date]
    X_test = X.loc[train_end_date:]
    
    close_prices_test = close_prices.loc[train_end_date:]

    return X_train, y_train, X_test, spy_returns, close_prices_test, features_wide

def main():
    """Main function to run the full quantitative strategy pipeline."""
    # Set seeds at the very beginning for reproducibility
    set_seeds(42)

    # --- Configuration ---
    tickers = get_sp500_tickers()
    # For demonstration, let's use a smaller subset to speed things up
    tickers = tickers
    
    start_date = '2015-01-01'
    end_date = '2025-12-15'
    train_end_date = '2025-11-30'
  
    horizon = 5 # 5-day holding period
    
    # Ensure the benchmark ticker is always included in the download request
    if 'SPY' not in tickers:
        tickers.append('SPY')

    # --- Data Preparation ---
    raw_data = download_data(tickers, start_date, end_date)
    if raw_data is None:
        exit()

    # --- Handle Survivorship Bias ---
    # Keep only the stocks that have a sufficient price history
    original_tickers = raw_data.columns.get_level_values(0).unique()
    # Count NaNs in the 'Close' price for each ticker
    nans_per_ticker = raw_data.xs('Close', level=1, axis=1).isna().sum()
    # Identify tickers to drop (more than 10 NaNs)
    tickers_to_drop = nans_per_ticker[nans_per_ticker > 10].index
    # --- Data Preparation (now encapsulated and leakage-free) ---
    X_train, y_train, X_test, spy_returns, close_prices_test, features_wide_train = prepare_data_for_modeling(
        tickers, start_date, end_date, train_end_date, horizon)
    
  
    # --- Strategy Selection ---
    # Choose one of the strategies to run by uncommenting it.
    
    # strategy_choice = 'gradient_boosting'
    # strategy_choice = 'regime_clustering'
    # strategy_choice = 'mlp_prediction'
    strategy_choice = 'autoencoder_features'
    # strategy_choice = 'comprehensive_strategy'
    # strategy_choice = 'cross_sectional_momentum'
    
    print(f"\n--- Running Strategy: {strategy_choice} ---")

    if strategy_choice == 'gradient_boosting':
        # === Strategy 1: Gradient Boosting for Cross-Sectional Ranking ===
        print("Training Gradient Boosting model...")
        # A more tuned set of parameters for a harder classification problem
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,      # More trees for a harder problem
            learning_rate=0.02,     # Lower learning rate to be more careful
            num_leaves=41,          # Slightly more complex trees
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=0.1,         # L2 regularization
            colsample_bytree=0.8,   # Feature subsampling
            subsample=0.8,          # Data subsampling
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        from sklearn.metrics import roc_auc_score
        train_preds_proba = model.predict_proba(X_train)[:, 1]
        fit_auc = roc_auc_score(y_train, train_preds_proba)
        print(f"LGBM (Gradient Boosting) Fit AUC: {fit_auc:.4f}")
        print("Generating predictions on test set...")
        predictions_flat = model.predict_proba(X_test)[:, 1]
        predictions = pd.Series(predictions_flat, index=X_test.index).unstack()

    elif strategy_choice == 'regime_clustering':
        predictions = get_regime_models_and_predictions(X_train, y_train, X_test, features_wide_train, n_regimes=3)

    elif strategy_choice == 'mlp_prediction':
        predictions = get_mlp_model_and_predictions(X_train, y_train, X_test)

    elif strategy_choice == 'autoencoder_features':
        predictions = get_autoencoder_features_and_predictions(X_train, y_train, X_test, train_models=TRAIN_MODELS)

    elif strategy_choice == 'comprehensive_strategy':
        predictions = get_comprehensive_strategy_predictions(X_train, y_train, X_test, features_wide_train)

    elif strategy_choice == 'cross_sectional_momentum':
        # === Strategy 3: Simple Cross-Sectional Momentum (Non-ML Benchmark) ===
        # This is a classic factor strategy. We use past returns as our "prediction".
        # It's a good sanity check for our ML models.
        print("Generating predictions based on past 50-day returns...")
        # The "prediction" is just the rank of the past 50-day return.
        # We need to unstack X to get the wide format back before filtering
        momentum_factor = X.unstack(level='Symbols').filter(like='ma_ratio_50').stack(level='Symbols')
        momentum_factor = momentum_factor.loc[X_test.index] # Align with test set
        predictions = momentum_factor.unstack()

    # --- Final Backtest and Visualization ---
    # Get test dates and tickers from the final predictions DataFrame
    test_dates = predictions.index
    test_tickers = predictions.columns

    # Ensure predictions df covers all necessary dates and tickers for backtesting
    predictions = predictions.reindex(index=test_dates, columns=test_tickers).fillna(0.5) # Fill missing with neutral
    
    # --- Alerting ---
    check_for_strong_signals(predictions, long_threshold=0.8, short_threshold=0.2)

    # Get the close prices for the test period
    test_close_prices = close_prices_test
    
    strategy_returns = run_backtest(predictions, test_close_prices, horizon=horizon, top_k=10)
    
    # Align benchmark returns with strategy returns
    benchmark_test_returns = spy_returns.loc[strategy_returns.index]
    
    plot_results(strategy_returns, benchmark_test_returns)

if __name__ == '__main__':
    main()
