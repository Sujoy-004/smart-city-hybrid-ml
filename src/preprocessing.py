
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_raw(data_path='/content/data/raw/delhi_aqi.csv', validate=True):
    """
    Load raw Delhi AQI dataset and perform basic validation.
    
    Parameters:
    -----------
    data_path : str
        Path to the raw CSV file
    validate : bool
        Whether to perform data validation checks
        
    Returns:
    --------
    pandas.DataFrame
        Raw dataset loaded from CSV
    """
    print("🔄 Loading raw data...")
    
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded {data_path}")
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if validate:
            print("🔍 Running validation checks...")
            
            # Check expected columns
            expected_cols = ['Date', 'Month', 'Year', 'Holidays_Count', 'Days', 
                           'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")
            
            # Check for completely empty dataset
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Check data types
            numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    print(f"⚠️  {col} is not numeric, will need type conversion")
            
            # Basic range validation
            if (df['AQI'] < 0).any() or (df['AQI'] > 500).any():
                print("⚠️  AQI values outside expected range [0, 500]")
            
            if (df['Month'] < 1).any() or (df['Month'] > 12).any():
                print("⚠️  Month values outside expected range [1, 12]")
            
            print("✅ Validation completed")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def clean(df, handle_outliers='keep', missing_strategy='none'):
    """
    Clean the dataset by handling missing values, outliers, and data types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset to clean
    handle_outliers : str
        Strategy for outliers: 'keep', 'cap', 'remove'  
    missing_strategy : str
        Strategy for missing values: 'none', 'drop', 'impute'
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    print("🧹 Cleaning data...")
    df_clean = df.copy()
    
    # Record initial shape
    initial_shape = df_clean.shape
    print(f"📊 Initial shape: {initial_shape[0]:,} rows × {initial_shape[1]} columns")
    
    # 1. Handle missing values (from audit: we found none, but good to have)
    missing_before = df_clean.isnull().sum().sum()
    if missing_before > 0:
        print(f"🔧 Found {missing_before} missing values")
        
        if missing_strategy == 'drop':
            df_clean = df_clean.dropna()
            print(f"   → Dropped rows with missing values")
            
        elif missing_strategy == 'impute':
            # Impute numerical columns with median
            numeric_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"   → Imputed {col} with median: {median_val:.2f}")
    else:
        print("✅ No missing values found")
    
    # 2. Fix data types  
    print("🔧 Ensuring correct data types...")
    
    # Ensure integer columns are integers
    int_cols = ['Date', 'Month', 'Year', 'Holidays_Count', 'Days', 'AQI']
    for col in int_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    
    # Ensure float columns are floats
    float_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
    for col in float_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(float)
    
    # 3. Handle data quality issues found in audit
    print("🔧 Fixing data quality issues...")
    
    # Cap data at reasonable maximums (from audit: found 1000 values)
    if handle_outliers == 'cap':
        # Cap PM2.5 and PM10 at 999 (instead of exactly 1000 which looks like data error)
        pm25_capped = (df_clean['PM2.5'] >= 1000).sum()
        pm10_capped = (df_clean['PM10'] >= 1000).sum()
        
        df_clean['PM2.5'] = df_clean['PM2.5'].clip(upper=999)
        df_clean['PM10'] = df_clean['PM10'].clip(upper=999)
        
        if pm25_capped > 0:
            print(f"   → Capped {pm25_capped} PM2.5 values at 999")
        if pm10_capped > 0:
            print(f"   → Capped {pm10_capped} PM10 values at 999")
    
    elif handle_outliers == 'remove':
        # Remove extreme outliers using IQR method
        initial_rows = len(df_clean)
        
        pollution_vars = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
        for var in pollution_vars:
            Q1 = df_clean[var].quantile(0.25)
            Q3 = df_clean[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
            upper_bound = Q3 + 3 * IQR
            
            outliers_mask = (df_clean[var] < lower_bound) | (df_clean[var] > upper_bound)
            outliers_removed = outliers_mask.sum()
            
            if outliers_removed > 0:
                df_clean = df_clean[~outliers_mask]
                print(f"   → Removed {outliers_removed} extreme outliers from {var}")
        
        final_rows = len(df_clean)
        if final_rows < initial_rows:
            print(f"   → Total rows removed: {initial_rows - final_rows}")
    
    # 4. Add data validation
    print("✅ Final validation...")
    
    # Ensure AQI is within valid range
    invalid_aqi = ((df_clean['AQI'] < 0) | (df_clean['AQI'] > 500)).sum()
    if invalid_aqi > 0:
        print(f"⚠️  Found {invalid_aqi} invalid AQI values")
        df_clean = df_clean[(df_clean['AQI'] >= 0) & (df_clean['AQI'] <= 500)]
    
    # Ensure months are valid
    invalid_months = ((df_clean['Month'] < 1) | (df_clean['Month'] > 12)).sum()
    if invalid_months > 0:
        print(f"⚠️  Found {invalid_months} invalid month values")
        df_clean = df_clean[(df_clean['Month'] >= 1) & (df_clean['Month'] <= 12)]
    
    # Final shape
    final_shape = df_clean.shape
    print(f"📊 Final shape: {final_shape[0]:,} rows × {final_shape[1]} columns")
    
    if final_shape[0] < initial_shape[0]:
        rows_removed = initial_shape[0] - final_shape[0]
        print(f"🗑️  Removed {rows_removed:,} rows ({rows_removed/initial_shape[0]*100:.1f}%)")
    
    return df_clean

def feature_engineer(df):
    """
    Apply comprehensive feature engineering to cleaned dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe with Date, Month, Year columns
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with engineered features (77 total features)
    """
    print("🔧 Starting feature engineering...")
    df_eng = df.copy()
    
    # Create proper datetime if needed
    if 'datetime' not in df_eng.columns:
        print("   → Creating datetime column...")
        df_eng['datetime'] = pd.to_datetime(df_eng[['Year', 'Month', 'Date']].rename(columns={'Date': 'day'}))
    
    # Sort by datetime to ensure proper time series order
    df_eng = df_eng.sort_values('datetime').reset_index(drop=True)
    
    # ===================================================================
    # TEMPORAL FEATURES
    # ===================================================================
    print("   → Creating temporal features...")
    
    # Basic temporal features
    df_eng['year'] = df_eng['datetime'].dt.year
    df_eng['month'] = df_eng['datetime'].dt.month  
    df_eng['day'] = df_eng['datetime'].dt.day
    df_eng['weekday'] = df_eng['datetime'].dt.weekday  # 0=Monday, 6=Sunday
    df_eng['day_of_year'] = df_eng['datetime'].dt.dayofyear
    df_eng['week_of_year'] = df_eng['datetime'].dt.isocalendar().week
    
    # Cyclical encoding for temporal features (important for ML models)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['month'] / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['month'] / 12)
    df_eng['day_sin'] = np.sin(2 * np.pi * df_eng['day_of_year'] / 365)
    df_eng['day_cos'] = np.cos(2 * np.pi * df_eng['day_of_year'] / 365)
    df_eng['weekday_sin'] = np.sin(2 * np.pi * df_eng['weekday'] / 7)
    df_eng['weekday_cos'] = np.cos(2 * np.pi * df_eng['weekday'] / 7)
    
    # Season encoding
    def get_season(month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    df_eng['season'] = df_eng['month'].apply(get_season)
    
    # ===================================================================
    # ROLLING WINDOW FEATURES
    # ===================================================================
    print("   → Creating rolling window features...")
    
    # Define pollutant columns for rolling features
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']
    
    # 3-day rolling averages
    for col in pollutant_cols:
        df_eng[f'{col}_rolling_3d'] = df_eng[col].rolling(window=3, min_periods=1).mean()
    
    # 7-day rolling averages  
    for col in pollutant_cols:
        df_eng[f'{col}_rolling_7d'] = df_eng[col].rolling(window=7, min_periods=1).mean()
    
    # Rolling standard deviations (volatility measures)
    for col in pollutant_cols:
        df_eng[f'{col}_rolling_7d_std'] = df_eng[col].rolling(window=7, min_periods=1).std()
    
    # ===================================================================
    # LAG FEATURES
    # ===================================================================
    print("   → Creating lag features...")
    
    # 1-day lag features (yesterday's values)
    for col in pollutant_cols:
        df_eng[f'{col}_lag_1d'] = df_eng[col].shift(1)
    
    # 7-day lag features (same day last week)
    for col in pollutant_cols:
        df_eng[f'{col}_lag_7d'] = df_eng[col].shift(7)
    
    # Difference features (change from previous day)
    for col in pollutant_cols:
        df_eng[f'{col}_diff_1d'] = df_eng[col] - df_eng[col].shift(1)
    
    # ===================================================================
    # INTERACTION AND DERIVED FEATURES
    # ===================================================================
    print("   → Creating interaction features...")
    
    # Pollutant ratios (often meaningful for air quality)
    df_eng['PM2.5_PM10_ratio'] = df_eng['PM2.5'] / (df_eng['PM10'] + 1e-6)  # avoid division by zero
    df_eng['NO2_SO2_ratio'] = df_eng['NO2'] / (df_eng['SO2'] + 1e-6)
    
    # Combined pollutant indices
    df_eng['total_particulates'] = df_eng['PM2.5'] + df_eng['PM10']
    df_eng['total_gases'] = df_eng['NO2'] + df_eng['SO2'] + df_eng['CO']
    
    # Weekend indicator
    df_eng['is_weekend'] = (df_eng['weekday'] >= 5).astype(int)
    
    # Holiday interaction with pollutants
    df_eng['holiday_pm25_interaction'] = df_eng['Holidays_Count'] * df_eng['PM2.5']
    
    # High pollution event indicators
    df_eng['high_aqi'] = (df_eng['AQI'] > df_eng['AQI'].quantile(0.75)).astype(int)
    df_eng['very_high_aqi'] = (df_eng['AQI'] > df_eng['AQI'].quantile(0.9)).astype(int)
    
    # Weather-like interactions (using existing pollutants as proxies)
    df_eng['ozone_temp_proxy'] = df_eng['Ozone'] * df_eng['month']  # Ozone often correlates with temperature
    
    # ===================================================================
    # HANDLE MISSING VALUES
    # ===================================================================
    print("   → Handling missing values from lag features...")
    
    # Fill initial lag values with backward fill then forward fill
    lag_columns = [col for col in df_eng.columns if 'lag' in col or 'diff' in col]
    df_eng[lag_columns] = df_eng[lag_columns].fillna(method='bfill').fillna(df_eng[lag_columns].mean())
    
    # Fill any remaining NaN values with median
    df_eng = df_eng.fillna(df_eng.median(numeric_only=True))
    
    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"✅ Feature engineering complete!")
    print(f"📊 Final shape: {df_eng.shape[0]:,} rows × {df_eng.shape[1]} columns")
    print(f"🆕 New features created: {df_eng.shape[1] - df.shape[1]}")
    
    # Feature summary
    temporal_features = len([col for col in df_eng.columns if any(x in col for x in ['year', 'month', 'day', 'weekday', 'season', '_sin', '_cos'])])
    rolling_features = len([col for col in df_eng.columns if 'rolling' in col])
    lag_features = len([col for col in df_eng.columns if 'lag' in col or 'diff' in col])
    interaction_features = len([col for col in df_eng.columns if any(x in col for x in ['ratio', 'total', 'interaction', 'high_', 'is_', 'proxy'])])
    
    print(f"   → Temporal features: {temporal_features}")
    print(f"   → Rolling features: {rolling_features}")
    print(f"   → Lag features: {lag_features}")
    print(f"   → Interaction features: {interaction_features}")
    
    return df_eng

def save_processed(df, output_path='/content/data/processed/traffic_pollution_clean.csv'):
    """
    Save processed dataset to CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned/engineered dataset to save
    output_path : str
        Path where to save the processed data
    """
    print(f"💾 Saving processed data to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"📁 File size: {os.path.getsize(output_path) / 1024:.1f} KB")

# ===================================================================
# WORKFLOW FUNCTIONS
# ===================================================================

def full_preprocessing_pipeline(data_path='/content/data/raw/delhi_aqi.csv', 
                               output_path='/content/data/processed/traffic_pollution_clean.csv',
                               handle_outliers='keep', 
                               missing_strategy='none',
                               apply_feature_engineering=True):
    """
    Run the complete preprocessing pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to raw data file
    output_path : str
        Path to save processed data
    handle_outliers : str
        Strategy for outliers: 'keep', 'cap', 'remove'
    missing_strategy : str
        Strategy for missing values: 'none', 'drop', 'impute'
    apply_feature_engineering : bool
        Whether to apply feature engineering
        
    Returns:
    --------
    pandas.DataFrame
        Fully processed dataset
    """
    print("🚀 Starting full preprocessing pipeline...")
    print("=" * 50)
    
    # Step 1: Load raw data
    df_raw = load_raw(data_path, validate=True)
    
    print("=" * 50)
    
    # Step 2: Clean data
    df_clean = clean(df_raw, handle_outliers=handle_outliers, missing_strategy=missing_strategy)
    
    print("=" * 50)
    
    # Step 3: Feature engineering (optional)
    if apply_feature_engineering:
        df_final = feature_engineer(df_clean)
    else:
        df_final = df_clean
        print("⏭️  Skipping feature engineering")
    
    print("=" * 50)
    
    # Step 4: Save processed data
    save_processed(df_final, output_path)
    
    print("=" * 50)
    print("🎉 Preprocessing pipeline completed successfully!")
    
    return df_final
