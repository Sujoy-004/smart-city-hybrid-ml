"""
Smart City Hybrid ML - Data Preprocessing Module
Functions: load_raw(), clean(), save_processed()
"""

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

def save_processed(df, output_path='/content/data/processed/traffic_pollution_clean.csv'):
    """
    Save processed dataset to CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset to save
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
