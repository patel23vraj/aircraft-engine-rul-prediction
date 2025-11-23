#!/usr/bin/env python3
"""
Aircraft Engine Remaining Useful Life (RUL) Prediction System
==============================================================

A production-quality machine learning system for predicting the Remaining Useful Life
of aircraft engines using synthetic data that mimics the NASA CMAPSS dataset structure.

This system demonstrates advanced aerospace engineering concepts combined with 
state-of-the-art machine learning techniques suitable for deployment in aerospace
companies like Lockheed Martin, Boeing, or Northrop Grumman.

Author: Aerospace ML Engineering Team
Version: 1.0
Date: 2024
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
from typing import Tuple, Dict, List, Any
import os
from datetime import datetime

# Configure plotting and warnings
plt.style.use('seaborn-v0_8')
warnings.filterwarnings('ignore')
np.random.seed(42)  # For reproducible results

# Global configuration
CONFIG = {
    'n_engines': 50,            # Number of engines in dataset
    'max_cycles': 200,          # Maximum operational cycles
    'n_sensors': 21,            # Number of sensor measurements
    'n_operational_settings': 3, # Number of operational settings
    'noise_level': 0.1,         # Sensor noise level
    'degradation_rate': 0.001,  # Base degradation rate
    'failure_threshold': 130,   # RUL threshold for failure prediction
}

print("="*80)
print("AIRCRAFT ENGINE RUL PREDICTION SYSTEM")
print("="*80)
print(f"Initializing system with {CONFIG['n_engines']} engines...")
print(f"Maximum operational cycles: {CONFIG['max_cycles']}")
print(f"Sensor measurements: {CONFIG['n_sensors']}")
print("="*80)

# ============================================================================
# 2. SYNTHETIC DATA GENERATION (NASA CMAPSS STRUCTURE)
# ============================================================================

def generate_synthetic_engine_data() -> pd.DataFrame:
    """
    Generate synthetic aircraft engine sensor data that mimics the NASA CMAPSS dataset.
    
    This function creates realistic engine operational data including:
    - Engine degradation patterns over time
    - Sensor measurements with realistic noise and correlations
    - Operational settings (altitude, throttle, bypass ratio)
    - Non-linear degradation patterns that accelerate near failure
    
    Returns:
        pd.DataFrame: Synthetic engine data with columns for engine_id, cycle, 
                     operational settings, and sensor measurements
    
    Engineering Context:
        Aircraft engines degrade over time due to wear, thermal cycling, and 
        operational stress. This degradation is captured through multiple sensors
        measuring temperature, pressure, vibration, and other critical parameters.
    """
    print("\n[STAGE 1] Generating synthetic engine data...")
    
    data = []
    
    for engine_id in range(1, CONFIG['n_engines'] + 1):
        # Each engine has a different total lifespan (realistic variation)
        total_cycles = np.random.randint(150, CONFIG['max_cycles'])
        
        # Generate operational settings that vary by flight profile
        base_altitude = np.random.uniform(0.7, 1.0)  # Normalized altitude
        base_throttle = np.random.uniform(0.6, 0.9)  # Normalized throttle
        base_bypass = np.random.uniform(0.5, 0.8)    # Normalized bypass ratio
        
        for cycle in range(1, total_cycles + 1):
            # Calculate degradation factor (accelerates near end of life)
            degradation_progress = cycle / total_cycles
            degradation_factor = 1 + CONFIG['degradation_rate'] * cycle * (1 + degradation_progress**2)
            
            # Operational settings with realistic flight-to-flight variation
            op_setting_1 = base_altitude + np.random.normal(0, 0.05)
            op_setting_2 = base_throttle + np.random.normal(0, 0.03)
            op_setting_3 = base_bypass + np.random.normal(0, 0.02)
            
            # Generate sensor measurements with physics-based correlations
            sensors = {}
            
            # Temperature sensors (affected by throttle and degradation)
            temp_base = 500 + 200 * op_setting_2  # Base temperature
            sensors['sensor_2'] = temp_base * degradation_factor + np.random.normal(0, 10)
            sensors['sensor_3'] = temp_base * 0.9 * degradation_factor + np.random.normal(0, 8)
            sensors['sensor_4'] = temp_base * 1.1 * degradation_factor + np.random.normal(0, 12)
            
            # Pressure sensors (affected by altitude and throttle)
            pressure_base = 100 + 50 * op_setting_1 * op_setting_2
            sensors['sensor_7'] = pressure_base * degradation_factor + np.random.normal(0, 5)
            sensors['sensor_8'] = pressure_base * 0.8 * degradation_factor + np.random.normal(0, 4)
            sensors['sensor_9'] = pressure_base * 1.2 * degradation_factor + np.random.normal(0, 6)
            
            # Flow and efficiency sensors
            flow_base = 300 + 100 * op_setting_3
            sensors['sensor_11'] = flow_base / degradation_factor + np.random.normal(0, 15)
            sensors['sensor_12'] = flow_base * 0.7 / degradation_factor + np.random.normal(0, 10)
            sensors['sensor_13'] = flow_base * 1.3 / degradation_factor + np.random.normal(0, 20)
            
            # Vibration and mechanical sensors (increase with degradation)
            vib_base = 10 + 5 * op_setting_2
            sensors['sensor_14'] = vib_base * degradation_factor**1.5 + np.random.normal(0, 2)
            sensors['sensor_15'] = vib_base * 0.8 * degradation_factor**1.3 + np.random.normal(0, 1.5)
            
            # Additional correlated sensors
            for i in [1, 5, 6, 10, 16, 17, 18, 19, 20, 21]:
                if i <= 10:
                    # Early sensors correlate with temperature/pressure
                    base_val = 50 + 30 * op_setting_1 + 20 * op_setting_2
                    sensors[f'sensor_{i}'] = base_val * degradation_factor + np.random.normal(0, 5)
                else:
                    # Later sensors show more complex degradation patterns
                    base_val = 100 + 50 * op_setting_3
                    sensors[f'sensor_{i}'] = base_val / (degradation_factor**0.5) + np.random.normal(0, 8)
            
            # Create row for this engine cycle
            row = {
                'engine_id': engine_id,
                'cycle': cycle,
                'operational_setting_1': op_setting_1,
                'operational_setting_2': op_setting_2,
                'operational_setting_3': op_setting_3,
                **sensors
            }
            
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Generated data for {CONFIG['n_engines']} engines")
    print(f"Total data points: {len(df):,}")
    print(f"Average cycles per engine: {len(df) / CONFIG['n_engines']:.1f}")
    
    return df

def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine cycle.
    
    RUL represents the number of operational cycles remaining before the engine
    requires major maintenance or replacement. This is the target variable for
    our machine learning models.
    
    Args:
        df: DataFrame with engine operational data
        
    Returns:
        pd.DataFrame: Original data with RUL column added
        
    Engineering Context:
        In aerospace maintenance, RUL prediction enables:
        - Predictive maintenance scheduling
        - Inventory optimization for spare parts
        - Flight safety assurance
        - Cost reduction through optimized maintenance intervals
    """
    print("\n[STAGE 2] Calculating Remaining Useful Life (RUL)...")
    
    df_with_rul = df.copy()
    
    # Calculate RUL for each engine
    rul_values = []
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()
        
        # RUL = max_cycle - current_cycle
        engine_rul = max_cycle - engine_data['cycle']
        rul_values.extend(engine_rul.tolist())
    
    df_with_rul['RUL'] = rul_values
    
    print(f"RUL Statistics:")
    print(f"  Mean RUL: {df_with_rul['RUL'].mean():.1f} cycles")
    print(f"  Max RUL: {df_with_rul['RUL'].max()} cycles")
    print(f"  Min RUL: {df_with_rul['RUL'].min()} cycles")
    print(f"  Std RUL: {df_with_rul['RUL'].std():.1f} cycles")
    
    return df_with_rul

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform comprehensive Exploratory Data Analysis on the engine dataset.
    
    This analysis helps understand:
    - Data distribution and quality
    - Sensor behavior patterns
    - Correlations between variables
    - Degradation trends over engine lifecycle
    
    Args:
        df: DataFrame with engine data including RUL
        
    Engineering Context:
        EDA is critical in aerospace applications to:
        - Identify sensor malfunctions or anomalies
        - Understand normal vs. abnormal operating conditions
        - Validate physics-based expectations
        - Guide feature engineering decisions
    """
    print("\n[STAGE 3] Performing Exploratory Data Analysis...")
    
    # Create comprehensive EDA visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Aircraft Engine Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. RUL Distribution
    axes[0, 0].hist(df['RUL'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('RUL Distribution')
    axes[0, 0].set_xlabel('Remaining Useful Life (cycles)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Engine Lifecycle Patterns
    sample_engines = np.random.choice(df['engine_id'].unique(), 5, replace=False)
    for engine in sample_engines:
        engine_data = df[df['engine_id'] == engine]
        axes[0, 1].plot(engine_data['cycle'], engine_data['RUL'], alpha=0.7, linewidth=2)
    axes[0, 1].set_title('RUL vs Cycle (Sample Engines)')
    axes[0, 1].set_xlabel('Operational Cycle')
    axes[0, 1].set_ylabel('RUL')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sensor Degradation Pattern
    key_sensors = ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_14']
    for sensor in key_sensors:
        # Normalize sensor values for comparison
        normalized_values = (df[sensor] - df[sensor].min()) / (df[sensor].max() - df[sensor].min())
        axes[0, 2].scatter(df['RUL'], normalized_values, alpha=0.1, s=1, label=sensor)
    axes[0, 2].set_title('Sensor Values vs RUL (Normalized)')
    axes[0, 2].set_xlabel('RUL')
    axes[0, 2].set_ylabel('Normalized Sensor Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Operational Settings Distribution
    settings_data = df[['operational_setting_1', 'operational_setting_2', 'operational_setting_3']]
    axes[1, 0].boxplot([settings_data[col] for col in settings_data.columns])
    axes[1, 0].set_title('Operational Settings Distribution')
    axes[1, 0].set_xticklabels(['Altitude', 'Throttle', 'Bypass Ratio'])
    axes[1, 0].set_ylabel('Normalized Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Correlation Heatmap (subset of sensors)
    sensor_cols = [col for col in df.columns if 'sensor' in col][:10]  # First 10 sensors
    corr_matrix = df[sensor_cols + ['RUL']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1, 1], fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[1, 1].set_title('Sensor Correlation Matrix')
    
    # 6. Engine Count by Lifecycle Stage
    df['lifecycle_stage'] = pd.cut(df['RUL'], bins=[0, 50, 100, 200, float('inf')], 
                                  labels=['Critical', 'Warning', 'Caution', 'Healthy'])
    stage_counts = df['lifecycle_stage'].value_counts()
    axes[1, 2].pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%', 
                   colors=['red', 'orange', 'yellow', 'green'])
    axes[1, 2].set_title('Engine Lifecycle Stage Distribution')
    
    # 7. Sensor Variance by RUL Range
    rul_bins = pd.cut(df['RUL'], bins=5)
    sensor_variance = df.groupby(rul_bins)['sensor_2'].var()
    axes[2, 0].bar(range(len(sensor_variance)), sensor_variance.values, color='lightcoral')
    axes[2, 0].set_title('Sensor Variance by RUL Range')
    axes[2, 0].set_xlabel('RUL Range (Low to High)')
    axes[2, 0].set_ylabel('Sensor Variance')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Degradation Rate Analysis
    sample_engine = df[df['engine_id'] == sample_engines[0]]
    axes[2, 1].plot(sample_engine['cycle'], sample_engine['sensor_2'], 'b-', linewidth=2, label='Temperature')
    axes[2, 1].plot(sample_engine['cycle'], sample_engine['sensor_14'], 'r-', linewidth=2, label='Vibration')
    axes[2, 1].set_title(f'Sensor Trends - Engine {sample_engines[0]}')
    axes[2, 1].set_xlabel('Operational Cycle')
    axes[2, 1].set_ylabel('Sensor Reading')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Data Quality Assessment
    missing_data = df.isnull().sum()
    sensor_cols_all = [col for col in df.columns if 'sensor' in col]
    missing_sensors = missing_data[sensor_cols_all]
    axes[2, 2].bar(range(len(missing_sensors)), missing_sensors.values, color='lightblue')
    axes[2, 2].set_title('Data Quality - Missing Values')
    axes[2, 2].set_xlabel('Sensor Index')
    axes[2, 2].set_ylabel('Missing Values Count')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    print("EDA visualization saved as 'eda_analysis.png'")
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"  Total engines: {df['engine_id'].nunique()}")
    print(f"  Total cycles: {len(df):,}")
    print(f"  Average engine lifespan: {df.groupby('engine_id')['cycle'].max().mean():.1f} cycles")
    print(f"  Sensor measurements: {len([col for col in df.columns if 'sensor' in col])}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features for improved RUL prediction accuracy.
    
    Feature engineering is crucial for capturing complex degradation patterns
    that simple sensor readings might miss. This function creates:
    - Rolling statistics to capture trends
    - Rate of change features for degradation velocity
    - Normalized cycle features for lifecycle position
    - Interaction features between operational settings and sensors
    
    Args:
        df: DataFrame with raw engine data
        
    Returns:
        pd.DataFrame: Enhanced dataset with engineered features
        
    Engineering Context:
        Advanced features help capture:
        - Non-linear degradation patterns
        - Sensor drift and calibration issues
        - Operational regime effects on degradation
        - Early warning indicators of impending failure
    """
    print("\n[STAGE 4] Engineering advanced features...")
    
    df_features = df.copy()
    
    # Sort by engine and cycle for proper time series operations
    df_features = df_features.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    
    print(f"Creating features for {len(sensor_cols)} sensors...")
    
    # Initialize lists to store new features
    new_features = []
    
    for engine_id in df_features['engine_id'].unique():
        engine_mask = df_features['engine_id'] == engine_id
        engine_data = df_features[engine_mask].copy()
        
        # 1. Normalized Cycle (position in lifecycle)
        max_cycle = engine_data['cycle'].max()
        engine_data['cycle_norm'] = engine_data['cycle'] / max_cycle
        
        # 2. Rolling Statistics (capture trends over time)
        window_sizes = [5, 10]
        for window in window_sizes:
            for sensor in sensor_cols[:5]:  # Limit to key sensors for performance
                # Rolling mean (trend)
                engine_data[f'{sensor}_rolling_mean_{window}'] = (
                    engine_data[sensor].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation (variability)
                engine_data[f'{sensor}_rolling_std_{window}'] = (
                    engine_data[sensor].rolling(window=window, min_periods=1).std().fillna(0)
                )
        
        # 3. Rate of Change Features (degradation velocity)
        for sensor in sensor_cols[:5]:
            # First difference (immediate rate of change)
            engine_data[f'{sensor}_diff'] = engine_data[sensor].diff().fillna(0)
            
            # Rate of change over 5 cycles
            engine_data[f'{sensor}_rate_5'] = (
                engine_data[sensor].diff(5).fillna(0) / 5
            )
        
        # 4. Cumulative Features (total degradation)
        for sensor in ['sensor_2', 'sensor_7', 'sensor_11', 'sensor_14']:
            engine_data[f'{sensor}_cumsum'] = engine_data[sensor].cumsum()
            engine_data[f'{sensor}_cummax'] = engine_data[sensor].cummax()
            engine_data[f'{sensor}_cummin'] = engine_data[sensor].cummin()
        
        # 5. Interaction Features (operational settings impact)
        key_sensors = ['sensor_2', 'sensor_7', 'sensor_11']
        for sensor in key_sensors:
            engine_data[f'{sensor}_x_throttle'] = (
                engine_data[sensor] * engine_data['operational_setting_2']
            )
            engine_data[f'{sensor}_x_altitude'] = (
                engine_data[sensor] * engine_data['operational_setting_1']
            )
        
        # 6. Statistical Features (distribution characteristics)
        for sensor in key_sensors:
            # Z-score (how far from normal for this engine)
            sensor_mean = engine_data[sensor].mean()
            sensor_std = engine_data[sensor].std()
            if sensor_std > 0:
                engine_data[f'{sensor}_zscore'] = (
                    (engine_data[sensor] - sensor_mean) / sensor_std
                )
            else:
                engine_data[f'{sensor}_zscore'] = 0
        
        # 7. Degradation Indicators
        # Temperature rise indicator
        if len(engine_data) > 10:
            temp_baseline = engine_data['sensor_2'].iloc[:10].mean()
            engine_data['temp_degradation'] = engine_data['sensor_2'] / temp_baseline
        else:
            engine_data['temp_degradation'] = 1.0
        
        # Efficiency decline indicator
        if len(engine_data) > 10:
            efficiency_baseline = engine_data['sensor_11'].iloc[:10].mean()
            engine_data['efficiency_decline'] = efficiency_baseline / engine_data['sensor_11']
        else:
            engine_data['efficiency_decline'] = 1.0
        
        new_features.append(engine_data)
    
    # Combine all engines
    df_engineered = pd.concat(new_features, ignore_index=True)
    
    # Handle any remaining NaN values
    # First handle categorical columns separately
    categorical_cols = df_engineered.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        df_engineered[col] = df_engineered[col].cat.add_categories([0]).fillna(0)
    
    # Handle numeric columns
    numeric_cols = df_engineered.select_dtypes(exclude=['category']).columns
    df_engineered[numeric_cols] = df_engineered[numeric_cols].ffill().fillna(0)
    
    print(f"Feature engineering complete:")
    print(f"  Original features: {len(df.columns)}")
    print(f"  Engineered features: {len(df_engineered.columns)}")
    print(f"  Total features added: {len(df_engineered.columns) - len(df.columns)}")
    
    return df_engineered

# ============================================================================
# 5. DATA PREPROCESSING AND TRAIN/TEST SPLIT
# ============================================================================

def prepare_data_for_modeling(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for machine learning by splitting and scaling.
    
    Critical considerations for time series data:
    - Split by engine ID, not randomly (prevents data leakage)
    - Scale features to prevent bias toward high-magnitude sensors
    - Maintain temporal order within each engine's data
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test, feature_names
        
    Engineering Context:
        Proper data splitting is crucial in aerospace applications:
        - Prevents overfitting to specific engines
        - Ensures model generalizes to unseen engines
        - Maintains realistic evaluation conditions
    """
    print("\n[STAGE 5] Preparing data for machine learning...")
    
    # Identify feature columns (exclude metadata and target)
    exclude_cols = ['engine_id', 'cycle', 'RUL', 'lifecycle_stage']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for modeling")
    
    # Split engines into train/test (80/20 split by engine ID)
    unique_engines = df['engine_id'].unique()
    np.random.shuffle(unique_engines)
    
    n_train_engines = int(0.8 * len(unique_engines))
    train_engines = unique_engines[:n_train_engines]
    test_engines = unique_engines[n_train_engines:]
    
    print(f"Training engines: {len(train_engines)}")
    print(f"Testing engines: {len(test_engines)}")
    
    # Create train/test splits
    train_mask = df['engine_id'].isin(train_engines)
    test_mask = df['engine_id'].isin(test_engines)
    
    X_train = df[train_mask][feature_cols].values
    X_test = df[test_mask][feature_cols].values
    y_train = df[train_mask]['RUL'].values
    y_test = df[test_mask]['RUL'].values
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Target variable range: {y_train.min():.1f} to {y_train.max():.1f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

# ============================================================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Train multiple machine learning models and evaluate their performance.
    
    This function implements three different algorithms:
    1. Random Forest: Ensemble method robust to overfitting
    2. Gradient Boosting: Sequential learning for complex patterns
    3. Linear Regression: Baseline linear model for comparison
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        
    Returns:
        Dict containing trained models and performance metrics
        
    Engineering Context:
        Multiple models provide:
        - Robustness through ensemble predictions
        - Insight into data complexity (linear vs non-linear)
        - Backup options for different deployment scenarios
    """
    print("\n[STAGE 6] Training machine learning models...")
    
    models = {}
    results = {}
    
    # 1. Random Forest Regressor
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'predictions': rf_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred)
    }
    
    # 2. Gradient Boosting Regressor
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {
        'predictions': gb_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'mae': mean_absolute_error(y_test, gb_pred),
        'r2': r2_score(y_test, gb_pred)
    }
    
    # 3. Linear Regression (baseline)
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = {
        'predictions': lr_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'mae': mean_absolute_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred)
    }
    
    # Print performance summary
    print("\nModel Performance Summary:")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.2f} cycles")
        print(f"  MAE:  {metrics['mae']:.2f} cycles")
        print(f"  R²:   {metrics['r2']:.3f}")
        print("-" * 40)
    
    return {
        'models': models,
        'results': results,
        'y_test': y_test
    }

# ============================================================================
# 7. RESULTS VISUALIZATION
# ============================================================================

def visualize_results(model_results: Dict[str, Any], feature_names: List[str]) -> None:
    """
    Create comprehensive visualizations of model performance and insights.
    
    This function generates multiple plots to analyze:
    - Prediction accuracy across models
    - Error distributions and patterns
    - Feature importance rankings
    - Residual analysis for model diagnostics
    
    Args:
        model_results: Dictionary containing models and results
        feature_names: List of feature column names
        
    Engineering Context:
        Visualization is critical for:
        - Model validation and trust building
        - Identifying systematic errors or biases
        - Understanding which factors drive predictions
        - Communicating results to stakeholders
    """
    print("\n[STAGE 7] Creating comprehensive result visualizations...")
    
    models = model_results['models']
    results = model_results['results']
    y_test = model_results['y_test']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Aircraft Engine RUL Prediction - Model Evaluation Results', 
                fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    model_names = list(results.keys())
    
    # 1. Predictions vs Actual (all models)
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        predictions = results[model_name]['predictions']
        axes[0, 0].scatter(y_test, predictions, alpha=0.6, color=color, 
                          label=f'{model_name} (R²={results[model_name]["r2"]:.3f})', s=20)
    
    # Perfect prediction line
    min_val, max_val = min(y_test.min(), min([r['predictions'].min() for r in results.values()])), \
                      max(y_test.max(), max([r['predictions'].max() for r in results.values()]))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Actual RUL')
    axes[0, 0].set_ylabel('Predicted RUL')
    axes[0, 0].set_title('Predictions vs Actual RUL')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Model Performance Comparison
    metrics = ['rmse', 'mae', 'r2']
    metric_values = {metric: [results[model][metric] for model in model_names] for metric in metrics}
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        if metric == 'r2':
            # R² should be maximized (closer to 1 is better)
            axes[0, 1].bar(x_pos + i*width, metric_values[metric], width, 
                          label=metric.upper(), alpha=0.8)
        else:
            # RMSE and MAE should be minimized
            axes[0, 1].bar(x_pos + i*width, metric_values[metric], width, 
                          label=metric.upper(), alpha=0.8)
    
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_title('Model Performance Comparison')
    axes[0, 1].set_xticks(x_pos + width)
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance (Random Forest)
    rf_model = models['Random Forest']
    feature_importance = rf_model.feature_importances_
    
    # Get top 15 most important features
    top_indices = np.argsort(feature_importance)[-15:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    axes[0, 2].barh(range(len(top_features)), top_importance, color='skyblue')
    axes[0, 2].set_yticks(range(len(top_features)))
    axes[0, 2].set_yticklabels(top_features, fontsize=8)
    axes[0, 2].set_xlabel('Feature Importance')
    axes[0, 2].set_title('Top 15 Feature Importance (Random Forest)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residual Analysis (Random Forest)
    rf_predictions = results['Random Forest']['predictions']
    residuals = y_test - rf_predictions
    
    axes[1, 0].scatter(rf_predictions, residuals, alpha=0.6, color='blue', s=20)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted RUL')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot (Random Forest)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Error Distribution
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        predictions = results[model_name]['predictions']
        errors = y_test - predictions
        axes[1, 1].hist(errors, bins=30, alpha=0.6, color=color, label=model_name, density=True)
    
    axes[1, 1].set_xlabel('Prediction Error (cycles)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Prediction Confidence Intervals (Random Forest)
    # Use prediction intervals from individual trees
    rf_model = models['Random Forest']
    tree_predictions = np.array([tree.predict(model_results.get('X_test', np.random.randn(100, len(feature_names)))) 
                                for tree in rf_model.estimators_[:10]])  # Use first 10 trees for speed
    
    if len(tree_predictions) > 0:
        pred_mean = np.mean(tree_predictions, axis=0)
        pred_std = np.std(tree_predictions, axis=0)
        
        # Sort by actual values for better visualization
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test[sort_idx]
        pred_mean_sorted = pred_mean[sort_idx] if len(pred_mean) == len(y_test) else rf_predictions[sort_idx]
        pred_std_sorted = pred_std[sort_idx] if len(pred_std) == len(y_test) else np.ones_like(y_test_sorted) * 10
        
        axes[1, 2].plot(y_test_sorted, label='Actual', color='blue', linewidth=2)
        axes[1, 2].plot(pred_mean_sorted, label='Predicted', color='red', linewidth=2)
        axes[1, 2].fill_between(range(len(y_test_sorted)), 
                               pred_mean_sorted - 1.96*pred_std_sorted,
                               pred_mean_sorted + 1.96*pred_std_sorted,
                               alpha=0.3, color='red', label='95% CI')
    else:
        # Fallback visualization
        sort_idx = np.argsort(y_test)
        axes[1, 2].plot(y_test[sort_idx], label='Actual', color='blue', linewidth=2)
        axes[1, 2].plot(rf_predictions[sort_idx], label='Predicted', color='red', linewidth=2)
    
    axes[1, 2].set_xlabel('Sample Index (sorted by actual RUL)')
    axes[1, 2].set_ylabel('RUL (cycles)')
    axes[1, 2].set_title('Prediction Confidence (Random Forest)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Learning Curves (Gradient Boosting)
    gb_model = models['Gradient Boosting']
    train_scores = gb_model.train_score_
    
    axes[2, 0].plot(train_scores, color='blue', linewidth=2, label='Training Score')
    axes[2, 0].set_xlabel('Boosting Iterations')
    axes[2, 0].set_ylabel('Training Score')
    axes[2, 0].set_title('Gradient Boosting Learning Curve')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Prediction Error by RUL Range
    rf_predictions = results['Random Forest']['predictions']
    errors = np.abs(y_test - rf_predictions)
    
    # Bin by RUL ranges
    rul_bins = pd.cut(y_test, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    error_by_bin = pd.DataFrame({'RUL_Range': rul_bins, 'Error': errors}).groupby('RUL_Range')['Error'].mean()
    
    axes[2, 1].bar(range(len(error_by_bin)), error_by_bin.values, color='lightcoral')
    axes[2, 1].set_xticks(range(len(error_by_bin)))
    axes[2, 1].set_xticklabels(error_by_bin.index, rotation=45)
    axes[2, 1].set_xlabel('RUL Range')
    axes[2, 1].set_ylabel('Mean Absolute Error')
    axes[2, 1].set_title('Prediction Error by RUL Range')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Model Complexity vs Performance
    model_complexity = {
        'Linear Regression': 1,
        'Random Forest': 100,  # n_estimators
        'Gradient Boosting': 100  # n_estimators
    }
    
    r2_scores = [results[model]['r2'] for model in model_names]
    complexity_scores = [model_complexity[model] for model in model_names]
    
    axes[2, 2].scatter(complexity_scores, r2_scores, s=100, c=colors, alpha=0.7)
    for i, model in enumerate(model_names):
        axes[2, 2].annotate(model, (complexity_scores[i], r2_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[2, 2].set_xlabel('Model Complexity (# estimators)')
    axes[2, 2].set_ylabel('R² Score')
    axes[2, 2].set_title('Model Complexity vs Performance')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("Model evaluation visualization saved as 'model_evaluation_results.png'")

# ============================================================================
# 8. PRODUCTION-READY PREDICTION PIPELINE
# ============================================================================

class RULPredictionPipeline:
    """
    Production-ready pipeline for aircraft engine RUL prediction.
    
    This class encapsulates the entire prediction workflow including:
    - Data preprocessing and feature engineering
    - Model ensemble for robust predictions
    - Confidence interval estimation
    - Maintenance recommendations based on predictions
    
    Attributes:
        models: Dictionary of trained ML models
        scaler: Fitted StandardScaler for feature normalization
        feature_names: List of feature column names
        is_fitted: Boolean indicating if pipeline is trained
        
    Engineering Context:
        Production pipelines in aerospace must be:
        - Robust to sensor failures and missing data
        - Provide uncertainty quantification
        - Generate actionable maintenance recommendations
        - Maintain audit trails for regulatory compliance
    """
    
    def __init__(self):
        """Initialize the RUL prediction pipeline."""
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
        self.model_weights = {}  # For ensemble predictions
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> None:
        """
        Train the prediction pipeline on historical engine data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector (RUL values)
            feature_names: List of feature column names
        """
        print("\n[PIPELINE] Training production prediction pipeline...")
        
        self.feature_names = feature_names
        
        # Fit scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train ensemble of models
        print("Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=150, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        self.models['rf'].fit(X_train_scaled, y_train)
        
        print("Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
        self.models['gb'].fit(X_train_scaled, y_train)
        
        # Calculate model weights based on cross-validation performance
        rf_cv_score = np.mean(cross_val_score(self.models['rf'], X_train_scaled, y_train, cv=5, scoring='r2'))
        gb_cv_score = np.mean(cross_val_score(self.models['gb'], X_train_scaled, y_train, cv=5, scoring='r2'))
        
        total_score = rf_cv_score + gb_cv_score
        self.model_weights = {
            'rf': rf_cv_score / total_score,
            'gb': gb_cv_score / total_score
        }
        
        self.is_fitted = True
        print(f"Pipeline training complete. Model weights: RF={self.model_weights['rf']:.3f}, GB={self.model_weights['gb']:.3f}")
    
    def predict(self, X: np.ndarray, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Make RUL predictions with confidence intervals and maintenance recommendations.
        
        Args:
            X: Feature matrix for prediction
            return_confidence: Whether to calculate confidence intervals
            
        Returns:
            Dictionary containing predictions, confidence intervals, and recommendations
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        rf_pred = self.models['rf'].predict(X_scaled)
        gb_pred = self.models['gb'].predict(X_scaled)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = (self.model_weights['rf'] * rf_pred + 
                        self.model_weights['gb'] * gb_pred)
        
        results = {
            'rul_prediction': ensemble_pred,
            'individual_predictions': {
                'random_forest': rf_pred,
                'gradient_boosting': gb_pred
            }
        }
        
        if return_confidence:
            # Calculate confidence intervals using prediction variance
            pred_variance = np.var([rf_pred, gb_pred], axis=0)
            confidence_interval = 1.96 * np.sqrt(pred_variance)  # 95% CI
            
            results['confidence_interval'] = confidence_interval
            results['lower_bound'] = ensemble_pred - confidence_interval
            results['upper_bound'] = ensemble_pred + confidence_interval
        
        # Generate maintenance recommendations
        recommendations = []
        for i, rul in enumerate(ensemble_pred):
            if rul <= 30:
                recommendations.append("CRITICAL: Immediate maintenance required")
            elif rul <= 60:
                recommendations.append("WARNING: Schedule maintenance within 2 weeks")
            elif rul <= 100:
                recommendations.append("CAUTION: Plan maintenance within 1 month")
            else:
                recommendations.append("HEALTHY: Continue normal operations")
        
        results['maintenance_recommendations'] = recommendations
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the Random Forest model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before getting feature importance")
        
        importance_scores = self.models['rf'].feature_importances_
        return dict(zip(self.feature_names, importance_scores))

# ============================================================================
# 9. PIPELINE DEMONSTRATION
# ============================================================================

def demonstrate_prediction_pipeline(pipeline: RULPredictionPipeline, 
                                  sample_data: pd.DataFrame) -> None:
    """
    Demonstrate the production pipeline with sample predictions.
    
    This function shows how the pipeline would be used in a real aerospace
    maintenance environment, including confidence intervals and actionable
    maintenance recommendations.
    
    Args:
        pipeline: Fitted RUL prediction pipeline
        sample_data: Sample engine data for demonstration
    """
    print("\n[STAGE 8] Demonstrating production prediction pipeline...")
    
    # Select 5 random samples for demonstration
    sample_indices = np.random.choice(len(sample_data), 5, replace=False)
    demo_samples = sample_data.iloc[sample_indices]
    
    # Prepare features (exclude metadata columns)
    exclude_cols = ['engine_id', 'cycle', 'RUL', 'lifecycle_stage']
    feature_cols = [col for col in sample_data.columns if col not in exclude_cols]
    X_demo = demo_samples[feature_cols].values
    
    # Make predictions
    predictions = pipeline.predict(X_demo, return_confidence=True)
    
    print("\nSample Predictions with Confidence Intervals:")
    print("="*80)
    
    for i in range(len(demo_samples)):
        engine_id = demo_samples.iloc[i]['engine_id']
        cycle = demo_samples.iloc[i]['cycle']
        actual_rul = demo_samples.iloc[i]['RUL']
        
        predicted_rul = predictions['rul_prediction'][i]
        lower_bound = predictions['lower_bound'][i]
        upper_bound = predictions['upper_bound'][i]
        recommendation = predictions['maintenance_recommendations'][i]
        
        print(f"Engine {engine_id} (Cycle {cycle}):")
        print(f"  Actual RUL:     {actual_rul:.1f} cycles")
        print(f"  Predicted RUL:  {predicted_rul:.1f} cycles")
        print(f"  95% CI:         [{lower_bound:.1f}, {upper_bound:.1f}] cycles")
        print(f"  Error:          {abs(actual_rul - predicted_rul):.1f} cycles")
        print(f"  Recommendation: {recommendation}")
        print("-" * 60)
    
    # Show feature importance
    feature_importance = pipeline.get_feature_importance()
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 Most Important Features:")
    print("="*50)
    for feature, importance in top_features:
        print(f"{feature:<30} {importance:.4f}")

# ============================================================================
# 10. MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function that orchestrates the entire RUL prediction workflow.
    
    This function demonstrates a complete machine learning pipeline for aerospace
    applications, from data generation through production deployment.
    """
    start_time = datetime.now()
    
    try:
        # Stage 1: Generate synthetic data
        df_raw = generate_synthetic_engine_data()
        
        # Stage 2: Calculate RUL
        df_with_rul = calculate_rul(df_raw)
        
        # Stage 3: Exploratory Data Analysis
        perform_eda(df_with_rul)
        
        # Stage 4: Feature Engineering
        df_engineered = engineer_features(df_with_rul)
        
        # Stage 5: Data Preparation
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_modeling(df_engineered)
        
        # Stage 6: Model Training and Evaluation
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        model_results['X_test'] = X_test  # Add for visualization
        
        # Stage 7: Results Visualization
        visualize_results(model_results, feature_names)
        
        # Stage 8: Production Pipeline
        pipeline = RULPredictionPipeline()
        pipeline.fit(X_train, y_train, feature_names)
        
        # Stage 9: Pipeline Demonstration
        demonstrate_prediction_pipeline(pipeline, df_engineered)
        
        # Final Summary
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("AIRCRAFT ENGINE RUL PREDICTION SYSTEM - EXECUTION COMPLETE")
        print("="*80)
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Dataset size: {len(df_engineered):,} data points")
        print(f"Number of engines: {df_engineered['engine_id'].nunique()}")
        print(f"Features engineered: {len(feature_names)}")
        
        print("\nModel Performance Summary:")
        best_model = min(model_results['results'].items(), key=lambda x: x[1]['rmse'])
        print(f"Best performing model: {best_model[0]}")
        print(f"  RMSE: {best_model[1]['rmse']:.2f} cycles")
        print(f"  MAE:  {best_model[1]['mae']:.2f} cycles")
        print(f"  R²:   {best_model[1]['r2']:.3f}")
        
        print("\nFiles Generated:")
        print("  - eda_analysis.png: Exploratory data analysis visualizations")
        print("  - model_evaluation_results.png: Model performance analysis")
        
        print("\nNext Steps for Production Deployment:")
        print("  1. Integrate with real-time sensor data streams")
        print("  2. Implement automated model retraining pipeline")
        print("  3. Set up monitoring and alerting systems")
        print("  4. Conduct extensive validation with historical maintenance records")
        print("  5. Obtain regulatory approval for safety-critical applications")
        
        print("\nThis system is ready for:")
        print("  ✓ Aerospace industry interviews")
        print("  ✓ Academic research projects")
        print("  ✓ Production deployment (with additional validation)")
        print("  ✓ Integration with existing maintenance systems")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check the error message and try again.")
        raise

if __name__ == "__main__":
    main()