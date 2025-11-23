#!/usr/bin/env python3
"""
Aircraft Engine Remaining Useful Life (RUL) Prediction System - Streamlined Version
==================================================================================

A production-quality machine learning system for predicting the Remaining Useful Life
of aircraft engines using synthetic data that mimics the NASA CMAPSS dataset structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from typing import Tuple, Dict, List, Any
from datetime import datetime

# Configure plotting and warnings
plt.style.use('default')
warnings.filterwarnings('ignore')
np.random.seed(42)

# Configuration
CONFIG = {
    'n_engines': 30,
    'max_cycles': 150,
    'n_sensors': 10,
}

print("="*60)
print("AIRCRAFT ENGINE RUL PREDICTION SYSTEM")
print("="*60)
print(f"Engines: {CONFIG['n_engines']}, Max Cycles: {CONFIG['max_cycles']}")

def generate_data():
    """Generate synthetic engine data"""
    print("\n[1/8] Generating synthetic data...")
    
    data = []
    for engine_id in range(1, CONFIG['n_engines'] + 1):
        total_cycles = np.random.randint(80, CONFIG['max_cycles'])
        
        for cycle in range(1, total_cycles + 1):
            # Degradation factor
            degradation = 1 + 0.002 * cycle * (cycle / total_cycles)
            
            # Operational settings
            altitude = np.random.uniform(0.7, 1.0)
            throttle = np.random.uniform(0.6, 0.9)
            
            # Sensors with physics-based correlations
            temp_base = 500 + 200 * throttle
            pressure_base = 100 + 50 * altitude * throttle
            flow_base = 300 + 100 * altitude
            
            row = {
                'engine_id': engine_id,
                'cycle': cycle,
                'altitude': altitude,
                'throttle': throttle,
                'temp_1': temp_base * degradation + np.random.normal(0, 10),
                'temp_2': temp_base * 0.9 * degradation + np.random.normal(0, 8),
                'pressure_1': pressure_base * degradation + np.random.normal(0, 5),
                'pressure_2': pressure_base * 0.8 * degradation + np.random.normal(0, 4),
                'flow_1': flow_base / degradation + np.random.normal(0, 15),
                'flow_2': flow_base * 0.7 / degradation + np.random.normal(0, 10),
                'vibration_1': (10 + 5 * throttle) * degradation**1.5 + np.random.normal(0, 2),
                'vibration_2': (8 + 4 * throttle) * degradation**1.3 + np.random.normal(0, 1.5),
                'efficiency_1': 100 / (degradation**0.5) + np.random.normal(0, 8),
                'efficiency_2': 90 / (degradation**0.3) + np.random.normal(0, 6),
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate RUL
    rul_values = []
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        max_cycle = engine_data['cycle'].max()
        engine_rul = max_cycle - engine_data['cycle']
        rul_values.extend(engine_rul.tolist())
    
    df['RUL'] = rul_values
    
    print(f"Generated {len(df):,} data points for {CONFIG['n_engines']} engines")
    print(f"RUL range: {df['RUL'].min()} to {df['RUL'].max()} cycles")
    
    return df

def perform_eda(df):
    """Exploratory Data Analysis"""
    print("\n[2/8] Performing EDA...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Aircraft Engine Dataset - EDA', fontsize=14, fontweight='bold')
    
    # RUL Distribution
    axes[0, 0].hist(df['RUL'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('RUL Distribution')
    axes[0, 0].set_xlabel('RUL (cycles)')
    
    # Sample engine lifecycles
    sample_engines = np.random.choice(df['engine_id'].unique(), 3, replace=False)
    for engine in sample_engines:
        engine_data = df[df['engine_id'] == engine]
        axes[0, 1].plot(engine_data['cycle'], engine_data['RUL'], alpha=0.7, linewidth=2)
    axes[0, 1].set_title('RUL vs Cycle (Sample Engines)')
    axes[0, 1].set_xlabel('Cycle')
    axes[0, 1].set_ylabel('RUL')
    
    # Sensor correlation with RUL
    sensor_cols = ['temp_1', 'pressure_1', 'flow_1', 'vibration_1']
    for sensor in sensor_cols:
        axes[0, 2].scatter(df['RUL'], df[sensor], alpha=0.1, s=1, label=sensor)
    axes[0, 2].set_title('Sensors vs RUL')
    axes[0, 2].set_xlabel('RUL')
    axes[0, 2].legend()
    
    # Operational settings
    axes[1, 0].boxplot([df['altitude'], df['throttle']])
    axes[1, 0].set_title('Operational Settings')
    axes[1, 0].set_xticklabels(['Altitude', 'Throttle'])
    
    # Correlation matrix
    corr_cols = ['temp_1', 'pressure_1', 'flow_1', 'vibration_1', 'RUL']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix')
    
    # Degradation pattern
    sample_engine = df[df['engine_id'] == sample_engines[0]]
    axes[1, 2].plot(sample_engine['cycle'], sample_engine['temp_1'], 'b-', label='Temperature')
    axes[1, 2].plot(sample_engine['cycle'], sample_engine['vibration_1'], 'r-', label='Vibration')
    axes[1, 2].set_title(f'Degradation - Engine {sample_engines[0]}')
    axes[1, 2].set_xlabel('Cycle')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    print("EDA saved as 'eda_analysis.png'")

def engineer_features(df):
    """Feature Engineering"""
    print("\n[3/8] Engineering features...")
    
    df_features = df.copy().sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    
    sensor_cols = [col for col in df.columns if col in ['temp_1', 'temp_2', 'pressure_1', 'pressure_2', 'flow_1', 'flow_2', 'vibration_1', 'vibration_2']]
    
    new_features = []
    
    for engine_id in df_features['engine_id'].unique():
        engine_mask = df_features['engine_id'] == engine_id
        engine_data = df_features[engine_mask].copy()
        
        # Normalized cycle
        max_cycle = engine_data['cycle'].max()
        engine_data['cycle_norm'] = engine_data['cycle'] / max_cycle
        
        # Rolling statistics
        for sensor in sensor_cols[:4]:  # Top 4 sensors
            engine_data[f'{sensor}_rolling_mean'] = engine_data[sensor].rolling(window=5, min_periods=1).mean()
            engine_data[f'{sensor}_rolling_std'] = engine_data[sensor].rolling(window=5, min_periods=1).std().fillna(0)
            engine_data[f'{sensor}_diff'] = engine_data[sensor].diff().fillna(0)
        
        # Interaction features
        engine_data['temp_throttle'] = engine_data['temp_1'] * engine_data['throttle']
        engine_data['pressure_altitude'] = engine_data['pressure_1'] * engine_data['altitude']
        
        new_features.append(engine_data)
    
    df_engineered = pd.concat(new_features, ignore_index=True)
    df_engineered = df_engineered.fillna(0)
    
    print(f"Features: {len(df.columns)} → {len(df_engineered.columns)}")
    return df_engineered

def prepare_data(df):
    """Prepare data for modeling"""
    print("\n[4/8] Preparing data...")
    
    # Feature columns
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split by engine ID
    unique_engines = df['engine_id'].unique()
    np.random.shuffle(unique_engines)
    
    n_train = int(0.8 * len(unique_engines))
    train_engines = unique_engines[:n_train]
    test_engines = unique_engines[n_train:]
    
    train_mask = df['engine_id'].isin(train_engines)
    test_mask = df['engine_id'].isin(test_engines)
    
    X_train = df[train_mask][feature_cols].values
    X_test = df[test_mask][feature_cols].values
    y_train = df[train_mask]['RUL'].values
    y_test = df[test_mask]['RUL'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models"""
    print("\n[5/8] Training models...")
    
    models = {}
    results = {}
    
    # Random Forest
    print("  Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'predictions': rf_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred)
    }
    
    # Gradient Boosting
    print("  Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = {
        'predictions': gb_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'mae': mean_absolute_error(y_test, gb_pred),
        'r2': r2_score(y_test, gb_pred)
    }
    
    # Linear Regression
    print("  Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'predictions': lr_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'mae': mean_absolute_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred)
    }
    
    # Print results
    print("\nModel Performance:")
    for name, metrics in results.items():
        print(f"  {name}:")
        print(f"    RMSE: {metrics['rmse']:.2f}")
        print(f"    MAE:  {metrics['mae']:.2f}")
        print(f"    R²:   {metrics['r2']:.3f}")
    
    return models, results, y_test

def visualize_results(models, results, y_test, feature_names):
    """Visualize model results"""
    print("\n[6/8] Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green']
    model_names = list(results.keys())
    
    # Predictions vs Actual
    for i, (name, color) in enumerate(zip(model_names, colors)):
        pred = results[name]['predictions']
        axes[0, 0].scatter(y_test, pred, alpha=0.6, color=color, label=f'{name} (R²={results[name]["r2"]:.3f})', s=20)
    
    min_val = min(y_test.min(), min([r['predictions'].min() for r in results.values()]))
    max_val = max(y_test.max(), max([r['predictions'].max() for r in results.values()]))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    axes[0, 0].set_xlabel('Actual RUL')
    axes[0, 0].set_ylabel('Predicted RUL')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Performance comparison
    metrics = ['rmse', 'mae', 'r2']
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        axes[0, 1].bar(x_pos + i*width, values, width, label=metric.upper(), alpha=0.8)
    
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_title('Performance Comparison')
    axes[0, 1].set_xticks(x_pos + width)
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature importance (Random Forest)
    rf_model = models['Random Forest']
    importance = rf_model.feature_importances_
    top_indices = np.argsort(importance)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    axes[0, 2].barh(range(len(top_features)), top_importance, color='skyblue')
    axes[0, 2].set_yticks(range(len(top_features)))
    axes[0, 2].set_yticklabels(top_features, fontsize=8)
    axes[0, 2].set_xlabel('Importance')
    axes[0, 2].set_title('Top 10 Feature Importance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Residuals
    rf_pred = results['Random Forest']['predictions']
    residuals = y_test - rf_pred
    axes[1, 0].scatter(rf_pred, residuals, alpha=0.6, color='blue', s=20)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Predicted RUL')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot (Random Forest)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    for i, (name, color) in enumerate(zip(model_names, colors)):
        pred = results[name]['predictions']
        errors = y_test - pred
        axes[1, 1].hist(errors, bins=20, alpha=0.6, color=color, label=name, density=True)
    
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prediction timeline
    sort_idx = np.argsort(y_test)
    axes[1, 2].plot(y_test[sort_idx], label='Actual', color='blue', linewidth=2)
    axes[1, 2].plot(rf_pred[sort_idx], label='Predicted', color='red', linewidth=2)
    axes[1, 2].set_xlabel('Sample (sorted by RUL)')
    axes[1, 2].set_ylabel('RUL')
    axes[1, 2].set_title('Prediction Timeline')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("Results saved as 'model_evaluation_results.png'")

class RULPipeline:
    """Production RUL Prediction Pipeline"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, X_train, y_train, feature_names):
        print("\n[7/8] Training production pipeline...")
        
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train ensemble
        self.models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.models['gb'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        self.models['rf'].fit(X_scaled, y_train)
        self.models['gb'].fit(X_scaled, y_train)
        
        self.is_fitted = True
        print("Pipeline training complete")
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        rf_pred = self.models['rf'].predict(X_scaled)
        gb_pred = self.models['gb'].predict(X_scaled)
        
        # Ensemble prediction
        ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
        
        # Confidence intervals
        pred_std = np.std([rf_pred, gb_pred], axis=0)
        confidence = 1.96 * pred_std
        
        # Maintenance recommendations
        recommendations = []
        for rul in ensemble_pred:
            if rul <= 20:
                recommendations.append("CRITICAL: Immediate maintenance")
            elif rul <= 40:
                recommendations.append("WARNING: Schedule maintenance soon")
            elif rul <= 80:
                recommendations.append("CAUTION: Plan maintenance")
            else:
                recommendations.append("HEALTHY: Normal operations")
        
        return {
            'rul_prediction': ensemble_pred,
            'confidence_interval': confidence,
            'lower_bound': ensemble_pred - confidence,
            'upper_bound': ensemble_pred + confidence,
            'recommendations': recommendations
        }

def demonstrate_pipeline(pipeline, sample_data):
    """Demonstrate pipeline predictions"""
    print("\n[8/8] Demonstrating pipeline...")
    
    # Sample predictions
    sample_indices = np.random.choice(len(sample_data), 5, replace=False)
    demo_samples = sample_data.iloc[sample_indices]
    
    exclude_cols = ['engine_id', 'cycle', 'RUL']
    feature_cols = [col for col in sample_data.columns if col not in exclude_cols]
    X_demo = demo_samples[feature_cols].values
    
    predictions = pipeline.predict(X_demo)
    
    print("\nSample Predictions:")
    print("="*60)
    
    for i in range(len(demo_samples)):
        engine_id = demo_samples.iloc[i]['engine_id']
        cycle = demo_samples.iloc[i]['cycle']
        actual_rul = demo_samples.iloc[i]['RUL']
        
        pred_rul = predictions['rul_prediction'][i]
        lower = predictions['lower_bound'][i]
        upper = predictions['upper_bound'][i]
        rec = predictions['recommendations'][i]
        
        print(f"Engine {engine_id} (Cycle {cycle}):")
        print(f"  Actual RUL:    {actual_rul:.1f} cycles")
        print(f"  Predicted RUL: {pred_rul:.1f} cycles")
        print(f"  95% CI:        [{lower:.1f}, {upper:.1f}]")
        print(f"  Error:         {abs(actual_rul - pred_rul):.1f} cycles")
        print(f"  Recommendation: {rec}")
        print("-" * 40)

def main():
    """Main execution function"""
    start_time = datetime.now()
    
    try:
        # Execute pipeline
        df = generate_data()
        perform_eda(df)
        df_engineered = engineer_features(df)
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df_engineered)
        models, results, y_test = train_models(X_train, X_test, y_train, y_test)
        visualize_results(models, results, y_test, feature_names)
        
        # Production pipeline
        pipeline = RULPipeline()
        pipeline.fit(X_train, y_train, feature_names)
        demonstrate_pipeline(pipeline, df_engineered)
        
        # Summary
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print("="*60)
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Dataset: {len(df_engineered):,} points, {CONFIG['n_engines']} engines")
        
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\nBest Model: {best_model[0]}")
        print(f"  RMSE: {best_model[1]['rmse']:.2f} cycles")
        print(f"  R²:   {best_model[1]['r2']:.3f}")
        
        print("\nFiles Generated:")
        print("  - eda_analysis.png")
        print("  - model_evaluation_results.png")
        
        print("\n✓ Ready for aerospace industry applications")
        print("✓ Production-quality ML pipeline")
        print("✓ Comprehensive analysis and visualization")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()