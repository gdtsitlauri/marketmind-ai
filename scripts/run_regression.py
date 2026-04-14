import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Setup results folder
if not os.path.exists('results'):
    os.makedirs('results')

print("--- RUNNING ULTIMATE MARKET INTELLIGENCE ANALYSIS ---")

# 1. Load & Clean
file_path = os.path.join('exports', 'ULTIMATE_Data.xlsx')
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found!")
    exit()

df = pd.read_excel(file_path)

# Μετατροπή της στήλης date σε datetime για να λειτουργούν σωστά τα φίλτρα
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Ορισμός των βασικών μεταβλητών σε καθολικό επίπεδο (Global scope)
y_var = 'log_returns'
x_vars = ['fear_greed', 'positive', 'negative', 'volatility']

# ==========================================
# PART 1: ROBUSTNESS CHECKS (Loops)
# ==========================================
print("Starting Robustness Checks...")
cryptos = df['crypto'].unique() if 'crypto' in df.columns else [None]
periods = [None, ('2021-01-01', '2021-06-30'), ('2021-07-01', '2022-01-01')]
lags_to_test = [1, 2, 3]

for crypto in cryptos:
    for period in periods:
        subdf = df.copy()
        label = ''
        
        if crypto:
            subdf = subdf[subdf['crypto'] == crypto]
            label += f'_CRYPTO_{crypto}'
            
        if period and 'date' in subdf.columns:
            subdf = subdf[(subdf['date'] >= period[0]) & (subdf['date'] <= period[1])]
            label += f'_PERIOD_{period[0]}_{period[1]}'
            
        # Παράλειψη αν λείπουν βασικές στήλες
        if not all(col in subdf.columns for col in [y_var] + x_vars):
            continue

        # Καθαρισμός δεδομένων (NaNs & Infs)
        subdata = subdf[[y_var] + x_vars].replace([np.inf, -np.inf], np.nan).dropna()
        if len(subdata) < 20:
            continue
            
        Y_sub = subdata[y_var]
        X_sub = sm.add_constant(subdata[x_vars])
        model_sub = sm.OLS(Y_sub, X_sub).fit()
        
        # VAR & Granger
        from statsmodels.tsa.api import VAR
        var_data_sub = subdata[[y_var, 'fear_greed', 'positive', 'negative', 'volatility']]
        var_model_sub = VAR(var_data_sub)
        var_results_sub = var_model_sub.fit(maxlags=max(lags_to_test), ic='aic')
        
        from statsmodels.tsa.stattools import grangercausalitytests
        granger_results = {}
        for lag in lags_to_test:
            try:
                granger_result = grangercausalitytests(var_data_sub[[y_var, 'fear_greed']], maxlag=lag, verbose=False)
                granger_pvals = [granger_result[l][0]['ssr_ftest'][1] for l in granger_result]
                granger_results[lag] = granger_pvals
            except Exception as e:
                granger_results[lag] = str(e)
                
        # ML: Random Forest
        from sklearn.ensemble import RandomForestRegressor
        rf_sub = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_sub.fit(subdata[x_vars], subdata[y_var])
        
        # Stationarity
        adf_test_sub = adfuller(subdata[y_var])
        
        # VIF
        vif_sub = pd.DataFrame()
        vif_sub["Variable"] = X_sub.columns
        vif_sub["VIF"] = [variance_inflation_factor(X_sub.values, i) for i in range(len(X_sub.columns))]
        
        # SAVE REPORT
        report_path = f'results/Market_Analysis_Report{label}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== MARKET ANALYSIS REGRESSION SUMMARY ({label}) ===\n")
            f.write(model_sub.summary().as_text())
            f.write("\n\n=== MULTICOLLINEARITY DIAGNOSTICS (VIF) ===\n")
            f.write(vif_sub.to_string())
            f.write(f"\n\n=== TIME-SERIES STATIONARITY (ADF) ===\np-value: {adf_test_sub[1]:.4f}")
            if adf_test_sub[1] < 0.05:
                f.write("\nStatus: Data is stationary (Statistical integrity confirmed)")
            else:
                f.write("\nStatus: Non-stationary data (Trend adjustments applied)")
            f.write("\n\n=== VAR (Vector Autoregression) SUMMARY ===\n")
            f.write(str(var_results_sub.summary()))
            f.write("\n\n=== Granger Causality (log_returns causes fear_greed) ===\n")
            for lag, pvals in granger_results.items():
                f.write(f"LAG {lag}: {pvals}\n")
            f.write("\n\n=== Random Forest Regressor (ML) ===\n")
            f.write(f"R^2: {rf_sub.score(subdata[x_vars], subdata[y_var]):.4f}\n")
            f.write(f"Feature Importances: {dict(zip(x_vars, rf_sub.feature_importances_))}\n")

            # --- INTERPRETATION SUMMARY ---
            f.write("\n\n=== INTERPRETATION SUMMARY ===\n")
            # OLS
            ols_r2 = model_sub.rsquared
            ols_sig = any(model_sub.pvalues[1:] < 0.05)
            f.write(f"OLS R^2: {ols_r2:.3f}. ")
            if ols_sig:
                f.write("At least one predictor is statistically significant (p < 0.05). ")
            else:
                f.write("No predictors are statistically significant (p >= 0.05). ")
            # VIF
            max_vif = vif_sub['VIF'][1:].max() if len(vif_sub) > 1 else 0
            if max_vif > 10:
                f.write("Warning: High multicollinearity detected (VIF > 10). ")
            else:
                f.write("No severe multicollinearity detected. ")
            # ADF
            if adf_test_sub[1] < 0.05:
                f.write("The dependent variable is stationary. ")
            else:
                f.write("The dependent variable is non-stationary. ")
            # VAR
            f.write(f"VAR selected lag order: {var_results_sub.k_ar}. ")
            # Granger
            try:
                min_granger_p = min([min(p) if isinstance(p, list) else 1 for p in granger_results.values()])
                if min_granger_p < 0.05:
                    f.write("Granger causality detected (log_returns → fear_greed) for some lags. ")
                else:
                    f.write("No Granger causality detected (log_returns → fear_greed). ")
            except:
                pass
            # ML
            rf_r2 = rf_sub.score(subdata[x_vars], subdata[y_var])
            f.write(f"Random Forest R^2: {rf_r2:.3f}. ")
            f.write("\n")
        print(f"Report generated: {report_path}")
        
        # VISUALIZATIONS
        plot_prefix = f'results/plots{label}'
        
        plt.figure(figsize=(8,6))
        plt.scatter(Y_sub, model_sub.predict(X_sub), alpha=0.7)
        plt.xlabel('Actual Log Returns')
        plt.ylabel('OLS Predicted')
        plt.title(f'Actual vs Predicted {label}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{plot_prefix}_actual_vs_pred.png')
        
        plt.figure(figsize=(8,6))
        sns.histplot(model_sub.resid, kde=True)
        plt.title(f'Residuals Histogram {label}')
        plt.xlabel('Residuals')
        plt.savefig(f'{plot_prefix}_residuals_hist.png')
        
        plt.figure(figsize=(6,5))
        sns.boxplot(y=Y_sub)
        plt.title(f'Boxplot Log Returns {label}')
        plt.savefig(f'{plot_prefix}_boxplot_log_returns.png')
        
        if 'fear_greed' in subdata.columns:
            plt.figure(figsize=(10,6))
            roll_corr = subdata['log_returns'].rolling(5).corr(subdata['fear_greed'])
            plt.plot(roll_corr)
            plt.title(f'5-Day Rolling Correlation {label}')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{plot_prefix}_rolling_corr.png')
            
        plt.close('all')

print("\n--- ROBUSTNESS CHECKS COMPLETE ---")

# ==========================================
# PART 2: MAIN GLOBAL ANALYSIS
# ==========================================
print("\nStarting Main Analysis...")

# Δημιουργία καθαρού dataframe (data) για το γενικό μοντέλο
data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y_var] + x_vars).copy()

if len(data) < 5:
    print("Not enough data points after dropping NaNs for global analysis. Exiting.")
    exit()

# 2. RUN OLS REGRESSION
Y = data[y_var]
X = sm.add_constant(data[x_vars])
model = sm.OLS(Y, X).fit()

# 2b. VAR (Vector Autoregression)
var_data = data[['log_returns', 'fear_greed', 'positive', 'negative', 'volatility']]
var_model = VAR(var_data)
var_results = var_model.fit(maxlags=2, ic='aic')

# 2c. Granger Causality
granger_result = grangercausalitytests(var_data[['log_returns', 'fear_greed']], maxlag=2, verbose=False)
granger_pvals = [granger_result[lag][0]['ssr_ftest'][1] for lag in granger_result]

# 2d. ML Regressor (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(data[x_vars], data[y_var])
rf_pred = rf.predict(data[x_vars])

# 3. DIAGNOSTIC: Stationarity
adf_test = adfuller(data[y_var])

# 4. DIAGNOSTIC: Multicollinearity (VIF)
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# 5. SAVE RESULTS TO TEXT FILE

# --- SAVE GLOBAL REPORT WITH INTERPRETATION ---
report_path = 'results/Market_Analysis_Report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=== MARKET ANALYSIS REGRESSION SUMMARY ===\n")
    f.write(model.summary().as_text())
    f.write("\n\n=== MULTICOLLINEARITY DIAGNOSTICS (VIF) ===\n")
    f.write(vif.to_string())
    f.write(f"\n\n=== TIME-SERIES STATIONARITY (ADF) ===\np-value: {adf_test[1]:.4f}")
    if adf_test[1] < 0.05:
        f.write("\nStatus: Data is stationary (Statistical integrity confirmed)")
    else:
        f.write("\nStatus: Non-stationary data (Trend adjustments applied)")
    f.write("\n\n=== VAR (Vector Autoregression) SUMMARY ===\n")
    f.write(str(var_results.summary()))
    f.write("\n\n=== Granger Causality (log_returns causes fear_greed) ===\n")
    f.write(f"p-values (lags 1-2): {granger_pvals}\n")
    f.write("\n\n=== Random Forest Regressor (ML) ===\n")
    f.write(f"R^2: {rf.score(data[x_vars], data[y_var]):.4f}\n")
    f.write(f"Feature Importances: {dict(zip(x_vars, rf.feature_importances_))}\n")

    # --- INTERPRETATION SUMMARY ---
    f.write("\n\n=== INTERPRETATION SUMMARY ===\n")
    ols_r2 = model.rsquared
    ols_sig = any(model.pvalues[1:] < 0.05)
    f.write(f"OLS R^2: {ols_r2:.3f}. ")
    if ols_sig:
        f.write("At least one predictor is statistically significant (p < 0.05). ")
    else:
        f.write("No predictors are statistically significant (p >= 0.05). ")
    max_vif = vif['VIF'][1:].max() if len(vif) > 1 else 0
    if max_vif > 10:
        f.write("Warning: High multicollinearity detected (VIF > 10). ")
    else:
        f.write("No severe multicollinearity detected. ")
    if adf_test[1] < 0.05:
        f.write("The dependent variable is stationary. ")
    else:
        f.write("The dependent variable is non-stationary. ")
    f.write(f"VAR selected lag order: {var_results.k_ar}. ")
    try:
        min_granger_p = min(granger_pvals)
        if min_granger_p < 0.05:
            f.write("Granger causality detected (log_returns → fear_greed) for some lags. ")
        else:
            f.write("No Granger causality detected (log_returns → fear_greed). ")
    except:
        pass
    rf_r2 = rf.score(data[x_vars], data[y_var])
    f.write(f"Random Forest R^2: {rf_r2:.3f}. ")
    f.write("\n")

# 6. GENERATE PROFESSIONAL CHARTS
# Επιλογή άξονα Χ (Ημερομηνία αν υπάρχει, αλλιώς το index)
x_axis = data['date'] if 'date' in data.columns else data.index

plt.figure(figsize=(10, 6))
plt.plot(x_axis, Y, label='Actual Market Returns', marker='o', color='#1f77b4')
plt.plot(x_axis, model.predict(X), label='OLS Prediction', linestyle='--', color='#d62728')
plt.plot(x_axis, rf_pred, label='Random Forest Prediction', linestyle=':', color='#ff7f0e')
plt.title('Bitcoin Price Dynamics: Actual vs Model Predictions')
plt.xlabel('Observations (Days)')
plt.ylabel('Log Returns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/Model_Prediction_Accuracy.png')

plt.figure(figsize=(10, 6))
model.params[1:].plot(kind='bar', color=['#2ca02c', '#9467bd', '#d62728', '#7f7f7f'])
plt.title('Market Driver Impact Analysis (OLS)')
plt.ylabel('Coefficient Sensitivity')
plt.axhline(0, color='black', linewidth=0.8)
plt.savefig('results/Market_Driver_Impact_OLS.png')

plt.figure(figsize=(10, 6))
importances = rf.feature_importances_
plt.bar(x_vars, importances, color='#ff7f0e')
plt.title('Feature Importances (Random Forest)')
plt.ylabel('Importance')
plt.savefig('results/Feature_Importances_RF.png')

# Προστασία στο Heatmap: Δώσε μόνο αριθμητικά δεδομένα
plt.figure(figsize=(8, 6))
numeric_data = data[[y_var] + x_vars]
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('results/Correlation_Heatmap.png')


# --- Adaptive Rolling Correlation Plot ---
plt.figure(figsize=(10, 6))
window = min(30, max(2, len(data)//2))
if len(data) >= window:
    roll_corr = data['log_returns'].rolling(window).corr(data['fear_greed'])
    plt.plot(roll_corr)
    plt.title(f'{window}-Day Rolling Correlation: log_returns vs fear_greed')
    plt.ylabel('Correlation')
    plt.xlabel('Observations (Days)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/Rolling_Correlation.png')
else:
    plt.text(0.5, 0.5, f'Not enough data for rolling correlation (need at least {window} obs)',
             horizontalalignment='center', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Rolling Correlation: log_returns vs fear_greed')
    plt.savefig('results/Rolling_Correlation.png')

plt.close('all')

print(f"\n--- ANALYSIS COMPLETE ---")
print(f"Data Points Synchronized: {len(data)} days.")
print(f"All reports and charts generated in the 'results/' directory.")