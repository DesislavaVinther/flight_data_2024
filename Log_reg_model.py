"""
LOGISTIC REGRESSION MODEL FOR FLIGHT DELAY ANALYSIS
====================================================
Project Goal: Compare airline delay rates before and after automatic refund legislation
Target Variable: Binary_delay (1 = severe delay >=180 minutes, 0 = no severe delay)

Research Question: Based on the 15 airlines, what is the probability that a passenger is delayed by 180 minutes
                    before and after the legislation on automatic refunds came into force on 25 June 2024?

Time Periods:
- Period 1 (Before law): July 2023 - June 2024
- Period 2 (After law): July 2024 - June 2025

Statistical Method: Logistic Regression with interaction terms
- Main parameters: airline & time period
- Interaction effect: airline × period (tests if law affects airlines differently)

Output: Beta coefficients for Wilks likelihood ratio test
"""

# ****************
# IMPORT LIBRARIES
# ****************

import pandas as pd                 # Data manipulation (reading CSV, filtering, grouping)
import numpy as np                  # Numerical operations (arrays, mathematical functions)
import statsmodels.formula.api as smf  # Formula interface for models (like R syntax)
import warnings                     # Python's warning system
warnings.filterwarnings('ignore')   # Suppress warning messages for cleaner output


# ****************
# SECTION 1: DATA LOADING AND PREPARATION
# ****************

def load_and_prepare_data(filepath):
    """
    Purpose: Load flight data and create time period variables for analysis

    Steps:
    1. Load CSV file
    2. Rename columns to match expected names
    3. Convert date column to datetime format
    4. Create period variable (0 = before law, 1 = after law)
    5. Filter data to only include the two 12-month periods

    Returns: pandas DataFrame with prepared data
    """

    # STEP 1: Load the CSV file
    # pd.read_csv() reads a CSV file and returns a DataFrame (table)
    # The 'r' before the string means "raw string" - backslashes are treated literally
    # Without 'r', \U will be interpreted as an escape sequence and cause errors
    df = pd.read_csv(r"C:\Users\desib\PycharmProjects\Flight data\data\flight_data_2023_2025_V_1.5.csv")


    # STEP 2: Rename columns to standard names
    # The CSV file has different column names than what the script expects
    # df.rename() creates a mapping: old_name → new_name
    df = df.rename(columns={
        'op_unique_carrier': 'airline',     # Airline code (e.g., 'AA' = American Airlines)
        'binary_delay': 'Binary_delay'      # 1 if delay >=180 min, 0 otherwise
    })


    # STEP 3: Convert date column to datetime format
    # pd.to_datetime() converts string dates (like "2024-07-01") to datetime objects
    # Datetime objects allow date comparisons and filtering (e.g., date1 < date2)
    df['fl_date'] = pd.to_datetime(df['fl_date'])


    # STEP 4: Create period variables for before/after law comparison
    # The "cutoff date" for the two periods is July 1, 2024
    cutoff_date = pd.Timestamp('2024-07-01')  # Create a datetime object for the cutoff

    # np.where() is like Excel's IF function: np.where(condition, value_if_true, value_if_false)
    # For each row: if fl_date < cutoff → period = 0, else period = 1
    df['period'] = np.where(df['fl_date'] < cutoff_date, 0, 1)


    # STEP 5: Filter data to only include the two 12-month periods
    # We want exactly 12 months before and after the law for fair comparison
    start_period1 = pd.Timestamp('2023-07-01')  # Start of Period 1
    end_period2 = pd.Timestamp('2025-06-30')    # End of Period 2

    # Boolean indexing: Keep only rows where date is between start and end
    # The & operator means AND (both conditions must be true)
    # Parentheses are required when using & with pandas comparisons
    df = df[(df['fl_date'] >= start_period1) & (df['fl_date'] <= end_period2)]

    return df


def create_model_data(df):
    """
    Purpose: Select only the columns needed for the logistic regression model

    The full dataset has many columns, but we only need 3 for this analysis:
    1. Binary_delay - Target variable (what we're predicting)
    2. airline - Predictor variable (which airline)
    3. period - Predictor variable (before or after law)

    Returns: Filtered DataFrame with only model variables
    """

    # Select specific columns and create a copy
    # .copy() creates an independent copy so changes don't affect the original df
    # [['col1', 'col2']] - double brackets select multiple columns
    model_df = df[['Binary_delay', 'airline', 'period']].copy()


    # Remove rows with ANY missing values
    # .dropna() removes any row that has at least one NaN value
    # Missing values cause errors in statistical models
    model_df = model_df.dropna()

    return model_df


# ****************
# SECTION 2: LOGISTIC REGRESSION MODEL
# ****************

def logistic_regression_statsmodels(df):
    """
    Purpose: Fit a logistic regression model with interaction terms

    MODEL SPECIFICATION:
    Binary_delay ~ airline + period + airline:period

    Translation:
    - Main effect of airline: Some airlines have higher/lower delay rates overall
    - Main effect of period: Delay rate changed from Period 1 to Period 2
    - Interaction effect (airline:period): Law affected airlines differently

    Returns:
    - model: Fitted statsmodels logistic regression model
    - reference_airline: Which airline is the baseline for comparisons
    """

    # CHOOSE REFERENCE CATEGORY
    # In regression with categorical variables, one category is the "reference"
    # All other categories are compared to this reference
    # Get all unique airlines and sort alphabetically
    # [0] selects the first airline alphabetically
    reference_airline = sorted(df['airline'].unique())[0]


    # BUILD MODEL FORMULA
    # statsmodels uses R-style formulas: "outcome ~ predictors"
    # C() treats variable as categorical
    # Treatment(reference="AA") sets AA as baseline category
    # * creates interaction: airline * period = airline + period + airline:period
    formula = 'Binary_delay ~ C(airline, Treatment(reference="' + reference_airline + '")) * C(period)'


    # FIT THE MODEL
    # smf.logit() creates a logistic regression model
    # .fit() estimates coefficients using maximum likelihood estimation
    # disp=0 suppresses convergence messages during fitting
    model = smf.logit(formula, data=df).fit(disp=0)

    # What does .fit() do?
    # - Finds coefficients that maximize likelihood of observed data
    # - Uses iterative algorithm (starts with guess, improves until convergence)
    # - Returns fitted model with coefficients, p-values, confidence intervals

    return model, reference_airline


def extract_beta_coefficients(model, reference_airline):
    """
    Purpose: Extract beta coefficients (β values) from the logistic regression model

    WHAT ARE BETA COEFFICIENTS?
    - Beta values (β) are the coefficients from the logistic regression
    - They represent log-odds ratios for each predictor
    - Formula: log(odds) = β₀ + β₁*X₁ + β₂*X₂ + ... + βₙ*Xₙ

    WHY EXTRACT BETA VALUES?
    - Needed for Wilks likelihood parameter test
    - Allows testing significance of individual predictors or groups of predictors
    - Comparing nested models (with and without certain predictors)

    WILKS LIKELIHOOD RATIO TEST:
    - Tests if adding predictors significantly improves model fit
    - Formula: -2 * (log-likelihood_restricted - log-likelihood_full)
    - Follows chi-square distribution with df = difference in number of parameters

    Returns:
    - beta_df: DataFrame with all beta coefficients, standard errors, and statistics
    """

    # EXTRACT COEFFICIENTS (BETA VALUES)
    # model.params contains all beta coefficients from the regression
    # These are the β values in the logistic regression equation
    beta_values = model.params

    # EXTRACT STANDARD ERRORS
    # Standard error measures the uncertainty in each coefficient estimate
    # Smaller SE = more precise estimate
    standard_errors = model.bse

    # EXTRACT Z-STATISTICS
    # Z-statistic = coefficient / standard error
    # Measures how many standard errors the coefficient is away from zero
    z_statistics = model.tvalues

    # EXTRACT P-VALUES
    # P-value tests: "Is this coefficient significantly different from zero?"
    # Small p-value (< 0.05) = statistically significant predictor
    p_values = model.pvalues

    # EXTRACT CONFIDENCE INTERVALS
    # 95% confidence interval for each coefficient
    # Range of plausible values for the true beta value
    conf_int = model.conf_int()


    # CREATE DATAFRAME WITH ALL STATISTICS
    beta_df = pd.DataFrame({
        'Beta_Coefficient': beta_values,      # β values (log-odds)
        'Std_Error': standard_errors,          # Standard errors
        'Z_Statistic': z_statistics,           # Z-test statistics
        'P_value': p_values,                   # P-values
        'CI_Lower_95': conf_int[0],            # Lower 95% CI
        'CI_Upper_95': conf_int[1]             # Upper 95% CI
    })

    # ADD SIGNIFICANCE MARKER
    # Mark coefficients as significant if p < 0.05
    beta_df['Significant'] = beta_df['P_value'] < 0.05


    # PRINT RESULTS
    print("\n" + "=" * 70)
    print("BETA COEFFICIENTS (LOG-ODDS) FOR LOGISTIC REGRESSION")
    print("=" * 70)
    print(f"Reference airline (baseline): {reference_airline}")
    print("\nInterpretation:")
    print("- Beta = 0: No effect on log-odds of delay")
    print("- Beta > 0: Increases log-odds of delay (higher delay probability)")
    print("- Beta < 0: Decreases log-odds of delay (lower delay probability)")
    print("=" * 70)
    print(beta_df.round(4))  # Round to 4 decimal places for readability


    # MODEL STATISTICS FOR WILKS TEST
    print("\n" + "=" * 70)
    print("MODEL FIT STATISTICS (for Wilks likelihood ratio test)")
    print("=" * 70)
    print(f"Log-Likelihood: {model.llf:.4f}")          # Log-likelihood of full model
    print(f"Number of parameters: {len(model.params)}")  # Number of beta coefficients
    print(f"AIC: {model.aic:.4f}")                      # Akaike Information Criterion
    print(f"BIC: {model.bic:.4f}")                      # Bayesian Information Criterion

    # Why print log-likelihood?
    # For Wilks test, you need log-likelihood from both full and restricted models
    # Test statistic = -2 * (LL_restricted - LL_full)
    # This follows chi-square distribution with df = difference in parameters

    return beta_df


# ****************
# SECTION 3: MAIN FUNCTION
# ****************

def main(filepath):
    """
    Purpose: Run analysis pipeline to extract beta coefficients

    WORKFLOW:
    1. Load and prepare data
    2. Fit logistic regression model
    3. Extract beta coefficients
    4. Save results

    OUTPUT FILES:
    - beta_coefficients.csv: Beta values needed for Wilks likelihood ratio test

    Returns: Dictionary with model, beta coefficients, and log-likelihood
    """

    # Print header
    print("=" * 70)
    print("LOGISTISK REGRESSION: BETA COEFFICIENT EXTRACTION")
    print("Output: Beta coefficients for Wilks likelihood ratio test")
    print("=" * 70)


    # STEP 1: LOAD AND PREPARE DATA
    print("\n>>> Loading and preparing data...")
    df = load_and_prepare_data(filepath)           # Load raw data
    model_df = create_model_data(df)               # Select model variables

    # Print basic statistics
    print(f"Total flights: {len(model_df):,}")
    print(f"Delay rate: {model_df['Binary_delay'].mean() * 100:.2f}%")
    print(f"Number of airlines: {model_df['airline'].nunique()}")


    # STEP 2: LOGISTIC REGRESSION
    print("\n>>> Fitting logistic regression model...")
    model, ref_airline = logistic_regression_statsmodels(model_df)


    # STEP 3: EXTRACT BETA COEFFICIENTS
    print("\n>>> Extracting beta coefficients...")
    beta_df = extract_beta_coefficients(model, ref_airline)


    # STEP 4: SAVE RESULTS
    print("\n>>> Saving results...")
    # Save beta coefficients to CSV file for later analysis
    beta_df.to_csv('beta_coefficients.csv')


    # PRINT SUMMARY
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated file:")
    print("  - beta_coefficients.csv (Beta coefficients for Wilks test)")


    # RETURN RESULTS
    # Return dictionary with all key results
    # This allows accessing results after running: results = main(filepath)
    return {
        'model': model,                      # Fitted statsmodels model
        'beta_coefficients': beta_df,        # Beta coefficients (for Wilks test)
        'log_likelihood': model.llf,         # Log-likelihood (needed for Wilks test)
        'n_params': len(model.params)        # Number of parameters (needed for Wilks test)
    }


# ****************
# SECTION 4: SCRIPT EXECUTION
# ****************

if __name__ == "__main__":
    # What is if __name__ == "__main__"?
    # - This code only runs if script is executed directly (python Log_reg_model.py)
    # - Doesn't run if script is imported in another file (import Log_reg_model)
    # - Allows using this file both as script and as module

    # SPECIFY DATA FILE PATH
    filepath = "din_flight_data.csv"  # Placeholder - actual path is in load_and_prepare_data()

    # RUN ANALYSIS
    results = main(filepath)
