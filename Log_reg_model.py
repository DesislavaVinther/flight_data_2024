"""
LOGISTIC REGRESSION MODEL FOR FLIGHT DELAY ANALYSIS
====================================================
Project Goal: Compare airline delay rates before and after automatic refund legislation
Target Variable: Binary_delay (1 = severe delay >=180 minutes, 0 = no severe delay)

Research Question: Did the automatic refund law (effective July 2024) change
                    the probability of severe delays for different airlines?

Time Periods:
- Period 1 (Before law): July 2023 - June 2024
- Period 2 (After law): July 2024 - June 2025

Statistical Method: Logistic Regression with interaction terms
- Main effects: airline, period

- Interaction effect: airline × period (tests if law affects airlines differently)
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
# Note: Each library serves a specific purpose in the analysis

import pandas as pd  # Data manipulation (reading CSV, filtering, grouping)
import numpy as np   # Numerical operations (arrays, mathematical functions)
import matplotlib.pyplot as plt  # Creating plots and visualizations

# Scikit-learn: Machine learning library for model training and evaluation
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.model_selection import train_test_split, cross_val_score  # Data splitting and validation
from sklearn.metrics import classification_report, accuracy_score  # Model performance metrics

# Statsmodels: Statistical modeling library (gives p-values and confidence intervals)
import statsmodels.api as sm  # Statistical models and tests
import statsmodels.formula.api as smf  # Formula interface for models (like R syntax)

import warnings  # Python's warning system
warnings.filterwarnings('ignore')  # Suppress warning messages for cleaner output
                                   # Note: Normally warnings are important, but they can clutter output


# =============================================================================
# SECTION 1: DATA LOADING AND PREPARATION
# =============================================================================
# This section loads flight data from CSV and prepares it for analysis

def load_and_prepare_data(filepath):
    """
    Purpose: Load flight data and create time period variables for analysis

    Steps:
    1. Load CSV file
    2. Rename columns to match expected names
    3. Convert date column to datetime format
    4. Create period variable (0 = before law, 1 = after law)
    5. Filter data to only include the two 12-month periods
    6. Print summary statistics

    Returns: pandas DataFrame with prepared data
    """

    # STEP 1: Load the CSV file
    # pd.read_csv() reads a CSV file and returns a DataFrame (table)
    # The 'r' before the string means "raw string" - backslashes are treated literally
    # Without 'r', Python would interpret \U as an escape sequence and cause errors
    df = pd.read_csv(r"C:\Users\desib\PycharmProjects\Flight data\data\flight_data_2023_2025_V_1.5.csv")

    # Note: At this point, df contains ALL columns from the CSV file
    # Example columns: op_unique_carrier, fl_date, binary_delay, arr_delay, etc.


    # STEP 2: Rename columns to standard names
    # The CSV file has different column names than what the script expects
    # df.rename() creates a mapping: old_name → new_name
    # This makes the code more readable and standardized
    df = df.rename(columns={
        'op_unique_carrier': 'airline',     # Airline code (e.g., 'AA' = American Airlines)
        'binary_delay': 'Binary_delay'      # 1 if delay >=180 min, 0 otherwise
    })

    # Why rename? The rest of the script expects 'airline' and 'Binary_delay'
    # This way, the script works with any dataset as long as these columns exist


    # STEP 3: Convert date column to datetime format
    # pd.to_datetime() converts string dates (like "2024-07-01") to datetime objects
    # Datetime objects allow date comparisons and filtering (e.g., date1 < date2)
    # Without this, dates would be treated as text, and comparisons wouldn't work
    df['fl_date'] = pd.to_datetime(df['fl_date'])

    # Example: "2024-07-01" (string) → Timestamp('2024-07-01 00:00:00')


    # STEP 4: Create period variables for before/after law comparison
    # The law went into effect on July 1, 2024 - this is the "cutoff date"
    cutoff_date = pd.Timestamp('2024-07-01')  # Create a datetime object for the cutoff

    # np.where() is like Excel's IF function: np.where(condition, value_if_true, value_if_false)
    # For each row: if fl_date < cutoff → period = 0, else period = 1
    df['period'] = np.where(df['fl_date'] < cutoff_date, 0, 1)

    # Create a readable label version (used for plots and tables)
    # Same logic but returns text labels instead of 0/1
    df['period_label'] = np.where(df['fl_date'] < cutoff_date,
                                  'Periode 1 (Før loven)',      # Before law
                                  'Periode 2 (Efter loven)')    # After law

    # Why two columns? period (0/1) is used in statistical models
    #                 period_label (text) is used in visualizations


    # STEP 5: Filter data to only include the two 12-month periods
    # We want exactly 12 months before and after the law for fair comparison
    start_period1 = pd.Timestamp('2023-07-01')  # Start of Period 1
    end_period2 = pd.Timestamp('2025-06-30')    # End of Period 2

    # Boolean indexing: Keep only rows where date is between start and end
    # The & operator means AND (both conditions must be true)
    # Parentheses are required when using & with pandas comparisons
    df = df[(df['fl_date'] >= start_period1) & (df['fl_date'] <= end_period2)]

    # After this line, df only contains flights from July 2023 to June 2025


    # STEP 6: Print summary statistics to understand the data
    # f-string format: f"text {variable}" inserts the variable value into the text
    # len(df) = number of rows (total flights)
    # :, adds thousand separators (e.g., 14,033,087)
    print(f"Total antal flyvninger: {len(df):,}")

    # Count flights in Period 1 (before law)
    # df[df['period'] == 0] filters to only Period 1 rows
    print(f"Periode 1 flyvninger: {len(df[df['period'] == 0]):,}")

    # Count flights in Period 2 (after law)
    print(f"Periode 2 flyvninger: {len(df[df['period'] == 1]):,}")

    # nunique() = number of unique values
    # Shows how many different airlines are in the dataset
    print(f"Antal airlines: {df['airline'].nunique()}")

    # Calculate overall delay rate
    # Since Binary_delay is 0 or 1, mean() gives the percentage of 1s
    # Example: [0, 1, 1, 0] → mean = 0.5 = 50% delayed
    # * 100 converts to percentage, :.2f formats to 2 decimal places
    print(f"\nDelay rate overall: {df['Binary_delay'].mean() * 100:.2f}%")

    return df  # Return the prepared DataFrame for use in other functions


def create_model_data(df):
    """
    Purpose: Select only the columns needed for the logistic regression model

    Why? The full dataset has 37 columns, but we only need 4 for this analysis:
    1. Binary_delay - Target variable (what we're predicting)
    2. airline - Predictor variable (which airline)
    3. period - Predictor variable (before or after law)
    4. period_label - For display purposes

    This makes the model faster and prevents using irrelevant variables

    Returns: Filtered DataFrame with only model variables
    """

    # Select specific columns and create a copy
    # .copy() creates an independent copy so changes don't affect the original df
    # [['col1', 'col2']] - double brackets select multiple columns
    model_df = df[['Binary_delay', 'airline', 'period', 'period_label']].copy()

    # Why these columns?
    # - Binary_delay: The outcome we're trying to predict (Y variable)
    # - airline: Categorical predictor - which airline operated the flight
    # - period: Binary predictor - time period (0 or 1)
    # - period_label: Human-readable version for output tables


    # Check for missing values (NaN = "Not a Number" in pandas)
    # .isnull() returns True/False for each cell
    # .sum() counts the True values (missing values) for each column
    print(f"\nMissing values:\n{model_df.isnull().sum()}")

    # Why check? Missing values cause errors in statistical models
    # Options: 1) Remove rows with missing values (dropna)
    #          2) Fill missing values (imputation)
    #          3) Use special algorithms that handle missing data


    # Remove rows with ANY missing values
    # .dropna() removes any row that has at least one NaN value
    # This is the simplest approach but may lose data
    model_df = model_df.dropna()

    # After dropna(), model_df has no missing values and is ready for modeling

    return model_df


# =============================================================================
# SECTION 2: DESCRIPTIVE STATISTICS
# =============================================================================
# Calculate and visualize delay rates by airline and time period

def descriptive_statistics(df):
    """
    Purpose: Calculate delay rates for each airline in each time period

    This creates a summary table showing:
    - How many flights each airline had
    - How many were delayed (>=180 min)
    - The delay rate (percentage)

    Returns:
    - delay_stats: Detailed statistics DataFrame
    - pivot: Summary table showing Period 1 vs Period 2
    """

    # GROUP BY and AGGREGATE
    # .groupby() groups data by airline and period (creates groups like "AA, Period 1")
    # .agg() applies functions to each group
    delay_stats = df.groupby(['airline', 'period_label']).agg(
        # For each group, calculate three things:

        # 1. Count total number of flights
        # ('Binary_delay', 'count') counts non-missing values in Binary_delay column
        total_flights=('Binary_delay', 'count'),

        # 2. Count delayed flights
        # ('Binary_delay', 'sum') adds up all the 1s (delayed flights)
        # Why sum? Since values are 0 or 1, sum gives the count of 1s
        delayed_flights=('Binary_delay', 'sum'),

        # 3. Calculate delay rate
        # ('Binary_delay', 'mean') gives the average
        # Since values are 0/1, mean = proportion of 1s = delay rate
        delay_rate=('Binary_delay', 'mean')
    ).reset_index()  # reset_index() converts group labels back to regular columns

    # Why group by these columns? We want separate statistics for:
    # - Each airline (AA, DL, UA, etc.)
    # - Each period (Before law, After law)
    # This allows us to compare how each airline changed over time


    # Convert delay rate from decimal to percentage
    # 0.0234 → 2.34 (multiply by 100)
    # Why separate column? Easier to display percentages in tables/plots
    delay_stats['delay_rate_pct'] = delay_stats['delay_rate'] * 100


    # Print header for output
    print("\n" + "=" * 70)  # Creates a line of 70 equal signs
    print("DESKRIPTIV STATISTIK: Delay Rate per Airline og Periode")
    print("=" * 70)


    # Create pivot table for easier comparison
    # .pivot() reshapes data from "long" to "wide" format
    # Before: Each airline-period combo is one row
    # After: Each airline is one row, periods are columns
    pivot = delay_stats.pivot(
        index='airline',          # Rows = airlines
        columns='period_label',   # Columns = periods
        values='delay_rate_pct'   # Cell values = delay rate percentages
    )

    # Example pivot table structure:
    #              Periode 1 (Før loven)  Periode 2 (Efter loven)
    # airline
    # AA                    2.34                     2.45
    # DL                    1.89                     1.76


    # Calculate the change in delay rate (Period 2 - Period 1)
    # Positive value = delays increased after law
    # Negative value = delays decreased after law
    pivot['Ændring (%-point)'] = pivot['Periode 2 (Efter loven)'] - pivot['Periode 1 (Før loven)']

    # Sort by change (most negative to most positive)
    # This shows which airlines improved most (negative) vs worsened most (positive)
    pivot = pivot.sort_values('Ændring (%-point)')


    # Print the pivot table rounded to 2 decimal places
    print(pivot.round(2))

    return delay_stats, pivot  # Return both detailed and summary statistics


def plot_delay_rates(delay_stats):
    """
    Purpose: Create a grouped bar chart comparing Period 1 vs Period 2 for each airline

    Visualization: Two bars per airline (blue = before, red = after)
    This makes it easy to visually compare delay rates across airlines and periods
    """

    # Create figure and axis objects
    # figsize=(14, 8) sets width and height in inches
    # fig = entire figure, ax = the plot area where we draw
    fig, ax = plt.subplots(figsize=(14, 8))

    # Why create fig and ax? This gives us control over every aspect of the plot
    # We can add titles, labels, adjust spacing, etc.


    # PREPARE DATA FOR GROUPED BAR CHART
    # Get unique airlines
    airlines = delay_stats['airline'].unique()

    # Create x-axis positions (0, 1, 2, 3, ...)
    # np.arange(n) creates array [0, 1, 2, ..., n-1]
    # We need numeric positions to place bars
    x = np.arange(len(airlines))

    # Set bar width (how wide each bar is)
    # 0.35 = 35% of the space between x-positions
    # This leaves room for two bars side by side
    width = 0.35


    # Filter data for each period
    # We need separate data for Period 1 (blue bars) and Period 2 (red bars)
    periode1 = delay_stats[delay_stats['period_label'] == 'Periode 1 (Før loven)']
    periode2 = delay_stats[delay_stats['period_label'] == 'Periode 2 (Efter loven)']


    # IMPORTANT: Ensure data is in same order as airlines array
    # .set_index('airline') makes airline the index
    # .reindex(airlines) reorders rows to match the airlines order
    # Why? Bar positions (x) match airlines order, so data must too
    periode1 = periode1.set_index('airline').reindex(airlines)
    periode2 = periode2.set_index('airline').reindex(airlines)


    # CREATE THE BARS
    # ax.bar() creates vertical bars
    # x position - width/2 → shifts bars slightly left (for Period 1)
    bars1 = ax.bar(x - width / 2,                    # X positions (shifted left)
                   periode1['delay_rate_pct'],       # Bar heights (delay rates)
                   width,                            # Bar width
                   label='Periode 1 (Før loven)',    # Legend label
                   color='#3498db',                  # Blue color (hex code)
                   alpha=0.8)                        # Transparency (0.8 = 80% opaque)

    # x position + width/2 → shifts bars slightly right (for Period 2)
    bars2 = ax.bar(x + width / 2,                    # X positions (shifted right)
                   periode2['delay_rate_pct'],       # Bar heights
                   width,                            # Bar width
                   label='Periode 2 (Efter loven)',  # Legend label
                   color='#e74c3c',                  # Red color
                   alpha=0.8)                        # Transparency

    # Why shift bars? Without shifts, they would overlap
    # Shifting left and right creates grouped bars


    # ADD LABELS AND TITLE
    ax.set_xlabel('Airline', fontsize=12)                      # X-axis label
    ax.set_ylabel('Sandsynlighed for Severe Delay (%)', fontsize=12)  # Y-axis label
    ax.set_title('Sandsynlighed for Forsinkelse >=180 min per Airline\n' +
                 'Foer og Efter Lovgivning om Automatisk Refusion',
                 fontsize=14, fontweight='bold')               # Title with line break (\n)

    # Set x-axis tick positions and labels
    ax.set_xticks(x)                                           # Where to put ticks (0, 1, 2, ...)
    ax.set_xticklabels(airlines, rotation=45, ha='right')      # Labels (AA, DL, ...), rotated 45°
    # ha='right' = horizontal alignment right (so rotated text doesn't overlap)

    ax.legend()                  # Add legend showing Period 1 (blue) and Period 2 (red)
    ax.grid(axis='y', alpha=0.3) # Add horizontal gridlines (alpha=0.3 makes them faint)
    # Why gridlines? Makes it easier to read exact values


    # ADD VALUE LABELS ON TOP OF EACH BAR
    # Loop through Period 1 bars
    for bar in bars1:
        height = bar.get_height()  # Get bar height (delay rate value)

        # ax.annotate() adds text to the plot
        ax.annotate(f'{height:.1f}%',                           # Text to display (e.g., "2.3%")
                    xy=(bar.get_x() + bar.get_width() / 2,      # X position (center of bar)
                        height),                                # Y position (top of bar)
                    xytext=(0, 3),                              # Offset: 3 points above bar
                    textcoords="offset points",                 # xytext is relative offset
                    ha='center', va='bottom',                   # Horizontal/vertical alignment
                    fontsize=8)                                 # Font size

    # Same for Period 2 bars
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

    # Why add labels? Easier to read exact values without checking y-axis


    # SAVE AND DISPLAY PLOT
    plt.tight_layout()  # Adjust spacing to prevent label cutoff

    plt.savefig('delay_rates_by_airline.png',  # Save to file
                dpi=300,                        # Resolution: 300 dots per inch (high quality)
                bbox_inches='tight')            # Crop whitespace around figure

    plt.show()  # Display plot in a window (or inline in Jupyter)

    return fig  # Return figure object (useful if we want to modify it later)


# =============================================================================
# SECTION 3: LOGISTIC REGRESSION MODEL
# =============================================================================
# Build statistical model to test if law affects delay probability

def logistic_regression_statsmodels(df):
    """
    Purpose: Fit a logistic regression model with interaction terms

    WHAT IS LOGISTIC REGRESSION?
    - Predicts probability of binary outcome (0 or 1)
    - In this case: P(Binary_delay = 1 | airline, period)
    - Uses sigmoid function to convert linear combination to probability (0 to 1)

    MODEL SPECIFICATION:
    Binary_delay ~ airline + period + airline:period

    Translation:
    - Main effect of airline: Some airlines have higher/lower delay rates overall
    - Main effect of period: Delay rate changed from Period 1 to Period 2
    - Interaction effect (airline:period): Law affected airlines differently

    WHY INTERACTION TERM?
    Without interaction: Assumes law has same effect on all airlines
    With interaction: Allows law to affect each airline differently

    EXAMPLE:
    Maybe AA improved after the law, but DL got worse
    Interaction term captures these airline-specific changes

    Returns:
    - model: Fitted statsmodels logistic regression model
    - reference_airline: Which airline is the baseline for comparisons
    """

    print("\n" + "=" * 70)
    print("LOGISTISK REGRESSION MODEL (Statsmodels)")
    print("=" * 70)


    # CHOOSE REFERENCE CATEGORY
    # In regression with categorical variables, one category is the "reference"
    # All other categories are compared to this reference

    # Get all unique airlines and sort alphabetically
    # sorted() returns a list in alphabetical order
    # [0] selects the first airline alphabetically
    reference_airline = sorted(df['airline'].unique())[0]
    print(f"\nReference airline: {reference_airline}")

    # Why choose reference? Coefficients are interpreted as:
    # "Compared to [reference_airline], airline X has Y% higher/lower odds"

    # Example: If AA is reference and DL coefficient = 0.5
    # Interpretation: DL has 50% higher log-odds of delay than AA


    # BUILD MODEL FORMULA
    # statsmodels uses R-style formulas: "outcome ~ predictors"
    # C() treats variable as categorical
    # Treatment(reference="AA") sets AA as baseline category
    # * creates interaction: airline * period = airline + period + airline:period
    formula = 'Binary_delay ~ C(airline, Treatment(reference="' + reference_airline + '")) * C(period)'

    # Example formula: Binary_delay ~ C(airline, Treatment(reference="AA")) * C(period)
    # This creates:
    # 1. Intercept (baseline probability for reference airline in Period 1)
    # 2. Coefficients for each airline (compared to reference)
    # 3. Coefficient for period (effect of law for reference airline)
    # 4. Interaction coefficients (how law effect differs by airline)


    # FIT THE MODEL
    # smf.logit() creates a logistic regression model
    # .fit() estimates coefficients using maximum likelihood estimation
    # disp=0 suppresses convergence messages during fitting
    model = smf.logit(formula, data=df).fit(disp=0)

    # What does .fit() do?
    # - Finds coefficients that maximize likelihood of observed data
    # - Uses iterative algorithm (starts with guess, improves until convergence)
    # - Returns fitted model with coefficients, p-values, confidence intervals


    # PRINT MODEL SUMMARY
    print("\n--- MODEL SUMMARY ---")
    print(model.summary())
    # Summary includes:
    # - Coefficients (log-odds)
    # - Standard errors (uncertainty in coefficients)
    # - z-values (coefficient / standard error)
    # - P-values (statistical significance)
    # - Confidence intervals
    # - Model fit statistics (Log-Likelihood, AIC, BIC)

    return model, reference_airline


def calculate_probabilities(model, df, reference_airline):
    """
    Purpose: Calculate predicted probabilities for each airline in each period

    Why? Model coefficients are in log-odds (hard to interpret)
    Probabilities are easier: "AA has 2.3% chance of severe delay in Period 1"

    Process:
    1. Create prediction data (all airline-period combinations)
    2. Use model to predict probability for each combination
    3. Organize results in a table for comparison

    Returns:
    - results_df: Detailed probabilities
    - prob_pivot: Summary table (airlines × periods)
    """

    # Get all unique airlines (sorted alphabetically)
    airlines = sorted(df['airline'].unique())

    # Initialize empty list to store results
    # We'll add one dictionary per airline-period combination
    results = []


    # LOOP THROUGH ALL COMBINATIONS
    # Outer loop: iterate through airlines
    for airline in airlines:
        # Inner loop: iterate through periods (0 and 1)
        for period in [0, 1]:

            # CREATE PREDICTION DATA
            # model.predict() needs a DataFrame with predictor values
            # Create a DataFrame with one row containing airline and period
            pred_data = pd.DataFrame({
                'airline': [airline],  # Note: [airline] creates a list (required for DataFrame)
                'period': [period]
            })
            # Example: {'airline': ['AA'], 'period': [0]}


            # PREDICT PROBABILITY
            # model.predict() returns array of probabilities
            # [0] extracts the first (and only) probability
            prob = model.predict(pred_data)[0]

            # What does predict() do?
            # 1. Calculates linear combination: β₀ + β₁*airline + β₂*period + β₃*airline*period
            # 2. Applies sigmoid function: prob = 1 / (1 + exp(-linear_combination))
            # 3. Returns probability between 0 and 1


            # STORE RESULTS
            # Create dictionary with results and append to list
            results.append({
                'airline': airline,
                'period': period,
                'period_label': 'Før loven' if period == 0 else 'Efter loven',  # Readable label
                'probability': prob,                # Probability as decimal (0.023)
                'probability_pct': prob * 100       # Probability as percentage (2.3)
            })

    # After loops, results contains one dictionary per airline-period combination


    # CONVERT TO DATAFRAME
    # pd.DataFrame() converts list of dictionaries to DataFrame
    results_df = pd.DataFrame(results)


    # CREATE PIVOT TABLE FOR EASY COMPARISON
    # Reshape from long to wide format
    prob_pivot = results_df.pivot(
        index='airline',              # Rows = airlines
        columns='period_label',       # Columns = periods
        values='probability_pct'      # Values = probabilities
    )

    # Example pivot structure:
    #          Før loven  Efter loven
    # airline
    # AA           2.34         2.45
    # DL           1.89         1.76


    # CALCULATE CHANGES
    # Absolute change (percentage points)
    # Example: 2.45% - 2.34% = +0.11 percentage points
    prob_pivot['Ændring (%-point)'] = prob_pivot['Efter loven'] - prob_pivot['Før loven']

    # Relative change (percent change)
    # Example: (2.45 - 2.34) / 2.34 * 100 = +4.7% increase
    prob_pivot['Relativ ændring (%)'] = (
        (prob_pivot['Efter loven'] - prob_pivot['Før loven']) / prob_pivot['Før loven']
    ) * 100

    # Why both? Absolute change shows magnitude, relative change shows proportional change


    # PRINT RESULTS
    print("\n" + "=" * 70)
    print("PREDICTED PROBABILITIES FOR SEVERE DELAY (>=180 min)")
    print("=" * 70)
    print(prob_pivot.round(2))  # Round to 2 decimal places for readability

    return results_df, prob_pivot


def plot_predicted_probabilities(prob_pivot):
    """
    Purpose: Create side-by-side visualizations of predicted probabilities

    Plot 1: Horizontal bar chart comparing Period 1 vs Period 2 for each airline
    Plot 2: Bar chart showing the change (improvement or worsening)

    Color coding:
    - Blue = Period 1 (before law)
    - Red = Period 2 (after law)
    - Green bars = Improvement (delays decreased)
    - Red bars = Worsening (delays increased)
    """

    # CREATE FIGURE WITH TWO SUBPLOTS
    # 1 row, 2 columns → side-by-side plots
    # figsize=(16, 8) → wide figure to fit both plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # axes is an array: axes[0] = left plot, axes[1] = right plot


    # SORT DATA
    # Sort airlines by Period 2 probability (lowest to highest)
    # ascending=True → lowest delay rate at bottom, highest at top
    # This makes the plot easier to read
    sorted_pivot = prob_pivot.sort_values('Efter loven', ascending=True)


    # =========================================================================
    # PLOT 1: BEFORE VS AFTER COMPARISON
    # =========================================================================
    ax1 = axes[0]  # Get first subplot

    # Create y-axis positions (0, 1, 2, 3, ...)
    # One position per airline
    y_pos = np.arange(len(sorted_pivot))


    # CREATE HORIZONTAL BARS
    # .barh() creates horizontal bars (x = value, y = position)
    # y_pos - 0.2 → shifts Period 1 bars slightly down
    ax1.barh(y_pos - 0.2,                      # Y positions (shifted down)
             sorted_pivot['Før loven'],        # Bar lengths (probabilities)
             0.4,                              # Bar height
             label='Før loven',                # Legend label
             color='#3498db',                  # Blue
             alpha=0.8)                        # 80% opaque

    # y_pos + 0.2 → shifts Period 2 bars slightly up
    ax1.barh(y_pos + 0.2,                      # Y positions (shifted up)
             sorted_pivot['Efter loven'],      # Bar lengths
             0.4,                              # Bar height
             label='Efter loven',              # Legend label
             color='#e74c3c',                  # Red
             alpha=0.8)                        # 80% opaque

    # Why horizontal bars? Easier to read airline names (no rotation needed)


    # ADD LABELS AND FORMATTING
    ax1.set_yticks(y_pos)                           # Y-axis tick positions
    ax1.set_yticklabels(sorted_pivot.index)         # Y-axis labels (airline codes)
    ax1.set_xlabel('Sandsynlighed for Severe Delay (%)')  # X-axis label
    ax1.set_title('Predicted Probability for Forsinkelse >=180 min\nper Airline',
                  fontweight='bold')                # Title
    ax1.legend()                                    # Show legend
    ax1.grid(axis='x', alpha=0.3)                   # Vertical gridlines


    # =========================================================================
    # PLOT 2: CHANGE IN PROBABILITY
    # =========================================================================
    ax2 = axes[1]  # Get second subplot

    # CREATE COLOR ARRAY
    # Green if change is negative (improvement), red if positive (worsening)
    # List comprehension: [expression for item in list]
    colors = ['#27ae60' if x < 0 else '#e74c3c'
              for x in sorted_pivot['Ændring (%-point)']]

    # Why conditional colors? Makes it immediately clear which airlines improved


    # CREATE HORIZONTAL BARS
    ax2.barh(y_pos,                                 # Y positions
             sorted_pivot['Ændring (%-point)'],     # Bar lengths (changes)
             color=colors,                          # Color array
             alpha=0.8)                             # 80% opaque

    # ADD VERTICAL LINE AT X=0
    # This line represents "no change"
    # Bars to the left = improvement, bars to the right = worsening
    ax2.axvline(x=0,                                # X position (vertical line at 0)
                color='black',                      # Black line
                linestyle='-',                      # Solid line
                linewidth=0.5)                      # Thin line


    # ADD LABELS AND FORMATTING
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_pivot.index)
    ax2.set_xlabel('Ændring i Sandsynlighed (%-point)')
    ax2.set_title('Ændring i Delay-sandsynlighed\nEfter Lovgivning (Grøn = Fald, Rød = Stigning)',
                  fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)


    # ADD VALUE LABELS TO BARS
    # Loop through changes and airline names simultaneously
    # enumerate() gives (index, value) pairs: (0, item0), (1, item1), ...
    # zip() combines two lists: zip([1,2], ['a','b']) → [(1,'a'), (2,'b')]
    for i, (v, airline) in enumerate(zip(sorted_pivot['Ændring (%-point)'],
                                         sorted_pivot.index)):

        # Annotate each bar with its value
        ax2.annotate(f'{v:+.2f}',                   # Text: +0.11 or -0.05 (:.2f = 2 decimals)
                     xy=(v, i),                      # Position at end of bar
                     xytext=(5 if v >= 0 else -5, 0), # Offset: right if positive, left if negative
                     textcoords='offset points',     # xytext is relative offset
                     ha='left' if v >= 0 else 'right',  # Horizontal alignment
                     va='center',                    # Vertical alignment (centered)
                     fontsize=9)

    # Why add labels? Shows exact change values without reading axis


    # SAVE AND DISPLAY
    plt.tight_layout()                              # Adjust spacing
    plt.savefig('predicted_probabilities.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


# =============================================================================
# SECTION 4: ODDS RATIOS AND FOREST PLOT
# =============================================================================
# Translate model coefficients to odds ratios (easier to interpret)

def calculate_odds_ratios(model):
    """
    Purpose: Convert logistic regression coefficients to odds ratios

    WHAT ARE ODDS RATIOS?
    - Odds = P(event) / P(no event)
    - Example: If P(delay) = 0.2, then Odds = 0.2/0.8 = 0.25

    - Odds Ratio (OR) = Odds₁ / Odds₀
    - Compares odds between two groups

    INTERPRETATION:
    - OR = 1.0 → No effect (odds are equal)
    - OR = 1.5 → Group 1 has 50% higher odds than Group 0
    - OR = 0.7 → Group 1 has 30% lower odds than Group 0

    WHY ODDS RATIOS?
    - Coefficients are in log-odds (hard to interpret)
    - Odds ratios are more intuitive: "1.5x more likely"

    MATH:
    - Coefficient (β) is log-odds ratio
    - OR = exp(β)
    - Example: β = 0.405 → OR = exp(0.405) = 1.5

    Returns: DataFrame with odds ratios and confidence intervals
    """

    # CALCULATE ODDS RATIOS
    # model.params contains the coefficients (β values)
    # np.exp() takes exponential: e^β
    # This converts log-odds to odds ratios
    odds_ratios = np.exp(model.params)

    # Example:
    # Coefficient β = 0.405 → OR = exp(0.405) = 1.50
    # Interpretation: 50% higher odds


    # CALCULATE CONFIDENCE INTERVALS
    # model.conf_int() returns 95% confidence intervals for coefficients
    # These are in log-odds, so we exponential them too
    conf_int = np.exp(model.conf_int())

    # What is a confidence interval?
    # - Range of plausible values for the true odds ratio
    # - 95% CI means: "95% confident true value is in this range"
    # - Narrow CI = precise estimate, wide CI = uncertain estimate

    # Example: OR = 1.50, 95% CI [1.20, 1.85]
    # Interpretation: Best estimate is 1.5x higher odds
    #                 But true value could be anywhere from 1.2x to 1.85x


    # CREATE RESULTS DATAFRAME
    or_df = pd.DataFrame({
        'Odds Ratio': odds_ratios,      # Point estimates
        'CI Lower': conf_int[0],        # Lower bound of 95% CI
        'CI Upper': conf_int[1],        # Upper bound of 95% CI
        'p-value': model.pvalues        # Statistical significance
    })

    # What is p-value?
    # - Probability of observing this coefficient if true effect is zero
    # - Small p-value (< 0.05) → statistically significant
    # - Large p-value → could be due to random chance


    # MARK SIGNIFICANT RESULTS
    # Create boolean column: True if p-value < 0.05
    or_df['Significant'] = or_df['p-value'] < 0.05

    # Why 0.05? Common threshold in statistics
    # p < 0.05 means "less than 5% chance of false positive"


    # PRINT RESULTS
    print("\n" + "=" * 70)
    print("ODDS RATIOS")
    print("=" * 70)
    print(or_df.round(4))  # Round to 4 decimal places

    return or_df


def plot_forest_plot(or_df):
    """
    Purpose: Create forest plot of interaction effects

    WHAT IS A FOREST PLOT?
    - Visualization of odds ratios with confidence intervals
    - Horizontal line = confidence interval
    - Dot = point estimate (odds ratio)
    - Vertical line at OR=1 = "no effect" reference

    WHY FOCUS ON INTERACTIONS?
    - Main effects show average differences
    - Interactions show how law affected each airline differently
    - This is the key research question!

    INTERPRETATION:
    - OR > 1 (red) → Law increased delay odds for that airline
    - OR < 1 (green) → Law decreased delay odds for that airline
    - CI crosses 1 → Not statistically significant
    """

    # FILTER TO INTERACTION TERMS ONLY
    # Interaction terms contain ':C' in their name
    # Example: 'C(airline)[T.DL]:C(period)[T.1]'
    # .str.contains() checks if string contains substring
    interaction_terms = or_df[or_df.index.str.contains(':C')]

    # Why filter? Forest plot would be too cluttered with all terms
    # Interactions are most interesting for this analysis


    # CHECK IF ANY INTERACTIONS EXIST
    if len(interaction_terms) == 0:
        print("Ingen interaktionsled fundet til forest plot")
        return None


    # CREATE FIGURE
    fig, ax = plt.subplots(figsize=(12, 10))


    # SORT BY ODDS RATIO
    # Show best performers (lowest OR) at bottom, worst at top
    interaction_terms = interaction_terms.sort_values('Odds Ratio')

    # Create y-axis positions
    y_pos = np.arange(len(interaction_terms))


    # CREATE COLOR ARRAY
    # Green if OR < 1 (improvement), red if OR > 1 (worsening)
    colors = ['#27ae60' if x < 1 else '#e74c3c'
              for x in interaction_terms['Odds Ratio']]


    # PLOT ERROR BARS (CONFIDENCE INTERVALS)
    # ax.errorbar() creates points with horizontal error bars

    # Calculate error bar lengths
    # xerr must be array of [left_errors, right_errors]
    # Left error = OR - CI_lower
    # Right error = CI_upper - OR
    ax.errorbar(
        interaction_terms['Odds Ratio'],                         # X positions (OR values)
        y_pos,                                                   # Y positions
        xerr=[interaction_terms['Odds Ratio'] - interaction_terms['CI Lower'],  # Left errors
              interaction_terms['CI Upper'] - interaction_terms['Odds Ratio']],  # Right errors
        fmt='o',              # Format: 'o' = circle markers
        color='black',        # Marker color
        ecolor='gray',        # Error bar color
        capsize=3,            # Cap width at ends of error bars
        markersize=8          # Marker size
    )


    # COLOR THE POINTS
    # Loop through odds ratios and colors
    for i, (or_val, color) in enumerate(zip(interaction_terms['Odds Ratio'], colors)):
        ax.scatter(or_val,    # X position
                   i,         # Y position
                   c=color,   # Color
                   s=100,     # Size
                   zorder=5)  # Z-order (draw on top of error bars)


    # ADD REFERENCE LINE AT OR = 1
    # This represents "no effect" - odds didn't change
    ax.axvline(x=1,                     # Vertical line at x=1
               color='red',             # Red color
               linestyle='--',          # Dashed line
               linewidth=1,             # Line width
               label='No effect (OR=1)') # Legend label

    # Why at 1? OR=1 means odds ratio of 1:1 (no change)


    # CLEAN UP Y-AXIS LABELS
    # Model generates ugly names like 'C(airline, Treatment(reference="AA"))[T.DL]:C(period)[T.1]'
    # Clean them to just show airline codes

    # Chain of .replace() calls to remove unwanted text
    cleaned_labels = []
    for name in interaction_terms.index:
        # Remove all the formula syntax, keep only airline code
        clean = (name.replace('C(airline, Treatment(reference="', '')
                     .replace('"))', '')
                     .replace('[T.', ' ')
                     .replace(']:C(period)[T.1]', ''))
        cleaned_labels.append(clean)


    # ADD LABELS AND FORMATTING
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cleaned_labels)
    ax.set_xlabel('Odds Ratio (95% CI)')
    ax.set_title('Interaktionseffekt: Airline × Periode\n(OR < 1 = Lavere odds efter loven)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)


    # SAVE AND DISPLAY
    plt.tight_layout()
    plt.savefig('forest_plot_interactions.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


# =============================================================================
# SECTION 5: MODEL VALIDATION
# =============================================================================
# Check if model makes accurate predictions on new data

def validate_model_sklearn(df):
    """
    Purpose: Validate model performance using train/test split and cross-validation

    WHY VALIDATE?
    - Ensure model generalizes to new data (not just memorizing training data)
    - Quantify prediction accuracy
    - Check for overfitting

    VALIDATION METHODS:
    1. Train/test split - Hold out 20% of data for testing
    2. Cross-validation - Split data into 5 folds, test on each

    METRICS USED:
    - Accuracy: Percentage of correct predictions
    - Classification Report: Precision, Recall, F1-score for each class
    - Cross-validation scores: Model performance across different data splits

    Note: Uses sklearn instead of statsmodels for prediction-focused validation
    """

    print("\n" + "=" * 70)
    print("MODEL VALIDERING (Scikit-learn)")
    print("=" * 70)


    # PREPARE FEATURES (X) AND TARGET (y)

    # Create dummy variables for categorical predictors
    # pd.get_dummies() converts categories to binary columns
    # Example: airline='AA' → airline_AA=1, airline_DL=0, airline_UA=0
    # drop_first=True removes one category to avoid multicollinearity
    X = pd.get_dummies(df[['airline', 'period']], drop_first=True)

    # What is multicollinearity?
    # - When predictors are perfectly correlated
    # - Example: airline_AA + airline_DL + airline_UA = 1 (always)
    # - Causes problems in regression (matrix not invertible)
    # - Solution: Drop one category (reference category)

    # Target variable (what we're predicting)
    y = df['Binary_delay']


    # TRAIN/TEST SPLIT
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,                       # Data to split
        test_size=0.2,              # 20% for testing
        random_state=42,            # Random seed (for reproducibility)
        stratify=y                  # Keep same proportion of 0/1 in both sets
    )

    # Why split?
    # - Train model on training set
    # - Test on unseen data (test set)
    # - If performance is similar → model generalizes well
    # - If training >> testing → overfitting

    # What is stratify?
    # - Ensures both sets have same delay rate
    # - Important for imbalanced data (rare events)
    # - Without stratify: might get all delays in training set by chance


    # FIT MODEL
    # Create logistic regression model
    model = LogisticRegression(
        max_iter=1000,      # Maximum iterations for convergence
        random_state=42     # Random seed for reproducibility
    )

    # Train model on training data
    # .fit() finds coefficients that minimize prediction error
    model.fit(X_train, y_train)


    # MAKE PREDICTIONS ON TEST SET

    # Predict class (0 or 1)
    # Uses threshold: if P(delay) > 0.5 → predict 1, else predict 0
    y_pred = model.predict(X_test)

    # Predict probabilities
    # .predict_proba() returns array with [P(class=0), P(class=1)]
    # [:, 1] selects second column (P(class=1))
    y_pred_proba = model.predict_proba(X_test)[:, 1]


    # CALCULATE METRICS

    # Accuracy: Percentage of correct predictions
    # (True Positives + True Negatives) / Total
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Classification Report
    # Shows precision, recall, F1-score for each class
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['No Severe Delay', 'Severe Delay']))

    # What are these metrics?
    # - Precision: Of predicted delays, how many were actually delayed?
    # - Recall: Of actual delays, how many did we predict?
    # - F1-score: Harmonic mean of precision and recall


    # CROSS-VALIDATION
    # Split data into 5 folds, train on 4, test on 1 (repeat 5 times)
    # More robust than single train/test split
    cv_scores = cross_val_score(
        model, X, y,           # Model and data
        cv=5,                  # 5-fold cross-validation
        scoring='accuracy'     # Metric to calculate
    )

    # Print mean and standard deviation of CV scores
    # std * 2 approximates 95% confidence interval
    print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Why cross-validation?
    # - Uses all data for both training and testing
    # - More stable estimate of performance
    # - Detects if performance varies across data subsets

    return model, cv_scores


# =============================================================================
# SECTION 6: MAIN FUNCTION
# =============================================================================
# Orchestrates the entire analysis workflow

def main(filepath):
    """
    Purpose: Run complete analysis pipeline from data loading to results

    WORKFLOW:
    1. Load and prepare data
    2. Calculate descriptive statistics
    3. Fit logistic regression model
    4. Calculate predicted probabilities
    5. Calculate odds ratios
    6. Validate model
    7. Save results

    This function ties together all previous functions into a complete analysis

    Returns: Dictionary with all results (models, tables, statistics)
    """

    # Print header
    print("=" * 70)
    print("LOGISTISK REGRESSION: FLIGHT DELAY ANALYSE")
    print("Sandsynlighed for Severe Delay (>=180 min) foer og efter lovgivning")
    print("=" * 70)


    # STEP 1: LOAD AND PREPARE DATA
    print("\n>>> STEP 1: Indlæser og forbereder data...")
    df = load_and_prepare_data(filepath)           # Load raw data
    model_df = create_model_data(df)               # Select model variables
    # After Step 1: Have clean dataset with airline, period, Binary_delay


    # STEP 2: DESCRIPTIVE STATISTICS
    print("\n>>> STEP 2: Beregner deskriptiv statistik...")
    delay_stats, desc_pivot = descriptive_statistics(model_df)  # Calculate delay rates
    plot_delay_rates(delay_stats)                                # Visualize delay rates
    # After Step 2: Know which airlines have high/low delay rates in each period


    # STEP 3: LOGISTIC REGRESSION
    print("\n>>> STEP 3: Kører logistisk regression...")
    model, ref_airline = logistic_regression_statsmodels(model_df)
    # After Step 3: Have statistical model with coefficients and p-values


    # STEP 4: PREDICTED PROBABILITIES
    print("\n>>> STEP 4: Beregner predicted probabilities...")
    prob_df, prob_pivot = calculate_probabilities(model, model_df, ref_airline)
    plot_predicted_probabilities(prob_pivot)
    # After Step 4: Know exact delay probability for each airline in each period


    # STEP 5: ODDS RATIOS
    print("\n>>> STEP 5: Beregner odds ratios...")
    or_df = calculate_odds_ratios(model)           # Calculate odds ratios
    plot_forest_plot(or_df)                        # Visualize in forest plot
    # After Step 5: Know which airlines improved/worsened after law


    # STEP 6: MODEL VALIDATION
    print("\n>>> STEP 6: Validerer model...")
    sklearn_model, cv_scores = validate_model_sklearn(model_df)
    # After Step 6: Know how well model predicts on new data


    # STEP 7: SAVE RESULTS
    print("\n>>> STEP 7: Gemmer resultater...")
    # Save key results to CSV files for later analysis
    prob_pivot.to_csv('predicted_probabilities.csv')   # Probability table
    or_df.to_csv('odds_ratios.csv')                    # Odds ratios table
    desc_pivot.to_csv('descriptive_statistics.csv')    # Descriptive stats table


    # PRINT SUMMARY
    print("\n" + "=" * 70)
    print("ANALYSE FÆRDIG!")
    print("=" * 70)
    print("\nGenererede filer:")
    print("  - delay_rates_by_airline.png")           # Bar chart visualization
    print("  - predicted_probabilities.png")          # Probability visualization
    print("  - forest_plot_interactions.png")         # Forest plot
    print("  - predicted_probabilities.csv")          # Probability data
    print("  - odds_ratios.csv")                      # Odds ratio data
    print("  - descriptive_statistics.csv")           # Summary statistics


    # RETURN RESULTS
    # Return dictionary with all key results
    # This allows accessing results after running: results = main(filepath)
    return {
        'model': model,                      # Fitted statsmodels model
        'probabilities': prob_pivot,         # Probability table
        'odds_ratios': or_df,                # Odds ratios table
        'descriptive_stats': desc_pivot      # Descriptive statistics
    }


# =============================================================================
# SECTION 7: SCRIPT EXECUTION
# =============================================================================
# This section runs when script is executed directly (not imported)

if __name__ == "__main__":
    # What is if __name__ == "__main__"?
    # - This code only runs if script is executed directly (python Log_reg_model.py)
    # - Doesn't run if script is imported in another file (import Log_reg_model)
    # - Allows using this file both as script and as module


    # SPECIFY DATA FILE PATH
    filepath = "din_flight_data.csv"  # Placeholder - user should change this

    # Why placeholder? Actual path is hardcoded in load_and_prepare_data()
    # This is just for reference/documentation


    # RUN ANALYSIS
    results = main(filepath)
    # This executes the entire analysis pipeline
    # After completion:
    # - 6 files are created (3 PNG images, 3 CSV tables)
    # - results variable contains all output (model, tables, statistics)


    # Note: This message still prints because main() runs before this point
    print("For at køre scriptet, opdater 'filepath' variablen")
    print("og fjern kommentar fra 'results = main(filepath)'")
    # This is outdated - the script actually runs now!
