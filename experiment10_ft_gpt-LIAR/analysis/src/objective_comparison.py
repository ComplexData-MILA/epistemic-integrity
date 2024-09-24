import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import numpy as np
import os
from scipy.stats import pearsonr

# Load the merged dataset
merged_data_path = '../out/survey_merged_data_with_variance.csv'
merged_data = pd.read_csv(merged_data_path)

print(merged_data.columns)


# 2. Combined density plot of the mean subjective assertivity scores for each level of Assertivity
plt.figure(figsize=(10, 6))
assertivity_levels = [-1, 0, 1]
colors = ['blue', 'green', 'red']
for i, level in enumerate(assertivity_levels):
    subset = merged_data[merged_data['Assertive'] == level]
    sns.distplot(subset['mean_subjective_assertivity'].dropna(
    ), kde=True, label=f'Assertivity Level {level}', color=colors[i])
plt.xlim(0, 1)
plt.title('Density Plot of Mean Subjective Assertivity Scores by Assertivity Level')
plt.xlabel('Mean Subjective Assertivity Score (0-1)')
plt.ylabel('Density')
plt.legend()
# plt.grid(True)
# plt.savefig('../out/plots/density_mean_subjective_assertivity_by_level.png')
plt.show()

# 3. Density plot of the objective assertivity scores by assertivity level
plt.figure(figsize=(10, 6))
for i, level in enumerate(assertivity_levels):
    subset = merged_data[merged_data['Assertive'] == level]
    # sns.kdeplot(subset['assertivity'].dropna(), shade=True,
    #             label=f'Assertivity Level {level}', color=colors[i])
    sns.distplot(subset['ft_gpt_norm'].dropna(), kde=True,
                label=f'Assertivity Level {level}', color=colors[i], hist_kws={'alpha':0.3})
plt.xlim(0, 1)
plt.title('Density Plot of Objective Assertivity Scores by Assertivity Level')
plt.xlabel('Objective Assertivity Score (0-1)')
plt.ylabel('Density')
plt.legend()
# plt.grid(True)
# plt.savefig('../out/plots/density_mean_objective_assertivity_by_level.png')
plt.show()


# Calculate the correlation between objective and subjective assertivity scores
correlation = merged_data[['ft_gpt_norm',
                           'mean_subjective_assertivity']].corr().iloc[0, 1]

# Create the scatter plot comparing objective and subjective assertivity scores
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data, x='ft_gpt_norm', y='mean_subjective_assertivity',
                hue='Assertive', palette='viridis', alpha=0.6)
plt.plot([0, 1], [0, 1], color='red',
         linestyle='--', label='Perfect Alignment')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Comparison of Objective and Subjective Assertivity Scores')
plt.xlabel('Objective Assertivity Score (0-1)')
plt.ylabel('Mean Subjective Assertivity Score (0-1)')

# Add the correlation to the legend
plt.legend(
    title=f'Assertive Level\nCorrelation: {correlation:.2f}', loc='upper left')
# plt.grid(True)
# plt.savefig('../out/plots/subjective_objective_correlation.png')
plt.show()


# Create a 2x2 grid of plots of comparing internal certainty with objective and subjective assertivity
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Density plot of recalibrated_uncertainty and objective assertivity (full sample)
sns.distplot(merged_data['recalibrated_uncertainty'].dropna(
), label='Recalibrated Uncertainty', kde=True, ax=axes[0, 0], hist_kws={'alpha':0.3})
sns.distplot(merged_data['ft_gpt_norm'].dropna(),
            label='Objective Assertivity', kde=True, ax=axes[0, 0], hist_kws={'alpha':0.3})
axes[0, 0].set_title(
    'Density Plot of Recalibrated Uncertainty and Objective Assertivity (Full Sample)')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
# axes[0, 0].grid(True)

# Plot 2: Density plot of recalibrated_uncertainty and objective assertivity by assertivity level
assertivity_levels = [-1, 0, 1]
colors = ['blue', 'green', 'red']
for i, level in enumerate(assertivity_levels):
    subset = merged_data[merged_data['Assertive'] == level]
    sns.distplot(subset['ft_gpt_norm'].dropna(), kde=True,
                label=f'Objective Assertivity Level {level}', color=colors[i], ax=axes[0, 1], hist_kws={'alpha':0.3})
sns.distplot(merged_data['recalibrated_uncertainty'].dropna(
), label='Recalibrated Uncertainty', kde=True, color='black', ax=axes[0, 1], hist_kws={'alpha':0.3})
axes[0, 1].set_title(
    'Density Plot of Recalibrated Uncertainty and Objective Assertivity by Level')
axes[0, 1].set_xlabel('Score')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
# axes[0, 1].grid(True)

# Plot 3: Density plot of recalibrated_uncertainty and subjective assertivity (full sample)
sns.distplot(merged_data['recalibrated_uncertainty'].dropna(
), label='Recalibrated Uncertainty', kde=True, ax=axes[1, 0], hist_kws={'alpha':0.3})
sns.distplot(merged_data['mean_subjective_assertivity'].dropna(
), label='Subjective Assertivity', kde=True, ax=axes[1, 0], hist_kws={'alpha':0.3})
axes[1, 0].set_title(
    'Density Plot of Recalibrated Uncertainty and Subjective Assertivity (Full Sample)')
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()
# axes[1, 0].grid(True)

# Plot 4: Density plot of recalibrated_uncertainty and subjective assertivity by assertivity level
for i, level in enumerate(assertivity_levels):
    subset = merged_data[merged_data['Assertive'] == level]
    sns.distplot(subset['mean_subjective_assertivity'].dropna(), kde=True,
                label=f'Subjective Assertivity Level {level}', color=colors[i], ax=axes[1, 1], hist_kws={'alpha':0.3})
sns.distplot(merged_data['recalibrated_uncertainty'].dropna(
), label='Recalibrated Uncertainty', kde=True, color='black', ax=axes[1, 1], hist_kws={'alpha':0.3})
axes[1, 1].set_title(
    'Density Plot of Recalibrated Uncertainty and Subjective Assertivity by Level')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
# axes[1, 1].grid(True)

# Adjust layout
plt.tight_layout()
# plt.savefig('../out/plots/4x4.png')
plt.show()

# Save the plots
output_dir = '../out/plots'
# os.makedirs(output_dir, exist_ok=True)
# plt.savefig(os.path.join(output_dir, 'density_plots.png'))


# Correlations
# Function to compute correlations and p-values
def compute_correlations(data, label):
    correlations = {}
    variables = [
        ("Objective Assertivity vs. Subjective Assertivity",
         "ft_gpt_norm", "mean_subjective_assertivity"),
        ("Internal Certainty vs. Objective Assertivity",
         "recalibrated_uncertainty", "ft_gpt_norm"),
        ("Internal Certainty vs. Subjective Assertivity",
         "recalibrated_uncertainty", "mean_subjective_assertivity")
    ]
    for name, var1, var2 in variables:
        if var1 in data.columns and var2 in data.columns:
            # Drop NaN values from both columns
            valid_data = data[[var1, var2]].dropna()
            if not valid_data.empty:
                corr, p_value = pearsonr(valid_data[var1], valid_data[var2])
                significance = ''
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                correlations[name] = f"{corr:.3f}{significance}"
            else:
                correlations[name] = "N/A"
    return correlations


# Load the merged dataset
merged_data_path = '../out/survey_merged_data_with_variance.csv'
merged_data = pd.read_csv(merged_data_path)

# Overall correlations
overall_correlations = compute_correlations(merged_data, "Overall")

# Correlations by assertivity level
assertivity_levels = [-1, 0, 1]
correlations_by_level = {}
for level in assertivity_levels:
    subset = merged_data[merged_data['Assertive'] == level]
    correlations_by_level[level] = compute_correlations(
        subset, f"Assertivity Level {level}")

# Combine the correlations into a DataFrame
correlation_data = {
    'Overall': overall_correlations,
    **{f"Level {level}": correlations_by_level[level] for level in assertivity_levels}
}
correlation_df = pd.DataFrame(correlation_data)

# Create a LaTeX table
latex_table = correlation_df.to_latex(index=True, caption="Correlation Analysis with Statistical Significance",
                                      label="tab:correlations", float_format="%.3f")

# Save the LaTeX table to a file
output_path_latex = '../out/correlations.tex'
with open(output_path_latex, 'w') as f:
    f.write(latex_table)

print(f"\nLaTeX table saved to {output_path_latex}")

# Display the LaTeX table
print(latex_table)
