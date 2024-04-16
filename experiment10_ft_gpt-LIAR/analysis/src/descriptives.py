# Descriptive statistics and variance analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged dataset
merged_data_path = '../out/survey_merged_data.csv'
merged_data = pd.read_csv(merged_data_path)

print(merged_data.columns)

# 1. Density plot of the mean assertivity scores for each explanation
# plt.figure(figsize=(10, 6))
# sns.kdeplot(merged_data['mean_subjective_assertivity'].dropna(), shade=True)
sns.distplot(merged_data['mean_subjective_assertivity'].dropna(), kde=True)
plt.xlim(0, 1)
plt.title('Mean Subjective Assertivity Scores for Each Explanation')
plt.xlabel('Mean Subjective Assertivity Score (0-1)')
plt.ylabel('Density')
# plt.grid(True)
# plt.savefig('../out/plots/density_mean_subjective_assertivity.png')
plt.show()

# 2. Combined density plot of the mean assertivity scores for each explanation, for each level of Assertivity
# plt.figure(figsize=(10, 6))
assertivity_levels = {-1: 'Low', 0: 'Medium', 1: 'High'}
colors = {-1: 'blue', 0: 'green', 1: 'red'}

for level, label in assertivity_levels.items():
    subset = merged_data[merged_data['Assertive'] == level]
    # sns.kdeplot(subset['mean_subjective_assertivity'].dropna(),
    #             shade=True, color=colors[level], label=label)
    sns.distplot(subset['mean_subjective_assertivity'].dropna(),
                kde=True, color=colors[level], label=label, hist_kws={'alpha':0.3})

plt.xlim(0, 1)
plt.title('Mean Subjective Assertivity Scores by Assertivity Level')
plt.xlabel('Mean Subjective Assertivity Score (0-1)')
plt.ylabel('Density')
plt.legend(title='Assertivity Level')
# plt.grid(True)
# plt.savefig('../out/plots/density_mean_subjective_assertivity_by_level.png')
plt.show()


# 3. Compute the variance of human assertivity scores for a given explanation
# Load the survey responses dataset
survey_responses_file_path = '../out/survey_responses_with_attentionchecks.csv'
survey_responses_data = pd.read_csv(survey_responses_file_path)

# Define the relevant columns
text_columns = ['RandomText1', 'RandomText3', 'RandomText4', 'RandomText5']
rating_columns = ['outcome_1_1', 'outcome_3_1', 'outcome_4_1', 'outcome_5_1']
answer_columns = ['RandomAnswer1', 'RandomAnswer3',
                  'RandomAnswer4', 'RandomAnswer5']

# Ensure respondent identifier exists
survey_responses_data['respondent_id'] = survey_responses_data.index

# Melt the survey responses data to have a long format DataFrame
survey_text_df = survey_responses_data.melt(
    id_vars='respondent_id', value_vars=text_columns, var_name='response', value_name='text')
survey_ratings_df = survey_responses_data.melt(
    id_vars='respondent_id', value_vars=rating_columns, var_name='rating', value_name='subjective_assertivity')
survey_answers_df = survey_responses_data.melt(
    id_vars='respondent_id', value_vars=answer_columns, var_name='answer', value_name='gpt-answer')

# Combine the melted DataFrames
survey_combined_df = pd.concat([survey_text_df['respondent_id'], survey_text_df['text'],
                                survey_answers_df['gpt-answer'], survey_ratings_df['subjective_assertivity']], axis=1)

# Convert subjective_assertivity to numeric, coercing errors to NaN
survey_combined_df['subjective_assertivity'] = pd.to_numeric(
    survey_combined_df['subjective_assertivity'], errors='coerce')

# Compute variance of human assertivity scores for each explanation
variance_subjective_assertivity = survey_combined_df.groupby(
    'gpt-answer')['subjective_assertivity'].var().reset_index()
variance_subjective_assertivity.rename(
    columns={'subjective_assertivity': 'variance_subjective_assertivity'}, inplace=True)

# Save the variance data
output_path_variance = '../out/variance_subjective_assertivity.csv'
variance_subjective_assertivity.to_csv(output_path_variance, index=False)

print(
    f"Variance of subjective assertivity scores saved to {output_path_variance}")

# Merge the variance data with the merged data
merged_data_with_variance = pd.merge(
    merged_data, variance_subjective_assertivity, on='gpt-answer', how='left')

# Save the merged dataset with variance
output_path_merged_with_variance = '../out/survey_merged_data_with_variance.csv'
merged_data_with_variance.to_csv(output_path_merged_with_variance, index=False)

print(f"Merged data with variance saved to {output_path_merged_with_variance}")

# Plot the variance of human assertivity scores for each explanation using a density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(
    variance_subjective_assertivity['variance_subjective_assertivity'].dropna(), shade=True)
plt.title(
    'Density Plot of Variance in Subjective Assertivity Scores for Each Explanation')
plt.xlabel('Variance in Subjective Assertivity Score')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('../out/plots/density_variance_subjective_assertivity.png')
plt.show()

# Additional Analysis: Compare standard deviations to benchmarks

# Calculate standard deviation of subjective assertivity scores
std_dev_subjective_assertivity = variance_subjective_assertivity['variance_subjective_assertivity'].apply(
    lambda x: x**0.5)

# 1. Compare standard deviations to the range of scores
range_of_scores = std_dev_subjective_assertivity.max(
) - std_dev_subjective_assertivity.min()
print(
    f"Range of standard deviations of subjective assertivity scores: {range_of_scores}")

# 2. Compare standard deviations to the mean of scores
mean_of_std_dev = std_dev_subjective_assertivity.mean()
print(
    f"Mean of standard deviations of subjective assertivity scores: {mean_of_std_dev}")

# 3. Distribution visualization of standard deviations
plt.figure(figsize=(10, 6))
sns.histplot(std_dev_subjective_assertivity.dropna(), kde=True, bins=30)
plt.title('Distribution of Standard Deviations of Subjective Assertivity Scores')
plt.xlabel('Standard Deviation of Subjective Assertivity Scores')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('../out/plots/distribution_std_dev_subjective_assertivity.png')
plt.show()

# 4. Calculate Interquartile Range (IQR)
Q1 = std_dev_subjective_assertivity.quantile(0.25)
Q3 = std_dev_subjective_assertivity.quantile(0.75)
IQR = Q3 - Q1
print(f"Interquartile Range (IQR) of standard deviations: {IQR}")
