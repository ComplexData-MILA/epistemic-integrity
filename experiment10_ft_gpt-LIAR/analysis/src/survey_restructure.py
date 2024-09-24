import pandas as pd

# Function to extract the correct option number from the 'Correct_Answer' column


def get_correct_option(correct_answer):
    # Extract the letter (A, B, C, D) and map it to a number (1, 2, 3, 4)
    option_letter = correct_answer.split(')')[0]
    return {'A': 1, 'B': 2, 'C': 3, 'D': 4}.get(option_letter)

# Function to calculate attention check score for each respondent


def calculate_score(row, question_columns, attention_columns, first_data_subset):
    score = 0
    for i, question_col in enumerate(question_columns):
        attention_col = attention_columns[i]
        question_text = row[question_col]

        # Skip if the question text is NaN
        if pd.isna(question_text):
            continue

        # Find the correct option for the question text
        matching_row = first_data_subset[first_data_subset['Attention_Check_Question'] == question_text]
        if not matching_row.empty:
            correct_option = matching_row['correct_option'].values[0]
            if f"${{e://Field/RandomOption{correct_option}_{i+1}}}" == row[attention_col]:
                score += 1
        else:
            print(
                f"No matching correct answer found for question: {question_text}")

    return score


# Load the datasets
#first_dataset_path = '/Users/bijeanghafouri/Dropbox/My Mac (Bijean’s MacBook Pro)/Documents/University/USC/projects/epistemic-integrity/dta/p7/llm_with_attention_checks.csv'
first_dataset_path = '../dta/llm_with_attention_checks.csv'
#survey_responses_file_path = '/Users/bijeanghafouri/Dropbox/My Mac (Bijean’s MacBook Pro)/Downloads/Assertivity_June 14, 2024_14.46.csv'
survey_responses_file_path = '../dta/Assertivity_June 14, 2024_14.46.csv'

first_data = pd.read_csv(first_dataset_path)
survey_responses_data = pd.read_csv(survey_responses_file_path)

# Extract relevant columns from the first dataset
first_data_subset = first_data[[
    'Attention_Check_Question', 'Correct_Answer']].copy()
first_data_subset['correct_option'] = first_data_subset['Correct_Answer'].apply(
    get_correct_option)

# Extract relevant columns from the survey responses dataset
attention_columns = ['attention_1', 'attention_2',
                     'attention_3', 'attention_4', 'attention_5']
question_columns = ['RandomQuestion1', 'RandomQuestion2',
                    'RandomQuestion3', 'RandomQuestion4', 'RandomQuestion5']
text_columns = ['RandomText1', 'RandomText3', 'RandomText4', 'RandomText5']
rating_columns = ['outcome_1_1', 'outcome_3_1', 'outcome_4_1', 'outcome_5_1']
answer_columns = ['RandomAnswer1', 'RandomAnswer3',
                  'RandomAnswer4', 'RandomAnswer5']

# Ensure the attention columns exist in the dataset
assert all(col in survey_responses_data.columns for col in attention_columns), "Not all attention columns are present in the survey responses data."

# Compute the attention check score for each respondent
survey_responses_data['attention_verification_1'] = survey_responses_data.apply(
    lambda row: calculate_score(row, question_columns, attention_columns, first_data_subset), axis=1)

# Add the new attention check verification
survey_responses_data['attention_verification_2'] = survey_responses_data['outcome_2_1'].apply(
    lambda x: 1 if '6' in str(x) else 0)

# Count the total number of respondents before filtering
total_respondents_before = len(survey_responses_data)

# Filter respondents based on attention checks
filtered_responses_data = survey_responses_data[
    (survey_responses_data['attention_verification_1'] >= 4) &
    (survey_responses_data['attention_verification_2'] == 1)
]

# Count the total number of respondents after filtering
total_respondents_after = len(filtered_responses_data)

# Calculate the number of respondents filtered out
filtered_out_count = total_respondents_before - total_respondents_after

print(f"Total respondents before filtering: {total_respondents_before}")
print(f"Total respondents after filtering: {total_respondents_after}")
print(f"Number of respondents filtered out: {filtered_out_count}")

# Save the filtered responses data
output_path_filtered = '../out/filtered_survey_responses.csv'
filtered_responses_data.to_csv(output_path_filtered, index=False)

print(f"Filtered survey responses saved to {output_path_filtered}")

# Ensure respondent identifier exists
filtered_responses_data['respondent_id'] = filtered_responses_data.index

# Melt the survey responses data to have a long format DataFrame
survey_text_df = filtered_responses_data.melt(
    id_vars='respondent_id', value_vars=text_columns, var_name='response', value_name='text')
survey_ratings_df = filtered_responses_data.melt(
    id_vars='respondent_id', value_vars=rating_columns, var_name='rating', value_name='subjective_assertivity')
survey_answers_df = filtered_responses_data.melt(
    id_vars='respondent_id', value_vars=answer_columns, var_name='answer', value_name='gpt-answer')

# Combine the melted DataFrames
survey_combined_df = pd.concat([survey_text_df['respondent_id'], survey_text_df['text'],
                                survey_answers_df['gpt-answer'], survey_ratings_df['subjective_assertivity']], axis=1)

# Convert subjective_assertivity to numeric, coercing errors to NaN
survey_combined_df['subjective_assertivity'] = pd.to_numeric(
    survey_combined_df['subjective_assertivity'], errors='coerce')

# Compute mean subjective assertivity and ratings count for each explanation
agg_subjective_assertivity = survey_combined_df.groupby(
    'gpt-answer')['subjective_assertivity'].agg(['mean', 'count', 'std']).reset_index()
agg_subjective_assertivity.rename(columns={'mean': 'mean_subjective_assertivity',
                                           'count': 'ratings_count', 'std': 'std_dev_subjective_assertivity'}, inplace=True)

# Normalize subjective assertivity scores
agg_subjective_assertivity['mean_subjective_assertivity'] = (agg_subjective_assertivity['mean_subjective_assertivity'] - agg_subjective_assertivity['mean_subjective_assertivity'].min()) / (
    agg_subjective_assertivity['mean_subjective_assertivity'].max() - agg_subjective_assertivity['mean_subjective_assertivity'].min())

# Compute descriptive statistics
descriptive_stats = agg_subjective_assertivity.describe()

# Display the descriptive statistics
print("Descriptive Statistics of Subjective Assertivity Scores:")
print(descriptive_stats)

# Extract relevant columns from the first dataset for merging
first_data_subset = first_data[['label_id', 'text', 'label',
                                'calibrated-uncertainty', 'assertivity', 'Assertive', 'gpt-answer', 'ft_gpt_response', 'ft_gpt_norm']]

# Recalibrate the 'calibrated-uncertainty' column
first_data_subset['recalibrated_uncertainty'] = first_data_subset['calibrated-uncertainty'].apply(
    lambda x: abs(0.5 - x) * 2)

# Merge the two datasets on 'gpt-answer' (the explanation text) with a left join
merged_data = pd.merge(
    first_data_subset, agg_subjective_assertivity, on='gpt-answer', how='left')

# Save the merged dataset
output_path_merged = '../out/survey_merged_data.csv'
merged_data.to_csv(output_path_merged, index=False)

print(f"Merged data saved to {output_path_merged}")

# Compute descriptive statistics for the merged data
descriptive_stats_merged = merged_data.describe()

# Display the descriptive statistics for the merged data
print("Descriptive Statistics of Merged Data:")
print(descriptive_stats_merged)

# Save the standard deviation data
output_path_std = '../out/std_dev_subjective_assertivity.csv'
agg_subjective_assertivity.to_csv(output_path_std, index=False)

print(
    f"Standard deviation of subjective assertivity scores saved to {output_path_std}")
