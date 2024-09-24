import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
#survey_data = pd.read_csv(
#    '/Users/bijeanghafouri/Dropbox/My Mac (Bijean’s MacBook Pro)/Downloads/Assertivity_June 14, 2024_14.46.csv')
survey_data = pd.read_csv(
    '../dta/Assertivity_June 14, 2024_14.46.csv')
#explanation_data = pd.read_csv(
#    '/Users/bijeanghafouri/Dropbox/My Mac (Bijean’s MacBook Pro)/Documents/University/USC/projects/epistemic-integrity/dta/p7/llm_with_attention_checks.csv')
explanation_data = pd.read_csv(
    '../dta/llm_with_attention_checks.csv')

# Preprocess survey data


def preprocess_survey_data(survey_data):
    # Education groups
    survey_data['education_group'] = survey_data['educ'].apply(
        lambda x: 'College + More' if x in ['2 year degree', '4 year degree', 'Professional degree', 'Graduate degree']
        else 'Lower than College'
    )

    # Party groups
    survey_data['party_group'] = survey_data.apply(
        lambda row: 'Democrat' if row['partyid'] == 'Democrat' or row['independent_lean'] == 'Lean Democrat' or row['other_lean'] == 'Lean Democrat'
        else ('Republican' if row['partyid'] == 'Republican' or row['independent_lean'] == 'Lean Republican' or row['other_lean'] == 'Lean Republican'
              else 'Other'), axis=1)

    # Political interest groups
    survey_data['interest_group'] = survey_data['interest'].apply(
        lambda x: 'High Political Interest' if x in ['Very interested', 'Extremely interested']
        else 'Low Political Interest'
    )

    # Political knowledge groups
    survey_data['knowledge_group'] = survey_data.apply(
        lambda row: 'High Political Knowledge' if row['knowl_1'] == 'The Supreme Court' and row['knowl_2'] == '2/3 (Two thirds)'
        else 'Low Political Knowledge', axis=1)

    return survey_data

# Merge survey data with explanation data


def merge_data(survey_subset, explanation_data):
    survey_subset['respondent_id'] = survey_subset.index
    text_columns = ['RandomText1', 'RandomText3', 'RandomText4', 'RandomText5']
    rating_columns = ['outcome_1_1', 'outcome_3_1',
                      'outcome_4_1', 'outcome_5_1']
    answer_columns = ['RandomAnswer1', 'RandomAnswer3',
                      'RandomAnswer4', 'RandomAnswer5']

    survey_text_df = survey_subset.melt(
        id_vars='respondent_id', value_vars=text_columns, var_name='response', value_name='text')
    survey_ratings_df = survey_subset.melt(
        id_vars='respondent_id', value_vars=rating_columns, var_name='rating', value_name='subjective_assertivity')
    survey_answers_df = survey_subset.melt(
        id_vars='respondent_id', value_vars=answer_columns, var_name='answer', value_name='gpt-answer')

    survey_combined_df = pd.concat([survey_text_df['respondent_id'], survey_text_df['text'],
                                    survey_answers_df['gpt-answer'], survey_ratings_df['subjective_assertivity']], axis=1)

    survey_combined_df['subjective_assertivity'] = pd.to_numeric(
        survey_combined_df['subjective_assertivity'], errors='coerce')

    agg_subjective_assertivity = survey_combined_df.groupby(
        'gpt-answer')['subjective_assertivity'].agg(['mean', 'count', 'std']).reset_index()
    agg_subjective_assertivity.rename(columns={'mean': 'mean_subjective_assertivity',
                                               'count': 'ratings_count', 'std': 'std_dev_subjective_assertivity'}, inplace=True)

    agg_subjective_assertivity['mean_subjective_assertivity'] = (agg_subjective_assertivity['mean_subjective_assertivity'] - agg_subjective_assertivity['mean_subjective_assertivity'].min()) / (
        agg_subjective_assertivity['mean_subjective_assertivity'].max() - agg_subjective_assertivity['mean_subjective_assertivity'].min())

    explanation_data_subset = explanation_data[[
        'label_id', 'text', 'label', 'calibrated-uncertainty', 'assertivity', 'Assertive', 'gpt-answer', 'ft_gpt_response',\
            'ft_gpt_norm']]
    explanation_data_subset['recalibrated_uncertainty'] = explanation_data_subset['calibrated-uncertainty'].apply(
        lambda x: abs(0.5 - x) * 2)

    merged_data = pd.merge(explanation_data_subset,
                           agg_subjective_assertivity, on='gpt-answer', how='left')

    return merged_data

# Generate density plots


def plot_density(data, group_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data['recalibrated_uncertainty'].dropna(),
                label='Recalibrated Uncertainty', shade=True, ax=ax)
    sns.kdeplot(data['mean_subjective_assertivity'].dropna(),
                label='Subjective Assertivity', shade=True, ax=ax)
    ax.set_title(
        f'Density Plot of Recalibrated Uncertainty and Subjective Assertivity ({group_name})')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to run the analysis


def main():
    # Preprocess the survey data
    survey_data_processed = preprocess_survey_data(survey_data)

    # Define the subgroups and their labels
    subgroups = {
        'education_group': ['Lower than College', 'College + More'],
        'party_group': ['Democrat', 'Republican', 'Other'],
        'interest_group': ['High Political Interest', 'Low Political Interest'],
        'knowledge_group': ['High Political Knowledge', 'Low Political Knowledge']
    }

    for group, values in subgroups.items():
        for value in values:
            # Subset survey data based on the group
            survey_subset = survey_data_processed[survey_data_processed[group] == value]

            # Merge with explanation data
            merged_data = merge_data(survey_subset, explanation_data)

            # Run the analysis (plotting density plots in this case)
            plot_density(merged_data, f'{group} - {value}')


if __name__ == "__main__":
    main()
