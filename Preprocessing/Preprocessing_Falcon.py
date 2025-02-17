import os
import re
import numpy as np
import pandas as pd
from Config import Config

def compute_percentiles_and_labels(df):
    """
    Compute the 75th percentile and labels for student attempts.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing student attempt data.
    
    Returns:
    pd.DataFrame: DataFrame with additional columns for percentiles and labels.
    """
    percentiles = df.groupby('problem_id')['num_snapshots'].apply(lambda x: np.percentile(x, 75)).reset_index()
    percentiles.columns = ['problem_id', '75th_percentile']
    df = df.merge(percentiles, on='problem_id')
    df = df[df['type'] != 'project']
    df['below_75th_percentile'] = df.apply(lambda row: row['num_snapshots'] <= row['75th_percentile'], axis=1)
    df['correct_eventually'] = df.apply(lambda row: row['max_score_x'] >= 100, axis=1)
    df['Label'] = df.apply(lambda row: row['correct_eventually'] & row['below_75th_percentile'], axis=1)
    return df

def create_df(config):
    """
    Create and preprocess the DataFrame from CSV files.
    
    Parameters:
    config (Config): Configuration object containing file paths.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame ready for analysis.
    """
    csv_file_path = os.path.join(config.falconcode_folder, 'cleaned_code.csv')
    df = pd.read_csv(csv_file_path, sep=',')
    df = df[df.columns[1:]]

    cleaned_file_path = os.path.join(config.falconcode_folder, 'cleaned_questions.csv')
    questions = pd.read_csv(cleaned_file_path, sep=',')
    questions = questions[questions.columns[1:]]

    # clean question
    questions['prompt'] = questions['prompt'].apply(lambda x: x.split('PROBLEM STATEMENT:')[-1] if x.__contains__("PROBLEM STATEMENT:") else x)
    questions['prompt'] = questions.prompt.apply(lambda text: re.sub(r'\bPROBLEM STATEMENT: \b', '', text).strip())
    
    df['problem_id'] = df['problem_id'].apply(lambda x: x.lower())
    questions['id'] = questions['id'].apply(lambda x: x.lower())

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values('timestamp', ascending=True)
    df = df.groupby(['student_id', 'course_id', 'problem_id']).apply(lambda x: pd.Series({
        'ServerTimestamp': x['timestamp'].tolist(),
        'score': x['score'].tolist(),
        'source_code': x['clean_code'].tolist()
    })).reset_index()

    def trim_after_max(score, snapshots, serverTimestamp):
        max_index = score.index(max(score))
        return score[:max_index+1], snapshots[:max_index+1], serverTimestamp[:max_index+1]

    # Apply the function to each row
    df['score'], df['source_code'], df['ServerTimestamp'] = zip(*df.apply(lambda row: trim_after_max(row['score'], row['source_code'], row['ServerTimestamp']), axis=1))
    df['start_time'] = [i[0] for i in df['ServerTimestamp']]
    df['end_time'] = [i[-1] for i in df['ServerTimestamp']]
    df['max_score'] = [max(i) for i in df['score']]
    df['num_snapshots'] = df['source_code'].apply(lambda x: len(x))

    a = df.groupby(['student_id', 'course_id']).apply(lambda x: pd.Series({
        'problem_id': x['problem_id'].tolist(),
        'score': x['score'].tolist(),
        'question_num': len(x)
    })).reset_index()

    valid_students = set(a[~a['score'].apply(lambda x: sum([item == -1 for sublist in x[:31] for item in sublist]) > 30)]['student_id']) # made more than 30 of question globaly
    df = df[df['student_id'].isin(valid_students)][df[df['student_id'].isin(valid_students)]['score'].apply(lambda x: not x.__contains__(-1))]
    print(len(df))

    df = df.merge(questions, left_on=['course_id','problem_id'], right_on=['course_id','id'])[['student_id', 'course_id', 'problem_id','score',
       'source_code', 'max_score_x', 'end_time', 'num_snapshots', 'type', 'prompt']]
    df = compute_percentiles_and_labels(df)

    df_sorted = df.sort_values(by=['student_id', 'end_time'])
    # create same row to each student
    df_sorted['row_number'] = df_sorted.groupby(['student_id', 'course_id']).cumcount() + 1
    print(df_sorted.columns)
    df_early = df_sorted[df_sorted['row_number'].apply(lambda x: x <= 30)] # if we need the early questions also
    df_early = df_early.groupby(['student_id', 'course_id']).apply(lambda x: pd.Series({
        'prev_tasks_id': x['problem_id'].astype(str).tolist(),
        'prev_tasks': x['source_code'].tolist(),
        'prev_scores': x['score'].tolist(),
        'num_tasks': len(x),  # Count number of past problems
        'prev_labels': x['Label'].tolist(),
    })).reset_index()
    
    df_late = df_sorted[df_sorted['row_number'].apply(lambda x: (x > 30) & (x <= 50))]
    df_late = df_late.merge(df_early, how='left', on=['student_id', 'course_id'])
    final_df = df_late[df_late['num_tasks'].apply(lambda x: x == 30)] # Keep only students with exactly 30 prior problems

    final_df = final_df.rename(columns={'problem_id': 'new_task_id'})
    final_df = final_df.rename(columns={'max_score_x': 'max_score'})
    final_df = final_df.rename(columns={'prompt': 'new_task'})
    # change label to 1 is struggling and 0 not
    final_df['prev_labels'] = final_df['prev_labels'].apply(lambda x: [not i for i in x])
    final_df['Label'] = final_df['Label'].apply(lambda x: not x)

    return final_df[['student_id', 'course_id', 'new_task_id', 'max_score', 'new_task', 
                 'prev_tasks_id', 'prev_tasks', 'prev_scores', 'prev_labels', 'Label']]

# Initialize configuration
config = Config()

# Create and preprocess DataFrame
dfa_30 = create_df(config)

# Filter students with history between 30 and 50 attempts
dfa_30 = dfa_30[dfa_30['prev_scores'].apply(lambda x: (len(x) >= 30 and len(x) < 50))]
dfa_30['prev_tasks'] = dfa_30['prev_tasks'].apply(lambda x: x[:30])
dfa_30['prev_scores'] = dfa_30['prev_scores'].apply(lambda x: x[:30])
dfa_30['prev_labels'] = dfa_30['prev_labels'].apply(lambda x: x[:30])
dfa_30['prev_tasks_id'] = dfa_30['prev_tasks_id'].apply(lambda x: x[:30])

# Print label value counts and save DataFrame to pickle file
print(dfa_30.Label.value_counts())
dfa_30.to_pickle(config.path_tosave_falcon)