import os
import pandas as pd
from Config import Config

def arrange(base):
    """
    Arrange and preprocess the data from the given base directory.
    
    Parameters:
    base (str): Base directory containing the data files.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame with relevant columns.
    """
    main_table = pd.read_csv(os.path.join(base, 'Data/MainTable.csv')) # Load the main table dataset
    main_table = main_table[main_table.EventType == 'Run.Program'] # Filter rows where EventType is 'Run.Program' (focus on executed code events)
    code = pd.read_csv(os.path.join(base, 'Data/CodeStates/CodeStates.csv')) # Load the code dataset
    early = pd.read_csv(os.path.join(base, 'early.csv'))
    late = pd.read_csv(os.path.join(base, 'late.csv'))

    # Identify and remove empty snapshots (missing values in 'code')
    code = code.dropna()

    # Filter the main table to only include rows with valid CodeStateIDs present in the code dataset
    main_table = main_table[main_table['CodeStateID'].isin(set(code['CodeStateID']))]
    print(len(main_table))

    # Convert all timestamps to UTC for consistency
    main_table.loc[main_table['ServerTimezone'] == '0', 'ServerTimestamp'] = pd.to_datetime(main_table.loc[main_table['ServerTimezone'] == '0', 'ServerTimestamp']).dt.tz_localize('US/Eastern')
    main_table.loc[main_table['ServerTimezone'] == 'UTC', 'ServerTimestamp'] = pd.to_datetime(main_table.loc[main_table['ServerTimezone'] == 'UTC', 'ServerTimestamp']).dt.tz_localize('UTC')
    main_table['ServerTimestamp'] = pd.to_datetime(main_table['ServerTimestamp'], utc=True)

    main_table = main_table[['SubjectID', 'Order', 'ServerTimestamp', 'AssignmentID', 'ProblemID', 'CodeStateID', 'Score']]

    # Merge the main table with the code dataset on 'CodeStateID' and sort by order of execution
    df = main_table.merge(code, how='left', on='CodeStateID').sort_values(by='Order')

    # Group data by student, assignment, and problem
    df = df.groupby(['SubjectID', 'AssignmentID', 'ProblemID']).apply(lambda x: pd.Series({
        'start_time': x['ServerTimestamp'].iloc[0],  # First timestamp (start time)
        'end_time': x['ServerTimestamp'].iloc[-1],  # Last timestamp (end time)
        'order': x['Order'].tolist(),  # List of order values
        'code': x['Code'].tolist(),  # List of code snapshots
        'score': x['Score'].tolist(),  # List of scores
    })).reset_index()

    # Compute the maximum score achieved for each problem
    df['max_score'] = df['score'].apply(max)

    # Keep only the code snapshots until the highest score is reached (if max score is 100)
    df['code'] = df.apply(lambda x: x['code'][:x['score'].index(max(x['score'])) + 1] if max(x['score']) == 100 else x['code'], axis=1)
    df = df.merge(questions, on=['AssignmentID', 'ProblemID'], how='inner')

    # Sort by end time for chronological order (Order is not relvant between diffrent questions)
    df = df.sort_values(by='end_time')

    # Merge processed data with early and late submissions datasets
    late = late.merge(df, how='left', on=['SubjectID', 'AssignmentID', 'ProblemID'])
    early = early.merge(df, how='left', on=['SubjectID', 'AssignmentID', 'ProblemID']).sort_values(by='end_time')

    # Group early submissions by student and compile history of previous submissions
    early = early.groupby(['SubjectID']).apply(lambda x: pd.Series({
        'prev_tasks_id': x['ProblemID'].astype(str).tolist(),
        'prev_tasks': x['code'].tolist(),
        'prev_scores': x['score'].tolist(),
        'num_problems': len(x),  # Count number of past problems
        'prev_labels': x['Label'].tolist(),
    })).reset_index()
    late = late.merge(early, how='left', on='SubjectID')
    
    late = late[late['num_problems'].apply(lambda x: x == 30)] # Keep only students with exactly 30 prior problems

    # Rename columns for consistency with falcon
    late = late.rename(columns={'SubjectID': 'student_id'})
    late = late.rename(columns={'AssignmentID': 'assignment_id'})
    late = late.rename(columns={'Requirement': 'new_task'})
    late = late.rename(columns={'ProblemID': 'new_task_id'})
    
    # Select final relevant columns
    late = late[['student_id', 'assignment_id', 'new_task_id', 'max_score', 'new_task', 
                 'prev_tasks_id', 'prev_tasks', 'prev_scores', 'prev_labels', 'Label']]
    late['prev_labels'] = late['prev_labels'].apply(lambda x: [not i for i in x])
    late['Label'] = late['Label'].apply(lambda x: not x)
    return late

# Initialize configuration
config = Config()

# Load questions data
questions = pd.read_excel(os.path.join(config.codeworkout_folder, 'questions.xlsx'))

# Initialize an empty DataFrame for codeworkout data
codeworkout = pd.DataFrame()

# Process each course in the configuration
for i in config.codeworkout_courses:
    late = arrange(os.path.join(config.codeworkout_folder, i))
    print(f"End preprocess for {i}, {late.Label.value_counts()}")
    codeworkout = pd.concat([codeworkout, late])

# Save the final DataFrame to a pickle file
codeworkout.to_pickle(config.path_tosave_codeworkout)