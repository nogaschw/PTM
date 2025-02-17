# Preprocessing Scripts for Codeworkout and Falcon Datasets

This directory contains scripts for preprocessing the Codeworkout and Falcon datasets. The preprocessing steps include loading data, filtering, merging, and generating embeddings for code snippets.

## Files

### Preprocessing_Codeworkout.py

This script preprocesses the Codeworkout dataset.
The datasets we used can found in here:
https://sites.google.com/ncsu.edu/csedm-dc-2021/dataset?authuser=0


### Preprocessing_Falcon.py

This script preprocesses the Falcon dataset.
Which the details can be found:
de Freitas, Adrian, Joel Coffman, Michelle de Freitas, Justin Wilson, and Troy Weingart. "Falconcode: A multiyear dataset of python code samples from an introductory computer science course." de Freitas, Adrian, Joel Coffman, Michelle de Freitas, Justin Wilson, and Troy Weingart. "Falconcode: A multiyear dataset of python code samples from an introductory computer science course." 

We used the same structure as in Codeworkout dataset. And we extract from the dataset the codes of the students and the coding tasks.

### CreateDataEmbedding.py

This script generates embeddings for code snippets using a pre-trained language model.
The script processes code snippets in batches and saves the embeddings to a pickle file.

### Config.py

This script contains the configuration settings for the preprocessing scripts.

#### Configuration Parameters:
- `codeworkout_folder`: Path to the Codeworkout data folder.
- `codeworkout_courses`: List of Codeworkout courses.
- `falconcode_folder`: Path to the Falcon data folder.
- `path_tosave_codeworkout`: Path to save the preprocessed Codeworkout data.
- `path_tosave_falcon`: Path to save the preprocessed Falcon data.
- `code_model_name`: Name of the pre-trained language model.
- `save_code_embedding`: Path to save the code embeddings.
- `max_model_len`: Maximum length for the model input.

## Usage

1. Ensure the data files are placed in the appropriate directories as specified in the configuration.
2. Run the preprocessing scripts to generate the preprocessed datasets and embeddings.
3. The preprocessed datasets will contain the following columns:
   - `student_id`
   - `course_id` / `assignment_id`
   - `new_task_id`
   - `max_score`
   - `new_task`
   - `prev_tasks_id`
   - `prev_tasks`
   - `prev_scores`
   - `prev_labels`
   - `Label`

## Example

To preprocess the Codeworkout dataset, run:
```bash
python Preprocessing_Codeworkout.py
```

To preprocess the Falcon dataset, run:
```bash
python Preprocessing_Falcon.py
```

To generate embeddings for code snippets, run:
```bash
python CreateDataEmbedding.py
```