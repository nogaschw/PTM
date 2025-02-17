import torch
import numpy as np
import itertools
import pandas as pd
from torch.utils.data import Dataset

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary

    """
    ixtoword = {}
    # period at the end of the sentence. make first dimension be end token
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword

def convert_to_idx(sample, node_word_index, path_word_index):
    """
    Converting to the index 
    Input:
    sample: list. One single training sample, which is a code, represented as a list of neighborhoods.
    node_word_index: dict. The node to word index dictionary.
    path_word_index: dict. The path to word index dictionary.

    """
    sample_index = []
    for line in sample:
        components = line.split(",")
        if components[0] in node_word_index:
            starting_node = node_word_index[components[0]]
        else:
            starting_node = node_word_index['UNK']
        if components[1] in path_word_index:
            path = path_word_index[components[1]]
        else:
            path = path_word_index['UNK']
        if components[2] in node_word_index:
            ending_node = node_word_index[components[2]]
        else:
            ending_node = node_word_index['UNK']
        
        sample_index.append([starting_node,path,ending_node])
    return sample_index

MAX_CODE_LEN = 100

class data_reader():
    def __init__(self, df,code_df, problems_d, maxstep, numofques):
        self.df = df
        self.code_lookup = dict(zip(code_df['Code'], code_df['RawASTPath']))   
        self.student_codes = dict(zip(code_df['Code'], code_df['SubjectID']))
        self.maxstep = maxstep
        self.numofques = numofques
        self.problems_d = problems_d
    
    def _create_nodes(self, train_students):        
        node_hist = {}
        path_hist = {}
        relevant_codes = [code for code, student in self.student_codes.items() 
                if student in train_students]
        for code in relevant_codes:
            raw_ast = self.code_lookup.get(code)
            if not isinstance(raw_ast, str):
                continue
            paths = raw_ast.split("@$")
            paths = [p for p in paths if p.strip() and len(p.strip()) > 3]
            
            starting_nodes = [p.split(",")[0] for p in paths]
            path = [p.split(",")[1] for p in paths]
            ending_nodes = [p.split(",")[2] for p in paths]
            for n in starting_nodes + ending_nodes:
                node_hist[n] = node_hist.get(n, 0) + 1
            for p in path:
                path_hist[p] = path_hist.get(p, 0) + 1
        return node_hist, path_hist
    
    def _create_student_dict(self, df):
        d = {}
        for j, row in df.iterrows():
            s = row['student_id']
            d[s] = {}
            attempts = [len(i) for i in row['prev_tasks']]
            d[s]["length"] = sum(attempts)
            d[s]["Problems"] = [self.problems_d[q] for i, q in enumerate(row['prev_tasks_id']) for _ in range(attempts[i])]
            d[s]["Result"] = [int(r) if idx == attempts[i] - 1 else 0 for i, r in enumerate(row['prev_labels']) for idx in range(attempts[i])]
            d[s]["CodeStates"] = [item for sublist in row['prev_tasks'] for item in sublist]
        return d

    def _create_rows(self, dict, node_word_index, path_word_index):
        processed_data = {}
        feature_size = 2 * self.numofques + MAX_CODE_LEN * 3
        code_features_cache = {}
        code_features_cache[None] = np.zeros(MAX_CODE_LEN * 3)

        for i, (student, student_data) in enumerate(dict.items()):
            lent = student_data["length"]
            ques = student_data["Problems"]
            ans = student_data["Result"]
            css = student_data["CodeStates"]
            temp = np.zeros(shape=[self.maxstep, feature_size]) # Skill DKT #1, original            

            if lent >= self.maxstep:
                steps = self.maxstep
                extra = 0
                ques = ques[-steps:]
                ans = ans[-steps:]
                css = css[-steps:]
            else:
                steps = lent
                extra = self.maxstep-steps
            for j in range(steps):
                if ans[j] == 1:
                    temp[j+extra][ques[j]] = 1
                else:
                    temp[j+extra][ques[j] + self.numofques] = 1
                current_code = css[j]
                if current_code not in code_features_cache:
                    code = self.code_lookup.get(current_code)
                    if isinstance(code, str):
                        code_paths = code.split("@$")
                        raw_features = convert_to_idx(code_paths, node_word_index, path_word_index)
                        if len(raw_features) < MAX_CODE_LEN:
                            raw_features += [[0,0,0]]*(MAX_CODE_LEN - len(raw_features))
                        else:
                            raw_features = raw_features[:MAX_CODE_LEN]
                        code_features_cache[current_code] = np.array(raw_features).reshape(-1)
                    else:
                        code_features_cache[current_code] = np.zeros(MAX_CODE_LEN * 3)
                
                temp[j+extra][2*self.numofques:] = code_features_cache[current_code]
            processed_data[student] = temp.tolist()
        return processed_data

    def get_data(self, train_students, test_students):
        node_hist, path_hist = self._create_nodes(train_students)
        node_count = len(node_hist)
        path_count = len(path_hist)

        # small frequency then abandon, for node and path
        valid_node = [node for node, count in node_hist.items()]
        valid_path = [path for path, count in path_hist.items()]

        # create ixtoword and wordtoix lists
        node_word_index, node_index_word = create_word_index_table(valid_node)
        path_word_index, path_index_word = create_word_index_table(valid_path)
    
        # Create train and test DataFrames
        test_dict = self._create_student_dict(self.df[self.df['student_id'].isin(test_students)])
        train_dict = self._create_student_dict(self.df[self.df['student_id'].isin(train_students)])
        data = {}
        data['test'] = self._create_rows(test_dict, node_word_index, path_word_index)
        print(f"finish test {len(test_dict)}")
        data['train'] = self._create_rows(train_dict, node_word_index, path_word_index)
        print(f"finish train {len(train_dict)}")
        data['node_count'] = node_count
        data['path_count'] = path_count
        self.data = data

class StudentDataset(Dataset):
    def __init__(self, df, handler, test='train'):
        self.df = df
        self.data_d = handler.data[test]
        self.problems_dict = handler.problems_d
        self.node_count = handler.data['node_count']
        self.path_count = handler.data['path_count']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        student_id = row['student_id']
        futures_problem_id = self.problems_dict[row['new_task_id']]
        hot_encoder = np.zeros(shape=[len(self.problems_dict)])
        hot_encoder[futures_problem_id] = 1

        return {
            'row': torch.tensor(self.data_d[student_id], dtype=torch.float32),
            'futures_problem_id': torch.tensor(hot_encoder, dtype=torch.float32),
            'label': torch.tensor(row['Label'])
        }