class Config:
    def __init__(self):
        self.codeworkout_questions_path = 'Datasets/codeworkout/questions.xlsx'
        self.falconcode_questions_path = 'Datasets/falcon/cleaned_questions.csv'
        self.path_saved_codeworkout = 'Datasets/codeworkout.pkl'
        self.path_saved_falcon = 'Datasets/falcon.pkl'
        self.text_model_name = 'google-bert/bert-base-cased'
        self.lr = 0.0001
        self.batch_size = 32
        self.max_len_code = 768
        self.run_model = 'PTM' # for Ablation change the name
        self.padding_size_code = 100 # 1 for no history (NoHis)
        self.number_coding_tasks = 30
        self.dataset = 1 # 0 for codeworkout or 1 for falcon
        self.save_model_path = 'PTM.pth'
        self.epoch = 15