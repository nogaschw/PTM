class Config:
    def __init__(self):
        self.codeworkout_folder = 'OriginalData/codeworkout'
        self.codeworkout_courses = ['Spring', 'Fall/Train', 'Fall/Test']
        self.falconcode_folder = '/home/nogaschw/PTM/Datasets/falcon'
        self.path_tosave_codeworkout = 'Datasets/codeworkout.pkl'
        self.path_tosave_falcon = 'Datasets/falcon.pkl'
        self.code_model_name = 'microsoft/codebert-base'
        self.save_code_embedding = 'Datasets/falcon_to_model_output.pkl'
        self.max_model_len = 512
        self.dataset = 1 # for embedding