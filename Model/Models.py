import torch
from helper import *
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence

class TBPP_Generation(nn.Module):
    def __init__(self, text_model_name, code_len, input_lstm, hidden_size_lstm, dropout_rate=0.2, num_skills=10):
        """
        Initialize the TBPP_Generation model.
        
        Parameters:
        text_model_name (str): Name of the pre-trained text model.
        code_len (int): Length of the code sequences.
        input_lstm (int): Input size for the LSTM.
        hidden_size_lstm (int): Hidden size for the LSTM.
        dropout_rate (float): Dropout rate.
        num_skills (int): Number of skills.
        """
        super(TBPP_Generation, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.lstm_snapshots = nn.LSTM(
            code_len, 
            input_lstm, 
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_lstm_output_prev_labels = nn.BatchNorm1d(input_lstm + 1) # +1 for prev_struggling

        self.lstm_submissions = nn.LSTM(
           input_lstm + 1,  # +1 for prev_struggling
           hidden_size_lstm,
           batch_first=True,
           bidirectional=False,
        )
        self.skills_rep = nn.Linear(hidden_size_lstm, num_skills)

    def _make_mlp(self, input_dim, hidden_dim, dropout_rate):
        """
        Create a multi-layer perceptron (MLP).
        
        Parameters:
        input_dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        dropout_rate (float): Dropout rate.
        
        Returns:
        nn.Sequential: MLP model.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
    
    def _represent_task_submissions(self, code_embedding, prev_struggling, code_num):
        """
        Represent task submissions using LSTM.
        
        Parameters:
        code_embedding (torch.Tensor): Code embeddings.
        prev_struggling (torch.Tensor): Previous struggling labels.
        code_num (torch.Tensor): Number of code sequences.
        
        Returns:
        torch.Tensor: Task submissions representation.
        """
        if (code_num == 1).all():
            snapshots_output = code_embedding.squeeze(2)
        else:
            batch_size, q_num, c_padding, len = code_embedding.size()
            snapshots_lstm = pack_padded_sequence(
                code_embedding.view(batch_size * q_num, c_padding, len),
                lengths=code_num.view(batch_size * q_num).to('cpu'),
                batch_first=True,
                enforce_sorted=False
            )
            _, (snapshots_h, _) = self.lstm_snapshots(snapshots_lstm)
            snapshots_output = self.dropout(snapshots_h).view(batch_size, q_num, -1)
        batch_size, q_num, _ = snapshots_output.size()
        task_submissions = self.norm_lstm_output_prev_labels(
            torch.cat((snapshots_output, prev_struggling.float().unsqueeze(-1)), dim=-1).view(batch_size * q_num, -1)
        ).view(batch_size, q_num, -1)
        return task_submissions
    
    def _process_submissions_sequence(self, task_submissions):
        """
        Process the sequence of task submissions using LSTM.
        
        Parameters:
        task_submissions (torch.Tensor): Task submissions representation.
        
        Returns:
        torch.Tensor: Skills representation.
        """
        batch_size = task_submissions.size(0)
        _, (final_submission_h, _) = self.lstm_submissions(task_submissions)
        final_submission_h = self.dropout(final_submission_h)
        return self.skills_rep(final_submission_h.view(batch_size, -1))
    
    def _process_text_input(self, text_input_ids, text_attention_mask):
        """
        Process the text input using the pre-trained text model.
        
        Parameters:
        text_input_ids (torch.Tensor): Text input IDs.
        text_attention_mask (torch.Tensor): Text attention mask.
        
        Returns:
        torch.Tensor: Text output representation.
        """
        with torch.no_grad():
            text_output = self.text_model(text_input_ids, text_attention_mask).last_hidden_state[:, 0, :]
        return text_output
    
    def createTBPP(self, text_input_ids, text_attention_mask, code_embedding, code_num, prev_struggling): 
        """
        Create the TBPP representation.
        
        Parameters:
        text_input_ids (torch.Tensor): Text input IDs.
        text_attention_mask (torch.Tensor): Text attention mask.
        code_embedding (torch.Tensor): Code embeddings.
        code_num (torch.Tensor): Number of code sequences.
        prev_struggling (torch.Tensor): Previous struggling labels.
        
        Returns:
        tuple: Text output and TBPP representation.
        """
        task_submissions = self._represent_task_submissions(code_embedding, prev_struggling, code_num) # (batch_size, num_tasks, hidden_layer)
        text_output = self._process_text_input(text_input_ids, text_attention_mask)
        TBPP = self._process_submissions_sequence(task_submissions)
        return text_output, TBPP

class AblationStudy(TBPP_Generation):
    def __init__(self, sequence_len, text_model_name, code_len, num_prog_concepts=10, dropout_rate=0.2):
        """
        Initialize the AblationStudy model.
        
        Parameters:
        name (str): Name of the ablation study.
        text_model_name (str): Name of the pre-trained text model.
        code_len (int): Length of the code sequences.
        num_prog_concepts (int): Number of programming concepts.
        dropout_rate (float): Dropout rate.
        """
        hidden_size_lstm = 512 if sequence_len != 1 else code_len 
        super(AblationStudy, self).__init__(text_model_name, code_len, input_lstm=code_len, hidden_size_lstm=hidden_size_lstm, num_skills=num_prog_concepts)
        self.MLP_text = self._make_mlp(self.text_model.config.hidden_size, num_prog_concepts, dropout_rate)
        self.MLP_code = self._make_mlp(num_prog_concepts, num_prog_concepts, dropout_rate) 
        self.weight_code = nn.Parameter(torch.tensor(0.5))  # Initialize with equal weights
        self.weight_text = nn.Parameter(torch.tensor(0.5))  # Initialize with equal weights
        self.fully_connected_layer = nn.Linear(num_prog_concepts, 1)

    def _generate_prediction(self, text_mlp, submissions_output):
        """
        Generate the final prediction.
        
        Parameters:
        text_mlp (torch.Tensor): Text MLP output.
        submissions_output (torch.Tensor): Submissions output.
        
        Returns:
        torch.Tensor: Final prediction.
        """
        code_output = self.MLP_code(submissions_output)
        text_output = self.MLP_text(text_mlp)
        weights = torch.softmax(torch.stack([self.weight_code, self.weight_text]), dim=0)
        combined = weights[0] * code_output + weights[1] * text_output
        return self.fully_connected_layer(combined)
    
    def forward(self, text_input_ids, text_attention_mask, code_embedding, code_num, prev_struggling): 
        """
        Forward pass of the AblationStudy model.
        
        Parameters:
        text_input_ids (torch.Tensor): Text input IDs.
        text_attention_mask (torch.Tensor): Text attention mask.
        code_embedding (torch.Tensor): Code embeddings.
        code_num (torch.Tensor): Number of code sequences.
        prev_struggling (torch.Tensor): Previous struggling labels.
        
        Returns:
        torch.Tensor: Model output.
        """
        text_output, TBPP = self.createTBPP(text_input_ids, text_attention_mask, code_embedding, code_num, prev_struggling)
        output = self._generate_prediction(text_output, TBPP)
        return output
    

class PTM(TBPP_Generation):
    def __init__(self, text_model_name, code_len, num_students, num_skills=13, num_prog_concepts=10):
        """
        Initialize the PTM model.
        
        Parameters:
        text_model_name (str): Name of the pre-trained text model.
        code_len (int): Length of the code sequences.
        num_students (int): Number of students.
        num_skills (int): Number of skills.
        num_prog_concepts (int): Number of programming concepts.
        """
        super(PTM, self).__init__(text_model_name, code_len, input_lstm=512, hidden_size_lstm=512, num_skills=num_prog_concepts)
        # latents skills represention
        self.latent_matrix = nn.Embedding(num_students, 3)
        self.fc_latent = nn.Linear(num_prog_concepts, 3)  # Learn latent features from known skills
        self.fc_interaction = nn.Linear(3 + 3, 3)  # Interaction layer for latent features

        #cross attention
        input_dim = self.text_model.config.hidden_size
        self.skill_query = nn.Linear(num_skills, input_dim)
        self.skill_key = nn.Linear(num_skills, input_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, dropout=0.1)

        # Skill importance
        self.skill_importance = nn.Sequential(
            nn.Linear(num_skills, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(input_dim + num_skills, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def _struggling_prediction(self, text_embedding, skill_vector, required_prog_concepts):
        """
        Predict struggling using cross attention and skill importance.
        
        Parameters:
        text_embedding (torch.Tensor): Text embeddings.
        skill_vector (torch.Tensor): Skill vector.
        required_prog_concepts (torch.Tensor): Required programming concepts.
        
        Returns:
        torch.Tensor: Struggling prediction.
        """
        batch_size = skill_vector.size(0)
        required_skills = torch.cat((required_prog_concepts, torch.ones((batch_size, 3)).to(required_prog_concepts.device)), dim=1)
        skill_weights = self.skill_importance(required_skills)# Skill importance weighting
        # Perform cross attention
        attended_features, _ = self.cross_attention(self.skill_query(skill_vector).unsqueeze(0), self.skill_key(skill_vector).unsqueeze(0), text_embedding.unsqueeze(0))
        # Skill-weighted feature transformation
        return self.output_mlp(torch.cat([attended_features.squeeze(0), skill_vector * skill_weights], dim=-1))
    
    def latent_skills_represention(self, student_id, skill_vec):
        """
        Represent latent skills using student ID and skill vector.
        
        Parameters:
        student_id (torch.Tensor): Student IDs.
        skill_vec (torch.Tensor): Skill vector.
        
        Returns:
        torch.Tensor: Latent skills representation.
        """
        latent_features_id = self.latent_matrix(student_id.long())  # Get latent features from embedding
        latent_features_skills = self.fc_latent(skill_vec)  # Latent features from skills
        latent_features = torch.cat((latent_features_id, latent_features_skills), dim=1)
        return self.fc_interaction(latent_features)

    def forward(self, text_input_ids, text_attention_mask, code_embedding, code_num, prev_struggling, required_prog_concepts, student_id): 
        """
        Forward pass of the PTM model.
        
        Parameters:
        text_input_ids (torch.Tensor): Target task text input IDs.
        text_attention_mask (torch.Tensor): Target task text attention mask.
        code_embedding (torch.Tensor): Code embeddings after LLM.
        code_num (torch.Tensor): Number of code sequences.
        prev_struggling (torch.Tensor): Previous struggling labels.
        required_prog_concepts (torch.Tensor): Required programming concepts.
        student_id (torch.Tensor): Student IDs.
        
        Returns:
        tuple: Model output and TBPP representation.
        """
        text_output, skill_vec = self.createTBPP(text_input_ids, text_attention_mask, code_embedding, code_num, prev_struggling) 
        latent_skills = self.latent_skills_represention(student_id, skill_vec)
        TBPP = torch.cat((skill_vec, latent_skills), dim=1) # concate with known skills
        output = self._struggling_prediction(text_output, TBPP, required_prog_concepts)
        return output, torch.sigmoid(TBPP)