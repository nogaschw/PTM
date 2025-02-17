# The following model taken from https://github.com/YangAzure/Code-DKT/tree/main
import torch
import torch.nn as nn

MAX_CODE_LEN = 100

class c2vRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, node_count, path_count, device):
        super(c2vRNNModel, self).__init__()

        self.embed_nodes = nn.Embedding(node_count+2, 100) # adding unk and end
        self.embed_paths = nn.Embedding(path_count+2, 100) # adding unk and end
        self.embed_dropout = nn.Dropout(0.2)
        self.path_transformation_layer = nn.Linear(input_dim+300,input_dim+300)
        self.attention_layer = nn.Linear(input_dim+300,1)
        self.prediction_layer = nn.Linear(input_dim+300,1)
        self.attention_softmax = nn.Softmax(dim=1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(2*input_dim+300,
                          hidden_dim,
                          layer_dim,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc_target = nn.Linear(hidden_dim + output_dim, 1)  # Predicting target question
        self.dropout = nn.Dropout(p=0.1)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, target_question, evaluating=False):  # shape of input: [batch_size, length, questions * 2+c2vnodes]       
        rnn_first_part = x[:, :, :self.input_dim] # (b,l,2q)
        rnn_attention_part = torch.stack([rnn_first_part]*MAX_CODE_LEN,dim=-2) # (b,l,c,2q)

        c2v_input = x[:, :, self.input_dim:].reshape(x.size(0), x.size(1), MAX_CODE_LEN, 3).long() # (b,l,c,3)

        starting_node_index = c2v_input[:,:,:,0] # (b,l,c,1)
        ending_node_index = c2v_input[:,:,:,2] # (b,l,c,1)
        path_index = c2v_input[:,:,:,1] # (b,l,c,1)

        starting_node_embed = self.embed_nodes(starting_node_index) # (b,l,c,1) -> (b,l,c,ne)
        ending_node_embed = self.embed_nodes(ending_node_index) # (b,l,c,1) -> (b,l,c,ne)
        path_embed = self.embed_paths(path_index) # (b,l,c,1) -> (b,l,c,pe)
        
        full_embed = torch.cat((starting_node_embed, ending_node_embed, path_embed, rnn_attention_part), dim=3) # (b,l,c,2ne+pe+q)
        if not evaluating:
            full_embed = self.embed_dropout(full_embed) # (b,l,c,2ne+pe+2q)
        
        full_embed_transformed = torch.tanh(self.path_transformation_layer(full_embed)) # (b,l,c,2ne+pe+2q)
        context_weights = self.attention_layer(full_embed_transformed) # (b,l,c,1)
        attention_weights = self.attention_softmax(context_weights) # (b,l,c,1)
        code_vectors = torch.sum(torch.mul(full_embed,attention_weights),dim=2) # (b,l,2ne+pe+2q)
        rnn_input = torch.cat((rnn_first_part,code_vectors), dim=2)
        out, hn = self.rnn(rnn_input)  # shape of out: [batch_size, length, hidden_size]
        all_question_preds = self.sig(self.fc(out))  # shape of res: [batch_size, length, question]

        # Modifiation : Use the last hidden state and concatenate with target question
        last_hidden_state = out[:, -1, :]  # [batch_size, hidden_dim]
        combined = torch.cat((last_hidden_state, target_question), dim=1)  # [batch_size, hidden_dim + output_dim]
        target_question_pred = self.sig(self.fc_target(combined))  # [batch_size, 1]
        
        return all_question_preds, target_question_pred

class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device

    def forward(self, all_pred, batch, target_label):
        loss = 0
        pred, target_q = all_pred
        pred = pred.to('cpu')
        target_q = target_q.to('cpu')
        batch = batch.to('cpu')
        target_label = target_label.to('cpu').unsqueeze(1)
        for student in range(pred.shape[0]):
            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:self.num_of_questions*2]  # shape: [length, questions]
            temp = pred[student][:self.max_step-1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step-1)]], dtype=torch.long)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:self.num_of_questions*2]).sum(1) + 1) // 2)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    p = p[i:]
                    a = a[i:]
                    break
            p = torch.cat([p, target_q[student]])
            a = torch.cat([a, target_label[student]])
            loss += self.crossEntropy(p, a)
        return loss