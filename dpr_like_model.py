import torch
from torch import nn
from noise import corrupt


class head_cls(nn.Module):
    def __init__(self, model, loss_type='NLLL') -> None:
        super(head_cls, self).__init__()

        self.transformer = model
        self.loss_type = loss_type

        # dense layer 1
        self.fc1 = nn.Linear(1536, 768)

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(768, 2)

        if self.loss_type=='NLLL':
            # softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)
            self.loss_fct = nn.NLLLoss()
        elif self.loss_type=='cross':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            print("loss error")

    def forward(self, batch):
        #In batch: 'question', 'context', and 'labels' 
        q = self.transformer(**batch['question'])[0]
        c = self.transformer(**batch['context'])[0]
        # x = x[:, 0, :]
        x = torch.cat((q[:, 0, :], c[:, 0, :]), dim=1)

        # Classification head to fine tune
        # x = self.dropout(x)
        x = self.fc1(x)
        #x = self.relu(x) # Relu or tanh not both
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
#TODO modifier ça pour output juste 0 ou 1 et pas la loss        
        # Loss fct
        if self.loss_type=='NLLL':
            x = self.softmax(x)
            loss = self.loss_fct(x, batch['labels'])
        elif self.loss_type=='cross':
            loss = self.loss_fct(x, batch['labels'])
            
        return loss, x


class collator_qc():
    def __init__(self, tokenizer, corruption_rate = 0.):
        self.corruption_rate = corruption_rate
        self.tokenizer = tokenizer

        # if corruption_rate != 0:
        #     self.nlp = stanza.Pipeline(lang="fr")
        # else: self.nlp=None
    
    def __call__(self, batch):
        # nlp=self.nlp, 
        batch = corrupt(batch, corruption_rate=self.corruption_rate)
        
        src_q = self.tokenizer(['<question> ' + sample['question'] for sample in batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        src_c = self.tokenizer(['<context> ' + sample['context'] for sample in batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        
        return {
            'question': {**src_q},
            'context': {**src_c},
            "labels": torch.as_tensor([sample['target'] for sample in batch])
        }


class head_colbert_like(nn.Module):
    def __init__(self, model, loss_type='NLLL') -> None:
        super(head_colbert_like, self).__init__()

        self.transformer = model
        self.loss_type = loss_type

        # dense layer 1
        self.fc1 = nn.Linear(1536, 1536)

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(1536, 2)

        if self.loss_type=='NLLL':
            # softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)
            self.loss_fct = nn.NLLLoss()
        elif self.loss_type=='cross':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            print("loss error")

    def forward(self, batch):
        #In batch: 'question', 'context', and 'labels' 
        q = self.transformer(**batch['question'])[0]
        c = self.transformer(**batch['context'])[0]
        mask_q = batch['question']['attention_mask']
        mask_c = batch['context']['attention_mask']

        # matrice to create the importance weights
        x = torch.einsum('bik, bjk -> bij', q, c)
        # retirer les mask puis rendre à -inf pr le softmax
        x = torch.einsum('bij, bi -> bij', x, mask_q)
        x = torch.einsum('bij, bj -> bij', x, mask_c)
        x[x==0.] = -float('inf')

        # Take the waigth matrix without the 'nan' created by 0. lines
        p = torch.softmax(x, dim=2).nan_to_num()
        pp = torch.softmax(x, dim=1).nan_to_num()


        a = torch.einsum('bjk, bij -> bk', c, p)
        b = torch.einsum('bik, bij -> bk', q, pp)

        x = torch.cat((a, b), dim=1)

        # Classification head to fine tune
        x = self.fc1(x)
        #x = self.relu(x) # Relu or tanh not both
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
#TODO modifier ça pour output juste 0 ou 1 et pas la loss        
        # Loss fct
        if self.loss_type=='NLLL':
            x = self.softmax(x)
            loss = self.loss_fct(x, batch['labels'])
        elif self.loss_type=='cross':
            loss = self.loss_fct(x, batch['labels'])
            
        return loss, x


"""
J'ai fait autre chose ds model.py

class distillation_model(nn.Module):
    def __init__(self, model, loss_type='NLLL'):
        super(head_cls, self).__init__()

        self.transformer = model
        self.loss_type = loss_type

        # dense layer 1
        self.fc1 = nn.Linear(768, 768)

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(768, 2)

        # if self.loss_type=='NLLL':
        #     # softmax activation function
        #     self.softmax = nn.LogSoftmax(dim=1)
        #     self.loss_fct = nn.NLLLoss()
        # elif self.loss_type=='cross':
        #     self.loss_fct = nn.CrossEntropyLoss()
        # else:
        #     print("loss error")

    def forward(self, batch):
        if batch['task'] == 'prediction':
            #In batch: 'question', 'context', and 'labels' 
            q = self.transformer(**batch['question'])[0]
            c = self.transformer(**batch['context'])[0]
            # x = x[:, 0, :]
            x = torch.cat((q[:, 0, :], c[:, 0, :]), dim=1)

            # Classification head to fine tune
            # x = self.dropout(x)
            x = self.fc1(x)
            #x = self.relu(x) # Relu or tanh not both
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.fc2(x)
    #TODO modifier ça pour output juste 0 ou 1 et pas la loss        
            # Loss fct
            if self.loss_type=='NLLL':
                x = self.softmax(x)
                loss = self.loss_fct(x, batch['labels'])
            elif self.loss_type=='cross':
                loss = self.loss_fct(x, batch['labels'])

        elif batch['task'] == 'trad':
            en = self.transformer(**batch['en'])[0]
            fr = self.transformer(**batch['fr'])[0]
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            x = cos(en, fr) # x ou loss?
        
        return loss, x

"""