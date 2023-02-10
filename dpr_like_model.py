import torch
from torch import nn
from noise import remove_subjects, crop_words


class collator_qc():
    def __init__(self, tokenizer, corruption_rate = 0.):
        self.corruption_rate = corruption_rate
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        batch = corrupt(batch, corruption_rate=self.corruption_rate)
        
        src_q = self.tokenizer(['<question> ' + sample['question'] for sample in batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        src_c = self.tokenizer(['<context> ' + sample['context'] for sample in batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        
        return {
            'question': {**src_q},
            'context': {**src_c},
            "labels": torch.as_tensor([sample['target'] for sample in batch])
        }


def corrupt(batch, corruption_rate=0.2):
    new_data = []
    ready = False
    for i, data in enumerate(batch):
        # Pass if we swithed two context so it's already in new_data
        if ready:
            ready = False
            pass
        
        # Convert and add the unanswerable question
        elif not data['answerable']:
            new_data.append({'question': data['question'], 'context': data['text'], 'target': 1})            
        
        # Randomly apply one of the four corruption function
        else:
            if random.random() > 1 - corruption_rate:
                p = random.random()
                if p < 0.33 and i+1<len(batch):
                    new_data.append({'question': data['question'], 'context': data['text'], 'target': 1})
                    new_data.append({'question': batch[i+1]['question'], 'context': batch[i+1]['text'], 'target': 1})
                    ready = True
                elif p < 0.66:
                    new_data.append({'question': crop_words(data['question']), 'context': data['text'], 'target': 1})
                else:
                    new_data.append({'question': remove_subjects(data['question']), 'context': data['text'], 'target': 1})
                # else:
                #     context = get_important_noun_phrases(data['text'], nlp=nlp) + ' </s> ' + data['text']
                #     new_data.append({'input': context, 'target': 1})

            # If no corruption, just add the data
            else:
                new_data.append({'question': data['question'], 'context': data['text'], 'target': 0})
    return new_data


'''
The two function with the special head I wanted to try with Thomas,
they are for the first experiences only on french squadv2.
The results were not really good.
'''

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
        x = self.fc1(x)
        #x = self.relu(x) # Relu or tanh not both
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)     
        # Loss fct
        if self.loss_type=='NLLL':
            x = self.softmax(x)
            loss = self.loss_fct(x, batch['labels'])
        elif self.loss_type=='cross':
            loss = self.loss_fct(x, batch['labels'])
            
        return loss, x


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
        # Loss fct
        if self.loss_type=='NLLL':
            x = self.softmax(x)
            loss = self.loss_fct(x, batch['labels'])
        elif self.loss_type=='cross':
            loss = self.loss_fct(x, batch['labels'])
            
        return loss, x


# Pytorch class to launch it:
class classification_test(pl.LightningModule):

    def __init__(
        self,
        model_name = "camembert-base",
        load_pretraned_model = False,
        validation_callback = None, 
        log_dir = None,
        num_labels = 2,
        head = 'head_cls'
        ):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['<question>', '<context>'], special_tokens=True)
        

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
        else:
            self.model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).roberta
            self.model.resize_token_embeddings(len(self.tokenizer))
            if head == 'head_cls':
                self.model = head_cls(self.model)
            else:
                self.model = head_colbert_like(self.model)
        
        self.validation_callback = validation_callback
        self.log_dir = log_dir
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.model(batch)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        # scheduler1 = LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 100.)
        # # scheduler2 = ReduceLROnPlateau(optimizer, 'min', patience=3)
        # scheduler2 = StepLR(optimizer, step_size=1000, gamma=0.5)
        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 1000.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]
        # return optimizer

    def validation_step(self, batch, batch_idx):
        loss, pred = self.model(batch)
        self.log("val_loss", loss, sync_dist=True)
        return {"predictions": pred.tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_"+k, v, sync_dist=True)
