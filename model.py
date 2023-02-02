import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR, StepLR
import pytorch_lightning as pl
from transformers import AutoTokenizer, CamembertForSequenceClassification
from transformers import RobertaForSequenceClassification, AutoModelForSequenceClassification
import os
# import stanza

# Import from project
from noise import corrupt_and_convert
from dpr_like_model import head_cls, head_colbert_like


class collator():
    def __init__(self, tokenizer, corruption_rate = 0., language = 'fr'):
        self.corruption_rate = corruption_rate
        self.tokenizer = tokenizer
        self.language = language

        # if corruption_rate != 0:
        #     self.nlp = stanza.Pipeline(lang="fr")
        # else: self.nlp=None
    
    def __call__(self, batch):
        # nlp=self.nlp, 
        batch = corrupt_and_convert(batch, language=self.language, corruption_rate=self.corruption_rate)
        
        src_txt = [sample['input'] for sample in batch]
        src_tok = self.tokenizer(src_txt, return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        
        return {
            **src_tok,
            "labels": torch.as_tensor([sample['target'] for sample in batch])
        }


class classification_model(pl.LightningModule):

    def __init__(
        self,
        model_name = "camembert-base",
        load_pretraned_model = False,
        validation_callback = None, 
        log_dir = None,
        num_labels = 2
        ):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
        else:
            self.model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        self.validation_callback = validation_callback
        self.log_dir = log_dir
    
    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output[0]
        self.log("train_loss", loss) #torch.as_tensor(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        scheduler = LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 100.)
        # scheduler2 = ReduceLROnPlateau(optimizer, 'min', patience=3)
        # scheduler2 = StepLR(optimizer, step_size=1000, gamma=0.5)
        scheduler = {
            # "scheduler": SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[3]),
            'scheduler': scheduler,
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]
        # return optimizer

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output[0]
        self.log("val_loss", loss)#torch.as_tensor(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"predictions": output.logits.tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])
        # predictions = np.concatenate(batch['predictions'], axis=0)
        # references = np.concatenate(batch['references'], axis=0)

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_"+k, v)
        # if self.log_dir != None:
        #     df = pd.DataFrame({"predictions": predictions, "references": references})
        #     df.to_csv(os.path.join(self.log_dir, "validation_prediction-"+str(self.current_epoch)+".csv"))

    # def predict_step(self, batch, batch_idx):
    #     with torch.no_grad():
    #         generated_batch = self.model.generate(
    #             input_ids = batch['input_ids'],
    #             attention_mask = batch['attention_mask']
    #         )

    #     generated_text = self.tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
    #     ground_truth_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
    #     return [{"qid":batch["qid"][i], "qtype":batch["qtype"][i], "default_selection": batch["default_selection"][i],
    #              "generated_text": generated_text[i], "ground_truth_text":ground_truth_text[i]} for i in range(len(batch["input_ids"]))]

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
        self.log("train_loss", loss)
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
        self.log("val_loss", loss)
        return {"predictions": pred.tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_"+k, v)


class train_and_distil(pl.LightningModule):

    def __init__(
        self,
        model_name = "xlm-roberta-base", 
        load_pretraned_model = False,
        validation_callback = None,
        log_dir = None,
        num_labels = 2,
        ):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #TODO
        self.tokenizer.add_tokens(['<en>', '<fr>'], special_tokens=True)
        

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
        else:
            # self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            # self.model = self.classi.roberta
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.validation_callback = validation_callback
        self.log_dir = log_dir
        
        # L2 or cosinesimilarity
        self.dist = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    
    def training_step(self, batch, batch_idx):
        # trouver un moyen de gérer les données!!
        output = self.model(**batch['classi'])

        sen = self.model.bert(**batch['trad']['en'])[0][:, 0, :] #TODO change bert and roberta auto
        sfr = self.model.bert(**batch['trad']['fr'])[0][:, 0, :]
        # Take the distance between cls vector of both language
        loss_trad = 1 - torch.mean(self.dist(sen, sfr))
        # loss_trad = torch.mean(torch.norm(sen-sfr, dim=1, p=2))

        loss = output.loss + loss_trad

        self.log("train_classi", output.loss)
        self.log("train_trad", loss_trad)
        self.log("train_loss", loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 1000.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch) # different here than two other algo
        self.log("val_loss", output.loss)
        return {"predictions": output.logits.tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_" + k, v)


class trad_collator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        en_tok = self.tokenizer('<en> ' + ' '.join([sample['en'] for sample in batch]), return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        fr_tok = self.tokenizer('<fr> ' + ' '.join([sample['fr'] for sample in batch]), return_tensors="pt",  padding='longest', truncation=True, max_length=512)

        return {
                'en': {**en_tok},
                'fr': {**fr_tok}
            }


# class collator_two_task():
#     def __init__(self, tokenizer, corruption_rate = 0.):
#         self.corruption_rate = corruption_rate
#         self.tokenizer = tokenizer
    
#     def __call__(self, batch):
        
#         t_batch = []
#         c_batch = []

#         for datum in batch:
#             if datum['task'] == 'classi':
#                 c_batch.append(datum['datum'])
#             else:
#                 t_batch.append(datum['datum'])
#         # print(t_batch, c_batch)
        
#         if len(c_batch) > 0:
#             c_batch = corrupt_and_convert(c_batch, corruption_rate=self.corruption_rate)
#             c_tok = self.tokenizer([sample['input'] for sample in c_batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        
#         if len(t_batch) > 0:
#             en_tok = self.tokenizer([sample['en'] for sample in t_batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
#             fr_tok = self.tokenizer([sample['fr'] for sample in t_batch], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
#         else: en_tok, fr_tok = {False, False}

#         return {
#             'classi': {
#                 **c_tok,
#                 "labels": torch.as_tensor([sample['target'] for sample in c_batch])
#             },
#             'trad': {
#                 'en': {**en_tok},
#                 'fr': {**fr_tok}
#             }
#         }