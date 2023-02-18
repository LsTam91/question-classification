import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR, StepLR
import pytorch_lightning as pl
from transformers import AutoTokenizer, CamembertForSequenceClassification
from transformers import AutoModelForSequenceClassification
import os
# import stanza

# Import from project
from noise import corrupt_and_convert


class collator():
    def __init__(self, tokenizer, corruption_rate = 0.):
        self.corruption_rate = corruption_rate
        self.tokenizer = tokenizer

        # if corruption_rate != 0:
        #     self.nlp = stanza.Pipeline(lang="fr")
        # else: self.nlp=None
    
    def __call__(self, batch):
        # nlp=self.nlp, 
        batch = corrupt_and_convert(batch, corruption_rate=self.corruption_rate)
        
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
            # self.model = self.load_state_dict(torch.load(load_pretraned_model))
            self.model = torch.load(load_pretraned_model)
        else:
            self.model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        self.validation_callback = validation_callback
        self.log_dir = log_dir
    
    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output[0]
        self.log("train_loss", loss, sync_dist=True) #torch.as_tensor(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        self.log("val_loss", loss, sync_dist=True)#torch.as_tensor(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
                self.log("val_"+k, v, sync_dist=True)
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


class train_and_distil(pl.LightningModule):

    def __init__(
        self,
        model_name = "xlm-roberta-base", # "bert-base-multilingual-uncased", #
        load_pretraned_model = False,
        validation_callback = None,
        log_dir = None,
        num_labels = 2,
        distance='cosine'
        ):

        super().__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #TODO
        self.tokenizer.add_tokens(['<en>', '<fr>'], special_tokens=True)
        

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
            # self.model = self.load_state_dict(torch.load(load_pretraned_model), map_location=torch.device('cpu'))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.validation_callback = validation_callback
        self.log_dir = log_dir
        
        # L2 or cosinesimilarity
        self.distance = distance
        if self.distance == 'cosine':
            self.dist = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Softmax fct:
        self.softmax = torch.nn.Softmax(dim=1)
        
    
    def training_step(self, batch, batch_idx):
        output = self.model(**batch['classi'])

        if "xlm-roberta-base" in self.model_name:
            sen = self.model.roberta(**batch['trad']['en'])[0][:, 0, :]
            sfr = self.model.roberta(**batch['trad']['fr'])[0][:, 0, :]
        else:
            sen = self.model.bert(**batch['trad']['en'])[0][:, 0, :]
            sfr = self.model.bert(**batch['trad']['fr'])[0][:, 0, :]

        # Take the distance between cls vector of both language
        if self.distance == 'cosine':
            # loss_trad = 1 - torch.mean(self.dist(sen, sfr))
            loss_trad = torch.mean(1-self.dist(sen, sfr))
        else:
            loss_trad = torch.mean(torch.norm(sen-sfr, dim=1, p=2)) * 0.1

        loss = output.loss + loss_trad

        self.log("train_classi", output.loss, sync_dist=True)
        self.log("train_trad", loss_trad, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 100.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch) # different here than two other algo
        self.log("val_loss", output.loss, sync_dist=True)
        return {"predictions": self.softmax(output.logits).tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_" + k, v, sync_dist=True)

    def test_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("test_loss", output.loss, sync_dist=True)
        return {"predictions": self.softmax(output.logits).tolist(), "references": batch['labels'].tolist()}
    
    def test_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("test_" + k, v, sync_dist=True)


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


class classification_multilanguage(pl.LightningModule):

    def __init__(
        self,
        model_name = "xlm-roberta-base",#"bert-base-multilingual-cased",
        load_pretraned_model = False,
        validation_callback = None, 
        log_dir = None,
        num_labels = 2
        ):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['<en>', '<fr>'], special_tokens=True)

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.validation_callback = validation_callback
        self.log_dir = log_dir
        
        self.softmax = torch.nn.Softmax(dim=1)
    
    def training_step(self, batch, batch_idx):
        output = self.model(**batch['classi'])

        self.log("train_loss", output.loss, sync_dist=True)
        return output.loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        # scheduler = LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 100.)
        # scheduler2 = ReduceLROnPlateau(optimizer, 'min', patience=3)
        # scheduler2 = StepLR(optimizer, step_size=1000, gamma=0.5)
        # scheduler = {
        #     # "scheduler": SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[3]),
        #     'scheduler': scheduler,
        #     "interval": "step",
        #     'name': 'lr_scheduler',
        #     "frequency": 1
        # }
        # return [optimizer], [scheduler]
        return optimizer

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        self.log("val_loss", output.loss, sync_dist=True)
        return {"predictions": self.softmax(output.logits).tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["predictions"] for b in batch], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in batch], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_"+k, v, sync_dist=True)