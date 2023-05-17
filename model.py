import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR #, ReduceLROnPlateau, SequentialLR, StepLR
import pytorch_lightning as pl
import transformers
from transformers import AutoTokenizer, CamembertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from typing import Optional, List, Dict, Any, Union, Tuple

# Import from project
from noise import corrupt_and_convert


class collator():

    """Data collator for text, question and target"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, corruption_rate: float = 0.0) -> None:
        self.corruption_rate = corruption_rate
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = corrupt_and_convert(batch, corruption_rate=self.corruption_rate)
        
        src_txt = [sample['input'] for sample in batch]
        src_tok = self.tokenizer(src_txt, return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        
        return {
            **src_tok,
            "labels": torch.as_tensor([sample['target'] for sample in batch])
        }


class trad_collator():

    """Data collator for the traduction task"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
    
    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        en_tok = self.tokenizer('<en> ' + ' '.join([sample['en'] for sample in batch]), return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        fr_tok = self.tokenizer('<fr> ' + ' '.join([sample['fr'] for sample in batch]), return_tensors="pt",  padding='longest', truncation=True, max_length=512)

        return {
                'en': {**en_tok},
                'fr': {**fr_tok}
            }


class MQ_classification(pl.LightningModule):
    """
    This class defines a PyTorch Lightning module for a multi-task
    classification problem or a classic classification problem using
    pre-trained transformer-based models.

    Args:
        model_name (str, optional): The name or path of the pre-trained transformer-based model. Default is "xlm-roberta-base".
        task (str, optional): define the task between classic transformer or my multi objectives method.
        load_pretrained_model (bool, optional): Whether to load a pre-trained model from a file. Default is False.
        validation_callback (function, optional): A function that takes the predictions and references as input and returns a dictionary of metrics. Default is None.
        log_dir (str, optional): The path to the directory where logs will be saved. Default is None.
        num_labels (int, optional): The number of labels for classification. Default is 2.
        distance (str, optional): distance type for the <cls> similarity, L2 or cosine

    Methods:
        training_step(batch: tuple, batch_idx: int) -> torch.Tensor: Processes a batch of training data and returns the loss.
        configure_optimizers() -> torch.optim.Optimizer: Configures optimizer and scheduler for training.
        validation_step(batch: tuple, batch_idx: int) -> dict: Processes a batch of validation data and returns a dictionary of predictions and references.
        validation_epoch_end(batch: list) -> None: Processes the validation results for an epoch.
    """
    def __init__(
        self,
        model_name = "xlm-roberta-base",
        task : str = 'multi_obj', # or 'classic'
        load_pretraned_model = False,
        validation_callback = None,
        log_dir = None,
        num_labels = 2,
        distance : str = 'cosine'
        ):
        super().__init__()

        self.model_name = model_name
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['<en>', '<fr>'], special_tokens=True)
        

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
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

        # Extract the name of the backbone from the model
        self.backbone_name = self.model.base_model_prefix
        
    
    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor: # Or batch: Dict[str, Dict[str, torch.Tensor]] for multi_obj
        if self.task == 'multi_obj':
            output = self.model(**batch['classi'])

            sen = getattr(self.model, self.backbone_name)(**batch['trad']['en'])[0][:, 0, :]
            sfr = getattr(self.model, self.backbone_name)(**batch['trad']['fr'])[0][:, 0, :]


            # Take the distance between cls vector of both language
            if self.distance == 'cosine':
                loss_trad = torch.mean(1-self.dist(sen, sfr))
            else: # L2
                loss_trad = torch.mean(torch.norm(sen-sfr, dim=1, p=2)) * 0.1

            loss = output.loss + loss_trad
            self.log("train_trad", loss_trad, sync_dist=True)
            self.log("train_classi", output.loss, sync_dist=True)

        else: # For classic classification training
            loss = self.model(**batch).loss

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[dict]]:
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 100.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        # Should return only optimizer if no scheduler is used
        return [optimizer], [scheduler]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        output = self.model(**batch)
        self.log("val_loss", output.loss, sync_dist=True)
        return {"predictions": self.softmax(output.logits).tolist(), "references": batch['labels'].tolist()}
    
    def validation_epoch_end(
        self, outputs: Dict[str, Any], *kargs, **kwargs
    ) -> None:
        predictions = sum([b["predictions"] for b in outputs], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in outputs], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_" + k, v, sync_dist=True)

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        output = self.model(**batch)
        self.log("test_loss", output.loss, sync_dist=True)
        return {"predictions": self.softmax(output.logits).tolist(), "references": batch['labels'].tolist()}
    
    def test_epoch_end(
        self, outputs: Dict[str, Any], *kargs, **kwargs
    ) -> None:
        predictions = sum([b["predictions"] for b in outputs], [])
        predictions = [(a[0] < a[1]) * 1 for a in predictions]
        references = sum([b["references"] for b in outputs], [])

        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("test_" + k, v, sync_dist=True)
