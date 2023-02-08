import json 
import os

# import torch
# from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

# import pytorch_lightning as pl
from pytorch_lightning.loggers import  TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer #, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse

# from this project:
from model import classification_model, collator, classification_test
from model import train_and_distil, trad_collator, classification_multilanguage

from evaluate_utils import HFMetric, MultiHFMetric
from dpr_like_model import collator_qc

# To disable the model message
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser(
    description='Train classi model'
    )
parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
                    help='do not use GPUs (for dev only)')
parser.add_argument('--ndevices', dest='ndevices', type=int, default=1)

parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=True, #action='store_true',
                    help='show progress bar' )

parser.add_argument('--name', dest="name", default="camembert-base000")
parser.add_argument('--model_name', dest="model_name", default="bert-base-multilingual-uncased")
parser.add_argument('--datasets-path', metavar='datasets_path',
                    default="QA/Traduction/")

parser.add_argument('--log-every-n-steps', dest="log_every_n_steps", default=64, type=int,
                    help='log frequency')
parser.add_argument('--batch-size', dest="batch_size", default=8, type=int)
parser.add_argument('--max-epochs', dest="max_epochs", default=100, type=int,
                    help='number of training epoch' )
parser.add_argument('--num-worker', dest="num_worker", default=16, type=int)
parser.add_argument('--noise', dest='noise', default=0.5,
                    help='amount of noise to add in data')
parser.add_argument('--distance', dest='distance', default='cosine',
                    help='cosine or l2')

parser.add_argument('--limit-train-batches', dest='limit_train_batches', default=2000, type=int)
parser.add_argument('--limit-val-batches', dest='limit_val_batches', default=10000, type=int)
parser.add_argument('--early-stop-criterion', dest='esc', type=str,
                    default="f1",
                    help='the name of the criterion used for early stopping (using validation set)')
parser.add_argument('--patience', dest='patience', default=10,
                    help='epochs before you stop training if no improvment')

parser.add_argument('--precision', dest='precision', default=32, type=int,
                    help='32bit precision or mixed 16bit precision')
parser.add_argument('--model-type', dest='model_type', default='distil', #action='store_true',
                    help='classic, dpr-like, or distil')

args = parser.parse_args()


class score:
    def __init__(self, name):
        self.name = name
    
    def __call__(self, x):
        return x[self.name]


def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Loading the metrics
    # Use of the different HuggingFace metrics f1, accuracy, recall
    validation_metrics = MultiHFMetric(
        accuracy = HFMetric('accuracy', score('accuracy')), # we define the fct sb_score and rouge_score above to avoid lambda fct
        f1 = HFMetric('f1', score('f1')),
        recall = HFMetric('recall', score('recall'))
    )

    # To store the logs
    log_folder = os.path.expandvars("logs")

    #Loading the model
    if args.model_type == 'dpr-like':
        # To try some hand made models
        model = classification_test(model_name = args.model_name,
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name
            )
        # train_collator = collator_qc(model.tokenizer, corruption_rate = 0.4)
        valid_collator = collator_qc(model.tokenizer)

    elif args.model_type == 'classic': 
        # classic transformer classification model:
        model = classification_model(model_name = args.model_name,
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name
            )
        # train_collator = collator(model.tokenizer, corruption_rate=0.4)
        valid_collator = collator(model.tokenizer)

    elif args.model_type == 'distil':
        model = train_and_distil(model_name = os.path.expandvars("$HF_HOME/" + args.model_name), #"xlm-roberta-base", # TODO: args.model_name,
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name,
            distance = args.distance
            )
        # train_collator = collator_two_task(model.tokenizer, corruption_rate=0.4)
        valid_collator = collator(model.tokenizer)
    else:
        model = classification_multilanguage(model_name = os.path.expandvars("$HF_HOME/" + args.model_name), #"xlm-roberta-base", # TODO: args.model_name,
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name
            )
        valid_collator = collator(model.tokenizer)

### TODO: Load dataset, first draft to modify later: (traduction of squad2, NQD and opus fr-en)
    
    train, valid = [], []
    for file in os.listdir('data/train'):
        with open('data/train/' + file, 'r') as fp:
            train.append(json.load(fp))

    for file in os.listdir('data/valid'):
        with open('data/valid/' + file, 'r') as fp:
            valid.append(json.load(fp))

    # # Loading the datasets
    # with open('data/train/squadv2_train', 'r') as fp:
    #     french_qa = json.load(fp)

    # with open("data/train/Natural_Question_Louis_tok", "r") as fp:
    #     english_qa = json.load(fp)

    with open('data/trad/opus_en_fr_st', 'r') as fp:
        opus = json.load(fp)

    # with open('data/valid/short_valid', 'r') as fp:
    #     valid_fr_qa = json.load(fp)

    loader_classi = DataLoader(ConcatDataset(train), # french_qa, 
                                batch_size=args.batch_size,
                                drop_last=True,
                                collate_fn=collator(model.tokenizer, corruption_rate=args.noise, language = 'fr'),
                                shuffle=True,
                                num_workers=args.num_worker
                                )

    loader_trad = DataLoader(opus,
                                batch_size=5,  #args.batch_size,
                                drop_last=True,
                                collate_fn=trad_collator(model.tokenizer),
                                shuffle=True,
                                num_workers=args.num_worker
                                )

    train_dataloader = {"classi": loader_classi, "trad": loader_trad}

    valid_dataloader = DataLoader(ConcatDataset(valid),
                                batch_size=args.batch_size,
                                drop_last=False,
                                collate_fn = valid_collator,
                                shuffle=True,
                                num_workers=args.num_worker
                                )
    
    # init the logger with the default tensorboard logger from lightning
    tb_logger = TensorBoardLogger(save_dir=log_folder, name=args.name)
    # tb_logger.log_hyperparams(vars(args))
    # We also log the learning rate, at each step
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # instanciate the differente callback for saving the model according to the different metrics
    checkpoint_callback_val_loss = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_accuracy = ModelCheckpoint(monitor='val_accuracy', save_top_k=1, mode="max", filename="val-accuracy-checkpoint-{epoch:02d}-{val_accuracy:.2f}")
    checkpoint_callback_val_f1 = ModelCheckpoint(monitor='val_f1', save_top_k=2, mode="max", filename="val-f1-checkpoint-{epoch:02d}-{val_f1:.2f}")
    checkpoint_callback_val_recall = ModelCheckpoint(monitor='val_recall', save_top_k=0, mode="max", filename="val-recall-checkpoint-{epoch:02d}-{val_recall:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_" + args.esc, min_delta=0.00, patience=args.patience, verbose=False, mode="max")

    callbacks = [
        lr_monitor,
        checkpoint_callback_val_loss,
        checkpoint_callback_val_accuracy,
        checkpoint_callback_val_f1,
        checkpoint_callback_val_recall,
        early_stop_callback
    ]

    # Explicitly specify the process group backend if you choose to
    ddp = DDPStrategy(process_group_backend="gloo")

    # Instanciate the trainer
    trainer = Trainer(
        logger=tb_logger, 
        log_every_n_steps=args.log_every_n_steps, 
        callbacks=callbacks, 
        enable_progress_bar=args.enable_progress_bar,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs, 
        deterministic=True,
        accumulate_grad_batches=8,
        accelerator='gpu' if(not args.cpu_only) else 'cpu',
        devices=args.ndevices,
        # auto_select_gpus=True,
        precision=args.precision,
        strategy=ddp #"ddp_find_unused_parameters_false" # strategy to train the model on different machine
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )
    

if __name__ == "__main__":
    main()
    
    