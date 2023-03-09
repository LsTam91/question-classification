import json 
import os

from torch.utils.data import DataLoader, ConcatDataset

# import pytorch_lightning as pl
from pytorch_lightning.loggers import  TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer #, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse

# from this project:
from model import classification_model, collator
from model import train_and_distil, trad_collator, classification_multilanguage

from evaluate_utils import HFMetric, MultiHFMetric

# To disable the model message
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser(
    description='Train classi model'
    )
parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
                    help='do not use GPUs (for dev only)')
parser.add_argument('--ndevices', dest='ndevices', type=int, default=1)

parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=False, action='store_true',
                    help='show progress bar' )

parser.add_argument('--name', dest="name", default="camembert-base000")
parser.add_argument('--model_name', dest="model_name", default="xlm-roberta-base", type=str)
parser.add_argument('--datasets-path', metavar='datasets_path', default="QA/Traduction/", type=str)

parser.add_argument('--log-every-n-steps', dest="log_every_n_steps", default=64, type=int,
                    help='log frequency')
parser.add_argument('--batch-size', dest="batch_size", default=8, type=int)
parser.add_argument('--max-epochs', dest="max_epochs", default=100, type=int,
                    help='number of training epoch' )
parser.add_argument('--save-top-k', dest="save_top_k", default=2, type=int)
parser.add_argument('--num-worker', dest="num_worker", default=32, type=int)
parser.add_argument('--noise', dest='noise', default=0.5, type=float,
                    help='amount of noise to add in data')
parser.add_argument('--distance', dest='distance', default='cosine', type=str,
                    help='cosine or l2')

parser.add_argument('--limit-train-batches', dest='limit_train_batches', default=2000, type=int)
parser.add_argument('--limit-val-batches', dest='limit_val_batches', default=10000, type=int)
parser.add_argument('--early-stop-criterion', dest='esc', type=str, default="f1",
                    help='the name of the criterion used for early stopping (using validation set)')
parser.add_argument('--patience', dest='patience', default=10, type=int,
                    help='epochs before you stop training if no improvment')

parser.add_argument('--precision', dest='precision', default=32, type=int,
                    help='32bit precision or mixed 16bit precision')
parser.add_argument('--model-type', dest='model_type', default='distil',
                    help='classic, dpr-like, or distil')

args = parser.parse_args()


# we define the class score above to avoid lambda fct in validation_metrics
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
        accuracy = HFMetric('accuracy', score('accuracy')),
        f1 = HFMetric('f1', score('f1')),
        recall = HFMetric('recall', score('recall')),
        precision = HFMetric('precision', score('precision'))
    )

    # To store the logs
    log_folder = os.path.expandvars("logs")

    #Loading the model
    if args.model_type == 'french': 
        # classic transformer classification model only in french:
        # TODO: le virer
        model = classification_model(model_name = args.model_name,
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name
            )

    elif args.model_type == 'distil':
        model = train_and_distil(model_name = os.path.expandvars("$HF_HOME/" + args.model_name),
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name,
            distance = args.distance
            )
        
        # Load en-fr traduction data
        trad = []
        for file in os.listdir('data/trad'):
            with open('data/trad/' + file, 'r') as fp:
                trad.append(json.load(fp))

        loader_trad = DataLoader(ConcatDataset(trad),
                                    batch_size=5,
                                    drop_last=True,
                                    collate_fn=trad_collator(model.tokenizer),
                                    shuffle=True,
                                    num_workers=args.num_worker
                                    )

    else:
        model = classification_multilanguage(model_name = os.path.expandvars("$HF_HOME/" + args.model_name),
            validation_callback = validation_metrics,
            log_dir = log_folder+args.name
            )


### TODO: Add more datasets and a test set (traduction of squad2, NQD and opus fr-en), add more data from davincii in french
    # # Load train and validation datasets
    train, valid, test = [], [], []
    for file in os.listdir('data/train'):
        with open('data/train/' + file, 'r') as fp:
            train.append(json.load(fp))

    for file in os.listdir('data/valid'):
        with open('data/valid/' + file, 'r') as fp:
            valid.append(json.load(fp))

    for file in os.listdir('data/test'):
        with open('data/test/' + file, 'r') as fp:
            test.append(json.load(fp))

    loader_classi = DataLoader(ConcatDataset(train),
                                batch_size=args.batch_size,
                                drop_last=True,
                                collate_fn=collator(model.tokenizer, corruption_rate=args.noise),
                                shuffle=True,
                                num_workers=args.num_worker
                                )

    valid_dataloader = DataLoader(ConcatDataset(valid),
                                batch_size=args.batch_size,
                                drop_last=False,
                                collate_fn = collator(model.tokenizer),
                                shuffle=False,
                                num_workers=args.num_worker
                                ) 
    
    test_dataloader = DataLoader(ConcatDataset(test),
                                batch_size=args.batch_size,
                                drop_last=False,
                                collate_fn = collator(model.tokenizer),
                                shuffle=False,
                                num_workers=args.num_worker
                                )

    # init the logger with the default tensorboard logger from lightning
    tb_logger = TensorBoardLogger(save_dir=log_folder, name=args.name)
    # tb_logger.log_hyperparams(vars(args))
    # We also log the learning rate, at each step
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # instanciate the differente callback for saving the model according to the different metrics
    checkpoint_callback_val_loss = ModelCheckpoint(monitor='val_loss', save_top_k=args.save_top_k, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_accuracy = ModelCheckpoint(monitor='val_accuracy', save_top_k=0, mode="max", filename="val-accuracy-checkpoint-{epoch:02d}-{val_accuracy:.2f}")
    checkpoint_callback_val_f1 = ModelCheckpoint(monitor='val_f1', save_top_k=args.save_top_k, mode="max", filename="val-f1-checkpoint-{epoch:02d}-{val_f1:.2f}")
    # checkpoint_callback_val_recall = ModelCheckpoint(monitor='val_recall', save_top_k=0, mode="max", filename="val-recall-checkpoint-{epoch:02d}-{val_recall:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_" + args.esc, min_delta=0.00, patience=args.patience, verbose=False, mode="max")

    callbacks = [
        lr_monitor,
        checkpoint_callback_val_loss,
        checkpoint_callback_val_accuracy,
        checkpoint_callback_val_f1,
        # checkpoint_callback_val_recall,
        early_stop_callback
    ]

    # Explicitly specify the process group backend if you choose to
    # ddp = DDPStrategy(process_group_backend="gloo")

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
        accumulate_grad_batches={0: 1, 400: max(64 // args.batch_size, 1)},
        accelerator='gpu' if(not args.cpu_only) else 'cpu',
        devices=args.ndevices,
        # auto_select_gpus=True,
        precision=args.precision
        # strategy=ddp #"ddp_find_unused_parameters_false" # strategy to train the model on different machine
    )

    trainer.fit(
        model=model,
        train_dataloaders={"classi": loader_classi, "trad": loader_trad} if args.model_type == 'distil' else loader_classi,
        val_dataloaders=valid_dataloader #{'valid': valid_dataloader, 'test': test_dataloader}
    )
    
    # test the model
    trainer.test(model, dataloaders=DataLoader(test_dataloader))

if __name__ == "__main__":
    main()
    
    