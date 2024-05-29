from transformers import Trainer, AutoModel, TrainingArguments, BertTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import simcse_unsup_loss,load_train_data_unsupervised
from dataset import TrainDataset
from torch.utils.data import Dataset, DataLoader
from model import SimcseModel
import argparse
import torch.nn as nn
#定义自己的Trainer
class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
        loss = simcse_unsup_loss(outputs,device='cuda')

        return (loss, outputs) if return_outputs else loss
    
    

def train(args):
    model = SimcseModel(args.pretrain_model_path,args.pooler)
    arguments = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=100,
        num_train_epochs=args.epochs,
        save_strategy='steps',
        save_total_limit = 3
    )
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    train_data = load_train_data_unsupervised(tokenizer, args)   

    train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=8)
    
    trainer = MyTrainer(model,
              arguments,
              train_dataset=train_dataset)
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='output')
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_step", type=int, default=100, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=512, help="max length of input")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--train_file", type=str, default=r"train.txt")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="stella-large-v2")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='pooler to use')
    parser.add_argument("--train_mode", type=str, default='unsupervise', choices=['unsupervise', 'supervise'], help="unsupervise or supervise")
    parser.add_argument("--overwrite_cache", action='store_true', default=False, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_predict", action='store_true', default=True)

    args = parser.parse_args()

    train(args)

        