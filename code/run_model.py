import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
import random
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, VisualBertModel
from models.visual_bert import VisualBertDualMasking
from data.dataloader import VisualStoryDataset
import wandb

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_num_object_classes(dataset):
    unique_classes = set()
    for item in dataset:
        if item is None:
            continue
        unique_classes.update(item['masked_classes'])
    return len(unique_classes)

def evaluate(model, data_loader, loss_fct, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(data_loader, desc='Validation', leave=False):
        if batch is None:
            continue
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        visual_embeddings = batch['visual_embeddings'].to(device)
        true_input_ids = batch['input_ids'].to(device)
        masked_classes = batch['masked_classes'].to(device)

        with torch.no_grad():
            prediction_scores, object_logits = model(input_ids, attention_mask, token_type_ids, visual_embeddings)
            mlm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), true_input_ids.view(-1))
            object_class_loss = loss_fct(object_logits, masked_classes)
            total_loss += (mlm_loss + object_class_loss).item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def train(model, train_loader, loss_fct, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc='Training', leave=False):
        if batch is None:
            continue
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        visual_embeddings = batch['visual_embeddings'].to(device)
        true_input_ids = batch['input_ids'].to(device)
        masked_classes = batch['masked_classes'].to(device)

        optimizer.zero_grad()
        prediction_scores, object_logits = model(input_ids, attention_mask, token_type_ids, visual_embeddings)
        mlm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), true_input_ids.view(-1))
        object_class_loss = loss_fct(object_logits, masked_classes)
        total_loss = mlm_loss + object_class_loss
        
        total_loss.backward()
        optimizer.step()
        wandb.log({'train_batch_loss': total_loss.item()})

    return total_loss / len(train_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='./data/VIST', help="Path to directory containing data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    wandb.login(key="a5607b415e91ffe216700bf7d3b4df114c44a534")
    wandb.init(project="visualbert", config=args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("==========Loading datasets...==========")
    train_dataset = VisualStoryDataset(f"{args.input_dir}/train-combined.json", args.input_dir, tokenizer)
    val_dataset = VisualStoryDataset(f"{args.input_dir}/val-combined.json", args.input_dir, tokenizer)

    # Wrap datasets with tqdm for progress bars
    train_loader = DataLoader(tqdm(train_dataset, desc='Loading Train Data'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tqdm(val_dataset, desc='Loading Val Data'), batch_size=args.batch_size, shuffle=False)


    print("==========Loading model...==========")

    num_object_classes = get_num_object_classes(train_dataset)
    model = VisualBertDualMasking('uclanlp/visualbert-nlvr2-coco-pre', num_object_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    print("==========Training...==========")

    torch.cuda.empty_cache()
    gc.collect()
    for epoch in range(args.epochs):
        avg_train_loss = train(model, train_loader, loss_fct, optimizer, device)
        avg_val_loss = evaluate(model, val_loader, loss_fct, device)
        
        wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss.item(), "avg_val_loss": avg_val_loss})
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {avg_train_loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}")

    # wandb.finish()

if __name__ == "__main__":
    main()
