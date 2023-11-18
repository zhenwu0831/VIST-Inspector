import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, VisualBertForPreTraining
import json
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
import os
import sys

# Initialize ResNet model without the final fully connected layer
resnet_model = models.resnet50(pretrained=True)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()

# Function to request and transform an image
def request_image(url, transform):
    rsp = requests.get(url, stream=True)
    img = Image.open(BytesIO(rsp.content)).convert('RGB')
    return transform(img)

# Function to load the VIST dataset
def load_vist_dataset(file_path, image_size=(224, 224)):
    with open(file_path, 'r') as file:
        vist_data = json.load(file)
    
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    processed_data = []

    for item in tqdm(vist_data):
        stories = item['stories']
        cur_url = item['urls']

        try:
            img_tensors = []
            for url in cur_url:
                img_tensors.append(request_image(url, transform))
            img_tensor = torch.stack(img_tensors)
        except:
            continue

        for story in stories:
            # processed_data.append({'url'})
            sto = " ".join(story)
            processed_data.append({'image': img_tensor, 'story': sto})

    return processed_data

def load_sorted_dataset_from_directory(directory_path, percentage=1):
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.json')]
    all_files.sort()
    selected_files = all_files[:int(len(all_files) * percentage)]

    dataset = []
    for file in selected_files:
        dataset.extend(load_vist_dataset(file))

    return dataset

# Load the VIST dataset
# vist_data_path = 'VIST-Inspector-main/data/VIST/train'
# vist_data = load_sorted_dataset_from_directory(vist_data_path)

train_data_path = './data/VIST/train'
train_data = load_sorted_dataset_from_directory(train_data_path)

val_data_path = './data/VIST/val'
val_data = load_sorted_dataset_from_directory(val_data_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# VIST dataset class
class VISTDataset(Dataset):
    def __init__(self, data, tokenizer, resnet_model, visual_embedding_dim=768, max_length=512, mask_probability=0.15):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.resnet_model = resnet_model
        # Projection layer to match VisualBERT's expected visual embedding size
        # self.projection = nn.Linear(2048, visual_embedding_dim)
    
    # ... other methods ...

    def __len__(self):
        return len(self.data)

    def mask_tokens(self, inputs):
        """ Randomly mask tokens for MLM with a mask_probability """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, self.mask_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # Mask token is the [MASK] token
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

    def __getitem__(self, idx):
        item = self.data[idx]
        story = item['story']

        image_feature_list = []
        # Extract features using ResNet
        for image in item['image']:
            with torch.no_grad():
                # Inside the __getitem__ method
                image_features = self.resnet_model(image.unsqueeze(0))
                # print("Shape after ResNet:", image_features.shape)

                image_features = image_features.view(image_features.size(0), -1)
                # print("Shape after flattening:", image_features.shape)

                image_feature_list.append(image_features)
                #image_features = self.projection(image_features)
                #print("Shape after projection:", image_features.shape)

        image_features = torch.cat(image_feature_list, dim=0)

        # Tokenize text and prepare inputs for VisualBERT
        inputs = self.tokenizer.encode_plus(
            story, 
            add_special_tokens=True,
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        inputs_ids, labels = self.mask_tokens(inputs['input_ids'].squeeze())

        num_visual_tokens = len(image_feature_list)
        visual_labels = torch.full((num_visual_tokens,), -100)  # MLM labels for visual tokens
        labels = torch.cat([labels, visual_labels], dim=0)

        # Ensure that visual_embeds is of correct dimensionality
        # image_features = image_features.squeeze(0)  # Remove batch dimension if necessary

        # Create an attention mask for the inputs
        attention_mask = inputs['attention_mask']
        
        # Create a visual attention mask with the same batch size and the number of visual features
        visual_attention_mask = torch.ones((image_features.size(0),), dtype=torch.long).unsqueeze(0)  # Add batch dimension

        return {
            'input_ids': inputs_ids, 
            'labels': labels, 
            'attention_mask': attention_mask, 
            'visual_embeds': image_features, 
            'visual_attention_mask': visual_attention_mask  # Add this line
        }

# Initialize dataset and dataloader

train_dataset = VISTDataset(train_data, tokenizer, resnet_model)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = VISTDataset(val_data, tokenizer, resnet_model)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Initialize the VisualBERT model
model = VisualBertForPreTraining.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

# Training loop function
def train(model, dataloader, epoch):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    for batch in dataloader:
        # ... your code to perform training ...
        input_ids = batch['input_ids']
        labels = batch['labels']
        visual_embeds = batch['visual_embeds']
        # print("Visual embeds shape:", visual_embeds.shape)
        attention_mask = batch['attention_mask']  # Add this
        visual_attention_mask = batch['visual_attention_mask']  # Add this

        outputs = model(
            input_ids=input_ids, 
            visual_embeds=visual_embeds, 
            attention_mask=attention_mask,  # Add this
            visual_attention_mask=visual_attention_mask,  # Add this
            labels=labels
        )
        loss = outputs.loss

        # ... rest of your training loop

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print(f"Train: Epoch: {epoch}, Loss: {loss.item()}")

def evaluate(model, dataloader, epoch):
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            visual_embeds = batch['visual_embeds']
            attention_mask = batch['attention_mask']  # Add this
            visual_attention_mask = batch['visual_attention_mask']  # Add this

            outputs = model(
                input_ids=input_ids, 
                visual_embeds=visual_embeds, 
                attention_mask=attention_mask,  # Add this
                visual_attention_mask=visual_attention_mask,  # Add this
                labels=labels
            )
            loss = outputs.loss

            # ... rest of your training loop
            
        print(f"Dev: Epoch: {epoch}, Loss: {loss.item()}")

num_epochs = 3

save_path = './results'

def run(model, train_loader, val_loader, epochs):
    for epoch in tqdm(range(epochs)):
        train(model, train_loader, epoch)
        evaluate(model, val_loader, epoch)

        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pt'))

run(model, dataloader, val_dataloader, num_epochs)