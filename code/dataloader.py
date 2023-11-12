import torch
from torch.utils.data import Dataset, DataLoader
import json
from .image_masking import *
from transformers import BertTokenizer

class VisualStoryDataset(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, transform=None):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

        with open(json_file, 'r') as file:
            self.data = []
            for item in json.load(file):
                for story_group in item['stories']:
                    for storylet in story_group:
                        self.data.append((item['image_info'], storylet))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info, storylet = self.data[idx]
        image_url = image_info['url_o']
        story_text = storylet['text']  

        print(f"Processing {image_url}...")
        image = download_image(image_url)
        if image is None:
            return None

        image = preprocess_image(image)

        linked_elements = advanced_story_image_analysis(story_text, image)
        masked_image, masked_classes = mask_image_based_on_linked_elements(image, linked_elements) 
        feature_vectors = convert_to_feature_vectors(masked_image)

        masked_classes = torch.tensor(masked_classes, dtype=torch.int64)

        # Tokenize the story text
        inputs = self.tokenizer(story_text, return_tensors='pt')

        inputs.update({'visual_embeddings': feature_vectors, 'masked_classes': masked_classes})

        return inputs