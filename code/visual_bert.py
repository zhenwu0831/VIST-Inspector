import torch
import torch.nn as nn
from transformers import VisualBertModel, VisualBertConfig
# import sys

class VisualBertDualMasking(nn.Module):
    def __init__(self, visual_bert_model_name, num_object_classes):
        super().__init__()
        config = VisualBertConfig.from_pretrained(visual_bert_model_name)
        self.visual_bert = VisualBertModel(config)

        self.visual_projection = nn.Linear(2048, config.hidden_size)
        
        # For classification of the masked object
        self.object_classifier = nn.Linear(config.hidden_size, num_object_classes)

    def forward(self, inputs):
        visual_embeddings = self.visual_projection(inputs['visual_embeddings'])
        # Create visual token type ids (all ones)
        visual_token_type_ids = torch.ones(inputs['visual_embeddings'].shape[:-1], dtype=torch.long)

        # Create a visual attention mask (all ones)
        visual_attention_mask = torch.ones(inputs['visual_embeddings'].shape[:-1], dtype=torch.float)

        # move to device
        device = inputs['visual_embeddings'].device
        visual_token_type_ids = visual_token_type_ids.to(device)
        visual_attention_mask = visual_attention_mask.to(device)

        # print(attention_mask.shape, visual_embeddings.shape, visual_attention_mask.shape)

        # sys.exit()

        # Forward pass through VisualBERT
        inputs.update(
            {
                "visual_embeds": visual_embeds,
                "visual_token_type_ids": visual_token_type_ids,
                "visual_attention_mask": visual_attention_mask,
            }
        )
        outputs = self.visual_bert(**inputs)

        pooled_output = outputs.pooler_output

        # Object Class Prediction
        object_logits = self.object_classifier(pooled_output)

        return object_logits
