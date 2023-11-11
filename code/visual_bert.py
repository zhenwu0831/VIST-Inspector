import torch
import torch.nn as nn
from transformers import VisualBertModel, VisualBertConfig

class VisualBertDualMasking(nn.Module):
    def __init__(self, visual_bert_model_name, num_object_classes):
        super().__init__()
        config = VisualBertConfig.from_pretrained(visual_bert_model_name)
        self.visual_bert = VisualBertModel(config)
        
        # For MLM task
        self.cls = nn.Linear(config.hidden_size, config.vocab_size)

        # For classification of the masked object
        self.object_classifier = nn.Linear(config.hidden_size, num_object_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, visual_embeddings, masked_classes=None):
        outputs = self.visual_bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   visual_embeds=visual_embeddings,
                                   output_hidden_states=True,
                                   return_dict=True)

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # MLM Prediction
        prediction_scores = self.cls(sequence_output)

        # Object Class Prediction
        object_logits = self.object_classifier(pooled_output)

        return prediction_scores, object_logits
