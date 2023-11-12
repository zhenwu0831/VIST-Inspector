import torch
import numpy as np
import spacy
import requests
from io import BytesIO
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
import cv2
from torchvision.models import resnet50  # For feature extraction

# Initialize object detection model, NLP model, and feature extractor
object_detector = fasterrcnn_resnet50_fpn(pretrained=True).eval()
nlp = spacy.load("en_core_web_sm")
feature_extractor = resnet50(pretrained=True)


def download_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except requests.RequestException as e:
        print(f"Error downloading {image_url}: {e}")
        return None

def preprocess_image(image):
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def detect_objects(image):
    if not isinstance(image, torch.Tensor):
        image = F.to_tensor(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        predictions = object_detector(image)

    detections = predictions[0]

    # Filter out low-confidence detections
    # 'scores' is a tensor of confidence scores
    # 'boxes' is a tensor of bounding box coordinates
    # 'labels' is a tensor of label indices
    high_confidence_indices = detections['scores'] > 0.5
    high_confidence_detections = {
        'boxes': detections['boxes'][high_confidence_indices],
        'labels': detections['labels'][high_confidence_indices],
        'scores': detections['scores'][high_confidence_indices]
    }

    return high_confidence_detections

def analyze_story(story_text):
    doc = nlp(story_text)
    entities = [ent.text.lower() for ent in doc.ents]
    actions = [token.lemma_.lower() for token in doc if token.pos_ == 'VERB']
    nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN']
    adjectives = [token.text.lower() for token in doc if token.pos_ == 'ADJ']
    return {
        "entities": entities,
        "actions": actions,
        "nouns": nouns,
        "adjectives": adjectives
    }

def advanced_story_image_analysis(story_text, image):
    story_analysis = analyze_story(story_text)
    detected_objects = detect_objects(image)
    linked_elements = {}
    for element in story_analysis:
        for obj, label in zip(detected_objects['boxes'], detected_objects['labels']):
            label_str = str(label.item()).lower()
            if element in label_str:
                if element not in linked_elements:
                    linked_elements[element] = []
                linked_elements[element].append(obj.tolist())
    return linked_elements


def mask_image_based_on_linked_elements(image, linked_elements, mask_type='blur'):
    print(linked_elements)
    # Clone the tensor to create a copy
    masked_image = image.clone()

    # Ensure the image is in NumPy array format for masking
    masked_image = masked_image.permute(1, 2, 0).numpy()

    masked_classes = []
    for element, bboxes in linked_elements.items():
        for bbox in bboxes:
            bbox = [int(x) for x in bbox]
            relevance_score = calculate_element_relevance(element, linked_elements)
            apply_contextual_mask(masked_image, bbox, mask_type, relevance_score)
            masked_class = int(element)
            masked_classes.append(element)

    # Convert back to tensor after masking
    masked_image = torch.from_numpy(masked_image).permute(2, 0, 1)

    return masked_image,  masked_classes

def apply_contextual_mask(image, bbox, mask_type, relevance_score):
    if mask_type == 'blur':
        kernel_size = int(23 * relevance_score) | 1  # Ensure kernel size is odd
        sigma = 30 * relevance_score
        image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.GaussianBlur(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], (kernel_size, kernel_size), sigma)
    elif mask_type == 'blackout' and relevance_score > 0.7:
        image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

def calculate_element_relevance(element, linked_elements):
    return len(linked_elements.get(element, [])) / len(linked_elements)

def convert_to_feature_vectors(masked_image):
    """
    Convert the processed (masked) image to a sequence of feature vectors.
    """
    preprocessed_img = preprocess_image(masked_image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        feature_vectors = feature_extractor(preprocessed_img)

    return feature_vectors.squeeze(0)