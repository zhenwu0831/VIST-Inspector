import json
import os
import argparse
import textstat
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from collections import Counter

nlp = spacy.load("en_core_web_sm")

stop_words = STOP_WORDS

# Feature extraction function
def extract_nlp_features(text):
    # Process the text with spaCy. This includes tokenization, lemmatization, POS tagging, and NER
    doc = nlp(text)

    # Calculate Type-Token Ratio (TTR) for lexical diversity
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    unique_words = set(tokens)
    ttr = len(unique_words) / len(tokens) if tokens else 0

    # Sentence complexity (average number of words per sentence)
    sentences = list(doc.sents)
    sentence_complexity = sum(len(sentence) for sentence in sentences) / len(sentences) if sentences else 0

    # Readability score using Flesch-Kincaid grade level
    readability_score = textstat.flesch_kincaid_grade(text)

    # POS distribution
    pos_dist = Counter(token.pos_ for token in doc)

    # Named Entity Recognition using spaCy
    ner_entities = [ent.text for ent in doc.ents]
    ner_freq_dist = Counter(ner_entities)

    # Merge the dictionaries
    features = {
        'type_token_ratio': ttr,
        'sentence_complexity': sentence_complexity,
        'readability_score': readability_score,
        'pos_distribution': dict(pos_dist),
        'named_entity_frequency': dict(ner_freq_dist)
    }
    return features

# Main function placeholder for demonstration
def main(args):
    input_dir = args.input_dir + args.model_name + '/' + args.data_split
    model_name = args.model_name
    data_split = args.data_split

    # for all json files in that directory, read the text and extract features, construct one big json file
    combined_json = []
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            file_path = os.path.join(input_dir, file)

            print('Processing file: ', file_path)

            with open(file_path, 'r') as f:
                data = json.load(f)

            album_id = data['album_id']
            image_urls = data['photo_sequence_urls']
            texts = [story['edited_story_text'] for story in data['edited_stories']]
            texts.append(data['auto_story_text_normalized'])

            for i, text in enumerate(texts):
                auto = True if i == len(texts) - 1 else False
                features = extract_nlp_features(text)
                combined_json.append({
                    'album_id': album_id,
                    'image_urls': image_urls,
                    'auto': auto,
                    'text': text,
                    'features': features
                })
            
            print('Finished processing file: ', file_path)
        
    # write the combined json file to disk
    output_path = '../data/' + model_name + '_' + data_split + '_features.json'
    with open(output_path, 'w') as f:
        json.dump(combined_json, f, indent=4)
    print('Finished writing combined json file to ' + output_path)

# Call the main function if this script is executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../data/VIST-Edit/')
    parser.add_argument('--model_name', type=str, default='AREL')
    parser.add_argument('--data_split', type=str, default='train')
    args = parser.parse_args()

    main(args)
