import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data from JSON file
with open('../data/ling_features/GLAC_train_features.json', 'r') as file:
    train_data = json.load(file)
with open('../data/ling_features/GLAC_dev_features.json', 'r') as file:
    dev_data = json.load(file)
with open('../data/ling_features/GLAC_test_features.json', 'r') as file:
    test_data = json.load(file)

# Combine data into one list
data = train_data + dev_data + test_data

# Initialize lists to store features for each group
auto_true_features = {'type_token_ratio': [], 'sentence_complexity': [], 'readability_score': []}
auto_false_features = {'type_token_ratio': [], 'sentence_complexity': [], 'readability_score': []}

# Initialize dictionaries to store POS counts for each group
auto_true_pos_distribution = {}
auto_false_pos_distribution = {}

# Initialize the POS tags we're interested in
pos_tags = ['PRON', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'PUNCT', 'AUX', 'ADV', 'PART', 'SCONJ', 'CCONJ']

# Ensure all POS tags are represented in the dictionaries, even if they're not in the first item
for tag in pos_tags:
    auto_true_pos_distribution[tag] = []
    auto_false_pos_distribution[tag] = []

# Populate the lists with data
for item in data:
    features = item['features']
    pos_dist = features['pos_distribution']
    
    # Populate features for t-tests
    if item['auto']:
        auto_true_features['type_token_ratio'].append(features['type_token_ratio'])
        auto_true_features['sentence_complexity'].append(features['sentence_complexity'])
        auto_true_features['readability_score'].append(features['readability_score'])
        # Populate POS counts
        for tag in pos_tags:
            auto_true_pos_distribution[tag].append(pos_dist.get(tag, 0))
    else:
        auto_false_features['type_token_ratio'].append(features['type_token_ratio'])
        auto_false_features['sentence_complexity'].append(features['sentence_complexity'])
        auto_false_features['readability_score'].append(features['readability_score'])
        # Populate POS counts
        for tag in pos_tags:
            auto_false_pos_distribution[tag].append(pos_dist.get(tag, 0))

# Perform statistical analysis (t-tests) for each feature
for feature in auto_true_features:
    true_mean = np.mean(auto_true_features[feature])
    false_mean = np.mean(auto_false_features[feature])
    t_stat, p_val = stats.ttest_ind(auto_true_features[feature], auto_false_features[feature])

    print(f"Feature: {feature}")
    print(f"Auto True Mean: {true_mean}, Auto False Mean: {false_mean}")
    print(f"T-statistic: {t_stat}, P-value: {p_val}\n")

    # Visualize the feature distributions
    plt.figure(figsize=(10, 4))
    plt.hist(auto_true_features[feature], alpha=0.5, label='Auto True')
    plt.hist(auto_false_features[feature], alpha=0.5, label='Auto False')
    plt.title(f"Distribution of Feature: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Function to perform t-test and plot histogram for POS distribution
def analyze_and_plot_pos_distribution(auto_true_pos, auto_false_pos, tag):
    true_mean = np.mean(auto_true_pos)
    false_mean = np.mean(auto_false_pos)
    t_stat, p_val = stats.ttest_ind(auto_true_pos, auto_false_pos)

    print(f"POS Tag: {tag}")
    print(f"Auto True Mean: {true_mean}, Auto False Mean: {false_mean}")
    print(f"T-statistic: {t_stat}, P-value: {p_val}\n")

    # Visualize the POS distributions
    plt.figure(figsize=(10, 4))
    bins = np.arange(min(auto_true_pos+auto_false_pos)-0.5, max(auto_true_pos+auto_false_pos)+1.5, 1)
    plt.hist(auto_true_pos, alpha=0.5, label='Auto True', bins=bins)
    plt.hist(auto_false_pos, alpha=0.5, label='Auto False', bins=bins)
    plt.title(f"Distribution of POS Tag: {tag}")
    plt.xlabel(f"Count of {tag}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Analyze and plot POS distribution
for tag in pos_tags:
    analyze_and_plot_pos_distribution(auto_true_pos_distribution[tag], auto_false_pos_distribution[tag], tag)
