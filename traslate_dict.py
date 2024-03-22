import json
with open('imagenet.json', 'r') as f:
    labels_dict = json.load(f)

print(list(labels_dict.values()))