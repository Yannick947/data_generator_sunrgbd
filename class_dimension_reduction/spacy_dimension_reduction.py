import json

import spacy

CLEANED_CLASSES_PATH = 'C:/Users/Yannick/Google Drive/instance_segmentation/data_generator_sunrgbd/class_dimension_reduction/cleaned_classes.json'
SIMILARITY_THRESHOLD = 0.5
PREDEFINED_CLASSES = ['bed ', 'tool', 'desk', 'chair', 'table', 'wardrobe', 'sofa', 'bookcase']

def main():
    nlp = spacy.load("en_vectors_web_lg")

    with open(CLEANED_CLASSES_PATH, 'r') as f:
        classes = json.load(f)
    
    # Put classes all in one string and tokenize
    joined_classes = " ".join(word for word in classes.values())
    tokens = nlp(joined_classes)

    joined_PREDEFINED_CLASSES = " ".join(word for word in PREDEFINED_CLASSES)
    predefined_class_tokens = nlp(joined_PREDEFINED_CLASSES)

    inverse_class_map = dict()
    for key, value in classes.items():
        if inverse_class_map.get(value) is not None:
            inverse_class_map[value].append(key)
        else: 
            inverse_class_map[value] = [key]

    class_map = dict()
    for class_token in tokens:

        # Try a blind load of the class otherwise continue and skip token to speed things up
        try:
            class_keys = inverse_class_map[class_token.text] 
        except:
            continue

        highest_similarity = 0
        highest_similarity_class = 'unknown'
        
        for predefined_class in predefined_class_tokens:
            similarity = predefined_class.similarity(class_token)
            if similarity > SIMILARITY_THRESHOLD and similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_class = predefined_class.text

        for class_key in class_keys:
            class_map[class_key] = highest_similarity_class

    if class_map.get(' ') is not None:
        class_map.pop(' ')

    classes_count = dict()
    for predefined_label in class_map.values():  
        if predefined_label in classes_count.keys():
            classes_count[predefined_label] += 1
        else: 
            classes_count[predefined_label] = 1

    for key in classes_count.keys():
        print(f'Number of labels assigned to the class {key} is {classes_count[key]}.')

    with open('./class_dimension_reduction/class_map_cleaned.json', 'w') as f:
        json.dump(class_map, f, indent=4)


if __name__ == '__main__':
    main()