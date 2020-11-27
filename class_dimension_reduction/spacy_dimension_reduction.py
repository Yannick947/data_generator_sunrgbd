import json

import spacy

CLASSES_PATH = 'C:/Users/Yannick/Google Drive/instance_segmentation/data_generator_sunrgbd/class_dimension_reduction/cleaned_classes.json'
# CLASSES_PATH = 'C:/Users/Yannick/Google Drive/instance_segmentation/data_generator_sunrgbd/class_map_detected_full.json'

predefined_classes = ['bed ', 'tool', 'desk', 'chair', 'table', 'wardrobe', 'sofa', 'bookcase']

def main():
    nlp = spacy.load("en_vectors_web_lg")

    with open(CLASSES_PATH, 'r') as f:
        classes = list(json.load(f)['classes']) 
    
    class_god_string = " ".join(word for word in classes)
    tokens = nlp(class_god_string)

    predefined_classes_god_string = " ".join(word for word in predefined_classes)
    predefined_class_tokens = nlp(predefined_classes_god_string)

    class_map = dict()
    for class_token in tokens:
        highest_similarity = 0
        highest_similarity_class = 'unknown'
        
        for predefined_class in predefined_class_tokens:
            similarity = predefined_class.similarity(class_token)
            if similarity > 0.5 and similarity > highest_similarity:
                highest_similarity = similarity
                highest_similarity_class = predefined_class.text

        class_map[class_token.text] = highest_similarity_class

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