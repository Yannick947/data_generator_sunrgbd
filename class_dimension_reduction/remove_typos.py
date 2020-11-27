import json

from spellchecker import SpellChecker

CLASSES_PATH = 'C:/Users/Yannick/Google Drive/instance_segmentation/data_generator_sunrgbd/sunrgbd_generator/class_map_detected_full.json'

def main():

    spell = SpellChecker()
    with open(CLASSES_PATH, 'r') as f:
        class_map = json.load(f)

    print('Number of initial classes: ', len(class_map.keys()))
    # Remove numbers
    for word in class_map.keys():
        is_alpha_word = ''.join(e for e in word.lower() if not e.isdigit())
        class_map[word] = is_alpha_word

    print(len(set(class_map.values())), ' number of classes remaining after removing special chars and numbers.')
    # find those words that may be misspelled
    misspelled = spell.unknown(list(class_map.values()))

    for i, word in enumerate(list(class_map.keys())):
        cleaned_word = class_map[word]

        # Update the class mapping if the cleaned word is misspelled according to SpellChecker
        if cleaned_word in misspelled:
            # Get the one `most likely` answer 
            correction = spell.correction(cleaned_word)
            class_map[word] = correction
        
        if i % 500 == 0 and i != 0:
            print(i)
            
    with open('cleaned_classes.json', 'w') as f:
        json.dump(class_map, f, indent=4)

    print(len(set(class_map.values())), ' number of classes remaining after removing spelling errors according to SpellChecker.')

if __name__ == '__main__':
    main()
