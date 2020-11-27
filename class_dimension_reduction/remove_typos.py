import json

from spellchecker import SpellChecker

CLASSES_PATH = 'C:/Users/Yannick/Google Drive/instance_segmentation/data_generator_sunrgbd/class_map_detected_full.json'

def main():

    spell = SpellChecker()
    with open(CLASSES_PATH, 'r') as f:
        classes = list(json.load(f).keys())

    print('Number of initial classes: ', len(set(classes)))
    # Remove chars which are not letters
    cleaned_words = list()    
    for word in classes:
        cleaned_words.append(''.join(e for e in word.lower() if e.isalpha()))

    # find those words that may be misspelled
    misspelled = spell.unknown(cleaned_words)

    for i, bad_word in enumerate(list(misspelled)):
        # Get the one `most likely` answer
        correction = spell.correction(bad_word)
        if correction != bad_word: 
            cleaned_words.remove(bad_word)
            cleaned_words.append(correction)
        
        if i % 100 == 0:
            print(i)
            
    print('Number of words in classes dict ', len(set(cleaned_words)))
    save_dict = {'classes': cleaned_words}

    with open('cleaned_classes.json', 'w') as f:
        json.dump(save_dict, f, indent=4)


if __name__ == '__main__':
    main()