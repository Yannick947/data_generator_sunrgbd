import os 
import json

from sunrgbd_generator.sunrgbd_to_maskrcnn import Sun_To_MASKRCNN

# Make sure that last character is slash ('/')
ROOT_DIR_SUNRGBD = 'C:/Users/Yannick/Downloads/SUNRGBD/'
PATH_CLASS_MAP = os.path.join(ROOT_DIR_SUNRGBD, )


def main():
    label_transformer = Sun_To_MASKRCNN(os.path.join(ROOT_DIR_SUNRGBD, 'seg37list.mat'), 
                                        known_classes_only=True)
    number_of_errors = 0

    for dirpath, _, filenames in os.walk(ROOT_DIR_SUNRGBD):
        for filename in filenames:                
            if  ('.jpg' in filename) or ('.png' in filename):
                image_filepath = os.path.join(dirpath, filename)
                if 'image' in image_filepath:
                    upper_dirname = dirpath[:dirpath.find('\\image')]

                    # specific transformations to get instance segmentation for maskrcnn
                    label_filepath = os.path.join(upper_dirname, 'annotation2Dfinal', 'index.json')

                    try: 
                        with open(label_filepath, 'r') as f:
                            single_label = json.load(f)
                        label_transformer.process_label(single_label,
                                                        image_filepath,
                                                        filename)
                    except Exception as e:
                        number_of_errors += 1 
                        print('Transformation failed for file ', label_filepath, '\n Error:', e)
                else: 
                    continue
    print('Annotation files which could not be opened due to invalid file syntax: ', number_of_errors)
    label_transformer.save_labels()


if __name__ == '__main__':
    main()