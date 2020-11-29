import os 
import json

from sunrgbd_generator.sunrgbd_to_maskrcnn import Sun_To_MASKRCNN

# Make sure that last character is slash ('/')
ROOT_DIR_SUNRGBD = 'C:/Users/Yannick/Downloads/SUNRGBD/'
ROOT_SAVE_PATH_LABELS = 'C:/Users/Yannick/Google Drive/instance_segmentation/data_generator_sunrgbd'
PATH_CLASS_MAP = os.path.join(ROOT_SAVE_PATH_LABELS, 'class_dimension_reduction', 'class_map_cleaned.json')


def main():
    label_transformer = Sun_To_MASKRCNN(path_to_class_map=PATH_CLASS_MAP, 
                                        known_classes_only=True, 
                                        include_image_size=True, 
                                        root_sunrgbd=ROOT_DIR_SUNRGBD)
    number_of_errors = 0

    for dirpath, _, filenames in os.walk(ROOT_DIR_SUNRGBD):
        for filename in filenames:                
            if  ('.jpg' in filename) or ('.png' in filename):
                image_filepath = os.path.join(dirpath, filename)
                if 'image' in image_filepath:
                    # TODO: Fix for using in unix
                    upper_dirname = dirpath[:dirpath.find('\\image')]

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