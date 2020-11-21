import os 
import json
import time 

ROOT_DIR_SUNRGBD = 'C:/Users/Yannick/Downloads/SUNRGBD'
BASE_PATH_TARGET_PLATFORM = '/cvhci/data/depth/SUNRGBD'

def main():
    label_transformer = Sun_To_MASKRCNN(os.path.join(ROOT_DIR_SUNRGBD, 'seg37list.mat'))
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
                        label_transformer.add_label(single_label,
                                                    image_filepath,
                                                    filename)
                    except Exception as e: 
                        print('Transformation failed for file ', label_filepath, '\n Error:', e)
                        number_of_errors += 1
                else: 
                    continue

    print(f'Number of files that could not be parsed {number_of_errors}')
    print(f'{label_transformer.label_id} labels processed, saving to json ..')

    label_transformer.save_labels()
    label_transformer.save_class_map()

class Sun_To_MASKRCNN(object):

    def __init__(self, path_to_class_map, known_classes_only=False):
        self.class_map = self.__get_class_map('class_map.json')
        self.label_dict = {'Generated on': time.time()}
        self.label_id = 0
        self.detected_classes = dict()
        self.num_classes = 0
        self.known_classes_only = known_classes_only

    def __get_class_map(self, path_to_class_map):
        with open (path_to_class_map) as f:
            return json.load(f)

    def add_label(self, single_label, path_to_image, image_name):
        ''' Specific transformation for 2D instance segmentation for MASK_RCNN of matterport'''

        frames = single_label['frames'][0]['polygon']
        classes = single_label['objects']

        self.label_dict[self.label_id] = {'path_to_image': path_to_image, 
                                     'image_name': image_name, 
                                     'regions':list(),
                                     'classes':list()}

        for frame in frames:
            class_of_object = classes[frame['object']]['name']
            if self.known_classes_only and class_of_object not in self.class_map.keys():
                continue

            self.label_dict[self.label_id]['regions'].append({"name": "polygon", 
                                                              "all_points_x":frame['x'], 
                                                              "all_points_y":frame['y']})

            if not bool(self.detected_classes.keys()) or class_of_object not in self.detected_classes.keys(): 
                self.detected_classes[class_of_object] = self.num_classes
                self.num_classes += 1

            self.label_dict[self.label_id]['classes'].append(class_of_object)

        self.label_id += 1

    def save_labels(self, save_path='./via_regions.json'):
        with open(save_path, 'w') as f:
            json.dump(self.label_dict, f, indent=4)
        
    def save_class_map(self, save_path='./class_map_detected.json'):
        with open(save_path, 'w') as f: 
            json.dump(self.detected_classes, f, indent=4)


if __name__ == '__main__':
    main()