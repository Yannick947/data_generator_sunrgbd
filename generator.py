import os 
import json
import time 

from PIL import Image

# Make sure that las character is slash ('/')
ROOT_DIR_SUNRGBD = 'C:/Users/Yannick/Downloads/SUNRGBD/'

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


class Sun_To_MASKRCNN(object):

    def __init__(self,
                 path_to_class_map=None,
                 known_classes_only=False,
                 include_image_size=True):
        
        if known_classes_only is True:

            assert path_to_class_map is not None
            self.class_map = self.__get_class_map('class_map.json')
            self.invalid_class_images = 0
            self.not_recognized_classes = dict()

        self.known_classes_only = known_classes_only
        self.label_dict = {'Generated on': time.time(), 'labels':list()}
        self.label_id = 0
        self.detected_classes = dict()
        self.num_classes = 0
        self.include_image_size = include_image_size
        self.num_images_parsed = 0

    def __get_class_map(self, path_to_class_map):
        with open (path_to_class_map) as f:
            return json.load(f)

    def add_image_size(self, single_label, path_to_image, image_name):

        with Image.open(path_to_image) as img:
            width, height = img.size

        self.label_dict['labels'][-1]['image_width'] = width
        self.label_dict['labels'][-1]['image_height'] = height

    def process_label(self, *kwargs):

        self.label_dict['labels'].append(dict())

        try:
            self.add_label(*kwargs)

            if self.include_image_size: 
                self.add_image_size(*kwargs)

            self.label_id += 1

        except Exception as e:
            
            # if anything went wrong, delete this label
            del self.label_dict['labels'][-1]

            raise Exception(e) 
        
        finally: 
            self.num_images_parsed += 1


    def add_label(self, single_label, path_to_image, image_name):
        ''' Specific transformation for 2D instance segmentation for MASK_RCNN of matterport'''

        frames = single_label['frames'][0]['polygon']
        classes = single_label['objects']

        self.label_dict['labels'][-1] = {'path_to_image': path_to_image[len(ROOT_DIR_SUNRGBD):], 
                                         'image_name': image_name, 
                                         'regions':list(),
                                         'classes':list(),
                                         'id':self.label_id}

        for frame in frames:
            class_of_object = classes[frame['object']]['name'].lower().strip()
            if class_of_object == 'wall1' or class_of_object == 'Wall1':
                print(frame)

            if self.known_classes_only and class_of_object not in self.class_map.keys():
                # Invalid class detected, don't continue with this image
                self.invalid_class_images += 1
                
                if class_of_object not in self.not_recognized_classes.keys():
                    self.not_recognized_classes[class_of_object] = 0
                else: 
                    self.not_recognized_classes[class_of_object] += 1

                continue

            self.label_dict['labels'][-1]['regions'].append({"name": "polygon", 
                                                             "all_points_x":frame['x'], 
                                                             "all_points_y":frame['y']})

            if not bool(self.detected_classes.keys()) or class_of_object not in self.detected_classes.keys(): 
                self.detected_classes[class_of_object] = self.num_classes
                self.num_classes += 1

            self.label_dict['labels'][-1]['classes'].append(class_of_object)

        if len(self.label_dict['labels'][-1]['classes']) == 0:
            raise ValueError('Found image with only unkown classes, skip this one.')

    def save_labels(self, save_path='./via_regions.json'):

        self.print_stats()
        with open(save_path, 'w') as f:
            json.dump(self.label_dict, f, indent=4)
        
    def save_class_map(self, save_path='./class_map_detected.json'):
        with open(save_path, 'w') as f: 
            json.dump(self.detected_classes, f, indent=4)

    def print_stats(self):
        print(f'Parsed {self.num_images_parsed} images in total.')
        print(f'{self.label_id} images succesfully parsed.')

        if self.known_classes_only is True:
            print(f'Detected {self.invalid_class_images} images with invalid labels.')
        else: 
            print(f'During parsing {len(self.detected_classes.keys())} classes were detected.')

        most_unkown_classes = dict((k, v) for k, v in self.not_recognized_classes.items() if v >= 20)

        print('Dict with most unknown classes: ', most_unkown_classes)
if __name__ == '__main__':
    main()