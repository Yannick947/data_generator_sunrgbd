import os
import time 
import json 

from PIL import Image

# Manually found mismatches between mapped classes
IGNORE_CLASSES = ['patio', 'headboard', 'kitchen', 'floor', 'toilet', 'tub', 'cottage', 'shower', 'bathroom',
                  'house', 'bath', 'room', 'pants', 'jeans', "outfit", 'furniture', 'pillow', 'bathtub', 'fireplace',
                  'Curtins']


class Sun_To_MASKRCNN(object):

    def __init__(self,
                 root_sunrgbd,
                 path_to_class_map=None,
                 known_classes_only=False,
                 include_image_size=True):
        """ Class to transform an annotation file from sunrgbd style into mask_rcnn format. Applies some custom transformtaions
        Arguments: 
            :param root_sunrgbd: Path to root of sunrgbd dataset
            :param path_to_class_map: Path to the previously extracted class map with the dimension_reduction utilities
            :param known_classes_only: If classes which were previously matched only shall be considered or all classes
            :param include_image_size: If image size shall be stored in the annotations file
        """
        if known_classes_only is True:

            assert path_to_class_map is not None, 'Need a path to class map if known_classes_only is True.'
            self.class_map = self.__get_class_map(path_to_class_map)
            self.invalid_class_images = 0
            self.unknown_classes = dict()

        self.known_classes_only = known_classes_only
        self.label_dict = {'Generated on': time.time(), 'labels':dict()}
        self.label_id = 0
        self.detected_classes = dict()
        self.num_classes = 0
        self.include_image_size = include_image_size
        self.num_images_parsed = 0
        self.root_sunrgbd = root_sunrgbd
        self.ignore_map = self.get_ignore_map()

    def __get_class_map(self, path_to_class_map):
        with open (path_to_class_map) as f:
            return json.load(f)

    def add_image_size(self, single_label, path_to_image, image_name):

        with Image.open(path_to_image) as img:
            width, height = img.size

        self.label_dict['labels'][self.label_id]['image_width'] = width
        self.label_dict['labels'][self.label_id]['image_height'] = height

    def process_label(self, *kwargs):

        
        self.label_dict['labels'][self.label_id] = dict()

        try:
            self.add_label(*kwargs)

            if self.include_image_size: 
                self.add_image_size(*kwargs)

            self.label_id += 1

        except Exception as e:
            
            # if anything went wrong, delete this label
            self.label_dict['labels'].pop(self.label_id, None)
            raise Exception(e) 
        
        finally: 
            self.num_images_parsed += 1


    def add_label(self, single_label, path_to_image, image_name):
        ''' Specific transformation for 2D instance segmentation for MASK_RCNN of matterport'''

        frames = single_label['frames'][0]['polygon']
        classes = single_label['objects']

        path_to_image_generic = path_to_image[len(self.root_sunrgbd):].replace('\\', '/')

        self.label_dict['labels'][self.label_id] = {'path_to_image': path_to_image_generic, 
                                                    'image_name': image_name, 
                                                    'regions':list(),
                                                    'classes':list(),
                                                    'id':self.label_id}

        for frame in frames:
            class_of_object = classes[frame['object']]['name']

            if not self.valid_frame(class_of_object, frame):
                continue

            self.label_dict['labels'][self.label_id]['regions'].append({"name": "polygon", 
                                                                        "all_points_x":frame['x'], 
                                                                        "all_points_y":frame['y']})

            if not bool(self.detected_classes.keys()) or class_of_object not in self.detected_classes.keys(): 
                self.detected_classes[class_of_object] = self.num_classes
                self.num_classes += 1

            if self.known_classes_only is True: 
                self.label_dict['labels'][self.label_id]['classes'].append(self.class_map[class_of_object])
            else:
                self.label_dict['labels'][self.label_id]['classes'].append(class_of_object)

        if len(self.label_dict['labels'][self.label_id]['classes']) == 0:
            raise ValueError('Found image with only unknown classes, skip this one.')
    
    def valid_frame(self, class_of_object, frame):
        if self.known_classes_only and\
           (self.class_map.get(class_of_object) is None or\
           self.class_map.get(class_of_object) == 'unknown') or\
           self.ignore_map.get(class_of_object) is True:
           # Invalid class detected, don't continue with this instance
            self.invalid_class_images += 1
                
            if class_of_object not in self.unknown_classes.keys():
                self.unknown_classes[class_of_object] = 0
            else: 
                self.unknown_classes[class_of_object] += 1

            return False
            
        if len(frame['x']) < 3 or (len(frame['x']) != len(frame['y'])):
            #invalid polygon encountered, continue
            return False
        
        return True

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
            print(f'Detected {self.invalid_class_images} instances with invalid labels.')
        else: 
            print(f'During parsing {len(self.detected_classes.keys())} classes were detected.')

        most_unkown_classes = dict((k, v) for k, v in self.unknown_classes.items() if v >= 20)

        print('Dict with most unknown classes: ', most_unkown_classes)

    def get_ignore_map(self):
        ignore_map = dict()
        try: 
            with open(os.path.join(self.root_sunrgbd, 'class_dimension_reduction', 'cleaned_classes.json'), 'r') as f:
                cleaned_classes = json.load(f)
        except FileNotFoundError:
            print('cleaned_classes.json not found, if you want to apply manually inserted IGNORE_CLASSES provide the file.')
            return ignore_map

        for key, value in cleaned_classes.items():
            if value in IGNORE_CLASSES:
                ignore_map[key] = True
            else: 
                ignore_map[key] = False

        return ignore_map
