import logging
import os
import cv2
import pickle
import numpy as np
import torch
import math
import socket
from torch.utils.data import Dataset
from skimage import io

hostname = socket.gethostname()
# if hostname.startswith(''):
root_path = '/media/jd4615/dataB/Datasets/affectnet/'

class dataloader(Dataset):
    _expressions = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger',
                    7: 'contempt', 8: 'none'}
    _expressions_indices = {8: [0, 1, 2, 3, 4, 5, 6, 7],
                            5: [0, 1, 2, 3, 6]}

    def __init__(self, subset='train',
                 transform_image_shape=None, transform_image=None,
                 n_expression=5, verbose=1, cleaned_set=True):
        self.image_path = torch.load(root_path+'index2img.pkl')
        self.root_path = root_path
        self.subset = subset
        if self.subset=='train':
            type_value = 1
        elif self.subset=='valid':
            type_value = 3
        elif self.subset=='test':
            type_value = 4
        else:
            raise ValueError(f'subset name should be train,valid,test, but got subset={self.subset}')
            
        self.transform_image_shape = transform_image_shape
        self.transform_image = transform_image
        self.verbose = verbose
        self.cleaned_set = cleaned_set
        if n_expression not in [5, 8]:
            raise ValueError(f'n_expression should be either 5 or 8, but got n_expression={n_expression}')
        self.n_expression = n_expression

        self.pickle_path = self.root_path+'/AffectNet.ctx.pkl'
        with open(self.pickle_path, 'br') as f:
            data = pickle.load(f)
        self.data = data

        # the keys are the image names (name.ext)
        self.keys = []
        self.skipped = {'other': [],'expression':[],'det':[],'annot':[],'missed':[]}
        # List of each expression to generate weights
        expressions = []
        for key, value in data.items():
            # if key == 'folder':
            #     continue
            if 'Manually' not in value['path']:
                continue

            image_file = self.root_path + '/' + self.image_path[key]
            if not os.path.isfile(image_file):
                self.skipped['missed'].append(key)
                continue

            if 'det' not in value.keys():
                self.skipped['det'].append(key)
                continue

            if 'bbox' not in value['det'].keys():
                self.skipped['det'].append(key)
                continue

            if 'annot' not in value.keys():
                self.skipped['annot'].append(key)
                continue

            if (int(value['annot']['expression']) not in self._expressions_indices[self.n_expression]):
                self.skipped['expression'].append(key)
                continue

            if not (int(value['type'])==type_value):
                # self.skipped['type'].append(key)
                continue



            expression = int(value['annot']['expression'])
            if self.cleaned_set:
                # Automatic cleaning : expression has to match the valence and arousal values
                valence = float(value['annot']['valence'])
                arousal = float(value['annot']['arousal'])
                
                intensity = math.sqrt(valence ** 2 + arousal ** 2)

                if expression == 0 and intensity >= 0.2:
                    self.skipped['other'].append(key)
                    continue
                elif expression == 1 and (valence <= 0 or intensity <= 0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 2 and (valence >= 0 or intensity <= 0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 3 and (arousal <= 0 or intensity <= 0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 4 and (not (arousal >= 0 and valence <= 0) or intensity <= 0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 5 and (valence >= 0 or intensity <= 0.3):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 6 and (arousal <= 0 or intensity <= 0.2):
                    self.skipped['other'].append(key)
                    continue
                elif expression == 7 and (valence >= 0 or intensity <= 0.2):
                    self.skipped['other'].append(key)
                    continue
                if self.n_expression == 5 and expression == 6:
                    expression = 4

            expressions.append(expression)
            self.keys.append(key)

        expressions = np.array(expressions)
        self.expressions = expressions
        self.sample_per_class = {label: np.sum(expressions == label) for label in np.unique(expressions)}
        self.expression_weights = np.array([1. / self.sample_per_class[e] for e in expressions])
        self.average_per_class = int(np.mean(list(self.sample_per_class.values())))
        self.weight = np.array([1.0/self.sample_per_class[ii] for ii in range(self.n_expression)])
        self.weight = torch.from_numpy(self.weight).float().squeeze()

        if self.verbose:
            skipped = sum([len(self.skipped[key]) for key in self.skipped])
            msg = f' --  {len(self.keys)} images, ' \
                  f'missed {len(self.skipped["missed"])} images, ' \
                  f'det {len(self.skipped["det"])} images, ' \
                  f'annot {len(self.skipped["annot"])} images, ' \
                  f'exp {len(self.skipped["expression"])} images,' \
                  f'other {len(self.skipped["other"])} with intensity),' \
                  f'Samples per class : {self.sample_per_class}'
            print(msg)
            logging.info(msg)


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # weight = self.expression_weights[index]
        key = self.keys[index]
        sample_data = self.data[key]
        image_file = self.root_path+'/'+self.image_path[key]
        valence = torch.tensor([float(sample_data['annot']['valence'])], dtype=torch.float32)
        arousal = torch.tensor([float(sample_data['annot']['arousal'])], dtype=torch.float32)
        expression = int(sample_data['annot']['expression'])

        if self.n_expression == 5 and expression == 6:
            expression = 4

        bounding_box = sample_data['det']['bbox'][0:4]

        if isinstance(bounding_box, list):
            bounding_box = np.array(bounding_box)

        # image = cv2.imread(image_file)
        # image = cv2.rectangle(image, (int(bounding_box[:2][0]),int(bounding_box[:2][1])), (int(bounding_box[2:][0]),int(bounding_box[2:][1])),(255, 0, 0),2)
        # cv2.imwrite('%s.jpg'%key,image)
        image = io.imread(image_file)
        if self.transform_image_shape is not None:
            # bounding_box = [landmarks.min(axis=0)[0], landmarks.min(axis=0)[1],
            #                 landmarks.max(axis=0)[0], landmarks.max(axis=0)[1]]
            # image, landmarks = self.transform_image_shape(image, shape=landmarks)
            image, landmarks = self.transform_image_shape(image, bb=bounding_box)
            # Fix for PyTorch currently not supporting negative stric
            image = np.ascontiguousarray(image)
        # cv2.imwrite('%s_2.jpg' % key, image)
        if self.transform_image is not None:
            image = self.transform_image(image)

        return dict(valence=valence, arousal=arousal, expression=expression, image=image,au=[])

