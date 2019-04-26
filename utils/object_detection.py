'''
    TensorFlow Object Detection API
'''
import numpy as np
import cv2
import tensorflow as tf
import os
import collections

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class ObjectDetection:

    def __init__(self, 
                 model='rfcn_resnet101_aquarium_fish_v2_22751',
                 labels='aquarium_fish_v2_label_map.pbtxt',
                 num_classes=3):
        # Model preparation
        self.model = model
        self.labels = labels
        self.num_classes = num_classes
        self.model_name = 'object_detection/saved_models/{}'.format(self.model)
        self.path_to_ckpt = self.model_name + '/frozen_inference_graph.pb'
        self.path_to_labels = os.path.join('object_detection/data', self.labels)

        # Load a (frozen) TensorFlow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Loading label map
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Visualization
        self.min_score_thresh = .5
        self.line_thickness = 4

        # Point(x, y) in queue
        self.point_buff_size = 64
        self.leftCam_point_buff = collections.deque(maxlen=self.point_buff_size)
        self.rightCam_point_buff = collections.deque(maxlen=self.point_buff_size)

    def run(self, image_np, display=True):
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                 feed_dict={self.image_tensor: image_np_expanded})
        if display==True:
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               self.category_index,
                                                               use_normalized_coordinates=True,
                                                               min_score_thresh=self.min_score_thresh,
                                                               line_thickness=4)
        elif display==False:
            pass

        return boxes, scores, classes, self.category_index

    def data_processing(self, image_np, boxes, scores, classes, category_index, point_buff):
        get_scores = np.squeeze(scores)
        get_category = np.array([category_index.get(i) for i in classes[0]])
        get_boxes = np.squeeze(boxes)

        num_objects = 0
        list_scores = []
        list_category = np.array([])
        for i in range(len(get_scores)):
            if scores is None or get_scores[i] > self.min_score_thresh:
                num_objects = num_objects + 1
                list_scores = np.append(list_scores, get_scores[i])
                list_category = np.append(list_category, get_category[i])

        '''
        (x1,y1) ----
            |       |
            |       |
            |		|
            ---- (x2,y2)
        x_min = x1, y_min = y1, x_max = x2, y_max = y2
        '''
        point = None
        x_min, y_min, x_max, y_max = None, None, None, None
        height, width, _ = image_np.shape
        for i in range(len(list_scores)):
            # Get boxes[y_min, x_min, y_max, x_max]
            box = get_boxes[i]
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)

            coordinates_x = int((x_max + x_min) / 2)
            coordinates_y = int((y_max + y_min) / 2)
            point = (coordinates_x, coordinates_y)
            # cv2.circle(image_np, point, 2, (0, 255, 0), -1)
        point_buff.appendleft(point)

        # Loop over the set of tracked points
        for i in range(1, len(point_buff)):
            # If either of the tracked points are None, ignore
            # them
            if point_buff[i - 1] is None or point_buff[i] is None:
                continue
            # Otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(self.point_buff_size / float(i + 1)) * .5)
            # cv2.line(image_np, point_buff[i - 1], point_buff[i], (0, 255, 0), thickness)

        return list_category, point, x_min, y_min, x_max, y_max, num_objects

    def close(self):
        self.sess.close()
        self.default_graph.close()
