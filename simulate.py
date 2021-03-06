import cv2
import time
from detectors.retinaface_detector import RetinaFaceDetector
from detectors.mtcnn_detector import MTCNNDetector

from basemodels.facenet import FacenetFeatureExtractor
from basemodels.facenet512 import Facenet512FeatureExtractor
from basemodels.arc_face import ArcFaceFeatureExtractor


# DO NOT MODIFY
DETECTOR_BACKENDS_MAPPING = {
    'mtcnn': MTCNNDetector,
    'retinaface': RetinaFaceDetector
}

# DO NOT MODIFY
REPRESENT_MODELS_MAPPING = {
    'Facenet': FacenetFeatureExtractor,
    'Facenet512': Facenet512FeatureExtractor,
    'ArcFace': ArcFaceFeatureExtractor
}


DETECTOR_BACKEND = "retinaface"
REPRESENT_MODEL = "ArcFace"
FACE_RECOGNITION_THRESHOLD = 0.90


class FaceDetectAndRepresentProcessor(object):
    face_detector = None
    face_features_extractor = None

    @classmethod
    def load_model(cls):
        # load face detector
        if cls.face_detector is None:
            print('Loading face detector')
            face_detector = DETECTOR_BACKENDS_MAPPING.get(DETECTOR_BACKEND)
            cls.face_detector = face_detector()

        # load face embedding vector extractor
        if cls.face_features_extractor is None:
            print('Loading face representer')
            face_features_extractor = REPRESENT_MODELS_MAPPING.get(REPRESENT_MODEL)
            cls.face_features_extractor = face_features_extractor()

        return cls.face_detector, cls.face_features_extractor

    @classmethod
    def predict(cls, image_data):
        iter_times = 20

        t1 = time.time()
        for _ in range(iter_times):
            face_detector, face_features_extractor = cls.load_model()
            raw_image_height, raw_image_width, _ = image_data.shape

            # detect and align faces in input image
            detected_and_aligned_faces = face_detector.detect_face(image_data, align=True)

            face_meta_data = list()

            for face_obj in detected_and_aligned_faces:
                aligned_face_image = face_obj['detected_face']      # BGR
                [x, y, w, h] = face_obj['bounding_box']
                confidence = face_obj['confidence']
                landmarks = face_obj['landmarks']

                if confidence < FACE_RECOGNITION_THRESHOLD:
                    continue

                # face embedding feature vector extraction
                face_embedding_vectors = face_features_extractor.represent(aligned_face_image)
                face_meta_data.append(
                    {
                        "image_height": raw_image_height,
                        "image_width": raw_image_width,
                        "bounding_box": [int(x), int(y), int(w), int(h)],
                        "confidence": float(confidence),
                        "landmarks": {
                            'left_eye': [float(e) for e in landmarks["left_eye"]],
                            'right_eye': [float(e) for e in landmarks["right_eye"]],
                            'nose': [float(e) for e in landmarks["nose"]],
                            'mouth_left': [float(e) for e in landmarks["mouth_left"]],
                            'mouth_right': [float(e) for e in landmarks["mouth_right"]],
                        },
                        "representation": face_embedding_vectors
                    }
                )

        t2 = time.time()
        print('Time cost = {} seconds'.format((t2 - t1) / iter_times))

        return face_meta_data


if __name__ == '__main__':
    processor = FaceDetectAndRepresentProcessor()
    processor.load_model()

    src_image = cv2.imread('./test_images/test_1_source.jpg', cv2.IMREAD_COLOR)
    tgt_image = cv2.imread('./test_images/test_1_target.jpg', cv2.IMREAD_COLOR)

    source_faces = processor.predict(image_data=src_image)
    target_faces = processor.predict(image_data=tgt_image)

    # import json
    # a = json.dumps(source_faces)
    # b = json.dumps(target_faces)
    # print(a)
    # print(b)

    # print("source_faces = {}".format(source_faces))
    # print("target_faces = {}".format(target_faces))
    #
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from distance import *
    # import time
    #
    # src_face_vector = np.array(source_faces[0]['representation'])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(src_image[:, :, ::-1])
    # plt.axis('off')
    #
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(tgt_image[:, :, ::-1])
    # colors = ['blue', 'yellow', 'red']
    # for index, face in enumerate(target_faces):
    #     tgt_face_vector = np.array(face['representation'])
    #
    #     distance_cosine = find_cosine_distance(src_face_vector, tgt_face_vector)
    #     distance_euclidean = find_euclidean_distance(src_face_vector, tgt_face_vector)
    #     distance_euclidean_l2 = find_euclidean_distance(l2_normalize(src_face_vector), l2_normalize(tgt_face_vector))
    #
    #     print('distance_cosine = {}, Threshold = {}'.format(distance_cosine, find_threshold(REPRESENT_MODEL, "cosine")))
    #     print('distance_euclidean = {}, Threshold = {}'.format(distance_euclidean, find_threshold(REPRESENT_MODEL, "euclidean")))
    #     print('distance_euclidean_l2 = {}, Threshold = {}'.format(distance_euclidean_l2, find_threshold(REPRESENT_MODEL, "euclidean_l2")))
    #     print('\n')
    #
    #     [x_min, y_min, width, height] = face['bounding_box']
    #     rect = plt.Rectangle((x_min, y_min), width, height, fill=False, edgecolor=colors[index])
    #     ax.add_patch(rect)
    #
    # plt.axis('off')
    #
    # plt.show()
