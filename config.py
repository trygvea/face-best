import os
import dlib
import cv2

intermediate_file = '.intermediate/faces.npy'

data_dir = os.path.expanduser('~/data')

# Globals
dlib_frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor_5 = dlib.shape_predictor(data_dir + '/dlib/shape_predictor_5_face_landmarks.dat')
shape_predictor_68 = dlib.shape_predictor(data_dir + '/dlib/shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')
face_classifier_opencv = cv2.CascadeClassifier(data_dir + '/opencv/haarcascade_frontalface_default.xml')


