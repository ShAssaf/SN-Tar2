import ast

import pandas as pd
from PIL import ImageDraw,Image


from vidToFrames import ALL_FRAMES_PATH, convert_ms_to_time_string, FACE_FEATURES
import face_recognition
import cv2
import os

facial_features = [
    'chin',
    'left_eyebrow',
    'right_eyebrow',
    'nose_bridge',
    'nose_tip',
    'left_eye',
    'right_eye',
    'top_lip',
    'bottom_lip']


def show_facial_features(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)

    # Find facial landmarks
    face_landmarks = face_recognition.face_landmarks(image)
    for land_mark in face_landmarks:
        for facial_feature in facial_features:
            if not os.path.exists(FACE_FEATURES + facial_feature):
                os.mkdir(FACE_FEATURES + facial_feature)
            # Get the coordinates of the right eye
            feature_cor = land_mark[facial_feature]


            # Get the coordinates of the bounding box of the right eye
            x_min = min(x[0] for x in feature_cor)
            y_min = min(x[1] for x in feature_cor)
            x_max = max(x[0] for x in feature_cor)
            y_max = max(x[1] for x in feature_cor)

            #img_obj = Image.fromarray(image)

            # draw lines upon facial features:
            # drawing = ImageDraw.Draw(img_obj)
            # drawing.rectangle((x_min,y_min,x_max,y_max), width=2, fill='blue')
            #
            # # show image
            # img_obj.show()

            # save facial features
            feature = image[y_min:y_max, x_min:x_max]
            feature_path = FACE_FEATURES + facial_feature
            cv2.imwrite(feature_path + '/' + str(len(os.listdir(feature_path))) + '.jpg', feature)


def extract_features(df):
    for idx, row in df['names'].items():
        if row != '':
            image_path = ALL_FRAMES_PATH + convert_ms_to_time_string(int(idx) * 1000) + '_.jpg'
            print(image_path)
            try:
                show_facial_features(image_path)
            except Exception as e:
                print(e)
if __name__ == '__main__':
    df = pd.read_csv('../csvs/data.csv').transpose()
    df.columns = df.iloc[0]
    df.drop(index='Unnamed: 0', axis=0, inplace=True)
    for idx, row in df['names'].items():
        df['names'][idx] = ' '.join(ast.literal_eval(row))
    extract_features(df)

    a = 1
