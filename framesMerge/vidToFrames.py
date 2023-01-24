import os
import shutil
from collections import defaultdict, Counter
from functools import partial
import face_recognition
import cv2
import pandas as pd

"""
this program transforming a video into its frames in a rate of 23.978 FPS
"""

BASE_PATH = '/Users/shlomo/PycharmProjects/SocialNetworks/'
MOVIE_PATH = BASE_PATH + 'tar2/cher2.mp4'
SRT_SCRIPT_PATH = BASE_PATH + 'tar2/framesMerge/csvs/srt-script.csv'
OUTPUT_PATH = BASE_PATH + 'tar2/outputs/'
ALL_FRAMES_PATH = OUTPUT_PATH + 'all_frames/'
ALL_CSV_FRAMES_PATH = OUTPUT_PATH + 'all_csv_frames/'
DATABASE_PATH = OUTPUT_PATH + 'database/'
FACE_FEATURES = OUTPUT_PATH + 'face_features/'
FPS = 23.978
FPS_INTERVAL = 1000
DELAY_MS = ((1 / FPS) * 1000)
global tolernace


def covert_time_to_ms(hhmmss,delimiter=':'):
    """this function converts the time to milliseconds"""
    hh = int(hhmmss.split(delimiter)[0])
    mm = int(hhmmss.split(delimiter)[1])
    ss = float(hhmmss.split(delimiter)[2].split(',')[0])
    return (hh * 3600 + mm * 60 + ss) * 1000


def convert_ms_to_time_string(ms):
    """this function converts milliseconds to time string"""
    seconds = ms / 1000
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return str(int(hours)) + ':' + str(int(minutes)) + ':' + str(int(seconds))


def get_frame_from_video(vid_obj, time_in_ms, name='', base_path=ALL_FRAMES_PATH):
    """this function gets a frame from the video"""
    vid_obj.set(cv2.CAP_PROP_POS_MSEC, time_in_ms)
    ret, frame = vid_obj.read()
    if ret:
        cv2.imwrite(base_path + convert_ms_to_time_string(time_in_ms) + '_' + name + '.jpg', frame)

    return frame


def parse_srt_script():
    """this function parses the srt script and returns a list of lists"""
    df = pd.read_csv(SRT_SCRIPT_PATH)
    times = df['start'].tolist()
    names = df['speaker'].tolist()
    return names, times


def frame_list():
    frames = list(set(list(map(lambda x: x[:-5], os.listdir(OUTPUT_PATH + 'all_csv_frames')))))
    name_counts = Counter([frame.split('_')[1] for frame in frames])
    sorted_list = sorted(frames, key=lambda x: name_counts[x.split('_')[1]], reverse=True)
    return sorted_list


def get_name_from_frame(frame) -> str:
    """this function returns the name of the frame"""
    return frame.split('_')[1]


def copy_file(src, dst):
    """this function copies a file from src to dst"""
    shutil.copyfile(src, dst)


def sample_all_vid(rate=FPS_INTERVAL):
    vidcap = cv2.VideoCapture(MOVIE_PATH)
    c = 0
    while True:
        get_frame_from_video(vidcap, c * rate)
        c += 1


def generate_srt_frames():
    """ this function generates the frames from the video and saves them in the output folder"""
    vidcap = cv2.VideoCapture(MOVIE_PATH)
    names, times = parse_srt_script()
    for name, time_start in zip(names, times):
        time_start_ms = covert_time_to_ms(time_start)
        if time_start_ms > 2220029:
            print('over the 500 srt ', time_start)
            break
        get_frame_from_video(vidcap, time_start_ms + DELAY_MS * 10, name + '1')
        get_frame_from_video(vidcap, time_start_ms + DELAY_MS * 11, name + '2')
        get_frame_from_video(vidcap, time_start_ms + DELAY_MS * 12, name + '3')

    vidcap.release()


def get_face_size(face_location):
    """this function returns the size of the face"""
    top, right, bottom, left = face_location
    return (bottom - top) * (right - left)


def get_large_face_location(image, frame_name, pixel_threshold=65000):
    face_locations = face_recognition.face_locations(image)
    print("{l} faces identified in {file_name}.".format(l=len(face_locations), file_name=frame_name + ".jpg"))
    face_locations_c = [face for face in face_locations if get_face_size(face) > pixel_threshold]
    print("{} faces left after filtering small images".format(len(face_locations_c)))
    return face_locations_c


def frame_crop(frame_name, face_dict, enc_dict):
    images = [cv2.imread(ALL_CSV_FRAMES_PATH + frame_name + '{}.jpg'.format(i + 1)) for i in range(3)]
    name = get_name_from_frame(frame_name)
    for image in images:
        face_locations = get_large_face_location(image, frame_name)
        if face_locations:
            face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=25)
            for loc in face_locations:
                crop_img = image[loc[0]:loc[2], loc[3]:loc[1]]
                face_dict[name].append(crop_img)
                if not os.path.exists(DATABASE_PATH + name):
                    os.makedirs(DATABASE_PATH + name)
                validate_face_f = validate_face(crop_img, enc_dict, name, current_enc=face_encodings)
                if validate_face_f is False:
                    cv2.imwrite(DATABASE_PATH + 'unknown' + '/' + frame_name + '.jpg', crop_img)
        else:
            print("no faces larger than 25KB in: " + frame_name)


def check_if_other_characters(enc_dict, current_enc, folder_name):
    for name in enc_dict:
        if name != folder_name and name != 'default_factory':
            sim = similarity(enc_dict, current_enc, name, tol=0.45)
            if sim > 0.85:
                print('this is {}% {}'.format(sim * 100, name))
                return True
    return False


def similarity(enc_dict, current_enc, folder_name, tol=None):
    if tol is None:
        global tolerance
        tol = tolerance
    if len(enc_dict[folder_name]) == 0:
        return 0
    return sum(face_recognition.compare_faces(enc_dict[folder_name], current_enc, tolerance=tol)) / len(
        enc_dict[folder_name])


def update_tolerance():
    global tolerance
    if tolerance > 0.4:
        tolerance -= 0.03
    elif tolerance > 0.2:
        tolerance -= 0.001
    else:
        tolerance -= 0.0001


def validate_face(face, enc_dict, folder_name, current_enc, min_size=10):
    global tolerance
    path = DATABASE_PATH + folder_name
    if len(enc_dict[folder_name]) < min_size:
        if check_if_other_characters(enc_dict, current_enc[0], folder_name):
            return False
        tolerance = 0.6
        cv2.imwrite(path + '/' + str(len(enc_dict[folder_name]) + tolerance) + '.jpg', face)
        enc_dict[folder_name].append(current_enc[0])
        print('added to {}'.format(folder_name))
    else:
        for enc in current_enc:
            # check if the face is of another character if so don't add it
            if check_if_other_characters(enc_dict, enc, folder_name):
                return False

            results = similarity(enc_dict, enc, folder_name)
            if results > 0.5:
                enc_dict[folder_name].append(enc)
                cv2.imwrite(path + '/' + str(len(enc_dict[folder_name]) + tolerance) + '.jpg', face)
                update_tolerance()
                print('added to {}'.format(folder_name))
                break
            else:
                print("its not {}".format(folder_name))
                return False
    return True


def create_database():
    """this function creates a database of the frames"""

    face_dict = defaultdict(list)
    enc_dict = defaultdict(list)
    frames = frame_list()
    for idx, f in enumerate(frames):
        print("creating database is {}% completed".format(round((idx / len(frames)) * 100, 2)))
        frame_crop(f, face_dict, enc_dict)
    return face_dict


def find_closest_match(enc_dict, current_enc, folder_name):
    bast_name = ''
    best_sim = 0
    for name in enc_dict:
        if name != folder_name and name != 'default_factory':
            sim = similarity(enc_dict, current_enc, name, tol=0.6)
            if sim > best_sim:
                best_sim = sim
                bast_name = name
    if best_sim < 0.5:
        return 'unknown'
    return bast_name


def clean_database():
    """this function cleans the database by removing the images that are not of the character"""
    enc_dict = create_classifier(jitters=10)
    for folder in os.listdir(DATABASE_PATH):
        if folder != 'unknown' and folder != '.DS_Store':
            for file in os.listdir(DATABASE_PATH + folder):
                if file != '.DS_Store':
                    current_enc = face_recognition.face_encodings(cv2.imread(DATABASE_PATH + folder + '/' + file))
                    if not current_enc:
                        continue
                    if similarity(enc_dict, current_enc[0], folder, tol=0.4) < 0.3:
                        print('removed {} from {}'.format(file, folder))
                        copy_file(DATABASE_PATH + folder + '/' + file,
                                  DATABASE_PATH + 'unknown' + '/' + folder + '_' + file)
                        os.remove(DATABASE_PATH + folder + '/' + file)


def handle_unknown():
    """this function handles the unknown folder by moving the images to the correct folder"""
    enc_dict = create_classifier(jitters=10)
    for file in os.listdir(DATABASE_PATH + 'unknown'):
        if file != '.DS_Store':
            current_enc = face_recognition.face_encodings(cv2.imread(DATABASE_PATH + 'unknown' + '/' + file))
            if not current_enc:
                continue
            char = find_closest_match(enc_dict, current_enc[0], 'unknown')
            if char != '':
                print('moved {} to {}'.format(file, char))
                copy_file(DATABASE_PATH + 'unknown' + '/' + file, DATABASE_PATH + char + '/' + file)
                os.remove(DATABASE_PATH + 'unknown' + '/' + file)


def create_classifier(jitters=50):
    enc_dict = {}
    for folder in os.listdir(DATABASE_PATH):
        if folder != 'unknown' and folder != '.DS_Store':
            folder_encoding = []
            for file in os.listdir(DATABASE_PATH + folder):
                if file != '.DS_Store':
                    folder_encoding.extend(
                        face_recognition.face_encodings(cv2.imread(DATABASE_PATH + folder + '/' + file),
                                                        num_jitters=jitters))
            enc_dict[folder] = folder_encoding
    return enc_dict


def identifier(encodings, classifier):
    options = []
    for enc in encodings:
        options.append(find_closest_match(classifier, enc, 'unknown'))
    names = list(set(options))
    if 'unknown' in names:
        names.remove('unknown')
    return names


if __name__ == "__main__":
    sample_all_vid()
    generate_srt_frames()
    create_database()
    for i in range(3):
        clean_database()
        handle_unknown()
    classifier = create_classifier()
    face_count = []
    face_bits = []
    total_amount_of_elements = []
    names_in_frame = []
    sorted_frames = sorted(os.listdir(ALL_FRAMES_PATH), key=lambda x: covert_time_to_ms(x.split('_')[0]))
    for idx,frame in enumerate(sorted_frames):
        if frame != '.DS_Store':
            img = cv2.imread(ALL_FRAMES_PATH + frame)
            locations = face_recognition.face_locations(img)
            encodings = face_recognition.face_encodings(img, locations)
            names_in_frame.append(identifier(encodings, classifier))  # 1g
            face_count.append(len(locations))  # 2a
            face_bits.append(sum([get_face_size(face_loc) for face_loc in locations]))  # 2b
            total_amount_of_elements.append(face_recognition.face_landmarks(img, locations))
        if idx % 100 == 0:
            print("processing is {}% completed".format(round((idx / len(sorted_frames)) * 100, 2)))
            data = [names_in_frame, face_count, total_amount_of_elements,face_bits]
            columns = ['names', 'face_count', 'total_amount_of_elements', 'face_bits']
            df = pd.DataFrame(data, columns)
            df.to_csv('data.csv')


