import argparse
import base64
import logging, os
from datetime import datetime
import shutil
import csv

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import utils_train_self_driving_car as utils
from utils_train_self_driving_car import rmse
from variational_autoencoder import VariationalAutoencoder

sio = socketio.Server()
app = Flask(__name__)
model = None

prev_image_array = None
anomaly_detection = None
autoenconder_model = None
frame_id = 0


@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])

        # The current throttle of the car
        throttle = float(data["throttle"])

        # The current brake of the car
        brake = float(data["brake"])

        # The current speed of the car
        speed = float(data["speed"])

        # The current way point and lap
        wayPoint = int(data["currentWayPoint"])
        lapNumber = int(data["lapNumber"])

        # whether an OBE or crash occurred
        isCrash = int(data["crash"])

        # the total number of OBEs and crashes so far
        number_obe = int(data["tot_obes"])
        number_crashes = int(data["tot_crashes"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        image_path = ''
        if args.data_dir != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.data_dir, args.track_name, "IMG", timestamp)
            image_path = '{}.jpg'.format(image_filename)
            image.save(image_path)

        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image_copy = np.copy(image)
            image_copy = autoenconder_model.normalize_and_reshape(image_copy)
            loss = anomaly_detection.test_on_batch(image_copy, image_copy)

            image = utils.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit

            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED

            if loss > args.threshold * 1.1:
                confidence = -1
            elif loss > args.threshold and loss <= args.threshold * 1.1:
                confidence = 0
            else:
                confidence = 1

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            global frame_id

            send_control(steering_angle, throttle, brake, confidence, loss, args.max_laps)
            if args.data_dir:
                csv_path = os.path.join(args.data_dir, args.track_name)
                writeCsvLine(csv_path,
                             [frame_id, args.model, args.anomaly_detector_name, args.threshold, args.track_name,
                              lapNumber, wayPoint, loss, steering_angle, throttle, brake, speed, isCrash,
                              image_path, number_obe, number_crashes])

                frame_id = frame_id + 1

        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0, 1, 0, 1)


def send_control(steering_angle, throttle, brake, confidence, loss, max_laps):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'brake': brake.__str__(),
            'confidence': confidence.__str__(),
            'loss': loss.__str__(),
            'max_laps': max_laps.__str__()
        },
        skip_sid=True)


def load_autoencoder(model):
    autoencoder = model.create_autoencoder()
    autoencoder.load_weights(model.model_name)
    assert (autoencoder is not None)
    return autoencoder


def writeCsvLine(filename, row):
    if filename is not None:
        filename += "/driving_log.csv"
        with open(filename, mode='a') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(row)
            result_file.flush()
            result_file.close()
    else:
        create_csv_results_file_header(filename)


def create_csv_results_file_header(file_name):
    if file_name is not None:
        file_name += "/driving_log.csv"
        with open(file_name, mode='w', newline='') as result_file:
            csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            fieldnames = ["FrameId", "Self Driving Model", "Anomaly Detector", "Threshold", "Track Name", "Lap Number",
                          "Check Point", "Loss", "Steering Angle", "Throttle", "Brake", "Speed", "Crashed", "center",
                          "Tot OBEs", "Tot Crashes"]
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            result_file.flush()
            result_file.close()

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving - Data Collection')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str,
                        # default='preliminary-runs')
                        default="")
    parser.add_argument('-m', help='path to the model', dest='model', type=str,
                        default="../models/epoch-dataset5-304.h5")
    parser.add_argument('-ad', help='path to the anomaly detector model', dest='anomaly_detector_name', type=str,
                        default="../models/trained-anomaly-detectors/variational-autoencoder-model-dataset2-050.h5")  # DO NOT CHANGE THIS
    parser.add_argument('-threshold', help='threshold for the outlier detector', dest='threshold', type=float,
                        default=0.035)
    parser.add_argument('-t', help='track name', dest='track_name', type=str, default='trial')
    parser.add_argument('-s', help='speed', dest='speed', type=int, default=20)
    parser.add_argument('-max_laps', help='number of laps in a simulation', dest='max_laps', type=int, default=2)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    if "chauffeur" in args.model:
        model = load_model(args.model, custom_objects={"rmse": rmse})
    else:
        model = load_model(args.model)

    MAX_SPEED = args.speed
    MIN_SPEED = 10
    speed_limit = MAX_SPEED

    anomaly_detector_name = args.anomaly_detector_name.split("/")[-1].split(".h5")[0]
    autoenconder_model = VariationalAutoencoder(args.anomaly_detector_name)
    anomaly_detection = load_autoencoder(autoenconder_model)
    anomaly_detection.compile(optimizer='adam', loss='mean_squared_error')

    if args.data_dir != '':
        path = os.path.join(args.data_dir, args.track_name, "IMG")
        csv_path = os.path.join(args.data_dir, args.track_name)
        print("Creating image folder at {}".format(path))
        if not os.path.exists(path):
            os.makedirs(path)
            create_csv_results_file_header(csv_path)
        else:
            shutil.rmtree(csv_path)
            os.makedirs(path)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
