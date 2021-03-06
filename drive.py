import argparse
import base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask, render_template
from keras.models import model_from_json
from PIL import Image


class Client(object):
    r'''Client for communication to the simulator.
    '''
    def __init__(self, args):
        r'''Create a new client.

            **Arguments:**

            `args`
                Initialization parameters. Must be a object with a string attribute `model` (a
                path to a JSON file) and a float attribute `throttle` (throttle value to be sent
                to the simulator).

            The `args` object must contain
        '''
        with open(args.model, 'r') as file:
            json = ''.join(file.readlines())
            self.model = model_from_json(json)

        weights_file = args.model.replace('json', 'h5')
        self.model.load_weights(weights_file)
        self.throttle = args.throttle

        self.sio = socketio.Server()
        self.sio.on('connect', self.connect)
        self.sio.on('telemetry', self.telemetry)

    def start(self):
        r'''Run the client.
        '''
        # Wrap Flask application with engineio's middleware
        app = socketio.Middleware(self.sio, Flask(__name__))

        # deploy as an eventlet WSGI server
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    def send_control(self, steering_angle, throttle):
        r'''Send a steering command to the simulator.
        '''
        command = {
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }

        self.sio.emit("steer", data=command, skip_sid=True)

    def connect(self, sid, environ):
        r'''Connect to the simulator.
        '''
        print("connect ", sid)
        self.send_control(0, 0)

    def telemetry(self, sid, data):
        r'''Event handler to receive data from the simulator.
        '''
        # The current steering angle of the car
        # steering_angle = data["steering_angle"]

        # The current throttle of the car
        # throttle = data["throttle"]

        # The current speed of the car
        # speed = data["speed"]

        # The current image from the center camera of the car
        image = data["image"]
        image = Image.open(BytesIO(base64.b64decode(image)))
        image = np.asarray(image)
        image = image[None, :, :, :]

        steering_angle = self.model.predict(image, batch_size=1).flat[0]
        self.send_control(steering_angle, self.throttle)
        print(steering_angle, self.throttle)


def main():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--throttle', type=float, default=0.2, help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()

    client = Client(args)
    client.start()


if __name__ == '__main__':
    main()
