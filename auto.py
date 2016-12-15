import random
from threading import Lock, Thread

from collections import Iterable
from collections import defaultdict

from keras.layers import Convolution2D
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from time import sleep, time
from sched import scheduler
import sys
import json
import math
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, Adam
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras import backend as K
from copy import copy

MAX_ENTRY = 300
COLLECTION_TIMEOUT = 60 * 60 * 10

COLLUSION_COUNT = 10
FAR_R = 10000000
FAR_P = -1
FAR_SNAKE = -1
BOT_COUNT = 8

DIMENSION = 16
CROSS_DIMENSION = 8

BASE = 2
BASE_MAX_POWER = 14
BASE_MIN_POWER = 9

WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400

RESIZE_HEIGHT = 100
RESIZE_WIDTH = 100

CHROME_BAR_HEIGHT = 74

INPUT_SIZE = DIMENSION + DIMENSION + (CROSS_DIMENSION - 1) * CROSS_DIMENSION / 2 + DIMENSION * (BASE_MAX_POWER - BASE_MIN_POWER) + 1

DEBUG_FEATURE = False

supported_platform = ["chrome", "phantomjs"]

platform = 1


def open_driver(platform_id):
    if platform_id == 0:
        capabilities = DesiredCapabilities.CHROME
        capabilities['loggingPrefs'] = {'browser': 'ALL'}

        driver = webdriver.Chrome(desired_capabilities=capabilities)

    elif platform_id == 1:
        capabilities = DesiredCapabilities.PHANTOMJS
        capabilities['loggingPrefs'] = {'browser': 'ALL'}

        driver = webdriver.PhantomJS(desired_capabilities=capabilities)
    else:
        driver = None

    return driver


class Bot(object):
    """Slither.io Bot"""
    POOLING_INTERVAL = 0.5
    START_SCRIPT = "window.play_btn.btnf.click(); window.autoRespawn = true;"
    END_SCRIPT = "window.autoRespawn = false; window.userInterface.quit();"


    def __init__(self, scheduler, predictor, log=None, debug=sys.stdout, name="Any Bot"):
        self.log = log
        self.debug = debug
        self.scheduler = scheduler
        self.is_running = False
        self.name = name
        self.predictor = predictor

        self.start_time = 0
        self.event = None

        self.last_status = None
        self.just_dead = 0

        self.last_update = None

        self.history_frame = np.zeros(shape=(4, RESIZE_WIDTH, RESIZE_WIDTH))

        self.direction = 0

    def debug_print(self, string):
        if self.debug:
            self.debug.write("DEBUG[" + self.name + "]: " + string + "\n")

    def log_print(self, string):
        if self.log:
            self.log.write(string + "\n")

    def __enter__(self):
        self.driver = open_driver(platform)
        if platform == 0:
            self.driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT + CHROME_BAR_HEIGHT)
        else:
            self.driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.driver.get('http://www.slither.io')
        self.debug_print("webpage ready")

        with open("bot.user.js", "r") as scriptFile:
            script = scriptFile.read()
        self.driver.execute_script(script)
        self.debug_print("bot ready")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()
        self.debug_print("bot destroyed")

    def _start_game(self):
        self.driver.execute_script(self.START_SCRIPT)

    def _end_game(self):
        self.driver.execute_script(self.END_SCRIPT)

    def run(self):
        if not self.is_running:
            self._start_game()
            self.is_running = True
            self.start_time = time()
            self.schedule_next()
            self.debug_print("bot running")
        else:
            self.debug_print("already running!")

    def stop(self):
        if self.is_running:
            self._end_game()
            self.is_running = False
            self.debug_print("bot stopped")
        else:
            self.debug_print("already stopped!")

    def schedule_next(self):
        self.start_time += self.POOLING_INTERVAL
        # self.debug_print("time: " + str(self.start_time) + ", now: " + str(time()))
        self.event = self.scheduler.enterabs(self.start_time, 1, self.play)

    def play(self):
        if self.is_running:
            self.schedule_next()
        result = self.driver.execute_script("return window.get_last_in_queue(window.message_queue)")

        new_update = time()
        if self.last_update is not None:
            self.debug_print(str(len(result)) + " updates in " + str(new_update - self.last_update) + "s")
        self.last_update = new_update

        self.process(result)

    def change_parameter(self, parameter):
        angle_dimension, boost = parameter
        self.direction = angle_dimension
        angle = (2 * math.pi / DIMENSION) * angle_dimension - math.pi
        x = 50 * math.cos(angle) + random.uniform(0, 5)
        y = 50 * math.sin(angle) + random.uniform(0, 5)
        if_boost = 1 if boost else 0
        # self.debug_print("executing script: " + 'canvasUtil.setMouseCoordinates({"x": %f, "y": %f}); window.setAcceleration(%d);' % (x, y, if_boost))
        self.driver.execute_script(
            'canvasUtil.setMouseCoordinates({"x": %f, "y": %f}); window.setAcceleration(%d);' % (x, y, if_boost))

    def process(self, result):
        screen = self.driver.get_screenshot_as_png()
        image = Image.open(BytesIO(screen))
        image = image.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.LANCZOS)
        image.save("./screen.png")
        if self.just_dead != 0:
            self.just_dead -= 1
        json_result = [json.loads(message) for message in result]
        dead = False
        last_message_obj = None
        for message in json_result:
            if "type" in message and message["type"] == "result":
                final_length = message["content"]["length"]
                dead = True
                self.just_dead = 10
            else:
                last_message_obj = message
        # self.debug_print(str(message_obj))
        if last_message_obj is not None and "type" in last_message_obj and last_message_obj["type"] == "status":
            content = last_message_obj["content"]
            length = content["length"]

            image_feature = np.asarray(image)
            np.delete(self.history_frame, 0)
            np.append(self.history_frame, image_feature)
            image_history_feature = self.history_frame
            self.debug_print("image feature shape: " + str(image_history_feature.shape))
            predict_result = self.predict(image_history_feature)
            action = predict_result[1]
            self.debug_print("weight: %f, action: %s" % (predict_result[0], str(predict_result[1])))
            self.change_parameter(action)
            if (self.just_dead == 0 or dead) and self.last_status is not None:
                last_feature, last_action, last_length = self.last_status
                if not dead:
                    last_reward = length - last_length
                    self.feedback(last_feature, last_action, last_reward, image_history_feature)
                else:
                    last_reward = - 1000
                    self.feedback(last_feature, last_action, last_reward, None)
                    self.debug_print("Game end, final length: " + str(last_length))
                    self.last_status = None
                    self.history_frame = np.zeros(shape=(4, RESIZE_WIDTH, RESIZE_WIDTH))
                    self.direction = 0
            self.last_status = (image_history_feature, action, length)

    def flatten(self, t):
        for t1 in t:
            if isinstance(t1, Iterable):
                yield from self.flatten(t1)
            else:
                yield 1.0 * t1

    def predict(self, feature):
        return self.predictor.action(feature)

    def feedback(self, feature, action, reward, new_state):
        # self.log_print(json.dumps((feature, action, reward, new_state)))
        self.predictor.feedback(feature, action, reward, new_state)


class Learning(object):
    ACTION = [(i, boost) for boost in [False, True] for i in range(DIMENSION)]
    DISCOUNT = 0.98
    EXPLORATION_PROB = 0.05
    BATCH_COUNT = 100

    def _create_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(4, RESIZE_WIDTH, RESIZE_HEIGHT), dim_ordering="th"))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(len(self.ACTION)))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        return model

    def __init__(self, explore = True, model_file = None, load = False, learning_rate = 0.05, discount = 0.98):
        if not explore:
            self.EXPLORATION_PROB = 0
        self.lock = Lock()
        self.sample = []
        self.trained = False
        self.predictor_file = model_file
        self.learning_rate = learning_rate
        self.discount = discount
        self.model = None
        if load:
            self.models= [load_model(model_file) for i in range(2)]
            self.select_model(0)
            self.trained = True
        else:
            self.models = [self._create_model(), self._create_model()]
            self.select_model(0)
            self.trained = False
        self.sess = tf.Session()
        K.set_session(self.sess)

    def q_action(self, state):
        X = state
        X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
        return self.model.predict(X, batch_size=1)[0]

    @staticmethod
    def action_to_index(action):
        angle_dimension, boost = action
        index = angle_dimension + (DIMENSION if boost else 0)
        return index

    def action(self, state):
        if random.random() < self.EXPLORATION_PROB or self.trained == False:
            return (0, random.choice(self.ACTION))
        else:
            result = self.q_action(state)
            index = np.argmax(result)
            value = np.max(result)
            return value, self.ACTION[index]

    def feedback(self, state, action, reward, new_state):
        index = self.action_to_index(action)
        if self.trained:
            q_action = self.q_action(state)
            if new_state is None:
                newValue = 0
            else:
                newValue = np.max(self.q_action(new_state))
            oldQ = q_action[index]
            newQ = (1 - self.learning_rate) * oldQ + \
                   self.learning_rate * (reward + self.discount * newValue)
            q_action[index] = newQ
        else:
            newQ = reward
            q_action = [0.0] * len(self.ACTION)
            q_action[index] = newQ
        print(len(state))
        self.sample += [(state, q_action)]
        print("sample count: %d\n" % len(self.sample))
        if len(self.sample) > self.BATCH_COUNT:
            training_sample = self.sample
            self.sample = []
            Thread(target=self.train, name="training thread", kwargs={"training_sample": training_sample}).start()

    def train(self, training_sample):
        if self.lock.acquire(False):
            with self.sess.graph.as_default():
                try:
                    print("start training!")
                    if self.trained:
                        self.model.save(self.predictor_file)
                    training_model = self.models[1]
                    X = [sample[0] for sample in training_sample]
                    Y = [sample[1] for sample in training_sample]
                    X = np.array(X)
                    Y = np.array(Y)
                    print(X.shape)
                    print(Y)
                    history = training_model.fit(X, Y, batch_size=self.BATCH_COUNT, nb_epoch=100, verbose=0)
                    print(history)
                    self.model = self.models[1]
                    self.models[0].set_weights(self.models[1].get_weights())
                    self.model = self.models[0]
                    self.trained = True
                    print("training complete!")
                except:
                    raise
                finally:
                    self.lock.release()
        else:
            print("last training is not complete, abort.")

class WithList(list):
    def __enter__(self):
        return [item.__enter__() for item in self]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for item in self:
            item.__exit__(exc_type, exc_val, exc_tb)

bot_scheduler = scheduler(time, sleep)
bot_predictor = Learning(explore=True, model_file="predictor.model", load=False)

with WithList([open("./log/bot_"+ str(i) +".log", "w") for i in range(BOT_COUNT)]) as files:
    with WithList([Bot(bot_scheduler, bot_predictor, files[i], sys.stdout, "Bot " + str(i)) for i in range(BOT_COUNT)]) as bots:
        for bot in bots:
            bot.run()
        start_time = time()
        while time() - start_time < COLLECTION_TIMEOUT:
            bot_scheduler.run(blocking=True)
        for bot in bots:
            bot.stop()
