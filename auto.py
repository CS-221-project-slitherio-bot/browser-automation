import random
from threading import Lock, Thread

from collections import Iterable
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from time import sleep, time
from sched import scheduler
import sys
import json
import math
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np

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
    POOLING_INTERVAL = 1.0
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

    def debug_print(self, string):
        if self.debug:
            self.debug.write("DEBUG[" + self.name + "]: " + string + "\n")

    def log_print(self, string):
        if self.log:
            self.log.write(string + "\n")

    def __enter__(self):
        self.driver = open_driver(platform)
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
        angle = (2 * math.pi / DIMENSION) * angle_dimension - math.pi
        x = 100 * math.sin(angle)
        y = 100 * math.cos(angle)
        if_boost = 1 if boost else 0
        # self.debug_print("executing script: " + 'canvasUtil.setMouseCoordinates({"x": %f, "y": %f}); window.setAcceleration(%d);' % (x, y, if_boost))
        self.driver.execute_script(
            'canvasUtil.setMouseCoordinates({"x": %f, "y": %f}); window.setAcceleration(%d);' % (x, y, if_boost))

    def process(self, result):
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
            width = content["width"]
            collusion = content["collusion"]
            food = content["food"]

            # Normalize functions
            if DEBUG_FEATURE:
                def normalize_distance(d):
                    return d

                def normalize_food(sz):
                    return sz
            else:
                def normalize_distance(d):
                    return 1.0 / (math.log2(d + 1.0) + 1.0)

                def normalize_food(sz):
                    return math.log(sz + 1.0, 10) / 5

            collusion_null = collusion + [None] * (DIMENSION - len(collusion))
            collusion_nd = [
                (point["snake"], point["distance"]) if point is not None else (FAR_SNAKE, FAR_R)
                for point in collusion_null]

            collusion_n_set = [
                set(filter(
                    lambda x: x != FAR_SNAKE,
                    [collusion_nd[j][0] for j in
                     range(int(DIMENSION * i / CROSS_DIMENSION),
                           int(DIMENSION * (i + 1) / CROSS_DIMENSION))])
                )
                for i in range(CROSS_DIMENSION)]
            collusion_cross = [[len(collusion_n_set[i] & collusion_n_set[j]) != 0 for j in range(i)] for i in range(CROSS_DIMENSION)]

            collusion_d = [nd[1] for nd in collusion_nd]

            normalized_collusion_d = [normalize_distance(d) for d in collusion_d]

            collusion_feature = (normalized_collusion_d, collusion_cross)

            def which_angle(angle):
                unit_angle = 2 * math.pi / DIMENSION
                which = int(math.ceil((angle + math.pi) / unit_angle) - 1)
                if which < 0 or which >= DIMENSION:
                    self.debug_print("which_angle error: " + str(angle))
                return which

            food_rp = defaultdict(int)
            for food in food:
                angle = food["a"]
                distance = food["distance"] / width
                log_distance = int(math.floor(math.log(distance, BASE)))
                if log_distance < BASE_MIN_POWER:
                    normalized_log_distance = BASE_MIN_POWER
                elif log_distance >= BASE_MAX_POWER:
                    normalized_log_distance = BASE_MAX_POWER
                else:
                    normalized_log_distance = log_distance
                angle_dimension = which_angle(angle)
                food_rp[(normalized_log_distance, angle_dimension)] += food["sz"]

            food_feature = [[normalize_food(food_rp[(dist, angle)]) for dist in range(BASE_MIN_POWER, BASE_MAX_POWER)] for angle in range(DIMENSION)]

            # self.debug_print("collusion_feature: " + str(collusion_feature))
            #
            # self.debug_print("food_feature: " + str(food_feature))

            flatten_feature = list(self.flatten((collusion_feature, food_feature)))

            # self.debug_print("feature length: " + str(len(flatten_feature)))
            action = self.predict(flatten_feature)
            self.debug_print("weight: %f, action: %s" % (action[0], str(action[1])))
            self.change_parameter(action[1])
            if (self.just_dead == 0 or dead) and self.last_status is not None:
                last_feature, last_action, last_length = self.last_status
                if not dead:
                    last_reward = length - last_length
                    self.feedback(last_feature, last_action, last_reward, flatten_feature)
                else:
                    last_reward = - last_length
                    self.feedback(last_feature, last_action, last_reward, None)
                    self.debug_print("Game end, final length: " + str(last_length))
                    self.last_status = None
            self.last_status = (flatten_feature, action, length)

    def flatten(self, t):
        for t1 in t:
            if isinstance(t1, Iterable):
                yield from self.flatten(t1)
            else:
                yield 1.0 * t1

    def predict(self, feature):
        return self.predictor.action(feature)

    def feedback(self, feature, action, reward, new_state):
        self.log_print(json.dumps((feature, action, reward, new_state)))
        self.predictor.feedback(feature, action, reward, new_state)


class Learning(object):
    ACTION = [(i, boost) for i in range(DIMENSION) for boost in [True, False]]
    DISCOUNT = 0.98
    EXPLORATION_PROB = 0.0
    BATCH_COUNT = 50

    @staticmethod
    def _create_predictor():
        return MLPRegressor(solver="adam", hidden_layer_sizes=(80, 40, 10, 5))

    def __init__(self, explore = True, predictor_file = None, load = False, learning_rate = 0.3, discount = 0.90):
        if not explore:
            self.EXPLORATION_PROB = 0
        self.lock = Lock()
        self.sample = []
        self.trained = False
        self.predictor_file = predictor_file
        self.learning_rate = learning_rate
        self.discount = discount
        if load:
            self.predictor = joblib.load(predictor_file)
            self.trained = True
        else:
            self.predictor = self._create_predictor()
            self.trained = False

    def q(self, state, action):
        X = state + self.action_to_array(action)
        X = np.array(X).reshape(1, -1)
        return self.predictor.predict(X)[0]

    @staticmethod
    def action_to_array(action):
        feature = [0.0] * DIMENSION + [0.0]
        angle_dimension, boost = action
        feature[angle_dimension] = 1.0
        feature[-1] = 1.0 if boost else 0.0
        return feature

    def action(self, state):
        if random.random() < self.EXPLORATION_PROB or self.trained == False:
            return (0, random.choice(self.ACTION))
        else:
            return max((self.q(state, action), action) for action in self.ACTION)

    def feedback(self, state, action, reward, new_state):
        if self.trained:
            if new_state is None:
                newValue = 0
            else:
                newValue, _ = max((self.q(new_state, action), action) for action in self.ACTION)
            newQ = (1 - self.learning_rate) * self.q(state, action) + \
                   self.learning_rate * (reward + self.discount * newValue)
        else:
            newQ = reward
        print(len(state + self.action_to_array(action)))
        self.sample += [(state + self.action_to_array(action), newQ)]
        if len(self.sample) > self.BATCH_COUNT:
            training_sample = self.sample
            self.sample = []
            Thread(target=self.train, name="training thread", kwargs={"training_sample": training_sample}).start()

    def train(self, training_sample):
        print("start training!")
        temp_predictor = sklearn.clone(self.predictor)
        X = [sample[0] for sample in training_sample]
        Y = [sample[1] for sample in training_sample]
        X = np.array(X)
        Y = np.array(Y)
        print(Y)
        temp_predictor.partial_fit(X, Y)
        self.predictor = temp_predictor
        joblib.dump(self.predictor, self.predictor_file)
        self.trained = True
        print("training complete!")

class WithList(list):
    def __enter__(self):
        return [item.__enter__() for item in self]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for item in self:
            item.__exit__(exc_type, exc_val, exc_tb)

bot_scheduler = scheduler(time, sleep)
bot_predictor = Learning(explore=True, predictor_file="predictor.model", load=False)

with WithList([open("./log/bot_"+ str(i) +".log", "w") for i in range(BOT_COUNT)]) as files:
    with WithList([Bot(bot_scheduler, bot_predictor, files[i], sys.stdout, "Bot " + str(i)) for i in range(BOT_COUNT)]) as bots:
        for bot in bots:
            bot.run()
        start_time = time()
        while time() - start_time < COLLECTION_TIMEOUT:
            bot_scheduler.run(blocking=True)
        for bot in bots:
            bot.stop()
