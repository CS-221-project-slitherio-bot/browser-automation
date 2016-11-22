import random
from threading import Lock, Thread

from collections import Iterable
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
COLLECTION_TIMEOUT = 60 * 60 * 2

COLLUSION_COUNT = 10
FAR_R = 10000000
FAR_P = -1
FAR_SNAKE = -1

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
    POOLING_INTERVAL = 1
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
        self.just_dead = False

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
        self.log_print(str(result))
        self.debug_print(str(len(result)) + "fps")
        self.process(result)

    def change_parameter(self, parameter):
        self.driver.execute_script("bot.opt.radiusMult = " + str(parameter))

    def process(self, result):
        if self.just_dead:
            self.just_dead = False
            return
        json_result = [json.loads(message) for message in result]
        dead = False
        for message in json_result:
            if "type" in message and message["type"] == "result":
                final_length = message["content"]["length"]
                self.debug_print("Game end, final length: " + str(final_length))
                dead = True
                self.just_dead = True
            else:
                last_message_obj = message
        # self.debug_print(str(message_obj))
        if last_message_obj and "type" in last_message_obj and last_message_obj["type"] == "status":
            content = last_message_obj["content"]
            length = content["length"]
            collusion_xy = content["collusion"]
            snake_xy = content["snake"]
            def rp(a, b):
                dx = a["xx"] - b["xx"]
                dy = a["yy"] - b["yy"]
                return (dx**2 + dy**2, math.atan2(dy, dx))
            collusion_rp = [(xy["snake"], rp(xy, snake_xy)) for xy in collusion_xy]
            sorted_collusion = sorted(collusion_rp + [(FAR_SNAKE, (FAR_R, FAR_P))] * COLLUSION_COUNT,
                                      key = lambda x: x[1][0])
            first_n_collusion = sorted_collusion[0: COLLUSION_COUNT]
            #TODO: retag the snake number of first_n_collusion
            feature_vector = (first_n_collusion, length)
            # self.debug_print(str(feature_vector))
            flatten_feature = list(self.flatten(feature_vector))
            action = self.predict(flatten_feature)
            self.change_parameter(action)
            if self.last_status != None:
                last_feature, last_action, last_length = self.last_status
                if not dead:
                    last_reward = length - last_length
                else:
                    last_reward = - 10 * final_length
                self.feedback(last_feature, last_action, last_reward)
            self.last_status = (flatten_feature, action, length)

    def flatten(self, t):
        for t1 in t:
            if isinstance(t1, Iterable):
                yield from self.flatten(t1)
            else:
                yield t1

    def predict(self, feature):
        return self.predictor.action(feature)

    def feedback(self, feature, action, reward):
        self.predictor.feedback(feature, action, reward)


class Learning(object):
    ACTION = [5, 10, 20, 30, 40, 60]
    DISCOUNT = 0.999
    EXPLORATION_PROB = 0.2
    BATCH_COUNT = 50

    @staticmethod
    def _create_predictor():
        return MLPRegressor(solver="adam", hidden_layer_sizes=(15, 8, 3))

    def __init__(self, explore = True, predictor_file = None, scaler_file = None, load = False):
        if not explore:
            self.EXPLORATION_PROB = 0
        self.lock = Lock()
        self.sample = []
        self.trained = False
        self.predictor_file = predictor_file
        self.scaler_file = scaler_file
        if load:
            self.predictor = joblib.load(predictor_file)
            self.scaler = joblib.load(scaler_file)
            self.trained = True
        else:
            self.predictor = self._create_predictor()
            self.scaler = StandardScaler()
            self.trained = False

    def q(self, state, action):
        X = state + [action]
        X = np.array(X).reshape(1, -1)
        return self.predictor.predict(X)

    def action(self, state):
        if random.random() < self.EXPLORATION_PROB or self.trained == False:
            return random.choice(self.ACTION)
        else:
            return max((self.q(state, action), action) for action in self.ACTION)[1]

    def feedback(self, state, action, reward):
        self.sample += [(state + [action], reward)]
        if len(self.sample) > self.BATCH_COUNT:
            training_sample = self.sample
            self.sample = []
            Thread(target=self.train, name="training thread", kwargs={"training_sample": training_sample}).start()

    def train(self, training_sample):
        print("start training!")
        temp_predictor = sklearn.clone(self.predictor)
        X = [sample[0] for sample in training_sample]
        if not self.trained:
            self.scaler.fit(X)
            joblib.dump(self.scaler, self.scaler_file)
        X = self.scaler.transform(X)
        Y = [sample[1] for sample in training_sample]
        temp_predictor.fit(X, Y)
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
bot_predictor = Learning(explore=True, predictor_file="predictor.model", scaler_file="scaler.model", load=True)

with WithList([Bot(bot_scheduler, bot_predictor, None, sys.stdout, "Bot " + str(i)) for i in range(8)]) as bots:
    for bot in bots:
        bot.run()
    start_time = time()
    while time() - start_time < COLLECTION_TIMEOUT:
        bot_scheduler.run(blocking=False)
    for bot in bots:
        bot.stop()
