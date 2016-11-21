from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from time import sleep, time
from sched import scheduler
import sys

MAX_ENTRY = 300
COLLECTION_TIMEOUT = 60 * 60 * 2

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

    def __init__(self, scheduler, log=None, debug=sys.stdout, name="Any Bot"):
        self.log = log
        self.debug = debug
        self.scheduler = scheduler
        self.is_running = False
        self.name = name

        self.start_time = 0
        self.event = None

    def debug_print(self, string):
        if self.debug:
            self.debug.write("DEBUG[" + self.name + "]: " + string + "\n")

    def log_print(self, string):
        if self.log:
            self.log.write(string + "\n")

    def __enter__(self):
        self.driver = open_driver(platform)
        self.driver.get('http://www.slither.io')

        with open("bot.user.js", "r") as scriptFile:
            script = scriptFile.read()
        self.driver.execute_script(script)
        self.debug_print("bot ready")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()
        self.debug_print("bot destroyed")

    def run(self):
        if not self.is_running:
            self.is_running = True
            self.start_time = time()
            self.schedule_next()
            self.debug_print("bot running")
        else:
            self.debug_print("already running!")

    def stop(self):
        if self.is_running:
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

    def process(self, result):
        pass

class WithList(list):
    def __enter__(self):
        return [item.__enter__() for item in self]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for item in self:
            item.__exit__(exc_type, exc_val, exc_tb)

bot_scheduler = scheduler(time, sleep)

with WithList([Bot(bot_scheduler, None, sys.stdout, "Bot " + str(i)) for i in range(8)]) as bots:
    for bot in bots:
        bot.run()
    start_time = time()
    while time() - start_time < 20:
        bot_scheduler.run(blocking=False)
    for bot in bots:
        bot.stop()
