from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from time import sleep
import threading

MAX_ENTRY = 300
COLLECTION_TIMEOUT = 60*60*2

supported_platform = ["chrome", "phantomjs"]

platform = 1

def play_slither_io(log_name, extra_script):
	if log_name != "":
		log_file = open(log_name, "w")
		def log(obj):
			log_file.write(str(obj) + "\n")
	else:
		def log(str):
			print(str)

	print("running: " + log_name + " with \"" + extra_script + "\"")

	run(log, extra_script)

	print("ending: " + log_name + " with \"" + extra_script + "\"")

	if log_name != "":
		log_file.close()

def run(log, extra_script):
	if platform == 0:
		capabilities = DesiredCapabilities.CHROME
		capabilities['loggingPrefs'] = { 'browser':'ALL' }

		driver = webdriver.Chrome(desired_capabilities=capabilities)

	elif platform == 1:
		capabilities = DesiredCapabilities.PHANTOMJS
		capabilities['loggingPrefs'] = { 'browser':'ALL' }

		driver = webdriver.PhantomJS(desired_capabilities=capabilities)

	driver.get('http://www.slither.io')

	with open("bot.user.js", "r") as scriptFile:
		script = scriptFile.read()

	driver.execute_script(script)
	driver.execute_script(extra_script)

	entry_count = 0
	refresh_times = 0
	# print console log messages
	while True:
		refresh_times += 1
		if platform != 1:
			for i, entry in enumerate(driver.get_log('browser')):
				if i % 10 == 0 or "Game" in entry['message']:
					entry_count += len(entry)
					log(entry)

		if entry_count > MAX_ENTRY:
			log("FINISHED" + str(refresh_times))
			break;

		if refresh_times > COLLECTION_TIMEOUT:
			log("TIMEOUT")
			break;
		sleep(1)

	if platform == 1:
		for i, entry in enumerate(driver.get_log('browser')):
			if i % 10 == 0 or "Game" in entry['message']:
				log(entry)

	driver.close()

workers = []

for multi in range(10, 30, 2):
	extra_script = "window.bot.opt.radiusMult = " + str(multi)
	for x in range(0,10):
		log_name = "play_m" + str(multi) + "." + str(x) + ".log"
		t = threading.Thread(target=play_slither_io, args=(log_name, extra_script))
		t.daemon = True
		t.start()
		workers += [t]

for t in workers:
	t.join()
