from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from time import sleep

capabilities = DesiredCapabilities.CHROME
capabilities['loggingPrefs'] = { 'browser':'ALL' }

MAX_ENTRY = 300
COLLECTION_TIMEOUT = 60

driver = webdriver.Chrome(desired_capabilities=capabilities)

# capabilities = DesiredCapabilities.PHANTOMJS
# capabilities['loggingPrefs'] = { 'browser':'ALL' }

# driver = webdriver.PhantomJS(desired_capabilities=capabilities)

driver.get('http://www.slither.io')

with open("bot.user.js", "r") as scriptFile:
	script = scriptFile.read()

driver.execute_script(script)

entry_count = 0
refresh_times = 0
log = []
# print console log messages
while True:
	refresh_times += 1
	for i, entry in enumerate(driver.get_log('browser')):
		if i % 10 == 0 or "Game" in entry['message']:
			entry_count += len(entry)
			print(entry)

	if entry_count > MAX_ENTRY:
		print("FINISHED" + str(refresh_times))
		break;

	if refresh_times > COLLECTION_TIMEOUT:
		print("TIMEOUT")
		break;
	sleep(1)


driver.close()