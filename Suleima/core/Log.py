import os
from datetime import datetime

class Log:
	def __init__(self, caseID, also_print=True):
		self.caseID = caseID
		self.logPath = os.path.join("data", "cases", caseID, "info.txt")
		os.makedirs(os.path.dirname(self.logPath), exist_ok=True)
		self.logs = []

		self.also_print = also_print

	def log(self, msg, do_print=True):
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"[{timestamp}] {msg}"
		self.logs.append(entry)
		with open(self.logPath, "a", encoding="utf-8") as f:
			f.write(entry + "\n")


		# Final decision to print
		#if do_print:
		if self.also_print is True:
			print(entry)

# Global variable to hold the active logger
global_log = None

def set_log(caseID, also_print):
	"""Initialise the global log instance."""
	global global_log
	global_log = Log(caseID, also_print=also_print)

def log(msg, do_print=True):
	"""Log a message using the global logger."""
	if global_log is None:
		raise RuntimeError("Logger not initialised. Call set_log(caseID) first.")
	global_log.log(msg, do_print=do_print)
