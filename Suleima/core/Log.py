import os
from datetime import datetime
from core.globals import *
import pandas as pd
from pathlib import Path

class Log:
	def __init__(self, caseID):
		self.caseID = caseID
		self.logPath = os.path.join("data", "cases", caseID, "info.txt")
		os.makedirs(os.path.dirname(self.logPath), exist_ok=True)
		self.logs = []


	def log(self, msg, do_print=F):
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"[{timestamp}] {msg}"
		self.logs.append(entry)
		with open(self.logPath, "a", encoding="utf-8") as f:
			f.write(entry + "\n")


		# Final decision to print
		if do_print is T:
			print(entry)

# Global variable to hold the active logger
global_log = None

def set_log(caseID):
	"""Initialise the global log instance."""
	global global_log
	global_log = Log(caseID)

def log(msg, do_print=F):
	"""Log a message using the global logger."""
	if global_log is None:
		raise RuntimeError("Logger not initialised. Call set_log(caseID) first.")
	global_log.log(msg, do_print=do_print)



'''
columns i want in my CSV per case:
	"caseId"

'''
def loginfo(outliers, path="data/intensity_outliers.csv"):
	"""Log outliers to a CSV file."""
	outliers = outliers.reset_index(drop=True)
	outliers.to_csv(path, index=False)
	#log(f"Outliers saved to data/intensity_outliers.csv", False)


def save_intensity_stats(stats_list, filepath="data/ct_intensity_summary.csv"):
	"""Append stats to CSV, writing header only if file doesn't exist."""
	filepath = Path(filepath)
	df_stats = pd.DataFrame(stats_list)
	# Write mode: append if file exists, else create
	write_header = not filepath.exists()
	df_stats.to_csv(filepath, mode='a', header=write_header, index=False)
	return df_stats