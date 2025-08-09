import shutil
import pandas as pd
import os


def create_segments(folder):
	# rename {folder}/ct_image.nii.gz to fullCT.nii.gz
	# place all files other than fullCT.nii.gz in a subfolder called "segments"
	old_path = os.path.join(folder, "ct_image.nii.gz")
	new_path = os.path.join(folder, "fullCT.nii.gz")
	if os.path.exists(old_path):
		shutil.move(old_path, new_path)
		print(f"Renamed {old_path} to {new_path}")
	else:
		print(f"{old_path} does not exist.")

	segments_dir = os.path.join(folder, "segments")
	os.makedirs(segments_dir, exist_ok=True)


	for file in os.listdir(folder):
		full_path = os.path.join(folder, file)

		if os.path.isdir(full_path) or file == "fullCT.nii.gz":
			continue

		new_file_path = os.path.join(segments_dir, file)
		shutil.move(full_path, new_file_path)
		print(f"Moved {full_path} to {new_file_path}")

def createCNTRLresampled(old_patient):
	'''
	all files in folder starting with cropped_ should be renamed/moved based on the following patterns:

	example normal_cases/{old_patient}/:
	AAP 50415783_61F
	rename/move cropped_ct:
	from          f"data/Outputs/normal_cases/AAP 50415783_61F/cropped_ct.nii.gz"
	to            f"data/cases/CNTRL_AAP_50415783_61F/resampledCT.nii.gz"

	all other files starting with cropped_ should be renamed/moved based on the following pattern:
	from          f"data/Outputs/normal_cases/AAP 50415783_61F/cropped_la.nii.gz"
	to           f"data/cases/CNTRL_AAP_50415783_61F/resampled/resampledLA.nii.gz"

	from         f"data/Outputs/normal_cases/AAP 50415783_61F/cropped_lv.nii.gz"
	to           f"data/cases/CNTRL_AAP_50415783_61F/resampled/resampledLV.nii.gz"
	etc.
	'''

	source_dir = f"data/Outputs/normal_cases/{old_patient}/"
	patient_clean = old_patient.replace(" ", "_")

	dest_base = f"data/cases/CNTRL_{patient_clean}"
	dest_resampled = os.path.join(dest_base, "resampled")

	os.makedirs(dest_resampled, exist_ok=True)

	for filename in os.listdir(source_dir):
		if not filename.startswith("cropped_") or not filename.endswith(".nii.gz"):
			continue

		src_path = os.path.join(source_dir, filename)
		if filename == "cropped_ct.nii.gz":
			dst_path = os.path.join(dest_base, "resampledCT.nii.gz")
		else:
			segment = filename.replace("cropped_", "").replace(".nii.gz", "").upper()
			dst_path = os.path.join(dest_resampled, f"resampled{segment}.nii.gz")
		shutil.move(src_path, dst_path)
		print(f"Moved {src_path} → {dst_path}")

def createTTSresampled(old_patient):
	'''
	all files in folder starting with cropped_ should be renamed/moved based on the following patterns:
	example takotsubo_cases/{old_patient}/:
	source_dir = data/Outputs/takotsubo_cases/AD 12191953/
	rename/move cropped_ct:
	from          f"data/Outputs/takotsubo_cases/AD 12191953/cropped_ct.nii.gz"
	to            f"data/cases/TTS_AD_12191953_51F/resampledCT.nii.gz"

	all other files starting with cropped_ should be renamed/moved based on the following pattern:
	from          f"data/Outputs/takotsubo_cases/AD 12191953/cropped_lv.nii.gz"
	to            f"data/cases/TTS_AD_12191953_51F/resampled/resampledLV.nii.gz"

	from         f"data/Outputs/takotsubo_cases/AD 12191953/cropped_rv.nii.gz"
	to           f"data/cases/TTS_AD_12191953_51F/resampled/resampledRV.nii.gz"
	etc.
	'''
	source_dir = f"data/Outputs/takotsubo_cases/{old_patient}/"
	TTS_metadata = pd.read_csv("data/takotsubo_cases_metadata.csv")
	TTS_metadata['PatientID'] = TTS_metadata['PatientID'].str.strip()
	TTS_metadata['Gender'] = TTS_metadata['Gender'].map({0: 'M', 1: 'F'})

	patient_key = old_patient.strip()

	match = TTS_metadata[TTS_metadata['PatientID'] == patient_key]
	if match.empty:
		print(f"⚠️ No metadata found for: {old_patient}")
		return
	age = int(match['Age'].values[0])
	gender = match['Gender'].values[0]

	patient_clean = old_patient.replace(" ", "_")
	new_name = f"TTS_{patient_clean}_{age}{gender}"

	dest_base = os.path.join("data/cases", new_name)
	dest_resampled = os.path.join(dest_base, "resampled")
	os.makedirs(dest_resampled, exist_ok=True)

	for filename in os.listdir(source_dir):
		if not filename.startswith("cropped_") or not filename.endswith(".nii.gz"):
			continue

		src_path = os.path.join(source_dir, filename)

		# Special case: cropped_ct.nii.gz
		if filename == "cropped_ct.nii.gz":
			dst_path = os.path.join(dest_base, "resampledCT.nii.gz")
		else:
			# General case: cropped_lv.nii.gz → resampledLV.nii.gz
			seg = filename.replace("cropped_", "").replace(".nii.gz", "").upper()
			dst_path = os.path.join(dest_resampled, f"resampled{seg}.nii.gz")

		shutil.move(src_path, dst_path)
		print(f"Moved {src_path} → {dst_path}")

def rename_takotsubo_cases(old_patient):
	source_dir = f"data/Outputs/takotsubo_cases/{old_patient}/"
	TTS_metadata = pd.read_csv("data/takotsubo_cases_metadata.csv")
	TTS_metadata['PatientID'] = TTS_metadata['PatientID'].str.strip()
	TTS_metadata['Gender'] = TTS_metadata['Gender'].map({0: 'M', 1: 'F'})

	patient_key = old_patient.strip()

	match = TTS_metadata[TTS_metadata['PatientID'] == patient_key]
	if match.empty:
		print(f"⚠️ No metadata found for: {old_patient}")
		return
	age = int(match['Age'].values[0])
	gender = match['Gender'].values[0]

	patient_clean = old_patient.replace(" ", "_")
	new_name = f"{patient_clean}_{age}{gender}"
	target_dir = f"data/Outputs/takotsubo_cases/{new_name}/"

	if not os.path.exists(source_dir):
		print(f"❌ Source folder not found: {source_dir}")
		return

	if os.path.exists(target_dir):
		print(f"⚠️ Target folder already exists: {target_dir}")
		return

	shutil.move(source_dir, target_dir)
	print(f"✅ Renamed {source_dir} → {target_dir}")

def rename_normal_cases(old_patient):
	source_dir = f"data/Outputs/normal_cases/{old_patient}/"
	patient_clean = old_patient.replace(" ", "_")

	target_dir = f"data/Outputs/normal_cases/{patient_clean}/"
	if not os.path.exists(source_dir):
		print(f"❌ Source folder not found: {source_dir}")
		return

	if os.path.exists(target_dir):
		print(f"⚠️ Target folder already exists: {target_dir}")
		return
	shutil.move(source_dir, target_dir)
	print(f"✅ Renamed {source_dir} → {target_dir}")

def move_Outputs_slices(case, patient):
	'''
	case = "data/Outputs/takotsubo_cases/" or "data/Outputs/normal_cases/"
	if case == "data/Outputs/takotsubo_cases/":
		# Handle takotsubo cases
		CASE = "TTS"
		pass
	elif case == "data/Outputs/normal_cases/":
		# Handle normal cases
		CASE = "CNTRL"
		pass

	files in  {case}/{patient}/   should be moved following this pattern:


	## CT SLICES: ctX_   &&    ctY_   &&    ctZ_
	from
				{case}/{patient}/nii_slices/ctX_{idx}.nii.gz
				{case}/{patient}/nii_slices/ctY_{idx}.nii.gz
				{case}/{patient}/nii_slices/ctZ_{idx}.nii.gz
	to
				data/cases/{CASE}_{patient}/ctSlices/ctX_{idx}.nii.gz
				data/cases/{CASE}_{patient}/ctSlices/ctY_{idx}.nii.gz
				data/cases/{CASE}_{patient}/ctSlices/ctZ_{idx}.nii.gz


	## MASK SLICES: maskX_   &&    maskY_   &&    maskZ_
	from
				{case}/{patient}/nii_slices/maskX_{idx}.nii.gz
				{case}/{patient}/nii_slices/maskY_{idx}.nii.gz
				{case}/{patient}/nii_slices/maskZ_{idx}.nii.gz
	to
				data/cases/{CASE}_{patient}/maskSlices/maskX_{idx}.nii.gz
				data/cases/{CASE}_{patient}/maskSlices/maskY_{idx}.nii.gz
				data/cases/{CASE}_{patient}/maskSlices/maskZ_{idx}.nii.gz


	## PNG SLICES: pngX_   &&    pngY_   &&    pngZ_
	from
				{case}/{patient}/png_slices/{ct or mask}X_{idx}.png
				{case}/{patient}/png_slices/{ct or mask}Y_{idx}.png
				{case}/{patient}/png_slices/{ct or mask}Z_{idx}.png
	to
				data/cases/{CASE}_{patient}/pngSlices/{ct or mask}X_{idx}.png
				data/cases/{CASE}_{patient}/pngSlices/{ct or mask}Y_{idx}.png
				data/cases/{CASE}_{patient}/pngSlices/{ct or mask}Z_{idx}.png

	'''

	if case == "data/Outputs/takotsubo_cases/":
		CASE = "TTS"
	elif case == "data/Outputs/normal_cases/":
		CASE = "CNTRL"
	else:
		print(f"❌ Unknown case type: {case}")
		return

	new_root = os.path.join("data/cases", f"{CASE}_{patient}")

	# === 1. CT Slices ===
	ct_source_dir = os.path.join(case, patient, "nii_slices")
	ct_target_dir = os.path.join(new_root, "ctSlices")
	os.makedirs(ct_target_dir, exist_ok=True)

	for file in os.listdir(ct_source_dir) if os.path.exists(ct_source_dir) else []:
		if file.startswith("ctX_") or file.startswith("ctY_") or file.startswith("ctZ_"):
			src = os.path.join(ct_source_dir, file)
			dst = os.path.join(ct_target_dir, file)
			shutil.move(src, dst)
			print(f"✅ Moved CT slice: {src} → {dst}")

	# === 2. Mask Slices ===
	mask_source_dir = os.path.join(case, patient, "nii_slices")
	mask_target_dir = os.path.join(new_root, "maskSlices")
	os.makedirs(mask_target_dir, exist_ok=True)

	for file in os.listdir(mask_source_dir) if os.path.exists(mask_source_dir) else []:
		if file.startswith("maskX_") or file.startswith("maskY_") or file.startswith("maskZ_"):
			src = os.path.join(mask_source_dir, file)
			dst = os.path.join(mask_target_dir, file)
			shutil.move(src, dst)
			print(f"✅ Moved mask slice: {src} → {dst}")

	# === 3. PNG Slices ===
	png_source_dir = os.path.join(case, patient, "png_slices")
	png_target_dir = os.path.join(new_root, "pngSlices")
	os.makedirs(png_target_dir, exist_ok=True)

	for file in os.listdir(png_source_dir) if os.path.exists(png_source_dir) else []:
		if file.startswith("ct") or file.startswith("mask"):
			src = os.path.join(png_source_dir, file)
			dst = os.path.join(png_target_dir, file)
			shutil.move(src, dst)
			print(f"✅ Moved PNG slice: {src} → {dst}")



def delete_files(casePath):
	p = casePath / "resampledCT.nii.gz"
	if p.exists():
		os.remove(p)
	# delete all files in casePath/resampled
	for file in os.listdir(casePath / "resampled"):
		p = casePath / "resampled" / file
		if p.is_file():
			os.remove(p)
			#print(f"Deleted: {p}")

	for file in os.listdir(casePath / "ctSlices"):
		p = casePath / "ctSlices" / file
		if p.is_file():
			os.remove(p)
			#print(f"Deleted: {p}")

	for file in os.listdir(casePath / "maskSlices"):
		p = casePath / "maskSlices" / file
		if p.is_file():
			os.remove(p)
			#print(f"Deleted: {p}")
	print(f"deleted from {casePath}: resampledCT, resampled files, ctSlices and maskSlices.")


def loop_cases():
	#loop=0
	from pathlib import Path
	root_dir = Path("data/cases")
	for caseID in os.listdir(root_dir):
		if caseID.startswith('.'):
			continue
		casePath = root_dir / caseID
		delete_files(casePath)


'''


################# USE WITH ###################
## OLD STRUCTURE:
output_normal_cases = "data/Outputs/normal_cases/"
output_takotsubo_cases = "data/Outputs/takotsubo_cases/"
for old_patient in os.listdir(output_takotsubo_cases):
	if old_patient.startswith('.'):
		continue
	print(f"Processing {old_patient}")


	#move_Outputs_slices(output_takotsubo_cases, old_patient)
	#move_Outputs_slices(output_normal_cases, old_patient)
	#rename_takotsubo_cases(old_patient)
	#createTTSresampled(old_patient)
	#rename_normal_cases(old_patient)
	#createCNTRLresampled(old_patient)
	#create_segments(old_patient)



## NEW STRUCTURE:
base_root = "data/cases_done/"
new_root = "data/cases/"
for caseID in os.listdir(base_root):
	if caseID.startswith('.'):
		continue
	from_path = os.path.join(base_root, caseID)
	to_path = os.path.join(new_root, caseID)
	os.makedirs(to_path, exist_ok=True)
	# copy: (cases_done/caseID/fullCT.nii.gz,
			# cases_done/caseID/segments/ <segment_files> )
	# to: (cases/caseID/fullCT.nii.gz,
			# cases/caseID/segments/ <segment_files> )




import os
import shutil
base_root = "data/cases_done/"
new_root = "data/cases/"

for caseID in os.listdir(base_root):
	if caseID.startswith('.'):
		continue

	from_path = os.path.join(base_root, caseID)
	to_path = os.path.join(new_root, caseID)

	os.makedirs(to_path, exist_ok=True)

	# === Copy fullCT.nii.gz ===
	src_ct = os.path.join(from_path, "fullCT.nii.gz")
	dst_ct = os.path.join(to_path, "fullCT.nii.gz")
	if os.path.exists(src_ct):
		shutil.copy2(src_ct, dst_ct)
	else:
		print(f"[WARNING] Missing fullCT.nii.gz in {caseID}")

	# === Copy segments folder ===
	src_segments = os.path.join(from_path, "segments")
	dst_segments = os.path.join(to_path, "segments")
	if os.path.exists(src_segments):
		shutil.copytree(src_segments, dst_segments, dirs_exist_ok=True)
	else:
		print(f"[WARNING] Missing segments folder in {caseID}")
'''

