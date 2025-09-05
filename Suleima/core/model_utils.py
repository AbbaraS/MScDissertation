import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import copy
from core.CNNmodel import *
from core.benchmarks import create_adapted_resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
import logging
from core.globals import *
from tqdm.notebook import tqdm
from core.Log import *
from core.CVsplits import *
from core.globals import *




def gradcam_eval(
	model,
	loader,
	target_layers,        # e.g., {"axial": model.axial_branch.bn3, ...}
	TH=0.5,
	case_ids_filter=None, # optional subset by CaseID(s)
	target="pred",        # "pred" | "pos" | "neg"
	max_cases=None,
	save_npz_path=None,
):
	device = next(model.parameters()).device
	model.eval()  # deterministic BN/Dropout

	# Hook buffers (per view)
	feats, grads = {k: None for k in target_layers}, {k: None for k in target_layers}
	handles = []

	def fwd_hook_factory(key):
		def _hook(m, inp, out):
			feats[key] = out           # [N,C,h,w]
		return _hook

	def bwd_hook_factory(key):
		def _hook(m, gin, gout):
			grads[key] = gout[0]       # [N,C,h,w] (∂y/∂feature)
		return _hook

	for k, layer in target_layers.items():
		handles.append(layer.register_forward_hook(fwd_hook_factory(k)))
		handles.append(layer.register_full_backward_hook(bwd_hook_factory(k)))

	# Collectors
	case_ids_all, y_true_all, y_prob_all, y_logit_all, y_hat_all = [], [], [], [], []
	cams_per_view = {k: [] for k in target_layers}  # view -> list of 2D np arrays

	seen = 0
	with torch.enable_grad():  # gradients needed for Grad-CAM
		for batch in loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).view(-1, 1).float()
			cids = list(batch["CaseID"])

			# Optional subsetting by CaseID
			if case_ids_filter is not None:
				mask_list = [cid in case_ids_filter for cid in cids]
				if not any(mask_list):
					continue
				mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
				axi, cor, sag, met, lbl = axi[mask], cor[mask], sag[mask], met[mask], lbl[mask]
				cids = [cid for cid, keep in zip(cids, mask_list) if keep]

			if axi.numel() == 0:
				continue

			# -------- Forward pass (batch) --------
			logits = model(axi, cor, sag, met)                 # [N,1]
			prob_vec = torch.sigmoid(logits).squeeze(1)        # [N]
			pred_vec = (prob_vec >= TH).long()                 # [N]

			# -------- BATCH-WISE BACKWARD HERE --------
			# Build signs per sample for the chosen target, then one backward for the whole batch
			logit_vec = logits.squeeze(1)                      # [N]
			if target == "pred":
				signs = torch.where(pred_vec == 1,
									 torch.ones_like(logit_vec),
									 -torch.ones_like(logit_vec))
			elif target == "pos":
				signs = torch.ones_like(logit_vec)
			elif target == "neg":
				signs = -torch.ones_like(logit_vec)
			else:
				raise ValueError("target must be 'pred', 'pos', or 'neg'")

			model.zero_grad(set_to_none=True)
			objective = (signs * logit_vec).sum()
			objective.backward()   # grads for the entire batch at once
			# -------- END BATCH-WISE BACKWARD --------

			# Build CAMs per view for each sample using the captured feats/grads
			N = logits.shape[0]
			# If you want to cap total processed cases, compute remaining count:
			take = N if (max_cases is None) else max(0, min(N, max_cases - seen))

			for i in range(take):
				for view, x_in in (("axial", axi), ("coronal", cor), ("sagittal", sag)):
					if feats[view] is None or grads[view] is None:
						cams_per_view.setdefault(view, []).append(None)
						continue

					Fview = feats[view][i]                 # [C,h,w]
					Gview = grads[view][i]                 # [C,h,w]
					w = Gview.mean(dim=(1, 2), keepdim=True)     # [C,1,1]
					cam = F.relu((w * Fview).sum(dim=0))         # [h,w]
					# normalize and upsample to that view's input spatial size
					m = cam.max()
					cam = cam / (m + 1e-12) if m > 0 else cam
					Hin, Win = x_in.shape[-2], x_in.shape[-1]
					cam_up = F.interpolate(cam[None, None], size=(Hin, Win),
										   mode="bilinear", align_corners=False).squeeze().detach()
					cams_per_view[view].append(cam_up.cpu().numpy())

				# Bookkeeping for this sample
				case_ids_all.append(cids[i])
				y_true_all.append(int(lbl[i].item()))
				y_prob_all.append(float(prob_vec[i].item()))
				y_logit_all.append(float(logit_vec[i].item()))
				y_hat_all.append(int(pred_vec[i].item()))
				seen += 1

			if max_cases is not None and seen >= max_cases:
				break

	# Clean up hooks
	for h in handles:
		h.remove()


	results = {
		"case_ids": case_ids_all,
		"y_true": np.asarray(y_true_all, dtype=int) ,
		"y_prob": np.asarray(y_prob_all, dtype=float) ,
		"y_logit": np.asarray(y_logit_all, dtype=float) ,
		"y_hat": np.asarray(y_hat_all, dtype=int) ,
		#"y_true":  y_true_all.tolist() ,
		#"y_prob":  y_prob_all.tolist() ,
		#"y_logit": y_logit_all.tolist(),
		#"y_hat":   y_hat_all.tolist()  ,
		"TH_used": float(TH),
		"cams": cams_per_view,   # dict: view -> list of HxW arrays (aligned with case_ids order)
		"target_mode": target,
	}
	if save_npz_path:
		# store ragged list of arrays using dtype=object for cams
		np.savez_compressed(
			save_npz_path,
			case_ids=np.array(case_ids_all, dtype=object),
			y_true=results["y_true"],
			y_prob=results["y_prob"],
			y_logit=results["y_logit"],
			y_hat=results["y_hat"],
			TH_used=results["TH_used"],
			target_mode=results["target_mode"],
			cams_axial=np.array(cams_per_view.get("axial", []), dtype=object),
			cams_coronal=np.array(cams_per_view.get("coronal", []), dtype=object),
			cams_sagittal=np.array(cams_per_view.get("sagittal", []), dtype=object),
		)
	return results

import numpy as np, matplotlib.pyplot as plt, torch
import matplotlib.cm as cm

def tensor_to_imgchw(x):
	"""x: torch.Tensor [C,H,W] on any device; returns float32 HxWxC in [0,1] (min-max per image)."""
	x = x.detach().float().cpu()
	# min-max normalise per image to [0,1] for display (safe if already [0,1])
	vmin, vmax = x.min(), x.max()
	if (vmax - vmin) > 1e-12:
		x = (x - vmin) / (vmax - vmin)
	return x.permute(1, 2, 0).numpy()

def unnormalize(img, mean=None, std=None):
	"""Optional: invert dataset normalisation (channel-wise). img: HxWxC float [0,1] or z-scored."""
	if mean is None or std is None:
		return img
	mean = np.array(mean).reshape(1,1,-1); std = np.array(std).reshape(1,1,-1)
	out = img * std + mean
	# re-normalise into [0,1] for display
	lo, hi = out.min(), out.max()
	return (out - lo) / (hi - lo + 1e-12)

def overlay_cam(img, cam, alpha=0.35, cmap_name="jet"):
	"""
	img: HxWxC in [0,1]; cam: HxW in [0,1].
	Returns an HxWxC float image with CAM heatmap overlaid.
	"""
	cmap = cm.get_cmap(cmap_name)
	heat = cmap(cam)[..., :3]                 # HxWx3 RGB heatmap
	# If img has 1 channel, repeat to 3 for overlay
	if img.shape[2] == 1:
		base = np.repeat(img, 3, axis=2)
	else:
		base = img[..., :3]
	return (1 - alpha) * base + alpha * heat

def find_case_in_batch(cids_list, target_cid):
	"""Return index of target_cid in list (or None)."""
	for i, cid in enumerate(cids_list):
		if cid == target_cid:
			return i
	return None

def show_case_cams(results, loader, case_id, mean=None, std=None, save_path=None):
	"""
	results: dict returned by gradcam_eval (with results['cams'] and results['case_ids'])
	loader : your test DataLoader (to fetch original images at display size)
	case_id: string/int matching CaseID
	mean/std: optional channel stats to unnormalize for nicer visuals
	"""
	# Locate the CAMs and metadata for this case
	idx = results["case_ids"].index(case_id)
	cams = results["cams"]  # dict: 'axial','coronal','sagittal' -> list of HxW arrays
	cam_ax, cam_co, cam_sa = cams.get("axial", [None])[idx], cams.get("coronal", [None])[idx], cams.get("sagittal", [None])[idx]
	y, p = results["y_true"][idx], results["y_prob"][idx]
	yhat, th = results["y_hat"][idx], results["TH_used"]
	title = f"Case {case_id} | y={y}, p={p:.3f}, ŷ={yhat} @ TH={th}"

	# Find the raw images from the loader
	img_ax = img_co = img_sa = None
	for batch in loader:
		cids = list(batch["CaseID"])
		j = find_case_in_batch(cids, case_id)
		if j is None:
			continue
		# take the j-th sample from this batch
		img_ax = tensor_to_imgchw(batch["axial_image"][j])
		img_co = tensor_to_imgchw(batch["coronal_image"][j])
		img_sa = tensor_to_imgchw(batch["sagittal_image"][j])
		break
	if img_ax is None:
		raise ValueError(f"CaseID {case_id} not found in loader.")

	# Optional: invert dataset normalisation (supply mean/std if you used them)
	img_ax = unnormalize(img_ax, mean, std)
	img_co = unnormalize(img_co, mean, std)
	img_sa = unnormalize(img_sa, mean, std)

	# Build overlays (gracefully handle None CAMs)
	ov_ax = overlay_cam(img_ax, cam_ax) if cam_ax is not None else img_ax
	ov_co = overlay_cam(img_co, cam_co) if cam_co is not None else img_co
	ov_sa = overlay_cam(img_sa, cam_sa) if cam_sa is not None else img_sa

	# Plot
	fig, axs = plt.subplots(1, 3, figsize=(12, 4))
	for ax, im, lbl in zip(axs, [ov_ax, ov_co, ov_sa], ["Axial", "Coronal", "Sagittal"]):
		ax.imshow(im)
		ax.set_title(lbl); ax.axis("off")
	fig.suptitle(title, fontsize=11)
	fig.tight_layout(rect=[0,0,1,0.95])

	if save_path:
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
	return fig, axs

def gallery(results, loader, where=lambda y, yhat: True, order_by="conf", K=6, mean=None, std=None, save_path=None):
	"""
	where: predicate(y, yhat) to filter cases (e.g., FN: lambda y,yh: (y==1 and yh==0))
	order_by: "conf"=|p-0.5| descending, or "p"=probability descending
	"""
	ids = results["case_ids"]
	y   = results["y_true"]
	yh  = results["y_hat"]
	p   = results["y_prob"]

	idxs = [i for i,(yi,yhi) in enumerate(zip(y,yh)) if where(yi, yhi)]
	if not idxs:
		print("No cases match the predicate.")
		return None, None

	scores = np.abs(p[idxs]-0.5) if order_by=="conf" else p[idxs]
	order = np.argsort(-scores)
	idxs = [idxs[i] for i in order[:K]]

	cols = 3  # 3 views
	rows = len(idxs)
	fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
	axs = np.atleast_2d(axs)

	for r, i in enumerate(idxs):
		cid, yi, pi, yhi = ids[i], y[i], p[i], yh[i]
		# draw images
		fig_row = show_case_cams(results, loader, cid, mean, std)[0]  # returns a fig; we instead reuse logic below if you prefer speed
		plt.close(fig_row)  # we don't need the single-case fig if using a grid; to keep simple, call show_case_cams above and skip grid assembly.
	# Simpler: just call show_case_cams in a loop and save per-case PNGs.

	# (If you want a real grid in one figure, refactor show_case_cams to return the raw overlays and blit them into axs[r,c].)
	return None, None







def TRAIN_MODEL(model, train_loader, val_loader, hypers):
	log = logging.getLogger('OUTER_train')
	log.info(f"		 ExpID; HP_Set;  Fold;   Epoch;   TrainLoss;   		 TrainAcc;             ValLoss;              	ValAcc; 	LR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	LR = hypers['LR']
	WD = hypers['WD']
	TH = hypers['TH']
	ExpID = hypers['ExpID']
	P = hypers['P']
	epochs = hypers['epochs']

	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=P, factor=0.5)
	criterion = nn.BCEWithLogitsLoss()

	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	pbar_epochs = tqdm(range(epochs), desc=f"	↳ Experiment {ExpID} | Training model... ", position=ExpID, leave=True)
	for epoch in pbar_epochs:
		model.train()
		running_loss = 0.0
		all_labels = []
		all_preds = []

		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad()
			outputs = model(axi, sag, cor, met)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > TH
			all_preds.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

		T_loss = running_loss / train_N
		T_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		model.eval()
		running_loss = 0.0
		all_labels = []
		all_preds = []
		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				outputs = model(axi, sag, cor, met)
				V_loss = criterion(outputs, lbl)

				running_loss += V_loss.item() * lbl.size(0)
				prediction = torch.sigmoid(outputs) > TH
				all_preds.extend(prediction.cpu().numpy())
				all_labels.extend(lbl.cpu().numpy())

		V_loss = running_loss / val_N
		V_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		scheduler.step(V_loss)
		if V_loss < best_V_loss:
			best_V_loss = V_loss
			best_model_state = copy.deepcopy(model.state_dict())
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"		{ExpID}; {hypers['HPset']}; {hypers['Fold']}; {epoch}; {T_loss}; {T_acc}; {V_loss}; {V_acc}; {optimizer.param_groups[0]['lr']}")
		if P_counter >= P: break
	model.load_state_dict(best_model_state)
	return model, best_model_state

def EVALUATE_MODEL1(model, test_loader, hypers):
	TH = hypers['TH']
	ExpID = hypers['ExpID']
	log = logging.getLogger('OUTER_evaluate')
	log.info(f"		 ExpID; HP_Set;  Fold;   CaseID;   prediction;   		 probability    ")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()
	criterion = nn.BCEWithLogitsLoss()
	eval_N = len(test_loader.dataset)

	running_loss = 0.0
	all_predictions = []
	all_probabilities = []
	all_labels = []
	with torch.no_grad():
		pbar_eval = tqdm(test_loader, desc=f"	↳ Experiment {ExpID} | Evaluating model... ", position=ExpID, leave=True)
		for batch in pbar_eval:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			CaseID = batch["CaseID"]

			outputs = model(axi, sag, cor, meta=met)
			loss = criterion(outputs, lbl)
			running_loss += loss.item() * lbl.size(0)

			probability = torch.sigmoid(outputs)
			all_probabilities.extend(probability.cpu().numpy())
			prediction = probability > TH

			all_predictions.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())
			current_batch_size = len(CaseID)
			for i in range(current_batch_size):
				#print(c)
				#c += 1
				# Use .item() to get the Python scalar value from the tensor
				#print(i)
				case_id = CaseID[i]
				pred = prediction[i].item()
				prob = probability[i].item()
				log.info(f"     {ExpID}; {hypers['HPset']}; {hypers['Fold']}; {case_id}; {pred}; {prob:.4f};")


	final_loss = running_loss / eval_N
	return final_loss, all_probabilities, all_labels, all_predictions

def train_INNER_model(model, train_loader, val_loader, experiment):
	log = logging.getLogger('INNER_5Ktrain')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	hypers = experiment['hypers']
	ExpID = experiment['ExpID']
	epochs = hypers['Epochs']

	LR = hypers['LR']
	WD = hypers['WD']
	P = hypers['P']

	optimizer = optim.Adam(model.parameters(), lr= LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2, factor=0.5,
								  threshold=1e-3, threshold_mode='rel',
								  cooldown=0, min_lr=1e-6)
	criterion = nn.BCEWithLogitsLoss()

	best_auc = -np.inf
	best_loss_at_best_auc = np.inf
	best_th = 0.5

	no_improve = 0

	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)


	print(f"	↳ Experiment {ExpID} | Training model... ")
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0

		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad(set_to_none=True)

			logits = model(axi, sag, cor, met)
			T_loss = criterion(logits, lbl)

			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)

		T_loss = running_loss / train_N

		model.eval()
		running_loss = 0.0
		all_labels = []  #y_true
		all_probs = []   #y_pred

		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				logits = model(axi, sag, cor, met)
				V_loss = criterion(logits, lbl)
				running_loss += V_loss.item() * lbl.size(0)

				all_labels.extend(lbl.cpu())
				all_probs.extend(torch.sigmoid(logits).cpu())

		V_loss = running_loss / val_N
		all_labels = torch.cat(all_labels).numpy().reshape(-1)
		all_probs = torch.cat(all_probs).numpy().reshape(-1)
		val_auc = roc_auc_score(all_labels, all_probs)

		scheduler.step(V_loss)
		th_star = best_threshold(all_labels, all_probs)
		improved = val_auc > best_auc + 1e-6

		log.info(f"{experiment['Model']};    {ExpID};    {experiment['OUTER_FOLD']};    {experiment['INNER_FOLD']};    {hypers['HPset']};    {epoch:02d};    {T_loss:.4f};    {V_loss:.4f}    {val_auc:.4f};    {th_star:.6f};    {optimizer.param_groups[0]['lr']};    {no_improve:02d};")

		if improved:
			best_auc = val_auc
			best_loss_at_best_auc = V_loss
			best_th = th_star
			best_epoch = epoch
			no_improve = 0
		else:
			no_improve += 1
		if no_improve >= P: break

	results = {
		"best_val_auc": float(best_auc),
		"best_val_loss": float(best_loss_at_best_auc),
		"best_threshold": float(best_th),
		"best_epoch": int(best_epoch),
		"epochs_ran": int(best_epoch + 1)}

	return results





def train_INNER_modelRESNET(model, train_loader, val_loader, experiment):
	log = logging.getLogger('INNER_5KtrainBENCHMARKS')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	feature_extractor, model = create_adapted_resnet18(device)

	hypers = experiment['hypers']
	ExpID = experiment['ExpID']
	epochs = hypers['Epochs']

	LR = hypers['LR']
	WD = hypers['WD']
	P = hypers['P']

	optimizer = optim.Adam(model.parameters(), lr= LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2, factor=0.5,
								  threshold=1e-3, threshold_mode='rel',
								  cooldown=0, min_lr=1e-6)
	criterion = nn.BCEWithLogitsLoss()

	best_auc = -np.inf
	best_loss_at_best_auc = np.inf
	best_th = 0.5

	no_improve = 0

	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)


	print(f"	↳ Experiment {ExpID} | Training model... ")
	for epoch in range(epochs):
		model.train()
		feature_extractor.eval()
		running_loss = 0.0

		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			with torch.no_grad():
				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)
			combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)

			optimizer.zero_grad(set_to_none=True)

			logits = model(combined_input)
			T_loss = criterion(logits, lbl)

			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)

		T_loss = running_loss / train_N

		model.eval()
		running_loss = 0.0
		all_labels = []  #y_true
		all_probs = []   #y_pred

		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)

				combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)

				logits = model(combined_input)
				V_loss = criterion(logits, lbl)
				running_loss += V_loss.item() * lbl.size(0)

				all_labels.extend(lbl.cpu())
				all_probs.extend(torch.sigmoid(logits).cpu())

		V_loss = running_loss / val_N
		all_labels = torch.cat(all_labels).numpy().reshape(-1)
		all_probs = torch.cat(all_probs).numpy().reshape(-1)
		val_auc = roc_auc_score(all_labels, all_probs)

		scheduler.step(V_loss)
		th_star = best_threshold(all_labels, all_probs)
		improved = val_auc > best_auc + 1e-6

		log.info(f"{experiment['Model']};    {ExpID};    {experiment['OUTER_FOLD']};    {experiment['INNER_FOLD']};    {hypers['HPset']};    {epoch:02d};    {T_loss:.4f};    {V_loss:.4f}    {val_auc:.4f};    {th_star:.6f};    {optimizer.param_groups[0]['lr']};    {no_improve:02d};")

		if improved:
			best_auc = val_auc
			best_loss_at_best_auc = V_loss
			best_th = th_star
			best_epoch = epoch
			no_improve = 0
		else:
			no_improve += 1
		if no_improve >= 3: break

	results = {
		"ExpID": ExpID,
		"Model": experiment['Model'],
		"best_val_auc": float(best_auc),
		"best_val_loss": float(best_loss_at_best_auc),
		"best_threshold": float(best_th),
		"best_epoch": int(best_epoch),
		"epochs_ran": int(epoch)}

	return results









def best_threshold(all_labels, all_probs, utility="youden"):
	"""
	Pick a post-hoc decision threshold on validation predictions.
	utility: "youden" (maximize TPR-FPR) or "f1".
	"""
	if utility == "f1":
		# scan unique probabilities for F1
		# (for speed you can sample a subset if very large)
		thr = np.unique(all_probs)
		f1s = [f1_score(all_labels, all_probs >= t) for t in thr]
		idx = int(np.argmax(f1s))
		return float(thr[idx])
	else:
		fpr, tpr, thr = roc_curve(all_labels, all_probs)
		j = tpr - fpr
		idx = int(np.argmax(j))
		return float(thr[idx])  # may be outside [0,1] if degenerate; fine.






def train_MLP(model, train_loader, val_loader, experiment):
	log = logging.getLogger('OUTER_train')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	hypers = experiment['hypers']
	epochs = hypers['Epochs']
	optimizer = optim.Adam(model.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2,
								  factor=0.9,
								  threshold=0.01,
								  min_lr=8e-4)   # 0.0008 < 0.0009
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	print(f"	↳ Experiment {experiment['ExpID']} | Training model... ")
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		all_labels = []
		all_preds = []

		for batch in train_loader:
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			optimizer.zero_grad()
			outputs = model(met)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > hypers['TH']
			all_preds.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

		T_loss = running_loss / train_N
		T_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		model.eval()
		running_loss = 0.0
		all_labels = []
		all_preds = []
		with torch.no_grad():
			for batch in val_loader:
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)
				outputs = model(met)
				V_loss = criterion(outputs, lbl)

				running_loss += V_loss.item() * lbl.size(0)
				prediction = torch.sigmoid(outputs) > hypers['TH']
				all_preds.extend(prediction.cpu().numpy())
				all_labels.extend(lbl.cpu().numpy())
		V_loss = running_loss / val_N
		V_acc = np.mean(np.array(all_preds) == np.array(all_labels))

		scheduler.step(V_loss)
		if V_loss < best_V_loss:
			best_V_loss = V_loss
			best_model_state = copy.deepcopy(model.state_dict())
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"	 MLP_META;	{experiment['ExpID']};	  {experiment['OUTER_FOLD']};		 {hypers['HPset']};  				 {epoch}; 			{T_loss}; 			{T_acc};					{V_loss}; 	 	{V_acc};		{P_counter}; 		{optimizer.param_groups[0]['lr']}")
		if P_counter >= hypers['P']: break
	model.load_state_dict(best_model_state)
	return model, best_model_state

def train_RESNET(train_loader, val_loader, experiment):
	log = logging.getLogger('OUTER_train')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	feature_extractor, classifier = create_adapted_resnet18(device)
	#model.to(device)
	hypers = experiment['hypers']
	epochs = hypers['Epochs']
	TH = hypers['TH']
	optimizer = optim.Adam(classifier.parameters(), lr= hypers['LR'], weight_decay=hypers['WD'])
	P_counter = 0
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
								  patience=2,
								  factor=0.9,
								  threshold=0.01,
								  min_lr=8e-4)   # 0.0008 < 0.0009
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)

	print(f"	↳ RESNET Experiment {experiment['ExpID']} | Training model... ")
	for epoch in range(epochs):

		classifier.train()
		feature_extractor.eval()

		all_labels = []
		all_preds = []

		running_loss = 0.0
		for batch in train_loader:
			axi = batch["axial_image"].to(device)
			cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)
			with torch.no_grad():
				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)

			combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)

			optimizer.zero_grad()
			outputs = classifier(combined_input)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > TH
			all_preds.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

		T_loss = running_loss / train_N
		T_acc = np.mean(np.array(all_preds) == np.array(all_labels))

		classifier.eval()
		running_loss = 0.0
		all_labels = []
		all_preds = []
		with torch.no_grad():
			for batch in val_loader:
				axi = batch["axial_image"].to(device)
				cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				axi_features = feature_extractor(axi)
				cor_features = feature_extractor(cor)
				sag_features = feature_extractor(sag)

				combined_input = torch.cat([axi_features, cor_features, sag_features, met], dim=1)
				outputs = classifier(combined_input)
				V_loss = criterion(outputs, lbl)

				running_loss += V_loss.item() * lbl.size(0)
				prediction = torch.sigmoid(outputs) > TH
				all_preds.extend(prediction.cpu().numpy())
				all_labels.extend(lbl.cpu().numpy())

		V_loss = running_loss / val_N
		V_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		scheduler.step(V_loss)

		if V_loss < best_V_loss:
			best_V_loss = V_loss
			best_model_state = copy.deepcopy(classifier.state_dict())
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"	 RESNET;	{experiment['ExpID']};	  {experiment['OUTER_FOLD']};		 {hypers['HPset']};  				 {epoch}; 			{T_loss}; 			{T_acc};					{V_loss}; 	 	{V_acc};		{P_counter}; 		{optimizer.param_groups[0]['lr']}")
		if P_counter >= hypers['P']: break
	classifier.load_state_dict(best_model_state)
	return classifier, best_model_state





def train_SINGLEVIEW(model, train_loader, val_loader, experiment):
	log = logging.getLogger('OUTER_train')
	#log.info(f"		 ExpID; HP_Set;  Fold;   Epoch;   TrainLoss;   		 TrainAcc;             ValLoss;              	ValAcc; 	LR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	hypers = experiment['hypers']
	LR = hypers['LR']
	WD = hypers['WD']
	TH = hypers['TH']
	ExpID = experiment['ExpID']
	P = hypers['P']
	epochs = hypers['Epochs']

	optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
	scheduler = ReduceLROnPlateau(optimizer, mode='min',
							   patience=2, factor=0.9, threshold=0.01)
	criterion = nn.BCEWithLogitsLoss()
	best_V_loss = float('inf')
	P_counter = 0
	val_N=len(val_loader.dataset)
	train_N=len(train_loader.dataset)
	print(f"	↳ Single View Experiment {ExpID} | Training model... ")
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		all_labels = []
		all_preds = []

		for batch in train_loader:
			#axi = batch["axial_image"].to(device)
			#cor = batch["coronal_image"].to(device)
			sag = batch["sagittal_image"].to(device)
			met = batch["meta"].to(device)
			lbl = batch["label"].to(device).unsqueeze(1)

			optimizer.zero_grad()
			#outputs = model(axi, met)
			outputs = model(sag, met)
			#outputs = model(cor, met)
			T_loss = criterion(outputs, lbl)
			T_loss.backward()
			optimizer.step()

			running_loss += T_loss.item() * lbl.size(0)
			prediction = torch.sigmoid(outputs) > TH
			all_preds.extend(prediction.cpu().numpy())
			all_labels.extend(lbl.cpu().numpy())

		T_loss = running_loss / train_N
		T_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		model.eval()
		running_loss = 0.0
		all_labels = []
		all_preds = []
		with torch.no_grad():
			for batch in val_loader:
				#axi = batch["axial_image"].to(device)
				#cor = batch["coronal_image"].to(device)
				sag = batch["sagittal_image"].to(device)
				met = batch["meta"].to(device)
				lbl = batch["label"].to(device).unsqueeze(1)

				#outputs = model(axi, met)
				outputs = model(sag, met)
				#outputs = model(cor, met)
				V_loss = criterion(outputs, lbl)

				running_loss += V_loss.item() * lbl.size(0)
				prediction = torch.sigmoid(outputs) > TH
				all_preds.extend(prediction.cpu().numpy())
				all_labels.extend(lbl.cpu().numpy())

		V_loss = running_loss / val_N
		V_acc = np.mean(np.array(all_preds) == np.array(all_labels))
		scheduler.step(V_loss)
		if V_loss < best_V_loss:
			best_V_loss = V_loss
			best_model_state = copy.deepcopy(model.state_dict())
			P_counter = 0
		else:
			P_counter += 1
		log.info(f"		SAGITTAL_VIEW; 	{ExpID};	 {hypers['HPset']}; 	{experiment['OUTER_FOLD']};		 {epoch}; 		{T_loss}; 		{T_acc}; 			{V_loss}; 		{V_acc}; 		{P_counter};		{optimizer.param_groups[0]['lr']}	")
		if P_counter >= P: break
	model.load_state_dict(best_model_state)
	return model, best_model_state





