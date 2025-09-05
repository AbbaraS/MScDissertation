import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter,MultipleLocator, MaxNLocator, LogLocator

def _save_fig(fig, save_path=None, dpi=300):
	if save_path:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
		plt.close(fig)
		print(f"Saved figure → {save_path}")
	else:
		plt.show()

def plot_inner_loss_AUCs2(df, outer_fold, param_map, param_name, use_ema=False, save_path=None):
	# ---- helper: pretty formatting for legend labels ----
	def fmt_val(v):
		try:
			v = float(v)
			# show 0.3, 0.4 etc., use sci only for very small/large
			if 0.001 <= abs(v) < 1000:
				# trim trailing zeros
				s = f"{v:.3f}".rstrip("0").rstrip(".")
				return s if s else "0"
			return format(v, ".0e").replace("e-0","e-").replace("e+0","e+")
		except Exception:
			return str(v)

	# ---- robust dtypes ----
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["TrainLoss", "ValLoss", "EMA_ValLoss", "AUC"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	target_hpsets = [int(k) for k in param_map.keys()]
	val_loss_col  = "EMA_ValLoss" if use_ema and "EMA_ValLoss" in df.columns else "ValLoss"

	sub = df[
		(df["OuterFold"] == int(outer_fold)) &
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["InnerFold", "HPset", "Epoch", "TrainLoss", val_loss_col, "AUC"])
	if sub.empty:
		raise ValueError(f"No rows for outer fold {outer_fold} and HPsets {target_hpsets}")

	# ---- LOSS long format & aggregate ----
	long_loss = sub.melt(
		id_vars=["InnerFold", "HPset", "Epoch"],
		value_vars=["TrainLoss", val_loss_col],
		var_name="Split", value_name="Loss"
	).replace({val_loss_col: "ValLoss"})

	g_loss = (long_loss.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
						.agg(mean=("Loss","mean"), std=("Loss","std"), count=("Loss","count")))
	g_loss["ci95"] = 1.96 * g_loss["std"].fillna(0) / np.sqrt(g_loss["count"].clip(lower=1))

	# ---- AUC aggregate ----
	long_auc = sub[["InnerFold","HPset","Epoch","AUC"]].copy()
	long_auc["Split"] = "AUC"
	g_auc = (long_auc.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
					  .agg(mean=("AUC","mean"), std=("AUC","std"), count=("AUC","count")))
	g_auc["ci95"] = 1.96 * g_auc["std"].fillna(0) / np.sqrt(g_auc["count"].clip(lower=1))

	inner_folds = sorted(g_loss["InnerFold"].dropna().unique().astype(int))
	n_panels = len(inner_folds)

	# ---- colours keyed by the varying parameter value (e.g., DR) ----
	prms_in_order = [param_map[h] for h in target_hpsets]  # preserves HPset order you passed in
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}

	style_map = {"TrainLoss": "-", "ValLoss": "--"}  # row 1 only

	# ---- figure: 2 rows × N columns ----
	fig, axes = plt.subplots(2, n_panels, figsize=(5*n_panels, 8), sharex=False)
	if n_panels == 1:
		axes = np.array([[axes[0]], [axes[1]]])

	max_epoch = int(sub["Epoch"].max()) if len(sub) else 30

	# ----- Row 1: LOSS -----
	for j, inner in enumerate(inner_folds):
		ax = axes[0, j]
		gi = g_loss[g_loss["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color   = prm_to_color[prm_val]
			for split, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split], color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
		ax.set_title(f"Inner Fold {inner}")
		ax.set_ylabel("Loss")
		ax.set_xticks(range(int(long_loss["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	# ----- Row 2: AUC -----
	for j, inner in enumerate(inner_folds):
		ax = axes[1, j]
		gi = g_auc[g_auc["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color   = prm_to_color[prm_val]
			ax.plot(d_h["Epoch"], d_h["mean"], linestyle="-", color=color, linewidth=2)
			ax.fill_between(d_h["Epoch"], d_h["mean"]-d_h["ci95"], d_h["mean"]+d_h["ci95"],
							color=color, alpha=0.15)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("AUC")
		ax.set_xticks(range(int(long_loss["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	# ---- legend & titles ----
	param_label = param_name[0] if isinstance(param_name, (list, tuple)) else str(param_name)
	fixed_text  = (param_name[1] if isinstance(param_name, (list, tuple)) and len(param_name) > 1 else "").strip()

	handles = [Line2D([0],[0], color=prm_to_color[prm], lw=2, label=f"{param_label}={fmt_val(prm)}")
			   for prm in prms_in_order]
	fig.legend(handles, [h.get_label() for h in handles],
			   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.92))

	fig.suptitle(f"Outer Fold {outer_fold} | Dropout Rates", fontsize=13, y=0.995)
	fig.text(0.5, 0.95,
			 f"Top row: solid=training loss, dashed=validation loss   |   Bottom row: AUC   |   {fixed_text}",
			 ha="center", va="center", fontsize=10, color="dimgray")

	plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
	_save_fig(fig, save_path=save_path)

from matplotlib.ticker import MultipleLocator, ScalarFormatter

def plot_lr_wd_sweeps_by_dr_and_fold(
	df,
	metric_base= "AUC",
	use_ci= False,
	figsize_per_panel= 4.5,
	save_path = None):

	# --- 1. Validate inputs ---
	metric_mean_col = f"{metric_base}_mean"
	metric_std_col = f"{metric_base}_std"
	required = {"OuterFold", "WD", "DR", "LR", metric_mean_col, metric_std_col, "n"}
	if not required.issubset(df.columns):
		raise ValueError(f"DataFrame missing required columns: {required - set(df.columns)}")

	data = df.copy()

	# --- 2. Get unique values for facets and lines ---
	outer_folds = sorted(data["OuterFold"].unique())
	wds = sorted(data["WD"].unique())
	drs = sorted(data["DR"].unique())
	lrs = sorted(data["LR"].unique())

	# --- MODIFIED: Rows are now based on DR ---
	n_rows, n_cols = len(drs), len(outer_folds)

	# --- 3. Create the subplot grid ---
	fig, axes = plt.subplots(n_rows, n_cols,
							 figsize=(figsize_per_panel * n_cols, 3 * n_rows),
							 sharex=True, sharey=True,
							 constrained_layout=True)
	if n_cols == 1 and n_rows == 1: axes = np.array([[axes]])
	elif n_rows == 1: axes = np.expand_dims(axes, axis=0)
	elif n_cols == 1: axes = np.expand_dims(axes, axis=1)

	# --- 4. Iterate through grid and plot (MODIFIED LOOP ORDER) ---
	for row_idx, dr_val in enumerate(drs):
		for col_idx, fold_val in enumerate(outer_folds):
			ax = axes[row_idx, col_idx]

			# --- MODIFIED: Inner loop is now for WD lines ---
			for wd_val in wds:
				sub = data[
					(data["OuterFold"] == fold_val) &
					(data["WD"] == wd_val) &
					(data["DR"] == dr_val)
				].sort_values(by="LR")

				if sub.empty: continue

				m = sub[metric_mean_col].values
				s = sub[metric_std_col].values
				n = sub["n"].clip(lower=1).values
				band = (1.6 * (s / np.sqrt(n))) if use_ci else s
				xx = sub["LR"].values

				ax.plot(xx, m, marker="o", linewidth=2, label=f"WD = {wd_val:.0e}")
				ax.fill_between(xx, m - band, m + band, alpha=0.15)

			# --- 5. Cosmetics for each subplot ---
			ax.grid(True, which="both", axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

			ax.set_xticks(lrs)
			ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])

			if metric_base.endswith("Acc") or metric_base == "AUC":
				ax.set_yticks(np.arange(0.45, 1.06, 0.10))
				ax.set_ylim(bottom=0.45, top=1.05)
				#ax.yaxis.set_major_locator(MultipleLocator(0.10))

			# Set titles and labels on the outer edges
			if row_idx == 0:
				ax.set_title(f"Fold {fold_val}", fontsize=10)
			if row_idx == n_rows - 1:
				ax.set_xlabel("Learning rate")
			if col_idx == 0:
				# --- MODIFIED: Y-label now shows DR ---
				ax.set_ylabel(f"Dropout rate = {dr_val}")
			#else: ax.set_ylabel("\n ")

	# --- 6. Final figure-level adjustments (MODIFIED LEGEND) ---
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, fontsize=9,
			   loc="upper center", ncol=len(handles),
			   bbox_to_anchor=(0.5, 1.04))
	if metric_base.startswith("Val"):metric_base = metric_base.replace("Val","Validation ")
	if metric_base.endswith("Acc"):metric_base = metric_base.replace("Acc","Accuracy")
	fig.suptitle(f"Change in {metric_base}", fontsize=12, y=1.06)

	_save_fig(fig, save_path)

def plot_lr_dr_sweeps_by_wd_and_fold(
	df ,
	metric_base  = "AUC",        # "AUC", "ValLoss", "ValAcc"
	use_ci  = False,            # if True, shade mean ± 1.96*SEM; else mean ± std
	figsize_per_panel  = 4.5,  # width per subplot
	save_path  = None):
	# --- 1. Validate inputs ---
	metric_mean_col = f"{metric_base}_mean"
	metric_std_col = f"{metric_base}_std"
	required = {"OuterFold", "WD", "DR", "LR", metric_mean_col, metric_std_col, "n"}
	if not required.issubset(df.columns):
		raise ValueError(f"DataFrame missing required columns: {required - set(df.columns)}")

	data = df.copy()

	# --- 2. Get unique values for facets and lines ---
	outer_folds = sorted(data["OuterFold"].unique())
	wds = sorted(data["WD"].unique(), reverse=True) # Show 1e-4 at the bottom
	drs = sorted(data["DR"].unique())
	lrs = sorted(data["LR"].unique())

	n_rows, n_cols = len(wds), len(outer_folds)
	#n_cols, n_rows = len(wds), len(outer_folds)

	# --- 3. Create the subplot grid ---
	fig, axes = plt.subplots(n_rows, n_cols,
							 figsize=(figsize_per_panel * n_cols, 3.8 * n_rows),
							 sharex=True, sharey=True,
							 constrained_layout=True)
	if n_cols == 1 and n_rows == 1: axes = np.array([[axes]])
	elif n_rows == 1: axes = np.expand_dims(axes, axis=0)
	elif n_cols == 1: axes = np.expand_dims(axes, axis=1)

	# --- 4. Iterate through grid and plot ---
	for row_idx, wd_val in enumerate(wds):
		for col_idx, fold_val in enumerate(outer_folds):
			ax = axes[row_idx, col_idx]

			# Inner loop to plot one line per Dropout Rate
			for dr_val in drs:
				sub = data[
					(data["OuterFold"] == fold_val) &
					(data["WD"] == wd_val) &
					(data["DR"] == dr_val)
				].sort_values(by="LR")

				if sub.empty: continue

				# Get data for plotting
				m = sub[metric_mean_col].values
				s = sub[metric_std_col].values
				n = sub["n"].clip(lower=1).values
				band = (1.96 * (s / np.sqrt(n))) if use_ci else s
				xx = sub["LR"].values

				# Plot line and shaded variance band
				ax.plot(xx, m, marker="o", linewidth=2, label=f"DR = {dr_val}")
				ax.fill_between(xx, m - band, m + band, alpha=0.15)

			# --- 5. Cosmetics for each subplot ---
			ax.grid(True, which="both", axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
			#ax.set_xscale("log")

			# Set specific x-ticks for the sweep values
			ax.set_xticks(lrs)
			ax.set_xticklabels([f"{lr:.0e}" for lr in lrs])

			if metric_base.endswith("Acc") or metric_base == "AUC":
				ax.set_ylim(bottom=0.5, top=1.05)
				ax.yaxis.set_major_locator(MultipleLocator(0.1))

			# Set titles and labels only on the outer edges of the grid
			if row_idx == 0:
				ax.set_title(f"Outer fold {fold_val}")
			if row_idx == n_rows - 1:
				ax.set_xlabel("Learning rate")
			if col_idx == 0:
				ax.set_ylabel(f"Weight decay = {wd_val:.0e}\n{metric_base}")

	# --- 6. Final figure-level adjustments ---
	handles, labels = ax.get_legend_handles_labels()
	fig.legend(handles, labels, title="Dropout Rate", bbox_to_anchor=(1.01, 0.95), loc='upper left')
	fig.suptitle(f"Change in {metric_base} vs. Learning Rate", fontsize=14, y=1.02)

	_save_fig(fig, save_path)

def plot_metric_vs_param_by_outerfold(
	df ,
	param= "LR",                 # one of {"LR","WD","DR"}
	metric_base = "AUC",          # "AUC", "ValLoss", "ValAcc"
	fixed = None,         # e.g., {"WD":1e-6, "DR":0.3} to isolate an LR sweep
	use_ci = False,              # if True, shade mean ± 1.96*SEM; else mean ± std
	figsize_per_panel = 4.2,    # width per subplot
	save_path = None
):

	# --- validate inputs ---
	metric_mean_col = f"{metric_base}_mean"
	metric_std_col  = f"{metric_base}_std"
	required = {"OuterFold", param, metric_mean_col, metric_std_col, "n"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"DataFrame missing columns: {missing}")

	# --- filter to sweep (hold others fixed if requested) ---
	data = df.copy()
	if fixed:
		for k, v in fixed.items():
			if k not in data.columns:
				raise ValueError(f"Fixed key '{k}' not in columns.")
			data = data.loc[np.isclose(data[k].astype(float), float(v)) if data[k].dtype!=object else data[k].eq(v)]
		if data.empty:
			raise ValueError("No rows left after applying 'fixed' constraints. Check values.")

	# --- outer folds present ---
	outer_folds = sorted(data["OuterFold"].unique())
	nF = len(outer_folds)
	if nF == 0:
		raise ValueError("No outer folds found after filtering.")

	# --- layout ---
	fig, axes = plt.subplots(1, nF, figsize=(figsize_per_panel*nF, 3.8), sharey=True,
						  constrained_layout=True)
	if nF == 1: axes = np.array([axes])

	# --- x scaling: LR/WD often benefit from log axis ---
	x = param
	log_x = (x in {"LR","WD"})
	x_label = {"LR":"Learning rate", "WD":"Weight decay", "DR":"Dropout rate"}.get(x, x)


	# --- per outer fold subplot ---
	# --- per outer fold subplot ---
	for j, (ax, of) in enumerate(zip(axes, outer_folds)):
		sub = data[data["OuterFold"] == of].copy()
		sub = sub.sort_values(by=x, key=lambda s: s.astype(float))

		# compute band
		m = sub[metric_mean_col].astype(float).values
		s = sub[metric_std_col].astype(float).values
		n = sub["n"].clip(lower=1).astype(int).values
		band = (1.96 * (s/np.sqrt(n))) if use_ci else s

		xx = sub[x].astype(float).values

		# plot line + shaded variance
		ax.plot(xx, m, marker="o", linewidth=2)
		ax.fill_between(xx, m - band, m + band, alpha=0.18)

		# cosmetics
		ax.set_title(f"Outer Fold {of}")
		ax.grid(alpha=0.3)

		ax.set_xlim(0.5, 1.0)
		ax.set_xlabel(x_label)

		if j == 0:ax.set_ylabel(metric_base.replace("Val","Validation "))
		if log_x:
			ax.set_xscale("log")
			ax.xaxis.set_major_locator(LogLocator(base=10))
			ax.xaxis.set_major_formatter(ScalarFormatter())
		else:
			ax.xaxis.set_major_locator(MaxNLocator(integer=False))
		# ensure sensible y ticks
		if metric_base.endswith("Acc") or metric_base=="AUC": ax.set_ylim(0.5, 1.0)

	# --- super title and fixed-text annotation ---

	title = f"Change in {metric_base} as {x_label} varies"
	fig.suptitle(title, fontsize=12)

	if fixed:
		fixed_txt = "  |  ".join([f"{k}={v if k=='DR' else f'{float(v):.0e}'.replace('e-0','e-').replace('e+0','e+')}"
								  for k,v in fixed.items()])
		fig.text(0.5, 0.98, f"Sweep: {x_label}    |    Fixed: {fixed_txt}",
				 ha="center", va="center", fontsize=10, color="dimgray")



	_save_fig(fig, save_path=save_path)

def _fmt_lr(x):
	"""Nice scientific formatting for LR/WD labels."""
	try:
		return (f"{float(x):.0e}"
				.replace("e-0","e-")
				.replace("e+0","e+"))
	except Exception:
		return str(x)

def _aggregate_grid(df, value, equal_weight_outer=False):
	"""
	Build a tidy table with columns: WD, DR, LR, mean_value, n
	value in {"AUC_at_Best", "LR_ratio"} where LR_ratio = LR_at_Best / LR.
	"""
	work = df.copy()
	# Coerce required cols
	need = ["OuterFold","InnerFold","WD","DR","LR","LR_at_Best","AUC_at_Best"]
	for c in need:
		work[c] = pd.to_numeric(work[c], errors="coerce")
	work = work.dropna(subset=need)

	if value == "LR_ratio":
		work["LR_ratio"] = work["LR_at_Best"] / work["LR"].replace(0, np.nan)
		metric_col = "LR_ratio"
	else:
		metric_col = "AUC_at_Best"

	if equal_weight_outer:
		# mean per OuterFold first, then mean across outers
		per_outer = (work.groupby(["OuterFold","WD","DR","LR"], as_index=False)
						  .agg(v=(metric_col,"mean"), n=("AUC_at_Best","count")))
		agg = (per_outer.groupby(["WD","DR","LR"], as_index=False)
						.agg(mean_value=("v","mean"),
							 n=("n","sum")))
	else:
		agg = (work.groupby(["WD","DR","LR"], as_index=False)
					.agg(mean_value=(metric_col,"mean"),
						 n=("AUC_at_Best","count")))
	# Sorting for tidy axes
	agg = agg.sort_values(["WD","DR","LR"]).reset_index(drop=True)
	return agg

def facet_heatmap_by_wd(inner_summary ,
						value = "AUC_at_Best",      # or "ValAcc_at_Best" or "LR_ratio"
						equal_weight_outer  = False,
						annotate  = True,
						cmap = "YlGnBu",
						save_path = None):
	"""
	Facet heatmaps by WD:
	  - columns = LR (categorical)
	  - rows    = DR (categorical)
	  - colour  = mean of 'value' per (LR, DR) averaged across folds
	"""
	df = inner_summary.copy()

	tbl = _aggregate_grid(df, value=value, equal_weight_outer=equal_weight_outer)

	# Axis categories
	lrs = np.sort(tbl["LR"].unique())
	drs = np.sort(tbl["DR"].unique())
	wds = np.sort(tbl["WD"].unique())
	n_panels = len(wds)

	#fig, axes = plt.subplots(1, n_panels, figsize=(4.8*n_panels, 4.6), sharey=True)
	fig, axes = plt.subplots(1, n_panels, figsize=(4.8*n_panels, 4.6),
							 sharex=False, sharey=True, constrained_layout=True)
	if n_panels == 1:
		axes = np.array([axes])

	# Global color scaling (use [0.5,1] for LR_ratio; else auto)
	v = tbl["mean_value"].values
	if value == "LR_ratio":
		vmin, vmax = 0.5, max(1.0, np.nanmax(v))
	else:
		vmin, vmax = np.nanmin(v), np.nanmax(v)
	norm = Normalize(vmin=vmin, vmax=vmax)

	last_im = None
	for ax, wd in zip(axes, wds):
		sub = tbl[tbl["WD"] == wd]
		pivot = sub.pivot(index="DR", columns="LR", values="mean_value").reindex(index=drs, columns=lrs)
		n_mat = sub.pivot(index="DR", columns="LR", values="n").reindex(index=drs, columns=lrs)

		im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
		last_im = im

		ax.set_xticks(np.arange(len(lrs))); ax.set_yticks(np.arange(len(drs)))
		ax.set_xticklabels([_fmt_lr(x) for x in lrs], rotation=45, ha="right")
		ax.set_yticklabels([f"{y:g}" for y in drs])
		ax.set_xlabel("Learning Rate (LR)")
		if ax is axes[0]:
			ax.set_ylabel("Dropout Rate (DR)")
		ax.set_title(f"WD = {_fmt_lr(wd)}")

		if annotate:
			data = pivot.values; counts = n_mat.values
			for i in range(len(drs)):
				for j in range(len(lrs)):
					val = data[i, j]
					if np.isnan(val): continue
					txt = f"{val:.3f}"
					txt += f"\n(n={int(counts[i,j])})"
					ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

		ax.grid(False)

	label_lookup = {
		"AUC_at_Best": "Mean AUC",
		"ValAcc_at_Best": "Mean Accuracy",
		"LR_ratio": "Mean Learning Rate Ratio"
	}

	cbar = fig.colorbar(last_im, ax=axes.ravel().tolist())
	cbar.set_label(label_lookup.get(value, f"Mean {value}"))

	# Suptitle
	sup_lookup = {
		"AUC_at_Best": "Mean AUC at Best Performance Across Folds, by Weight Decay",
		"ValAcc_at_Best": "Mean Accuracy at Best Performance across folds, by Weight Decay",
		"LR_ratio": "Mean Learning Rate Ratio at Best Performance, by Weight Decay" # — <1 implies best after LR decay
	}
	fig.suptitle(sup_lookup.get(value, value), fontsize=12)

	_save_fig(fig, save_path=save_path)

def _aggregate_grid_per_OuterFold(df , value ):
	"""
	Build a tidy table grouped by OuterFold, WD, DR, and LR.
	The value column can be "AUC_at_Best", "ValAcc_at_Best", or "LR_ratio".
	"""
	work = df.copy()

	# Define required columns based on the selected metric
	metric_map = {
		"AUC_at_Best": "AUC_at_Best",
		"ValAcc_at_Best": "ValAcc_at_Best",
		"LR_ratio": "LR_ratio"
	}
	if value not in metric_map:
		raise ValueError(f"Value '{value}' is not a valid metric.")

	metric_col = metric_map[value]

	# Determine all columns needed for calculation and grouping
	required_cols = ["OuterFold", "InnerFold", "WD", "DR", "LR"]
	if value == "LR_ratio":
		required_cols.extend(["LR_at_Best"])
	else:
		required_cols.append(metric_col)

	for col in required_cols:
		if col in work.columns:
			work[col] = pd.to_numeric(work[col], errors="coerce")
	work = work.dropna(subset=required_cols)

	if value == "LR_ratio":
		work["LR_ratio"] = work["LR_at_Best"] / work["LR"].replace(0, np.nan)

	# Group by each OuterFold without averaging them together
	agg = (work.groupby(["OuterFold", "WD", "DR", "LR"], as_index=False)
			   .agg(
				   mean_value=(metric_col, "mean"),
				   n=(metric_col, "count")
			   ))

	agg = agg.sort_values(["OuterFold", "WD", "DR", "LR"]).reset_index(drop=True)
	return agg

def facet_heatmap_by_wd_per_OuterFold(inner_summary ,
									  value = "AUC_at_Best",
									  cmap = "YlGnBu",
									  save_path = None):
	cmap = cm.get_cmap(cmap)
	bright_cmap = colors.LinearSegmentedColormap.from_list(
	f"bright_{cmap.name}",
	cmap(np.linspace(0.2, 0.9, 256))
	)
	df = inner_summary.copy()
	tbl = _aggregate_grid_per_OuterFold(df, value=value)

	# Get unique values for grid dimensions and axes
	folds = np.sort(tbl["OuterFold"].unique())
	wds = np.sort(tbl["WD"].unique())
	lrs = np.sort(tbl["LR"].unique())
	drs = np.sort(tbl["DR"].unique())

	n_rows, n_cols = len(folds), len(wds)

	fig, axes = plt.subplots(n_rows, n_cols,
							 figsize=(2.8 * n_cols + 1, 3 * n_rows),
							 sharex=False, sharey=True,
							 constrained_layout=True)

	# Ensure axes is always a 2D array for consistent indexing
	if n_rows == 1 and n_cols == 1:axes = np.array([[axes]])
	elif n_rows == 1:axes = np.expand_dims(axes, axis=0)
	elif n_cols == 1:axes = np.expand_dims(axes, axis=1)

	# Global color scaling
	v = tbl["mean_value"].values
	vmin, vmax = (np.nanmin(v), np.nanmax(v))
	if value == "LR_ratio": vmin, vmax = 0.5, max(1.0, np.nanmax(v) if not np.all(np.isnan(v)) else 1.0)
	norm = Normalize(vmin=vmin, vmax=vmax)
	label_lookup = {"AUC_at_Best": "Mean AUC", "ValAcc_at_Best": "Mean Accuracy", "LR_ratio": "Mean LR Ratio"}
	cbar_label = label_lookup.get(value, f"Mean {value}")
	for row_idx, fold in enumerate(folds):
		images_in_row = []
		for col_idx, wd in enumerate(wds):
			ax = axes[row_idx, col_idx]

			sub = tbl[(tbl["OuterFold"] == fold) & (tbl["WD"] == wd)]
			pivot = sub.pivot(index="DR", columns="LR", values="mean_value").reindex(index=drs, columns=lrs)
			n_mat = sub.pivot(index="DR", columns="LR", values="n").reindex(index=drs, columns=lrs)

			im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap=bright_cmap, norm=norm)
			images_in_row.append(im)

			if row_idx == 0:ax.set_title(f"WD = {_fmt_lr(wd)}", fontsize=10)
			if col_idx == 0:ax.set_ylabel(f"Fold {int(fold)}\n\nDropout rate (DR)")
			if row_idx == n_rows - 1: ax.set_xlabel("Learning rate (LR)\n")

			# Configure tick labels
			ax.set_xticks(np.arange(len(lrs)))
			ax.set_yticks(np.arange(len(drs)))
			ax.set_xticklabels([_fmt_lr(x) for x in lrs], rotation=45, ha="right")
			ax.set_yticklabels([f"{y:g}" for y in drs])

			data, counts = pivot.values, n_mat.values
			for i, j in np.ndindex(data.shape):
				val = data[i, j]
				if np.isnan(val): continue
				count_val = counts[i, j]
				if np.isnan(count_val): continue
				txt = f"{val:.3f}"
				ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
			ax.grid(False)
		cbar = fig.colorbar(images_in_row[0], ax=axes[row_idx, :].tolist(), pad=0.02)
		cbar.set_label(cbar_label)

	# Add a main title for the entire figure
	sup_lookup = {
		"AUC_at_Best": "Mean AUC at Best Performance",
		"ValAcc_at_Best": "Mean Accuracy at Best Performance",
		"LR_ratio": "Mean Learning Rate Ratio"
	}
	fig.suptitle(f"{sup_lookup.get(value, value)}, by Weight Decay", fontsize=12)

	# This assumes _save_fig and _fmt_lr are defined elsewhere
	_save_fig(fig, save_path=save_path)

def plot_hp_heatmap(inner_summary , WD  = 1e-4, save_path = None,
					auc_col = "AUC_at_Best", lr_col = "LR", dr_col = "DR",
					wd_col = "WD", round_lr = 6, round_dr = 4, tol  = 1e-12):
	"""
	Heatmap of Hyperparameter Interactions (LR × DR) with cell colour = mean AUC_at_Best,
	computed from inner_summary rows at a fixed WD.

	Parameters
	----------
	inner_summary
		Must contain columns: ['LR','DR','WD','AUC_at_Best'] (others are ignored here).
	WD  , default=1e-4
		Weight decay value to isolate (rows are filtered where WD ~= this, within 'tol').
	save_path  or None
		If provided, saves the figure to this path. Otherwise, displays it.
	auc_col, lr_col, dr_col, wd_col
		Column names in the provided dataframe.
	round_lr, round_dr : int
		Decimal rounding applied to LR and DR to stabilise grouping against float noise.
	tol
		Absolute tolerance for matching WD (np.isclose).

	Returns
	-------
	pd.DataFrame
		The pivot table (rows: DR, cols: LR) of mean AUC values used for plotting.
	"""
	# --- coerce dtypes (robust against mixed types) ---
	for c in [auc_col, lr_col, dr_col, wd_col]:
		if c in inner_summary:
			inner_summary[c] = pd.to_numeric(inner_summary[c], errors="coerce")
	df = inner_summary.dropna(subset=[auc_col, lr_col, dr_col, wd_col]).copy()

	# --- filter the requested WD (with tolerance) ---
	df = df[np.isclose(df[wd_col].values, float(WD), rtol=0.0, atol=tol)]
	if df.empty:
		raise ValueError(f"No rows found at WD={WD} (±{tol}).")

	# --- stabilise float keys (rounding) for grouping/pivoting ---
	df["_LR_r"] = df[lr_col].round(round_lr)
	df["_DR_r"] = df[dr_col].round(round_dr)

	# --- aggregate: mean AUC + count per (LR, DR) ---
	#grp = (df.groupby(["_LR_r", "_DR_r"], as_index=False)
	#		 .agg(mean_auc=(auc_col, "mean"),
	#			  n=(auc_col, "count")))


	# first: mean per (OuterFold, LR, DR)
	df1 = (df.groupby(["OuterFold", "_LR_r", "_DR_r"], as_index=False)
			 .agg(of_mean_auc=(auc_col, "mean")))
	# then: mean across outer folds
	grp = (df1.groupby(["_LR_r", "_DR_r"], as_index=False)
			  .agg(mean_auc=("of_mean_auc","mean"),
				   n=("of_mean_auc","count")))

	# sort axes numerically
	lrs = np.sort(grp["_LR_r"].unique())
	drs = np.sort(grp["_DR_r"].unique())

	# pivot to matrix (rows=DR, cols=LR)
	pivot_mean = grp.pivot(index="_DR_r", columns="_LR_r", values="mean_auc").reindex(index=drs, columns=lrs)
	pivot_n    = grp.pivot(index="_DR_r", columns="_LR_r", values="n"       ).reindex(index=drs, columns=lrs)

	# --- plot ---
	fig, ax = plt.subplots(figsize=(1.2*len(lrs)+2.5, 1.0*len(drs)+2.5))
	im = ax.imshow(pivot_mean.values, origin="lower", aspect="auto")  # default colormap

	# ticks & labels
	def fmt_lr(x):
		# scientific formatting without awkward e+00
		s = f"{x:.0e}".replace("e-0","e-").replace("e+0","e+")
		return s
	ax.set_xticks(np.arange(len(lrs)))
	ax.set_yticks(np.arange(len(drs)))
	ax.set_xticklabels([fmt_lr(x) for x in lrs], rotation=45, ha="right")
	ax.set_yticklabels([f"{y:g}" for y in drs])

	ax.set_xlabel("Learning Rate (LR)")
	ax.set_ylabel("Dropout Rate (DR)")
	title_wd = f"{fmt_lr(WD) if WD>0 else WD}"
	ax.set_title(f"Hyperparameter Interaction Heatmap (fixed WD={title_wd})\nCell colour: mean AUC_at_Best")

	# annotate cells with mean (and n if available)
	data = pivot_mean.values
	nmat = pivot_n.values
	for i in range(len(drs)):
		for j in range(len(lrs)):
			val = data[i, j]
			if np.isnan(val):
				# optionally show a small dot or leave blank
				continue
			ax.text(j, i, f"{val:.3f}\n(n={int(nmat[i,j])})",
					ha="center", va="center", fontsize=9, color="white",
					bbox=dict(boxstyle="round,pad=0.2", facecolor=(0,0,0,0.25), edgecolor="none"))

	# colorbar
	cbar = fig.colorbar(im, ax=ax, shrink=0.9)
	cbar.set_label("Mean AUC_at_Best")

	fig.tight_layout()
	_save_fig(fig, save_path=save_path)

	# return pivot for logging/reporting
	pivot_mean.index.name = "DR"
	pivot_mean.columns.name = "LR"
	return pivot_mean

def plot_inner_losses(df, outer_fold, param_map, param_name, use_ema=False, save_path=None):
	"""
	For the specified outer_fold, plot Train vs Validation loss curves for each inner fold.
	Legend shows only the chosen parameter (e.g., LR/WD/DR) using HPset -> param_map.
	"""
	# robust dtypes
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["TrainLoss", "ValLoss", "EMA_ValLoss"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	target_hpsets = list(map(int, param_map.keys()))
	val_col = "EMA_ValLoss" if use_ema and "EMA_ValLoss" in df.columns else "ValLoss"

	sub = df[
		(df["OuterFold"] == int(outer_fold)) &
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["InnerFold", "HPset", "Epoch", "TrainLoss", val_col])
	if sub.empty:
		raise ValueError(f"No rows found for outer fold {outer_fold} and HPsets {target_hpsets}")

	long = sub.melt(
		id_vars=["InnerFold", "HPset", "Epoch"],
		value_vars=["TrainLoss", val_col],
		var_name="Split", value_name="Loss"
	).replace({val_col: "ValLoss"})

	g = (long.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
			  .agg(mean=("Loss","mean"), std=("Loss","std"), count=("Loss","count")))
	g["ci95"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))

	inner_folds = sorted(g["InnerFold"].dropna().unique().astype(int))
	n_panels = len(inner_folds)

	# consistent colours by parameter value (keep order of your HP sets)
	prms_in_order = [param_map[h] for h in target_hpsets]
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}

	style_map = {"TrainLoss": "-", "ValLoss": "--"}

	fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4), sharex=True, sharey=True)
	if n_panels == 1: axes = [axes]
	max_epoch = int(sub["Epoch"].max()) if len(sub) else 30

	for inner, ax in zip(inner_folds, axes):
		gi = g[g["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for split, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split], color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
		ax.set_title(f"Inner Fold {inner}")
		ax.set_xlabel("Epoch")
		ax.set_xticks(range(int(long["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	axes[0].set_ylabel("Loss")
	#HP = ("LR", "WD: 1e-4, DR: 0.2")
	HP = param_name[0]
	others = param_name[1]

	# legend: ONLY the parameter values
	prm_handles = [Line2D([0],[0], color=prm_to_color[prm], lw=2, label=f"{HP}={prm:g}")
				   for prm in prms_in_order]
	fig.legend(prm_handles, [h.get_label() for h in prm_handles],
			   loc="upper center", ncol=len(prm_handles), bbox_to_anchor=(0.5, 1))

	fig.text(0.5, 1.015, "solid: training, dashed: validation", ha="center", va="bottom", fontsize=11)
	fig.suptitle(f"Outer Fold {outer_fold}: Training vs Validation Loss ({others})", y=1.12)
	plt.tight_layout()
	_save_fig(fig, save_path=save_path)

def plot_inner_accuracies(df, outer_fold, param_map, param_name, save_path=None):
	"""
	For the specified outer_fold, plot Train vs Validation accuracy curves for each inner fold.
	Legend shows only the chosen parameter using HPset -> param_map.
	"""
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["TrainAcc", "ValAcc"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	target_hpsets = list(map(int, param_map.keys()))
	sub = df[
		(df["OuterFold"] == int(outer_fold)) &
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["InnerFold", "HPset", "Epoch", "TrainAcc", "ValAcc"])
	if sub.empty:
		raise ValueError(f"No rows found for outer fold {outer_fold} and HPsets {target_hpsets}")

	long = sub.melt(
		id_vars=["InnerFold", "HPset", "Epoch"],
		value_vars=["TrainAcc", "ValAcc"],
		var_name="Split", value_name="Acc"
	)

	g = (long.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
			  .agg(mean=("Acc","mean"), std=("Acc","std"), count=("Acc","count")))
	g["ci95"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))

	inner_folds = sorted(g["InnerFold"].dropna().unique().astype(int))
	n_panels = len(inner_folds)

	prms_in_order = [param_map[h] for h in target_hpsets]
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}

	style_map = {"TrainAcc": "-", "ValAcc": "--"}

	fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4), sharex=True, sharey=True)
	if n_panels == 1: axes = [axes]
	max_epoch = int(sub["Epoch"].max()) if len(sub) else 30

	for inner, ax in zip(inner_folds, axes):
		gi = g[g["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for split, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split], color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
		ax.set_title(f"Inner Fold {inner}")
		ax.set_xlabel("Epoch")
		ax.set_xticks(range(int(long["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	axes[0].set_ylabel("Accuracy")

	HP = param_name[0]
	others = param_name[1]

	prm_handles = [Line2D([0],[0], color=prm_to_color[prm], lw=2, label=f"{HP}={prm:g}")
				   for prm in prms_in_order]
	fig.legend(prm_handles, [h.get_label() for h in prm_handles],
			   loc="upper center", ncol=len(prm_handles), bbox_to_anchor=(0.5, 1))

	fig.text(0.5, 1.015, "solid: training, dashed: validation", ha="center", va="bottom", fontsize=11)
	fig.suptitle(f"Outer Fold {outer_fold}: Training vs Validation Accuracy ({others})", y=1.12)
	plt.tight_layout()
	_save_fig(fig, save_path=save_path)

def plot_inner_AUCs(df, outer_fold, param_map, param_name, save_path=None):
	"""
	For the specified outer_fold, plot Validation AUC per epoch for each inner fold.
	Legend shows only the chosen parameter using HPset -> param_map.
	"""
	# dtypes
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["AUC"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	target_hpsets = list(map(int, param_map.keys()))
	sub = df[
		(df["OuterFold"] == int(outer_fold)) &
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["InnerFold", "HPset", "Epoch", "AUC"])
	if sub.empty:
		raise ValueError(f"No rows found for outer fold {outer_fold} and HPsets {target_hpsets}")

	# keep pipeline uniform without melt()
	long = sub[["InnerFold", "HPset", "Epoch", "AUC"]].copy()
	long["Split"] = "AUC"   # single series, just for consistent grouping

	g = (long.groupby(["InnerFold", "HPset", "Epoch", "Split"], as_index=False)
			  .agg(mean=("AUC", "mean"), std=("AUC", "std"), count=("AUC", "count")))
	g["ci95"] = 1.96 * g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))

	inner_folds = sorted(g["InnerFold"].dropna().unique().astype(int))
	n_panels = len(inner_folds)

	# colours by parameter value (order follows your HP sets)
	prms_in_order = [param_map[h] for h in target_hpsets]
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}

	fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4), sharex=True, sharey=True)
	if n_panels == 1:
		axes = [axes]
	max_epoch = int(sub["Epoch"].max()) if len(sub) else 30

	for inner, ax in zip(inner_folds, axes):
		gi = g[g["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			# single split "AUC"
			for _, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle="-", color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"] - d_s["ci95"], d_s["mean"] + d_s["ci95"],
								color=color, alpha=0.15)

		ax.set_title(f"Inner Fold {inner}")
		ax.set_xlabel("Epoch")
		ax.set_xticks(range(int(long["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	axes[0].set_ylabel("AUC")

	HP = param_name[0]
	others = param_name[1]

	# legend: ONLY the parameter values
	prm_handles = [Line2D([0],[0], color=prm_to_color[prm], lw=2, label=f"{HP}={prm:g}")
				   for prm in prms_in_order]
	fig.legend(prm_handles, [h.get_label() for h in prm_handles],
			   loc="upper center", ncol=len(prm_handles), bbox_to_anchor=(0.5, 1))

	fig.suptitle(f"Outer Fold {outer_fold}: Validation AUC ({others})", y=1.12)
	plt.tight_layout()
	_save_fig(fig, save_path=save_path)

def plot_inner_loss_AUCs(df, outer_fold, param_map, param_name, use_ema=False, save_path=None):
	"""
	One figure, 2 rows x N inner folds:
	  Row 1: Train vs Validation Loss (solid=train, dashed=validation)
	  Row 2: Validation AUC
	Legend shows only the chosen parameter values using HPset -> param_map.
	"""
	# ---- robust dtypes ----
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["TrainLoss", "ValLoss", "EMA_ValLoss", "AUC"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	#target_hpsets = list(map(int, param_map.keys()))
	target_hpsets = [int(k) for k in param_map.keys()]
	val_loss_col  = "EMA_ValLoss" if use_ema and "EMA_ValLoss" in df.columns else "ValLoss"

	sub = df[
		(df["OuterFold"] == int(outer_fold)) &
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["InnerFold", "HPset", "Epoch", "TrainLoss", val_loss_col, "AUC"])
	if sub.empty:
		raise ValueError(f"No rows for outer fold {outer_fold} and HPsets {target_hpsets}")

	# ---- LOSS long format & aggregate ----
	long_loss = sub.melt(
		id_vars=["InnerFold", "HPset", "Epoch"],
		value_vars=["TrainLoss", val_loss_col],
		var_name="Split", value_name="Loss"
	).replace({val_loss_col: "ValLoss"})

	g_loss = (long_loss.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
						.agg(mean=("Loss","mean"), std=("Loss","std"), count=("Loss","count")))
	g_loss["ci95"] = 1.96 * g_loss["std"].fillna(0) / np.sqrt(g_loss["count"].clip(lower=1))

	# ---- AUC aggregate (no melt; avoid name clash) ----
	long_auc = sub[["InnerFold","HPset","Epoch","AUC"]].copy()
	long_auc["Split"] = "AUC"
	g_auc = (long_auc.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
					  .agg(mean=("AUC","mean"), std=("AUC","std"), count=("AUC","count")))
	g_auc["ci95"] = 1.96 * g_auc["std"].fillna(0) / np.sqrt(g_auc["count"].clip(lower=1))

	inner_folds = sorted(g_loss["InnerFold"].dropna().unique().astype(int))
	n_panels = len(inner_folds)

	# ---- colours keyed by parameter value (stable across rows) ----
	prms_in_order = [param_map[h] for h in target_hpsets]
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}

	style_map = {"TrainLoss": "-", "ValLoss": "--"}  # row 1 only

	# ---- figure: 2 rows × N columns ----
	fig, axes = plt.subplots(2, n_panels, figsize=(5*n_panels, 8), sharex=False)
	if n_panels == 1:
		axes = np.array([[axes[0]], [axes[1]]])  # make 2×1 array

	max_epoch = int(sub["Epoch"].max()) if len(sub) else 30

	# ----- Row 1: LOSS -----
	for j, inner in enumerate(inner_folds):
		ax = axes[0, j]
		gi = g_loss[g_loss["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for split, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split], color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
		ax.set_title(f"Inner Fold {inner}")
		ax.set_ylabel("Loss")
		ax.set_xticks(range(int(long_loss["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	# ----- Row 2: AUC -----
	for j, inner in enumerate(inner_folds):
		ax = axes[1, j]
		gi = g_auc[g_auc["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for _, d_s in d_h.groupby("Split"):  # single "AUC"
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle="-", color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("AUC")
		ax.set_xticks(range(int(long_loss["Epoch"].min()), max_epoch+1, 2))
		ax.grid(alpha=0.3)

	# ---- legend & titles ----
	HP = param_name[0] if isinstance(param_name, (list, tuple)) else str(param_name)
	others = (param_name[1] if isinstance(param_name, (list, tuple)) and len(param_name) > 1 else "").strip()
	#fmt = format(prm, ".0e").replace("e-0", "e-").replace("e+0", "e+")
	handles = [
		Line2D([0],[0],
			   color=prm_to_color[prm], lw=2,
			   label=f"{HP}={format(prm, '.0e').replace('e-0','e-').replace('e+0','e+')}")
		for prm in prms_in_order
	]
	fig.legend(handles, [h.get_label() for h in handles],
			   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.92),  #0.965
			   #title=f"{others}"
			   )

	#fig.text(0.5, 0.90, "row 1: solid=train, dashed=validation • row 2: AUC",
	#		 ha="center", va="top", fontsize=11)

	#fig.suptitle(f"Outer Fold {outer_fold}: Loss (top) and AUC (bottom){others}", y=0.995)
	fig.suptitle(f"Outer Fold {outer_fold} | Learning Rates ", fontsize=13, y=0.995)

	fig.text(0.5, 0.95,       #0.965,, 0.89
		 f"Top row: solid=training loss, dashed=validation loss   |   Bottom row: AUC   |   Hyperparams: {others}",
		 ha="center", va="center", fontsize=10, color="dimgray")
	plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))  # leave space for legend/supertitle

	_save_fig(fig, save_path=save_path)

def _fmt_lr(x):
	return format(float(x), ".0e").replace("e-0","e-").replace("e+0","e+")

def plot_inner_loss_AUCs_LRchanges(df, outer_fold, param_map, param_name, use_ema=False, save_path=None):
	"""
	One figure, 2 rows x N inner folds:
	  Row 1: Train vs Validation Loss (solid=train, dashed=validation)
	  Row 2: Validation AUC
	Legend shows only the chosen parameter values using HPset -> param_map.
	Also marks points where LR changes and labels them with the new LR value.
	"""
	# ---- robust dtypes ----
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["TrainLoss", "ValLoss", "EMA_ValLoss", "AUC", "LR"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	target_hpsets = [int(k) for k in param_map.keys()]
	val_loss_col  = "EMA_ValLoss" if use_ema and "EMA_ValLoss" in df.columns else "ValLoss"

	sub = df[
		(df["OuterFold"] == int(outer_fold)) &
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["InnerFold", "HPset", "Epoch", "TrainLoss", val_loss_col, "AUC", "LR"])
	if sub.empty:
		raise ValueError(f"No rows for outer fold {outer_fold} and HPsets {target_hpsets}")

	# ---- LOSS long format & aggregate ----
	long_loss = sub.melt(
		id_vars=["InnerFold", "HPset", "Epoch"],
		value_vars=["TrainLoss", val_loss_col],
		var_name="Split", value_name="Loss"
	).replace({val_loss_col: "ValLoss"})

	g_loss = (long_loss.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
						.agg(mean=("Loss","mean"), std=("Loss","std"), count=("Loss","count")))
	g_loss["ci95"] = 1.96 * g_loss["std"].fillna(0) / np.sqrt(g_loss["count"].clip(lower=1))

	# ---- AUC aggregate ----
	long_auc = sub[["InnerFold","HPset","Epoch","AUC"]].copy()
	long_auc["Split"] = "AUC"
	g_auc = (long_auc.groupby(["InnerFold","HPset","Epoch","Split"], as_index=False)
					  .agg(mean=("AUC","mean"), std=("AUC","std"), count=("AUC","count")))
	g_auc["ci95"] = 1.96 * g_auc["std"].fillna(0) / np.sqrt(g_auc["count"].clip(lower=1))

	# ★ NEW: detect LR change points per (InnerFold, HPset)
	# We call a "change" when LR != LR.shift() inside each group. We keep the new LR and the epoch it starts.
	sub_sorted = sub.sort_values(["InnerFold","HPset","Epoch"])
	def _lr_changes(group):
		s = group["LR"]
		changed = s.ne(s.shift()) & s.notna() & s.shift().notna()
		out = group.loc[changed, ["InnerFold","HPset","Epoch","LR"]].drop_duplicates(["Epoch"])
		return out
	lr_changes = (sub_sorted.groupby(["InnerFold","HPset"], as_index=False, group_keys=False)
							 .apply(_lr_changes))
	# If there are duplicates across repeated logging, dedupe by epoch
	lr_changes = lr_changes.drop_duplicates(["InnerFold","HPset","Epoch"])

	inner_folds = sorted(g_loss["InnerFold"].dropna().unique().astype(int))
	n_panels = len(inner_folds)

	# ---- colours keyed by parameter value (stable across rows) ----
	prms_in_order = [param_map[h] for h in target_hpsets]
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}

	style_map = {"TrainLoss": "-", "ValLoss": "--"}  # row 1 only

	# ---- figure: 2 rows × N columns ----
	fig, axes = plt.subplots(2, n_panels, figsize=(5*n_panels, 8), sharex=False)
	if n_panels == 1:
		axes = np.array([[axes[0]], [axes[1]]])  # make 2×1 array

	max_epoch = int(sub["Epoch"].max()) if len(sub) else 30
	min_epoch = int(sub["Epoch"].min()) if len(sub) else 0
	xticks = range(min_epoch, max_epoch+1, max(1, (max_epoch-min_epoch)//6 or 1))

	# ----- Row 1: LOSS -----
	for j, inner in enumerate(inner_folds):
		ax = axes[0, j]
		gi = g_loss[g_loss["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for split, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split], color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
			# ★ NEW: mark LR change points on ValLoss curve
			changes_h = lr_changes[(lr_changes["InnerFold"]==inner) & (lr_changes["HPset"]==hp_id)]
			if not changes_h.empty:
				# get ValLoss means at those epochs for marker y-coordinates
				valloss_at_e = (d_h[d_h["Split"]=="ValLoss"]
								.set_index("Epoch")["mean"])
				for _, r in changes_h.iterrows():
					e = int(r["Epoch"])
					if e in valloss_at_e.index:
						y = float(valloss_at_e.loc[e])
						ax.scatter([e],[y], marker="D", s=40, color=color, edgecolor="k", zorder=5)
						ax.annotate(f"LR={_fmt_lr(r['LR'])}", (e, y),
									textcoords="offset points", xytext=(4,4),
									fontsize=8, color="black")

		ax.set_title(f"Inner Fold {inner}")
		ax.set_ylabel("Loss")
		ax.set_xticks(xticks)
		ax.grid(alpha=0.3)

	# ----- Row 2: AUC -----
	for j, inner in enumerate(inner_folds):
		ax = axes[1, j]
		gi = g_auc[g_auc["InnerFold"] == inner]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for _, d_s in d_h.groupby("Split"):  # single "AUC"
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle="-", color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
			# ★ NEW: mark LR change points on AUC curve
			changes_h = lr_changes[(lr_changes["InnerFold"]==inner) & (lr_changes["HPset"]==hp_id)]
			if not changes_h.empty:
				auc_at_e = d_h.set_index("Epoch")["mean"]  # only "AUC" split exists
				for _, r in changes_h.iterrows():
					e = int(r["Epoch"])
					if e in auc_at_e.index:
						y = float(auc_at_e.loc[e])
						ax.scatter([e],[y], marker="D", s=40, color=color, edgecolor="k", zorder=5)
						ax.annotate(f"LR={_fmt_lr(r['LR'])}", (e, y),
									textcoords="offset points", xytext=(4,4),
									fontsize=8, color="black")

		ax.set_xlabel("Epoch")
		ax.set_ylabel("AUC")
		ax.set_xticks(xticks)
		ax.grid(alpha=0.3)

	# ---- legend & titles ----
	HP = param_name[0] if isinstance(param_name, (list, tuple)) else str(param_name)
	others = (param_name[1] if isinstance(param_name, (list, tuple)) and len(param_name) > 1 else "").strip()
	handles = [
		Line2D([0],[0],
			   color=prm_to_color[prm], lw=2,
			   label=f"{HP}={_fmt_lr(prm)}")
		for prm in prms_in_order
	]
	fig.legend(handles, [h.get_label() for h in handles],
			   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.92))

	fig.suptitle(f"Outer Fold {outer_fold} | Learning Rates", fontsize=13, y=0.995)
	fig.text(0.5, 0.95,
			 f"Top: solid=training loss, dashed=validation loss  |  Bottom: AUC  |  Hyperparams: {others}",
			 ha="center", va="center", fontsize=10, color="dimgray")

	plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
	_save_fig(fig, save_path=save_path)

def plot_outer_loss_AUCs(df, param_map, param_name, use_auc=True, save_path=None):
	"""
	One figure, 2 rows x N outer folds:
	  Row 1: Train vs Validation Loss (solid=train, dashed=validation)
	  Row 2: Validation AUC  (default)  OR  Train vs Validation Accuracy if use_auc=False

	Legend shows only the chosen parameter values using HPset -> param_map.
	df must contain columns:
	  OuterFold, InnerFold, HPset, Epoch, TrainLoss, ValLoss, [AUC] and/or [TrainAcc, ValAcc]
	"""
	# ---- dtypes (robust) ----
	int_cols   = ["OuterFold", "InnerFold", "HPset", "Epoch"]
	float_cols = ["TrainLoss", "ValLoss", "AUC", "TrainAcc", "ValAcc"]
	for c in int_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
	for c in float_cols:
		if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

	target_hpsets = [int(k) for k in param_map.keys()]

	# required columns for the chosen bottom-row metric
	bottom_cols = ["AUC"] if use_auc else ["TrainAcc", "ValAcc"]

	sub = df[
		(df["HPset"].isin(target_hpsets))
	].dropna(subset=["OuterFold", "InnerFold", "HPset", "Epoch",
					 "TrainLoss", "ValLoss"] + bottom_cols)
	if sub.empty:
		raise ValueError(f"No rows found for HPsets {target_hpsets} with required columns.")

	# ---- aggregate across INNER folds within each OUTER fold ----
	def agg_ci(g, value_col):
		out = (g.groupby(["OuterFold","HPset","Epoch"], as_index=False)
				 .agg(mean=(value_col,"mean"),
					  std=(value_col,"std"),
					  count=(value_col,"count")))
		out["ci95"] = 1.96 * out["std"].fillna(0) / np.sqrt(out["count"].clip(lower=1))
		return out

	g_train_loss = agg_ci(sub, "TrainLoss")
	g_val_loss   = agg_ci(sub, "ValLoss")

	if use_auc:
		g_auc = agg_ci(sub, "AUC")
	else:
		g_train_acc = agg_ci(sub, "TrainAcc")
		g_val_acc   = agg_ci(sub, "ValAcc")

	# combine and mark split (for linestyle)
	g_loss = pd.concat(
		[g_train_loss.assign(Split="Train"), g_val_loss.assign(Split="Val")],
		ignore_index=True
	)

	outer_folds = sorted(g_loss["OuterFold"].dropna().unique().astype(int))
	n_panels = len(outer_folds)

	# ---- colours keyed by parameter value (stable across rows) ----
	prms_in_order = [param_map[h] for h in target_hpsets]
	tab10 = mpl.colormaps["tab10"].colors
	prm_to_color = {prm: tab10[i % len(tab10)] for i, prm in enumerate(prms_in_order)}
	style_map = {"Train": "-", "Val": "--"}  # solid=train, dashed=val

	# ---- figure: 2 rows × N columns ----
	fig, axes = plt.subplots(2, n_panels, figsize=(5*n_panels, 7), sharex=False)
	if n_panels == 1:
		axes = np.array([[axes[0]], [axes[1]]])

	min_epoch = int(g_loss["Epoch"].min())
	max_epoch = int(g_loss["Epoch"].max())
	step = max(1, (max_epoch - min_epoch) // 5)

	# ----- Row 1: LOSS -----
	for j, outer in enumerate(outer_folds):
		ax = axes[0, j]
		gi = g_loss[g_loss["OuterFold"] == outer]
		for hp_id, d_h in gi.groupby("HPset"):
			prm_val = param_map[int(hp_id)]
			color = prm_to_color[prm_val]
			for split, d_s in d_h.groupby("Split"):
				ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split],
						color=color, linewidth=2)
				ax.fill_between(d_s["Epoch"],
								d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
								color=color, alpha=0.15)
		ax.set_title(f"Outer Fold {outer}")
		ax.set_ylabel("Loss")
		ax.set_xticks(range(min_epoch, max_epoch+1, step))
		ax.grid(alpha=0.3)

	# ----- Row 2: bottom metric -----
	for j, outer in enumerate(outer_folds):
		ax = axes[1, j]
		if use_auc:
			gi = g_auc[g_auc["OuterFold"] == outer]
			for hp_id, d_h in gi.groupby("HPset"):
				prm_val = param_map[int(hp_id)]
				color = prm_to_color[prm_val]
				ax.plot(d_h["Epoch"], d_h["mean"], linestyle="-", color=color, linewidth=2)
				ax.fill_between(d_h["Epoch"], d_h["mean"]-d_h["ci95"], d_h["mean"]+d_h["ci95"],
								color=color, alpha=0.15)
			ax.set_ylabel("AUC")
			ax.set_ylim(0.5, 1.0)
		else:
			# accuracy: train vs val with styles
			g_acc = pd.concat(
				[g_train_acc.assign(Split="Train"), g_val_acc.assign(Split="Val")],
				ignore_index=True
			)
			gi = g_acc[g_acc["OuterFold"] == outer]
			for hp_id, d_h in gi.groupby("HPset"):
				prm_val = param_map[int(hp_id)]
				color = prm_to_color[prm_val]
				for split, d_s in d_h.groupby("Split"):
					ax.plot(d_s["Epoch"], d_s["mean"], linestyle=style_map[split],
							color=color, linewidth=2)
					ax.fill_between(d_s["Epoch"], d_s["mean"]-d_s["ci95"], d_s["mean"]+d_s["ci95"],
									color=color, alpha=0.15)
			ax.set_ylabel("Accuracy")
			ax.set_ylim(0.5, 1.0)

		ax.set_xlabel("Epoch")
		ax.set_xticks(range(min_epoch, max_epoch+1, step))
		ax.grid(alpha=0.3)

	# ---- legend & titles (same “system”) ----
	HP = param_name[0] if isinstance(param_name, (list, tuple)) else str(param_name)
	others = (param_name[1] if isinstance(param_name, (list, tuple)) and len(param_name) > 1 else "").strip()

	handles = [
		Line2D([0],[0],
			   color=prm_to_color[prm], lw=2,
			   label=f"{HP}={format(prm, '.0e').replace('e-0','e-').replace('e+0','e+')}")
		for prm in prms_in_order
	]
	fig.legend(handles, [h.get_label() for h in handles],
			   loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.91))

	bottom_name = "AUC" if use_auc else "Accuracy"
	fig.suptitle(f"MultiViewCNN: Loss (top) and {bottom_name} (bottom)", fontsize=13, y=0.995)
	fig.text(0.5, 0.94,
			 ("Top row: solid=training loss, dashed=validation loss   |   " +
			  (f"Bottom row: validation AUC   |   " if use_auc
			   else "Bottom row: solid=training accuracy, dashed=validation accuracy   |   ") +
			  others),
			 ha="center", va="center", fontsize=10, color="dimgray")

	plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
	_save_fig(fig, save_path=save_path)

def plot_fold_history(history: pd.DataFrame, out: int, hpset: int,
					  save_path: str = None, include_ema: bool = False):
	"""
	Plot TrainLoss, ValLoss (and EMA_ValLoss), AUC, and Brier across epochs
	for a specific (OuterFold, HPset) from the OUTER history CSV.

	Parameters
	----------
	history : pd.DataFrame
		DataFrame with columns:
		["Timestamp","Model","OuterFold","HPset","Epoch","TrainLoss","TrainAcc",
		 "ValLoss","ValAcc","AUC","Brier","EMA_ValLoss","LR","NoImprove",
		 "LrDrop","EsTriggered","BestValLoss","BestEpoch"]
	out : int
		OuterFold index to filter.
	hpset : int
		HPset id to filter.
	save_path : str, optional
		If provided, save the figure here (PNG/PDF, etc.).
	include_ema : bool, optional
		If True and column exists, overlay EMA_ValLoss on the ValLoss panel.

	Returns
	-------
	fig, axes, df_sel : (matplotlib.figure.Figure, np.ndarray, pd.DataFrame)
		Handles to the figure, axes array (2x2), and the filtered/sorted frame.
	"""
	# --- filter ---
	df_sel = history[(history["OuterFold"] == out) & (history["HP"] == hpset)].copy()
	if df_sel.empty:
		raise ValueError(f"No rows found for OuterFold={out}, HPset={hpset}")

	# --- coerce numerics just in case ---
	num_cols = ["Epoch","TrainLoss","ValLoss","AUC","Brier","EMA_ValLoss",
				"LR","NoImprove","LrDrop","EsTriggered","BestValLoss","BestEpoch"]
	for c in num_cols:
		if c in df_sel.columns:
			df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
	df_sel = df_sel.sort_values("Epoch")

	# --- best epoch (prefer logged BestEpoch from last row, else argmin ValLoss) ---
	if "BestEpoch" in df_sel.columns and df_sel["BestEpoch"].notna().any():
		best_epoch = int(df_sel["BestEpoch"].dropna().iloc[-1])
	else:
		best_epoch = int(df_sel.loc[df_sel["ValLoss"].idxmin(), "Epoch"])
	lr_drops = df_sel.loc[df_sel.get("LrDrop", pd.Series([], dtype=float)) == 1, "Epoch"].tolist()

	# --- labels/metadata for title ---
	model = df_sel["Model"].iloc[0] if "Model" in df_sel.columns else ""
	lr = df_sel["LR"].iloc[0] if "LR" in df_sel.columns else None

	fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
	max_epoch = int(50)
	min_epoch = int(0)
	xticks = range(min_epoch, max_epoch+1, 5)
	# 1) TrainLoss
	ax = axes[0, 0]
	ax.plot(df_sel["Epoch"], df_sel["TrainLoss"], lw=2)
	ax.set_title("Train Loss (BCE)")
	ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
	ax.set_xticks(xticks)


	# 2) ValLoss (+ EMA if available)
	ax = axes[0, 1]
	ax.plot(df_sel["Epoch"], df_sel["ValLoss"], lw=2, label="ValLoss")
	if include_ema and "EMA_ValLoss" in df_sel.columns and df_sel["EMA_ValLoss"].notna().any():
		ax.plot(df_sel["Epoch"], df_sel["EMA_ValLoss"], lw=2, ls="--", label="EMA ValLoss")
	ax.set_title("Validation Loss")
	ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_xticks(xticks)
	ax.legend(loc="best")

	# 3) AUC
	ax = axes[1, 0]
	ax.plot(df_sel["Epoch"], df_sel["AUC"], lw=2)
	ax.set_title("Validation AUC")
	ax.set_xlabel("Epoch"); ax.set_ylabel("AUC"); ax.set_ylim(0, 1); ax.set_xticks(xticks)

	# 4) Brier
	ax = axes[1, 1]
	ax.plot(df_sel["Epoch"], df_sel["Brier"], lw=2)
	ax.set_title("Validation Brier score")
	ax.set_xlabel("Epoch"); ax.set_ylabel("Brier"); ax.set_xticks(xticks)

	# --- annotate best epoch + LR drops across all panels ---
	for ax in axes.ravel():
		ax.axvline(best_epoch, color="k", ls="--", alpha=0.35, label=None)
		if lr_drops:
			for e in lr_drops:
				ax.axvline(e, color="C3", ls=":", alpha=0.35)

	supt = f"{model} | Fold {out}, HPset {hpset}"
	if lr is not None and not np.isnan(lr):
		supt += f", LR={lr:g}"
	fig.suptitle(supt, fontsize=12)

	_save_fig(fig, save_path=save_path)
	#return fig, axes, df_sel

#fig, axes, df = plot_fold_history(history, out=2, hpset=11, save_path=None)
#plt.show()
#print(df.head(5))



def add_metrics_at_best(df_logs: pd.DataFrame, inner_summary: pd.DataFrame):
	"""
	For each row in inner_summary, fetch metrics from df_logs at its BestEpoch:
	  Adds columns: ValAcc_at_Best, TrainAcc_at_Best, TrainLoss_at_Best, Timestamp_at_Best

	Keys used to join: (OuterFold, InnerFold, HPset, Epoch=BestEpoch)

	df_logs must contain (at least):
	  ['OuterFold','InnerFold','HPset','Epoch','ValAcc','TrainAcc','TrainLoss'] and optionally 'Timestamp'

	inner_summary must contain:
	  ['OuterFold','InnerFold','HPset','BestEpoch']
	"""
	logs = df_logs.copy()
	summ = inner_summary.copy()

	# ---- Coerce dtypes for safety ----
	for c in ["OuterFold", "InnerFold", "HPset", "Epoch"]:
		if c in logs: logs[c] = pd.to_numeric(logs[c], errors="coerce")
	for c in ["ValAcc", "TrainAcc", "TrainLoss"]:
		if c in logs: logs[c] = pd.to_numeric(logs[c], errors="coerce")


	for c in ["OuterFold", "InnerFold", "HPset", "BestEpoch"]:
		if c in summ: summ[c] = pd.to_numeric(summ[c], errors="coerce")

	# ---- Drop rows with missing keys ----
	logs = logs.dropna(subset=["OuterFold","InnerFold","HPset","Epoch"])
	summ = summ.dropna(subset=["OuterFold","InnerFold","HPset","BestEpoch"])

	# ---- Deduplicate logs at the (folds, HPset, epoch) level ----
	key_cols = ["OuterFold","InnerFold","HPset","Epoch"]
	metric_cols = [c for c in ["ValAcc","TrainAcc","TrainLoss","Timestamp"] if c in logs.columns]

	if "Timestamp" in logs.columns and not logs["Timestamp"].isna().all():
		# Keep the last record by Timestamp within each (folds,HPset,epoch)
		logs = logs.sort_values(key_cols + ["Timestamp"])
		logs = logs.drop_duplicates(subset=key_cols, keep="last")
		# No aggregation needed; we now have one row per key
		logs_compact = logs[key_cols + metric_cols]
	else:
		# Aggregate duplicates; take mean for numeric metrics, keep NA for timestamp
		agg_map = {m: "mean" for m in metric_cols if m != "Timestamp"}
		logs_compact = (logs.groupby(key_cols, as_index=False)
							.agg(**{m: (m, agg_map[m]) for m in agg_map}))
		# attach a null Timestamp column so downstream merge has a consistent schema
		if "Timestamp" in metric_cols and "Timestamp" not in logs_compact.columns:
			logs_compact["Timestamp"] = pd.NaT

	# ---- Prepare lookup rows: rename BestEpoch -> Epoch to join on logs ----
	lookups = summ[["OuterFold","InnerFold","HPset","BestEpoch"]].rename(columns={"BestEpoch":"Epoch"})

	# ---- Merge once, then rename outputs ----
	merged = lookups.merge(
		logs_compact,
		on=["OuterFold","InnerFold","HPset","Epoch"],
		how="left",
		validate="m:1"  # each summary row should map to <=1 log row
	).rename(columns={
		"ValAcc":   "ValAcc_at_Best",
		"TrainAcc": "TrainAcc_at_Best",
		"TrainLoss":"TrainLoss_at_Best",
		"Timestamp":"Timestamp_at_Best",
	})

	# ---- Attach back to inner_summary on the original keys ----
	out = summ.merge(
		merged.rename(columns={"Epoch":"BestEpoch"}),
		on=["OuterFold","InnerFold","HPset","BestEpoch"],
		how="left",
		validate="1:1"
	)

	# ---- Sanity report (optional) ----
	miss = out["ValAcc_at_Best"].isna().sum()
	if miss:
		print(f"[add_metrics_at_best] Note: {miss} rows had no matching log record at BestEpoch.")

	return out