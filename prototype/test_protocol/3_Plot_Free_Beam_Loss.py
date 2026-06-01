"""Plot Free Beam Loss / Gain (Protocol 3)
=====================================

This script plots the results produced by the manual Protocol 3 capture
(`3_Free_Beam_Loss.py`).

It supports two common workflows:

1) Single frequency/signal (one CSV):
   - Plot gain vs angle.

2) Sweep across frequencies (multiple CSVs):
   - Plot gain vs frequency with one curve per angle.

Notes
-----
- Protocol 3 CSV filenames often do not encode the frequency. For the multi-CSV
  workflow, provide frequencies explicitly via --freqs, or ensure your filenames
  contain a recognizable "500Hz" / "2kHz" pattern.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


PLOT_TITLE_SIZE = 16
PLOT_LABEL_SIZE = 13
PLOT_TICK_SIZE = 12
PLOT_LEGEND_SIZE = 12
PLOT_ANNOTATION_SIZE = 11
PLOT_CONE_LABEL_SIZE = 13


@dataclass(frozen=True)
class _CsvSeries:
	path: Path
	freq_hz: float | None
	df: pd.DataFrame


def _parse_freq_from_text(text: str) -> float | None:
	"""Best-effort frequency parser from a filename/label.

	Recognizes patterns like:
	- 500Hz, 500 Hz
	- 2kHz, 2 kHz
	- 2.5kHz
	"""
	if not text:
		return None

	lowered = text.lower()
	# Prefer kHz first to avoid matching "2" in "2kHz" as 2 Hz.
	m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*k\s*hz", lowered)
	if m:
		return float(m.group("val")) * 1000.0
	m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*hz", lowered)
	if m:
		return float(m.group("val"))
	return None


def _load_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if df.empty:
		raise ValueError(f"CSV is empty: {path}")
	return df


def _detect_angle_column(df: pd.DataFrame, preferred: str | None = None) -> str:
	if preferred:
		if preferred not in df.columns:
			raise ValueError(f"Requested angle column '{preferred}' not present in CSV")
		return preferred
	for candidate in ("expected_angle", "angle_deg", "angle"):
		if candidate in df.columns:
			return candidate
	raise ValueError("No angle column found; expected one of: expected_angle, angle_deg, angle")


def _detect_gain_column(df: pd.DataFrame, preferred: str | None = None) -> str:
	if preferred:
		if preferred not in df.columns:
			raise ValueError(f"Requested gain column '{preferred}' not present in CSV")
		return preferred
	for candidate in ("gain_db", "gain", "rms_dbfs"):
		if candidate in df.columns:
			return candidate
	raise ValueError("No gain column found; expected one of: gain_db, gain, rms_dbfs")


def _as_float_series(series: pd.Series) -> np.ndarray:
	return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def _close_xy_trace(angles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Append the first sample to the end so the XY trace closes on itself.

	This makes the circular measurement layout visible in a regular x-y plot.
	"""
	if angles.size == 0 or values.size == 0:
		return angles, values
	return np.concatenate([angles, angles[:1]]), np.concatenate([values, values[:1]])


def _mirror_signed_seam(angles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Mirror the signed seam so the 180° value also appears at -180° when needed.

	This is only used for the right-hand cartesian plot.
	"""
	angles = np.asarray(angles, dtype=float)
	values = np.asarray(values, dtype=float)
	if angles.size == 0 or values.size == 0:
		return angles, values

	left_idx = np.where(np.isclose(angles, -180.0))[0]
	right_idx = np.where(np.isclose(angles, 180.0))[0]
	x_parts = [angles]
	y_parts = [values]

	if right_idx.size > 0 and left_idx.size == 0:
		x_parts.insert(0, np.array([-180.0], dtype=float))
		y_parts.insert(0, values[right_idx[-1] : right_idx[-1] + 1])
	elif left_idx.size > 0 and right_idx.size == 0:
		x_parts.append(np.array([180.0], dtype=float))
		y_parts.append(values[left_idx[0] : left_idx[0] + 1])

	return np.concatenate(x_parts), np.concatenate(y_parts)


def _wrap_signed_angles(angles: np.ndarray) -> np.ndarray:
	"""Wrap angles into the signed $[-180, 180)$ display range."""
	angles = np.asarray(angles, dtype=float)
	return ((angles + 180.0) % 360.0) - 180.0


def _add_polar_cone_overlay(ax: plt.Axes, *, half_width_deg: float = 25.0,
							cone_color: str = "tab:green", alpha: float = 0.12,
							label: str = "focus region") -> None:
	"""Draw a translucent cone centered on 0° that spans +/- half_width_deg.

	The patch is rendered in polar data coordinates so it follows the current
	theta configuration of the axis. It is intentionally subtle and sits behind
	the measured curve.
	"""
	if half_width_deg <= 0:
		return

	r_min, r_max = ax.get_ylim()
	if not np.isfinite(r_min) or not np.isfinite(r_max):
		return
	if r_max <= r_min:
		return

	theta = np.deg2rad(np.linspace(-float(half_width_deg), float(half_width_deg), 128))
	ax.fill_between(
		theta,
		r_min,
		r_max,
		color=cone_color,
		alpha=alpha,
		zorder=0,
		label=label,
	)

	# Add an in-plot tag so the highlighted region is named directly on the chart.
	r_mid = r_min + 0.72 * (r_max - r_min)
	ax.text(
		np.deg2rad(0.0),
		r_mid,
		label,
		ha="center",
		va="center",
		fontsize=PLOT_CONE_LABEL_SIZE,
		fontweight="bold",
		color=cone_color,
		bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=cone_color, alpha=0.75),
		zorder=3,
	)
	# Keep the overlay behind the data line and preserve the radial scale.
	ax.set_ylim(r_min, r_max)


def _prepare_series(paths: list[Path], freqs_hz: list[float | None] | None) -> list[_CsvSeries]:
	if freqs_hz is not None and len(freqs_hz) != len(paths):
		raise ValueError("--freqs must have the same count as CSV files")

	series_list: list[_CsvSeries] = []
	for idx, path in enumerate(paths):
		df = _load_csv(path)
		freq = None
		if freqs_hz is not None:
			freq = freqs_hz[idx]
		if freq is None:
			freq = _parse_freq_from_text(path.stem)
		series_list.append(_CsvSeries(path=path, freq_hz=freq, df=df))
	return series_list


def plot_gain_vs_angle(
	csv_path: Path,
	*,
	angle_col: str | None = None,
	gain_col: str | None = None,
	doa_col: str | None = None,
	split_plots: bool = False,
	polar_min: float | None = None,
	show_cone: bool = True,
	cone_half_width_deg: float = 25.0,
	title: str | None = None,
	real_angles: list[float] | None = None,
	save_path: Path | None = None,
	show: bool = True,
):
	df = _load_csv(csv_path)
	angle_col = _detect_angle_column(df, angle_col)
	gain_col = _detect_gain_column(df, gain_col)
	doa_col = doa_col if doa_col is not None else ("doa_deg" if "doa_deg" in df.columns else None)

	angles = _as_float_series(df[angle_col])
	gains = _as_float_series(df[gain_col])
	order = np.argsort(angles)
	angles = angles[order]
	gains = gains[order]
	doa = _as_float_series(df[doa_col])[order] if doa_col is not None and doa_col in df.columns else None

	freq = _parse_freq_from_text(csv_path.stem)
	default_title = "Free beam gain vs angle"
	if freq is not None:
		default_title += f" ({freq:.0f} Hz)"

	if split_plots:
		fig = plt.figure(figsize=(12, 5.5))
		ax_gain = fig.add_subplot(1, 2, 1, projection="polar")
		ax_error = fig.add_subplot(1, 2, 2)

		# Close polar trace so 345->0 is connected visually. Use closed arrays
		# so the polar plot wraps correctly.
		angles_closed, gains_closed = _close_xy_trace(angles, gains)
		angles_rad = np.deg2rad(angles_closed)
		ax_gain.plot(angles_rad, gains_closed, marker="o", linewidth=2)
		# Apply optional radial minimum (zoom control)
		if polar_min is not None:
			try:
				ax_gain.set_ylim(bottom=float(polar_min))
			except Exception:
				# Fallback for older matplotlib versions
				ax_gain.set_rmin(float(polar_min))
		# Annotate polar title with the maximum output level in dBFS (preferred),
		# falling back to the relative gain if no absolute dBFS columns are present.
		max_output_dbfs = None
		for col in ("rms_dbfs_avgch", "rms_dbfs", "peak_dbfs", "input_rms_dbfs_avgch", "input_rms_dbfs"):
			if col in df.columns:
				vals = _as_float_series(df[col])
				finite = vals[np.isfinite(vals)]
				if finite.size:
					max_output_dbfs = float(np.nanmax(finite))
					break
		if max_output_dbfs is not None and np.isfinite(max_output_dbfs):
			ax_gain.set_title(f"Gain (polar) [{max_output_dbfs:.1f} dBFS]", fontsize=PLOT_TITLE_SIZE)
		else:
			# Fallback: show the maximum relative gain (dB) if dBFS isn't available
			try:
				max_rel_gain = float(np.nanmax(gains_closed))
			except Exception:
				max_rel_gain = None
			if max_rel_gain is None or not np.isfinite(max_rel_gain):
				ax_gain.set_title("Gain (polar)", fontsize=PLOT_TITLE_SIZE)
			else:
				ax_gain.set_title(f"Gain (polar) [{max_rel_gain:.1f} dB]", fontsize=PLOT_TITLE_SIZE)
		ax_gain.set_theta_zero_location("N")
		ax_gain.set_theta_direction(-1)
		ax_gain.grid(True, alpha=0.3)
		ax_gain.set_rlabel_position(135)
		ax_gain.tick_params(labelsize=PLOT_TICK_SIZE)
		
		# Set radial ticks to 5dB increments
		try:
			data_min_db = float(np.floor(np.nanmin(gains_closed)))
			if not np.isfinite(data_min_db):
				data_min_db = -20.0
			# Round down to nearest 5 dB
			db_floor = int(5 * np.floor(data_min_db / 5.0))
			# Create ticks from db_floor to just below 0
			db_ticks = np.arange(db_floor, 1, 5)
			if len(db_ticks) > 0:
				# Set radial limits and ticks
				ax_gain.set_ylim((db_floor, 0))
				ax_gain.set_yticks(db_ticks)
				ax_gain.yaxis.set_major_formatter(FormatStrFormatter('%d'))
		except Exception:
			pass  # Silently ignore if setting ticks fails
		
		ax_gain.grid(True, alpha=0.3)
		if show_cone:
			_add_polar_cone_overlay(
				ax_gain,
				half_width_deg=cone_half_width_deg,
				cone_color="tab:green",
				alpha=0.12,
				label=f"Focus ±{cone_half_width_deg:g}°",
			)

		if doa is None or not np.isfinite(doa).any():
			raise ValueError(
				"split_plots=True requires a DOA column (expected 'doa_deg' by default)"
			)

		# Cartesian summary: DOA error versus real angle.
		# Keep the right-hand plot centered on 0° by wrapping only the angles
		# above 180° into their negative counterparts for display.
		if real_angles is not None and len(real_angles) == len(angles):
			reference_angles = _wrap_signed_angles(np.asarray(real_angles, dtype=float))
			ref_label = "Real angle (reference)"
		else:
			reference_angles = _wrap_signed_angles(angles)
			ref_label = "Measured angle (reference)"

		error_angles_plot = _wrap_signed_angles(angles)
		error_deg = doa - reference_angles
		error_order = np.argsort(error_angles_plot)
		error_angles_plot = error_angles_plot[error_order]
		error_deg = error_deg[error_order]
		error_angles_plot, error_deg_plot = _mirror_signed_seam(error_angles_plot, error_deg)
		ax_error.axhline(0.0, color="0.35", linewidth=1.2, linestyle="--", zorder=1)
		# Highlight the focus region as a vertical band matching the polar cone.
		ax_error.axvspan(-float(cone_half_width_deg), float(cone_half_width_deg), color="tab:green", alpha=0.10, zorder=0, label=f"Focus Region ±{cone_half_width_deg:g}°")
		ax_error.plot(error_angles_plot, error_deg_plot, marker="o", linewidth=1.8, color="tab:orange", label="DOA error")
		ax_error.fill_between(error_angles_plot, 0.0, error_deg_plot, color="tab:orange", alpha=0.08)
		ax_error.set_xlabel("Signal angle (deg)", fontsize=PLOT_LABEL_SIZE)
		ax_error.set_ylabel("DOA error (deg)", fontsize=PLOT_LABEL_SIZE)
		ax_error.set_title("DOA vs real angle error", fontsize=PLOT_TITLE_SIZE)
		ax_error.grid(True, alpha=0.3)
		ax_error.tick_params(axis="both", labelsize=PLOT_TICK_SIZE)
		# Add a small horizontal padding so points near the edges are visible.
		pad_deg = 5.0
		ax_error.set_xlim(-180.0 - pad_deg, 180.0 + pad_deg)
		ax_error.set_xticks(np.arange(-180, 181, 45))
		finite_error = error_deg_plot[np.isfinite(error_deg_plot)]
		if finite_error.size:
			# Compute an automatic ylim but enforce a minimum span of ±20° so
			# small error datasets remain readable.
			y_max = max(5.0, float(np.nanmax(np.abs(finite_error))) * 1.15, 20.0)
		else:
			# No finite values -> fall back to a sensible default range.
			y_max = 20.0
		ax_error.set_ylim(-y_max, y_max)
		if ref_label:
			ax_error.text(
				0.02,
				0.98,
				ref_label,
				transform=ax_error.transAxes,
				va="top",
				ha="left",
				fontsize=PLOT_ANNOTATION_SIZE,
				bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.85),
			)
		ax_error.legend(loc="upper right", fontsize=PLOT_LEGEND_SIZE)
		fig.suptitle(title or default_title, fontsize=PLOT_TITLE_SIZE + 1)
	else:
		fig = plt.figure(figsize=(10, 5))
		ax = fig.add_subplot(111)
		ax.plot(angles, gains, marker="o", linewidth=2)
		ax.set_xlabel("Signal Angle", fontsize=PLOT_LABEL_SIZE)
		ax.set_xticks(angles, labels=angles.astype(int))
		ax.set_ylabel("Gain (dB)", fontsize=PLOT_LABEL_SIZE)
		ax.set_title(title or default_title, fontsize=PLOT_TITLE_SIZE)
		ax.grid(True, alpha=0.3)
		ax.set_xlim(float(np.nanmin(angles)), float(np.nanmax(angles)))

		if show_cone:
			# In non-split mode the cone is mainly a visual reference on the x-axis plot.
			# We only add it when the caller uses a polar-like display elsewhere.
			pass

		# Optional: visualize DOA estimate (if present).
		if doa is not None:
			if np.isfinite(doa).any():
				ax2 = ax.twinx()
				ax2.plot(angles, doa, color="tab:orange", linestyle="--", marker="x", alpha=0.7)
				ax2.axhline(0.0, color="0.75", linewidth=1.0, alpha=0.5, linestyle="-")
				ax2.set_ylabel("Estimated Angle (doa)")

	fig.tight_layout()

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=160)

	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_gain_vs_frequency_per_angle(
	csv_paths: list[Path],
	*,
	freqs_hz: list[float | None] | None = None,
	angle_col: str | None = None,
	gain_col: str | None = None,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
):
	series_list = _prepare_series(csv_paths, freqs_hz)
	if any(s.freq_hz is None for s in series_list):
		missing = [str(s.path) for s in series_list if s.freq_hz is None]
		raise ValueError(
			"Could not infer frequencies for one or more CSVs. "
			"Provide --freqs (e.g. '--freqs 500,750,1000') or encode like '500Hz' in filenames. "
			f"Missing: {missing}"
		)

	# Use the first CSV to pick column names.
	angle_col = _detect_angle_column(series_list[0].df, angle_col)
	gain_col = _detect_gain_column(series_list[0].df, gain_col)

	# Gather (freq, gain) points for each angle.
	angle_to_points: dict[float, list[tuple[float, float]]] = {}
	for s in series_list:
		freq = float(s.freq_hz)
		df = s.df
		if angle_col not in df.columns or gain_col not in df.columns:
			raise ValueError(
				f"CSV {s.path} missing required columns '{angle_col}' and/or '{gain_col}'."
			)

		angles = _as_float_series(df[angle_col])
		gains = _as_float_series(df[gain_col])
		for a, g in zip(angles, gains, strict=False):
			if not np.isfinite(a) or not np.isfinite(g):
				continue
			angle_key = float(a)
			angle_to_points.setdefault(angle_key, []).append((freq, float(g)))

	if not angle_to_points:
		raise ValueError("No valid (angle, gain) points found across the provided CSVs")

	# Sort angles for deterministic legend order.
	angles_sorted = sorted(angle_to_points.keys())

	fig = plt.figure(figsize=(11, 6))
	ax = fig.add_subplot(111)

	for angle in angles_sorted:
		points = angle_to_points[angle]
		points.sort(key=lambda p: p[0])
		freqs = np.array([p[0] for p in points], dtype=float)
		gains = np.array([p[1] for p in points], dtype=float)
		ax.plot(freqs, gains, marker="o", linewidth=2, label=f"{angle:g}°")

	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel(f"{gain_col} (dB)")
	ax.set_title(title or "Free beam gain vs frequency (one curve per angle)")
	ax.grid(True, alpha=0.3)
	ax.legend(title=f"{angle_col}", ncol=2, fontsize=9)
	ax.set_xscale("log")
	ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

	fig.tight_layout()

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=160)

	if show:
		plt.show()
	else:
		plt.close(fig)


def _parse_freq_list(raw: str) -> list[float]:
	values: list[float] = []
	for part in raw.split(","):
		text = part.strip()
		if not text:
			continue
		values.append(float(text))
	if not values:
		raise ValueError("--freqs cannot be empty")
	return values


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Plot Protocol 3 free beam gain outputs")
	parser.add_argument("csv_files", nargs="+", type=str, help="One or more Protocol 3 CSV files")
	parser.add_argument(
		"--mode",
		choices=("auto", "angle", "freq"),
		default="auto",
		help="auto: 1 CSV -> angle, N CSVs -> freq; angle: gain vs angle; freq: gain vs frequency per angle",
	)
	parser.add_argument(
		"--freqs",
		type=str,
		default=None,
		help="Comma-separated frequencies (Hz) corresponding to each CSV file in order (required if filenames have no Hz/kHz)",
	)
	parser.add_argument("--angle-col", type=str, default=None, help="Angle column to use (default: auto)")
	parser.add_argument("--gain-col", type=str, default=None, help="Gain column to use (default: gain_db)")
	parser.add_argument(
		"--split-plots",
		action="store_true",
		help="In angle mode, show gain on a polar plot and DOA on a separate x-y plot",
	)
	parser.add_argument(
		"--no-cone",
		action="store_true",
		help="Disable the translucent target cone overlay in angle mode",
	)
	parser.add_argument(
		"--cone-half-width",
		type=float,
		default=25.0,
		help="Half-width of the target cone overlay in degrees (default: 25)",
	)
	parser.add_argument(
		"--real-angles",
		type=str,
		default=None,
		help="Comma-separated real angle values for DOA plot (e.g., '0,-15,-25,-25,-25,0,25,25,25,15,0')",
	)
	parser.add_argument("--title", type=str, default=None, help="Optional plot title")
	parser.add_argument("--no-show", action="store_true", help="Do not show an interactive window")
	parser.add_argument("--no-save", action="store_true", help="Do not save the plot")
	parser.add_argument(
		"--out",
		type=str,
		default=None,
		help="Output PNG path (default: alongside first CSV)",
	)

	args = parser.parse_args(argv)

	csv_paths = [Path(p) for p in args.csv_files]
	for p in csv_paths:
		if not p.exists():
			raise FileNotFoundError(str(p))

	show = not bool(args.no_show)
	mode = args.mode
	if mode == "auto":
		mode = "angle" if len(csv_paths) == 1 else "freq"

	out_path: Path | None
	if args.no_save:
		out_path = None
	elif args.out is not None:
		out_path = Path(args.out)
	else:
		# Single file: use source CSV name; multiple files: use timestamped generic name
		if len(csv_paths) == 1:
			out_path = csv_paths[0].with_suffix(".png")
		else:
			timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
			out_path = csv_paths[0].parent / f"free_beam_loss_{timestamp}.png"

	freqs = _parse_freq_list(args.freqs) if args.freqs is not None else None

	if mode == "angle":
		if len(csv_paths) != 1:
			raise ValueError("mode=angle expects exactly 1 CSV")
		# Parse real angles if provided
		real_angles = None
		if args.real_angles is not None:
			try:
				real_angles = [float(x.strip()) for x in args.real_angles.split(",")]
			except ValueError:
				raise ValueError(f"Invalid real-angles format: {args.real_angles}")
		plot_gain_vs_angle(
			csv_paths[0],
			angle_col=args.angle_col,
			gain_col=args.gain_col,
			doa_col="doa_deg",
			split_plots=bool(args.split_plots),
			title=args.title,
			real_angles=real_angles,
			show_cone=not args.no_cone,
			cone_half_width_deg=float(args.cone_half_width),
			save_path=out_path,
			show=show,
		)
		return 0

	if mode == "freq":
		plot_gain_vs_frequency_per_angle(
			csv_paths,
			freqs_hz=(list(freqs) if freqs is not None else None),
			angle_col=args.angle_col,
			gain_col=args.gain_col,
			title=args.title,
			save_path=out_path,
			show=show,
		)
		return 0

	raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
	raise SystemExit(main())

