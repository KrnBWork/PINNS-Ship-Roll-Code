"""
===============================================================================
PHYSICS-INFORMED LSTM — REAL KRISO ROLL FITTING (REV M COMPARISON)
===============================================================================

Purpose
-------
Architecture-comparison revision of the supplied Rev M model. The Transformer
backbone has been replaced with a standard causal stacked LSTM. In this NoPINN
variant the physics-informed training terms are disabled while retaining:

    * the same Excel/data preparation and independent ship/wave clocks
    * the same wave alignment, feature engineering, and chronological splits
    * the same roll and roll-rate targets
    * the same diagnostic equation-of-motion utilities and coefficient storage
    * the same force decomposition, gates, calibration, and turning correction
    * the same non-PINN loss terms and active loss weightings
    * the same rolling forecast, full-wave ODE diagnostic, Bayesian search,
      logging, checkpoints, plots, metrics, and unit-check infrastructure

LSTM change
-----------
The numeric sequence is processed by a unidirectional ``nn.LSTM``. It is causal
by recurrence, so no attention masks or positional encoding are required. The
network preserves the original four-output interface used by every physics and
forecast component.

Diagnostic equation of motion
-----------------------------
The ODE diagnostic remains available but is not used as a training constraint:

    phi_t = v
    v_t + c1*v + c2*abs(v)*v + k1*phi = M_wave(history)

where:
    phi    = scaled roll angle
    v      = scaled roll rate
    c1     = linear damping
    c2     = quadratic damping
    k1     = linear restoring coefficient
    M_wave = measured-wave coupling + LSTM residual forcing

===============================================================================
"""

from __future__ import annotations

import argparse
import collections
import copy
import gc
import importlib.util
import itertools
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# A stale local torch/ directory in a project folder can shadow site-packages.
# Remove the script directory from import search only when it contains such a folder.
_SCRIPT_DIR = Path(__file__).resolve().parent
if (_SCRIPT_DIR / "torch").is_dir():
    sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != _SCRIPT_DIR]


def resolve_runtime_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    return (_SCRIPT_DIR / raw).resolve(strict=False)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _load_seed_sweep_helpers() -> Tuple[Tuple[int, ...], Callable[..., Dict[str, object]]]:
    helper_path = _SCRIPT_DIR / "seed_sweep_5pct.py"
    spec = importlib.util.spec_from_file_location("seed_sweep_5pct_helper", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load seed sweep helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.SEED_SWEEP_5PCT_SEEDS), module.run_seed_sweep_5pct


SEED_SWEEP_5PCT_SEEDS, run_seed_sweep_5pct = _load_seed_sweep_helpers()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:  # pragma: no cover - plotting is optional
    HAS_PLT = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG: Dict[str, object] = {
    # General
    "seed": 42,
    # metrics2.json Bayes trial 19 is the attached Transformer reference baseline.
    # These metrics are retained only for architecture comparison; each LSTM run recomputes
    # metrics from the current data and trained model.
    "baseline_run_id": "revm_metrics2_trial19_direct_lstm_no_ode",
    "baseline_run_metrics": {
        "fit_r2": 0.9616780572893926,
        "fit_rms_error_deg": 0.4424532898669103,
        "train_r2": 0.9722054011300854,
        "train_rms_error_deg": 0.3505198301681422,
        "validation_r2": 0.9271532844397941,
        "validation_rms_error_deg": 0.5942441098847692,
        "source_forecast_window_s": 9.439994810000002,
        "forecast_r2": 0.9680923279046043,
        "forecast_rms_error_deg": 0.5828017023253252,
        "forecast_phase_lag_s": 0.04000091000000339,
        "forecast_amplitude_ratio": 0.905803242306859,
        "forecast_extrema_relative_error": 0.04785167651386014,
        "full_ode_r2": -0.7955922662366366,
        "full_ode_rms_error_deg": 3.028639157685876,
        "best_epoch": 20,
        "best_metric": 0.9680923279046043,
        "c_roll": 0.3853868842124939,
        "c_quad": 9.492147364653647e-05,
        "k_roll": 4.861016273498535,
    },
    # Runtime/resource choices remain portable and are resolved on this host.
    # Only active Bayes-selected model/training values from metrics2 trial 19 seed searches.
    "device": "auto",                 # auto, cuda, or cpu
    "cuda_benchmark": True,
    "cuda_allow_tf32": True,
    "float32_matmul_precision": "high",
    # Runtime-only tuning for a host allocation of up to 32 vCPUs. These settings do not
    # alter batches, losses, optimiser steps, model precision, or Bayes quality.
    "vcpu_limit": 32,
    "torch_num_threads": 32,
    "torch_num_interop_threads": 1,   # avoid nested thread-pool oversubscription
    "dataloader_num_workers": 0,      # tensors are already resident in shared RAM
    "dataloader_prefetch_factor": 4,
    "dataloader_persistent_workers": True,
    "precompute_windows": True,
    "precompute_windows_cpu_only": True,
    "runtime_profile": "cpu_32vcpu",
    # Disable the legacy auto-profile because it changed batch size, rollout
    # frequency, and Bayes candidate-pool size, which can affect accuracy.
    "cpu_large_memory_auto": False,
    "cpu_large_memory_min_ram_gb": 48.0,
    "cpu_large_memory_min_batch_size": 128,
    "output_dir": "outputs_revm_lstm_comparison",
    "checkpoint_dir": "checkpoints_revm_lstm_comparison",
    "log_dir": "logs_revm_lstm_comparison",
    "save_checkpoints": True,
    # Disk-conscious Rev M policy: overwrite one lean best.pt checkpoint whenever
    # the hidden forecast R2 improves. No last, periodic, or final checkpoint is
    # written.
    "checkpoint_policy": "best_only",
    "save_best_checkpoint": True,
    "save_periodic_checkpoints": False,
    "checkpoint_every": 0,
    "save_last_checkpoint": False,
    "save_final_checkpoint": False,
    "final_checkpoint_name": "best.pt",
    "keep_best_state": True,
    "save_plots": True,

    # Pure-seas baseline: use the complete loaded workbook with no default
    # time-window truncation.
    "disable_data_time_window": True,
    "data_time_start_s": None,
    "data_time_end_s": None,
    "data_trim_final_s": 2.0,
    # Contiguous chronological split: training first, then 15% validation,
    # then 10% hidden post-training rolling forecast.
    "validation_window_s": None,
    "val_frac": 0.15,
    "forecast_frac": 0.10,
    "exploratory_assessment_dir": "exploratory_data_forecast_lengths_nopinn",
    "metrics_exclude_initial_s": 0.0,
    "forecast_window_s": None,         # optional seconds-based CLI override
    "forecast_start_s": None,          # optional explicit forecast start override
    "forecast_plot_context_s": 1.0,
    "forecast_uncertainty_enabled": True,
    "forecast_uncertainty_confidence": 0.95,
    "forecast_uncertainty_growth_power": 0.5,
    "forecast_uncertainty_floor_deg": 0.05,
    # Primary rolling forecast path. The direct LSTM forecast is the
    # current target while the recursive ODE remains available as a diagnostic.
    "forecast_use_ode": False,
    "direct_forecast_amplitude_calibration_enabled": True,
    "direct_forecast_amplitude_calibration_gain_min": 0.85,
    "direct_forecast_amplitude_calibration_gain_max": 1.25,
    "direct_forecast_amplitude_calibration_gain_steps": 41,
    "direct_forecast_amplitude_calibration_extrema_weight": 0.35,
    "forecast_force_replay_lag_min_s": -1.0,
    "forecast_force_replay_lag_max_s": 1.0,
    "forecast_force_replay_lag_steps": 41,
    "forecast_force_replay_gain_min": 0.4,
    "forecast_force_replay_gain_max": 1.4,
    "forecast_force_replay_gain_steps": 41,
    "forecast_coefficient_replay_multiplier_min": 0.25,
    "forecast_coefficient_replay_multiplier_max": 7.50,
    "forecast_coefficient_replay_multiplier_steps": 46,
    "forecast_coefficient_replay_selection_metric": "r2",
    # Actual force timing applied inside the recursive forecast ODE.
    # Positive values delay the learned force: force(t) uses learned force(t-lag).
    "forecast_force_lag_s": -0.18,
    "forecast_orientation_ablation_lag_min_s": -2.0,
    "forecast_orientation_ablation_lag_max_s": 2.0,
    "forecast_orientation_ablation_lag_steps": 41,
    "forecast_orientation_ablation_strength_min": -1.0,
    "forecast_orientation_ablation_strength_max": 1.5,
    "forecast_orientation_ablation_strength_steps": 51,
    "orientation_envelope_lag_max_s": 5.0,
    "orientation_envelope_lag_candidates": 151,
    # Diagnostic-only forecast post-correction seeded from the replay sweep.
    # The uncorrected forecast is still recorded in metrics/csv for comparison.
    "forecast_force_post_correction_enabled": False,
    "forecast_force_post_correction_lag_s": -0.40,
    "forecast_force_post_correction_gain": 0.425,
    "bayes_full_ode_outputs": True,
    # Pure-seas positional workbook layout: column 1 time [s], columns 2/3 wave
    # probes [m], columns 4/5 global X/Y [m], and column 13 roll angle [deg].
    "default_data_file": "KTHTest73Excel.xlsx",
    "excel_run_sheets": ["Active"],
    "forecast_sheet": "KTHTest73",
    "training_sheets": ["KTHTest73"],
    "run_gap_s": 120.0,                # plotting/time-axis gap only; sequence windows never cross runs

    # Synthetic fallback used only for unit checks or running without --data.
    "t_end_demo": 120.0,
    "dt_demo": 0.04,
    "noise_std_deg": 0.08,

    # Feature engineering
    "freq_multipliers": [1],
    "wave_feature_set": [
        "wave_signal",
        "wave_slope",
        "wave_abs_slope",
        "wave_curvature",
        "wave_abs_curvature",
        "wave_signal_slope_product",
        "wave_envelope",
        "wave_envelope_slow",
    ],
    # Force-target lag diagnostic showed the inferred-force target is best
    # aligned about 0.95 s earlier than the original wave-to-roll lag.
    "fixed_wave_lag_s": 3.0,
    "fixed_wave_lag_source": "validation_force_lag_scan",
    "wave_alignment_source_file": "../LSTMPINNS/PINNSLSTM_REAL_KRISO_RevR.py",
    "wave_lag_feature_offsets_s": [-0.96, -0.48, 0.0, 0.48, 0.96],
    "max_wave_lag_s": 4.0,
    "n_wave_lag_candidates": 81,
    "bandpass_low_period_factor": 1.8,
    "bandpass_high_period_factor": 0.35,
    "wave_envelope_seconds": 1.5,
    "wave_envelope_slow_seconds": 3.0,
    "wave_quadrature_basis_enabled": True,
    "wave_quadrature_signal_init": -0.08,
    "wave_quadrature_slope_init": 0.18,
    # Keep the remaining wave channels as context for the LSTM/gate, but
    # stop them dominating the explicit force while diagnosing sign/phase.
    "wave_auxiliary_force_gain": 0.0,
    "wave_shape_correction_enabled": False,
    "wave_shape_correction_hidden": 8,
    "wave_shape_correction_gain": 0.0,
    "use_position_features": True,
    "apply_position_encounter_shift": False,
    "wave_direction_xy": [-1.0, 0.0],       # measured waves propagate toward decreasing vessel X
    "wave_phase_speed_mps": None,           # None -> deep-water estimate from dominant roll period
    "encounter_position_reference": "initial",
    "encounter_time_shift_clip_s": 30.0,
    "use_orientation_features": True,
    "orientation_smoothing_seconds": 0.5,
    "orientation_min_speed_mps": 1.0e-5,
    # Keep orientation/position features available to the network, but do not
    # hard-multiply the pure wave force by the orientation gate in this
    # experiment. The previous diagnostics showed the gate envelope was often
    # anti-correlated with the inferred force envelope.
    "wave_orientation_gate_enabled": False,
    "wave_parallel_min_gain": 0.3173611847161887,
    "wave_parallel_velocity_blend": 0.5855481030715662,
    "wave_parallel_velocity_reference_quantile": 0.8800043044906735,
    "wave_parallel_obliquity_power": 2.459497372056859,
    "wave_parallel_gate_smoothing_seconds": 0.60,
    "wave_parallel_gate_state_force": False,
    "use_motion_feedback": False,
    "motion_feedback_delay_s": 0.08,
    # Strictly causal feedback only: never expose the current roll target.
    "motion_feedback_delay_offsets_s": [0.20, 0.40, 0.80],
    # Feedback-aware rollouts rerun the LSTM once per simulated step.
    # Bound only this expensive loss path; ordinary fitting retains batch 32
    # and the final forecast still spans the complete 15% holdout.
    # Cover at least one complete ≈2.56 s roll cycle so recursive training sees
    # both a peak and a trough instead of learning only a short decay segment.
    "motion_feedback_rollout_window_s": 5.08,
    "motion_feedback_rollout_batch_size": 1,
    # Do not let measured motion history leak into direct roll/rate or residual
    # predictions. Retain it only as a weak correction to the wave-force gate.
    "motion_feedback_backbone_gain": 0.0,
    "motion_feedback_gate_gain": 0.0,
    "roll_rate_smoothing_seconds": 0.12,
    "scale_roll_rate_target": False,

    # Standard causal stacked LSTM. No tokenisation, positional encoding, or learned embeddings are used.
    # The numeric channels are passed through a small feature adapter because
    # The numeric feature adapter preserves the original hidden-width comparison setting.
    # 160 samples span 6.36 s at 25 Hz, matching the attached metrics trial while still
    # exposing a moderate direct forecast horizon.
    "seq_len": 192,
    "stride": 2,
    # Oversample windows around measured roll turning points inside the
    # training split only. This increases exposure to local mins/maxs without
    # changing the dataset or leaking validation/forecast points.
    "turning_point_sampling_enabled": True,
    "turning_point_sampling_repeats": 2,
    "turning_point_sampling_min_prominence_deg": 0.08,
    "turning_point_sampling_neighbourhood": 4,
    "turning_point_sampling_window_fracs": [0.35, 0.50, 0.65],
    # Overlap output windows; causal context-aware stitching retains the
    # prediction with the most preceding history at each timestamp.
    "prediction_stride": 128,
    "inference_batch_size": 64,       # evaluation only; does not alter training batches
    "lstm_hidden_size": 64,
    "lstm_layers": 2,
    "lstm_dropout": 0.10453660952184797,
    "fc_hidden": 64,

    # Training
    "epochs": 50,
    # Bold anti-overfit pass: slower optimiser and heavy regularisation, with
    # fit quality allowed to fall if validation/autonomous dynamics improve.
    "batch_size": 16,
    "learning_rate": 0.0004986359006149152,
    "weight_decay": 4.326508084466064e-06,
    "grad_clip_norm": 1.0,
    "use_amp": True,
    "amp_dtype": "float16",
    "use_fused_adamw": True,
    "rollout_every_n_batches": 1,
    "rollout_warmup_epochs": 0,
    "rollout_eval": True,
    "val_every": 5,
    "log_every": 50,
    "early_stopping_patience": 18,
    "early_stopping_min_delta": 1.0e-5,
    "training_selection_metric": "forecast_r2",
    "direct_forecast_selection_r2_weight": 1.0,
    "direct_forecast_selection_loss_weight": 0.35,
    "direct_forecast_selection_peak_weight": 1.00,
    "direct_forecast_selection_extrema_weight": 0.75,
    "direct_forecast_selection_asym_extrema_weight": 1.00,
    "direct_forecast_selection_spectral_weight": 0.25,

    # Active loss weights adjusted from Metrics2.json to reduce teacher-forced
    # overfit while preserving the already-good waveform shape.
    "lambda_data": 1.60,
    "lambda_r2_data": 17.028917094784482,
    "lambda_rate_data": 0.0,
    "lambda_roll_slope": 0.0,
    "lambda_kinematic": 0.0,
    "lambda_physics": 0.0,
    "lambda_boundary": 0.0,
    "lambda_force_reg": 3.0e-5,
    "lambda_force_smooth": 2.5e-5,
    # Train gently against inferred total force where the envelope indicates
    # force bursts.
    "lambda_wave_force_target": 0.0,
    "lambda_pure_wave_force_target": 0.0,
    "lambda_total_force_target": 0.35,
    "total_force_target_phase_weight": 2.0,
    "total_force_target_underforce_weight": 3.0,
    "lambda_total_force_shape": 0.0,
    "total_force_shape_overshoot_weight": 5.0,
    "lambda_total_force_tail": 0.0,
    # Emphasise inferred-force burst events so quiet regions do not dominate the
    # force fit while peak timing/envelope errors remain unresolved.
    "lambda_total_force_event": 1.5,
    "total_force_event_quantile": 0.72,
    "total_force_event_weight": 6.0,
    # Match local force waveform shape in the specific bands where diagnostics
    # show the right trend but imperfect peak decomposition.
    "lambda_total_force_band_shape": 0.90,
    "total_force_band_shape_freqs_hz": [0.45, 1.70, 3.20],
    "total_force_band_shape_weights": [0.25, 5.0, 6.0],
    "total_force_band_shape_phase_weight": 0.45,
    "total_force_band_shape_event_weight": 10.0,
    "total_force_band_shape_envelope_weight": 1.20,
    # Frequency-domain correction for the force-shape regions still visible in
    # diagnostics. The low-frequency band is excess-only so the wave force is
    # not asked to learn arbitrary slow bias, while the 1.6 Hz and 3.2 Hz bands
    # receive direct magnitude/shape pressure.
    "lambda_total_force_spectral_shape": 1.60,
    "total_force_spectral_shape_band_lows_hz": [0.02, 0.95, 1.55],
    "total_force_spectral_shape_band_highs_hz": [0.95, 1.55, 3.20],
    "total_force_spectral_shape_weights": [1.0, 3.0, 18.0],
    "total_force_spectral_shape_excess_only": [1.0, 0.0, 0.0],
    "total_force_spectral_shape_underfit_weight": 16.00,
    "total_force_spectral_shape_complex_weight": 0.60,
    "total_force_spectral_shape_low_excess_margin": 1.02,
    "force_regime_parallel_max_gain": 0.45,
    "force_regime_side_min_gain": 0.75,
    "force_regime_parallel_weight": 1.0,
    "force_regime_oblique_weight": 1.0,
    "force_regime_side_weight": 2.0,
    "force_regime_envelope_weight": 0.25,
    "lambda_total_force_regime_balance": 0.0,
    "lambda_force_mean": 0.0,
    "lambda_force_residual": 0.0,
    "lambda_wave_envelope_gate_reg": 0.0,
    "lambda_wave_envelope_gate_target": 0.0,
    "lambda_wave_envelope_gate_shape": 0.0,
    "wave_envelope_gate_target_huber_delta": 0.35,
    "wave_envelope_gate_target_base_floor_ratio": 0.08,
    "wave_force_target_smoothing_seconds": 0.06,
    "force_target_band_mse_weight": 1.00,
    "force_target_raw_weight": 0.25,
    "force_target_huber_delta": 1.50,
    "force_target_band_huber_delta": 1.25,
    "wave_force_target_phase_weight": 3.00,
    "wave_force_target_amplitude_weight": 1.00,
    "wave_force_target_underforce_weight": 3.00,
    "lambda_wave_force_highpass": 1.5,
    "wave_force_highpass_window_s": 0.45,
    "lambda_force_band_amplitude": 0.0,
    "lambda_force_band_phase": 0.0,
    "lambda_force_lag_penalty": 0.0,
    "force_lag_penalty_max_s": 1.2,
    "force_lag_penalty_steps": 7,
    "force_band_amplitude_underfit_weight": 4.00,
    # Match the local burst envelope of the learned total force without asking
    # the model to reproduce every inferred-force spike pointwise.
    "lambda_force_envelope": 0.4,
    "force_envelope_window_s": 0.80,
    "force_envelope_huber_delta": 0.60,
    "wave_envelope_gate_enabled": False,
    "wave_envelope_gate_gain": 0.60,
    "wave_envelope_gate_min": 0.35,
    "wave_envelope_gate_max": 1.25,
    # Extrema fidelity is deliberately dominant in Rev M. The pointwise term
    # upweights high-|roll| samples, while the explicit extrema terms enforce
    # signed peak/trough values, timing, local prominence and recursive cycling.
    "lambda_peak_data": 4.50,
    "lambda_peak_trough": 4.196567737966396,
    "lambda_local_prominence": 0.0,
    "lambda_amplitude_underfit": 7.584956872866984,
    "lambda_extrema_window_underfit": 12.00,
    "extrema_window_radius": 3,
    "extrema_window_quantile": 0.60,
    "extrema_window_overshoot_weight": 0.12,
    "extrema_window_peak_weight": 1.0,
    "extrema_window_trough_weight": 1.0,
    "extrema_window_scale_floor_ratio": 0.08,
    "lambda_high_amplitude_underfit": 6.00,
    "high_amplitude_underfit_quantile": 0.68,
    "high_amplitude_underfit_overshoot_weight": 0.08,
    "high_amplitude_underfit_scale_floor_ratio": 0.08,
    "roll_amplitude_calibration_enabled": True,
    "roll_amplitude_calibration_gain_init": 0.0742194652557373,
    "roll_amplitude_calibration_gain_max": 0.40,
    "roll_amplitude_calibration_threshold_scaled": 0.75,
    "roll_amplitude_calibration_softness_scaled": 0.25,
    "roll_amplitude_calibration_rate_scale": True,
    # Kept for compatibility with the Rev M config; ignored while
    # ``pinn_enabled`` is false.
    "lambda_rollout": 4.0,
    "lambda_rollout_rate": 0.15,
    "lambda_rollout_amplitude": 8.0,
    "lambda_rollout_phase": 3.0,
    "lambda_rollout_peak_trough": 4.0,
    "lambda_rollout_global_extrema": 1.0,
    "lambda_rollout_extrema_window_underfit": 3.00,
    "lambda_rollout_high_amplitude_underfit": 1.50,
    "lambda_high_pass_residual": 0.70,
    "lambda_low_pass_residual": 1.80,
    "lambda_roll_spectral_shape": 1.90,
    "roll_spectral_shape_band_lows_hz": [0.00, 0.85, 1.55],
    "roll_spectral_shape_band_highs_hz": [0.28, 1.15, 3.20],
    "roll_spectral_shape_weights": [2.5, 4.0, 3.5],
    "roll_spectral_shape_raw_bands": [1.0, 0.0, 0.0],
    "roll_spectral_shape_underfit_weight": 3.25,
    "roll_spectral_shape_complex_weight": 0.18,
    "roll_spectral_shape_mag_floor_ratio": 0.025,
    "curvature_high_pass_window_s": 0.22,
    "curvature_scale_floor_ratio": 0.04,
    "lambda_roll_curvature": 0.0,
    "lambda_rollout_local_prominence": 0.5,
    "lambda_rollout_high_pass_residual": 0.70,
    "lambda_rollout_roll_curvature": 0.0,
    "lambda_rollout_roll_spectral_shape": 0.45,
    # Direct LSTM forecast pressure. This mirrors the production
    # no-ODE forecast path, with deliberately stronger peak/spectrum penalties
    # on the tail of each training window.
    "direct_forecast_loss_enabled": True,
    "direct_forecast_loss_window_s": 7.60,
    "direct_forecast_loss_horizons_s": [1.52, 3.04, 5.08, 7.60],
    "direct_forecast_loss_horizon_weights": [1.1, 1.2, 1.3, 1.6],
    "direct_forecast_tail_fraction": 0.0,
    "direct_forecast_tail_multiplier": 1.0,
    "direct_forecast_loss_min_steps": 8,
    "direct_forecast_loss_batch_size": 3,
    "direct_forecast_turning_point_sampling_enabled": True,
    "direct_forecast_turning_point_weight": 0.85,
    "direct_forecast_turning_point_max_tasks": 3,
    "direct_forecast_turning_point_min_prominence_deg": 0.08,
    "direct_forecast_turning_point_neighbourhood": 4,
    "direct_forecast_turning_point_window_fracs": [0.35, 0.50, 0.65],
    "lambda_direct_forecast": 6.0,
    "lambda_direct_forecast_rate": 0.20,
    "lambda_direct_forecast_amplitude": 24.0,
    "lambda_direct_forecast_phase": 4.0,
    "lambda_direct_forecast_peak_trough": 26.0,
    "lambda_direct_forecast_global_extrema": 2.5,
    "lambda_direct_forecast_extrema_window_underfit": 13.0,
    "lambda_direct_forecast_high_amplitude_underfit": 8.0,
    "lambda_direct_forecast_asymmetric_extrema": 10.0,
    "direct_forecast_asym_extrema_radius": 4,
    "direct_forecast_asym_extrema_quantile": 0.45,
    "direct_forecast_peak_underfit_weight": 4.0,
    "direct_forecast_peak_overshoot_weight": 0.35,
    "direct_forecast_trough_overshoot_weight": 4.5,
    "direct_forecast_trough_underfit_weight": 0.75,
    "direct_forecast_asym_extrema_scale_floor_ratio": 0.06,
    "lambda_direct_forecast_local_prominence": 2.0,
    "lambda_direct_forecast_high_pass_residual": 1.20,
    "lambda_direct_forecast_roll_spectral_shape": 2.50,
    "lambda_direct_forecast_roll_curvature": 0.25,
    "lambda_global_extrema": 2.1055894117187552,
    "peak_data_quantile": 0.667388952818952,
    "peak_data_alpha": 4.3262995135899995,
    "peak_trough_quantile": 0.4389764516006415,
    "peak_trough_neighbourhood": 2,
    "peak_trough_underfit_weight": 7.80563940980747,
    "global_extrema_location_weight": 0.8403640235763261,
    "global_extrema_softmax_beta": 18.0,
    "amplitude_window_s": 2.1415163861745268,
    "amplitude_quantile": 0.6685759594361653,
    "amplitude_underfit_margin": 0.08326781960578375,
    "amplitude_underfit_power": 2.0,
    # Local-prominence settings are shared by the rollout and direct-forecast
    # prominence losses.
    "local_prominence_window_s": 0.7123666413820411,
    "local_prominence_quantile": 0.39967794041856036,
    "local_prominence_neighbourhood": 4,
    "local_prominence_scale_floor_ratio": 0.12758345334202084,
    "local_prominence_value_weight": 2.0,
    "local_prominence_shape_weight": 1.4686959475754362,
    # High-pass residual loss explicitly supervises sub-cycle structure.
    "high_pass_window_s": 0.28,
    "high_pass_scale_floor_ratio": 0.07725865132793107,
    "low_pass_window_s": 4.80,
    "low_pass_scale_floor_ratio": 0.08,
    "rollout_window_s": 5.08,
    "rollout_min_steps": 8,
    "lambda_reversal": 0.0,
    "lambda_turning_point": 0.0,
    "reversal_tanh_scale": 4.0,
    "turning_point_neighbourhood": 4,
    "kinematic_weight_quantile": 0.40404924836305306,
    "kinematic_weight_alpha": 7.869243775021384,

    # Physics coefficient initial guesses and constraints in scaled coordinates.
    # KVLCC2 is a VLCC/tanker hullform, so bias the roll dynamics toward a slow,
    # softly restoring, lightly damped roll response instead of a stiff small-craft response.
    # metrics2 trial 19 non-architecture parameters seed the LSTM comparison run.
    "c_roll_init": 0.20610502092309096,
    "c_quad_init": 4.4562293397215795e-05,
    "k_roll_init": 5.286357774176246,
    "freeze_physics_coefficients": False,
    "bayes_freeze_physics_coefficients": False,
    "c_roll_min": 1.0e-4,
    "c_quad_min": 1.0e-6,
    "c_quad_max": 0.01,
    "k_roll_min": 1.0e-4,
    # Forcing scale terms
    "force_residual_scale": 0.0,
    "wave_forcing_gain": 4.0,
    "state_forcing_gain": 0.0,
    "wave_gate_bias": 1.0,
    "wave_gate_gain": 0.0,
    "turn_moment_scale": 0.0,

    # Bayesian hyperparameter optimisation. Each trial trains for 150 epochs by
    # default. Search stops after the joint fit/forecast target, or when either
    # full-wave ODE target is reached. Moving-vessel studies are intentionally
    # incompatible with this stationary, two-probe beam-sea dataset.
    "bayes_study_version": "revm_lstm_architecture_dynamics_v1",
    "bayes_compatible_study_versions": [
        "revm_lstm_architecture_dynamics_v1",
    ],
    "bayes_opt_dir": "Bayes_RevM_LSTM_ArchitectureDynamics_v1",
    "bayes_trials": 100000,
    "bayes_init_points": 16,
    "bayes_trial_epochs": 150,
    "bayes_log_every": 50,            # compact progress-log interval
    "bayes_candidate_pool": 2048,
    "bayes_ei_jitter": 0.01,
    "bayes_r2_objective_weight": 50.0,
    "bayes_rmse_objective_weight": 10.0,
    "bayes_fit_r2_objective_weight": 50.0,
    "bayes_fit_rmse_objective_weight": 10.0,
    "bayes_phase_objective_weight": 0.5,
    "bayes_amplitude_objective_weight": 0.5,
    "bayes_extrema_objective_weight": 20.0,
    "bayes_forecast_extrema_rmse_objective_weight": 1.5,
    "bayes_forecast_peak_underfit_objective_weight": 6.0,
    "bayes_forecast_trough_overshoot_objective_weight": 6.0,
    "bayes_turning_objective_weight": 0.25,
    "bayes_physics_objective_weight": 0.25,
    "bayes_kinematic_objective_weight": 0.25,
    "bayes_stop_on_target": True,
    "bayes_target_r2": 0.75,
    "bayes_target_rms_error_deg": 0.15,
    "bayes_target_fit_r2": 0.80,
    "bayes_target_fit_rms_error_deg": 0.96,
    "bayes_target_full_ode_r2": 0.95,
    "bayes_target_full_ode_rms_error_deg": 0.15,
    "bayes_resume": True,
    "bayes_checkpoint_every": 0,
    "bayes_save_trial_checkpoints": False,
    # Bayes trials generate the same fit, forecast, wave-force, lag, loss, per-run,
    # and full-ODE images as standard training.
    "bayes_save_trial_plots": True,
    "bayes_retrain_best": True,
    "bayes_reuse_prepared_data": True,
    "bayes_reuse_loader_cache": True,
    # Four possible training strides are searched. Keep their prepared arrays,
    # base tensors, and train/validation windows in RAM between CPU trials.
    "bayes_prepared_data_cache_max_entries": 4,
    "bayes_tensor_cache_max_entries": 4,
    "bayes_window_cache_max_entries": 8,
    "bayes_keep_window_cache_between_trials": True,
    # Deterministic grid-search controls. The grid path locks every parameter
    # except the seven spectral/rollout amplitude knobs requested for Rev M.
    "grid_opt_dir": "Grid_RevM_LSTM_SpectralRollout",
    "grid_step": 0.25,
    "grid_max_trials": 0,              # 0 means exhaustive; use CLI cap for chunked runs
    "grid_resume": True,
    "grid_save_trial_plots": True,
}


# Capture the loss weightings exactly where they are declared in the Training
# component. These values remain authoritative when the historical source-trial
# parameters below are applied, and they seed the first Bayesian trial.
REVM_TRAINING_WEIGHT_KEYS = (
    "lambda_data",
    "lambda_r2_data",
    "lambda_force_reg",
    "lambda_force_smooth",
    "lambda_force_mean",
    "lambda_force_residual",
    "lambda_wave_force_target",
    "lambda_pure_wave_force_target",
    "lambda_total_force_target",
    "total_force_target_phase_weight",
    "total_force_target_underforce_weight",
    "lambda_total_force_shape",
    "total_force_shape_overshoot_weight",
    "lambda_total_force_tail",
    "lambda_total_force_event",
    "total_force_event_quantile",
    "total_force_event_weight",
    "lambda_total_force_band_shape",
    "total_force_band_shape_freqs_hz",
    "total_force_band_shape_weights",
    "total_force_band_shape_phase_weight",
    "total_force_band_shape_event_weight",
    "total_force_band_shape_envelope_weight",
    "lambda_total_force_spectral_shape",
    "total_force_spectral_shape_band_lows_hz",
    "total_force_spectral_shape_band_highs_hz",
    "total_force_spectral_shape_weights",
    "total_force_spectral_shape_excess_only",
    "total_force_spectral_shape_underfit_weight",
    "total_force_spectral_shape_complex_weight",
    "total_force_spectral_shape_low_excess_margin",
    "force_regime_parallel_max_gain",
    "force_regime_side_min_gain",
    "force_regime_parallel_weight",
    "force_regime_oblique_weight",
    "force_regime_side_weight",
    "force_regime_envelope_weight",
    "lambda_total_force_regime_balance",
    "lambda_wave_envelope_gate_reg",
    "lambda_wave_envelope_gate_target",
    "lambda_wave_envelope_gate_shape",
    "wave_envelope_gate_target_huber_delta",
    "wave_envelope_gate_target_base_floor_ratio",
    "wave_force_target_smoothing_seconds",
    "force_target_band_mse_weight",
    "force_target_raw_weight",
    "force_target_huber_delta",
    "force_target_band_huber_delta",
    "wave_force_target_amplitude_weight",
    "wave_force_target_underforce_weight",
    "lambda_wave_force_highpass",
    "wave_force_highpass_window_s",
    "lambda_force_band_amplitude",
    "lambda_force_band_phase",
    "lambda_force_lag_penalty",
    "force_lag_penalty_max_s",
    "force_lag_penalty_steps",
    "force_band_amplitude_underfit_weight",
    "lambda_force_envelope",
    "force_envelope_window_s",
    "force_envelope_huber_delta",
    "wave_envelope_gate_gain",
    "lambda_peak_data",
    "lambda_peak_trough",
    "lambda_local_prominence",
    "lambda_amplitude_underfit",
    "lambda_extrema_window_underfit",
    "extrema_window_radius",
    "extrema_window_quantile",
    "extrema_window_overshoot_weight",
    "extrema_window_peak_weight",
    "extrema_window_trough_weight",
    "extrema_window_scale_floor_ratio",
    "lambda_high_amplitude_underfit",
    "high_amplitude_underfit_quantile",
    "high_amplitude_underfit_overshoot_weight",
    "high_amplitude_underfit_scale_floor_ratio",
    "lambda_reversal",
    "lambda_turning_point",
    "reversal_tanh_scale",
    "lambda_rollout",
    "lambda_rollout_rate",
    "lambda_rollout_amplitude",
    "lambda_rollout_phase",
    "lambda_rollout_peak_trough",
    "lambda_rollout_global_extrema",
    "lambda_rollout_extrema_window_underfit",
    "lambda_rollout_high_amplitude_underfit",
    "lambda_high_pass_residual",
    "lambda_low_pass_residual",
    "lambda_roll_spectral_shape",
    "roll_spectral_shape_band_lows_hz",
    "roll_spectral_shape_band_highs_hz",
    "roll_spectral_shape_weights",
    "roll_spectral_shape_raw_bands",
    "roll_spectral_shape_underfit_weight",
    "roll_spectral_shape_complex_weight",
    "roll_spectral_shape_mag_floor_ratio",
    "curvature_high_pass_window_s",
    "curvature_scale_floor_ratio",
    "lambda_roll_curvature",
    "lambda_rollout_local_prominence",
    "lambda_rollout_high_pass_residual",
    "lambda_rollout_roll_spectral_shape",
    "lambda_rollout_roll_curvature",
    "direct_forecast_loss_enabled",
    "direct_forecast_loss_window_s",
    "direct_forecast_loss_horizons_s",
    "direct_forecast_loss_horizon_weights",
    "direct_forecast_tail_fraction",
    "direct_forecast_tail_multiplier",
    "direct_forecast_loss_min_steps",
    "direct_forecast_loss_batch_size",
    "direct_forecast_turning_point_sampling_enabled",
    "direct_forecast_turning_point_weight",
    "direct_forecast_turning_point_max_tasks",
    "direct_forecast_turning_point_min_prominence_deg",
    "direct_forecast_turning_point_neighbourhood",
    "direct_forecast_turning_point_window_fracs",
    "lambda_direct_forecast",
    "lambda_direct_forecast_rate",
    "lambda_direct_forecast_amplitude",
    "lambda_direct_forecast_phase",
    "lambda_direct_forecast_peak_trough",
    "lambda_direct_forecast_global_extrema",
    "lambda_direct_forecast_extrema_window_underfit",
    "lambda_direct_forecast_high_amplitude_underfit",
    "lambda_direct_forecast_asymmetric_extrema",
    "direct_forecast_asym_extrema_radius",
    "direct_forecast_asym_extrema_quantile",
    "direct_forecast_peak_underfit_weight",
    "direct_forecast_peak_overshoot_weight",
    "direct_forecast_trough_overshoot_weight",
    "direct_forecast_trough_underfit_weight",
    "direct_forecast_asym_extrema_scale_floor_ratio",
    "lambda_direct_forecast_local_prominence",
    "lambda_direct_forecast_high_pass_residual",
    "lambda_direct_forecast_roll_spectral_shape",
    "lambda_direct_forecast_roll_curvature",
    "direct_forecast_selection_r2_weight",
    "direct_forecast_selection_loss_weight",
    "direct_forecast_selection_peak_weight",
    "direct_forecast_selection_extrema_weight",
    "direct_forecast_selection_asym_extrema_weight",
    "direct_forecast_selection_spectral_weight",
    "lambda_global_extrema",
)
REVM_TRAINING_WEIGHTINGS: Dict[str, object] = {
    key: copy.deepcopy(CONFIG[key]) for key in REVM_TRAINING_WEIGHT_KEYS
}

# Rev M LSTM keeps the attached result's data, physics, training, and loss settings;
# the three architecture settings are mapped directly by width/layer/dropout. These are
# configuration/hyperparameter values, not trained neural-network weights. Normal
# training and the first Bayes trial use the active Rev M subset. Terminal coefficients
# remain embedded separately for the full-wave ODE integration command.
REVM_SOURCE_TRIAL_INDEX = 19
REVM_SOURCE_TRIAL_PARAMS: Dict[str, object] = {
    "seq_len": 192,
    "stride": 2,
    # Oversample windows around measured roll turning points inside the
    # training split only. This increases exposure to local mins/maxs without
    # changing the dataset or leaking validation/forecast points.
    "turning_point_sampling_enabled": True,
    "turning_point_sampling_repeats": 2,
    "turning_point_sampling_min_prominence_deg": 0.08,
    "turning_point_sampling_neighbourhood": 4,
    "turning_point_sampling_window_fracs": [0.35, 0.50, 0.65],
    "prediction_stride": 128,
    "batch_size": 16,
    "learning_rate": 0.0004986359006149152,
    "weight_decay": 4.326508084466064e-06,
    "lstm_hidden_size": 64,
    "lstm_layers": 2,
    "lstm_dropout": 0.10453660952184797,
    "fc_hidden": 64,
    "epochs": 50,
    "lambda_data": 1.60,
    "lambda_r2_data": 17.028917094784482,
    "lambda_force_reg": 3.0e-5,
    "lambda_force_smooth": 2.5e-5,
    "lambda_total_force_target": 0.35,
    "total_force_target_phase_weight": 2.0,
    "total_force_target_underforce_weight": 3.0,
    "lambda_total_force_event": 1.5,
    "total_force_event_quantile": 0.72,
    "total_force_event_weight": 6.0,
    "lambda_total_force_band_shape": 0.90,
    "total_force_band_shape_freqs_hz": [0.45, 1.70, 3.20],
    "total_force_band_shape_weights": [0.25, 5.0, 6.0],
    "total_force_band_shape_phase_weight": 0.45,
    "total_force_band_shape_event_weight": 10.0,
    "total_force_band_shape_envelope_weight": 1.20,
    "lambda_total_force_spectral_shape": 1.60,
    "total_force_spectral_shape_band_lows_hz": [0.02, 0.95, 1.55],
    "total_force_spectral_shape_band_highs_hz": [0.95, 1.55, 3.20],
    "total_force_spectral_shape_weights": [1.0, 3.0, 18.0],
    "total_force_spectral_shape_excess_only": [1.0, 0.0, 0.0],
    "total_force_spectral_shape_underfit_weight": 16.00,
    "total_force_spectral_shape_complex_weight": 0.60,
    "total_force_spectral_shape_low_excess_margin": 1.02,
    "force_regime_parallel_max_gain": 0.45,
    "force_regime_side_min_gain": 0.75,
    "force_regime_parallel_weight": 1.0,
    "force_regime_oblique_weight": 1.0,
    "force_regime_side_weight": 2.0,
    "force_regime_envelope_weight": 0.25,
    "wave_force_target_phase_weight": 3.00,
    "wave_force_target_amplitude_weight": 1.00,
    "wave_force_target_underforce_weight": 3.00,
    "lambda_wave_force_highpass": 1.5,
    "wave_force_highpass_window_s": 0.45,
    "force_band_amplitude_underfit_weight": 4.00,
    "lambda_force_envelope": 0.4,
    "force_envelope_window_s": 0.80,
    "force_envelope_huber_delta": 0.60,
    "wave_envelope_gate_gain": 0.60,
    "lambda_peak_data": 4.50,
    "lambda_peak_trough": 4.196567737966396,
    "lambda_amplitude_underfit": 7.584956872866984,
    "lambda_rollout": 4.0,
    "lambda_rollout_rate": 0.15,
    "lambda_rollout_amplitude": 8.0,
    "lambda_rollout_phase": 3.0,
    "lambda_rollout_peak_trough": 4.0,
    "lambda_rollout_global_extrema": 1.0,
    "lambda_global_extrema": 2.1055894117187552,
    "lambda_high_pass_residual": 0.70,
    "lambda_low_pass_residual": 1.80,
    "lambda_roll_spectral_shape": 1.90,
    "roll_spectral_shape_band_lows_hz": [0.00, 0.85, 1.55],
    "roll_spectral_shape_band_highs_hz": [0.28, 1.15, 3.20],
    "roll_spectral_shape_weights": [2.5, 4.0, 3.5],
    "roll_spectral_shape_raw_bands": [1.0, 0.0, 0.0],
    "roll_spectral_shape_underfit_weight": 3.25,
    "roll_spectral_shape_complex_weight": 0.18,
    "roll_spectral_shape_mag_floor_ratio": 0.025,
    "curvature_high_pass_window_s": 0.22,
    "curvature_scale_floor_ratio": 0.04,
    "lambda_rollout_local_prominence": 0.5,
    "lambda_rollout_high_pass_residual": 0.70,
    "lambda_rollout_roll_spectral_shape": 0.45,
    "peak_data_quantile": 0.667388952818952,
    "peak_data_alpha": 4.3262995135899995,
    "peak_trough_quantile": 0.4389764516006415,
    "peak_trough_neighbourhood": 2,
    "peak_trough_underfit_weight": 7.80563940980747,
    "global_extrema_location_weight": 0.8403640235763261,
    "amplitude_window_s": 2.1415163861745268,
    "amplitude_quantile": 0.6685759594361653,
    "amplitude_underfit_margin": 0.08326781960578375,
    "amplitude_underfit_power": 2.0,
    "local_prominence_window_s": 0.7123666413820411,
    "local_prominence_quantile": 0.39967794041856036,
    "local_prominence_neighbourhood": 4,
    "local_prominence_scale_floor_ratio": 0.12758345334202084,
    "local_prominence_shape_weight": 1.4686959475754362,
    "high_pass_window_s": 0.28,
    "high_pass_scale_floor_ratio": 0.07725865132793107,
    "low_pass_window_s": 4.80,
    "low_pass_scale_floor_ratio": 0.08,
    "rollout_window_s": 5.08,
    "turning_point_neighbourhood": 4,
    "kinematic_weight_quantile": 0.40404924836305306,
    "kinematic_weight_alpha": 7.869243775021384,
    "c_roll_init": 0.20610502092309096,
    "c_quad_init": 4.4562293397215795e-05,
    "k_roll_init": 5.286357774176246,
    "force_residual_scale": 0.0,
    "wave_forcing_gain": 4.0,
    "wave_gate_bias": 1.0,
    "wave_gate_gain": 0.0,
    "wave_parallel_min_gain": 0.3173611847161887,
    "wave_parallel_velocity_blend": 0.5855481030715662,
    "wave_parallel_velocity_reference_quantile": 0.8800043044906735,
    "wave_parallel_obliquity_power": 2.459497372056859,
}
REVM_SOURCE_LEARNED_PHYSICS: Dict[str, float] = {
    # Metrics2 current post-training coefficients.
    "c_roll": 0.3853868842124939,
    "c_quad": 9.492147364653647e-05,
    "k_roll": 4.861016273498535,
}
CONFIG.update(REVM_SOURCE_TRIAL_PARAMS)
CONFIG.update({
    # Temporarily route forcing through measured-wave coupling only; state
    # features remain available to the wave gate.
    "state_forcing_gain": 0.0,
    "force_residual_scale": 0.0,
    "turn_moment_scale": 0.0,
    "baseline_run_id": "revm_metrics2_trial19_direct_lstm_no_ode",
    "baseline_run_metrics": {
        "fit_r2": 0.9616780572893926,
        "fit_rms_error_deg": 0.4424532898669103,
        "train_r2": 0.9722054011300854,
        "train_rms_error_deg": 0.3505198301681422,
        "validation_r2": 0.9271532844397941,
        "validation_rms_error_deg": 0.5942441098847692,
        "source_forecast_window_s": 9.439994810000002,
        "forecast_r2": 0.9680923279046043,
        "forecast_rms_error_deg": 0.5828017023253252,
        "forecast_phase_lag_s": 0.04000091000000339,
        "forecast_amplitude_ratio": 0.905803242306859,
        "forecast_extrema_relative_error": 0.04785167651386014,
        "full_ode_r2": -0.7955922662366366,
        "full_ode_rms_error_deg": 3.028639157685876,
        "best_epoch": 20,
        "best_metric": 0.9680923279046043,
        **REVM_SOURCE_LEARNED_PHYSICS,
    },
    "source_metrics_file": "metrics2.json",
    "source_trial_index": REVM_SOURCE_TRIAL_INDEX,
    "source_revision": "Rev M metrics2 Bayes trial 19 baseline",
    "source_learned_physics_parameters": copy.deepcopy(REVM_SOURCE_LEARNED_PHYSICS),
    "full_ode_coefficient_source": "source",
    "seq_len": 192,
    "batch_size": 16,
    "rollout_window_s": 5.08,
    "c_roll_init": 0.20610502092309096,
    "c_quad_init": 4.4562293397215795e-05,
    "k_roll_init": 5.286357774176246,
    "freeze_physics_coefficients": False,
    "bayes_freeze_physics_coefficients": False,
    "output_dir": "outputs_revm_lstm_comparison",
    "checkpoint_dir": "checkpoints_revm_lstm_comparison",
    "log_dir": "logs_revm_lstm_comparison",
    "bayes_study_version": "revm_lstm_architecture_dynamics_v1",
    "bayes_compatible_study_versions": [
        "revm_lstm_architecture_dynamics_v1",
    ],
    "bayes_opt_dir": "Bayes_RevM_LSTM_ArchitectureDynamics_v1",
})
# Do this last so historical source-trial values cannot silently replace the
# weightings currently declared in Rev M's Training component.
CONFIG.update(REVM_TRAINING_WEIGHTINGS)
CONFIG["pinn_enabled"] = False


BAYES_SEARCH_SPACE: Dict[str, Dict[str, object]] = {
    # LSTM/data-window hyperparameters
    "seq_len": {"type": "categorical", "values": [192]},
    "stride": {"type": "categorical", "values": [8, 16, 24, 32]},
    "prediction_stride": {"type": "categorical", "values": [32, 64, 96, 128, 192]},
    "batch_size": {"type": "categorical", "values": [16, 32, 64]},
    "learning_rate": {"type": "float", "low": 2.0e-5, "high": 1.0e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1.0e-8, "high": 2.0e-2, "log": True},
    "lstm_hidden_size": {"type": "categorical", "values": [32]},
    "lstm_layers": {"type": "categorical", "values": [2]},
    "lstm_dropout": {"type": "float", "low": 0.0, "high": 0.45},
    "fc_hidden": {"type": "categorical", "values": [128]},

    # Loss weighting hyperparameters
    "lambda_data": {"type": "float", "low": 0.05, "high": 3.00, "log": True},
    "lambda_r2_data": {"type": "float", "low": 5.0, "high": 50.0, "log": True},
    "lambda_force_reg": {"type": "float", "low": 1.0e-5, "high": 5.0e-2, "log": True},
    "lambda_force_smooth": {"type": "float", "low": 1.0e-5, "high": 2.0e-2, "log": True},
    "total_force_target_phase_weight": {"type": "float", "low": 0.0, "high": 10.0},
    "total_force_event_quantile": {"type": "float", "low": 0.60, "high": 0.90},
    "total_force_event_weight": {"type": "float", "low": 0.0, "high": 8.0},
    "total_force_band_shape_phase_weight": {"type": "float", "low": 0.0, "high": 1.5},
    "total_force_band_shape_event_weight": {"type": "float", "low": 0.0, "high": 10.0},
    "total_force_band_shape_envelope_weight": {"type": "float", "low": 0.0, "high": 2.0},
    "lambda_total_force_spectral_shape": {"type": "categorical", "values": [0.55]},
    "total_force_spectral_shape_underfit_weight": {"type": "float", "low": 0.0, "high": 8.0},
    "total_force_spectral_shape_complex_weight": {"type": "float", "low": 0.0, "high": 1.0},
    "total_force_spectral_shape_low_excess_margin": {"type": "float", "low": 1.0, "high": 2.0},
    "force_regime_parallel_max_gain": {"type": "float", "low": 0.25, "high": 0.55},
    "force_regime_side_min_gain": {"type": "float", "low": 0.65, "high": 0.90},
    "force_regime_side_weight": {"type": "float", "low": 1.0, "high": 4.0},
    "force_regime_envelope_weight": {"type": "float", "low": 0.0, "high": 1.0},
    "wave_shape_correction_gain": {"type": "float", "low": 0.0, "high": 0.5},
    "wave_force_target_amplitude_weight": {"type": "float", "low": 0.0, "high": 4.0},
    "wave_force_target_underforce_weight": {"type": "float", "low": 0.0, "high": 8.0},
    "force_band_amplitude_underfit_weight": {"type": "float", "low": 0.0, "high": 10.0},
    "force_envelope_window_s": {"type": "float", "low": 0.45, "high": 2.4},
    "force_envelope_huber_delta": {"type": "float", "low": 0.25, "high": 2.0},
    "wave_envelope_gate_gain": {"type": "float", "low": 0.0, "high": 1.2},
    "lambda_peak_data": {"type": "float", "low": 1.0, "high": 8.00},
    "lambda_peak_trough": {"type": "float", "low": 3.0, "high": 20.0, "log": True},
    "lambda_amplitude_underfit": {"type": "float", "low": 2.0, "high": 14.0},
    "lambda_rollout": {"type": "categorical", "values": [4.0]},
    "lambda_rollout_rate": {"type": "categorical", "values": [0.15]},
    "lambda_rollout_amplitude": {"type": "categorical", "values": [8.0]},
    "lambda_rollout_phase": {"type": "categorical", "values": [3.0]},
    "lambda_rollout_peak_trough": {"type": "categorical", "values": [4.0]},
    "lambda_rollout_global_extrema": {"type": "categorical", "values": [1.0]},
    "lambda_global_extrema": {"type": "float", "low": 1.0, "high": 12.0, "log": True},
    "lambda_high_pass_residual": {"type": "float", "low": 0.02, "high": 3.00, "log": True},
    "lambda_rollout_local_prominence": {"type": "categorical", "values": [0.5]},
    "lambda_rollout_high_pass_residual": {"type": "categorical", "values": [0.70]},
    "peak_data_quantile": {"type": "float", "low": 0.55, "high": 0.80},
    "peak_data_alpha": {"type": "float", "low": 3.0, "high": 10.0},
    "peak_trough_quantile": {"type": "float", "low": 0.40, "high": 0.70},
    "peak_trough_neighbourhood": {"type": "categorical", "values": [1, 2, 3]},
    "peak_trough_underfit_weight": {"type": "float", "low": 4.0, "high": 12.0},
    "global_extrema_location_weight": {"type": "float", "low": 0.50, "high": 2.00},
    "amplitude_window_s": {"type": "float", "low": 0.4, "high": 3.0},
    "amplitude_quantile": {"type": "float", "low": 0.45, "high": 0.95},
    "amplitude_underfit_margin": {"type": "float", "low": 0.0, "high": 0.10},
    "amplitude_underfit_power": {"type": "categorical", "values": [1.0, 1.5, 2.0, 3.0]},
    "local_prominence_window_s": {"type": "float", "low": 0.4, "high": 1.4},
    "local_prominence_quantile": {"type": "float", "low": 0.15, "high": 0.45},
    "local_prominence_neighbourhood": {"type": "categorical", "values": [2, 3, 4]},
    "local_prominence_scale_floor_ratio": {"type": "float", "low": 0.02, "high": 0.20, "log": True},
    "local_prominence_shape_weight": {"type": "float", "low": 0.50, "high": 3.00},
    "high_pass_window_s": {"type": "float", "low": 0.3, "high": 1.2},
    "wave_force_highpass_window_s": {"type": "float", "low": 0.2, "high": 1.2},
    "high_pass_scale_floor_ratio": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
    # Keep Bayes rollout training aligned with the 128-sample forecast horizon.
    "rollout_window_s": {"type": "categorical", "values": [5.08]},
    "turning_point_neighbourhood": {"type": "categorical", "values": [1, 2, 3, 4]},
    "kinematic_weight_quantile": {"type": "float", "low": 0.40, "high": 0.95},
    "kinematic_weight_alpha": {"type": "float", "low": 0.0, "high": 10.0},

    # Physics/forcing scale hyperparameters
    "c_roll_init": {"type": "float", "low": 0.12, "high": 0.22},
    "c_quad_init": {"type": "categorical", "values": [3.0e-05]},
    "k_roll_init": {"type": "float", "low": 5.0, "high": 5.6},
    "force_residual_scale": {"type": "categorical", "values": [0.0]},
    "wave_forcing_gain": {"type": "float", "low": 0.5, "high": 120.0, "log": True},
    "wave_gate_bias": {"type": "float", "low": 0.0, "high": 3.0},
    "wave_gate_gain": {"type": "float", "low": 0.0, "high": 5.0},
    "wave_parallel_min_gain": {"type": "float", "low": 0.05, "high": 0.45},
    "wave_parallel_velocity_blend": {"type": "float", "low": 0.35, "high": 0.85},
    "wave_parallel_velocity_reference_quantile": {"type": "float", "low": 0.75, "high": 0.98},
    "wave_parallel_obliquity_power": {"type": "float", "low": 0.70, "high": 3.00},
}


BAYES_SEARCH_SPACE = {
    "seq_len": {"type": "categorical", "values": [128, 160, 192, 224, 256]},
    "prediction_stride": {"type": "categorical", "values": [32, 64, 96, 128, 192]},
    "batch_size": {"type": "categorical", "values": [16, 32, 64]},
    "learning_rate": {"type": "float", "low": 2.0e-5, "high": 1.0e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1.0e-8, "high": 2.0e-2, "log": True},
    "lstm_hidden_size": {"type": "categorical", "values": [32, 48, 64, 96, 128]},
    "lstm_layers": {"type": "categorical", "values": [1, 2, 3, 4]},
    "lstm_dropout": {"type": "float", "low": 0.0, "high": 0.30},
    "fc_hidden": {"type": "categorical", "values": [64, 96, 128, 192, 256]},
    "c_roll_init": {"type": "float", "low": 0.12, "high": 0.22},
    "c_quad_init": {"type": "float", "low": 1.0e-6, "high": 5.0e-4, "log": True},
    "k_roll_init": {"type": "float", "low": 5.0, "high": 5.6},
}


GRID_SEARCH_SPACE: Dict[str, Tuple[float, float]] = {
    "lambda_r2_data": (6.0, 18.0),
    "lambda_peak_trough": (3.0, 20.0),
    "lambda_amplitude_underfit": (2.0, 14.0),
    "lambda_global_extrema": (1.0, 12.0),
    "lambda_high_pass_residual": (0.02, 3.0),
    "lstm_dropout": (0.0, 0.45),
    "weight_decay": (1.0e-8, 2.0e-2),
    "c_roll_init": (0.12, 0.22),
    "k_roll_init": (5.0, 5.6),
}


GRID_SEARCH_STEPS: Dict[str, float] = {
    "lambda_r2_data": 3.0,
    "lambda_peak_trough": 0.5,
    "lambda_amplitude_underfit": 1.0,
    "lambda_global_extrema": 0.5,
    "lambda_high_pass_residual": 0.3,
    "lstm_dropout": 0.05,
    "weight_decay": 2.0e-3,
    "c_roll_init": 0.03,
    "k_roll_init": 0.25,
}


# =============================================================================
# LOGGING / DEVICE / UTILITY
# =============================================================================

def setup_logging(log_dir: str = "logs_revm") -> logging.Logger:
    requested_log_dir = Path(log_dir).expanduser()
    log_warnings: List[str] = []
    try:
        requested_log_dir.mkdir(parents=True, exist_ok=True)
        if not requested_log_dir.is_dir():
            raise NotADirectoryError(f"{requested_log_dir} exists but is not a directory")
        probe_path = requested_log_dir / f".pinn_lstm_write_test_{os.getpid()}"
        with open(probe_path, "w", encoding="utf-8") as probe:
            probe.write("ok\n")
        probe_path.unlink(missing_ok=True)
        active_log_dir = requested_log_dir
    except Exception as exc:
        active_log_dir = Path(tempfile.gettempdir()) / "pinn_lstm_revm_logs"
        active_log_dir.mkdir(parents=True, exist_ok=True)
        log_warnings.append(
            f"Could not use requested log directory {requested_log_dir!s}: {exc!r}. "
            f"Using fallback log directory {active_log_dir!s}."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = active_log_dir / f"pinn_lstm_revm_{stamp}.log"

    lg = logging.getLogger(f"NOPINN_LSTM_REVM_{stamp}")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    lg.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
    except Exception as exc:
        fallback_log_dir = Path(tempfile.gettempdir()) / "pinn_lstm_revm_logs"
        fallback_log_dir.mkdir(parents=True, exist_ok=True)
        fallback_log_path = fallback_log_dir / f"pinn_lstm_revm_{stamp}.log"
        log_warnings.append(
            f"Could not open log file {log_path!s}: {exc!r}. "
            f"Using fallback log file {fallback_log_path!s}."
        )
        try:
            fh = logging.FileHandler(fallback_log_path, encoding="utf-8")
            log_path = fallback_log_path
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Unable to open requested log file {log_path!s} or fallback log file "
                f"{fallback_log_path!s}: {fallback_exc!r}"
            ) from fallback_exc
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(sh)
    for warning in log_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
        lg.warning(warning)
    lg.info("Log file: %s", log_path)
    return lg



def available_cpu_count() -> int:
    counts: List[int] = []
    try:
        counts.append(len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        pass
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            value = int(os.environ.get(name, "0"))
        except ValueError:
            value = 0
        if value > 0:
            counts.append(value)
    counts.append(os.cpu_count() or 1)
    return max(1, min(c for c in counts if c > 0))


def physical_memory_gb() -> Optional[float]:
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 ** 2)
    except Exception:
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))  # type: ignore[attr-defined]
        pages = int(os.sysconf("SC_PHYS_PAGES"))  # type: ignore[attr-defined]
        if page_size > 0 and pages > 0:
            return float(page_size * pages) / (1024.0 ** 3)
    except Exception:
        pass
    return None


def _auto_or_int(value: object, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "default"}:
            return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def configured_cpu_count(cfg: Dict[str, object]) -> Tuple[int, int]:
    """Return usable and detected CPU counts after applying the configured cap."""
    detected = available_cpu_count()
    limit = _auto_or_int(cfg.get("vcpu_limit", 0), detected)
    if limit <= 0:
        limit = detected
    return max(1, min(detected, int(limit))), int(detected)


def configure_torch_threads(cfg: Dict[str, object], lg: Optional[logging.Logger] = None) -> None:
    available, detected = configured_cpu_count(cfg)
    configured_limit = _auto_or_int(cfg.get("vcpu_limit", 0), detected)
    if configured_limit <= 0:
        configured_limit = detected
    n = _auto_or_int(cfg.get("torch_num_threads", 0), available)
    if n <= 0:
        n = available
    n = max(1, min(int(n), available))
    if str(cfg.get("runtime_profile", "auto")).strip().lower() == "cpu_large_memory":
        interop_default = 1
    else:
        interop_default = max(1, min(4, n))
    interop = _auto_or_int(cfg.get("torch_num_interop_threads", 0), interop_default)
    if interop <= 0:
        interop = interop_default
    interop = max(1, min(int(interop), available))

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        # Interop threads can only be set before parallel work starts. Safe to ignore.
        pass

    precision = str(cfg.get("float32_matmul_precision", "high")).strip().lower()
    if precision in {"highest", "high", "medium"}:
        try:
            torch.set_float32_matmul_precision(precision)
        except Exception:
            pass
    allow_tf32 = bool(cfg.get("cuda_allow_tf32", True))
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    except Exception:
        pass

    cfg["detected_cpu_count"] = int(detected)
    cfg["available_cpu_count"] = int(available)
    cfg["resolved_vcpu_limit"] = int(configured_limit)
    cfg["usable_cpu_count"] = int(available)
    cfg["resolved_torch_num_threads"] = int(torch.get_num_threads())
    cfg["resolved_torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    if lg:
        lg.info(
            "Torch CPU threads: detected=%d configured_vcpu_limit=%d usable=%d intraop=%d interop=%d matmul_precision=%s tf32=%s",
            detected,
            configured_limit,
            available,
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            precision if precision in {"highest", "high", "medium"} else "default",
            allow_tf32,
        )

def set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(cfg: Dict[str, object], lg: Optional[logging.Logger] = None) -> torch.device:
    requested = str(cfg.get("device", "auto")).strip().lower()
    if requested in {"gpu", "cuda:0"}:
        requested = "cuda"
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")

    cuda_available = torch.cuda.is_available()
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if cuda_available else "cpu")

    cfg["resolved_device"] = str(device)
    cfg["cuda_available_at_start"] = bool(cuda_available)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("cuda_benchmark", True))
        idx = int(device.index or 0)
        props = torch.cuda.get_device_properties(idx)
        cfg["cuda_device_name"] = torch.cuda.get_device_name(idx)
        cfg["cuda_device_total_memory_gb"] = float(props.total_memory / (1024 ** 3))
        cfg["torch_cuda_version"] = str(torch.version.cuda)
        if lg:
            lg.info("CUDA active: torch=%s cuda=%s device=%s memory=%.2f GB cudnn_benchmark=%s tf32=%s",
                    torch.__version__, torch.version.cuda, cfg["cuda_device_name"], cfg["cuda_device_total_memory_gb"],
                    bool(cfg.get("cuda_benchmark", True)), bool(cfg.get("cuda_allow_tf32", True)))
    elif lg:
        lg.warning("Using CPU. torch=%s cuda_available=%s torch_cuda=%s",
                   torch.__version__, cuda_available, torch.version.cuda)
    return device


def safe_json_dump(payload: Dict[str, object], path: Path) -> None:
    def clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, torch.Tensor):
            return float(v.detach().cpu().item()) if v.numel() == 1 else v.detach().cpu().tolist()
        if isinstance(v, dict):
            return {str(k): clean(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, Path):
            return str(v)
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(payload), fh, indent=2)


def moving_average_time(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return values.copy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    n = max(int(round(float(window_s) / max(dt, 1e-8))), 1)
    n = min(n, int(values.size))
    if n <= 1:
        return values.copy()
    kernel = np.ones(n, dtype=float) / float(n)
    smoothed = np.convolve(values, kernel, mode="same")
    if smoothed.size != values.size:
        start = max(0, (smoothed.size - values.size) // 2)
        smoothed = smoothed[start:start + values.size]
    return smoothed


def highpass_time(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    return np.asarray(x, dtype=float) - moving_average_time(x, t, window_s)


def rolling_rms_time(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(moving_average_time(x * x, t, window_s), 0.0))


def estimate_dominant_period(x: np.ndarray, t: np.ndarray, idx: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)[idx]
    tt = np.asarray(t, dtype=float)[idx]
    if len(x) < 16:
        return max(float(np.median(np.diff(t)) * 20.0), 1.0)
    dt = float(np.median(np.diff(tt)))
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=max(dt, 1e-8))
    if len(freqs) <= 1:
        return max(dt * 20.0, 1.0)
    power = np.abs(spec) ** 2
    power[0] = 0.0
    mask = (freqs > 1.0 / max((tt[-1] - tt[0]), 1e-6)) & (freqs < 0.5 / max(dt, 1e-8))
    if not np.any(mask):
        return max(dt * 20.0, 1.0)
    sel_freqs = freqs[mask]
    sel_power = power[mask]
    f_dom = float(sel_freqs[np.argmax(sel_power)])
    return 1.0 / max(f_dom, 1e-6)


def bandpass_time(x: np.ndarray, t: np.ndarray, center_period_s: float,
                  low_period_factor: float = 1.6, high_period_factor: float = 0.45) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    low_window = max(float(center_period_s) * float(low_period_factor), dt * 3.0)
    high_window = max(float(center_period_s) * float(high_period_factor), dt * 2.0)
    return moving_average_time(highpass_time(x, t, low_window), t, high_window)


def robust_roll_rate(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = np.gradient(np.asarray(phi, dtype=float), np.asarray(t, dtype=float))
    if len(v) >= 5:
        kernel = np.ones(5) / 5.0
        v = np.convolve(v, kernel, mode="same")
    return v


def normalise_segments(raw_segments: Optional[object], n_points: int) -> List[Tuple[int, int]]:
    n = int(n_points)
    if n <= 0:
        return []
    segments: List[Tuple[int, int]] = []
    if raw_segments is not None:
        arr = np.asarray(raw_segments, dtype=int)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            for row in arr:
                lo = int(max(0, min(n - 1, row[0])))
                hi = int(max(0, min(n - 1, row[1])))
                if hi >= lo:
                    segments.append((lo, hi))
    if not segments:
        segments = [(0, n - 1)]
    return segments


def run_names_from_pack(wave_pack: Dict[str, np.ndarray], n_segments: int) -> List[str]:
    raw = wave_pack.get("run_names")
    if raw is None:
        return [f"run_{i + 1}" for i in range(int(n_segments))]
    names = [str(v) for v in list(raw)]
    if len(names) < int(n_segments):
        names.extend(f"run_{i + 1}" for i in range(len(names), int(n_segments)))
    return names[:int(n_segments)]


def run_segments_from_pack(wave_pack: Dict[str, np.ndarray], n_points: int) -> Tuple[List[Tuple[int, int]], List[str]]:
    segments = normalise_segments(wave_pack.get("run_segments"), n_points)
    return segments, run_names_from_pack(wave_pack, len(segments))


def wave_segments_from_pack(wave_pack: Dict[str, np.ndarray], n_points: int, n_expected: int) -> List[Tuple[int, int]]:
    segments = normalise_segments(wave_pack.get("wave_run_segments"), n_points)
    if len(segments) != int(n_expected):
        segments = normalise_segments(None, n_points)
    return segments


def values_by_segment(x: np.ndarray, t: np.ndarray, segments: List[Tuple[int, int]],
                      fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    out = np.empty_like(x_arr, dtype=float)
    for lo, hi in segments:
        sl = slice(int(lo), int(hi) + 1)
        out[sl] = np.asarray(fn(x_arr[sl], t_arr[sl]), dtype=float)
    return out


def gradient_time_by_segments(x: np.ndarray, t: np.ndarray, segments: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    segs = normalise_segments(segments, len(x_arr))

    def _grad(xx: np.ndarray, tt: np.ndarray) -> np.ndarray:
        if len(xx) < 2:
            return np.zeros_like(xx, dtype=float)
        return np.gradient(xx, tt)

    return values_by_segment(x_arr, t_arr, segs, _grad)


def moving_average_time_by_segments(x: np.ndarray, t: np.ndarray, window_s: float,
                                    segments: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    segs = normalise_segments(segments, len(x_arr))
    return values_by_segment(x_arr, t_arr, segs, lambda xx, tt: moving_average_time(xx, tt, window_s))


def rolling_rms_time_by_segments(x: np.ndarray, t: np.ndarray, window_s: float,
                                 segments: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    segs = normalise_segments(segments, len(x_arr))
    return values_by_segment(x_arr, t_arr, segs, lambda xx, tt: rolling_rms_time(xx, tt, window_s))


def bandpass_time_by_segments(x: np.ndarray, t: np.ndarray, center_period_s: float,
                              low_period_factor: float = 1.6, high_period_factor: float = 0.45,
                              segments: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    segs = normalise_segments(segments, len(x_arr))
    return values_by_segment(
        x_arr,
        t_arr,
        segs,
        lambda xx, tt: bandpass_time(xx, tt, center_period_s, low_period_factor, high_period_factor),
    )


def robust_roll_rate_by_segments(phi: np.ndarray, t: np.ndarray,
                                 segments: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    phi_arr = np.asarray(phi, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    segs = normalise_segments(segments, len(phi_arr))
    return values_by_segment(phi_arr, t_arr, segs, robust_roll_rate)


def estimate_dominant_period_by_segments(x: np.ndarray, t: np.ndarray, idx: np.ndarray,
                                         segments: Optional[List[Tuple[int, int]]] = None) -> float:
    idx_arr = np.asarray(idx, dtype=int)
    segs = normalise_segments(segments, len(np.asarray(t)))
    periods: List[float] = []
    weights: List[int] = []
    for lo, hi in segs:
        run_idx = idx_arr[(idx_arr >= int(lo)) & (idx_arr <= int(hi))]
        if len(run_idx) >= 16:
            periods.append(float(estimate_dominant_period(x, t, run_idx)))
            weights.append(int(len(run_idx)))
    if not periods:
        return estimate_dominant_period(x, t, idx_arr)
    return float(np.average(np.asarray(periods, dtype=float), weights=np.asarray(weights, dtype=float)))


def segment_for_index(segments: Optional[List[Tuple[int, int]]] | object, n_points: int, idx: int) -> Tuple[int, int]:
    segs = normalise_segments(segments, n_points)
    idx_i = int(idx)
    for lo, hi in segs:
        if int(lo) <= idx_i <= int(hi):
            return int(lo), int(hi)
    return 0, max(0, int(n_points) - 1)


# =============================================================================
# EXCEL LOADING
# =============================================================================

def normalize_column_name(col: object) -> str:
    text = str(col).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def match_column(cols: Iterable[object], keys: Iterable[str], exclude: Optional[Iterable[object]] = None) -> Optional[object]:
    exclude_set = set(exclude or [])
    norm_cols = [(c, normalize_column_name(c)) for c in cols if c not in exclude_set]
    norm_keys = [normalize_column_name(k) for k in keys if normalize_column_name(k)]

    for key in norm_keys:
        for col, norm_col in norm_cols:
            if norm_col == key:
                return col
    for key in norm_keys:
        for col, norm_col in norm_cols:
            if key in norm_col.split():
                return col
    for key in norm_keys:
        if len(key) < 3:
            continue
        for col, norm_col in norm_cols:
            if key in norm_col:
                return col
    return None


def make_unique_columns(cols: Iterable[object]) -> List[str]:
    cleaned: List[str] = []
    counts: Dict[str, int] = {}
    for col in cols:
        base = str(col).strip()
        idx = counts.get(base, 0)
        cleaned.append(base if idx == 0 else f"{base}.{idx}")
        counts[base] = idx + 1
    return cleaned


def resolve_data_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    candidates = [raw] if raw.is_absolute() else [_SCRIPT_DIR / raw, Path.cwd() / raw, raw]
    seen = set()
    for candidate in candidates:
        key = os.fspath(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return resolve_runtime_path(raw)


def infer_roll_units(col_name: object, raw_series: pd.Series) -> Optional[str]:
    name_l = str(col_name).lower()
    if "deg" in name_l:
        return "deg"
    if "rad" in name_l:
        return "rad"
    text_values: List[str] = []
    for value in raw_series.dropna().tolist():
        if isinstance(value, str):
            text_values.append(value.lower().strip())
        if len(text_values) >= 5:
            break
    if any("deg" in v for v in text_values):
        return "deg"
    if any("rad" in v for v in text_values):
        return "rad"
    return None


def angle_values_to_radians(raw_values: np.ndarray, col_name: object,
                            unit_hint: Optional[str] = None) -> Tuple[np.ndarray, str]:
    raw = np.asarray(raw_values, dtype=float)
    if unit_hint == "deg":
        return np.deg2rad(raw), "deg"
    if unit_hint == "rad":
        return raw, "rad"
    max_abs = float(np.nanmax(np.abs(raw))) if raw.size else 0.0
    if max_abs > 2 * np.pi * 1.5:
        return np.deg2rad(raw), "deg inferred"
    return raw, "rad inferred"


def clean_optional_vessel_signal(vessel_df: pd.DataFrame, col: Optional[object], t_col: object) -> Optional[np.ndarray]:
    if col is None:
        return None
    s = pd.to_numeric(vessel_df[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    tmp = pd.DataFrame({"t": vessel_df[t_col], "y": s}).dropna(subset=["t"]).sort_values("t")
    tmp["y"] = tmp["y"].interpolate(method="linear", limit_direction="both")
    return tmp["y"].to_numpy(dtype=float)


def find_wave_time_and_signal(df: pd.DataFrame) -> Tuple[object, object]:
    cols = list(df.columns)
    wave_time_col = match_column(cols, ["wave_time", "wavetime", "wave time"])
    if wave_time_col is None:
        raise ValueError("Could not detect Wave_Time. Expected wave signal immediately to its right.")
    idx = cols.index(wave_time_col)
    if idx + 1 >= len(cols):
        raise ValueError(f"Found {wave_time_col!r}, but there is no wave signal column to its right.")
    return wave_time_col, cols[idx + 1]


def load_excel_sheet(data_path: Path, xl: pd.ExcelFile, sheet: str,
                     lg: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    lg.info("Reading Excel sheet: %s [%s]", data_path, sheet)
    df = pd.read_excel(xl, sheet_name=sheet)
    df.columns = make_unique_columns(df.columns)

    cols = list(df.columns)
    wave_2_col = heading_col = pitch_col = yaw_angle_col = None
    rudder_col = rps_col = None
    layout_note = "legacy named-column layout"
    named_ship_time = match_column(cols, ["ship_time", "ship time"])
    named_roll = match_column(cols, ["roll"])
    named_x = match_column(cols, ["x"])
    named_y = match_column(cols, ["y"])
    named_wave_time = match_column(cols, ["wave_time", "wave time", "wavetime"])
    combined_active_layout = all(
        col is not None
        for col in (named_ship_time, named_roll, named_x, named_y, named_wave_time)
    )
    legacy_positional_layout = not combined_active_layout and len(cols) >= 13

    if combined_active_layout:
        # Exact combined original/final layout used by Rev I for
        # my_file.xlsx[Active]. Ship and wave rows have independent clocks.
        t_col = named_ship_time
        roll_col = named_roll
        x_col = named_x
        y_col = named_y
        wave_time_col, wave_signal_col = find_wave_time_and_signal(df)
        rudder_col = match_column(cols, ["rudder feedback", "rudder"])
        rps_col = match_column(cols, ["rps feedback", "rps"])
        pitch_col = match_column(cols, ["pitch"])
        yaw_angle_col = match_column(cols, ["yaw"], exclude={wave_time_col, wave_signal_col})
        speed_col = match_column(cols, ["speed", "surge speed", "forward velocity"])
        yawrate_col = match_column(cols, ["yaw rate", "yaw_rate", "yawrate"])
        layout_note = (
            "combined Active layout (A Ship_Time, B Rudder Feedback, "
            "C RPS Feedback, D/E X/Y, F/G/H Roll/Pitch/Yaw, "
            "I Speed, J Yaw rate, L Wave_Time, M SS5-seed1)"
        )
    elif legacy_positional_layout:
        t_col = cols[0]              # A / column 1: Time (seconds)
        wave_time_col = t_col
        wave_signal_col = cols[1]    # B / column 2: Wave probe 1 (metres)
        wave_2_col = cols[2]         # C / column 3: Wave probe 2 (metres)
        x_col = cols[3]              # D / column 4: Vessel x position (metres)
        y_col = cols[4]              # E / column 5: Vessel y position (metres)
        roll_col = cols[12]          # M / column 13: Roll angle (degrees)
        speed_col = None
        yawrate_col = None
        layout_note = (
            "pure-seas positional layout (A time [s], B/C wave probes [m], "
            "D/E global x/y [m], M roll [deg])"
        )
    else:
        t_col = match_column(df.columns, ["ship_time", "vessel_time", "vessel time", "ship time", "time", "t_s", "time_s", "seconds", "sec"])
        roll_col = match_column(df.columns, ["roll_deg", "roll_rad", "roll", "motion", "phi"], exclude={t_col})
        speed_col = match_column(df.columns, ["speed", "u", "surge speed", "forward velocity"], exclude={t_col, roll_col})
        yawrate_col = match_column(df.columns, ["yawrate", "yaw_rate", "yaw rate", "r_deg_s", "r"], exclude={t_col, roll_col, speed_col})
        x_col = match_column(df.columns, ["x", "x_pos", "x position", "x_position", "x coordinate", "x_coord", "ship x pos"], exclude={t_col, roll_col, speed_col, yawrate_col})
        y_col = match_column(df.columns, ["y", "y_pos", "y position", "y_position", "y coordinate", "y_coord", "ship y pos"], exclude={t_col, roll_col, speed_col, yawrate_col, x_col})
        pitch_col = match_column(df.columns, ["pitch"], exclude={t_col, roll_col, speed_col, yawrate_col, x_col, y_col})
        yaw_angle_col = match_column(df.columns, ["yaw", "yaw angle"], exclude={t_col, roll_col, speed_col, yawrate_col, x_col, y_col, pitch_col})
        rudder_col = match_column(df.columns, ["rudder feedback", "rudder"])
        rps_col = match_column(df.columns, ["rps feedback", "rps"])
        wave_time_col, wave_signal_col = find_wave_time_and_signal(df)

    if t_col is None or roll_col is None:
        raise ValueError(f"Could not detect required Time/Roll columns. Columns: {list(df.columns)}")

    unit_hint = "deg" if combined_active_layout or legacy_positional_layout else infer_roll_units(roll_col, df[roll_col])
    optional_vessel_cols: List[object] = []
    for col in [
        rudder_col, rps_col, speed_col, yawrate_col, x_col, y_col,
        heading_col, yaw_angle_col, pitch_col,
    ]:
        if col is not None and col not in {t_col, roll_col} and col not in optional_vessel_cols:
            optional_vessel_cols.append(col)

    vessel_df = df[[t_col, roll_col] + optional_vessel_cols].copy()
    vessel_df[t_col] = pd.to_numeric(vessel_df[t_col], errors="coerce")
    vessel_df[roll_col] = pd.to_numeric(vessel_df[roll_col], errors="coerce")
    for col in optional_vessel_cols:
        vessel_df[col] = pd.to_numeric(vessel_df[col], errors="coerce")
    vessel_df = vessel_df.dropna(subset=[t_col, roll_col]).sort_values(t_col)

    wave_cols = [wave_time_col, wave_signal_col]
    if wave_2_col is not None and wave_2_col not in wave_cols:
        wave_cols.append(wave_2_col)
    wave_df = df[wave_cols].copy()
    wave_df[wave_time_col] = pd.to_numeric(wave_df[wave_time_col], errors="coerce")
    wave_df[wave_signal_col] = pd.to_numeric(wave_df[wave_signal_col], errors="coerce")
    if wave_2_col is not None and wave_2_col in wave_df:
        wave_df[wave_2_col] = pd.to_numeric(wave_df[wave_2_col], errors="coerce")
    wave_df = wave_df.dropna(subset=[wave_time_col, wave_signal_col]).sort_values(wave_time_col)
    if wave_2_col is not None and wave_2_col in wave_df and wave_df[wave_2_col].notna().sum() > 0:
        wave_df[wave_2_col] = wave_df[wave_2_col].interpolate(method="linear", limit_direction="both")

    if vessel_df.empty:
        raise ValueError("No numeric Time/Roll rows found after cleaning.")
    if wave_df.empty:
        raise ValueError("No numeric wave time/wave-signal rows found after cleaning.")

    t = vessel_df[t_col].to_numpy(dtype=float)
    phi_raw = vessel_df[roll_col].to_numpy(dtype=float)
    wave_t = wave_df[wave_time_col].to_numpy(dtype=float)
    wave_1_raw = wave_df[wave_signal_col].to_numpy(dtype=float)
    wave_2_wave_raw = None
    wave_source_note = f"wave elevation from {wave_signal_col} (metres)"
    if wave_2_col is not None and wave_2_col in wave_df and wave_df[wave_2_col].notna().sum() > 0:
        wave_2_wave_raw = wave_df[wave_2_col].to_numpy(dtype=float)
        wave_raw = 0.5 * (wave_1_raw + wave_2_wave_raw)
        wave_cross_beam_raw = 0.5 * (wave_2_wave_raw - wave_1_raw)
        wave_source_note = "midpoint average of Wave 1 and Wave 2; half-difference retained as cross-beam gradient proxy"
    else:
        wave_raw = wave_1_raw
        wave_cross_beam_raw = np.zeros_like(wave_raw)

    phi, units = angle_values_to_radians(phi_raw, roll_col, unit_hint)
    pack: Dict[str, np.ndarray] = {
        "wave_time_raw": wave_t,
        "wave_signal_raw": wave_raw,
        "wave_1_raw": wave_1_raw,
        "wave_midpoint_raw": wave_raw,
        "wave_cross_beam_raw": wave_cross_beam_raw,
    }
    if wave_2_wave_raw is not None:
        pack["wave_2_raw"] = wave_2_wave_raw
    speed_raw = clean_optional_vessel_signal(vessel_df, speed_col, t_col)
    yawrate_direct_raw = clean_optional_vessel_signal(vessel_df, yawrate_col, t_col)
    x_raw = clean_optional_vessel_signal(vessel_df, x_col, t_col)
    y_raw = clean_optional_vessel_signal(vessel_df, y_col, t_col)
    heading_raw = clean_optional_vessel_signal(vessel_df, heading_col, t_col)
    yaw_angle_raw = clean_optional_vessel_signal(vessel_df, yaw_angle_col, t_col)
    rudder_raw = clean_optional_vessel_signal(vessel_df, rudder_col, t_col)
    rps_raw = clean_optional_vessel_signal(vessel_df, rps_col, t_col)
    pitch_raw = clean_optional_vessel_signal(vessel_df, pitch_col, t_col)
    yawrate_raw = None
    yawrate_note = "not found"
    if yawrate_direct_raw is not None:
        yawrate_raw = np.deg2rad(yawrate_direct_raw)  # legacy files supplied yaw rate in deg/s
        yawrate_note = f"{yawrate_col} (direct deg/s -> rad/s)"
    else:
        yaw_source_col = yaw_angle_col if yaw_angle_raw is not None else heading_col
        yaw_source_raw = yaw_angle_raw if yaw_angle_raw is not None else heading_raw
        if yaw_source_raw is not None and len(t) >= 2:
            yaw_hint = infer_roll_units(yaw_source_col, df[yaw_source_col])
            yaw_rad, yaw_units = angle_values_to_radians(yaw_source_raw, yaw_source_col, yaw_hint)
            yawrate_raw = np.gradient(np.unwrap(yaw_rad), t)
            yawrate_note = f"{yaw_source_col} angle derivative ({yaw_units} -> rad/s)"
    if speed_raw is not None:
        pack["speed_raw"] = speed_raw
    if yawrate_raw is not None:
        pack["yawrate_raw"] = yawrate_raw
    if x_raw is not None:
        pack["x_position_raw"] = x_raw
    if y_raw is not None:
        pack["y_position_raw"] = y_raw
    if heading_raw is not None:
        heading_hint = infer_roll_units(heading_col, df[heading_col])
        pack["ship_heading_raw"] = angle_values_to_radians(heading_raw, heading_col, heading_hint)[0]
    if yaw_angle_raw is not None:
        yaw_hint = "deg" if combined_active_layout else infer_roll_units(yaw_angle_col, df[yaw_angle_col])
        pack["yaw_angle_raw"] = angle_values_to_radians(yaw_angle_raw, yaw_angle_col, yaw_hint)[0]
    if rudder_raw is not None:
        rudder_hint = "deg" if combined_active_layout else infer_roll_units(rudder_col, df[rudder_col])
        pack["rudder_feedback_raw"] = angle_values_to_radians(rudder_raw, rudder_col, rudder_hint)[0]
    if rps_raw is not None:
        pack["rps_feedback_raw"] = rps_raw
    if pitch_raw is not None:
        pitch_hint = "deg" if combined_active_layout else infer_roll_units(pitch_col, df[pitch_col])
        pack["pitch_angle_raw"] = angle_values_to_radians(pitch_raw, pitch_col, pitch_hint)[0]
    lg.info("  Sheet:              %s", sheet)
    lg.info("  Layout:             %s", layout_note)
    lg.info("  Time column:        %s", t_col)
    lg.info("  Roll column:        %s (%s -> internal radians)", roll_col, units)
    lg.info("  Wave time column:   %s", wave_time_col)
    lg.info("  Wave signal column: %s (metres)", wave_signal_col)
    lg.info("  Wave probe 2 column:%s", f"{wave_2_col} (metres)" if wave_2_col is not None else "not found")
    lg.info("  Active wave signal: %s", wave_source_note)
    lg.info("  Rudder column:      %s", rudder_col if rudder_col is not None else "not found")
    lg.info("  RPS column:         %s", rps_col if rps_col is not None else "not found")
    lg.info("  Pitch column:       %s", pitch_col if pitch_col is not None else "not found")
    lg.info("  Speed column:       %s", speed_col if speed_col is not None else "not found")
    lg.info("  Heading column:     %s", heading_col if heading_col is not None else "not found")
    lg.info("  Yaw angle column:   %s", yaw_angle_col if yaw_angle_col is not None else "not found")
    lg.info("  Yaw-rate source:    %s", yawrate_note)
    lg.info("  X position column:  %s", x_col if x_col is not None else "not found")
    lg.info("  Y position column:  %s", y_col if y_col is not None else "not found")
    return t, phi, pack


def _cfg_string_list(cfg: Optional[Dict[str, object]], key: str) -> List[str]:
    if cfg is None:
        return []
    raw = cfg.get(key, [])
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(part).strip() for part in raw if str(part).strip()]
    return []


def select_excel_sheets(sheet_names: List[str], cfg: Optional[Dict[str, object]], lg: logging.Logger) -> List[str]:
    requested = _cfg_string_list(cfg, "excel_run_sheets")
    by_lower = {str(name).lower(): str(name) for name in sheet_names}
    if requested and all(name.lower() in by_lower for name in requested):
        return [by_lower[name.lower()] for name in requested]
    if requested:
        missing = [name for name in requested if name.lower() not in by_lower]
        lg.info("Requested multi-run sheets not all present; missing=%s. Falling back to single-sheet selection.", missing)
    return [next((s for s in sheet_names if s.lower() == "timeseries"), sheet_names[0])]


def combine_excel_runs(runs: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, np.ndarray]]],
                       cfg: Optional[Dict[str, object]], lg: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if len(runs) == 1:
        name, t, phi, pack = runs[0]
        pack = dict(pack)
        t_raw = np.asarray(t, dtype=float)
        wave_t_raw = np.asarray(pack["wave_time_raw"], dtype=float)
        t = t_raw - float(t_raw[0])
        wave_t = wave_t_raw - float(wave_t_raw[0])
        pack["wave_time_raw"] = wave_t
        pack["run_names"] = np.asarray([name], dtype=object)
        pack["run_segments"] = np.asarray([[0, len(t) - 1]], dtype=int)
        pack["wave_run_segments"] = np.asarray([[0, len(wave_t) - 1]], dtype=int)
        pack["run_local_time"] = np.asarray(t, dtype=float).copy()
        pack["wave_run_local_time"] = wave_t.copy()
        lg.info(
            "Rebased single Excel run %s to local time origin: vessel %.6g->0 s, wave %.6g->0 s.",
            name,
            float(t_raw[0]),
            float(wave_t_raw[0]),
        )
        pack["forecast_run_name"] = str((cfg or {}).get("forecast_sheet", name))
        training = _cfg_string_list(cfg, "training_sheets") or [name]
        pack["training_run_names"] = np.asarray(training, dtype=object)
        return t, phi, pack

    gap_s = float((cfg or {}).get("run_gap_s", 120.0))
    vessel_keys = [
        "speed_raw", "yawrate_raw", "x_position_raw", "y_position_raw",
        "ship_heading_raw", "yaw_angle_raw", "rudder_feedback_raw",
        "rps_feedback_raw", "pitch_angle_raw",
    ]
    wave_keys = [
        "wave_signal_raw", "wave_1_raw", "wave_2_raw", "wave_midpoint_raw", "wave_cross_beam_raw",
    ]
    present_vessel_keys = sorted({key for _, _, _, pack in runs for key in vessel_keys if key in pack})
    present_wave_keys = sorted({key for _, _, _, pack in runs for key in wave_keys if key in pack})

    t_parts: List[np.ndarray] = []
    phi_parts: List[np.ndarray] = []
    local_t_parts: List[np.ndarray] = []
    wave_t_parts: List[np.ndarray] = []
    wave_local_t_parts: List[np.ndarray] = []
    vessel_parts: Dict[str, List[np.ndarray]] = {key: [] for key in present_vessel_keys}
    wave_parts: Dict[str, List[np.ndarray]] = {key: [] for key in present_wave_keys}
    run_segments: List[Tuple[int, int]] = []
    wave_segments: List[Tuple[int, int]] = []
    run_names: List[str] = []
    cursor = 0.0
    v_start = 0
    w_start = 0

    for run_idx, (name, t_raw, phi_raw, pack) in enumerate(runs):
        t_arr = np.asarray(t_raw, dtype=float)
        phi_arr = np.asarray(phi_raw, dtype=float)
        wave_t_raw = np.asarray(pack["wave_time_raw"], dtype=float)
        local_t = t_arr - float(t_arr[0])
        wave_local_t = wave_t_raw - float(wave_t_raw[0])
        if run_idx == 0:
            offset = 0.0
        else:
            offset = cursor + max(gap_s, 0.0)
        t_global = local_t + offset
        wave_t_global = wave_local_t + offset
        cursor = max(float(t_global[-1]), float(wave_t_global[-1]))

        t_parts.append(t_global)
        phi_parts.append(phi_arr)
        local_t_parts.append(local_t)
        wave_t_parts.append(wave_t_global)
        wave_local_t_parts.append(wave_local_t)
        run_segments.append((v_start, v_start + len(t_global) - 1))
        wave_segments.append((w_start, w_start + len(wave_t_global) - 1))
        run_names.append(str(name))

        for key in present_vessel_keys:
            if key in pack and len(np.asarray(pack[key])) == len(t_arr):
                vessel_parts[key].append(np.asarray(pack[key], dtype=float))
            else:
                vessel_parts[key].append(np.zeros(len(t_arr), dtype=float))
                lg.warning("Sheet %s has no usable %s; filling zeros for the combined dataset.", name, key)
        for key in present_wave_keys:
            if key in pack and len(np.asarray(pack[key])) == len(wave_t_raw):
                wave_parts[key].append(np.asarray(pack[key], dtype=float))
            elif key == "wave_cross_beam_raw":
                wave_parts[key].append(np.zeros(len(wave_t_raw), dtype=float))
            elif key in {"wave_1_raw", "wave_2_raw", "wave_midpoint_raw", "wave_signal_raw"}:
                wave_parts[key].append(np.asarray(pack["wave_signal_raw"], dtype=float))
            else:
                wave_parts[key].append(np.zeros(len(wave_t_raw), dtype=float))

        v_start += len(t_global)
        w_start += len(wave_t_global)

    out_pack: Dict[str, np.ndarray] = {
        "wave_time_raw": np.concatenate(wave_t_parts),
        "run_local_time": np.concatenate(local_t_parts),
        "wave_run_local_time": np.concatenate(wave_local_t_parts),
        "run_names": np.asarray(run_names, dtype=object),
        "run_segments": np.asarray(run_segments, dtype=int),
        "wave_run_segments": np.asarray(wave_segments, dtype=int),
        "forecast_run_name": str((cfg or {}).get("forecast_sheet", run_names[0])),
        "training_run_names": np.asarray(_cfg_string_list(cfg, "training_sheets") or run_names, dtype=object),
    }
    for key, parts in wave_parts.items():
        out_pack[key] = np.concatenate(parts)
    if "wave_signal_raw" not in out_pack and "wave_midpoint_raw" in out_pack:
        out_pack["wave_signal_raw"] = np.asarray(out_pack["wave_midpoint_raw"], dtype=float)
    if "wave_midpoint_raw" not in out_pack:
        out_pack["wave_midpoint_raw"] = np.asarray(out_pack["wave_signal_raw"], dtype=float)
    if "wave_cross_beam_raw" not in out_pack:
        out_pack["wave_cross_beam_raw"] = np.zeros_like(np.asarray(out_pack["wave_signal_raw"], dtype=float))
    for key, parts in vessel_parts.items():
        out_pack[key] = np.concatenate(parts)

    lg.info("Combined %d non-continuous Excel runs: %s", len(run_names), run_names)
    for name, (lo, hi), (wlo, whi) in zip(run_names, run_segments, wave_segments):
        lg.info("  Run %-10s vessel idx=%d-%d time=%.6g-%.6g s | wave idx=%d-%d",
                name, lo, hi, float(np.concatenate(t_parts)[lo]), float(np.concatenate(t_parts)[hi]), wlo, whi)
    return np.concatenate(t_parts), np.concatenate(phi_parts), out_pack


def load_excel(path: str | Path, lg: logging.Logger,
               cfg: Optional[Dict[str, object]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data_path = resolve_data_path(path)
    lg.info("Reading Excel file: %s", data_path)
    if not data_path.exists() or not data_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {data_path}")

    with pd.ExcelFile(data_path) as xl:
        sheets = select_excel_sheets(list(xl.sheet_names), cfg, lg)
        runs = [(sheet, *load_excel_sheet(data_path, xl, sheet, lg)) for sheet in sheets]
    return combine_excel_runs(runs, cfg, lg)


# =============================================================================
# SYNTHETIC DEMO
# =============================================================================

def generate_internal_demo(cfg: Dict[str, object], lg: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    lg.info("No data file supplied; using internal wave-driven roll demo.")
    rng = np.random.default_rng(int(cfg["seed"]))
    t = np.arange(0.0, float(cfg["t_end_demo"]) + float(cfg["dt_demo"]) / 2.0, float(cfg["dt_demo"]))
    wave = 0.55 * np.sin(0.19 * t) + 0.30 * np.sin(0.44 * t + 0.8) + 0.06 * rng.standard_normal(len(t))
    forcing = 0.45 * wave + 0.22 * np.gradient(wave, t)
    c1_true, c2_true, k1_true = 0.34, 0.018, 0.48
    phi = np.zeros_like(t)
    dphi = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        rhs = forcing[i - 1] - c1_true * dphi[i - 1] - c2_true * abs(dphi[i - 1]) * dphi[i - 1] - k1_true * phi[i - 1]
        dphi[i] = dphi[i - 1] + dt * rhs
        phi[i] = phi[i - 1] + dt * dphi[i]
    phi += np.deg2rad(float(cfg["noise_std_deg"])) * rng.standard_normal(len(phi))
    speed = 1.0 + 0.05 * np.sin(0.015 * t)
    yawrate = np.deg2rad(0.3 * np.sin(0.025 * t))
    y_pos = -0.25 * t
    x_pos = 2.0 * np.sin(0.01 * t)
    return t, phi, {
        "wave_time_raw": t.copy(),
        "wave_signal_raw": wave.copy(),
        "speed_raw": speed,
        "yawrate_raw": yawrate,
        "x_position_raw": x_pos,
        "y_position_raw": y_pos,
    }


# =============================================================================
# FEATURE ENGINEERING AND TURNING CORRECTIONS
# =============================================================================

def interp_to_vessel_time(vessel_t: np.ndarray, src_t: np.ndarray, src_y: np.ndarray,
                          lag_s: float = 0.0, encounter_time_shift_s: Optional[np.ndarray] = None) -> np.ndarray:
    query_t = np.asarray(vessel_t, dtype=float) - float(lag_s)
    if encounter_time_shift_s is not None:
        query_t = query_t + np.asarray(encounter_time_shift_s, dtype=float)
    return np.interp(query_t, src_t, src_y, left=src_y[0], right=src_y[-1])


def interp_to_vessel_time_by_segments(vessel_t: np.ndarray, src_t: np.ndarray, src_y: np.ndarray,
                                      lag_s: float = 0.0,
                                      encounter_time_shift_s: Optional[np.ndarray] = None,
                                      vessel_segments: Optional[List[Tuple[int, int]]] = None,
                                      source_segments: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    vessel_arr = np.asarray(vessel_t, dtype=float)
    src_t_arr = np.asarray(src_t, dtype=float)
    src_y_arr = np.asarray(src_y, dtype=float)
    v_segments = normalise_segments(vessel_segments, len(vessel_arr))
    s_segments = normalise_segments(source_segments, len(src_t_arr))
    if len(v_segments) != len(s_segments):
        return interp_to_vessel_time(vessel_arr, src_t_arr, src_y_arr, lag_s, encounter_time_shift_s)

    out = np.empty_like(vessel_arr, dtype=float)
    shift = None if encounter_time_shift_s is None else np.asarray(encounter_time_shift_s, dtype=float)
    for (vlo, vhi), (slo, shi) in zip(v_segments, s_segments):
        v_sl = slice(int(vlo), int(vhi) + 1)
        s_sl = slice(int(slo), int(shi) + 1)
        query_t = vessel_arr[v_sl] - float(lag_s)
        if shift is not None:
            query_t = query_t + shift[v_sl]
        src_t_run = src_t_arr[s_sl]
        src_y_run = src_y_arr[s_sl]
        out[v_sl] = np.interp(query_t, src_t_run, src_y_run, left=src_y_run[0], right=src_y_run[-1])
    return out


def deep_water_phase_speed(period_s: float) -> float:
    return 9.80665 * float(max(period_s, 1e-8)) / (2.0 * math.pi)


def position_encounter_shift_from_xy(t: np.ndarray,
                                     x_position: Optional[np.ndarray],
                                     y_position: Optional[np.ndarray],
                                     train_idx: np.ndarray,
                                     cfg: Dict[str, object],
                                     lg: Optional[logging.Logger],
                                     dominant_wave_period_s: Optional[float] = None,
                                     run_segments: Optional[List[Tuple[int, int]]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    t = np.asarray(t, dtype=float)
    shift = np.zeros_like(t, dtype=float)
    ratio = np.ones_like(t, dtype=float)
    info: Dict[str, object] = {
        "active": False,
        "phase_speed_mps": None,
        "direction_xy": [1.0, 0.0],
        "position_reference_m": 0.0,
        "time_shift_min_s": 0.0,
        "time_shift_max_s": 0.0,
        "encounter_frequency_ratio_min": 1.0,
        "encounter_frequency_ratio_max": 1.0,
    }
    if not bool(cfg.get("apply_position_encounter_shift", True)):
        return shift, ratio, info
    if x_position is None and y_position is None:
        return shift, ratio, info

    direction = np.asarray(cfg.get("wave_direction_xy", [1.0, 0.0]), dtype=float).reshape(-1)
    if direction.size != 2 or not np.all(np.isfinite(direction)):
        direction = np.asarray([0.0, -1.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    direction = np.asarray([1.0, 0.0], dtype=float) if norm < 1e-8 else direction / norm

    x = np.zeros_like(t) if x_position is None else np.asarray(x_position, dtype=float)
    y = np.zeros_like(t) if y_position is None else np.asarray(y_position, dtype=float)
    phase_speed = cfg.get("wave_phase_speed_mps")
    if phase_speed is None or float(phase_speed) <= 0.0:
        period = dominant_wave_period_s if dominant_wave_period_s is not None else 1.0
        phase_speed = deep_water_phase_speed(period)
    phase_speed = float(max(phase_speed, 1e-6))

    along_position = direction[0] * x + direction[1] * y
    ref_mode = str(cfg.get("encounter_position_reference", "initial")).lower()
    train_idx_arr = np.asarray(train_idx, dtype=int)
    segs = normalise_segments(run_segments, len(t))
    pos_refs: List[float] = []
    for lo, hi in segs:
        sl = slice(int(lo), int(hi) + 1)
        run_train = train_idx_arr[(train_idx_arr >= int(lo)) & (train_idx_arr <= int(hi))]
        if ref_mode in {"train_mean", "training_mean"} and len(run_train) > 0:
            pos_ref = float(np.mean(along_position[run_train]))
        else:
            pos_ref = float(along_position[int(lo)])
        pos_refs.append(pos_ref)
        shift[sl] = -(along_position[sl] - pos_ref) / phase_speed

    clip_s = cfg.get("encounter_time_shift_clip_s")
    if clip_s is not None and float(clip_s) > 0.0:
        shift = np.clip(shift, -float(clip_s), float(clip_s))
    ratio = gradient_time_by_segments(t + shift, t, segs) if len(t) > 1 else ratio
    ratio = np.clip(ratio, 0.05, 5.0)

    info.update({
        "active": True,
        "phase_speed_mps": phase_speed,
        "direction_xy": [float(direction[0]), float(direction[1])],
        "position_reference_m": float(pos_refs[0]) if pos_refs else 0.0,
        "position_reference_by_run_m": [float(v) for v in pos_refs],
        "time_shift_min_s": float(np.min(shift)),
        "time_shift_max_s": float(np.max(shift)),
        "encounter_frequency_ratio_min": float(np.min(ratio)),
        "encounter_frequency_ratio_max": float(np.max(ratio)),
    })
    if lg:
        lg.info("  Position encounter shift: direction=%s phase_speed=%.6f m/s shift=[%.4f, %.4f] s freq_ratio=[%.4f, %.4f]",
                info["direction_xy"], phase_speed, info["time_shift_min_s"], info["time_shift_max_s"],
                info["encounter_frequency_ratio_min"], info["encounter_frequency_ratio_max"])
    return shift, ratio, info


def vessel_wave_orientation_features_from_xy(t: np.ndarray,
                                             x_position: Optional[np.ndarray],
                                             y_position: Optional[np.ndarray],
                                             cfg: Dict[str, object],
                                             lg: Optional[logging.Logger] = None,
                                             run_segments: Optional[List[Tuple[int, int]]] = None,
                                             train_idx: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Approximate vessel course relative to waves from XY track.

    Velocity is resolved onto the configured wave-propagation direction and its
    perpendicular crest direction. The signs are retained so following and
    opposing motion can be learned.
    """
    info: Dict[str, object] = {
        "active": False,
        "coordinate_convention": "configured_wave_direction_and_perpendicular_crest",
    }
    if not bool(cfg.get("use_position_features", True)) or not bool(cfg.get("use_orientation_features", True)):
        return {}, info
    if x_position is None and y_position is None:
        return {}, info

    tt = np.asarray(t, dtype=float)
    x = np.zeros_like(tt) if x_position is None else np.asarray(x_position, dtype=float)
    y = np.zeros_like(tt) if y_position is None else np.asarray(y_position, dtype=float)
    if len(tt) < 2:
        return {}, info

    segs = normalise_segments(run_segments, len(tt))
    vx = gradient_time_by_segments(x, tt, segs)
    vy = gradient_time_by_segments(y, tt, segs)
    smooth_s = float(cfg.get("orientation_smoothing_seconds", 0.0))
    if smooth_s > 0.0:
        vx = moving_average_time_by_segments(vx, tt, smooth_s, segs)
        vy = moving_average_time_by_segments(vy, tt, smooth_s, segs)

    direction = np.asarray(cfg.get("wave_direction_xy", [1.0, 0.0]), dtype=float).reshape(-1)
    if direction.size != 2 or not np.all(np.isfinite(direction)):
        direction = np.asarray([1.0, 0.0], dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    direction = np.asarray([1.0, 0.0], dtype=float) if direction_norm < 1e-8 else direction / direction_norm
    crest_direction = np.asarray([-direction[1], direction[0]], dtype=float)

    velocity_perp = direction[0] * vx + direction[1] * vy
    velocity_parallel = crest_direction[0] * vx + crest_direction[1] * vy
    speed_xy = np.sqrt(vx ** 2 + vy ** 2)
    min_speed = float(max(cfg.get("orientation_min_speed_mps", 1.0e-5), 1.0e-8))
    denom = np.maximum(speed_xy, min_speed)
    heading_perp = np.clip(velocity_perp / denom, -1.0, 1.0)
    heading_parallel = np.clip(velocity_parallel / denom, -1.0, 1.0)
    obliquity = np.abs(heading_perp)

    features = {
        "track_speed_xy": speed_xy,
        "wave_perpendicular_velocity": velocity_perp,
        "wave_parallel_velocity": velocity_parallel,
        "heading_perpendicular_to_waves": heading_perp,
        "heading_parallel_to_waves": heading_parallel,
        "heading_obliquity_abs": obliquity,
    }
    orientation_gate_active = bool(cfg.get("wave_orientation_gate_enabled", True))
    min_gain = float(np.clip(float(cfg.get("wave_parallel_min_gain", 0.20)), 0.0, 1.0))
    blend = float(np.clip(float(cfg.get("wave_parallel_velocity_blend", 0.65)), 0.0, 1.0))
    power = float(max(1.0e-6, cfg.get("wave_parallel_obliquity_power", 1.50)))
    q = float(np.clip(float(cfg.get("wave_parallel_velocity_reference_quantile", 0.90)), 0.01, 0.99))
    perp_abs = np.abs(velocity_perp)
    ref_idx = np.asarray(train_idx, dtype=int) if train_idx is not None else np.arange(len(perp_abs), dtype=int)
    ref_idx = ref_idx[(ref_idx >= 0) & (ref_idx < len(perp_abs))]
    ref_values = perp_abs[ref_idx] if ref_idx.size else perp_abs
    ref_values = ref_values[np.isfinite(ref_values)]
    velocity_ref = float(np.quantile(ref_values, q)) if ref_values.size else float("nan")
    if not math.isfinite(velocity_ref) or velocity_ref < 1.0e-8:
        finite_perp = perp_abs[np.isfinite(perp_abs)]
        velocity_ref = float(np.max(finite_perp)) if finite_perp.size else 1.0
    velocity_ref = max(velocity_ref, 1.0e-8)
    velocity_drive = np.clip(perp_abs / velocity_ref, 0.0, 1.0)
    heading_drive = np.clip(obliquity, 0.0, 1.0)
    drive = np.clip((1.0 - blend) * heading_drive + blend * velocity_drive, 0.0, 1.0)
    orientation_gain = min_gain + (1.0 - min_gain) * (drive ** power)
    gate_smooth_s = float(cfg.get("wave_parallel_gate_smoothing_seconds", 0.0))
    if gate_smooth_s > 0.0:
        orientation_gain = moving_average_time_by_segments(orientation_gain, tt, gate_smooth_s, segs)
        orientation_gain = np.clip(orientation_gain, min_gain, 1.0)
    features["wave_orientation_effect_gain"] = orientation_gain

    info.update({
        "active": True,
        "wave_direction_xy": [float(direction[0]), float(direction[1])],
        "crest_direction_xy": [float(crest_direction[0]), float(crest_direction[1])],
        "track_speed_xy_min": float(np.min(speed_xy)),
        "track_speed_xy_max": float(np.max(speed_xy)),
        "perpendicular_velocity_min": float(np.min(velocity_perp)),
        "perpendicular_velocity_max": float(np.max(velocity_perp)),
        "parallel_velocity_min": float(np.min(velocity_parallel)),
        "parallel_velocity_max": float(np.max(velocity_parallel)),
        "obliquity_min": float(np.min(obliquity)),
        "obliquity_max": float(np.max(obliquity)),
    })
    info.update({
        "wave_orientation_gate_active": orientation_gate_active,
        "wave_orientation_gain_min": float(np.min(orientation_gain)),
        "wave_orientation_gain_max": float(np.max(orientation_gain)),
        "wave_orientation_gain_mean": float(np.mean(orientation_gain)),
        "wave_parallel_min_gain": min_gain,
        "wave_parallel_velocity_blend": blend,
        "wave_parallel_velocity_reference_mps": float(velocity_ref),
        "wave_parallel_velocity_reference_quantile": q,
        "wave_parallel_obliquity_power": power,
        "wave_parallel_gate_smoothing_seconds": gate_smooth_s,
        "wave_orientation_gate_note": (
            "1.0 means high wave-normal encounter; lower values indicate more "
            "nearly parallel encounter. In the current experiment this signal "
            "is retained as a feature/diagnostic, but is not hard-multiplied "
            "into pure wave force unless wave_orientation_gate_enabled is true."
        ),
    })
    if lg:
        lg.info(
            "  Track orientation from XY: wave_direction=%s crest_direction=%s | v_perp=[%.4f, %.4f] m/s v_parallel=[%.4f, %.4f] m/s obliquity=[%.4f, %.4f]",
            info["wave_direction_xy"], info["crest_direction_xy"],
            info["perpendicular_velocity_min"], info["perpendicular_velocity_max"],
            info["parallel_velocity_min"], info["parallel_velocity_max"],
            info["obliquity_min"], info["obliquity_max"],
        )
        lg.info(
            "  Wave-orientation force gain feature: active_gate=%s min=%.4f max=%.4f mean=%.4f | v_perp_ref=%.4f m/s blend=%.2f power=%.2f",
            info["wave_orientation_gate_active"],
            info["wave_orientation_gain_min"],
            info["wave_orientation_gain_max"],
            info["wave_orientation_gain_mean"],
            info["wave_parallel_velocity_reference_mps"],
            info["wave_parallel_velocity_blend"],
            info["wave_parallel_obliquity_power"],
        )
    return features, info


def estimate_best_wave_lag(vessel_t: np.ndarray, phi: np.ndarray, wave_t: np.ndarray, wave_raw: np.ndarray,
                           train_idx: np.ndarray, cfg: Dict[str, object], lg: logging.Logger,
                           vessel_segments: Optional[List[Tuple[int, int]]] = None,
                           wave_segments: Optional[List[Tuple[int, int]]] = None,
                           encounter_time_shift_s: Optional[np.ndarray] = None,
                           dominant_period_s: Optional[float] = None) -> Tuple[float, np.ndarray, np.ndarray, float]:
    v_segments = normalise_segments(vessel_segments, len(vessel_t))
    w_segments = normalise_segments(wave_segments, len(wave_t))
    dominant_period = (
        float(dominant_period_s)
        if dominant_period_s is not None
        else estimate_dominant_period_by_segments(phi, vessel_t, train_idx, v_segments)
    )
    roll_rate = robust_roll_rate_by_segments(phi, vessel_t, v_segments)
    roll_rate_bp = bandpass_time_by_segments(
        roll_rate, vessel_t, dominant_period,
        float(cfg["bandpass_low_period_factor"]), float(cfg["bandpass_high_period_factor"]), v_segments,
    )
    wave_slope_raw = gradient_time_by_segments(wave_raw, wave_t, w_segments)
    wave_proxy = bandpass_time_by_segments(
        wave_slope_raw, wave_t, dominant_period,
        float(cfg["bandpass_low_period_factor"]), float(cfg["bandpass_high_period_factor"]), w_segments,
    )

    lags = np.linspace(0.0, float(cfg["max_wave_lag_s"]), int(cfg["n_wave_lag_candidates"]))
    rr = roll_rate_bp[train_idx] - np.mean(roll_rate_bp[train_idx])
    rr_sd = max(float(np.std(rr)), 1e-10)
    best_lag = 0.0
    best_score = -np.inf
    scores: List[float] = []
    for lag in lags:
        aligned = interp_to_vessel_time_by_segments(
            vessel_t, wave_t, wave_proxy, lag_s=float(lag),
            encounter_time_shift_s=encounter_time_shift_s,
            vessel_segments=v_segments, source_segments=w_segments,
        )[train_idx]
        aligned = aligned - np.mean(aligned)
        a_sd = float(np.std(aligned))
        score = -np.inf if a_sd < 1e-10 else float(np.mean((aligned / a_sd) * (rr / rr_sd)))
        scores.append(score)
        if abs(score) > best_score:
            best_score = abs(score)
            best_lag = float(lag)
    lg.info("  Estimated causal wave-response lag: %.3f s | |corr|=%.4f | dominant roll period≈%.3f s",
            best_lag, best_score, dominant_period)
    return best_lag, lags, np.asarray(scores), dominant_period


def wave_to_force_lag_scores(vessel_t: np.ndarray,
                             inferred_force: np.ndarray,
                             wave_t: np.ndarray,
                             wave_raw: np.ndarray,
                             sample_idx: np.ndarray,
                             cfg: Dict[str, object],
                             vessel_segments: Optional[List[Tuple[int, int]]] = None,
                             wave_segments: Optional[List[Tuple[int, int]]] = None,
                             encounter_time_shift_s: Optional[np.ndarray] = None,
                             dominant_period_s: Optional[float] = None,
                             lag_grid_s: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    v_segments = normalise_segments(vessel_segments, len(vessel_t))
    w_segments = normalise_segments(wave_segments, len(wave_t))
    idx = np.asarray(sample_idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < len(vessel_t)) & (idx < len(inferred_force))]
    if lag_grid_s is None:
        max_lag = float(cfg.get("force_lag_diagnostic_max_s", cfg.get("max_wave_lag_s", 4.0)))
        n_lags = int(max(5, cfg.get("force_lag_diagnostic_candidates", 121)))
        lags = np.linspace(-max_lag, max_lag, n_lags)
    else:
        lags = np.asarray(lag_grid_s, dtype=float)
    scores = np.full(len(lags), np.nan, dtype=np.float64)
    if len(idx) < 8:
        return lags, scores
    dominant_period = (
        float(dominant_period_s)
        if dominant_period_s is not None and math.isfinite(float(dominant_period_s))
        else estimate_dominant_period_by_segments(inferred_force, vessel_t, idx, v_segments)
    )
    force_bp = bandpass_time_by_segments(
        inferred_force,
        vessel_t,
        dominant_period,
        float(cfg["bandpass_low_period_factor"]),
        float(cfg["bandpass_high_period_factor"]),
        v_segments,
    )
    wave_slope_raw = gradient_time_by_segments(wave_raw, wave_t, w_segments)
    wave_proxy = bandpass_time_by_segments(
        wave_slope_raw,
        wave_t,
        dominant_period,
        float(cfg["bandpass_low_period_factor"]),
        float(cfg["bandpass_high_period_factor"]),
        w_segments,
    )
    target = force_bp[idx]
    target = target - np.nanmean(target)
    target_sd = float(np.nanstd(target))
    if not math.isfinite(target_sd) or target_sd < 1.0e-10:
        return lags, scores
    target_z = target / target_sd
    for i, lag in enumerate(lags):
        aligned = interp_to_vessel_time_by_segments(
            vessel_t,
            wave_t,
            wave_proxy,
            lag_s=float(lag),
            encounter_time_shift_s=encounter_time_shift_s,
            vessel_segments=v_segments,
            source_segments=w_segments,
        )[idx]
        valid = np.isfinite(aligned) & np.isfinite(target_z)
        if int(np.sum(valid)) < 8:
            continue
        aligned = aligned[valid] - float(np.mean(aligned[valid]))
        aligned_sd = float(np.std(aligned))
        if not math.isfinite(aligned_sd) or aligned_sd < 1.0e-10:
            continue
        scores[i] = float(np.mean((aligned / aligned_sd) * target_z[valid]))
    return lags, scores


def wave_proxy_to_force_lag_scores(vessel_t: np.ndarray,
                                   inferred_force: np.ndarray,
                                   wave_t: np.ndarray,
                                   wave_proxy_raw: np.ndarray,
                                   sample_idx: np.ndarray,
                                   cfg: Dict[str, object],
                                   vessel_segments: Optional[List[Tuple[int, int]]] = None,
                                   wave_segments: Optional[List[Tuple[int, int]]] = None,
                                   encounter_time_shift_s: Optional[np.ndarray] = None,
                                   dominant_period_s: Optional[float] = None,
                                   lag_grid_s: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    v_segments = normalise_segments(vessel_segments, len(vessel_t))
    w_segments = normalise_segments(wave_segments, len(wave_t))
    idx = np.asarray(sample_idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < len(vessel_t)) & (idx < len(inferred_force))]
    if lag_grid_s is None:
        max_lag = float(cfg.get("force_lag_diagnostic_max_s", cfg.get("max_wave_lag_s", 4.0)))
        n_lags = int(max(5, cfg.get("force_lag_diagnostic_candidates", 121)))
        lags = np.linspace(-max_lag, max_lag, n_lags)
    else:
        lags = np.asarray(lag_grid_s, dtype=float)
    scores = np.full(len(lags), np.nan, dtype=np.float64)
    proxy_raw = np.asarray(wave_proxy_raw, dtype=float)
    if len(idx) < 8 or len(wave_t) == 0 or proxy_raw.shape[:1] != np.asarray(wave_t).shape[:1]:
        return lags, scores
    dominant_period = (
        float(dominant_period_s)
        if dominant_period_s is not None and math.isfinite(float(dominant_period_s))
        else estimate_dominant_period_by_segments(inferred_force, vessel_t, idx, v_segments)
    )
    force_bp = bandpass_time_by_segments(
        inferred_force,
        vessel_t,
        dominant_period,
        float(cfg["bandpass_low_period_factor"]),
        float(cfg["bandpass_high_period_factor"]),
        v_segments,
    )
    proxy_bp = bandpass_time_by_segments(
        proxy_raw,
        wave_t,
        dominant_period,
        float(cfg["bandpass_low_period_factor"]),
        float(cfg["bandpass_high_period_factor"]),
        w_segments,
    )
    target = force_bp[idx]
    target = target - np.nanmean(target)
    target_sd = float(np.nanstd(target))
    if not math.isfinite(target_sd) or target_sd < 1.0e-10:
        return lags, scores
    target_z = target / target_sd
    for i, lag in enumerate(lags):
        aligned = interp_to_vessel_time_by_segments(
            vessel_t,
            wave_t,
            proxy_bp,
            lag_s=float(lag),
            encounter_time_shift_s=encounter_time_shift_s,
            vessel_segments=v_segments,
            source_segments=w_segments,
        )[idx]
        valid = np.isfinite(aligned) & np.isfinite(target_z)
        if int(np.sum(valid)) < 8:
            continue
        aligned = aligned[valid] - float(np.mean(aligned[valid]))
        aligned_sd = float(np.std(aligned))
        if not math.isfinite(aligned_sd) or aligned_sd < 1.0e-10:
            continue
        scores[i] = float(np.mean((aligned / aligned_sd) * target_z[valid]))
    return lags, scores


def build_wave_force_lag_proxies(data: Dict[str, object]) -> Dict[str, np.ndarray]:
    wave_t = np.asarray(data.get("wave_time_raw", []), dtype=float)
    wave_signal = np.asarray(data.get("wave_signal_raw", []), dtype=float)
    if len(wave_t) == 0 or wave_signal.shape[:1] != wave_t.shape[:1]:
        return {}
    wave_segments = normalise_segments(data.get("wave_run_segments"), len(wave_t))
    wave_cross = np.asarray(
        data.get("wave_cross_beam_raw", np.zeros_like(wave_signal)),
        dtype=float,
    )
    if wave_cross.shape[:1] != wave_signal.shape[:1]:
        wave_cross = np.zeros_like(wave_signal)
    wave_slope = gradient_time_by_segments(wave_signal, wave_t, wave_segments)
    wave_cross_slope = gradient_time_by_segments(wave_cross, wave_t, wave_segments)
    cross_combo = wave_cross + 0.25 * wave_cross_slope
    proxies = {
        "wave signal": wave_signal,
        "-wave signal": -wave_signal,
        "wave slope": wave_slope,
        "-wave slope": -wave_slope,
        "cross beam": wave_cross,
        "-cross beam": -wave_cross,
        "cross beam + slope": cross_combo,
        "-cross beam - slope": -cross_combo,
    }
    return {
        name: arr for name, arr in proxies.items()
        if np.asarray(arr).shape[:1] == wave_t.shape[:1]
        and np.any(np.isfinite(np.asarray(arr, dtype=float)))
    }


def multi_wave_proxy_force_lag_scan(data: Dict[str, object],
                                    inferred_force: np.ndarray,
                                    cfg: Dict[str, object],
                                    current_wave_lag_s: float,
                                    base_lags_s: Optional[np.ndarray] = None) -> Dict[str, object]:
    vessel_t = np.asarray(data.get("t", []), dtype=float)
    wave_t = np.asarray(data.get("wave_time_raw", []), dtype=float)
    if len(vessel_t) == 0 or len(wave_t) == 0:
        return {
            "note": "Unavailable: missing vessel or wave time arrays.",
            "current_wave_lag_s": float(current_wave_lag_s),
            "lags_s": np.asarray([], dtype=float),
            "proxies": {},
        }
    train_idx = np.asarray(data.get("train_idx", []), dtype=int)
    val_idx = np.asarray(data.get("val_idx", []), dtype=int)
    vessel_segments = normalise_segments(data.get("run_segments"), len(vessel_t))
    wave_segments = normalise_segments(data.get("wave_run_segments"), len(wave_t))
    encounter_shift = np.asarray(
        data.get("encounter_time_shift_s", np.zeros(len(vessel_t))),
        dtype=float,
    )
    if encounter_shift.shape[:1] != vessel_t.shape[:1]:
        encounter_shift = np.zeros(len(vessel_t), dtype=float)
    dominant_period_s = float(data.get("dominant_roll_period_s", cfg.get("dominant_roll_period_s", 1.0)))
    proxies_raw = build_wave_force_lag_proxies(data)
    lag_grid = (
        np.asarray(base_lags_s, dtype=float)
        if base_lags_s is not None and len(np.asarray(base_lags_s, dtype=float)) > 0
        else None
    )
    proxy_scans: Dict[str, Dict[str, object]] = {}
    common_lags = np.asarray([], dtype=float)
    for name, proxy_raw in proxies_raw.items():
        train_lags, train_scores = wave_proxy_to_force_lag_scores(
            vessel_t,
            inferred_force,
            wave_t,
            proxy_raw,
            train_idx,
            cfg,
            vessel_segments,
            wave_segments,
            encounter_shift,
            dominant_period_s,
            lag_grid,
        )
        val_lags, val_scores = wave_proxy_to_force_lag_scores(
            vessel_t,
            inferred_force,
            wave_t,
            proxy_raw,
            val_idx,
            cfg,
            vessel_segments,
            wave_segments,
            encounter_shift,
            dominant_period_s,
            train_lags,
        )
        common_lags = train_lags
        proxy_scans[name] = {
            "train_scores": train_scores,
            "validation_scores": val_scores,
            "train": force_lag_scan_summary(train_lags, train_scores, current_wave_lag_s),
            "validation": force_lag_scan_summary(val_lags, val_scores, current_wave_lag_s),
        }
    return {
        "note": (
            "Band-passed candidate wave proxies scanned directly against inferred "
            "measured force; sign-flipped proxies diagnose half-period/sign ambiguity."
        ),
        "current_wave_lag_s": float(current_wave_lag_s),
        "lags_s": common_lags,
        "proxies": proxy_scans,
    }


def force_lag_scan_summary(lags: np.ndarray,
                           scores: np.ndarray,
                           current_lag_s: float) -> Dict[str, float]:
    lag_arr = np.asarray(lags, dtype=float)
    score_arr = np.asarray(scores, dtype=float)
    finite = np.isfinite(lag_arr) & np.isfinite(score_arr)
    if not np.any(finite):
        return {
            "best_lag_s": float("nan"),
            "best_corr": float("nan"),
            "current_lag_s": float(current_lag_s),
            "current_corr": float("nan"),
            "lag_difference_s": float("nan"),
        }
    lag_f = lag_arr[finite]
    score_f = score_arr[finite]
    best_i = int(np.argmax(np.abs(score_f)))
    current_corr = float(np.interp(float(current_lag_s), lag_f, score_f))
    return {
        "best_lag_s": float(lag_f[best_i]),
        "best_corr": float(score_f[best_i]),
        "best_abs_corr": float(abs(score_f[best_i])),
        "current_lag_s": float(current_lag_s),
        "current_corr": current_corr,
        "current_abs_corr": float(abs(current_corr)),
        "lag_difference_s": float(lag_f[best_i] - float(current_lag_s)),
    }


def encode_time(t_norm: np.ndarray, freq_multipliers: Iterable[int]) -> np.ndarray:
    feats = [t_norm.reshape(-1, 1)]
    for k in freq_multipliers:
        arg = float(k) * math.pi * t_norm.reshape(-1, 1)
        feats.append(np.sin(arg))
        feats.append(np.cos(arg))
    return np.concatenate(feats, axis=1)


def active_wave_feature_names(cfg: Dict[str, object]) -> List[str]:
    valid = [
        "wave_signal",
        "wave_slope",
        "wave_abs_slope",
        "wave_curvature",
        "wave_abs_curvature",
        "wave_signal_slope_product",
        "wave_envelope",
        "wave_envelope_slow",
        "wave_cross_beam_gradient",
    ]
    names = [str(n) for n in cfg.get("wave_feature_set", ["wave_signal", "wave_slope"]) if str(n) in valid]
    return names if names else ["wave_signal", "wave_slope"]


def scale_train_region(arr: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(arr, dtype=float)
    mu = float(np.mean(arr[train_idx]))
    sd = max(float(np.std(arr[train_idx])), 1e-8)
    return (arr - mu) / sd, mu, sd


def motion_feedback_delay_steps(cfg: Dict[str, object], dt_s: float) -> List[int]:
    raw = cfg.get("motion_feedback_delay_offsets_s", None)
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        delays_s = [float(v) for v in raw]
    else:
        delays_s = [float(cfg.get("motion_feedback_delay_s", 0.2))]
    steps = [int(max(0, round(delay_s / max(float(dt_s), 1.0e-8)))) for delay_s in delays_s]
    return sorted(set(steps))


def effective_rollout_window_s(cfg: Dict[str, object], feedback_active: bool) -> float:
    window_s = float(max(0.0, cfg.get("rollout_window_s", 5.08)))
    if not feedback_active:
        return window_s
    feedback_window_s = float(max(0.0, cfg.get("motion_feedback_rollout_window_s", window_s)))
    return min(window_s, feedback_window_s)


def effective_rollout_batch_size(cfg: Dict[str, object], batch_size: int,
                                 feedback_active: bool) -> int:
    batch_size = max(1, int(batch_size))
    if not feedback_active:
        return batch_size
    feedback_batch = max(1, int(cfg.get("motion_feedback_rollout_batch_size", batch_size)))
    return min(batch_size, feedback_batch)


def unit_checks_raw(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray]) -> None:
    if len(t) != len(phi):
        raise ValueError("Ship time and roll arrays must have the same length.")
    if len(t) < 16:
        raise ValueError("Too few vessel samples for sequence modelling.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(phi)):
        raise ValueError("Non-finite values found in vessel time or roll.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Time must be strictly increasing after cleaning.")
    wt = np.asarray(wave_pack["wave_time_raw"], dtype=float)
    wy = np.asarray(wave_pack["wave_signal_raw"], dtype=float)
    if len(wt) != len(wy) or len(wt) < 16:
        raise ValueError("Wave_Time and wave signal must have the same useful length.")
    if not np.all(np.isfinite(wt)) or not np.all(np.isfinite(wy)):
        raise ValueError("Non-finite values found in wave time or signal.")
    if np.any(np.diff(wt) <= 0.0):
        raise ValueError("Wave_Time must be strictly increasing after cleaning.")


def apply_data_time_window(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                           cfg: Dict[str, object], lg: logging.Logger) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    t_arr = np.asarray(t, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    trim_final_s = float(max(0.0, cfg.get("data_trim_final_s", 0.0)))
    disabled = bool(cfg.get("disable_data_time_window", False))
    if disabled and trim_final_s <= 0.0:
        if cfg.get("data_time_start_s", None) is not None or cfg.get("data_time_end_s", None) is not None:
            lg.info(
                "Data time-window truncation is disabled; ignoring configured start/end window."
            )
        return t, phi, wave_pack, {
            "active": False,
            "disabled": True,
            "start_s": None,
            "end_s": None,
            "reason": "disable_data_time_window is true",
        }
    if disabled:
        if cfg.get("data_time_start_s", None) is not None or cfg.get("data_time_end_s", None) is not None:
            lg.info(
                "Data time-window truncation is disabled; ignoring configured start/end window before applying final trim."
            )
        start_s = None
        end_s = None
    else:
        start_s = cfg.get("data_time_start_s", None)
        end_s = cfg.get("data_time_end_s", None)
    trim_end_s = None
    if trim_final_s > 0.0:
        trim_end_s = float(t_arr[-1]) - trim_final_s
        end_s = trim_end_s if end_s is None else min(float(end_s), trim_end_s)
    if start_s is None and end_s is None:
        return t, phi, wave_pack, {"active": False, "start_s": None, "end_s": None}

    lo = -float("inf") if start_s is None else float(start_s)
    hi = float("inf") if end_s is None else float(end_s)
    if hi < lo:
        raise ValueError(f"Invalid data time window: end {hi} s is before start {lo} s.")

    vessel_mask = (t_arr >= lo) & (t_arr <= hi)
    if int(np.count_nonzero(vessel_mask)) < 16:
        raise ValueError(
            f"Data time window [{lo}, {hi}] s leaves too few vessel samples "
            f"({int(np.count_nonzero(vessel_mask))})."
        )

    out_pack: Dict[str, np.ndarray] = {}
    wave_sample_keys = {
        "wave_time_raw", "wave_signal_raw", "wave_1_raw", "wave_2_raw",
        "wave_midpoint_raw", "wave_cross_beam_raw", "wave_run_local_time",
    }
    metadata_keys = {"run_names", "run_segments", "wave_run_segments", "forecast_run_name", "training_run_names"}
    for key, value in wave_pack.items():
        arr = np.asarray(value)
        if key in wave_sample_keys or key in metadata_keys:
            continue
        if arr.shape[:1] == t_arr.shape[:1]:
            out_pack[key] = arr[vessel_mask]
        else:
            out_pack[key] = arr.copy()

    wave_t = np.asarray(wave_pack["wave_time_raw"], dtype=float)
    wave_raw = np.asarray(wave_pack["wave_signal_raw"], dtype=float)
    wave_mask = (wave_t >= lo) & (wave_t <= hi)
    if int(np.count_nonzero(wave_mask)) < 16:
        raise ValueError(
            f"Data time window [{lo}, {hi}] s leaves too few wave samples "
            f"({int(np.count_nonzero(wave_mask))})."
        )
    out_pack["wave_time_raw"] = wave_t[wave_mask]
    out_pack["wave_signal_raw"] = wave_raw[wave_mask]
    for key in ("wave_1_raw", "wave_2_raw", "wave_midpoint_raw", "wave_cross_beam_raw", "wave_run_local_time"):
        if key in wave_pack:
            arr = np.asarray(wave_pack[key])
            if arr.shape[:1] == wave_t.shape[:1]:
                out_pack[key] = arr[wave_mask]
            else:
                out_pack[key] = arr.copy()

    t_out = t_arr[vessel_mask]
    phi_out = phi_arr[vessel_mask]
    orig_run_segments, orig_run_names = run_segments_from_pack(wave_pack, len(t_arr))
    orig_wave_segments = wave_segments_from_pack(wave_pack, len(wave_t), len(orig_run_segments))
    if "run_segments" in wave_pack or len(orig_run_segments) > 1:
        old_v_to_new = np.full(len(t_arr), -1, dtype=int)
        old_v_to_new[np.flatnonzero(vessel_mask)] = np.arange(len(t_out), dtype=int)
        old_w_to_new = np.full(len(wave_t), -1, dtype=int)
        old_w_to_new[np.flatnonzero(wave_mask)] = np.arange(len(out_pack["wave_time_raw"]), dtype=int)
        new_run_segments: List[Tuple[int, int]] = []
        new_wave_segments: List[Tuple[int, int]] = []
        new_run_names: List[str] = []
        for name, (lo, hi), (wlo, whi) in zip(orig_run_names, orig_run_segments, orig_wave_segments):
            v_idx = np.arange(int(lo), int(hi) + 1, dtype=int)
            w_idx = np.arange(int(wlo), int(whi) + 1, dtype=int)
            kept_v = v_idx[vessel_mask[v_idx]]
            kept_w = w_idx[wave_mask[w_idx]]
            if kept_v.size == 0 or kept_w.size == 0:
                continue
            new_run_segments.append((int(old_v_to_new[int(kept_v[0])]), int(old_v_to_new[int(kept_v[-1])])))
            new_wave_segments.append((int(old_w_to_new[int(kept_w[0])]), int(old_w_to_new[int(kept_w[-1])])))
            new_run_names.append(str(name))
        if new_run_segments:
            out_pack["run_segments"] = np.asarray(new_run_segments, dtype=int)
            out_pack["wave_run_segments"] = np.asarray(new_wave_segments, dtype=int)
            out_pack["run_names"] = np.asarray(new_run_names, dtype=object)
    for key in ("forecast_run_name", "training_run_names"):
        if key in wave_pack:
            out_pack[key] = np.asarray(wave_pack[key]).copy()
    info = {
        "active": True,
        "disabled_manual_window": bool(disabled),
        "start_s": None if start_s is None else float(start_s),
        "end_s": None if end_s is None else float(end_s),
        "trim_final_s": float(trim_final_s),
        "trim_end_s": trim_end_s,
        "vessel_points_before": int(len(t_arr)),
        "vessel_points_after": int(len(t_out)),
        "wave_points_before": int(len(wave_t)),
        "wave_points_after": int(len(out_pack["wave_time_raw"])),
        "actual_start_s": float(t_out[0]),
        "actual_end_s": float(t_out[-1]),
    }
    lg.info(
        "  Data time window:     %.6g-%.6g s | trim_final=%.6g s | vessel %d -> %d | wave %d -> %d",
        lo,
        hi,
        trim_final_s,
        info["vessel_points_before"],
        info["vessel_points_after"],
        info["wave_points_before"],
        info["wave_points_after"],
    )
    return t_out, phi_out, out_pack, info


DATA_PREP_CACHE_KEYS = (
    "seed",
    "device",
    "disable_data_time_window",
    "data_time_start_s",
    "data_time_end_s",
    "data_trim_final_s",
    "excel_run_sheets",
    "forecast_sheet",
    "training_sheets",
    "run_gap_s",
    "validation_window_s",
    "val_frac",
    "forecast_frac",
    "forecast_window_s",
    "forecast_start_s",
    "freq_multipliers",
    "wave_feature_set",
    "fixed_wave_lag_s",
    "fixed_wave_lag_source",
    "wave_lag_feature_offsets_s",
    "max_wave_lag_s",
    "n_wave_lag_candidates",
    "bandpass_low_period_factor",
    "bandpass_high_period_factor",
    "wave_envelope_seconds",
    "wave_envelope_slow_seconds",
    "use_position_features",
    "apply_position_encounter_shift",
    "wave_direction_xy",
    "wave_phase_speed_mps",
    "encounter_position_reference",
    "encounter_time_shift_clip_s",
    "use_orientation_features",
    "orientation_smoothing_seconds",
    "orientation_min_speed_mps",
    "use_motion_feedback",
    "motion_feedback_delay_s",
    "motion_feedback_delay_offsets_s",
    "roll_rate_smoothing_seconds",
    "scale_roll_rate_target",
    "stride",
)


def data_prep_cache_key(cfg: Dict[str, object]) -> str:
    payload = {key: cfg.get(key) for key in DATA_PREP_CACHE_KEYS}
    val_frac = min(max(float(cfg.get("val_frac", 0.0)), 0.0), 0.9)
    val_window = cfg.get("validation_window_s", None)
    val_window_active = val_window is not None and float(val_window) > 0.0
    payload["seq_len_for_split"] = int(cfg["seq_len"]) if (val_frac > 0.0 or val_window_active) else 0
    return json.dumps(payload, sort_keys=True, default=str)


def rough_data_payload_mb(data: Dict[str, object]) -> float:
    seen: set[int] = set()
    total = 0
    for value in data.values():
        if isinstance(value, np.ndarray):
            ptr = int(value.__array_interface__.get("data", (0,))[0])
            if ptr and ptr in seen:
                continue
            if ptr:
                seen.add(ptr)
            total += int(value.nbytes)
    return float(total / (1024 ** 2))


class BayesDataCache:
    def __init__(self, t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray]):
        self.t = t
        self.phi = phi
        self.wave_pack = wave_pack
        self.prepared: Dict[str, Dict[str, object]] = {}
        self.loader_cache: Dict[object, object] = {}

    @staticmethod
    def _cache_keys_with_prefix(cache: Dict[object, object], prefix: str) -> List[object]:
        return [key for key in cache if isinstance(key, tuple) and len(key) > 0 and key[0] == prefix]

    def _prune_prepared(self, active_key: str, cfg: Dict[str, object], lg: logging.Logger) -> None:
        max_entries = max(1, int(cfg.get("bayes_prepared_data_cache_max_entries", 1)))
        while len(self.prepared) > max_entries:
            stale_key = next((key for key in self.prepared if key != active_key), None)
            if stale_key is None:
                break
            self.prepared.pop(stale_key, None)
            lg.info("[BO] Released stale prepared dataset from RAM cache (entries=%d/%d).",
                    len(self.prepared), max_entries)

    def _prune_tensor_cache(self, cfg: Dict[str, object], lg: Optional[logging.Logger] = None) -> None:
        max_entries = max(0, int(cfg.get("bayes_tensor_cache_max_entries", 1)))
        tensor_keys = self._cache_keys_with_prefix(self.loader_cache, "tensors")
        removed = 0
        while len(tensor_keys) > max_entries:
            stale_key = tensor_keys.pop(0)
            self.loader_cache.pop(stale_key, None)
            removed += 1
        if lg is not None and removed > 0:
            lg.info("[BO] Released %d stale tensor cache entries (kept=%d).", removed, len(tensor_keys))

    def get_prepared(self, cfg: Dict[str, object], lg: logging.Logger) -> Dict[str, object]:
        key = data_prep_cache_key(cfg)
        cached = self.prepared.get(key)
        if cached is not None:
            self.prepared[key] = self.prepared.pop(key)
            self._prune_prepared(key, cfg, lg)
            lg.info("[BO] Reusing prepared dataset from RAM cache (%.2f MB).", rough_data_payload_mb(cached))
            return cached
        data = prepare_data(self.t, self.phi, self.wave_pack, cfg, lg)
        self.prepared[key] = data
        self._prune_prepared(key, cfg, lg)
        lg.info("[BO] Cached prepared dataset in RAM for reuse (%.2f MB; entries=%d).",
                rough_data_payload_mb(data), len(self.prepared))
        return data

    def loader_shared_cache(self, cfg: Dict[str, object]) -> Optional[Dict[object, object]]:
        if not bool(cfg.get("bayes_reuse_loader_cache", True)):
            return None
        return self.loader_cache

    def release_between_trials(self, cfg: Dict[str, object], lg: logging.Logger) -> None:
        window_keys = self._cache_keys_with_prefix(self.loader_cache, "windows")
        cached_tensors = [
            value
            for cached in self.loader_cache.values()
            if isinstance(cached, dict)
            for value in cached.values()
            if isinstance(value, torch.Tensor)
        ]
        cache_is_cpu = not any(value.is_cuda for value in cached_tensors)
        keep_windows = bool(cfg.get("bayes_keep_window_cache_between_trials", False)) and cache_is_cpu
        removed_windows = 0
        if not keep_windows:
            for key in window_keys:
                self.loader_cache.pop(key, None)
                removed_windows += 1
        else:
            max_windows = max(1, int(cfg.get("bayes_window_cache_max_entries", 8)))
            while len(window_keys) > max_windows:
                stale_key = window_keys.pop(0)
                self.loader_cache.pop(stale_key, None)
                removed_windows += 1
        self._prune_tensor_cache(cfg, lg)
        if HAS_PLT:
            plt.close("all")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if removed_windows:
            lg.info("[BO] Released %d precomputed window cache entries after trial; prepared datasets kept=%d tensor caches kept=%d.",
                    removed_windows,
                    len(self.prepared),
                    len(self._cache_keys_with_prefix(self.loader_cache, "tensors")))
        elif keep_windows and window_keys:
            lg.info("[BO] Retained %d/%d CPU window-cache entries in RAM for reuse.",
                    len(window_keys), int(cfg.get("bayes_window_cache_max_entries", 8)))


def window_starts_ending_at_or_before(n_points: int, seq_len: int, stride: int, end_idx: int) -> np.ndarray:
    n_points = int(n_points)
    seq_len = int(max(2, seq_len))
    stride = int(max(1, stride))
    end_idx = int(min(max(0, end_idx), n_points - 1))
    last_start = end_idx - seq_len + 1
    if n_points <= 0 or last_start < 0:
        return np.asarray([], dtype=int)
    starts = list(range(0, last_start + 1, stride))
    if last_start not in starts:
        starts.append(last_start)
    return np.asarray(sorted(set(int(s) for s in starts)), dtype=int)


def point_indices_from_window_starts(starts: np.ndarray, seq_len: int, n_points: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=int)
    if starts.size == 0:
        return np.asarray([], dtype=int)
    mask = np.zeros(int(n_points), dtype=bool)
    seq_len = int(max(1, seq_len))
    for s in starts:
        lo = int(max(0, s))
        hi = int(min(n_points, lo + seq_len))
        if hi > lo:
            mask[lo:hi] = True
    return np.flatnonzero(mask).astype(int)


def augment_window_starts_for_turning_points(
    t: np.ndarray,
    phi: np.ndarray,
    base_starts: np.ndarray,
    train_idx: np.ndarray,
    seq_len: int,
    cfg: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    base = np.asarray(base_starts, dtype=int).reshape(-1)
    if base.size == 0 or not bool(cfg.get("turning_point_sampling_enabled", False)):
        return base, {"enabled": False, "base_windows": int(base.size), "extra_windows": 0}

    tt = np.asarray(t, dtype=float).reshape(-1)
    yy = np.asarray(phi, dtype=float).reshape(-1)
    n_points = int(min(tt.size, yy.size))
    seq_len = int(max(2, seq_len))
    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    if n_points < 3 or train_idx.size == 0:
        return base, {"enabled": False, "base_windows": int(base.size), "extra_windows": 0}

    train_mask = np.zeros(n_points, dtype=bool)
    train_mask[train_idx[(train_idx >= 0) & (train_idx < n_points)]] = True
    finite = np.isfinite(tt[:n_points]) & np.isfinite(yy[:n_points]) & train_mask
    if int(np.count_nonzero(finite)) < seq_len:
        return base, {"enabled": False, "base_windows": int(base.size), "extra_windows": 0}

    dy_prev = yy[1:-1] - yy[:-2]
    dy_next = yy[2:] - yy[1:-1]
    turning = ((dy_prev > 0.0) & (dy_next <= 0.0)) | ((dy_prev < 0.0) & (dy_next >= 0.0))
    candidate_idx = np.flatnonzero(turning).astype(int) + 1
    candidate_idx = candidate_idx[finite[candidate_idx]]
    if candidate_idx.size == 0:
        return base, {"enabled": True, "base_windows": int(base.size), "extra_windows": 0}

    neighbourhood = int(max(1, cfg.get("turning_point_sampling_neighbourhood", cfg.get("turning_point_neighbourhood", 4))))
    min_prom = math.radians(float(max(0.0, cfg.get("turning_point_sampling_min_prominence_deg", 0.0))))
    selected: List[int] = []
    for idx in candidate_idx:
        lo = int(max(0, idx - neighbourhood))
        hi = int(min(n_points - 1, idx + neighbourhood))
        if hi <= lo or not np.all(finite[lo:hi + 1]):
            continue
        left = yy[lo:idx + 1]
        right = yy[idx:hi + 1]
        if dy_prev[idx - 1] > 0.0:
            prominence = min(float(yy[idx] - np.min(left)), float(yy[idx] - np.min(right)))
        else:
            prominence = min(float(np.max(left) - yy[idx]), float(np.max(right) - yy[idx]))
        if prominence >= min_prom:
            selected.append(int(idx))

    if not selected:
        return base, {
            "enabled": True,
            "base_windows": int(base.size),
            "turning_points": int(candidate_idx.size),
            "selected_turning_points": 0,
            "extra_windows": 0,
        }

    fracs = config_float_list(cfg, "turning_point_sampling_window_fracs", [0.5])
    fracs = [float(np.clip(frac, 0.0, 1.0)) for frac in fracs] or [0.5]
    repeats = int(max(0, cfg.get("turning_point_sampling_repeats", 0)))
    extra_unique: List[int] = []
    seen_extra: set[int] = set()
    for idx in selected:
        for frac in fracs:
            start = int(round(float(idx) - frac * float(seq_len - 1)))
            start = int(np.clip(start, 0, max(0, n_points - seq_len)))
            end = start + seq_len
            if end > n_points or not np.all(train_mask[start:end]):
                continue
            if start not in seen_extra:
                seen_extra.add(start)
                extra_unique.append(start)

    if repeats <= 0 or not extra_unique:
        return base, {
            "enabled": True,
            "base_windows": int(base.size),
            "turning_points": int(candidate_idx.size),
            "selected_turning_points": int(len(selected)),
            "extra_unique_windows": int(len(extra_unique)),
            "extra_windows": 0,
        }

    extra = np.repeat(np.asarray(extra_unique, dtype=int), repeats)
    augmented = np.sort(np.concatenate([base, extra])).astype(int)
    return augmented, {
        "enabled": True,
        "base_windows": int(base.size),
        "turning_points": int(candidate_idx.size),
        "selected_turning_points": int(len(selected)),
        "extra_unique_windows": int(len(extra_unique)),
        "repeat_factor": int(repeats),
        "extra_windows": int(extra.size),
        "total_windows": int(augmented.size),
    }


def window_starts_within_segment(start_idx: int, end_idx: int, seq_len: int, stride: int) -> np.ndarray:
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    seq_len = int(max(2, seq_len))
    stride = int(max(1, stride))
    last_start = int(end_idx) - seq_len + 1
    if last_start < start_idx:
        return np.asarray([], dtype=int)
    starts = list(range(start_idx, last_start + 1, stride))
    if last_start not in starts:
        starts.append(last_start)
    return np.asarray(sorted(set(int(s) for s in starts)), dtype=int)


def split_preforecast_windows(t: np.ndarray, cfg: Dict[str, object], lg: logging.Logger,
                              run_segments: Optional[List[Tuple[int, int]]] = None,
                              run_names: Optional[List[str]] = None) -> Dict[str, object]:
    tt = np.asarray(t, dtype=float)
    n_points = int(len(tt))
    seq_len = int(cfg["seq_len"])
    stride = int(max(1, cfg["stride"]))
    segments = normalise_segments(run_segments, n_points)
    names = list(run_names or [f"run_{i + 1}" for i in range(len(segments))])
    if len(names) < len(segments):
        names.extend(f"run_{i + 1}" for i in range(len(names), len(segments)))
    names = names[:len(segments)]

    name_lowers = {name.lower() for name in names}
    forecast_name = str(cfg.get("forecast_sheet", names[0] if names else "")).strip()
    forecast_name_l = forecast_name.lower()
    if forecast_name_l not in name_lowers:
        forecast_name_l = names[0].lower() if names else ""

    requested_training = _cfg_string_list(cfg, "training_sheets")
    requested_training_l = {name.lower() for name in requested_training}
    if not requested_training_l or not any(name.lower() in requested_training_l for name in names):
        requested_training_l = set(name_lowers)

    forecast_start_idx = n_points - 1
    forecast_end_idx = n_points - 1
    forecast_enabled = False
    forecast_region: Optional[Tuple[int, int]] = None
    train_start_parts: List[np.ndarray] = []
    val_start_parts: List[np.ndarray] = []
    val_mask = np.zeros(n_points, dtype=bool)
    val_frac = min(max(float(cfg.get("val_frac", 0.0)), 0.0), 0.9)
    validation_window_cfg = cfg.get("validation_window_s", None)
    validation_window_s = (
        None if validation_window_cfg is None else max(0.0, float(validation_window_cfg))
    )
    candidate_windows = 0
    preforecast_points = 0
    run_infos: List[Dict[str, object]] = []

    for name, (lo, hi) in zip(names, segments):
        lo_i = int(lo)
        hi_i = int(hi)
        is_forecast_run = name.lower() == forecast_name_l
        if is_forecast_run:
            local_region = choose_forecast_region(tt[lo_i:hi_i + 1], cfg)
            if local_region is not None:
                local_start, local_end = local_region
                forecast_start_idx = lo_i + int(local_start)
                forecast_end_idx = lo_i + int(local_end)
                forecast_region = (int(forecast_start_idx), int(forecast_end_idx))
                forecast_enabled = True
                fit_end_idx_run = int(max(lo_i, forecast_start_idx - 1))
            else:
                fit_end_idx_run = hi_i
        else:
            fit_end_idx_run = hi_i

        is_training_run = name.lower() in requested_training_l
        candidate_starts = (
            window_starts_within_segment(lo_i, fit_end_idx_run, seq_len, stride)
            if is_training_run else np.asarray([], dtype=int)
        )
        candidate_windows += int(candidate_starts.size)
        fit_points_run = int(max(0, fit_end_idx_run - lo_i + 1))
        preforecast_points += fit_points_run if is_training_run else 0
        run_val_starts = np.asarray([], dtype=int)
        run_train_starts = candidate_starts
        train_end_idx_run = fit_end_idx_run
        val_start_idx_run: Optional[int] = None
        mode = "train_all_pre_forecast_windows"
        if (
            is_training_run
            and ((validation_window_s is not None and validation_window_s > 0.0) or val_frac > 0.0)
            and candidate_starts.size > 1
        ):
            if validation_window_s is not None and validation_window_s > 0.0:
                local_fit_t = tt[lo_i:fit_end_idx_run + 1]
                val_start_t = float(tt[fit_end_idx_run]) - float(validation_window_s)
                proposed_val_start = lo_i + int(np.searchsorted(local_fit_t, val_start_t, side="left"))
                requested_val_points = int(max(1, fit_end_idx_run - proposed_val_start + 1))
            else:
                # val_frac is a fraction of the complete run, not of the
                # remaining pre-forecast subset.
                full_run_points = int(hi_i - lo_i + 1)
                requested_val_points = int(max(1, round(full_run_points * val_frac)))
            requested_val_points = int(min(requested_val_points, max(0, fit_points_run - seq_len)))
            proposed_val_start = int(fit_end_idx_run - requested_val_points + 1)
            proposed_train_end = int(proposed_val_start - 1)
            train_points_run = int(max(0, proposed_train_end - lo_i + 1))
            if requested_val_points >= seq_len and train_points_run >= seq_len:
                train_end_idx_run = proposed_train_end
                val_start_idx_run = proposed_val_start
                run_train_starts = window_starts_within_segment(
                    lo_i, train_end_idx_run, seq_len, stride
                )
                run_val_starts = window_starts_within_segment(
                    val_start_idx_run, fit_end_idx_run, seq_len, stride
                )
                val_mask[val_start_idx_run:fit_end_idx_run + 1] = True
                mode = "contiguous_chronological_train_then_validation"

        if run_train_starts.size > 0:
            train_start_parts.append(run_train_starts)
        if run_val_starts.size > 0:
            val_start_parts.append(run_val_starts)
        run_infos.append({
            "name": name,
            "start_idx": lo_i,
            "end_idx": hi_i,
            "start_time_s": float(tt[lo_i]),
            "end_time_s": float(tt[hi_i]),
            "training_run": bool(is_training_run),
            "forecast_run": bool(is_forecast_run),
            "fit_end_idx": int(fit_end_idx_run),
            "fit_end_time_s": float(tt[fit_end_idx_run]),
            "train_end_idx": int(train_end_idx_run),
            "train_end_time_s": float(tt[train_end_idx_run]),
            "validation_start_idx": int(val_start_idx_run) if val_start_idx_run is not None else None,
            "validation_start_time_s": float(tt[val_start_idx_run]) if val_start_idx_run is not None else None,
            "candidate_windows": int(candidate_starts.size),
            "train_windows": int(run_train_starts.size),
            "validation_windows": int(run_val_starts.size),
            "mode": mode,
        })

    train_starts = np.concatenate(train_start_parts).astype(int) if train_start_parts else np.asarray([], dtype=int)
    val_starts = np.concatenate(val_start_parts).astype(int) if val_start_parts else np.asarray([], dtype=int)
    train_starts = np.asarray(sorted(set(int(s) for s in train_starts)), dtype=int)
    val_starts = np.asarray(sorted(set(int(s) for s in val_starts)), dtype=int)
    mode_prefix = "" if len(segments) == 1 else "multi_run_"
    mode = mode_prefix + (
        "contiguous_chronological_train_validation_forecast"
        if val_starts.size > 0 else "train_all_pre_forecast_windows"
    )

    if train_starts.size == 0:
        raise ValueError("Chronological split left no training windows. Reduce seq_len, val_frac, or forecast_frac.")

    train_idx = point_indices_from_window_starts(train_starts, seq_len, n_points)
    val_idx = np.flatnonzero(val_mask).astype(int) if val_starts.size > 0 else np.asarray([], dtype=int)
    if train_idx.size == 0:
        train_idx = point_indices_from_window_starts(train_starts, seq_len, n_points)

    forecast_points = (
        int(forecast_end_idx - forecast_start_idx + 1)
        if forecast_enabled else 0
    )
    actual_train_frac = float(train_idx.size / max(n_points, 1))
    actual_val_frac = float(val_idx.size / max(n_points, 1))
    actual_forecast_frac = float(forecast_points / max(n_points, 1))
    fit_end_idx = int(max(int(info["fit_end_idx"]) for info in run_infos)) if run_infos else n_points - 1
    info = {
        "mode": mode,
        "validation_window_s": float(validation_window_s) if validation_window_s is not None else None,
        "val_frac": float(val_frac),
        "forecast_frac": float(cfg.get("forecast_frac", 0.0)),
        "actual_train_point_frac": actual_train_frac,
        "actual_val_point_frac": actual_val_frac,
        "actual_forecast_point_frac": actual_forecast_frac,
        "forecast_holdout_active": bool(forecast_enabled),
        "forecast_start_idx": int(forecast_start_idx),
        "forecast_end_idx": int(forecast_end_idx),
        "forecast_start_time_s": float(tt[forecast_start_idx]) if n_points else None,
        "forecast_end_time_s": float(tt[forecast_end_idx]) if n_points else None,
        "fit_end_idx": int(fit_end_idx),
        "fit_end_time_s": float(tt[fit_end_idx]) if n_points else None,
        "preforecast_points": int(preforecast_points),
        "train_points": int(train_idx.size),
        "validation_points": int(val_idx.size),
        "candidate_windows": int(candidate_windows),
        "train_windows": int(train_starts.size),
        "validation_windows": int(val_starts.size),
        "forecast_run_name": next((name for name in names if name.lower() == forecast_name_l), forecast_name),
        "training_run_names": [name for name in names if name.lower() in requested_training_l],
        "runs": run_infos,
    }
    lg.info(
        "  Forecast holdout:     %s | run=%s | %.6g-%.6g s",
        "active" if forecast_enabled else "inactive",
        info["forecast_run_name"],
        info["forecast_start_time_s"] if info["forecast_start_time_s"] is not None else float("nan"),
        info["forecast_end_time_s"] if info["forecast_end_time_s"] is not None else float("nan"),
    )
    lg.info(
        "  Chronological split:   %s | train=%d (%.1f%%) val=%d (%.1f%%) forecast=%d (%.1f%%) | train_windows=%d val_windows=%d",
        mode,
        int(train_idx.size), 100.0 * actual_train_frac,
        int(val_idx.size), 100.0 * actual_val_frac,
        int(forecast_points), 100.0 * actual_forecast_frac,
        int(train_starts.size), int(val_starts.size),
    )
    for run_info in run_infos:
        lg.info("    Run %-10s training=%s forecast=%s fit_end=%.6g s train_windows=%d val_windows=%d",
                run_info["name"], run_info["training_run"], run_info["forecast_run"],
                run_info["fit_end_time_s"], run_info["train_windows"], run_info["validation_windows"])
    return {
        "train_idx": train_idx.astype(int),
        "val_idx": val_idx.astype(int),
        "train_window_starts": train_starts.astype(int),
        "val_window_starts": val_starts.astype(int),
        "forecast_region": forecast_region,
        "validation_split": info,
    }


def prepare_data(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                 cfg: Dict[str, object], lg: logging.Logger) -> Dict[str, object]:
    lg.info("Preparing fitting-only KRISO dataset...")
    t, phi, wave_pack, data_window = apply_data_time_window(t, phi, wave_pack, cfg, lg)
    unit_checks_raw(t, phi, wave_pack)
    N = len(t)
    seq_len = int(cfg["seq_len"])
    if N < max(seq_len + 4, 50):
        raise ValueError("Sequence too short for configured seq_len.")

    vessel_segments, run_names = run_segments_from_pack(wave_pack, N)
    raw_wave_t_for_segments = np.asarray(wave_pack["wave_time_raw"], dtype=float)
    wave_segments = wave_segments_from_pack(wave_pack, len(raw_wave_t_for_segments), len(vessel_segments))
    split_info = split_preforecast_windows(t, cfg, lg, vessel_segments, run_names)
    train_idx = np.asarray(split_info["train_idx"], dtype=int)
    val_idx = np.asarray(split_info["val_idx"], dtype=int)
    train_window_starts = np.asarray(split_info["train_window_starts"], dtype=int)
    val_window_starts = np.asarray(split_info["val_window_starts"], dtype=int)
    train_window_starts, turning_sampling = augment_window_starts_for_turning_points(
        t, phi, train_window_starts, train_idx, seq_len, cfg
    )
    split_info["train_window_starts"] = train_window_starts.astype(int)
    split_info["turning_point_sampling"] = turning_sampling
    if isinstance(split_info.get("validation_split"), dict):
        split_info["validation_split"]["base_train_windows"] = int(
            turning_sampling.get("base_windows", train_window_starts.size)
        )
        split_info["validation_split"]["turning_point_extra_windows"] = int(
            turning_sampling.get("extra_windows", 0)
        )
        split_info["validation_split"]["train_windows"] = int(train_window_starts.size)
    if bool(turning_sampling.get("enabled", False)):
        lg.info(
            "  Turning-point sampling: selected=%d unique_windows=%d extra=%d total_train_windows=%d",
            int(turning_sampling.get("selected_turning_points", 0)),
            int(turning_sampling.get("extra_unique_windows", 0)),
            int(turning_sampling.get("extra_windows", 0)),
            int(train_window_starts.size),
        )

    time_feature_t = np.asarray(wave_pack.get("run_local_time", t), dtype=float)
    if time_feature_t.shape[:1] != t.shape[:1]:
        time_feature_t = t
    t_min = float(time_feature_t.min())
    t_scale = float(max(time_feature_t.max() - time_feature_t.min(), 1e-8))
    t_norm = (time_feature_t - t_min) / t_scale
    X_time = encode_time(t_norm, cfg.get("freq_multipliers", [1]))

    wave_t = np.asarray(wave_pack["wave_time_raw"], dtype=float)
    wave_raw = np.asarray(wave_pack["wave_signal_raw"], dtype=float)
    wave_cross_raw = np.asarray(wave_pack.get("wave_cross_beam_raw"), dtype=float) if wave_pack.get("wave_cross_beam_raw") is not None else np.zeros_like(wave_raw)
    if wave_cross_raw.shape[:1] != wave_raw.shape[:1]:
        wave_cross_raw = np.zeros_like(wave_raw)
    x_raw = np.asarray(wave_pack.get("x_position_raw"), dtype=float) if wave_pack.get("x_position_raw") is not None else None
    y_raw = np.asarray(wave_pack.get("y_position_raw"), dtype=float) if wave_pack.get("y_position_raw") is not None else None
    dominant_roll_period_s = estimate_dominant_period_by_segments(
        phi, t, train_idx, vessel_segments
    )
    encounter_shift_s, encounter_ratio, encounter_stats = position_encounter_shift_from_xy(
        t, x_raw, y_raw, train_idx, cfg, lg,
        dominant_wave_period_s=dominant_roll_period_s, run_segments=vessel_segments
    )
    orientation_raw, orientation_stats = vessel_wave_orientation_features_from_xy(
        t,
        x_raw,
        y_raw,
        cfg,
        lg,
        run_segments=vessel_segments,
        train_idx=train_idx,
    )
    fixed_wave_lag = cfg.get("fixed_wave_lag_s")
    if fixed_wave_lag is None:
        best_lag_s, lag_grid_s, lag_scores, dominant_roll_period_s = estimate_best_wave_lag(
            t, phi, wave_t, wave_raw, train_idx, cfg, lg,
            vessel_segments, wave_segments,
            encounter_time_shift_s=encounter_shift_s,
            dominant_period_s=dominant_roll_period_s,
        )
        wave_lag_source = str(
            cfg.get(
                "wave_alignment_source_file",
                "training-only band-limited roll/wave correlation scan",
            )
        )
    else:
        best_lag_s = float(fixed_wave_lag)
        if not math.isfinite(best_lag_s):
            raise ValueError("fixed_wave_lag_s must be finite or None.")
        lag_grid_s = np.asarray([best_lag_s], dtype=float)
        lag_scores = np.asarray([np.nan], dtype=float)
        wave_lag_source = str(cfg.get("fixed_wave_lag_source", "configuration"))
        lg.info(
            "  Fixed causal wave-response lag: %.9f s | source=%s | "
            "target-based lag scan disabled",
            best_lag_s,
            wave_lag_source,
        )
    cfg["dominant_roll_period_s"] = float(dominant_roll_period_s)
    lag_offsets_s = [float(v) for v in cfg.get("wave_lag_feature_offsets_s", [0.0])]
    wave_lags_s = [best_lag_s + off for off in lag_offsets_s]

    wave_slope_raw = gradient_time_by_segments(wave_raw, wave_t, wave_segments)
    wave_cross_slope_raw = gradient_time_by_segments(wave_cross_raw, wave_t, wave_segments)
    wave_bp_raw = bandpass_time_by_segments(
        wave_raw, wave_t, dominant_roll_period_s,
        float(cfg["bandpass_low_period_factor"]), float(cfg["bandpass_high_period_factor"]), wave_segments,
    )
    wave_slope_bp_raw = bandpass_time_by_segments(
        wave_slope_raw, wave_t, dominant_roll_period_s,
        float(cfg["bandpass_low_period_factor"]), float(cfg["bandpass_high_period_factor"]), wave_segments,
    )
    wave_cross_bp_raw = bandpass_time_by_segments(
        wave_cross_raw, wave_t, dominant_roll_period_s,
        float(cfg["bandpass_low_period_factor"]), float(cfg["bandpass_high_period_factor"]), wave_segments,
    )
    wave_cross_slope_bp_raw = bandpass_time_by_segments(
        wave_cross_slope_raw, wave_t, dominant_roll_period_s,
        float(cfg["bandpass_low_period_factor"]), float(cfg["bandpass_high_period_factor"]), wave_segments,
    )
    wave_abs_slope_bp_raw = np.abs(wave_slope_bp_raw)
    wave_curvature_bp_raw = gradient_time_by_segments(wave_slope_bp_raw, wave_t, wave_segments)
    wave_abs_curvature_bp_raw = np.abs(wave_curvature_bp_raw)
    wave_signal_slope_product_raw = wave_bp_raw * wave_slope_bp_raw
    wave_env_raw = rolling_rms_time_by_segments(wave_slope_bp_raw, wave_t, float(cfg["wave_envelope_seconds"]), wave_segments)
    wave_env_slow_raw = rolling_rms_time_by_segments(wave_slope_bp_raw, wave_t, float(cfg["wave_envelope_slow_seconds"]), wave_segments)

    aligned_feat: Dict[str, List[np.ndarray]] = {
        "wave_signal": [],
        "wave_slope": [],
        "wave_abs_slope": [],
        "wave_curvature": [],
        "wave_abs_curvature": [],
        "wave_signal_slope_product": [],
        "wave_envelope": [],
        "wave_envelope_slow": [],
        "wave_cross_beam_gradient": [],
    }
    raw_sources = {
        "wave_signal": wave_bp_raw,
        "wave_slope": wave_slope_bp_raw,
        "wave_abs_slope": wave_abs_slope_bp_raw,
        "wave_curvature": wave_curvature_bp_raw,
        "wave_abs_curvature": wave_abs_curvature_bp_raw,
        "wave_signal_slope_product": wave_signal_slope_product_raw,
        "wave_envelope": wave_env_raw,
        "wave_envelope_slow": wave_env_slow_raw,
        "wave_cross_beam_gradient": wave_cross_bp_raw + 0.25 * wave_cross_slope_bp_raw,
    }
    for lag_s in wave_lags_s:
        for feat_name, source in raw_sources.items():
            aligned_feat[feat_name].append(
                interp_to_vessel_time_by_segments(
                    t, wave_t, source, lag_s=lag_s, encounter_time_shift_s=encounter_shift_s,
                    vessel_segments=vessel_segments, source_segments=wave_segments,
                )
            )

    scaled_feat: Dict[str, List[np.ndarray]] = {k: [] for k in aligned_feat}
    wave_stats: List[Dict[str, float]] = []
    for i, lag_s in enumerate(wave_lags_s):
        row: Dict[str, float] = {"lag_s": float(lag_s)}
        for feat_name in aligned_feat:
            arr = aligned_feat[feat_name][i]
            scaled, mu, sd = scale_train_region(arr, train_idx)
            scaled_feat[feat_name].append(scaled)
            row[f"{feat_name}_mean"] = mu
            row[f"{feat_name}_std"] = sd
        wave_stats.append(row)

    phi_std = max(float(np.std(phi[train_idx])), 1e-8)
    phi_scaled = phi / phi_std
    v_est = robust_roll_rate_by_segments(phi_scaled, t, vessel_segments)
    rate_smooth_s = float(cfg.get("roll_rate_smoothing_seconds", 0.0))
    if rate_smooth_s > 0.0:
        v_est = moving_average_time_by_segments(v_est, t, rate_smooth_s, vessel_segments)
    if bool(cfg.get("scale_roll_rate_target", False)):
        v_std = max(float(np.std(v_est[train_idx])), 1e-8)
        v_scaled = v_est / v_std
    else:
        v_std = 1.0
        v_scaled = v_est

    speed_raw = np.asarray(wave_pack.get("speed_raw"), dtype=float) if wave_pack.get("speed_raw") is not None else None
    yawrate_raw = np.asarray(wave_pack.get("yawrate_raw"), dtype=float) if wave_pack.get("yawrate_raw") is not None else None
    state_stats: Dict[str, Dict[str, float]] = {}

    speed_scaled = yawrate_scaled = x_scaled = y_scaled = encounter_ratio_scaled = encounter_shift_scaled = None
    orientation_scaled: Dict[str, np.ndarray] = {}
    if speed_raw is not None:
        speed_scaled, mu, sd = scale_train_region(speed_raw, train_idx)
        state_stats["speed"] = {"mean": mu, "std": sd}
    if yawrate_raw is not None:
        yawrate_scaled, mu, sd = scale_train_region(yawrate_raw, train_idx)
        state_stats["yawrate"] = {"mean": mu, "std": sd}
    if x_raw is not None and bool(cfg.get("use_position_features", True)):
        x_scaled, mu, sd = scale_train_region(x_raw, train_idx)
        state_stats["x_position"] = {"mean": mu, "std": sd}
    if y_raw is not None and bool(cfg.get("use_position_features", True)):
        y_scaled, mu, sd = scale_train_region(y_raw, train_idx)
        state_stats["y_position"] = {"mean": mu, "std": sd}
    if encounter_stats.get("active", False):
        encounter_ratio_scaled, mu, sd = scale_train_region(encounter_ratio, train_idx)
        state_stats["encounter_frequency_ratio"] = {"mean": mu, "std": sd}
        encounter_shift_scaled, mu, sd = scale_train_region(encounter_shift_s, train_idx)
        state_stats["encounter_time_shift"] = {"mean": mu, "std": sd}
    for name, arr in orientation_raw.items():
        if name == "wave_orientation_effect_gain":
            arr = np.asarray(arr, dtype=float)
            orientation_scaled[name] = arr
            train_values = arr[train_idx] if len(train_idx) else arr
            state_stats[name] = {
                "mean": float(np.mean(train_values)),
                "std": float(np.std(train_values)),
                "scaled": False,
                "note": "bounded physical wave-force multiplier retained in raw 0-1 coordinates",
            }
            continue
        scaled, mu, sd = scale_train_region(arr, train_idx)
        orientation_scaled[name] = scaled
        state_stats[name] = {"mean": mu, "std": sd}

    feature_parts: List[np.ndarray] = [X_time]
    wave_feature_indices: Dict[str, List[int]] = {k: [] for k in scaled_feat}
    state_feature_indices: Dict[str, List[int]] = {
        "speed": [], "yawrate": [], "x_position": [], "y_position": [],
        "encounter_frequency_ratio": [], "encounter_time_shift": [],
        "track_speed_xy": [], "wave_perpendicular_velocity": [], "wave_parallel_velocity": [],
        "heading_perpendicular_to_waves": [], "heading_parallel_to_waves": [], "heading_obliquity_abs": [],
        "wave_orientation_effect_gain": []
    }
    next_idx = int(X_time.shape[1])
    active_wave = active_wave_feature_names(cfg)
    for feat_name in active_wave:
        for ch in scaled_feat[feat_name]:
            feature_parts.append(ch.reshape(-1, 1))
            wave_feature_indices[feat_name].append(next_idx)
            next_idx += 1

    def append_state(name: str, arr: Optional[np.ndarray]) -> None:
        nonlocal next_idx
        if arr is not None:
            feature_parts.append(arr.reshape(-1, 1))
            state_feature_indices[name].append(next_idx)
            next_idx += 1

    append_state("speed", speed_scaled)
    append_state("yawrate", yawrate_scaled)
    append_state("x_position", x_scaled)
    append_state("y_position", y_scaled)
    append_state("encounter_frequency_ratio", encounter_ratio_scaled)
    append_state("encounter_time_shift", encounter_shift_scaled)
    for name in [
        "track_speed_xy",
        "wave_perpendicular_velocity",
        "wave_parallel_velocity",
        "heading_perpendicular_to_waves",
        "heading_parallel_to_waves",
        "heading_obliquity_abs",
        "wave_orientation_effect_gain",
    ]:
        append_state(name, orientation_scaled.get(name))
    X_exog = np.concatenate(feature_parts, axis=1)

    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    requested_rollout_steps = int(round(float(cfg.get("rollout_window_s", 5.08)) / max(dt_med, 1.0e-8)))
    resolved_rollout_steps = int(min(max(0, seq_len - 1), max(0, requested_rollout_steps)))
    cfg["resolved_rollout_steps"] = resolved_rollout_steps
    cfg["resolved_rollout_duration_s"] = float(resolved_rollout_steps * dt_med)
    cfg["full_rollout_horizon_active"] = bool(resolved_rollout_steps >= requested_rollout_steps)
    if not bool(cfg["full_rollout_horizon_active"]):
        lg.warning(
            "Configured sequence holds only %.3f s of the requested %.3f s rollout; increase seq_len for full-horizon training.",
            float(cfg["resolved_rollout_duration_s"]),
            float(cfg.get("rollout_window_s", 5.08)),
        )
    delay_steps_list = motion_feedback_delay_steps(cfg, dt_med)
    motion_feature_indices = {"phi_fb": [], "v_fb": []}
    if bool(cfg.get("use_motion_feedback", False)):
        motion_parts: List[np.ndarray] = []
        phi_cols: List[int] = []
        v_cols: List[int] = []
        next_motion_col = int(X_exog.shape[1])
        for delay_steps in delay_steps_list:
            phi_fb = np.empty_like(phi_scaled)
            v_fb = np.empty_like(v_scaled)
            for lo, hi in vessel_segments:
                sl = slice(int(lo), int(hi) + 1)
                if delay_steps <= 0:
                    phi_fb[sl] = phi_scaled[sl]
                    v_fb[sl] = v_scaled[sl]
                else:
                    run_phi = phi_scaled[sl]
                    run_v = v_scaled[sl]
                    n_run = len(run_phi)
                    fill = min(int(delay_steps), n_run)
                    phi_fb[int(lo):int(lo) + fill] = run_phi[0]
                    v_fb[int(lo):int(lo) + fill] = run_v[0]
                    if n_run > fill:
                        phi_fb[int(lo) + fill:int(hi) + 1] = run_phi[:n_run - fill]
                        v_fb[int(lo) + fill:int(hi) + 1] = run_v[:n_run - fill]
            motion_parts.extend([phi_fb.reshape(-1, 1), v_fb.reshape(-1, 1)])
            phi_cols.append(next_motion_col)
            v_cols.append(next_motion_col + 1)
            next_motion_col += 2
        motion_feature_indices = {"phi_fb": phi_cols, "v_fb": v_cols}
        X = np.concatenate([X_exog, *motion_parts], axis=1)
    else:
        X = X_exog

    device = resolve_torch_device(cfg, lg)
    unit_checks_prepared(X, phi_scaled, v_scaled, wave_feature_indices, cfg)

    lg.info("  Total points:          %d", N)
    lg.info("  Train points:          %d", len(train_idx))
    lg.info("  Validation points:     %d", len(val_idx))
    lg.info("  phi_std:               %.8f rad", phi_std)
    lg.info("  v_std:                 %.8f scaled-rad/s", v_std)
    lg.info("  Active wave features:  %s", active_wave)
    lg.info("  Wave lags used (s):    %s", [round(float(v), 4) for v in wave_lags_s])
    lg.info("  Active state features: %s", [k for k, v in state_feature_indices.items() if v])
    if bool(cfg.get("use_motion_feedback", False)):
        lg.info("  Roll feedback delays:  %s steps (%s s)",
                delay_steps_list, [round(float(v * dt_med), 6) for v in delay_steps_list])
    else:
        lg.info("  Roll feedback:         disabled (physics/wave-driven mode)")
    lg.info("  Input feature count:   %d", X.shape[1])
    lg.info("  Forecast training window: %d steps (%.3f s) | full horizon=%s",
            resolved_rollout_steps, float(cfg["resolved_rollout_duration_s"]), bool(cfg["full_rollout_horizon_active"]))
    lg.info("  Device:                %s", device)

    return {
        "t": t,
        "X": X.astype(np.float32),
        "X_exog": X_exog.astype(np.float32),
        "phi": phi,
        "phi_scaled": phi_scaled.astype(np.float32),
        "phi_std": float(phi_std),
        "v_est_scaled": v_scaled.astype(np.float32),
        "v_std": float(v_std),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_window_starts": train_window_starts,
        "val_window_starts": val_window_starts,
        "forecast_region": split_info.get("forecast_region"),
        "validation_split": split_info.get("validation_split"),
        "turning_point_sampling": split_info.get("turning_point_sampling", {}),
        "run_names": np.asarray(run_names, dtype=object),
        "run_segments": np.asarray(vessel_segments, dtype=int),
        "wave_run_segments": np.asarray(wave_segments, dtype=int),
        "run_local_time": np.asarray(wave_pack.get("run_local_time", t), dtype=float),
        "input_dim": int(X.shape[1]),
        "device": device,
        "wave_time_raw": wave_t,
        "wave_signal_raw": wave_raw,
        "wave_1_raw": np.asarray(wave_pack.get("wave_1_raw", wave_raw), dtype=float),
        "wave_2_raw": np.asarray(wave_pack.get("wave_2_raw", wave_raw), dtype=float),
        "wave_cross_beam_raw": wave_cross_raw,
        "wave_signal_aligned": aligned_feat["wave_signal"][0],
        "wave_slope_aligned": aligned_feat["wave_slope"][0],
        "wave_lag_source": wave_lag_source,
        "best_wave_lag_s": float(best_lag_s),
        "wave_lags_s": [float(v) for v in wave_lags_s],
        "lag_grid_s": lag_grid_s,
        "lag_scores": lag_scores,
        "dominant_roll_period_s": float(dominant_roll_period_s),
        "wave_feature_indices": wave_feature_indices,
        "state_feature_indices": state_feature_indices,
        "motion_feature_indices": motion_feature_indices,
        "state_stats": state_stats,
        "wave_stats": wave_stats,
        "encounter_time_shift_s": encounter_shift_s,
        "encounter_frequency_ratio": encounter_ratio,
        "encounter_stats": encounter_stats,
        "orientation_stats": orientation_stats,
        "motion_delay_steps": int(delay_steps_list[-1]) if delay_steps_list else 0,
        "motion_delay_steps_list": [int(v) for v in delay_steps_list],
        "data_time_window": data_window,
        "t_min": t_min,
        "t_scale": t_scale,
    }


def unit_checks_prepared(X: np.ndarray, phi_scaled: np.ndarray, v_scaled: np.ndarray,
                         wave_feature_indices: Dict[str, List[int]], cfg: Dict[str, object]) -> None:
    if X.ndim != 2:
        raise ValueError("Prepared feature matrix X must be 2D.")
    if len(X) != len(phi_scaled) or len(X) != len(v_scaled):
        raise ValueError("Prepared X, phi_scaled, and v_scaled lengths must match.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Prepared feature matrix contains non-finite values.")
    if not np.all(np.isfinite(phi_scaled)) or not np.all(np.isfinite(v_scaled)):
        raise ValueError("Prepared target arrays contain non-finite values.")
    active = active_wave_feature_names(cfg)
    if not any(len(wave_feature_indices.get(name, [])) > 0 for name in active):
        raise ValueError("No active wave features were added. Check Wave_Time and wave signal columns.")


# =============================================================================
# DATASET
# =============================================================================

class FitWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, t: np.ndarray, phi_s: np.ndarray, v_s: np.ndarray,
                 seq_len: int, stride: int, device: torch.device,
                 start_idx: int, end_idx: int, allow_context_before_start: bool = False,
                 tensor_cache: Optional[Dict[str, torch.Tensor]] = None,
                 precompute_windows: bool = False,
                 window_cache: Optional[Dict[str, torch.Tensor]] = None,
                 explicit_starts: Optional[Iterable[int]] = None):
        if tensor_cache is None:
            tensor_cache = {
                "X": torch.as_tensor(X, dtype=torch.float32, device=device),
                "t": torch.as_tensor(t, dtype=torch.float32, device=device).view(-1, 1),
                "phi": torch.as_tensor(phi_s, dtype=torch.float32, device=device).view(-1, 1),
                "v": torch.as_tensor(v_s, dtype=torch.float32, device=device).view(-1, 1),
            }
        self.X = tensor_cache["X"]
        self.t = tensor_cache["t"]
        self.phi = tensor_cache["phi"]
        self.v = tensor_cache["v"]
        self.seq_len = int(seq_len)
        self.starts: List[int] = []
        self.window_cache: Optional[Dict[str, torch.Tensor]] = None
        N = len(phi_s)
        seq_len = self.seq_len
        stride = max(1, int(stride))
        start_idx = int(max(0, start_idx))
        end_idx = int(min(N - 1, end_idx))
        if end_idx - start_idx + 1 < 2:
            return

        if explicit_starts is not None:
            starts = [int(s) for s in explicit_starts]
        else:
            if allow_context_before_start:
                s0 = max(0, start_idx - seq_len + 1)
            else:
                s0 = start_idx
            last_start = end_idx - seq_len + 1
            if last_start < s0:
                last_start = max(0, end_idx - seq_len + 1)
            starts = list(range(s0, max(s0, last_start) + 1, stride))
            final_start = max(0, end_idx - seq_len + 1)
            if final_start not in starts and final_start >= 0:
                starts.append(final_start)
            starts = sorted(set(starts))

        for s in starts:
            e = s + seq_len
            if s < 0 or e > N:
                continue
            if explicit_starts is None and not allow_context_before_start and s < start_idx:
                continue
            if explicit_starts is None and e - 1 > end_idx:
                continue
            self.starts.append(int(s))

        if window_cache is not None:
            self.window_cache = window_cache
        elif bool(precompute_windows) and len(self.starts) > 0:
            starts_tensor = torch.as_tensor(self.starts, dtype=torch.long, device=self.X.device)
            offsets = torch.arange(self.seq_len, dtype=torch.long, device=self.X.device).view(1, -1)
            idx = starts_tensor.view(-1, 1) + offsets
            x_windows = self.X[idx]
            self.window_cache = {
                "src": x_windows,
                "tgt": x_windows,
                "t": self.t[idx],
                "y_phi": self.phi[idx],
                "y_v": self.v[idx],
                "start": starts_tensor.view(-1, 1),
            }

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.window_cache is not None:
            return {k: v[idx] for k, v in self.window_cache.items()}
        s = self.starts[idx]
        e = s + self.seq_len
        return {
            "src": self.X[s:e],
            "tgt": self.X[s:e],
            "t": self.t[s:e],
            "y_phi": self.phi[s:e],
            "y_v": self.v[s:e],
            "start": torch.tensor([s], dtype=torch.long, device=self.X.device),
        }


def resolve_dataloader_workers(cfg: Dict[str, object], device: torch.device, dataset_on_device: bool) -> int:
    raw = cfg.get("dataloader_num_workers", "auto")
    available, _ = configured_cpu_count(cfg)
    if isinstance(raw, str) and raw.strip().lower() in {"", "auto", "default"}:
        if device.type == "cuda" and dataset_on_device:
            return 0
        return max(0, min(4, available - 1))
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = 0
    if device.type == "cuda" and dataset_on_device and workers > 0:
        return 0
    return max(0, min(workers, max(0, available - 1)))


def dataloader_kwargs(cfg: Dict[str, object], device: torch.device,
                      dataset_on_device: bool, shuffle: bool) -> Dict[str, object]:
    workers = resolve_dataloader_workers(cfg, device, dataset_on_device)
    kwargs: Dict[str, object] = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("dataloader_persistent_workers", True))
        kwargs["prefetch_factor"] = max(1, int(cfg.get("dataloader_prefetch_factor", 4)))
    return kwargs


def prune_window_cache_entries(shared_cache: Dict[object, object], max_entries: int) -> None:
    if max_entries < 1:
        max_entries = 1
    window_keys = [key for key in shared_cache if isinstance(key, tuple) and key[:1] == ("windows",)]
    while len(window_keys) >= max_entries:
        oldest = window_keys.pop(0)
        shared_cache.pop(oldest, None)


def shared_tensor_cache(data: Dict[str, object], device: torch.device,
                        shared_cache: Optional[Dict[object, object]]) -> Dict[str, torch.Tensor]:
    key = (
        "tensors",
        id(data["X"]),
        id(data["t"]),
        id(data["phi_scaled"]),
        id(data["v_est_scaled"]),
        str(device),
    )
    if shared_cache is not None and key in shared_cache:
        cached = shared_cache[key]
        if isinstance(cached, dict):
            return cached
    tensor_cache = {
        "X": torch.as_tensor(data["X"], dtype=torch.float32, device=device),
        "t": torch.as_tensor(data["t"], dtype=torch.float32, device=device).view(-1, 1),
        "phi": torch.as_tensor(data["phi_scaled"], dtype=torch.float32, device=device).view(-1, 1),
        "v": torch.as_tensor(data["v_est_scaled"], dtype=torch.float32, device=device).view(-1, 1),
    }
    if shared_cache is not None:
        shared_cache[key] = tensor_cache
    return tensor_cache


def shared_window_cache_key(split_name: str, data: Dict[str, object], cfg: Dict[str, object],
                            start_idx: int, end_idx: int, allow_context_before_start: bool,
                            tensor_cache: Dict[str, torch.Tensor],
                            explicit_starts: Optional[np.ndarray] = None) -> Tuple[object, ...]:
    if explicit_starts is None:
        starts_sig: Tuple[object, ...] = ("range",)
    else:
        starts = np.asarray(explicit_starts, dtype=np.int64)
        starts_sig = (
            "explicit",
            int(starts.size),
            int(starts[0]) if starts.size else -1,
            int(starts[-1]) if starts.size else -1,
            int(np.sum(starts, dtype=np.int64)) if starts.size else 0,
            int(np.sum(starts * starts, dtype=np.int64)) if starts.size else 0,
        )
    return (
        "windows",
        split_name,
        id(tensor_cache["X"]),
        int(cfg["seq_len"]),
        int(max(1, cfg["stride"])),
        int(start_idx),
        int(end_idx),
        bool(allow_context_before_start),
        starts_sig,
    )


def get_cached_windows(shared_cache: Optional[Dict[object, object]], key: Tuple[object, ...],
                       cfg: Dict[str, object]) -> Optional[Dict[str, torch.Tensor]]:
    if shared_cache is None or not bool(cfg.get("precompute_windows", False)):
        return None
    cached = shared_cache.get(key)
    if isinstance(cached, dict):
        shared_cache[key] = shared_cache.pop(key)
        return cached
    prune_window_cache_entries(shared_cache, int(cfg.get("bayes_window_cache_max_entries", 2)))
    return None


def remember_cached_windows(shared_cache: Optional[Dict[object, object]], key: Tuple[object, ...],
                            ds: FitWindowDataset, cfg: Dict[str, object]) -> None:
    if shared_cache is None or not bool(cfg.get("precompute_windows", False)) or ds.window_cache is None:
        return
    shared_cache[key] = ds.window_cache


def build_loaders(data: Dict[str, object], cfg: Dict[str, object], lg: logging.Logger,
                  shared_cache: Optional[Dict[object, object]] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_idx = np.asarray(data["train_idx"], dtype=int)
    val_idx = np.asarray(data["val_idx"], dtype=int)
    train_starts = np.asarray(data.get("train_window_starts", []), dtype=int)
    val_starts = np.asarray(data.get("val_window_starts", []), dtype=int)
    device = data["device"]
    precompute_windows = bool(cfg.get("precompute_windows", False)) and (
        device.type == "cpu" or not bool(cfg.get("precompute_windows_cpu_only", True))
    )
    tensor_cache = shared_tensor_cache(data, device, shared_cache)
    dataset_on_device = True
    if train_starts.size == 0 and train_idx.size > 0:
        train_starts = window_starts_ending_at_or_before(
            len(np.asarray(data["t"])), int(cfg["seq_len"]), int(cfg["stride"]), int(train_idx[-1])
        )
    if train_starts.size == 0:
        raise ValueError("No training windows. Reduce seq_len, val_frac, forecast_frac, or the forecast-window override.")
    train_window_key = shared_window_cache_key(
        "train", data, cfg, int(train_starts[0]), int(train_starts[-1] + int(cfg["seq_len"]) - 1),
        False, tensor_cache, explicit_starts=train_starts,
    )
    train_window_cache = get_cached_windows(shared_cache, train_window_key, cfg)
    train_ds = FitWindowDataset(
        data["X"], data["t"], data["phi_scaled"], data["v_est_scaled"],
        int(cfg["seq_len"]), int(cfg["stride"]), device,
        int(train_starts[0]), int(train_starts[-1] + int(cfg["seq_len"]) - 1), allow_context_before_start=False,
        tensor_cache=tensor_cache, precompute_windows=precompute_windows,
        window_cache=train_window_cache, explicit_starts=train_starts,
    )
    if len(train_ds) == 0:
        raise ValueError("No training windows. Reduce seq_len or provide more data.")
    remember_cached_windows(shared_cache, train_window_key, train_ds, cfg)
    train_loader = DataLoader(train_ds, **dataloader_kwargs(cfg, device, dataset_on_device, shuffle=True))

    val_loader = None
    if val_starts.size > 0:
        val_window_key = shared_window_cache_key(
            "val", data, cfg, int(val_starts[0]), int(val_starts[-1] + int(cfg["seq_len"]) - 1),
            False, tensor_cache, explicit_starts=val_starts,
        )
        val_window_cache = get_cached_windows(shared_cache, val_window_key, cfg)
        val_ds = FitWindowDataset(
            data["X"], data["t"], data["phi_scaled"], data["v_est_scaled"],
            int(cfg["seq_len"]), int(max(1, cfg["stride"])), device,
            int(val_starts[0]), int(val_starts[-1] + int(cfg["seq_len"]) - 1), allow_context_before_start=False,
            tensor_cache=tensor_cache, precompute_windows=precompute_windows,
            window_cache=val_window_cache, explicit_starts=val_starts,
        )
        remember_cached_windows(shared_cache, val_window_key, val_ds, cfg)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, **dataloader_kwargs(cfg, device, dataset_on_device, shuffle=False))
    loader_workers = resolve_dataloader_workers(cfg, device, dataset_on_device)
    if device.type == "cuda" and str(cfg.get("dataloader_num_workers", "auto")).strip().lower() not in {"auto", "default", "", "0"}:
        lg.warning("DataLoader workers forced to 0 because window tensors are resident on CUDA.")
    cfg["resolved_dataloader_num_workers"] = int(loader_workers)
    cfg["resolved_dataloader_prefetch_factor"] = int(cfg.get("dataloader_prefetch_factor", 4)) if loader_workers > 0 else 0
    lg.info("Sequence windows: train=%d val=%s", len(train_ds), len(val_loader.dataset) if val_loader else 0)
    if precompute_windows:
        def cache_bytes(ds: FitWindowDataset) -> int:
            if ds.window_cache is None:
                return 0
            seen: set[int] = set()
            total = 0
            for v in ds.window_cache.values():
                ptr = int(v.data_ptr())
                if ptr in seen:
                    continue
                seen.add(ptr)
                total += int(v.numel() * v.element_size())
            return total

        seen_cache_ptrs: set[int] = set()
        cached = cache_bytes(train_ds)
        if train_ds.window_cache is not None:
            seen_cache_ptrs.update(int(v.data_ptr()) for v in train_ds.window_cache.values())
        if val_loader is not None and getattr(val_loader.dataset, "window_cache", None) is not None:
            for v in val_loader.dataset.window_cache.values():
                ptr = int(v.data_ptr())
                if ptr not in seen_cache_ptrs:
                    seen_cache_ptrs.add(ptr)
                    cached += int(v.numel() * v.element_size())
        cfg["precomputed_window_cache_mb"] = float(cached / (1024 ** 2))
        lg.info("Precomputed sequence windows in RAM: %.2f MB", cfg["precomputed_window_cache_mb"])
        if shared_cache is not None:
            window_entries = sum(1 for key in shared_cache if isinstance(key, tuple) and key[:1] == ("windows",))
            lg.info("Shared Bayes window-cache entries in RAM: %d/%d", window_entries,
                    int(cfg.get("bayes_window_cache_max_entries", 2)))
    lg.info("DataLoader: workers=%d prefetch=%s persistent=%s dataset_device=%s",
            loader_workers,
            cfg["resolved_dataloader_prefetch_factor"] if loader_workers > 0 else "n/a",
            bool(cfg.get("dataloader_persistent_workers", True)) if loader_workers > 0 else False,
            device)
    return train_loader, val_loader


# =============================================================================
# MODEL
# =============================================================================

def inv_softplus(y: float) -> float:
    y = float(max(y, 1e-8))
    return math.log(math.expm1(y))


class PINNLSTM(nn.Module):
    """Causal stacked-LSTM backbone with the unchanged Rev M output interface.

    The measured numeric channels are passed through a linear width adapter and a
    standard unidirectional ``nn.LSTM``. No tokenisation, learned embeddings,
    positional encoding, attention, or bidirectional recurrence is used.

    The public forward interface is intentionally identical to the original model:
    it returns scaled roll angle, scaled roll rate, total forcing without turning,
    and residual forcing for every sample in the input window. This lets the
    existing data pipeline, non-PINN losses, forecasts, checkpoints,
    diagnostics, and output writers remain unchanged.
    """

    def __init__(self, cfg: Dict[str, object], input_dim: int,
                 wave_feature_indices: Dict[str, List[int]],
                 state_feature_indices: Dict[str, List[int]],
                 motion_feature_indices: Dict[str, List[int]]):
        super().__init__()
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.wave_feature_indices = wave_feature_indices or {}
        self.state_feature_indices = state_feature_indices or {}
        self.motion_feature_indices = motion_feature_indices or {}

        backbone_scale = torch.ones(self.input_dim, dtype=torch.float32)
        feedback_backbone_gain = float(cfg.get("motion_feedback_backbone_gain", 1.0))
        for key in ("phi_fb", "v_fb"):
            for idx in self.motion_feature_indices.get(key, []):
                backbone_scale[int(idx)] = feedback_backbone_gain
        self.register_buffer(
            "backbone_feature_scale",
            backbone_scale,
            persistent=True,
        )

        hidden_size = int(cfg["lstm_hidden_size"])
        n_layers = int(cfg["lstm_layers"])
        dropout = float(cfg["lstm_dropout"])
        head_hidden = int(cfg["fc_hidden"])
        self.hidden_size = hidden_size

        # Keep the original numeric feature-width adapter, but replace the entire
        # attention/positional-encoding stack with a standard unidirectional LSTM.
        self.feature_adapter = nn.Linear(self.input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.phi_head = nn.Sequential(nn.Linear(hidden_size, head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))
        self.v_head = nn.Sequential(nn.Linear(hidden_size, head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))
        self.force_residual_head = nn.Sequential(nn.Linear(hidden_size, head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))
        gain_max = float(max(1.0e-6, cfg.get("roll_amplitude_calibration_gain_max", 0.40)))
        gain_init = float(np.clip(float(cfg.get("roll_amplitude_calibration_gain_init", 0.08)), 0.0, gain_max))
        gain_frac = float(np.clip(gain_init / gain_max, 1.0e-6, 1.0 - 1.0e-6))
        self.roll_amplitude_gain_logit = nn.Parameter(
            torch.tensor(math.log(gain_frac / (1.0 - gain_frac)), dtype=torch.float32)
        )

        self.wave_envelope_gate_sources = (
            ("wave_abs_slope", self.wave_feature_indices),
            ("wave_envelope", self.wave_feature_indices),
            ("wave_envelope_slow", self.wave_feature_indices),
            ("wave_cross_beam_gradient", self.wave_feature_indices),
            ("x_position", self.state_feature_indices),
            ("y_position", self.state_feature_indices),
            ("encounter_frequency_ratio", self.state_feature_indices),
            ("encounter_time_shift", self.state_feature_indices),
            ("track_speed_xy", self.state_feature_indices),
            ("wave_perpendicular_velocity", self.state_feature_indices),
            ("wave_parallel_velocity", self.state_feature_indices),
            ("heading_perpendicular_to_waves", self.state_feature_indices),
            ("heading_parallel_to_waves", self.state_feature_indices),
            ("heading_obliquity_abs", self.state_feature_indices),
            ("wave_orientation_effect_gain", self.state_feature_indices),
        )
        envelope_gate_in_dim = sum(
            len(src.get(key, [])) for key, src in self.wave_envelope_gate_sources
        )
        self.wave_envelope_gate_input_dim = int(envelope_gate_in_dim)
        self.wave_envelope_gate = nn.Sequential(
            nn.Linear(max(1, envelope_gate_in_dim), head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )
        self.wave_shape_correction_sources = (
            ("wave_signal", self.wave_feature_indices),
            ("wave_slope", self.wave_feature_indices),
            ("wave_abs_slope", self.wave_feature_indices),
            ("wave_envelope", self.wave_feature_indices),
            ("wave_envelope_slow", self.wave_feature_indices),
            ("wave_cross_beam_gradient", self.wave_feature_indices),
        )
        wave_shape_input_dim = sum(
            len(src.get(key, [])) for key, src in self.wave_shape_correction_sources
        )
        self.wave_shape_correction_input_dim = int(wave_shape_input_dim)
        shape_hidden = max(1, int(cfg.get("wave_shape_correction_hidden", 8)))
        self.wave_shape_correction = nn.Sequential(
            nn.Linear(max(1, wave_shape_input_dim), shape_hidden),
            nn.Tanh(),
            nn.Linear(shape_hidden, 1),
        )

        gate_in_dim = hidden_size
        for key, src in (
            ("phi_fb", self.motion_feature_indices), ("v_fb", self.motion_feature_indices),
            ("speed", self.state_feature_indices), ("yawrate", self.state_feature_indices),
            ("x_position", self.state_feature_indices), ("y_position", self.state_feature_indices),
            ("encounter_frequency_ratio", self.state_feature_indices), ("encounter_time_shift", self.state_feature_indices),
            ("track_speed_xy", self.state_feature_indices), ("wave_perpendicular_velocity", self.state_feature_indices),
            ("wave_parallel_velocity", self.state_feature_indices), ("heading_perpendicular_to_waves", self.state_feature_indices),
            ("heading_parallel_to_waves", self.state_feature_indices), ("heading_obliquity_abs", self.state_feature_indices),
            ("wave_orientation_effect_gain", self.state_feature_indices),
            ("wave_envelope", self.wave_feature_indices), ("wave_envelope_slow", self.wave_feature_indices),
            ("wave_cross_beam_gradient", self.wave_feature_indices),
        ):
            gate_in_dim += len(src.get(key, []))
        self.wave_gate = nn.Sequential(nn.Linear(gate_in_dim, head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))

        # Measured-wave and vessel-state coupling coefficients are unchanged.
        n_wave = max(len(self.wave_feature_indices.get("wave_signal", [])), 1)
        n_slope = max(len(self.wave_feature_indices.get("wave_slope", [])), 1)
        n_abs = max(len(self.wave_feature_indices.get("wave_abs_slope", [])), 1)
        n_env = max(len(self.wave_feature_indices.get("wave_envelope", [])), 1)
        n_env_slow = max(len(self.wave_feature_indices.get("wave_envelope_slow", [])), 1)
        n_cross_beam = max(len(self.wave_feature_indices.get("wave_cross_beam_gradient", [])), 1)
        n_speed = max(len(self.state_feature_indices.get("speed", [])), 1)
        n_yaw = max(len(self.state_feature_indices.get("yawrate", [])), 1)
        n_x = max(len(self.state_feature_indices.get("x_position", [])), 1)
        n_y = max(len(self.state_feature_indices.get("y_position", [])), 1)
        n_track = max(len(self.state_feature_indices.get("track_speed_xy", [])), 1)
        n_perp_vel = max(len(self.state_feature_indices.get("wave_perpendicular_velocity", [])), 1)
        n_parallel_vel = max(len(self.state_feature_indices.get("wave_parallel_velocity", [])), 1)
        n_heading_perp = max(len(self.state_feature_indices.get("heading_perpendicular_to_waves", [])), 1)
        n_heading_parallel = max(len(self.state_feature_indices.get("heading_parallel_to_waves", [])), 1)
        n_obliquity = max(len(self.state_feature_indices.get("heading_obliquity_abs", [])), 1)
        self.a_wave = nn.Parameter(torch.full(
            (n_wave,),
            float(cfg.get("wave_quadrature_signal_init", -0.08)),
            dtype=torch.float32,
        ))
        self.a_wave_slope = nn.Parameter(torch.full(
            (n_slope,),
            float(cfg.get("wave_quadrature_slope_init", 0.18)),
            dtype=torch.float32,
        ))
        self.a_wave_abs_slope = nn.Parameter(torch.full((n_abs,), 0.12, dtype=torch.float32))
        self.a_wave_env = nn.Parameter(torch.full((n_env,), 0.10, dtype=torch.float32))
        self.a_wave_env_slow = nn.Parameter(torch.full((n_env_slow,), 0.08, dtype=torch.float32))
        self.a_wave_cross_beam_gradient = nn.Parameter(torch.full((n_cross_beam,), 0.10, dtype=torch.float32))
        self.a_speed = nn.Parameter(torch.full((n_speed,), 0.05, dtype=torch.float32))
        self.a_yawrate = nn.Parameter(torch.full((n_yaw,), 0.12, dtype=torch.float32))
        self.a_x_position = nn.Parameter(torch.full((n_x,), 0.05, dtype=torch.float32))
        self.a_y_position = nn.Parameter(torch.full((n_y,), 0.05, dtype=torch.float32))
        self.a_track_speed_xy = nn.Parameter(torch.full((n_track,), 0.05, dtype=torch.float32))
        self.a_wave_perpendicular_velocity = nn.Parameter(torch.full((n_perp_vel,), 0.10, dtype=torch.float32))
        self.a_wave_parallel_velocity = nn.Parameter(torch.full((n_parallel_vel,), 0.05, dtype=torch.float32))
        self.a_heading_perpendicular_to_waves = nn.Parameter(torch.full((n_heading_perp,), 0.08, dtype=torch.float32))
        self.a_heading_parallel_to_waves = nn.Parameter(torch.full((n_heading_parallel,), 0.04, dtype=torch.float32))
        self.a_heading_obliquity_abs = nn.Parameter(torch.full((n_obliquity,), 0.08, dtype=torch.float32))

        # Optional moving-vessel correction; inactive for this stationary dataset.
        self.a_turn_yaw = nn.Parameter(torch.zeros((n_yaw,), dtype=torch.float32))
        self.a_turn_yaw_abs = nn.Parameter(torch.zeros((n_yaw,), dtype=torch.float32))
        self.a_turn_speed_yaw = nn.Parameter(torch.zeros((n_yaw,), dtype=torch.float32))

        # Positive physics parameters in scaled coordinates are unchanged.
        self.c_roll_param = nn.Parameter(torch.tensor(inv_softplus(float(cfg["c_roll_init"]) - float(cfg["c_roll_min"]))))
        self.c_quad_param = nn.Parameter(torch.tensor(inv_softplus(float(cfg["c_quad_init"]) - float(cfg["c_quad_min"]))))
        self.k_roll_param = nn.Parameter(torch.tensor(inv_softplus(float(cfg["k_roll_init"]) - float(cfg["k_roll_min"]))))
        self.reset_parameters()
        self.reset_wave_envelope_gate_identity()
        self.reset_wave_shape_correction_identity()
        if (not bool(cfg.get("pinn_enabled", True))) or bool(cfg.get("freeze_physics_coefficients", False)):
            self.c_roll_param.requires_grad_(False)
            self.c_quad_param.requires_grad_(False)
            self.k_roll_param.requires_grad_(False)

    def reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if name.endswith(("c_roll_param", "c_quad_param", "k_roll_param")):
                continue
            if "weight" in name and p.ndim >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)



    def reset_wave_envelope_gate_identity(self) -> None:
        final = self.wave_envelope_gate[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def reset_wave_shape_correction_identity(self) -> None:
        final = self.wave_shape_correction[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @property
    def c_roll(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.c_roll_param) + float(self.cfg["c_roll_min"])

    @property
    def c_quad(self) -> torch.Tensor:
        raw = torch.nn.functional.softplus(self.c_quad_param) + float(self.cfg["c_quad_min"])
        return torch.clamp(raw, max=float(self.cfg.get("c_quad_max", 0.02)))

    @property
    def k_roll(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.k_roll_param) + float(self.cfg["k_roll_min"])

    @property
    def roll_amplitude_gain(self) -> torch.Tensor:
        gain_max = float(max(0.0, self.cfg.get("roll_amplitude_calibration_gain_max", 0.40)))
        return gain_max * torch.sigmoid(self.roll_amplitude_gain_logit)

    def calibrate_roll_amplitude(self, phi_raw: torch.Tensor, v_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(self.cfg.get("roll_amplitude_calibration_enabled", False)):
            unity = 1.0 + 0.0 * phi_raw
            return phi_raw, v_raw, unity
        threshold = float(max(0.0, self.cfg.get("roll_amplitude_calibration_threshold_scaled", 0.75)))
        softness = float(max(1.0e-6, self.cfg.get("roll_amplitude_calibration_softness_scaled", 0.25)))
        gate = torch.sigmoid((torch.abs(phi_raw) - threshold) / softness)
        boost = 1.0 + self.roll_amplitude_gain.to(dtype=phi_raw.dtype, device=phi_raw.device) * gate
        phi = phi_raw * boost
        if bool(self.cfg.get("roll_amplitude_calibration_rate_scale", True)):
            v = v_raw * boost
        else:
            v = v_raw
        return phi, v, boost

    def _sum_feature_coupling(self, X: torch.Tensor, idxs: List[int], param: torch.Tensor) -> torch.Tensor:
        if len(idxs) == 0:
            return 0.0 * X[..., :1]
        values = X[..., idxs]
        return torch.sum(values * param[:len(idxs)].view(1, 1, -1), dim=-1, keepdim=True)

    def wave_orientation_effect_gain(self, X: torch.Tensor) -> torch.Tensor:
        idxs = self.state_feature_indices.get("wave_orientation_effect_gain", [])
        if not bool(self.cfg.get("wave_orientation_gate_enabled", True)) or len(idxs) == 0:
            return 1.0 + 0.0 * X[..., :1]
        gain = torch.mean(X[..., idxs], dim=-1, keepdim=True)
        min_gain = float(np.clip(float(self.cfg.get("wave_parallel_min_gain", 0.20)), 0.0, 1.0))
        return torch.clamp(gain, min=min_gain, max=1.0)

    def learned_wave_envelope_gate(self, X: torch.Tensor) -> torch.Tensor:
        if (
            not bool(self.cfg.get("wave_envelope_gate_enabled", False))
            or self.wave_envelope_gate_input_dim <= 0
        ):
            return 1.0 + 0.0 * X[..., :1]
        parts: List[torch.Tensor] = []
        for key, src in self.wave_envelope_gate_sources:
            idxs = src.get(key, [])
            if len(idxs) > 0:
                parts.append(X[..., idxs])
        if not parts:
            return 1.0 + 0.0 * X[..., :1]
        raw = self.wave_envelope_gate(torch.cat(parts, dim=-1))
        base = float(self.cfg.get("wave_envelope_gate_gain", 0.60))
        gate_min = float(self.cfg.get("wave_envelope_gate_min", 0.35))
        gate_max = float(self.cfg.get("wave_envelope_gate_max", 1.25))
        return torch.clamp(1.0 + base * torch.tanh(raw), min=gate_min, max=gate_max)

    def wave_shape_correction_force(self, X: torch.Tensor) -> torch.Tensor:
        if (
            not bool(self.cfg.get("wave_shape_correction_enabled", False))
            or float(self.cfg.get("wave_shape_correction_gain", 0.0)) <= 0.0
            or self.wave_shape_correction_input_dim <= 0
        ):
            return 0.0 * X[..., :1]
        parts: List[torch.Tensor] = []
        for key, src in self.wave_shape_correction_sources:
            idxs = src.get(key, [])
            if len(idxs) > 0:
                parts.append(X[..., idxs])
        if not parts:
            return 0.0 * X[..., :1]
        correction = self.wave_shape_correction(torch.cat(parts, dim=-1))
        return float(self.cfg.get("wave_shape_correction_gain", 0.20)) * correction

    def wave_forcing_breakdown(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        signal_force = self._sum_feature_coupling(
            X,
            self.wave_feature_indices.get("wave_signal", []),
            self.a_wave,
        )
        slope_force = self._sum_feature_coupling(
            X,
            self.wave_feature_indices.get("wave_slope", []),
            self.a_wave_slope,
        )
        auxiliary_force = 0.0 * X[..., :1]
        auxiliary_force = auxiliary_force + self._sum_feature_coupling(X, self.wave_feature_indices.get("wave_abs_slope", []), self.a_wave_abs_slope)
        auxiliary_force = auxiliary_force + self._sum_feature_coupling(X, self.wave_feature_indices.get("wave_envelope", []), self.a_wave_env)
        auxiliary_force = auxiliary_force + self._sum_feature_coupling(X, self.wave_feature_indices.get("wave_envelope_slow", []), self.a_wave_env_slow)
        auxiliary_force = auxiliary_force + self._sum_feature_coupling(X, self.wave_feature_indices.get("wave_cross_beam_gradient", []), self.a_wave_cross_beam_gradient)
        aux_gain = float(self.cfg.get("wave_auxiliary_force_gain", 0.0))
        if not bool(self.cfg.get("wave_quadrature_basis_enabled", True)):
            aux_gain = 1.0
        shape_correction_force = self.wave_shape_correction_force(X)
        raw_wave_force = signal_force + slope_force + aux_gain * auxiliary_force + shape_correction_force
        state_force = 0.0 * X[..., :1]
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("speed", []), self.a_speed)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("yawrate", []), self.a_yawrate)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("x_position", []), self.a_x_position)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("y_position", []), self.a_y_position)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("track_speed_xy", []), self.a_track_speed_xy)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("wave_perpendicular_velocity", []), self.a_wave_perpendicular_velocity)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("wave_parallel_velocity", []), self.a_wave_parallel_velocity)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("heading_perpendicular_to_waves", []), self.a_heading_perpendicular_to_waves)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("heading_parallel_to_waves", []), self.a_heading_parallel_to_waves)
        state_force = state_force + self._sum_feature_coupling(X, self.state_feature_indices.get("heading_obliquity_abs", []), self.a_heading_obliquity_abs)
        orientation_gain = self.wave_orientation_effect_gain(X)
        envelope_gate = self.learned_wave_envelope_gate(X)
        wave_gain = float(self.cfg["wave_forcing_gain"])
        pure_gate = orientation_gain * envelope_gate
        pure_wave_force_without_envelope_gate = orientation_gain * wave_gain * raw_wave_force
        signal_force = pure_gate * wave_gain * signal_force
        slope_force = pure_gate * wave_gain * slope_force
        auxiliary_force = pure_gate * wave_gain * aux_gain * auxiliary_force
        wave_force = pure_gate * wave_gain * raw_wave_force
        if bool(self.cfg.get("wave_parallel_gate_state_force", True)):
            state_force = orientation_gain * state_force
        state_force = float(self.cfg["state_forcing_gain"]) * state_force
        return {
            "wave_signal_force": signal_force,
            "wave_slope_force": slope_force,
            "wave_auxiliary_force": auxiliary_force,
            "wave_shape_correction_force": pure_gate * wave_gain * shape_correction_force,
            "pure_wave_force": wave_force,
            "pure_wave_force_without_envelope_gate": pure_wave_force_without_envelope_gate,
            "state_force": state_force,
            "wave_orientation_gate": orientation_gain,
            "wave_envelope_gate": envelope_gate,
            "pure_wave_gate": pure_gate,
        }

    def wave_forcing_components(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        breakdown = self.wave_forcing_breakdown(X)
        return breakdown["pure_wave_force"], breakdown["state_force"]

    def pure_wave_forcing(self, X: torch.Tensor) -> torch.Tensor:
        wave_force, _ = self.wave_forcing_components(X)
        return wave_force

    def wave_forcing_base(self, X: torch.Tensor) -> torch.Tensor:
        wave_force, state_force = self.wave_forcing_components(X)
        return wave_force + state_force

    def turn_moment(self, X: torch.Tensor) -> torch.Tensor:
        yaw_idx = self.state_feature_indices.get("yawrate", [])
        if len(yaw_idx) == 0:
            return 0.0 * X[..., :1]
        yaw = X[..., yaw_idx]
        turn = torch.sum(yaw * self.a_turn_yaw[:len(yaw_idx)].view(1, 1, -1), dim=-1, keepdim=True)
        turn = turn + torch.sum(yaw * torch.abs(yaw) * self.a_turn_yaw_abs[:len(yaw_idx)].view(1, 1, -1), dim=-1, keepdim=True)
        speed_idx = self.state_feature_indices.get("speed", [])
        if len(speed_idx) > 0:
            speed = torch.mean(X[..., speed_idx], dim=-1, keepdim=True)
            turn = turn + torch.sum(speed * yaw * self.a_turn_speed_yaw[:len(yaw_idx)].view(1, 1, -1), dim=-1, keepdim=True)
        return float(self.cfg.get("turn_moment_scale", 1.0)) * turn

    def state_dependent_wave_gain(self, recurrent_state: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        parts = [recurrent_state]
        for key, src in (
            ("phi_fb", self.motion_feature_indices), ("v_fb", self.motion_feature_indices),
            ("speed", self.state_feature_indices), ("yawrate", self.state_feature_indices),
            ("x_position", self.state_feature_indices), ("y_position", self.state_feature_indices),
            ("encounter_frequency_ratio", self.state_feature_indices), ("encounter_time_shift", self.state_feature_indices),
            ("track_speed_xy", self.state_feature_indices), ("wave_perpendicular_velocity", self.state_feature_indices),
            ("wave_parallel_velocity", self.state_feature_indices), ("heading_perpendicular_to_waves", self.state_feature_indices),
            ("heading_parallel_to_waves", self.state_feature_indices), ("heading_obliquity_abs", self.state_feature_indices),
            ("wave_orientation_effect_gain", self.state_feature_indices),
            ("wave_envelope", self.wave_feature_indices), ("wave_envelope_slow", self.wave_feature_indices),
            ("wave_cross_beam_gradient", self.wave_feature_indices),
        ):
            idxs = src.get(key, [])
            if len(idxs) > 0:
                values = X[..., idxs]
                if key in {"phi_fb", "v_fb"}:
                    values = float(self.cfg.get("motion_feedback_gate_gain", 1.0)) * values
                parts.append(values)
        gate_raw = self.wave_gate(torch.cat(parts, dim=-1))
        gate = float(self.cfg.get("wave_gate_bias", 1.0)) + float(self.cfg.get("wave_gate_gain", 1.0)) * torch.tanh(gate_raw)
        return torch.clamp(gate, min=0.25, max=2.5)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The Rev M dataset supplies identical causal numeric windows as src/tgt.
        # Using the target window directly preserves the existing call signature
        # while ensuring that output t can only depend on samples <= t.
        sequence = src if tgt is None else tgt
        scale = self.backbone_feature_scale.to(dtype=sequence.dtype, device=sequence.device)
        adapted = self.feature_adapter(sequence * scale)
        recurrent_state, _ = self.lstm(adapted)
        phi_raw = self.phi_head(recurrent_state)
        v_raw = self.v_head(recurrent_state)
        phi, v, _ = self.calibrate_roll_amplitude(phi_raw, v_raw)
        force_residual = float(self.cfg["force_residual_scale"]) * self.force_residual_head(recurrent_state)
        force_base = self.wave_forcing_base(sequence)
        force_wave = self.state_dependent_wave_gain(recurrent_state, sequence) * force_base
        force_total_without_turn = force_wave + force_residual
        return phi, v, force_total_without_turn, force_residual


# =============================================================================
# LOSSES
# =============================================================================

def central_diff_first(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dt = t[:, 2:, :] - t[:, :-2, :]
    return (y[:, 2:, :] - y[:, :-2, :]) / torch.clamp(dt, min=1e-8)


def central_diff_second(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dt = t[:, 2:, :] - t[:, :-2, :]
    return 4.0 * (y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :]) / torch.clamp(dt ** 2, min=1e-8)


def mse_with_peak_weight(pred: torch.Tensor, target: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    se = (pred - target) ** 2
    lam_peak = float(cfg.get("lambda_peak_data", 0.0))
    if lam_peak <= 0.0:
        return torch.mean(se)
    with torch.no_grad():
        abs_target = torch.abs(target)
        threshold = torch.quantile(abs_target.flatten(), float(cfg.get("peak_data_quantile", 0.8)))
        threshold = torch.clamp(threshold, min=1e-8)
        # lambda_peak_data is the strength of the excess extrema weighting, not
        # merely an on/off switch. This makes the declared training/Bayes weight
        # actually control how much high-|roll| samples matter.
        weights = 1.0 + lam_peak * float(cfg.get("peak_data_alpha", 2.0)) * torch.clamp(
            (abs_target - threshold) / threshold,
            min=0.0,
        )
    return torch.sum(weights * se) / torch.clamp(torch.sum(weights), min=1e-8)


def r2_data_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_centered = target - torch.mean(target)
    ss_tot = torch.sum(target_centered ** 2)
    ss_res = torch.sum((pred - target) ** 2)
    return ss_res / torch.clamp(ss_tot, min=1e-8)


def kinematic_loss(phi_pred: torch.Tensor, v_pred: torch.Tensor, t_phys: torch.Tensor,
                   y_v: Optional[torch.Tensor] = None, cfg: Optional[Dict[str, object]] = None) -> torch.Tensor:
    phi_t = central_diff_first(phi_pred, t_phys)
    se = (phi_t - v_pred[:, 1:-1, :]) ** 2
    if y_v is None or cfg is None:
        return torch.mean(se)
    alpha = float(cfg.get("kinematic_weight_alpha", 0.0))
    if alpha <= 0.0:
        return torch.mean(se)
    with torch.no_grad():
        ref_rate = torch.abs(y_v[:, 1:-1, :])
        threshold = torch.quantile(ref_rate.flatten(), float(cfg.get("kinematic_weight_quantile", 0.75)))
        threshold = torch.clamp(threshold, min=1e-8)
        weights = 1.0 + alpha * torch.clamp((ref_rate - threshold) / threshold, min=0.0)
    return torch.sum(weights * se) / torch.clamp(torch.sum(weights), min=1e-8)


def roll_slope_data_loss(phi_pred: torch.Tensor, y_v: torch.Tensor, t_phys: torch.Tensor,
                         cfg: Dict[str, object]) -> torch.Tensor:
    phi_t = central_diff_first(phi_pred, t_phys)
    target_rate = y_v[:, 1:-1, :]
    se = (phi_t - target_rate) ** 2
    alpha = float(cfg.get("kinematic_weight_alpha", 0.0))
    if alpha <= 0.0:
        return torch.mean(se)
    with torch.no_grad():
        ref_rate = torch.abs(target_rate)
        threshold = torch.quantile(ref_rate.flatten(), float(cfg.get("kinematic_weight_quantile", 0.75)))
        threshold = torch.clamp(threshold, min=1e-8)
        weights = 1.0 + alpha * torch.clamp((ref_rate - threshold) / threshold, min=0.0)
    return torch.sum(weights * se) / torch.clamp(torch.sum(weights), min=1e-8)


def peak_trough_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    k = int(max(1, cfg.get("peak_trough_neighbourhood", 1)))
    if phi_pred.shape[1] < 2 * k + 1:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    center = y_phi[:, k:-k, :]
    pred_center = phi_pred[:, k:-k, :]
    is_peak = torch.ones_like(center, dtype=torch.bool)
    is_trough = torch.ones_like(center, dtype=torch.bool)
    for offset in range(1, k + 1):
        left = y_phi[:, k - offset:-k - offset, :]
        right = y_phi[:, k + offset:y_phi.shape[1] - k + offset, :]
        is_peak &= (center >= left) & (center >= right)
        is_trough &= (center <= left) & (center <= right)

    with torch.no_grad():
        amp = torch.abs(center)
        threshold = torch.quantile(torch.abs(y_phi).flatten(), float(cfg.get("peak_trough_quantile", 0.70)))
        threshold = torch.clamp(threshold, min=1e-8)
        extrema = (is_peak | is_trough) & (amp >= threshold)
        underfit = torch.abs(pred_center) < amp
        weights = torch.ones_like(center)
        weights = torch.where(underfit, weights * float(cfg.get("peak_trough_underfit_weight", 1.5)), weights)
        weights = weights * extrema.float()

    denom = torch.sum(weights)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    return torch.sum(weights * (pred_center - center) ** 2) / torch.clamp(denom, min=1e-8)


def extrema_window_underfit_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                                 cfg: Dict[str, object]) -> torch.Tensor:
    """Shift-tolerant underfit loss on target local peaks and troughs."""
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    detect_k = int(max(1, cfg.get("peak_trough_neighbourhood", 1)))
    if phi_pred.shape[1] < 2 * detect_k + 1:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    radius = int(max(0, cfg.get("extrema_window_radius", 3)))
    kernel = 2 * radius + 1
    pred_cf = phi_pred.transpose(1, 2)
    if radius > 0:
        padded = nn.functional.pad(pred_cf, (radius, radius), mode="replicate")
        local_max = nn.functional.max_pool1d(padded, kernel_size=kernel, stride=1).transpose(1, 2)
        local_min = -nn.functional.max_pool1d(-padded, kernel_size=kernel, stride=1).transpose(1, 2)
    else:
        local_max = phi_pred
        local_min = phi_pred

    center = y_phi[:, detect_k:-detect_k, :]
    pred_max = local_max[:, detect_k:-detect_k, :]
    pred_min = local_min[:, detect_k:-detect_k, :]
    is_peak = torch.ones_like(center, dtype=torch.bool)
    is_trough = torch.ones_like(center, dtype=torch.bool)
    for offset in range(1, detect_k + 1):
        left = y_phi[:, detect_k - offset:-detect_k - offset, :]
        right = y_phi[:, detect_k + offset:y_phi.shape[1] - detect_k + offset, :]
        is_peak &= (center >= left) & (center >= right)
        is_trough &= (center <= left) & (center <= right)

    with torch.no_grad():
        amp = torch.abs(center)
        threshold = torch.quantile(
            torch.abs(y_phi).flatten(),
            float(cfg.get("extrema_window_quantile", cfg.get("peak_trough_quantile", 0.55))),
        )
        threshold = torch.clamp(threshold, min=1.0e-8)
        active = (is_peak | is_trough) & (amp >= threshold)
        scale_floor = threshold * float(max(0.0, cfg.get("extrema_window_scale_floor_ratio", 0.08)))
        scale = torch.clamp(amp, min=torch.clamp(scale_floor, min=1.0e-8))
        weights = active.float() * (1.0 + torch.clamp((amp - threshold) / threshold, min=0.0))

    peak_deficit = torch.relu(center - pred_max)
    peak_overshoot = torch.relu(pred_max - center)
    trough_deficit = torch.relu(pred_min - center)
    trough_overshoot = torch.relu(center - pred_min)
    overshoot_weight = float(max(0.0, cfg.get("extrema_window_overshoot_weight", 0.10)))
    peak_penalty = (peak_deficit ** 2 + overshoot_weight * peak_overshoot ** 2) / (scale ** 2)
    trough_penalty = (trough_deficit ** 2 + overshoot_weight * trough_overshoot ** 2) / (scale ** 2)
    penalty = (
        float(max(0.0, cfg.get("extrema_window_peak_weight", 1.0))) * is_peak.float() * peak_penalty
        + float(max(0.0, cfg.get("extrema_window_trough_weight", 1.0))) * is_trough.float() * trough_penalty
    )
    denom = torch.sum(weights)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    return torch.sum(weights * penalty) / torch.clamp(denom, min=1.0e-8)


def global_extrema_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                        t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    pred = phi_pred.squeeze(-1)
    target = y_phi.squeeze(-1)
    tt = t_phys.squeeze(-1)
    duration = torch.clamp(tt[:, -1:] - tt[:, :1], min=1.0e-8)
    t_norm = (tt - tt[:, :1]) / duration
    beta = float(max(1.0e-6, cfg.get("global_extrema_softmax_beta", 12.0)))
    max_weights = torch.softmax(beta * pred, dim=1)
    min_weights = torch.softmax(-beta * pred, dim=1)
    pred_max_value = torch.sum(max_weights * pred, dim=1)
    pred_min_value = torch.sum(min_weights * pred, dim=1)
    pred_max_time = torch.sum(max_weights * t_norm, dim=1)
    pred_min_time = torch.sum(min_weights * t_norm, dim=1)

    target_max_value, target_max_idx = torch.max(target, dim=1)
    target_min_value, target_min_idx = torch.min(target, dim=1)
    target_max_time = torch.gather(t_norm, 1, target_max_idx.view(-1, 1)).squeeze(1)
    target_min_time = torch.gather(t_norm, 1, target_min_idx.view(-1, 1)).squeeze(1)

    value_loss = (pred_max_value - target_max_value) ** 2 + (pred_min_value - target_min_value) ** 2
    time_loss = (pred_max_time - target_max_time) ** 2 + (pred_min_time - target_min_time) ** 2
    return torch.mean(value_loss + float(cfg.get("global_extrema_location_weight", 0.35)) * time_loss)


def amplitude_underfit_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                            t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = torch.median(torch.clamp(dt, min=1.0e-8))
        window_steps = int(round(float(cfg.get("amplitude_window_s", 1.0)) / max(float(dt_med.detach().cpu().item()), 1.0e-8)))
        window_steps = max(3, min(window_steps, int(phi_pred.shape[1])))
        if window_steps % 2 == 0:
            window_steps = max(3, window_steps - 1)

    pad = window_steps // 2
    pred_env = torch.sqrt(
        nn.functional.avg_pool1d(
            phi_pred.transpose(1, 2) ** 2,
            kernel_size=window_steps,
            stride=1,
            padding=pad,
        ).transpose(1, 2).clamp_min(0.0)
        + 1.0e-12
    )
    target_env = torch.sqrt(
        nn.functional.avg_pool1d(
            y_phi.transpose(1, 2) ** 2,
            kernel_size=window_steps,
            stride=1,
            padding=pad,
        ).transpose(1, 2).clamp_min(0.0)
        + 1.0e-12
    )
    margin = float(max(0.0, cfg.get("amplitude_underfit_margin", 0.0)))
    deficit = torch.relu(target_env * (1.0 - margin) - pred_env)
    power = float(cfg.get("amplitude_underfit_power", 2.0))
    if abs(power - 1.0) < 1.0e-8:
        penalty = deficit / torch.clamp(target_env, min=1.0e-8)
    else:
        penalty = (deficit / torch.clamp(target_env, min=1.0e-8)) ** power

    with torch.no_grad():
        threshold = torch.quantile(target_env.flatten(), float(cfg.get("amplitude_quantile", 0.70)))
        threshold = torch.clamp(threshold, min=1.0e-8)
        weights = 1.0 + torch.clamp((target_env - threshold) / threshold, min=0.0)
    return torch.sum(weights * penalty) / torch.clamp(torch.sum(weights), min=1.0e-8)


def high_amplitude_underfit_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                                 cfg: Dict[str, object]) -> torch.Tensor:
    """Directional loss on high-|roll| samples, including extrema shoulders."""
    if phi_pred.shape[1] < 2:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    with torch.no_grad():
        amp = torch.abs(y_phi)
        threshold = torch.quantile(
            amp.flatten(),
            float(cfg.get("high_amplitude_underfit_quantile", 0.68)),
        )
        threshold = torch.clamp(threshold, min=1.0e-8)
        active = amp >= threshold
        scale_floor = threshold * float(max(0.0, cfg.get("high_amplitude_underfit_scale_floor_ratio", 0.08)))
        scale = torch.clamp(amp, min=torch.clamp(scale_floor, min=1.0e-8))
        weights = active.float() * (1.0 + torch.clamp((amp - threshold) / threshold, min=0.0))

    positive_deficit = torch.relu(y_phi - phi_pred)
    positive_overshoot = torch.relu(phi_pred - y_phi)
    negative_deficit = torch.relu(phi_pred - y_phi)
    negative_overshoot = torch.relu(y_phi - phi_pred)
    overshoot_weight = float(max(0.0, cfg.get("high_amplitude_underfit_overshoot_weight", 0.08)))
    positive_penalty = positive_deficit ** 2 + overshoot_weight * positive_overshoot ** 2
    negative_penalty = negative_deficit ** 2 + overshoot_weight * negative_overshoot ** 2
    penalty = torch.where(y_phi >= 0.0, positive_penalty, negative_penalty) / (scale ** 2)
    denom = torch.sum(weights)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    return torch.sum(weights * penalty) / torch.clamp(denom, min=1.0e-8)


def asymmetric_extrema_forecast_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                                     cfg: Dict[str, object]) -> torch.Tensor:
    """Penalise weak positive peaks and over-deep negative troughs in forecast roll."""
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    detect_k = int(max(1, cfg.get("peak_trough_neighbourhood", 1)))
    if phi_pred.shape[1] < 2 * detect_k + 1:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    radius = int(max(0, cfg.get("direct_forecast_asym_extrema_radius", cfg.get("extrema_window_radius", 3))))
    kernel = 2 * radius + 1
    pred_cf = phi_pred.transpose(1, 2)
    if radius > 0:
        padded = nn.functional.pad(pred_cf, (radius, radius), mode="replicate")
        pred_local_max = nn.functional.max_pool1d(padded, kernel_size=kernel, stride=1).transpose(1, 2)
        pred_local_min = -nn.functional.max_pool1d(-padded, kernel_size=kernel, stride=1).transpose(1, 2)
    else:
        pred_local_max = phi_pred
        pred_local_min = phi_pred

    center = y_phi[:, detect_k:-detect_k, :]
    pred_max = pred_local_max[:, detect_k:-detect_k, :]
    pred_min = pred_local_min[:, detect_k:-detect_k, :]
    is_peak = torch.ones_like(center, dtype=torch.bool)
    is_trough = torch.ones_like(center, dtype=torch.bool)
    for offset in range(1, detect_k + 1):
        left = y_phi[:, detect_k - offset:-detect_k - offset, :]
        right = y_phi[:, detect_k + offset:y_phi.shape[1] - detect_k + offset, :]
        is_peak &= (center >= left) & (center >= right)
        is_trough &= (center <= left) & (center <= right)

    with torch.no_grad():
        amp = torch.abs(center)
        threshold = torch.quantile(
            torch.abs(y_phi).flatten(),
            float(cfg.get("direct_forecast_asym_extrema_quantile", cfg.get("peak_trough_quantile", 0.45))),
        )
        threshold = torch.clamp(threshold, min=1.0e-8)
        scale_floor = threshold * float(max(0.0, cfg.get("direct_forecast_asym_extrema_scale_floor_ratio", 0.06)))
        scale = torch.clamp(amp, min=torch.clamp(scale_floor, min=1.0e-8))
        peak_mask = is_peak & (center > 0.0) & (amp >= threshold)
        trough_mask = is_trough & (center < 0.0) & (amp >= threshold)
        weights = (peak_mask | trough_mask).float() * (1.0 + torch.clamp((amp - threshold) / threshold, min=0.0))

    peak_underfit = torch.relu(center - pred_max)
    peak_overshoot = torch.relu(pred_max - center)
    trough_overshoot = torch.relu(center - pred_min)
    trough_underfit = torch.relu(pred_min - center)
    peak_loss = (
        float(cfg.get("direct_forecast_peak_underfit_weight", 4.0)) * peak_underfit ** 2
        + float(cfg.get("direct_forecast_peak_overshoot_weight", 0.35)) * peak_overshoot ** 2
    )
    trough_loss = (
        float(cfg.get("direct_forecast_trough_overshoot_weight", 4.5)) * trough_overshoot ** 2
        + float(cfg.get("direct_forecast_trough_underfit_weight", 0.75)) * trough_underfit ** 2
    )
    penalty = (peak_mask.float() * peak_loss + trough_mask.float() * trough_loss) / (scale ** 2)
    denom = torch.sum(weights)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    return torch.sum(weights * penalty) / torch.clamp(denom, min=1.0e-8)


def time_window_steps(t_phys: torch.Tensor, window_s: float, sequence_length: int) -> int:
    """Convert a physical window to a safe odd number of sequence samples."""
    if sequence_length < 3 or window_s <= 0.0:
        return 1
    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item()) if dt.numel() else 1.0
    steps = int(round(float(window_s) / max(dt_med, 1.0e-8)))
    max_odd = sequence_length if sequence_length % 2 == 1 else sequence_length - 1
    steps = max(3, min(steps, max_odd))
    if steps % 2 == 0:
        steps = max(3, steps - 1)
    return steps


def moving_average_sequence(y: torch.Tensor, window_steps: int) -> torch.Tensor:
    """Centred moving average with replicated edges and unchanged length."""
    if window_steps <= 1:
        return y
    pad = int(window_steps) // 2
    channels_first = y.transpose(1, 2)
    padded = nn.functional.pad(channels_first, (pad, pad), mode="replicate")
    return nn.functional.avg_pool1d(padded, kernel_size=int(window_steps), stride=1).transpose(1, 2)


def high_pass_residual_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                            t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    """Scale-normalised loss on sub-cycle roll structure."""
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    steps = time_window_steps(
        t_phys,
        float(cfg.get("high_pass_window_s", 0.6)),
        int(phi_pred.shape[1]),
    )
    pred_hp = phi_pred - moving_average_sequence(phi_pred, steps)
    target_hp = y_phi - moving_average_sequence(y_phi, steps)
    target_hp_rms = torch.sqrt(torch.mean(target_hp ** 2, dim=1, keepdim=True) + 1.0e-12)
    target_rms = torch.sqrt(torch.mean(y_phi ** 2, dim=1, keepdim=True) + 1.0e-12)
    floor = float(max(0.0, cfg.get("high_pass_scale_floor_ratio", 0.05))) * target_rms
    scale = torch.maximum(target_hp_rms, floor + 1.0e-8)
    normalised_se = ((pred_hp - target_hp) / scale) ** 2
    return torch.mean(torch.log1p(normalised_se))


def low_pass_residual_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                           t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    """Scale-normalised loss on slow roll trend/envelope structure."""
    if phi_pred.shape[1] < 3:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    steps = time_window_steps(
        t_phys,
        float(cfg.get("low_pass_window_s", 3.2)),
        int(phi_pred.shape[1]),
    )
    pred_lp = moving_average_sequence(phi_pred, steps)
    target_lp = moving_average_sequence(y_phi, steps)
    target_lp_rms = torch.sqrt(torch.mean(target_lp ** 2, dim=1, keepdim=True) + 1.0e-12)
    target_rms = torch.sqrt(torch.mean(y_phi ** 2, dim=1, keepdim=True) + 1.0e-12)
    floor = float(max(0.0, cfg.get("low_pass_scale_floor_ratio", 0.05))) * target_rms
    scale = torch.maximum(target_lp_rms, floor + 1.0e-8)
    normalised_se = ((pred_lp - target_lp) / scale) ** 2
    return torch.mean(torch.log1p(normalised_se))


def roll_curvature_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                        t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    """Compare high-passed roll curvature to recover fast local reversals."""
    if float(cfg.get("lambda_roll_curvature", 0.0)) <= 0.0 or phi_pred.shape[1] < 5:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    steps = time_window_steps(
        t_phys,
        float(cfg.get("curvature_high_pass_window_s", cfg.get("high_pass_window_s", 0.25))),
        int(phi_pred.shape[1]),
    )
    pred_hp = phi_pred - moving_average_sequence(phi_pred, steps)
    target_hp = y_phi - moving_average_sequence(y_phi, steps)
    pred_curv = central_diff_second(pred_hp, t_phys)
    target_curv = central_diff_second(target_hp, t_phys)
    target_curv_rms = torch.sqrt(torch.mean(target_curv ** 2, dim=1, keepdim=True) + 1.0e-12)
    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item()) if dt.numel() else 1.0
        if not math.isfinite(dt_med) or dt_med <= 0.0:
            dt_med = 1.0
    target_rms = torch.sqrt(torch.mean(y_phi ** 2, dim=1, keepdim=True) + 1.0e-12)
    floor = (
        float(max(0.0, cfg.get("curvature_scale_floor_ratio", 0.04)))
        * target_rms
        / max(dt_med ** 2, 1.0e-8)
    )
    scale = torch.maximum(target_curv_rms, floor + 1.0e-8)
    normalised_se = ((pred_curv - target_curv) / scale) ** 2
    return torch.mean(torch.log1p(normalised_se))


def roll_spectral_shape_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                             t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    """Match measured roll spectrum in targeted low, 1 Hz, and tail bands."""
    if float(cfg.get("lambda_roll_spectral_shape", 0.0)) <= 0.0 or phi_pred.shape[1] < 8:
        return torch.mean((phi_pred * 0.0) ** 2)

    n = int(phi_pred.shape[1])
    lows = config_float_list(cfg, "roll_spectral_shape_band_lows_hz", [0.0, 0.85, 1.55])
    highs = config_float_list(cfg, "roll_spectral_shape_band_highs_hz", [0.28, 1.15, 4.20])
    weights = config_float_list(cfg, "roll_spectral_shape_weights", [2.5, 4.0, 3.5])
    raw_bands = config_float_list(cfg, "roll_spectral_shape_raw_bands", [1.0, 0.0, 0.0])
    n_bands = min(len(lows), len(highs))
    if len(weights) < n_bands:
        weights = weights + [1.0] * (n_bands - len(weights))
    if len(raw_bands) < n_bands:
        raw_bands = raw_bands + [0.0] * (n_bands - len(raw_bands))

    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item()) if dt.numel() else 1.0
        if not math.isfinite(dt_med) or dt_med <= 0.0:
            dt_med = 1.0
    freqs = torch.fft.rfftfreq(n, d=dt_med, device=phi_pred.device)
    window = torch.hann_window(
        n,
        periodic=False,
        dtype=phi_pred.dtype,
        device=phi_pred.device,
    ).view(1, n)
    norm = torch.clamp(torch.sum(window), min=1.0e-8)

    def spectrum(values: torch.Tensor, *, demean: bool) -> torch.Tensor:
        x = values[..., 0]
        if demean:
            x = x - torch.mean(x, dim=1, keepdim=True)
        return torch.fft.rfft(x * window, dim=1) / norm

    pred_raw = spectrum(phi_pred, demean=False)
    target_raw = spectrum(y_phi, demean=False).detach()
    pred_demeaned = spectrum(phi_pred, demean=True)
    target_demeaned = spectrum(y_phi, demean=True).detach()

    zero = torch.mean((phi_pred * 0.0) ** 2)
    total_loss = zero
    total_weight = 0.0
    underfit_weight = float(cfg.get("roll_spectral_shape_underfit_weight", 2.0))
    complex_weight = float(cfg.get("roll_spectral_shape_complex_weight", 0.15))
    mag_floor_ratio = max(0.0, float(cfg.get("roll_spectral_shape_mag_floor_ratio", 0.025)))

    for idx in range(n_bands):
        low_hz = float(lows[idx])
        high_hz = float(highs[idx])
        band_weight = float(weights[idx])
        if high_hz <= low_hz or band_weight <= 0.0:
            continue
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        if int(torch.sum(band_mask).detach().cpu().item()) <= 0:
            continue

        use_raw = float(raw_bands[idx]) > 0.5
        pred_spec = pred_raw[:, band_mask] if use_raw else pred_demeaned[:, band_mask]
        target_spec = target_raw[:, band_mask] if use_raw else target_demeaned[:, band_mask]
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec).detach()
        target_power = torch.clamp(torch.mean(target_mag ** 2).detach(), min=1.0e-8)
        mag_floor = torch.clamp(torch.sqrt(target_power) * mag_floor_ratio, min=1.0e-6)
        pred_safe = torch.clamp(pred_mag, min=mag_floor)
        target_safe = torch.clamp(target_mag, min=mag_floor)
        with torch.no_grad():
            bin_weight = target_mag / torch.clamp(
                torch.mean(target_mag, dim=1, keepdim=True),
                min=mag_floor,
            )
            bin_weight = torch.clamp(bin_weight, min=0.25, max=6.0)

        log_error = torch.log(pred_safe) - torch.log(target_safe)
        amp_loss = torch.sum(bin_weight * log_error ** 2) / torch.clamp(
            torch.sum(bin_weight),
            min=1.0e-8,
        )
        under_loss = torch.sum(
            bin_weight * torch.relu(torch.log(target_safe) - torch.log(pred_safe)) ** 2
        ) / torch.clamp(torch.sum(bin_weight), min=1.0e-8)
        band_loss = amp_loss + underfit_weight * under_loss
        if complex_weight > 0.0:
            band_loss = band_loss + complex_weight * (
                torch.mean(torch.abs(pred_spec - target_spec) ** 2) / target_power
            )
        total_loss = total_loss + band_weight * band_loss
        total_weight += band_weight

    if total_weight <= 0.0:
        return zero
    return total_loss / total_weight


def local_prominence_extrema_loss(phi_pred: torch.Tensor, y_phi: torch.Tensor,
                                  t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    """Score extrema by local prominence instead of absolute roll amplitude."""
    T = int(phi_pred.shape[1])
    k = int(max(1, cfg.get("local_prominence_neighbourhood", 3)))
    if T < 2 * k + 1:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0

    steps = time_window_steps(t_phys, float(cfg.get("local_prominence_window_s", 0.8)), T)
    pred_residual = phi_pred - moving_average_sequence(phi_pred, steps)
    target_residual = y_phi - moving_average_sequence(y_phi, steps)
    target_local_rms = torch.sqrt(
        moving_average_sequence(target_residual ** 2, steps).clamp_min(0.0) + 1.0e-12
    )
    target_rms = torch.sqrt(torch.mean(y_phi ** 2, dim=1, keepdim=True) + 1.0e-12)
    scale_floor = float(max(0.0, cfg.get("local_prominence_scale_floor_ratio", 0.08))) * target_rms
    local_scale = torch.maximum(target_local_rms, scale_floor + 1.0e-8)

    center = y_phi[:, k:-k, :]
    is_peak = torch.ones_like(center, dtype=torch.bool)
    is_trough = torch.ones_like(center, dtype=torch.bool)
    for offset in range(1, k + 1):
        left = y_phi[:, k - offset:T - k - offset, :]
        right = y_phi[:, k + offset:T - k + offset, :]
        is_peak &= (center >= left) & (center >= right)
        is_trough &= (center <= left) & (center <= right)

    target_residual_center = target_residual[:, k:-k, :]
    pred_residual_center = pred_residual[:, k:-k, :]
    scale_center = local_scale[:, k:-k, :]
    with torch.no_grad():
        relative_prominence = torch.abs(target_residual_center) / torch.clamp(scale_center, min=1.0e-8)
        threshold = torch.quantile(
            relative_prominence.flatten(),
            float(cfg.get("local_prominence_quantile", 0.35)),
        )
        extrema_mask = (is_peak | is_trough) & (relative_prominence >= threshold)
        weights = extrema_mask.float() * torch.clamp(relative_prominence, min=0.5, max=3.0)

    denom = torch.sum(weights)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean((phi_pred - y_phi) ** 2) * 0.0
    scale_center = torch.clamp(scale_center, min=1.0e-8)
    signed_error = ((pred_residual_center - target_residual_center) / scale_center) ** 2
    prominence_error = (
        (torch.abs(pred_residual_center) - torch.abs(target_residual_center)) / scale_center
    ) ** 2
    value_weight = float(cfg.get("local_prominence_value_weight", 1.0))
    shape_weight = float(cfg.get("local_prominence_shape_weight", 0.5))
    local_error = value_weight * signed_error + shape_weight * prominence_error
    return torch.sum(weights * torch.log1p(local_error)) / torch.clamp(denom, min=1.0e-8)


def physics_loss(model: PINNLSTM, phi_pred: torch.Tensor, v_pred: torch.Tensor,
                 force_pred: torch.Tensor, t_phys: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    v_t = central_diff_first(v_pred, t_phys)
    v_mid = v_pred[:, 1:-1, :]
    phi_mid = phi_pred[:, 1:-1, :]
    force_mid = force_pred[:, 1:-1, :] + model.turn_moment(X)[:, 1:-1, :]
    residual = v_t + model.c_roll * v_mid + model.c_quad * torch.abs(v_mid) * v_mid + model.k_roll * phi_mid - force_mid
    return torch.mean(residual ** 2)


def boundary_loss(phi_pred: torch.Tensor, v_pred: torch.Tensor, y_phi: torch.Tensor, y_v: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(phi_pred[:, :1, :], y_phi[:, :1, :]) + nn.functional.mse_loss(v_pred[:, :1, :], y_v[:, :1, :])


def force_smoothness_loss(force_pred: torch.Tensor, t_phys: torch.Tensor) -> torch.Tensor:
    return torch.mean(central_diff_first(force_pred, t_phys) ** 2)


def smooth_sequence_seconds(values: torch.Tensor, t_phys: torch.Tensor,
                            smoothing_seconds: float) -> torch.Tensor:
    """Replicate-padded moving average for a [batch, time, channel] tensor."""
    if values.shape[1] < 3 or float(smoothing_seconds) <= 0.0:
        return values
    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item())
        kernel = max(1, int(round(float(smoothing_seconds) / max(dt_med, 1.0e-8))))
        if kernel % 2 == 0:
            kernel += 1
        max_kernel = int(values.shape[1])
        if max_kernel % 2 == 0:
            max_kernel -= 1
        kernel = min(kernel, max(1, max_kernel))
    if kernel <= 1:
        return values
    pad = kernel // 2
    padded = nn.functional.pad(values.transpose(1, 2), (pad, pad), mode="replicate")
    return nn.functional.avg_pool1d(padded, kernel_size=kernel, stride=1).transpose(1, 2)


def measured_wave_force_target(model: PINNLSTM,
                               y_phi: torch.Tensor, y_v: torch.Tensor,
                               t_phys: torch.Tensor, X: torch.Tensor,
                               cfg: Dict[str, object]) -> torch.Tensor:
    """External wave force implied by measured roll/rate in scaled coordinates."""
    v_t = central_diff_first(y_v, t_phys)
    v_mid = y_v[:, 1:-1, :]
    phi_mid = y_phi[:, 1:-1, :]
    turn_mid = model.turn_moment(X)[:, 1:-1, :]
    target = (
        v_t
        + model.c_roll * v_mid
        + model.c_quad * torch.abs(v_mid) * v_mid
        + model.k_roll * phi_mid
        - turn_mid
    )
    target = smooth_sequence_seconds(
        target,
        t_phys[:, 1:-1, :],
        float(cfg.get("wave_force_target_smoothing_seconds", 0.40)),
    )
    # This loss supervises the wave path, not the learned c/k coefficients.
    return target.detach()


def force_target_amplitude_loss(pred: torch.Tensor, target: torch.Tensor,
                                cfg: Dict[str, object]) -> torch.Tensor:
    pred_rms = torch.sqrt(torch.mean(pred ** 2) + 1.0e-12)
    target_rms = torch.sqrt(torch.mean(target ** 2) + 1.0e-12).detach()
    log_ratio = torch.log(torch.clamp(pred_rms, min=1.0e-8)) - torch.log(torch.clamp(target_rms, min=1.0e-8))
    ratio = pred_rms / torch.clamp(target_rms, min=1.0e-8)
    underforce = torch.relu(1.0 - ratio) ** 2
    amp_weight = float(cfg.get("wave_force_target_amplitude_weight", 0.0))
    under_weight = float(cfg.get("wave_force_target_underforce_weight", 0.0))
    return amp_weight * log_ratio ** 2 + under_weight * underforce


def force_target_highpass_loss(pred: torch.Tensor, target: torch.Tensor,
                               t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    weight = float(cfg.get("lambda_wave_force_highpass", 0.0))
    if weight <= 0.0 or pred.shape[1] < 3:
        return torch.mean((pred - target) ** 2) * 0.0
    window_s = float(cfg.get("wave_force_highpass_window_s", cfg.get("high_pass_window_s", 0.6)))
    pred_hp = pred - smooth_sequence_seconds(pred, t_phys, window_s)
    target_hp = target - smooth_sequence_seconds(target, t_phys, window_s)
    target_hp_power = torch.clamp(torch.mean(target_hp ** 2).detach(), min=1.0e-4)
    normalized_mse = torch.mean((pred_hp - target_hp) ** 2) / target_hp_power
    return weight * (normalized_mse + 0.5 * phase_correlation_loss(pred_hp, target_hp))


def force_target_bandpass(pred: torch.Tensor, target: torch.Tensor,
                          t_phys: torch.Tensor,
                          cfg: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_period = cfg.get("force_band_amplitude_center_period_s", cfg.get("dominant_roll_period_s", 1.0))
    try:
        center_period_s = float(raw_period)
    except (TypeError, ValueError):
        center_period_s = 1.0
    if not math.isfinite(center_period_s) or center_period_s <= 0.0:
        center_period_s = 1.0
    low_factor = float(cfg.get("bandpass_low_period_factor", 1.8))
    high_factor = float(cfg.get("bandpass_high_period_factor", 0.35))
    low_window_s = max(center_period_s * low_factor, 1.0e-6)
    high_window_s = max(center_period_s * high_factor, 1.0e-6)

    pred_hp = pred - smooth_sequence_seconds(pred, t_phys, low_window_s)
    target_hp = target - smooth_sequence_seconds(target, t_phys, low_window_s)
    pred_band = smooth_sequence_seconds(pred_hp, t_phys, high_window_s)
    target_band = smooth_sequence_seconds(target_hp, t_phys, high_window_s)
    return pred_band, target_band


def config_float_list(cfg: Dict[str, object], key: str,
                      default: Iterable[float]) -> List[float]:
    raw = cfg.get(key, default)
    if isinstance(raw, str):
        items = raw.replace(";", ",").split(",")
    elif isinstance(raw, np.ndarray):
        items = raw.tolist()
    elif isinstance(raw, Iterable):
        items = list(raw)
    else:
        items = [raw]
    values: List[float] = []
    for item in items:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if values:
        return values
    return [float(v) for v in default]


def normalized_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                          target_power: torch.Tensor,
                          beta: float) -> torch.Tensor:
    scale = torch.sqrt(torch.clamp(target_power.detach(), min=1.0e-8))
    residual = (pred - target) / torch.clamp(scale, min=1.0e-8)
    return nn.functional.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=max(float(beta), 1.0e-6),
    )


def robust_force_target_fit_loss(pred: torch.Tensor, target: torch.Tensor,
                                 t_phys: torch.Tensor,
                                 cfg: Dict[str, object]) -> torch.Tensor:
    band_weight = float(cfg.get("force_target_band_mse_weight", 1.0))
    raw_weight = float(cfg.get("force_target_raw_weight", 0.0))
    if pred.shape[1] < 5:
        band_weight = 0.0
        raw_weight = max(raw_weight, 1.0)
    if band_weight <= 0.0 and raw_weight <= 0.0:
        return torch.mean((pred - target) ** 2) * 0.0

    loss = torch.mean((pred - target) ** 2) * 0.0
    if band_weight > 0.0:
        pred_band, target_band = force_target_bandpass(pred, target, t_phys, cfg)
        band_power = torch.clamp(torch.mean(target_band ** 2).detach(), min=1.0e-4)
        loss = loss + band_weight * normalized_huber_loss(
            pred_band,
            target_band,
            band_power,
            float(cfg.get("force_target_band_huber_delta", cfg.get("force_target_huber_delta", 1.5))),
        )
    if raw_weight > 0.0:
        raw_power = torch.clamp(torch.mean(target ** 2).detach(), min=1.0e-4)
        loss = loss + raw_weight * normalized_huber_loss(
            pred,
            target,
            raw_power,
            float(cfg.get("force_target_huber_delta", 1.5)),
        )
    return loss


def force_target_band_amplitude_loss(pred: torch.Tensor, target: torch.Tensor,
                                     t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    weight = float(cfg.get("lambda_force_band_amplitude", 0.0))
    under_weight = float(cfg.get("force_band_amplitude_underfit_weight", 0.0))
    if (weight <= 0.0 and under_weight <= 0.0) or pred.shape[1] < 5:
        return torch.mean((pred - target) ** 2) * 0.0
    pred_band, target_band = force_target_bandpass(pred, target, t_phys, cfg)
    pred_rms = torch.sqrt(torch.mean(pred_band ** 2) + 1.0e-12)
    target_rms = torch.sqrt(torch.mean(target_band ** 2) + 1.0e-12).detach()
    log_ratio = torch.log(torch.clamp(pred_rms, min=1.0e-8)) - torch.log(torch.clamp(target_rms, min=1.0e-8))
    ratio = pred_rms / torch.clamp(target_rms, min=1.0e-8)
    underfit = torch.relu(1.0 - ratio) ** 2
    return weight * log_ratio ** 2 + under_weight * underfit


def force_target_band_phase_loss(pred: torch.Tensor, target: torch.Tensor,
                                 t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    weight = float(cfg.get("lambda_force_band_phase", 0.0))
    if weight <= 0.0 or pred.shape[1] < 5:
        return torch.mean((pred - target) ** 2) * 0.0
    pred_band, target_band = force_target_bandpass(pred, target, t_phys, cfg)
    return weight * phase_correlation_loss(pred_band, target_band)


def sequence_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - torch.mean(pred, dim=1, keepdim=True)
    target_c = target - torch.mean(target, dim=1, keepdim=True)
    numerator = torch.sum(pred_c * target_c, dim=(1, 2))
    denom = torch.sqrt(
        torch.sum(pred_c ** 2, dim=(1, 2))
        * torch.sum(target_c ** 2, dim=(1, 2))
        + 1.0e-12
    )
    corr = numerator / torch.clamp(denom, min=1.0e-8)
    return torch.mean(corr)


def lagged_sequence_correlation(pred: torch.Tensor, target: torch.Tensor,
                                lag_steps: int) -> torch.Tensor:
    lag = int(lag_steps)
    if lag < 0:
        return sequence_correlation(pred[:, :lag, :], target[:, -lag:, :])
    if lag > 0:
        return sequence_correlation(pred[:, lag:, :], target[:, :-lag, :])
    return sequence_correlation(pred, target)


def force_target_lag_penalty_loss(pred: torch.Tensor, target: torch.Tensor,
                                  t_phys: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    weight = float(cfg.get("lambda_force_lag_penalty", 0.0))
    if weight <= 0.0 or pred.shape[1] < 7:
        return torch.mean((pred - target) ** 2) * 0.0
    pred_band, target_band = force_target_bandpass(pred, target, t_phys, cfg)
    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item())
        max_s = max(0.0, float(cfg.get("force_lag_penalty_max_s", 1.2)))
        max_lag_steps = int(round(max_s / max(dt_med, 1.0e-8)))
        max_lag_steps = int(min(max_lag_steps, max(1, pred_band.shape[1] // 3)))
        n_lags = int(max(3, cfg.get("force_lag_penalty_steps", 7)))
        lag_steps = np.unique(
            np.round(np.linspace(-max_lag_steps, max_lag_steps, n_lags)).astype(int)
        )
        if 0 not in lag_steps:
            lag_steps = np.sort(np.append(lag_steps, 0))
    if max_lag_steps <= 0 or len(lag_steps) <= 1:
        return weight * phase_correlation_loss(pred_band, target_band)

    corr_values: List[torch.Tensor] = []
    valid_lags: List[int] = []
    for lag in lag_steps:
        lag_i = int(lag)
        if pred_band.shape[1] - abs(lag_i) < 3:
            continue
        corr_values.append(lagged_sequence_correlation(pred_band, target_band, lag_i))
        valid_lags.append(lag_i)
    if not corr_values:
        return weight * phase_correlation_loss(pred_band, target_band)

    corrs = torch.stack(corr_values)
    lag_abs = torch.as_tensor(
        [abs(v) / max(float(max_lag_steps), 1.0) for v in valid_lags],
        dtype=pred_band.dtype,
        device=pred_band.device,
    )
    zero_idx = valid_lags.index(0) if 0 in valid_lags else int(torch.argmin(lag_abs).detach().cpu().item())
    zero_corr = corrs[zero_idx]
    lag_weights = torch.softmax(8.0 * corrs, dim=0)
    expected_abs_lag = torch.sum(lag_weights * lag_abs)
    return weight * ((1.0 - zero_corr) + expected_abs_lag)


def force_target_envelope_loss(pred: torch.Tensor, target: torch.Tensor,
                               t_phys: torch.Tensor,
                               cfg: Dict[str, object]) -> torch.Tensor:
    if pred.shape[1] < 5:
        return torch.mean((pred - target) ** 2) * 0.0
    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    if window_s <= 0.0:
        return torch.mean((pred - target) ** 2) * 0.0
    pred_band, target_band = force_target_bandpass(pred, target, t_phys, cfg)
    pred_env = smooth_sequence_seconds(torch.abs(pred_band), t_phys, window_s)
    target_env = smooth_sequence_seconds(torch.abs(target_band), t_phys, window_s)
    env_power = torch.clamp(torch.mean(target_env ** 2).detach(), min=1.0e-4)
    fit = normalized_huber_loss(
        pred_env,
        target_env,
        env_power,
        float(cfg.get("force_envelope_huber_delta", 0.85)),
    )
    pred_rms = torch.sqrt(torch.mean(pred_env ** 2) + 1.0e-12)
    target_rms = torch.sqrt(torch.mean(target_env ** 2) + 1.0e-12).detach()
    log_ratio = (
        torch.log(torch.clamp(pred_rms, min=1.0e-8))
        - torch.log(torch.clamp(target_rms, min=1.0e-8))
    )
    underfit = torch.relu(1.0 - pred_rms / torch.clamp(target_rms, min=1.0e-8)) ** 2
    return fit + 0.25 * log_ratio ** 2 + 0.50 * underfit


def force_mean_loss(force_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.mean(force_pred, dim=1) ** 2)


def wave_envelope_gate_regularization_loss(model: PINNLSTM,
                                           X: torch.Tensor,
                                           cfg: Dict[str, object]) -> torch.Tensor:
    if (
        not bool(cfg.get("wave_envelope_gate_enabled", True))
        or float(cfg.get("lambda_wave_envelope_gate_reg", 0.0)) <= 0.0
    ):
        return torch.mean((X[..., :1] * 0.0) ** 2)
    gate = model.learned_wave_envelope_gate(X)
    return torch.mean((gate - 1.0) ** 2)


def wave_envelope_gate_target_terms(model: PINNLSTM,
                                    y_phi: torch.Tensor, y_v: torch.Tensor,
                                    t_phys: torch.Tensor, X: torch.Tensor,
                                    cfg: Dict[str, object]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not bool(cfg.get("wave_envelope_gate_enabled", True)) or X.shape[1] < 5:
        return None
    breakdown = model.wave_forcing_breakdown(X)
    gate = breakdown["wave_envelope_gate"][:, 1:-1, :]
    base_wave = breakdown["pure_wave_force_without_envelope_gate"][:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    if gate.shape[1] < 5 or base_wave.shape[1] < 5 or target.shape[1] < 5:
        return None

    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    base_band, target_band = force_target_bandpass(base_wave, target, target_t, cfg)
    base_env = smooth_sequence_seconds(torch.abs(base_band), target_t, window_s)
    target_env = smooth_sequence_seconds(torch.abs(target_band), target_t, window_s)
    gate_env = smooth_sequence_seconds(gate, target_t, window_s)

    with torch.no_grad():
        target_level = torch.sqrt(torch.mean(target_env ** 2) + 1.0e-12)
        floor_ratio = float(cfg.get("wave_envelope_gate_target_base_floor_ratio", 0.08))
        base_floor = torch.clamp(floor_ratio * target_level, min=1.0e-4)
        base_ref = torch.clamp(base_env.detach(), min=base_floor)
        target_gate = target_env.detach() / base_ref
        target_gate = torch.clamp(
            target_gate,
            min=float(cfg.get("wave_envelope_gate_min", 0.35)),
            max=float(cfg.get("wave_envelope_gate_max", 2.20)),
        )
        reliability = base_env.detach() / (base_env.detach() + base_floor)
        reliability = torch.clamp(reliability, min=0.05, max=1.0)
    return gate_env, target_gate, reliability


def wave_envelope_gate_target_loss(model: PINNLSTM,
                                   y_phi: torch.Tensor, y_v: torch.Tensor,
                                   t_phys: torch.Tensor, X: torch.Tensor,
                                   cfg: Dict[str, object]) -> torch.Tensor:
    if float(cfg.get("lambda_wave_envelope_gate_target", 0.0)) <= 0.0:
        return torch.mean((X[..., :1] * 0.0) ** 2)
    terms = wave_envelope_gate_target_terms(model, y_phi, y_v, t_phys, X, cfg)
    if terms is None:
        return torch.mean((X[..., :1] * 0.0) ** 2)
    gate_env, target_gate, reliability = terms

    err = gate_env - target_gate
    abs_err = torch.abs(err)
    delta = float(cfg.get("wave_envelope_gate_target_huber_delta", 0.35))
    huber = torch.where(
        abs_err <= delta,
        0.5 * err ** 2,
        delta * (abs_err - 0.5 * delta),
    )
    return torch.sum(reliability * huber) / torch.clamp(torch.sum(reliability), min=1.0e-8)


def wave_envelope_gate_shape_loss(model: PINNLSTM,
                                  y_phi: torch.Tensor, y_v: torch.Tensor,
                                  t_phys: torch.Tensor, X: torch.Tensor,
                                  cfg: Dict[str, object]) -> torch.Tensor:
    if float(cfg.get("lambda_wave_envelope_gate_shape", 0.0)) <= 0.0:
        return torch.mean((X[..., :1] * 0.0) ** 2)
    terms = wave_envelope_gate_target_terms(model, y_phi, y_v, t_phys, X, cfg)
    if terms is None:
        return torch.mean((X[..., :1] * 0.0) ** 2)
    gate_env, target_gate, reliability = terms
    denom = torch.clamp(torch.sum(reliability), min=1.0e-8)
    gate_mean = torch.sum(reliability * gate_env) / denom
    with torch.no_grad():
        target_mean = torch.sum(reliability * target_gate) / denom
        target_centered = target_gate - target_mean
        target_scale = torch.sqrt(
            torch.sum(reliability * target_centered ** 2) / denom + 1.0e-12
        )
        target_shape = target_centered / torch.clamp(target_scale, min=1.0e-4)
    gate_shape = (gate_env - gate_mean) / torch.clamp(target_scale, min=1.0e-4)
    return torch.sum(reliability * (gate_shape - target_shape) ** 2) / denom


def wave_force_target_loss(model: PINNLSTM,
                           force_pred: torch.Tensor, force_residual: torch.Tensor,
                           y_phi: torch.Tensor, y_v: torch.Tensor,
                           t_phys: torch.Tensor, X: torch.Tensor,
                           cfg: Dict[str, object]) -> torch.Tensor:
    """Require the explicitly wave-driven force to explain measured dynamics."""
    wave_force = (force_pred - force_residual)[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    robust_fit = robust_force_target_fit_loss(wave_force, target, target_t, cfg)
    phase_pred, phase_target = force_target_bandpass(wave_force, target, target_t, cfg)
    phase_weight = float(cfg.get("wave_force_target_phase_weight", 0.50))
    return (
        robust_fit
        + phase_weight * phase_correlation_loss(phase_pred, phase_target)
        + force_target_amplitude_loss(wave_force, target, cfg)
        + force_target_highpass_loss(wave_force, target, target_t, cfg)
        + force_target_band_amplitude_loss(wave_force, target, target_t, cfg)
        + force_target_band_phase_loss(wave_force, target, target_t, cfg)
        + force_target_lag_penalty_loss(wave_force, target, target_t, cfg)
    )


def pure_wave_force_target_loss(model: PINNLSTM,
                                y_phi: torch.Tensor, y_v: torch.Tensor,
                                t_phys: torch.Tensor, X: torch.Tensor,
                                cfg: Dict[str, object]) -> torch.Tensor:
    """Identify measured-wave couplings without state, feedback or residual bypasses."""
    pure_wave_force = model.pure_wave_forcing(X)[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    robust_fit = robust_force_target_fit_loss(pure_wave_force, target, target_t, cfg)
    phase_pred, phase_target = force_target_bandpass(pure_wave_force, target, target_t, cfg)
    phase_weight = float(cfg.get("wave_force_target_phase_weight", 1.0))
    return (
        robust_fit
        + phase_weight * phase_correlation_loss(phase_pred, phase_target)
        + force_target_amplitude_loss(pure_wave_force, target, cfg)
        + force_target_highpass_loss(pure_wave_force, target, target_t, cfg)
        + force_target_band_amplitude_loss(pure_wave_force, target, target_t, cfg)
        + force_target_band_phase_loss(pure_wave_force, target, target_t, cfg)
        + force_target_lag_penalty_loss(pure_wave_force, target, target_t, cfg)
    )


def total_force_underforce_loss(pred: torch.Tensor, target: torch.Tensor,
                                t_phys: torch.Tensor,
                                cfg: Dict[str, object]) -> torch.Tensor:
    """Penalise missing signed force bursts more than excess force elsewhere."""
    weight = float(cfg.get("total_force_target_underforce_weight", 0.0))
    if weight <= 0.0 or pred.shape[1] < 5:
        return torch.mean((pred - target) ** 2) * 0.0
    pred_band, target_band = force_target_bandpass(pred, target, t_phys, cfg)
    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    target_env = smooth_sequence_seconds(torch.abs(target_band), t_phys, window_s)
    target_power = torch.clamp(torch.mean(target_band ** 2).detach(), min=1.0e-4)
    scale = torch.sqrt(target_power)
    with torch.no_grad():
        q = float(np.clip(float(cfg.get("total_force_event_quantile", 0.72)), 0.0, 1.0))
        threshold = torch.quantile(target_env.detach().flatten(), q)
        event_gate = torch.sigmoid(
            8.0 * (target_env.detach() - threshold) / torch.clamp(scale, min=1.0e-8)
        )
        target_sign = torch.sign(target_band.detach())
    signed_pred = pred_band * target_sign
    target_mag = torch.abs(target_band.detach())
    shortfall = torch.relu(target_mag - signed_pred) / torch.clamp(scale, min=1.0e-8)
    return weight * torch.sum(event_gate * shortfall ** 2) / torch.clamp(torch.sum(event_gate), min=1.0e-8)


def total_force_target_loss(model: PINNLSTM,
                            force_pred: torch.Tensor,
                            y_phi: torch.Tensor, y_v: torch.Tensor,
                            t_phys: torch.Tensor, X: torch.Tensor,
                            cfg: Dict[str, object]) -> torch.Tensor:
    """Align the full learned ODE force with the force implied by measured motion."""
    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    robust_fit = robust_force_target_fit_loss(total_force, target, target_t, cfg)
    phase_pred, phase_target = force_target_bandpass(total_force, target, target_t, cfg)
    phase_weight = float(cfg.get("total_force_target_phase_weight", 5.0))
    return (
        robust_fit
        + phase_weight * phase_correlation_loss(phase_pred, phase_target)
        + total_force_underforce_loss(total_force, target, target_t, cfg)
        + force_target_amplitude_loss(total_force, target, cfg)
        + force_target_highpass_loss(total_force, target, target_t, cfg)
        + force_target_band_amplitude_loss(total_force, target, target_t, cfg)
        + force_target_band_phase_loss(total_force, target, target_t, cfg)
        + force_target_lag_penalty_loss(total_force, target, target_t, cfg)
    )


def total_force_shape_loss(model: PINNLSTM,
                           force_pred: torch.Tensor,
                           y_phi: torch.Tensor, y_v: torch.Tensor,
                           t_phys: torch.Tensor, X: torch.Tensor,
                           cfg: Dict[str, object]) -> torch.Tensor:
    """Penalise local learned-force shape/overshoot against inferred force."""
    if force_pred.shape[1] < 5:
        return torch.mean((force_pred * 0.0) ** 2)
    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    pred_band, target_band = force_target_bandpass(total_force, target, target_t, cfg)
    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    target_env = smooth_sequence_seconds(torch.abs(target_band), target_t, window_s)
    target_floor = torch.clamp(
        torch.sqrt(torch.mean(target_band ** 2).detach() + 1.0e-12) * 0.15,
        min=1.0e-4,
    )
    local_scale = torch.clamp(target_env.detach(), min=target_floor)
    signed_shape = ((pred_band - target_band) / local_scale) ** 2
    overshoot = (
        torch.relu((torch.abs(pred_band) - torch.abs(target_band.detach())) / local_scale)
        ** 2
    )
    overshoot_weight = float(cfg.get("total_force_shape_overshoot_weight", 2.0))
    return torch.mean(torch.log1p(signed_shape + overshoot_weight * overshoot))


def total_force_tail_loss(model: PINNLSTM,
                          force_pred: torch.Tensor,
                          y_phi: torch.Tensor, y_v: torch.Tensor,
                          t_phys: torch.Tensor, X: torch.Tensor,
                          cfg: Dict[str, object]) -> torch.Tensor:
    """Discourage excess total-force energy above the configured roll/wave band."""
    if force_pred.shape[1] < 5:
        return torch.mean((force_pred * 0.0) ** 2)
    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    raw_period = cfg.get("force_band_amplitude_center_period_s", cfg.get("dominant_roll_period_s", 1.0))
    try:
        center_period_s = float(raw_period)
    except (TypeError, ValueError):
        center_period_s = 1.0
    if not math.isfinite(center_period_s) or center_period_s <= 0.0:
        center_period_s = 1.0
    high_window_s = max(
        center_period_s * float(cfg.get("bandpass_high_period_factor", 0.35)),
        1.0e-6,
    )
    pred_tail = total_force - smooth_sequence_seconds(total_force, target_t, high_window_s)
    target_tail = target - smooth_sequence_seconds(target, target_t, high_window_s)
    target_power = torch.clamp(torch.mean(target_tail ** 2).detach(), min=1.0e-4)
    pred_power = torch.mean(pred_tail ** 2)
    excess_ratio = torch.relu(pred_power / target_power - 1.0)
    tail_fit = normalized_huber_loss(
        pred_tail,
        target_tail.detach(),
        target_power,
        float(cfg.get("force_target_band_huber_delta", cfg.get("force_target_huber_delta", 1.25))),
    )
    return excess_ratio ** 2 + 0.25 * tail_fit


def total_force_event_loss(model: PINNLSTM,
                           force_pred: torch.Tensor,
                           y_phi: torch.Tensor, y_v: torch.Tensor,
                           t_phys: torch.Tensor, X: torch.Tensor,
                           cfg: Dict[str, object]) -> torch.Tensor:
    """Prioritise high inferred-force events when fitting total learned force."""
    if force_pred.shape[1] < 5:
        return torch.mean((force_pred * 0.0) ** 2)
    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    pred_band, target_band = force_target_bandpass(total_force, target, target_t, cfg)
    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    pred_env = smooth_sequence_seconds(torch.abs(pred_band), target_t, window_s)
    target_env = smooth_sequence_seconds(torch.abs(target_band), target_t, window_s)
    target_power = torch.clamp(torch.mean(target_band ** 2).detach(), min=1.0e-4)
    scale = torch.sqrt(target_power)

    with torch.no_grad():
        q = float(np.clip(float(cfg.get("total_force_event_quantile", 0.75)), 0.0, 1.0))
        threshold = torch.quantile(target_env.detach().flatten(), q)
        excess = torch.relu(target_env.detach() - threshold)
        event_strength = excess / torch.clamp(torch.max(excess), min=1.0e-8)
        event_weight = 1.0 + float(cfg.get("total_force_event_weight", 4.0)) * event_strength

    residual = (pred_band - target_band.detach()) / torch.clamp(scale, min=1.0e-8)
    abs_residual = torch.abs(residual)
    delta = float(cfg.get("force_target_band_huber_delta", cfg.get("force_target_huber_delta", 1.25)))
    huber = torch.where(
        abs_residual <= delta,
        0.5 * residual ** 2,
        delta * (abs_residual - 0.5 * delta),
    )
    env_residual = ((pred_env - target_env.detach()) / torch.clamp(scale, min=1.0e-8)) ** 2
    weighted_fit = torch.sum(event_weight * huber) / torch.clamp(torch.sum(event_weight), min=1.0e-8)
    weighted_env = torch.sum(event_weight * env_residual) / torch.clamp(torch.sum(event_weight), min=1.0e-8)
    return weighted_fit + 0.25 * weighted_env


def total_force_band_shape_loss(model: PINNLSTM,
                                force_pred: torch.Tensor,
                                y_phi: torch.Tensor, y_v: torch.Tensor,
                                t_phys: torch.Tensor, X: torch.Tensor,
                                cfg: Dict[str, object]) -> torch.Tensor:
    """Match total learned-force waveform shape in configured frequency bands."""
    if float(cfg.get("lambda_total_force_band_shape", 0.0)) <= 0.0 or force_pred.shape[1] < 5:
        return torch.mean((force_pred * 0.0) ** 2)
    freqs_hz = config_float_list(
        cfg,
        "total_force_band_shape_freqs_hz",
        [0.45, 1.70, 3.20],
    )
    band_weights = config_float_list(
        cfg,
        "total_force_band_shape_weights",
        [1.0, 2.0, 0.75],
    )
    if len(band_weights) < len(freqs_hz):
        band_weights = band_weights + [1.0] * (len(freqs_hz) - len(band_weights))

    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    phase_weight = float(cfg.get("total_force_band_shape_phase_weight", 0.25))
    event_weight_scale = float(cfg.get("total_force_band_shape_event_weight", 0.0))
    envelope_weight = float(cfg.get("total_force_band_shape_envelope_weight", 0.0))
    huber_delta = float(cfg.get("force_target_band_huber_delta", cfg.get("force_target_huber_delta", 1.25)))
    zero = torch.mean((total_force * 0.0) ** 2)
    total_loss = zero
    total_weight = 0.0

    for freq_hz, band_weight in zip(freqs_hz, band_weights):
        if freq_hz <= 0.0 or band_weight <= 0.0:
            continue
        band_cfg = dict(cfg)
        band_cfg["force_band_amplitude_center_period_s"] = 1.0 / freq_hz
        pred_band, target_band = force_target_bandpass(total_force, target, target_t, band_cfg)
        target_power = torch.clamp(torch.mean(target_band ** 2).detach(), min=1.0e-4)
        scale = torch.sqrt(torch.clamp(target_power, min=1.0e-8))
        residual = (pred_band - target_band.detach()) / torch.clamp(scale, min=1.0e-8)
        abs_residual = torch.abs(residual)
        huber = torch.where(
            abs_residual <= huber_delta,
            0.5 * residual ** 2,
            huber_delta * (abs_residual - 0.5 * huber_delta),
        )
        pred_env = smooth_sequence_seconds(
            torch.abs(pred_band),
            target_t,
            float(cfg.get("force_envelope_window_s", 1.2)),
        )
        target_env = smooth_sequence_seconds(
            torch.abs(target_band),
            target_t,
            float(cfg.get("force_envelope_window_s", 1.2)),
        )
        if event_weight_scale > 0.0:
            with torch.no_grad():
                q = float(np.clip(float(cfg.get("total_force_band_shape_event_quantile", cfg.get("total_force_event_quantile", 0.75))), 0.0, 1.0))
                threshold = torch.quantile(target_env.detach().flatten(), q)
                excess = torch.relu(target_env.detach() - threshold)
                event_strength = excess / torch.clamp(torch.max(excess), min=1.0e-8)
                sample_weight = 1.0 + event_weight_scale * event_strength
            waveform_loss = torch.sum(sample_weight * huber) / torch.clamp(torch.sum(sample_weight), min=1.0e-8)
        else:
            waveform_loss = torch.mean(huber)
        band_loss = waveform_loss
        if envelope_weight > 0.0:
            env_residual = ((pred_env - target_env.detach()) / torch.clamp(scale, min=1.0e-8)) ** 2
            if event_weight_scale > 0.0:
                envelope_loss = torch.sum(sample_weight * env_residual) / torch.clamp(torch.sum(sample_weight), min=1.0e-8)
            else:
                envelope_loss = torch.mean(env_residual)
            band_loss = band_loss + envelope_weight * envelope_loss
        if phase_weight > 0.0:
            band_loss = band_loss + phase_weight * phase_correlation_loss(
                pred_band,
                target_band.detach(),
            )
        total_loss = total_loss + float(band_weight) * band_loss
        total_weight += float(band_weight)

    if total_weight <= 0.0:
        return zero
    return total_loss / total_weight


def total_force_spectral_shape_loss(model: PINNLSTM,
                                    force_pred: torch.Tensor,
                                    y_phi: torch.Tensor, y_v: torch.Tensor,
                                    t_phys: torch.Tensor, X: torch.Tensor,
                                    cfg: Dict[str, object]) -> torch.Tensor:
    """Match total learned-force spectrum in targeted diagnostic bands."""
    if (
        float(cfg.get("lambda_total_force_spectral_shape", 0.0)) <= 0.0
        or force_pred.shape[1] < 8
    ):
        return torch.mean((force_pred * 0.0) ** 2)

    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    n = int(total_force.shape[1])
    if n < 8:
        return torch.mean((total_force * 0.0) ** 2)

    lows = config_float_list(
        cfg,
        "total_force_spectral_shape_band_lows_hz",
        [0.02, 1.45, 2.90],
    )
    highs = config_float_list(
        cfg,
        "total_force_spectral_shape_band_highs_hz",
        [0.55, 1.85, 3.40],
    )
    weights = config_float_list(
        cfg,
        "total_force_spectral_shape_weights",
        [1.50, 6.0, 3.0],
    )
    excess_only = config_float_list(
        cfg,
        "total_force_spectral_shape_excess_only",
        [1.0, 0.0, 0.0],
    )
    n_bands = min(len(lows), len(highs))
    if len(weights) < n_bands:
        weights = weights + [1.0] * (n_bands - len(weights))
    if len(excess_only) < n_bands:
        excess_only = excess_only + [0.0] * (n_bands - len(excess_only))

    with torch.no_grad():
        dt = torch.diff(target_t, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item())
        if not math.isfinite(dt_med) or dt_med <= 0.0:
            dt_med = 1.0
    freqs = torch.fft.rfftfreq(n, d=dt_med, device=total_force.device)
    window = torch.hann_window(
        n,
        periodic=False,
        dtype=total_force.dtype,
        device=total_force.device,
    ).view(1, n)
    norm = torch.clamp(torch.sum(window), min=1.0e-8)

    zero = torch.mean((total_force * 0.0) ** 2)
    total_loss = zero
    total_weight = 0.0
    underfit_weight = float(cfg.get("total_force_spectral_shape_underfit_weight", 1.5))
    complex_weight = float(cfg.get("total_force_spectral_shape_complex_weight", 0.20))
    low_margin = max(1.0, float(cfg.get("total_force_spectral_shape_low_excess_margin", 1.02)))

    def spectrum(values: torch.Tensor, *, demean: bool) -> torch.Tensor:
        x = values[..., 0]
        if demean:
            x = x - torch.mean(x, dim=1, keepdim=True)
        return torch.fft.rfft(x * window, dim=1) / norm

    pred_raw_spec = spectrum(total_force, demean=False)
    target_raw_spec = spectrum(target, demean=False).detach()
    pred_demeaned_spec = spectrum(total_force, demean=True)
    target_demeaned_spec = spectrum(target, demean=True).detach()

    for idx in range(n_bands):
        low_hz = float(lows[idx])
        high_hz = float(highs[idx])
        band_weight = float(weights[idx])
        if high_hz <= low_hz or band_weight <= 0.0:
            continue
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        if int(torch.sum(band_mask).detach().cpu().item()) <= 0:
            continue

        is_excess_only = float(excess_only[idx]) > 0.5
        pred_spec = pred_raw_spec[:, band_mask] if is_excess_only else pred_demeaned_spec[:, band_mask]
        target_spec = target_raw_spec[:, band_mask] if is_excess_only else target_demeaned_spec[:, band_mask]
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec).detach()
        target_power = torch.clamp(torch.mean(target_mag ** 2).detach(), min=1.0e-8)

        if is_excess_only:
            pred_power = torch.mean(pred_mag ** 2)
            band_loss = torch.relu(pred_power / (target_power * low_margin) - 1.0) ** 2
        else:
            mag_floor = torch.clamp(torch.sqrt(target_power) * 0.02, min=1.0e-6)
            pred_safe = torch.clamp(pred_mag, min=mag_floor)
            target_safe = torch.clamp(target_mag, min=mag_floor)
            with torch.no_grad():
                bin_weight = target_mag / torch.clamp(
                    torch.mean(target_mag, dim=1, keepdim=True),
                    min=mag_floor,
                )
                bin_weight = torch.clamp(bin_weight, min=0.25, max=6.0)
            log_error = torch.log(pred_safe) - torch.log(target_safe)
            amp_loss = torch.sum(bin_weight * log_error ** 2) / torch.clamp(
                torch.sum(bin_weight),
                min=1.0e-8,
            )
            under_loss = torch.sum(
                bin_weight * torch.relu(torch.log(target_safe) - torch.log(pred_safe)) ** 2
            ) / torch.clamp(torch.sum(bin_weight), min=1.0e-8)
            complex_loss = torch.mean(torch.abs(pred_spec - target_spec) ** 2) / target_power
            band_loss = amp_loss + underfit_weight * under_loss + complex_weight * complex_loss

        total_loss = total_loss + band_weight * band_loss
        total_weight += band_weight

    if total_weight <= 0.0:
        return zero
    return total_loss / total_weight


def raw_orientation_effect_gain_tensor(model: PINNLSTM,
                                       X: torch.Tensor) -> torch.Tensor:
    """Return the geometry-derived encounter gain without applying hard gating."""
    idxs = model.state_feature_indices.get("wave_orientation_effect_gain", [])
    if len(idxs) == 0:
        return 1.0 + 0.0 * X[..., :1]
    gain = torch.mean(X[..., idxs], dim=-1, keepdim=True)
    gain = torch.where(torch.isfinite(gain), gain, torch.ones_like(gain))
    return torch.clamp(gain, min=0.0, max=1.0)


def weighted_masked_huber(pred: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor, scale: torch.Tensor,
                          delta: float) -> torch.Tensor:
    residual = (pred - target.detach()) / torch.clamp(scale, min=1.0e-8)
    abs_residual = torch.abs(residual)
    huber = torch.where(
        abs_residual <= delta,
        0.5 * residual ** 2,
        delta * (abs_residual - 0.5 * delta),
    )
    denom = torch.clamp(torch.sum(mask), min=1.0e-8)
    return torch.sum(mask * huber) / denom


def total_force_regime_balance_loss(model: PINNLSTM,
                                    force_pred: torch.Tensor,
                                    y_phi: torch.Tensor, y_v: torch.Tensor,
                                    t_phys: torch.Tensor, X: torch.Tensor,
                                    cfg: Dict[str, object]) -> torch.Tensor:
    """Match total inferred force separately by vessel/wave encounter regime."""
    if float(cfg.get("lambda_total_force_regime_balance", 0.0)) <= 0.0 or force_pred.shape[1] < 5:
        return torch.mean((force_pred * 0.0) ** 2)

    total_force = force_pred[:, 1:-1, :]
    target = measured_wave_force_target(model, y_phi, y_v, t_phys, X, cfg)
    target_t = t_phys[:, 1:-1, :]
    pred_band, target_band = force_target_bandpass(total_force, target, target_t, cfg)
    orientation_gain = raw_orientation_effect_gain_tensor(model, X)[:, 1:-1, :]

    parallel_max = float(np.clip(float(cfg.get("force_regime_parallel_max_gain", 0.45)), 0.0, 1.0))
    side_min = float(np.clip(float(cfg.get("force_regime_side_min_gain", 0.75)), parallel_max, 1.0))
    regime_specs = (
        (orientation_gain <= parallel_max, float(cfg.get("force_regime_parallel_weight", 1.0))),
        ((orientation_gain > parallel_max) & (orientation_gain < side_min), float(cfg.get("force_regime_oblique_weight", 1.0))),
        (orientation_gain >= side_min, float(cfg.get("force_regime_side_weight", 2.0))),
    )
    huber_delta = float(cfg.get("force_target_band_huber_delta", cfg.get("force_target_huber_delta", 1.25)))
    envelope_weight = float(cfg.get("force_regime_envelope_weight", 0.25))
    zero = torch.mean((total_force * 0.0) ** 2)
    total_loss = zero
    total_weight = 0.0

    pred_env = None
    target_env = None
    if envelope_weight > 0.0:
        pred_env = smooth_sequence_seconds(
            torch.abs(pred_band),
            target_t,
            float(cfg.get("force_envelope_window_s", 1.2)),
        )
        target_env = smooth_sequence_seconds(
            torch.abs(target_band),
            target_t,
            float(cfg.get("force_envelope_window_s", 1.2)),
        )

    for regime_mask_bool, regime_weight in regime_specs:
        if regime_weight <= 0.0:
            continue
        mask = regime_mask_bool.to(dtype=pred_band.dtype)
        count = torch.sum(mask)
        if float(count.detach().cpu().item()) < 3.0:
            continue
        target_power = torch.clamp(
            torch.sum(mask * target_band.detach() ** 2) / torch.clamp(count, min=1.0),
            min=1.0e-4,
        )
        scale = torch.sqrt(target_power)
        regime_loss = weighted_masked_huber(
            pred_band,
            target_band,
            mask,
            scale,
            huber_delta,
        )
        if envelope_weight > 0.0 and pred_env is not None and target_env is not None:
            env_residual = ((pred_env - target_env.detach()) / torch.clamp(scale, min=1.0e-8)) ** 2
            env_loss = torch.sum(mask * env_residual) / torch.clamp(torch.sum(mask), min=1.0e-8)
            regime_loss = regime_loss + envelope_weight * env_loss
        total_loss = total_loss + regime_weight * regime_loss
        total_weight += regime_weight

    if total_weight <= 0.0:
        return zero
    return total_loss / total_weight


def phase_correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - torch.mean(pred, dim=1, keepdim=True)
    target_c = target - torch.mean(target, dim=1, keepdim=True)
    numerator = torch.sum(pred_c * target_c, dim=1)
    denom = torch.sqrt(torch.sum(pred_c ** 2, dim=1) * torch.sum(target_c ** 2, dim=1) + 1.0e-12)
    corr = numerator / torch.clamp(denom, min=1.0e-8)
    return torch.mean(1.0 - corr)


def reversal_shape_loss(pred_v: torch.Tensor, target_v: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    scale = float(max(1.0e-6, cfg.get("reversal_tanh_scale", 4.0)))
    sign_shape = torch.mean((torch.tanh(scale * pred_v) - torch.tanh(scale * target_v)) ** 2)
    return sign_shape + 0.25 * phase_correlation_loss(pred_v, target_v)


def turning_point_loss(pred_v: torch.Tensor, target_v: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    if pred_v.shape[1] < 3:
        return torch.mean((pred_v - target_v) ** 2) * 0.0
    target_prod = target_v[:, :-1, :] * target_v[:, 1:, :]
    crossing = (target_prod <= 0.0).float()
    neighbourhood = int(max(0, cfg.get("turning_point_neighbourhood", 2)))
    if neighbourhood > 0:
        crossing = nn.functional.max_pool1d(
            crossing.transpose(1, 2),
            kernel_size=2 * neighbourhood + 1,
            stride=1,
            padding=neighbourhood,
        ).transpose(1, 2)
    denom = torch.sum(crossing)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.mean((pred_v - target_v) ** 2) * 0.0
    pred_prod = pred_v[:, :-1, :] * pred_v[:, 1:, :]
    missed_crossing = torch.relu(pred_prod)
    local_speed = 0.5 * (pred_v[:, :-1, :] ** 2 + pred_v[:, 1:, :] ** 2)
    return torch.sum(crossing * (missed_crossing + local_speed)) / torch.clamp(denom, min=1.0e-8)


def _set_tensor_motion_feedback(X_work: torch.Tensor, idx: int, start_idx: int, delay_steps: List[int],
                                pred_phi_steps: List[torch.Tensor], pred_v_steps: List[torch.Tensor],
                                y_phi: torch.Tensor, y_v: torch.Tensor,
                                model: PINNLSTM) -> None:
    phi_idxs = model.motion_feature_indices.get("phi_fb", [])
    v_idxs = model.motion_feature_indices.get("v_fb", [])
    if len(phi_idxs) == 0 and len(v_idxs) == 0:
        return
    if len(delay_steps) < max(len(phi_idxs), len(v_idxs)):
        delay_steps = delay_steps + [delay_steps[-1] if delay_steps else 0] * (max(len(phi_idxs), len(v_idxs)) - len(delay_steps))
    for i, col in enumerate(phi_idxs):
        ref = max(0, int(idx) - int(delay_steps[i]))
        if ref >= start_idx:
            phi_ref = pred_phi_steps[min(ref - start_idx, len(pred_phi_steps) - 1)]
        else:
            phi_ref = y_phi[:, ref:ref + 1, :]
        X_work[:, idx:idx + 1, int(col):int(col) + 1] = phi_ref
    for i, col in enumerate(v_idxs):
        ref = max(0, int(idx) - int(delay_steps[i]))
        if ref >= start_idx:
            v_ref = pred_v_steps[min(ref - start_idx, len(pred_v_steps) - 1)]
        else:
            v_ref = y_v[:, ref:ref + 1, :]
        X_work[:, idx:idx + 1, int(col):int(col) + 1] = v_ref


def recursive_rollout_losses(model: PINNLSTM, force_pred: torch.Tensor,
                             X: torch.Tensor, y_phi: torch.Tensor, y_v: torch.Tensor,
                             t_phys: torch.Tensor, cfg: Dict[str, object]) -> Dict[str, torch.Tensor]:
    zero = torch.mean((force_pred * 0.0) ** 2)
    T = int(y_phi.shape[1])
    if T < 3:
        return {
            "rollout": zero,
            "rollout_rate": zero,
            "rollout_amplitude": zero,
            "rollout_phase": zero,
            "rollout_peak_trough": zero,
            "rollout_global_extrema": zero,
            "rollout_extrema_window_underfit": zero,
            "rollout_high_amplitude_underfit": zero,
            "reversal": zero,
            "turning_point": zero,
            "rollout_local_prominence": zero,
            "rollout_high_pass_residual": zero,
            "rollout_roll_spectral_shape": zero,
            "rollout_roll_curvature": zero,
        }

    feedback_active = bool(
        model.motion_feature_indices.get("phi_fb", [])
        or model.motion_feature_indices.get("v_fb", [])
    )
    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item()) if dt.numel() else 1.0
        rollout_window_s = effective_rollout_window_s(cfg, feedback_active)
        requested = int(round(rollout_window_s / max(dt_med, 1.0e-8)))
        min_steps = int(max(2, cfg.get("rollout_min_steps", 8)))
        rollout_steps = min(T - 1, max(min_steps, requested))
    if rollout_steps < 2:
        return {
            "rollout": zero,
            "rollout_rate": zero,
            "rollout_amplitude": zero,
            "rollout_phase": zero,
            "rollout_peak_trough": zero,
            "rollout_global_extrema": zero,
            "rollout_extrema_window_underfit": zero,
            "rollout_high_amplitude_underfit": zero,
            "rollout_local_prominence": zero,
            "rollout_high_pass_residual": zero,
            "rollout_roll_spectral_shape": zero,
            "reversal": zero,
            "turning_point": zero,
            "rollout_roll_curvature": zero,
        }

    start = max(0, T - rollout_steps - 1)
    seq_len = int(cfg.get("seq_len", T))
    delay_steps = motion_feedback_delay_steps(cfg, dt_med)
    X_work = X.clone() if feedback_active else X
    # With exogenous-only inputs, the causal full-window forward pass already
    # gives the force available at every time step. Reuse it so a genuine 7 s
    # dynamics rollout needs one LSTM pass rather than ~175 passes.
    causal_force = force_pred + model.turn_moment(X) if not feedback_active else None
    phi_j = y_phi[:, start:start + 1, :]
    v_j = y_v[:, start:start + 1, :]
    phi_seq = [phi_j]
    v_seq = [v_j]
    for j in range(start, T - 1):
        if feedback_active:
            X_step = X_work.clone()
            _set_tensor_motion_feedback(X_step, j, start, delay_steps, phi_seq, v_seq, y_phi, y_v, model)
            s = max(0, j - seq_len + 1)
            window = X_step[:, s:j + 1, :]
            _, _, force_without_turn, _ = model(window, window)
            force_j = force_without_turn[:, -1:, :] + model.turn_moment(window)[:, -1:, :]
        else:
            X_step = X_work
            assert causal_force is not None
            force_j = causal_force[:, j:j + 1, :]
        dt_j = torch.clamp(t_phys[:, j + 1:j + 2, :] - t_phys[:, j:j + 1, :], min=1.0e-8)
        accel_j = force_j - model.c_roll * v_j - model.c_quad * torch.abs(v_j) * v_j - model.k_roll * phi_j
        v_j = v_j + dt_j * accel_j
        phi_j = phi_j + dt_j * v_j
        v_seq.append(v_j)
        phi_seq.append(phi_j)
        if feedback_active:
            X_work = X_step.clone()
            _set_tensor_motion_feedback(X_work, j + 1, start, delay_steps, phi_seq, v_seq, y_phi, y_v, model)

    pred_phi = torch.cat(phi_seq, dim=1)
    pred_v = torch.cat(v_seq, dim=1)
    target_phi = y_phi[:, start:T, :]
    target_v = y_v[:, start:T, :]
    target_t = t_phys[:, start:T, :]
    return {
        "rollout": nn.functional.mse_loss(pred_phi, target_phi),
        "rollout_rate": nn.functional.mse_loss(pred_v, target_v),
        "rollout_amplitude": amplitude_underfit_loss(pred_phi, target_phi, target_t, cfg),
        "rollout_phase": phase_correlation_loss(pred_phi, target_phi),
        "rollout_peak_trough": peak_trough_loss(pred_phi, target_phi, cfg),
        "rollout_global_extrema": global_extrema_loss(pred_phi, target_phi, target_t, cfg),
        "rollout_extrema_window_underfit": extrema_window_underfit_loss(pred_phi, target_phi, cfg),
        "rollout_high_amplitude_underfit": high_amplitude_underfit_loss(pred_phi, target_phi, cfg),
        "reversal": reversal_shape_loss(pred_v, target_v, cfg),
        "turning_point": turning_point_loss(pred_v, target_v, cfg),
        "rollout_local_prominence": local_prominence_extrema_loss(pred_phi, target_phi, target_t, cfg),
        "rollout_high_pass_residual": high_pass_residual_loss(pred_phi, target_phi, target_t, cfg),
        "rollout_roll_spectral_shape": roll_spectral_shape_loss(pred_phi, target_phi, target_t, cfg),
        "rollout_roll_curvature": roll_curvature_loss(pred_phi, target_phi, target_t, cfg),
    }


def direct_lstm_rollout_losses(model: PINNLSTM,
                                      phi_pred: torch.Tensor,
                                      v_pred: torch.Tensor,
                                      X: torch.Tensor,
                                      y_phi: torch.Tensor,
                                      y_v: torch.Tensor,
                                      t_phys: torch.Tensor,
                                      cfg: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """Train the no-ODE forecast path on the tail of each sequence."""
    zero = torch.mean((phi_pred * 0.0) ** 2)
    T = int(y_phi.shape[1])
    keys = (
        "direct_forecast",
        "direct_forecast_rate",
        "direct_forecast_amplitude",
        "direct_forecast_phase",
        "direct_forecast_peak_trough",
        "direct_forecast_global_extrema",
        "direct_forecast_extrema_window_underfit",
        "direct_forecast_high_amplitude_underfit",
        "direct_forecast_asymmetric_extrema",
        "direct_forecast_local_prominence",
        "direct_forecast_high_pass_residual",
        "direct_forecast_roll_spectral_shape",
        "direct_forecast_roll_curvature",
    )
    if T < 3:
        return {key: zero for key in keys}

    feedback_active = bool(
        model.motion_feature_indices.get("phi_fb", [])
        or model.motion_feature_indices.get("v_fb", [])
    )
    with torch.no_grad():
        dt = torch.diff(t_phys, dim=1)
        dt_med = float(torch.median(torch.clamp(dt, min=1.0e-8)).detach().cpu().item()) if dt.numel() else 1.0
        horizons_s = config_float_list(
            cfg,
            "direct_forecast_loss_horizons_s",
            [float(cfg.get("direct_forecast_loss_window_s", 5.08))],
        )
        horizon_weights = config_float_list(
            cfg,
            "direct_forecast_loss_horizon_weights",
            [1.0] * len(horizons_s),
        )
        if len(horizon_weights) < len(horizons_s):
            horizon_weights.extend([horizon_weights[-1] if horizon_weights else 1.0] * (len(horizons_s) - len(horizon_weights)))
        horizon_weights = horizon_weights[:len(horizons_s)]
        min_steps = int(max(2, cfg.get("direct_forecast_loss_min_steps", 8)))
        tail_fraction = float(np.clip(float(cfg.get("direct_forecast_tail_fraction", 0.0)), 0.0, 1.0))
        tail_extra = max(0.0, float(cfg.get("direct_forecast_tail_multiplier", 1.0)) - 1.0)
        horizon_specs = []
        for horizon_s, raw_weight in zip(horizons_s, horizon_weights):
            weight = float(max(0.0, raw_weight))
            if weight <= 0.0:
                continue
            requested = int(round(float(max(0.0, horizon_s)) / max(dt_med, 1.0e-8)))
            forecast_steps = min(T - 1, max(min_steps, requested))
            if forecast_steps >= 2:
                horizon_specs.append((forecast_steps, weight))
    if not horizon_specs:
        return {key: zero for key in keys}

    seq_len = int(cfg.get("seq_len", T))
    delay_steps = motion_feedback_delay_steps(cfg, dt_med) if feedback_active else []
    losses = {key: zero for key in keys}
    total_weight = 0.0
    target_task_weight = float(max(0.0, cfg.get("direct_forecast_turning_point_weight", 0.0)))

    def segment_losses(
        start: int,
        end: int,
        X_seg: torch.Tensor,
        phi_pred_seg: torch.Tensor,
        v_pred_seg: torch.Tensor,
        y_phi_seg: torch.Tensor,
        y_v_seg: torch.Tensor,
        t_seg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        start_i = int(max(0, min(start, T - 2)))
        end_i = int(max(start_i + 2, min(end, T)))
        target_phi = y_phi_seg[:, start_i:end_i, :]
        target_v = y_v_seg[:, start_i:end_i, :]
        target_t = t_seg[:, start_i:end_i, :]
        if not feedback_active:
            pred_phi = torch.cat(
                [y_phi_seg[:, start_i:start_i + 1, :], phi_pred_seg[:, start_i + 1:end_i, :]],
                dim=1,
            )
            pred_v = torch.cat(
                [y_v_seg[:, start_i:start_i + 1, :], v_pred_seg[:, start_i + 1:end_i, :]],
                dim=1,
            )
        else:
            X_work = X_seg.clone()
            phi_seq = [y_phi_seg[:, start_i:start_i + 1, :]]
            v_seq = [y_v_seg[:, start_i:start_i + 1, :]]
            for j in range(start_i + 1, end_i):
                _set_tensor_motion_feedback(X_work, j, start_i, delay_steps, phi_seq, v_seq, y_phi_seg, y_v_seg, model)
                s = max(0, j - seq_len + 1)
                window = X_work[:, s:j + 1, :]
                phi_j, v_j, _, _ = model(window, window)
                phi_seq.append(phi_j[:, -1:, :])
                v_seq.append(v_j[:, -1:, :])
            pred_phi = torch.cat(phi_seq, dim=1)
            pred_v = torch.cat(v_seq, dim=1)

        return {
            "direct_forecast": nn.functional.mse_loss(pred_phi, target_phi),
            "direct_forecast_rate": nn.functional.mse_loss(pred_v, target_v),
            "direct_forecast_amplitude": amplitude_underfit_loss(pred_phi, target_phi, target_t, cfg),
            "direct_forecast_phase": phase_correlation_loss(pred_phi, target_phi),
            "direct_forecast_peak_trough": peak_trough_loss(pred_phi, target_phi, cfg),
            "direct_forecast_global_extrema": global_extrema_loss(pred_phi, target_phi, target_t, cfg),
            "direct_forecast_extrema_window_underfit": extrema_window_underfit_loss(pred_phi, target_phi, cfg),
            "direct_forecast_high_amplitude_underfit": high_amplitude_underfit_loss(pred_phi, target_phi, cfg),
            "direct_forecast_asymmetric_extrema": asymmetric_extrema_forecast_loss(pred_phi, target_phi, cfg),
            "direct_forecast_local_prominence": local_prominence_extrema_loss(pred_phi, target_phi, target_t, cfg),
            "direct_forecast_high_pass_residual": high_pass_residual_loss(pred_phi, target_phi, target_t, cfg),
            "direct_forecast_roll_spectral_shape": roll_spectral_shape_loss(pred_phi, target_phi, target_t, cfg),
            "direct_forecast_roll_curvature": roll_curvature_loss(pred_phi, target_phi, target_t, cfg),
        }

    def add_losses(weight: float, values: Dict[str, torch.Tensor], denom_weight: Optional[float] = None) -> None:
        nonlocal total_weight
        for key, value in values.items():
            losses[key] = losses[key] + float(weight) * value
        total_weight += float(weight if denom_weight is None else denom_weight)

    def turning_point_task_starts(forecast_steps: int) -> List[Tuple[int, int]]:
        if (
            target_task_weight <= 0.0
            or not bool(cfg.get("direct_forecast_turning_point_sampling_enabled", False))
            or forecast_steps < 2
        ):
            return []
        max_tasks = int(max(0, cfg.get("direct_forecast_turning_point_max_tasks", 0)))
        if max_tasks <= 0:
            return []
        fracs = config_float_list(
            cfg,
            "direct_forecast_turning_point_window_fracs",
            config_float_list(cfg, "turning_point_sampling_window_fracs", [0.5]),
        )
        fracs = [float(np.clip(v, 0.0, 1.0)) for v in fracs] or [0.5]
        neighbourhood = int(max(1, cfg.get("direct_forecast_turning_point_neighbourhood", cfg.get("turning_point_sampling_neighbourhood", 4))))
        min_prom = math.radians(float(max(0.0, cfg.get(
            "direct_forecast_turning_point_min_prominence_deg",
            cfg.get("turning_point_sampling_min_prominence_deg", 0.0),
        ))))
        phi_np = y_phi.detach().cpu().numpy()[..., 0]
        v_np = y_v.detach().cpu().numpy()[..., 0]
        tasks: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for row in range(int(phi_np.shape[0])):
            products = v_np[row, :-1] * v_np[row, 1:]
            candidates = np.flatnonzero(products <= 0.0).astype(int) + 1
            for idx in candidates:
                lo = int(max(0, idx - neighbourhood))
                hi = int(min(T - 1, idx + neighbourhood))
                if hi <= lo:
                    continue
                left = phi_np[row, lo:idx + 1]
                right = phi_np[row, idx:hi + 1]
                if v_np[row, max(0, idx - 1)] > 0.0:
                    prominence = min(float(phi_np[row, idx] - np.min(left)), float(phi_np[row, idx] - np.min(right)))
                else:
                    prominence = min(float(np.max(left) - phi_np[row, idx]), float(np.max(right) - phi_np[row, idx]))
                if not math.isfinite(prominence) or prominence < min_prom:
                    continue
                for frac in fracs:
                    start = int(round(float(idx) - frac * float(forecast_steps)))
                    start = int(np.clip(start, 0, max(0, T - forecast_steps - 1)))
                    end = int(min(T, start + forecast_steps + 1))
                    if end - start < 3:
                        continue
                    task = (int(row), int(start))
                    if task not in seen:
                        seen.add(task)
                        tasks.append(task)
                    if len(tasks) >= max_tasks:
                        return tasks
        return tasks

    for forecast_steps, weight in horizon_specs:
        start = max(0, T - forecast_steps - 1)
        horizon_losses = segment_losses(start, T, X, phi_pred, v_pred, y_phi, y_v, t_phys)
        if tail_fraction > 0.0 and tail_extra > 0.0:
            horizon_len = int(T - start)
            tail_steps = int(math.ceil(float(horizon_len) * tail_fraction))
            tail_steps = int(min(horizon_len, max(2, tail_steps)))
            if tail_steps >= 2:
                tail_start = int(max(start, T - tail_steps))
                tail_losses = segment_losses(tail_start, T, X, phi_pred, v_pred, y_phi, y_v, t_phys)
                for key, value in tail_losses.items():
                    horizon_losses[key] = horizon_losses[key] + float(tail_extra) * value
        add_losses(float(weight), horizon_losses, float(weight) * (1.0 + float(tail_extra)))
        for row, task_start in turning_point_task_starts(forecast_steps):
            end = int(min(T, int(task_start) + int(forecast_steps) + 1))
            task_losses = segment_losses(
                int(task_start),
                end,
                X[row:row + 1],
                phi_pred[row:row + 1],
                v_pred[row:row + 1],
                y_phi[row:row + 1],
                y_v[row:row + 1],
                t_phys[row:row + 1],
            )
            add_losses(float(weight) * target_task_weight, task_losses)

    return {key: value / max(total_weight, 1.0e-8) for key, value in losses.items()}


def compute_loss(model: PINNLSTM, batch: Dict[str, torch.Tensor],
                 cfg: Dict[str, object], rollout_enabled: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    src = batch["src"]
    tgt = batch["tgt"]
    t_phys = batch["t"]
    y_phi = batch["y_phi"]
    y_v = batch["y_v"]
    phi_pred, v_pred, force_pred, force_residual = model(src, tgt)
    pinn_enabled = bool(cfg.get("pinn_enabled", True))
    zero_aux = torch.mean((phi_pred * 0.0) ** 2)
    l_data = mse_with_peak_weight(phi_pred, y_phi, cfg)
    l_r2_data = r2_data_loss(phi_pred, y_phi)
    l_rate = nn.functional.mse_loss(v_pred, y_v)
    l_peak_trough = peak_trough_loss(phi_pred, y_phi, cfg)
    l_global_extrema = global_extrema_loss(phi_pred, y_phi, t_phys, cfg)
    l_amp_underfit = amplitude_underfit_loss(phi_pred, y_phi, t_phys, cfg)
    l_local_prominence = local_prominence_extrema_loss(phi_pred, y_phi, t_phys, cfg)
    l_extrema_window_underfit = extrema_window_underfit_loss(phi_pred, y_phi, cfg)
    l_high_amplitude_underfit = high_amplitude_underfit_loss(phi_pred, y_phi, cfg)
    l_high_pass_residual = high_pass_residual_loss(phi_pred, y_phi, t_phys, cfg)
    l_low_pass_residual = low_pass_residual_loss(phi_pred, y_phi, t_phys, cfg)
    l_roll_spectral_shape = roll_spectral_shape_loss(phi_pred, y_phi, t_phys, cfg)
    l_roll_curvature = roll_curvature_loss(phi_pred, y_phi, t_phys, cfg)
    l_roll_slope = roll_slope_data_loss(phi_pred, y_v, t_phys, cfg)
    if pinn_enabled:
        l_kin = kinematic_loss(phi_pred, v_pred, t_phys, y_v, cfg)
        l_phys = physics_loss(model, phi_pred, v_pred, force_pred, t_phys, tgt)
        l_bc = boundary_loss(phi_pred, v_pred, y_phi, y_v)
        l_force = torch.mean(force_pred ** 2)
        l_force_smooth = force_smoothness_loss(force_pred, t_phys)
        l_force_mean = force_mean_loss(force_pred)
        l_force_residual = torch.mean(force_residual ** 2)
        l_wave_envelope_gate = wave_envelope_gate_regularization_loss(model, tgt, cfg)
        l_wave_envelope_gate_target = wave_envelope_gate_target_loss(
            model,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_wave_envelope_gate_shape = wave_envelope_gate_shape_loss(
            model,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_wave_force_target = wave_force_target_loss(
            model,
            force_pred,
            force_residual,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_pure_wave_force_target = pure_wave_force_target_loss(
            model,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_target = total_force_target_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_shape = total_force_shape_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_tail = total_force_tail_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_event = total_force_event_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_regime_balance = total_force_regime_balance_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_band_shape = total_force_band_shape_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_total_force_spectral_shape = total_force_spectral_shape_loss(
            model,
            force_pred,
            y_phi,
            y_v,
            t_phys,
            tgt,
            cfg,
        )
        l_force_envelope = force_target_envelope_loss(
            model.pure_wave_forcing(tgt)[:, 1:-1, :],
            measured_wave_force_target(model, y_phi, y_v, t_phys, tgt, cfg),
            t_phys[:, 1:-1, :],
            cfg,
        )
        rollout_weights = (
            float(cfg.get("lambda_rollout", 0.0)),
            float(cfg.get("lambda_rollout_rate", 0.0)),
            float(cfg.get("lambda_rollout_amplitude", 0.0)),
            float(cfg.get("lambda_rollout_phase", 0.0)),
            float(cfg.get("lambda_rollout_peak_trough", 0.0)),
            float(cfg.get("lambda_rollout_global_extrema", 0.0)),
            float(cfg.get("lambda_rollout_extrema_window_underfit", 0.0)),
            float(cfg.get("lambda_rollout_high_amplitude_underfit", 0.0)),
            float(cfg.get("lambda_reversal", 0.0)),
            float(cfg.get("lambda_turning_point", 0.0)),
            float(cfg.get("lambda_rollout_local_prominence", 0.0)),
            float(cfg.get("lambda_rollout_high_pass_residual", 0.0)),
            float(cfg.get("lambda_rollout_roll_spectral_shape", 0.0)),
            float(cfg.get("lambda_rollout_roll_curvature", 0.0)),
        )
    else:
        l_kin = zero_aux
        l_phys = zero_aux
        l_bc = zero_aux
        l_force = zero_aux
        l_force_smooth = zero_aux
        l_force_mean = zero_aux
        l_force_residual = zero_aux
        l_wave_envelope_gate = zero_aux
        l_wave_envelope_gate_target = zero_aux
        l_wave_envelope_gate_shape = zero_aux
        l_wave_force_target = zero_aux
        l_pure_wave_force_target = zero_aux
        l_total_force_target = zero_aux
        l_total_force_shape = zero_aux
        l_total_force_tail = zero_aux
        l_total_force_event = zero_aux
        l_total_force_regime_balance = zero_aux
        l_total_force_band_shape = zero_aux
        l_total_force_spectral_shape = zero_aux
        l_force_envelope = zero_aux
        rollout_weights = (0.0,) * 14
    direct_weights = (
        float(cfg.get("lambda_direct_forecast", 0.0)),
        float(cfg.get("lambda_direct_forecast_rate", 0.0)),
        float(cfg.get("lambda_direct_forecast_amplitude", 0.0)),
        float(cfg.get("lambda_direct_forecast_phase", 0.0)),
        float(cfg.get("lambda_direct_forecast_peak_trough", 0.0)),
        float(cfg.get("lambda_direct_forecast_global_extrema", 0.0)),
        float(cfg.get("lambda_direct_forecast_extrema_window_underfit", 0.0)),
        float(cfg.get("lambda_direct_forecast_high_amplitude_underfit", 0.0)),
        float(cfg.get("lambda_direct_forecast_asymmetric_extrema", 0.0)),
        float(cfg.get("lambda_direct_forecast_local_prominence", 0.0)),
        float(cfg.get("lambda_direct_forecast_high_pass_residual", 0.0)),
        float(cfg.get("lambda_direct_forecast_roll_spectral_shape", 0.0)),
        float(cfg.get("lambda_direct_forecast_roll_curvature", 0.0)),
    )
    feedback_active = bool(
        model.motion_feature_indices.get("phi_fb", [])
        or model.motion_feature_indices.get("v_fb", [])
    )
    if rollout_enabled and any(w > 0.0 for w in rollout_weights):
        rollout_batch = effective_rollout_batch_size(
            cfg, int(y_phi.shape[0]), feedback_active
        )
        rollout = recursive_rollout_losses(
            model,
            force_pred[:rollout_batch],
            tgt[:rollout_batch],
            y_phi[:rollout_batch],
            y_v[:rollout_batch],
            t_phys[:rollout_batch],
            cfg,
        )
    else:
        zero_rollout = zero_aux if not pinn_enabled else torch.mean((force_pred * 0.0) ** 2)
        rollout = {
            "rollout": zero_rollout,
            "rollout_rate": zero_rollout,
            "rollout_amplitude": zero_rollout,
            "rollout_phase": zero_rollout,
            "rollout_peak_trough": zero_rollout,
            "rollout_global_extrema": zero_rollout,
            "rollout_extrema_window_underfit": zero_rollout,
            "rollout_high_amplitude_underfit": zero_rollout,
            "reversal": zero_rollout,
            "turning_point": zero_rollout,
            "rollout_local_prominence": zero_rollout,
            "rollout_high_pass_residual": zero_rollout,
            "rollout_roll_spectral_shape": zero_rollout,
            "rollout_roll_curvature": zero_rollout,
        }
    if (
        rollout_enabled
        and bool(cfg.get("direct_forecast_loss_enabled", True))
        and any(w > 0.0 for w in direct_weights)
    ):
        direct_batch = int(y_phi.shape[0])
        if feedback_active:
            direct_batch = min(
                direct_batch,
                max(1, int(cfg.get("direct_forecast_loss_batch_size", 1))),
            )
        direct_rollout = direct_lstm_rollout_losses(
            model,
            phi_pred[:direct_batch],
            v_pred[:direct_batch],
            tgt[:direct_batch],
            y_phi[:direct_batch],
            y_v[:direct_batch],
            t_phys[:direct_batch],
            cfg,
        )
    else:
        zero_direct = torch.mean((phi_pred * 0.0) ** 2)
        direct_rollout = {
            "direct_forecast": zero_direct,
            "direct_forecast_rate": zero_direct,
            "direct_forecast_amplitude": zero_direct,
            "direct_forecast_phase": zero_direct,
            "direct_forecast_peak_trough": zero_direct,
            "direct_forecast_global_extrema": zero_direct,
            "direct_forecast_extrema_window_underfit": zero_direct,
            "direct_forecast_high_amplitude_underfit": zero_direct,
            "direct_forecast_asymmetric_extrema": zero_direct,
            "direct_forecast_local_prominence": zero_direct,
            "direct_forecast_high_pass_residual": zero_direct,
            "direct_forecast_roll_spectral_shape": zero_direct,
            "direct_forecast_roll_curvature": zero_direct,
        }

    total = (
        float(cfg["lambda_data"]) * l_data
        + float(cfg.get("lambda_r2_data", 0.0)) * l_r2_data
        + float(cfg.get("lambda_peak_trough", 0.0)) * l_peak_trough
        + float(cfg.get("lambda_global_extrema", 0.0)) * l_global_extrema
        + float(cfg.get("lambda_amplitude_underfit", 0.0)) * l_amp_underfit
        + float(cfg.get("lambda_local_prominence", 0.0)) * l_local_prominence
        + float(cfg.get("lambda_extrema_window_underfit", 0.0)) * l_extrema_window_underfit
        + float(cfg.get("lambda_high_amplitude_underfit", 0.0)) * l_high_amplitude_underfit
        + float(cfg.get("lambda_high_pass_residual", 0.0)) * l_high_pass_residual
        + float(cfg.get("lambda_low_pass_residual", 0.0)) * l_low_pass_residual
        + float(cfg.get("lambda_roll_spectral_shape", 0.0)) * l_roll_spectral_shape
        + float(cfg.get("lambda_roll_curvature", 0.0)) * l_roll_curvature
        + float(cfg["lambda_rate_data"]) * l_rate
        + float(cfg.get("lambda_roll_slope", 0.0)) * l_roll_slope
        + float(cfg["lambda_kinematic"]) * l_kin
        + float(cfg["lambda_physics"]) * l_phys
        + float(cfg["lambda_boundary"]) * l_bc
        + float(cfg["lambda_force_reg"]) * l_force
        + float(cfg["lambda_force_smooth"]) * l_force_smooth
        + float(cfg.get("lambda_force_mean", 0.0)) * l_force_mean
        + float(cfg.get("lambda_force_residual", 0.0)) * l_force_residual
        + float(cfg.get("lambda_wave_force_target", 0.0)) * l_wave_force_target
        + float(cfg.get("lambda_pure_wave_force_target", 0.0)) * l_pure_wave_force_target
        + float(cfg.get("lambda_total_force_target", 0.0)) * l_total_force_target
        + float(cfg.get("lambda_total_force_shape", 0.0)) * l_total_force_shape
        + float(cfg.get("lambda_total_force_tail", 0.0)) * l_total_force_tail
        + float(cfg.get("lambda_total_force_event", 0.0)) * l_total_force_event
        + float(cfg.get("lambda_total_force_band_shape", 0.0)) * l_total_force_band_shape
        + float(cfg.get("lambda_total_force_spectral_shape", 0.0)) * l_total_force_spectral_shape
        + float(cfg.get("lambda_total_force_regime_balance", 0.0)) * l_total_force_regime_balance
        + float(cfg.get("lambda_force_envelope", 0.0)) * l_force_envelope
        + float(cfg.get("lambda_wave_envelope_gate_reg", 0.0)) * l_wave_envelope_gate
        + float(cfg.get("lambda_wave_envelope_gate_target", 0.0)) * l_wave_envelope_gate_target
        + float(cfg.get("lambda_wave_envelope_gate_shape", 0.0)) * l_wave_envelope_gate_shape
        + float(cfg.get("lambda_rollout", 0.0)) * rollout["rollout"]
        + float(cfg.get("lambda_rollout_rate", 0.0)) * rollout["rollout_rate"]
        + float(cfg.get("lambda_rollout_amplitude", 0.0)) * rollout["rollout_amplitude"]
        + float(cfg.get("lambda_rollout_phase", 0.0)) * rollout["rollout_phase"]
        + float(cfg.get("lambda_rollout_peak_trough", 0.0)) * rollout["rollout_peak_trough"]
        + float(cfg.get("lambda_rollout_global_extrema", 0.0)) * rollout["rollout_global_extrema"]
        + float(cfg.get("lambda_rollout_extrema_window_underfit", 0.0)) * rollout["rollout_extrema_window_underfit"]
        + float(cfg.get("lambda_rollout_high_amplitude_underfit", 0.0)) * rollout["rollout_high_amplitude_underfit"]
        + float(cfg.get("lambda_reversal", 0.0)) * rollout["reversal"]
        + float(cfg.get("lambda_turning_point", 0.0)) * rollout["turning_point"]
        + float(cfg.get("lambda_rollout_local_prominence", 0.0)) * rollout["rollout_local_prominence"]
        + float(cfg.get("lambda_rollout_high_pass_residual", 0.0)) * rollout["rollout_high_pass_residual"]
        + float(cfg.get("lambda_rollout_roll_spectral_shape", 0.0)) * rollout["rollout_roll_spectral_shape"]
        + float(cfg.get("lambda_rollout_roll_curvature", 0.0)) * rollout["rollout_roll_curvature"]
        + float(cfg.get("lambda_direct_forecast", 0.0)) * direct_rollout["direct_forecast"]
        + float(cfg.get("lambda_direct_forecast_rate", 0.0)) * direct_rollout["direct_forecast_rate"]
        + float(cfg.get("lambda_direct_forecast_amplitude", 0.0)) * direct_rollout["direct_forecast_amplitude"]
        + float(cfg.get("lambda_direct_forecast_phase", 0.0)) * direct_rollout["direct_forecast_phase"]
        + float(cfg.get("lambda_direct_forecast_peak_trough", 0.0)) * direct_rollout["direct_forecast_peak_trough"]
        + float(cfg.get("lambda_direct_forecast_global_extrema", 0.0)) * direct_rollout["direct_forecast_global_extrema"]
        + float(cfg.get("lambda_direct_forecast_extrema_window_underfit", 0.0)) * direct_rollout["direct_forecast_extrema_window_underfit"]
        + float(cfg.get("lambda_direct_forecast_high_amplitude_underfit", 0.0)) * direct_rollout["direct_forecast_high_amplitude_underfit"]
        + float(cfg.get("lambda_direct_forecast_asymmetric_extrema", 0.0)) * direct_rollout["direct_forecast_asymmetric_extrema"]
        + float(cfg.get("lambda_direct_forecast_local_prominence", 0.0)) * direct_rollout["direct_forecast_local_prominence"]
        + float(cfg.get("lambda_direct_forecast_high_pass_residual", 0.0)) * direct_rollout["direct_forecast_high_pass_residual"]
        + float(cfg.get("lambda_direct_forecast_roll_spectral_shape", 0.0)) * direct_rollout["direct_forecast_roll_spectral_shape"]
        + float(cfg.get("lambda_direct_forecast_roll_curvature", 0.0)) * direct_rollout["direct_forecast_roll_curvature"]
    )
    pieces = {
        "total": total,
        "data": l_data,
        "r2_data": l_r2_data,
        "peak_trough": l_peak_trough,
        "global_extrema": l_global_extrema,
        "amplitude_underfit": l_amp_underfit,
        "local_prominence": l_local_prominence,
        "extrema_window_underfit": l_extrema_window_underfit,
        "high_amplitude_underfit": l_high_amplitude_underfit,
        "high_pass_residual": l_high_pass_residual,
        "low_pass_residual": l_low_pass_residual,
        "roll_spectral_shape": l_roll_spectral_shape,
        "roll_curvature": l_roll_curvature,
        "rate": l_rate,
        "roll_slope": l_roll_slope,
        "kinematic": l_kin,
        "physics": l_phys,
        "boundary": l_bc,
        "force_reg": l_force,
        "force_smooth": l_force_smooth,
        "force_mean": l_force_mean,
        "force_residual": l_force_residual,
        "wave_force_target": l_wave_force_target,
        "pure_wave_force_target": l_pure_wave_force_target,
        "total_force_target": l_total_force_target,
        "total_force_shape": l_total_force_shape,
        "total_force_tail": l_total_force_tail,
        "total_force_event": l_total_force_event,
        "total_force_band_shape": l_total_force_band_shape,
        "total_force_spectral_shape": l_total_force_spectral_shape,
        "total_force_regime_balance": l_total_force_regime_balance,
        "force_envelope": l_force_envelope,
        "wave_envelope_gate": l_wave_envelope_gate,
        "wave_envelope_gate_target": l_wave_envelope_gate_target,
        "wave_envelope_gate_shape": l_wave_envelope_gate_shape,
        "rollout": rollout["rollout"],
        "rollout_rate": rollout["rollout_rate"],
        "rollout_amplitude": rollout["rollout_amplitude"],
        "rollout_phase": rollout["rollout_phase"],
        "rollout_peak_trough": rollout["rollout_peak_trough"],
        "rollout_global_extrema": rollout["rollout_global_extrema"],
        "rollout_extrema_window_underfit": rollout["rollout_extrema_window_underfit"],
        "rollout_high_amplitude_underfit": rollout["rollout_high_amplitude_underfit"],
        "reversal": rollout["reversal"],
        "turning_point": rollout["turning_point"],
        "rollout_local_prominence": rollout["rollout_local_prominence"],
        "rollout_high_pass_residual": rollout["rollout_high_pass_residual"],
        "rollout_roll_spectral_shape": rollout["rollout_roll_spectral_shape"],
        "rollout_roll_curvature": rollout["rollout_roll_curvature"],
        "direct_forecast": direct_rollout["direct_forecast"],
        "direct_forecast_rate": direct_rollout["direct_forecast_rate"],
        "direct_forecast_amplitude": direct_rollout["direct_forecast_amplitude"],
        "direct_forecast_phase": direct_rollout["direct_forecast_phase"],
        "direct_forecast_peak_trough": direct_rollout["direct_forecast_peak_trough"],
        "direct_forecast_global_extrema": direct_rollout["direct_forecast_global_extrema"],
        "direct_forecast_extrema_window_underfit": direct_rollout["direct_forecast_extrema_window_underfit"],
        "direct_forecast_high_amplitude_underfit": direct_rollout["direct_forecast_high_amplitude_underfit"],
        "direct_forecast_asymmetric_extrema": direct_rollout["direct_forecast_asymmetric_extrema"],
        "direct_forecast_local_prominence": direct_rollout["direct_forecast_local_prominence"],
        "direct_forecast_high_pass_residual": direct_rollout["direct_forecast_high_pass_residual"],
        "direct_forecast_roll_spectral_shape": direct_rollout["direct_forecast_roll_spectral_shape"],
        "direct_forecast_roll_curvature": direct_rollout["direct_forecast_roll_curvature"],
        "rollout_active": torch.as_tensor(1.0 if rollout_enabled and any(w > 0.0 for w in rollout_weights) else 0.0,
                                          dtype=force_pred.dtype, device=force_pred.device),
        "direct_forecast_active": torch.as_tensor(
            1.0 if (
                rollout_enabled
                and bool(cfg.get("direct_forecast_loss_enabled", True))
                and any(w > 0.0 for w in direct_weights)
            ) else 0.0,
            dtype=force_pred.dtype,
            device=force_pred.device,
        ),
    }
    return total, pieces


# =============================================================================
# TRAINING / CHECKPOINTS
# =============================================================================

def build_model(cfg: Dict[str, object], data: Dict[str, object]) -> PINNLSTM:
    return PINNLSTM(
        cfg,
        int(data["input_dim"]),
        data["wave_feature_indices"],
        data["state_feature_indices"],
        data["motion_feature_indices"],
    ).to(data["device"])


def log_model_summary(model: PINNLSTM, lg: logging.Logger) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = sum(1 for m in model.modules() if isinstance(m, nn.Embedding))
    lg.info("Model: standard causal stacked LSTM, numeric feature adapter, no positional encoding or attention")
    if not bool(model.cfg.get("pinn_enabled", True)):
        lg.info("PINN training terms disabled; ODE/force components are diagnostics only")
    lg.info("Total parameters: %d", n_params)
    lg.info("nn.Embedding layers: %d", n_embed)
    lg.info("Initial c1=%.6f c2=%.6f k1=%.6f", model.c_roll.item(), model.c_quad.item(), model.k_roll.item())
    if n_embed != 0:
        raise RuntimeError("The LSTM comparison must not contain nn.Embedding layers.")


def resolve_amp_settings(cfg: Dict[str, object], device: torch.device) -> Tuple[bool, torch.dtype, str]:
    if device.type != "cuda" or not bool(cfg.get("use_amp", True)):
        return False, torch.float32, "off"
    requested = str(cfg.get("amp_dtype", "float16")).strip().lower()
    if requested in {"bf16", "bfloat16"}:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, "bfloat16"
        return True, torch.float16, "float16"
    return True, torch.float16, "float16"


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch builds
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model: PINNLSTM, cfg: Dict[str, object],
                    device: torch.device, lg: logging.Logger) -> torch.optim.Optimizer:
    kwargs = {
        "lr": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
    }
    if device.type == "cuda" and bool(cfg.get("use_fused_adamw", True)):
        try:
            opt = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
            cfg["resolved_fused_adamw"] = True
            return opt
        except (TypeError, RuntimeError) as exc:
            lg.warning("Fused AdamW unavailable; falling back to standard AdamW: %s", exc)
    cfg["resolved_fused_adamw"] = False
    return torch.optim.AdamW(model.parameters(), **kwargs)



def _format_log_duration(seconds: float) -> str:
    try:
        total = int(max(0.0, round(float(seconds))))
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _format_log_value(value: object, fmt: str = ".5e") -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return format(val, fmt)


def training_run_condition(cfg: Dict[str, object]) -> str:
    explicit = str(cfg.get("run_condition", "")).strip()
    if explicit:
        return explicit
    text = " ".join(str(cfg.get(k, "")) for k in ("output_dir", "checkpoint_dir", "log_dir", "bayes_opt_dir")).lower()
    if "exploratory_requested_train_pct" in cfg:
        return (
            f"exploratory train={int(cfg.get('exploratory_requested_train_pct', 0))}% "
            f"val={int(cfg.get('exploratory_requested_val_pct', 0))}% "
            f"forecast={int(cfg.get('exploratory_requested_forecast_pct', 0))}%"
        )
    if "seed_sweep" in text:
        return f"seed-sweep seed={cfg.get('seed', 'n/a')}"
    if bool(cfg.get("bayes_compact_logging", False)):
        if "grid" in text:
            mode = "grid"
        elif "best_run_outputs" in text:
            mode = "bayes-best-retrain"
        else:
            mode = "bayes"
        try:
            trial = int(cfg.get("bayes_trial_index", 0))
        except (TypeError, ValueError):
            trial = 0
        try:
            total = int(cfg.get("bayes_total_trials", 0))
        except (TypeError, ValueError):
            total = 0
        if trial > 0 and total > 0:
            return f"{mode} trial={trial:05d}/{total:05d}"
        if trial > 0:
            return f"{mode} trial={trial:05d}"
        return mode
    return "standard"

def scheduled_rollout_enabled(cfg: Dict[str, object], epoch: int, batch_idx: int) -> bool:
    warmup = max(0, int(cfg.get("rollout_warmup_epochs", 0)))
    every = max(1, int(cfg.get("rollout_every_n_batches", 1)))
    # batch_idx is one-based. Anchoring the schedule at batch 1 guarantees at
    # least one rollout in small/high-memory loaders whose epoch has fewer than
    # ``every`` batches so sparse schedules still activate recursive training.
    return int(epoch) > warmup and (int(batch_idx) - 1) % every == 0


def save_checkpoint(path: Path, model: PINNLSTM, optimizer: Optional[torch.optim.Optimizer],
                    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], epoch: int,
                    metrics: Dict[str, object], cfg: Dict[str, object], data: Dict[str, object], lg: logging.Logger,
                    checkpoint_kind: str = "training") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "checkpoint_kind": str(checkpoint_kind),
        "metrics": metrics,
        "config": copy.deepcopy(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "data_meta": {
            "phi_std": data.get("phi_std"),
            "v_std": data.get("v_std"),
            "best_wave_lag_s": data.get("best_wave_lag_s"),
            "wave_lags_s": data.get("wave_lags_s"),
            "dominant_roll_period_s": data.get("dominant_roll_period_s"),
            "wave_feature_indices": data.get("wave_feature_indices"),
            "state_feature_indices": data.get("state_feature_indices"),
            "motion_feature_indices": data.get("motion_feature_indices"),
            "motion_delay_steps_list": data.get("motion_delay_steps_list"),
        },
    }
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    lg.info("  Checkpoint saved: %s", path)


def load_checkpoint(path: str | Path, model: PINNLSTM,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                    lg: Optional[logging.Logger] = None) -> Tuple[int, Dict[str, float]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if lg:
        lg.info("Loaded checkpoint %s at epoch %s", path, ck.get("epoch"))
    return int(ck.get("epoch", 0)), ck.get("metrics", {})


@torch.no_grad()
def evaluate_loader(model: PINNLSTM, loader: Optional[DataLoader], cfg: Dict[str, object]) -> Dict[str, float]:
    if loader is None:
        return {"loss": float("nan")}
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    rollout_eval = bool(cfg.get("rollout_eval", False))
    for batch in loader:
        loss, pieces = compute_loss(model, batch, cfg, rollout_enabled=rollout_eval)
        bs = int(batch["src"].shape[0])
        n += bs
        for k, v in pieces.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().cpu().item()) * bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def direct_forecast_selection_objective(metrics: Dict[str, float],
                                        prefix: str,
                                        cfg: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
    """Composite validation objective for the direct LSTM forecast path."""
    prefix = str(prefix).strip("_")
    def value(name: str) -> float:
        return finite_or(metrics.get(f"{prefix}_{name}"), float("nan"))

    components = {
        "r2_data": value("r2_data"),
        "direct_forecast": value("direct_forecast"),
        "direct_forecast_peak_trough": value("direct_forecast_peak_trough"),
        "direct_forecast_extrema_window_underfit": value("direct_forecast_extrema_window_underfit"),
        "direct_forecast_asymmetric_extrema": value("direct_forecast_asymmetric_extrema"),
        "direct_forecast_roll_spectral_shape": value("direct_forecast_roll_spectral_shape"),
    }
    weighted_terms = [
        float(cfg.get("direct_forecast_selection_r2_weight", 1.0)) * components["r2_data"],
        float(cfg.get("direct_forecast_selection_loss_weight", 0.35)) * components["direct_forecast"],
        float(cfg.get("direct_forecast_selection_peak_weight", 1.0)) * components["direct_forecast_peak_trough"],
        float(cfg.get("direct_forecast_selection_extrema_weight", 0.75)) * components["direct_forecast_extrema_window_underfit"],
        float(cfg.get("direct_forecast_selection_asym_extrema_weight", 1.0)) * components["direct_forecast_asymmetric_extrema"],
        float(cfg.get("direct_forecast_selection_spectral_weight", 0.25)) * components["direct_forecast_roll_spectral_shape"],
    ]
    finite_terms = [float(v) for v in weighted_terms if math.isfinite(float(v))]
    objective = float(sum(finite_terms)) if finite_terms else float("nan")
    return objective, components


def train_model(model: PINNLSTM, train_loader: DataLoader,
                val_loader: Optional[DataLoader], data: Dict[str, object],
                cfg: Dict[str, object], lg: logging.Logger,
                resume: Optional[str] = None) -> Dict[str, object]:
    device = data["device"]
    opt = build_optimizer(model, cfg, device, lg)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(cfg["epochs"])))
    amp_enabled, amp_dtype, amp_label = resolve_amp_settings(cfg, device)
    cfg["resolved_amp_enabled"] = bool(amp_enabled)
    cfg["resolved_amp_dtype"] = amp_label
    scaler = make_grad_scaler(amp_enabled)
    progress_interval = max(50, int(cfg.get("log_every", 50)))
    lg.info(
        "Run condition: %s | epochs=%d | progress_every=%d | val_every=%d | selection=%s | PINN=%s | direct_forecast=%s | AMP=%s %s | fused_adamw=%s",
        training_run_condition(cfg),
        int(cfg["epochs"]),
        progress_interval,
        int(cfg.get("val_every", 1)),
        str(cfg.get("training_selection_metric", "r2")),
        "on" if bool(cfg.get("pinn_enabled", False)) else "off",
        "on" if bool(cfg.get("direct_forecast_loss_enabled", True)) else "off",
        "on" if amp_enabled else "off",
        amp_label,
        bool(cfg.get("resolved_fused_adamw", False)),
    )
    if bool(cfg.get("use_motion_feedback", False)):
        lg.info(
            "Motion feedback: delays=%s s feedback_window=%.3f s feedback_batch=%d (ordinary batch=%d)",
            [float(v) for v in cfg.get("motion_feedback_delay_offsets_s", [])],
            float(cfg.get("motion_feedback_rollout_window_s", cfg.get("rollout_window_s", 5.08))),
            max(1, int(cfg.get("motion_feedback_rollout_batch_size", cfg["batch_size"]))),
            int(cfg["batch_size"]),
        )
    if bool(cfg.get("direct_forecast_loss_enabled", True)):
        direct_horizons = config_float_list(
            cfg,
            "direct_forecast_loss_horizons_s",
            [float(cfg.get("direct_forecast_loss_window_s", 5.08))],
        )
        direct_horizon_weights = config_float_list(
            cfg,
            "direct_forecast_loss_horizon_weights",
            [1.0] * len(direct_horizons),
        )
        lg.info(
            "Direct LSTM forecast loss: horizons=%s weights=%s tail=%.0f%%x%.2g batch=%d peak=%.3g extrema=%.3g spectrum=%.3g",
            [float(v) for v in direct_horizons],
            [float(v) for v in direct_horizon_weights],
            100.0 * float(cfg.get("direct_forecast_tail_fraction", 0.0)),
            float(cfg.get("direct_forecast_tail_multiplier", 1.0)),
            int(cfg.get("direct_forecast_loss_batch_size", 1)),
            float(cfg.get("lambda_direct_forecast_peak_trough", 0.0)),
            float(cfg.get("lambda_direct_forecast_extrema_window_underfit", 0.0)),
            float(cfg.get("lambda_direct_forecast_roll_spectral_shape", 0.0)),
        )
    start_epoch = 0
    resume_metrics: Dict[str, float] = {}
    if resume:
        start_epoch, resume_metrics = load_checkpoint(resume, model, opt, sch, lg)

    best_metric = float("inf")
    best_epoch = start_epoch
    if resume_metrics:
        for key in ("best_monitor_objective", "best_metric", "monitored_objective", "monitored_loss"):
            try:
                candidate = float(resume_metrics.get(key, float("nan")))
            except (TypeError, ValueError):
                candidate = float("nan")
            if math.isfinite(candidate):
                best_metric = candidate
                break
        try:
            resumed_best_epoch = int(resume_metrics.get("best_epoch", start_epoch))
            if resumed_best_epoch > 0:
                best_epoch = resumed_best_epoch
        except (TypeError, ValueError):
            best_epoch = start_epoch
    best_state = None
    no_improve = 0
    history: List[Dict[str, float]] = []
    last_checkpoint_metrics: Dict[str, object] = {}
    final_checkpoint_path: Optional[Path] = None
    best_checkpoint_path: Optional[Path] = None
    t0 = time.time()
    bayes_compact_logging = bool(cfg.get("bayes_compact_logging", False))

    for ep in range(start_epoch + 1, int(cfg["epochs"]) + 1):
        model.train()
        accum: Dict[str, float] = {}
        n = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            rollout_enabled = scheduled_rollout_enabled(cfg, ep, batch_idx)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, pieces = compute_loss(model, batch, cfg, rollout_enabled=rollout_enabled)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
                opt.step()
            bs = int(batch["src"].shape[0])
            n += bs
            for k, v in pieces.items():
                accum[k] = accum.get(k, 0.0) + float(v.detach().cpu().item()) * bs
        sch.step()
        train_metrics = {f"train_{k}": v / max(n, 1) for k, v in accum.items()}

        if bayes_compact_logging:
            should_val = (ep == int(cfg["epochs"]) or ep % int(cfg["val_every"]) == 0)
        else:
            should_val = (ep == 1 or ep == int(cfg["epochs"]) or ep % int(cfg["val_every"]) == 0)
        validation_available = val_loader is not None
        validation_evaluated = bool(validation_available and should_val)
        val_metrics = (
            {f"val_{k}": v for k, v in evaluate_loader(model, val_loader, cfg).items()}
            if validation_evaluated else {"val_total": float("nan")}
        )
        loss_monitor = float(val_metrics.get("val_total", float("nan")))
        loss_monitor_name = "validation_loss"
        if not math.isfinite(loss_monitor) and not validation_available:
            loss_monitor = float(train_metrics["train_total"])
            loss_monitor_name = "training_loss"
        elif not math.isfinite(loss_monitor):
            loss_monitor_name = "validation_loss_pending"

        monitored = loss_monitor
        monitor_name = loss_monitor_name
        monitor_r2 = float("nan")
        monitor_forecast_r2 = float("nan")
        monitor_forecast_rmse_deg = float("nan")
        monitor_direct_components: Dict[str, float] = {}
        selection_metric = str(cfg.get("training_selection_metric", "r2")).strip().lower()
        forecast_selection = selection_metric in {"forecast_r2", "forecast", "hidden_forecast_r2"}
        forecast_evaluated = False
        if forecast_selection and should_val:
            forecast_payload = forecast_roll_region(model, data, cfg)
            forecast_metrics = (
                forecast_payload.get("metrics", {})
                if isinstance(forecast_payload, dict) and bool(forecast_payload.get("enabled", False))
                else {}
            )
            if isinstance(forecast_metrics, dict):
                monitor_forecast_r2 = finite_or(forecast_metrics.get("r2"), float("nan"))
                monitor_forecast_rmse_deg = finite_or(
                    forecast_metrics.get("rmse_deg"),
                    finite_or(forecast_metrics.get("error_rms_deg"), float("nan")),
                )
            forecast_evaluated = math.isfinite(monitor_forecast_r2)
            if forecast_evaluated:
                monitored = -monitor_forecast_r2
                monitor_name = "forecast_r2"
                monitor_r2 = monitor_forecast_r2
        elif selection_metric in {"direct_forecast", "direct", "forecast_direct", "lstm_forecast"}:
            if validation_available:
                direct_objective, direct_components = direct_forecast_selection_objective(
                    val_metrics,
                    "val",
                    cfg,
                )
                r2_objective = direct_components.get("r2_data", float("nan"))
                direct_source = (
                    "validation_direct_forecast"
                    if validation_evaluated
                    else "validation_direct_forecast_pending"
                )
            else:
                direct_objective, direct_components = direct_forecast_selection_objective(
                    train_metrics,
                    "train",
                    cfg,
                )
                r2_objective = direct_components.get("r2_data", float("nan"))
                direct_source = "training_direct_forecast"
            if math.isfinite(direct_objective):
                monitored = direct_objective
                monitor_name = direct_source
                monitor_direct_components = direct_components
                if math.isfinite(r2_objective):
                    monitor_r2 = 1.0 - r2_objective
        elif selection_metric in {"r2", "fit_r2", "r2_primary"}:
            if validation_available:
                r2_objective = float(val_metrics.get("val_r2_data", float("nan")))
                r2_source = "validation_r2" if validation_evaluated else "validation_r2_pending"
            else:
                r2_objective = float(train_metrics.get("train_r2_data", float("nan")))
                r2_source = "training_r2"
            if math.isfinite(r2_objective):
                monitored = r2_objective
                monitor_name = r2_source
                monitor_r2 = 1.0 - r2_objective

        selection_ready = (
            math.isfinite(monitored)
            and (
                forecast_evaluated
                if forecast_selection
                else (not validation_available or validation_evaluated)
            )
        )
        checkpoint_best_metric = float(best_metric)
        checkpoint_best_epoch = float(best_epoch)
        if selection_ready and monitored < best_metric:
            checkpoint_best_metric = float(monitored)
            checkpoint_best_epoch = float(ep)
        checkpoint_best_forecast_r2 = (
            -checkpoint_best_metric
            if forecast_selection and math.isfinite(checkpoint_best_metric)
            else float("nan")
        )
        row = {
            "epoch": float(ep),
            "lr": float(opt.param_groups[0]["lr"]),
            "c_roll": float(model.c_roll.detach().cpu().item()),
            "c_quad": float(model.c_quad.detach().cpu().item()),
            "k_roll": float(model.k_roll.detach().cpu().item()),
            "roll_amplitude_gain": float(model.roll_amplitude_gain.detach().cpu().item()),
            "monitor_objective": float(monitored),
            "monitor_metric": monitor_name,
            "monitor_r2": float(monitor_r2),
            "monitor_forecast_r2": float(monitor_forecast_r2),
            "monitor_forecast_rmse_deg": float(monitor_forecast_rmse_deg),
            "monitor_loss": float(loss_monitor),
            "monitor_direct_forecast": float(monitor_direct_components.get("direct_forecast", float("nan"))),
            "monitor_direct_forecast_peak_trough": float(monitor_direct_components.get("direct_forecast_peak_trough", float("nan"))),
            "monitor_direct_forecast_extrema_window_underfit": float(monitor_direct_components.get("direct_forecast_extrema_window_underfit", float("nan"))),
            "monitor_direct_forecast_asymmetric_extrema": float(monitor_direct_components.get("direct_forecast_asymmetric_extrema", float("nan"))),
            "monitor_direct_forecast_roll_spectral_shape": float(monitor_direct_components.get("direct_forecast_roll_spectral_shape", float("nan"))),
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

        checkpoint_metrics = {
            "monitored_objective": float(monitored),
            "best_monitor_objective": float(checkpoint_best_metric),
            "monitored_loss": float(loss_monitor),
            "monitor_metric": monitor_name,
            "monitored_r2": float(monitor_r2),
            "monitored_forecast_r2": float(monitor_forecast_r2),
            "monitored_forecast_rmse_deg": float(monitor_forecast_rmse_deg),
            "best_forecast_r2": float(checkpoint_best_forecast_r2),
            "monitored_direct_forecast": float(monitor_direct_components.get("direct_forecast", float("nan"))),
            "monitored_direct_forecast_peak_trough": float(monitor_direct_components.get("direct_forecast_peak_trough", float("nan"))),
            "monitored_direct_forecast_extrema_window_underfit": float(monitor_direct_components.get("direct_forecast_extrema_window_underfit", float("nan"))),
            "monitored_direct_forecast_roll_spectral_shape": float(monitor_direct_components.get("direct_forecast_roll_spectral_shape", float("nan"))),
            "best_metric": (
                float(-checkpoint_best_metric)
                if forecast_selection and math.isfinite(checkpoint_best_metric)
                else float(checkpoint_best_metric)
            ),
            "best_epoch": float(checkpoint_best_epoch),
            "train_total": float(train_metrics.get("train_total", float("nan"))),
            "val_total": float(val_metrics.get("val_total", float("nan"))),
            "train_r2": float(1.0 - float(train_metrics.get("train_r2_data", float("nan")))),
            "val_r2": float(1.0 - float(val_metrics.get("val_r2_data", float("nan")))),
        }
        last_checkpoint_metrics = copy.deepcopy(checkpoint_metrics)

        improved = (
            selection_ready
            and monitored < best_metric - float(cfg.get("early_stopping_min_delta", 0.0))
        )
        if improved:
            best_metric = monitored
            best_epoch = ep
            no_improve = 0
            checkpoint_metrics["best_monitor_objective"] = float(best_metric)
            checkpoint_metrics["best_metric"] = (
                float(-best_metric)
                if forecast_selection and math.isfinite(best_metric)
                else float(best_metric)
            )
            if forecast_selection and math.isfinite(best_metric):
                checkpoint_metrics["best_forecast_r2"] = float(-best_metric)
            checkpoint_metrics["best_epoch"] = float(best_epoch)
            if bool(cfg.get("keep_best_state", True)):
                best_state = copy.deepcopy(model.state_dict())
            if bool(cfg.get("save_checkpoints", True)) and bool(cfg.get("save_best_checkpoint", False)):
                best_checkpoint_path = Path(str(cfg["checkpoint_dir"])) / "best.pt"
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    None,
                    None,
                    ep,
                    checkpoint_metrics,
                    cfg,
                    data,
                    lg,
                    checkpoint_kind="best_epoch_model_weights",
                )
        elif selection_ready:
            no_improve += 1

        checkpoint_every = int(cfg.get("checkpoint_every", 0))
        if (
            bool(cfg.get("save_checkpoints", True))
            and bool(cfg.get("save_periodic_checkpoints", False))
            and checkpoint_every > 0
            and ep % checkpoint_every == 0
        ):
            save_checkpoint(Path(str(cfg["checkpoint_dir"])) / f"epoch_{ep:05d}.pt", model, opt, sch, ep,
                            checkpoint_metrics, cfg, data, lg)
        if bool(cfg.get("save_checkpoints", True)) and bool(cfg.get("save_last_checkpoint", False)):
            last_every = max(1, checkpoint_every)
            if ep == int(cfg["epochs"]) or ep % last_every == 0:
                save_checkpoint(Path(str(cfg["checkpoint_dir"])) / "last.pt", model, opt, sch, ep,
                                checkpoint_metrics, cfg, data, lg)

        should_log = (ep == int(cfg["epochs"]) or ep % progress_interval == 0)
        if should_log:
            elapsed_s = time.time() - t0
            completed_epochs = max(1, ep - start_epoch)
            total_epochs = max(1, int(cfg["epochs"]) - start_epoch)
            eta_s = max(0.0, elapsed_s / completed_epochs * max(0, total_epochs - completed_epochs))
            if bayes_compact_logging:
                best_text = (
                    f"composite={cfg.get('bayes_best_composite_run', 'none')},"
                    f"r2={cfg.get('bayes_best_r2_run', 'none')},"
                    f"rmse={cfg.get('bayes_best_rmse_run', 'none')}"
                )
            else:
                best_text = f"epoch={best_epoch}" if best_epoch > 0 else "pending"
            lg.info(
                "Progress | condition=%s | status=running | epoch=%d/%d | elapsed=%s | eta=%s | "
                "train=%s | val=%s | R2=%s | obj=%s:%s | direct=%s | phys=%s | kin=%s | lr=%s | best=%s",
                training_run_condition(cfg),
                ep,
                int(cfg["epochs"]),
                _format_log_duration(elapsed_s),
                _format_log_duration(eta_s),
                _format_log_value(train_metrics.get("train_total", float("nan"))),
                _format_log_value(val_metrics.get("val_total", float("nan"))),
                _format_log_value(monitor_r2, ".5f"),
                monitor_name,
                _format_log_value(monitored),
                _format_log_value(train_metrics.get("train_direct_forecast", float("nan"))),
                _format_log_value(train_metrics.get("train_physics", float("nan"))),
                _format_log_value(train_metrics.get("train_kinematic", float("nan"))),
                _format_log_value(row.get("lr", float("nan")), ".3e"),
                best_text,
            )

        patience = int(cfg.get("early_stopping_patience", 0))
        if patience > 0 and no_improve >= patience:
            lg.info("Early stopping at epoch %d. Best epoch=%d monitored_objective=%.6e", ep, best_epoch, best_metric)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        lg.info("Restored best validation/checkpoint state into memory.")
    elif resume and bool(cfg.get("keep_best_state", True)):
        best_path = Path(str(cfg["checkpoint_dir"])) / "best.pt"
        if best_path.exists():
            try:
                load_checkpoint(best_path, model, None, None, lg)
                best_checkpoint_path = best_path
                lg.info("Restored existing best checkpoint after resumed training.")
            except Exception as exc:
                lg.warning("Could not restore existing best checkpoint %s: %s", best_path, exc)
    if bool(cfg.get("save_checkpoints", True)) and bool(cfg.get("save_final_checkpoint", True)):
        final_metrics = copy.deepcopy(last_checkpoint_metrics)
        final_best_metric = (
            float(-best_metric)
            if str(cfg.get("training_selection_metric", "r2")).strip().lower() in {"forecast_r2", "forecast", "hidden_forecast_r2"}
            and math.isfinite(best_metric)
            else float(best_metric)
        )
        final_metrics.update({
            "best_monitor_objective": float(best_metric),
            "best_metric": final_best_metric,
            "best_forecast_r2": (
                final_best_metric
                if str(cfg.get("training_selection_metric", "r2")).strip().lower() in {"forecast_r2", "forecast", "hidden_forecast_r2"}
                else final_metrics.get("best_forecast_r2", float("nan"))
            ),
            "best_epoch": float(best_epoch),
            "selected_model_state": "best monitored state restored in memory",
        })
        final_name = str(cfg.get("final_checkpoint_name", "final.pt")).strip() or "final.pt"
        final_checkpoint_path = Path(str(cfg["checkpoint_dir"])) / final_name
        save_checkpoint(
            final_checkpoint_path,
            model,
            None,
            None,
            int(best_epoch if best_epoch > 0 else start_epoch),
            final_metrics,
            cfg,
            data,
            lg,
            checkpoint_kind="final_selected_model",
        )
    if data["device"].type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(data["device"]) / (1024 ** 3)
        lg.info("CUDA peak memory allocated: %.3f GB", peak_gb)
    lg.info("Training complete in %.2f min", (time.time() - t0) / 60.0)
    reported_best_metric = (
        float(-best_metric)
        if str(cfg.get("training_selection_metric", "r2")).strip().lower() in {"forecast_r2", "forecast", "hidden_forecast_r2"}
        and math.isfinite(best_metric)
        else float(best_metric)
    )
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_metric": reported_best_metric,
        "best_metric_name": str(cfg.get("training_selection_metric", "r2")),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "final_checkpoint": str(final_checkpoint_path or best_checkpoint_path) if (final_checkpoint_path is not None or best_checkpoint_path is not None) else None,
    }


# =============================================================================
# PREDICTION / METRICS / OUTPUTS
# =============================================================================

def stitch_window_by_causal_context(stitched: Dict[str, np.ndarray],
                                    best_context_steps: np.ndarray,
                                    indices: np.ndarray,
                                    candidates: Dict[str, np.ndarray]) -> None:
    """Keep the overlapping prediction with the longest causal history.

    A token later in a sequence has observed more preceding samples than the
    same timestamp near the beginning of a later overlapping window. Selecting
    the largest token position avoids both low-context boundary predictions and
    the peak blurring that can result from averaging phase-shifted predictions.
    """
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return
    context_steps = np.arange(idx.size, dtype=int)
    replace = context_steps > best_context_steps[idx]
    if not np.any(replace):
        return
    selected_idx = idx[replace]
    for name, output in stitched.items():
        candidate = np.asarray(candidates[name], dtype=float)
        if candidate.shape[0] != idx.size:
            raise ValueError(f"Context-stitch candidate {name!r} has the wrong length.")
        output[selected_idx] = candidate[replace]
    best_context_steps[selected_idx] = context_steps[replace]


@torch.no_grad()
def predict_full_teacher_forced(model: PINNLSTM, data: Dict[str, object],
                                cfg: Dict[str, object]) -> Tuple[
                                    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
                                ]:
    model.eval()
    X = np.asarray(data["X"], dtype=np.float32)
    t = np.asarray(data["t"], dtype=np.float32)
    N = len(X)
    seq_len = int(cfg["seq_len"])
    pred_stride = max(1, int(cfg.get("prediction_stride", cfg.get("stride", 1))))
    run_segments = normalise_segments(data.get("run_segments"), N)
    starts: List[int] = []
    for lo, hi in run_segments:
        run_starts = window_starts_within_segment(int(lo), int(hi), seq_len, pred_stride)
        starts.extend(int(s) for s in run_starts)
    if not starts:
        starts = list(range(0, max(N - seq_len + 1, 1), pred_stride))
        final_start = max(0, N - seq_len)
        if final_start not in starts:
            starts.append(final_start)
    starts = sorted(set(starts))

    stitched = {
        "phi": np.zeros(N, dtype=np.float64),
        "v": np.zeros(N, dtype=np.float64),
        "force": np.zeros(N, dtype=np.float64),
        "force_residual": np.zeros(N, dtype=np.float64),
        "turn": np.zeros(N, dtype=np.float64),
    }
    best_context_steps = np.full(N, -1, dtype=int)

    inference_batch_size = max(1, int(cfg.get("inference_batch_size", 64)))
    valid_starts = [int(s) for s in starts if int(s) + seq_len <= N]
    for batch_start in range(0, len(valid_starts), inference_batch_size):
        batch_starts = valid_starts[batch_start:batch_start + inference_batch_size]
        windows = np.stack([X[s:s + seq_len] for s in batch_starts], axis=0)
        src = torch.as_tensor(windows, dtype=torch.float32, device=data["device"])
        phi_p, v_p, force_p, force_residual_p = model(src, src)
        turn_p = model.turn_moment(src)
        phi_np = phi_p.squeeze(-1).detach().cpu().numpy()
        v_np = v_p.squeeze(-1).detach().cpu().numpy()
        force_np = force_p.squeeze(-1).detach().cpu().numpy()
        force_residual_np = force_residual_p.squeeze(-1).detach().cpu().numpy()
        turn_np = turn_p.squeeze(-1).detach().cpu().numpy()
        for row, s in enumerate(batch_starts):
            idx = np.arange(s, s + seq_len, dtype=int)
            stitch_window_by_causal_context(
                stitched,
                best_context_steps,
                idx,
                {
                    "phi": phi_np[row],
                    "v": v_np[row],
                    "force": force_np[row],
                    "force_residual": force_residual_np[row],
                    "turn": turn_np[row],
                },
            )

    # In the unlikely event a point was missed, run the nearest final window.
    missed = best_context_steps < 0
    if np.any(missed):
        for lo, hi in run_segments:
            idx_run = np.arange(int(lo), int(hi) + 1, dtype=int)
            if not np.any(missed[idx_run]):
                continue
            s = max(int(lo), int(hi) - seq_len + 1)
            e = int(hi) + 1
            if e - s < seq_len:
                continue
            src = torch.tensor(X[s:e], dtype=torch.float32, device=data["device"]).unsqueeze(0)
            phi_p, v_p, force_p, force_residual_p = model(src, src)
            turn_p = model.turn_moment(src)
            idx = np.arange(s, e, dtype=int)
            stitch_window_by_causal_context(
                stitched,
                best_context_steps,
                idx,
                {
                    "phi": phi_p.squeeze(0).squeeze(-1).detach().cpu().numpy(),
                    "v": v_p.squeeze(0).squeeze(-1).detach().cpu().numpy(),
                    "force": force_p.squeeze(0).squeeze(-1).detach().cpu().numpy(),
                    "force_residual": force_residual_p.squeeze(0).squeeze(-1).detach().cpu().numpy(),
                    "turn": turn_p.squeeze(0).squeeze(-1).detach().cpu().numpy(),
                },
            )

    return (
        stitched["phi"],
        stitched["v"],
        stitched["force"],
        stitched["turn"],
        stitched["force_residual"],
    )


def forecast_window_description(cfg: Dict[str, object]) -> str:
    duration_cfg = cfg.get("forecast_window_s", None)
    if duration_cfg is not None:
        return f"final {float(duration_cfg):g} seconds"
    return f"final {100.0 * float(cfg.get('forecast_frac', 0.10)):g}% of data"


def choose_forecast_region(t: np.ndarray, cfg: Dict[str, object]) -> Optional[Tuple[int, int]]:
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return None

    start_cfg = cfg.get("forecast_start_s", None)
    duration_cfg = cfg.get("forecast_window_s", None)
    if start_cfg is None and duration_cfg is None:
        fraction = min(max(float(cfg.get("forecast_frac", 0.10)), 0.0), 0.9)
        if fraction <= 0.0:
            return None
        forecast_points = int(min(len(t) - 1, max(2, round(len(t) * fraction))))
        return int(len(t) - forecast_points), int(len(t) - 1)

    duration = None if duration_cfg is None else max(0.0, float(duration_cfg))
    if duration is not None and duration <= 0.0:
        return None
    if start_cfg is None:
        assert duration is not None
        start_t = float(t[-1]) - duration
    else:
        start_t = float(start_cfg)
    start_t = min(max(start_t, float(t[1])), float(t[-2]))
    end_t = float(t[-1]) if duration is None else min(float(t[-1]), start_t + duration)
    # Select the nearest sampled endpoints. A strict searchsorted boundary can
    # lose one point when decimal sample times such as 0.04 are represented a
    # few ulps above the requested boundary (127 instead of 128 samples).
    start_idx = int(np.argmin(np.abs(t - start_t)))
    end_idx = int(np.argmin(np.abs(t - end_t)))
    start_idx = max(1, min(start_idx, len(t) - 2))
    end_idx = max(start_idx + 1, min(end_idx, len(t) - 1))
    return start_idx, end_idx


def _set_forecast_motion_feedback(X_work: np.ndarray, idx: int, start_idx: int, delay_steps: int | List[int],
                                  pred_phi_s: np.ndarray, pred_v_s: np.ndarray,
                                  data: Dict[str, object]) -> None:
    motion_indices = data.get("motion_feature_indices", {})
    if not isinstance(motion_indices, dict):
        return
    phi_idxs = motion_indices.get("phi_fb", [])
    v_idxs = motion_indices.get("v_fb", [])
    if len(phi_idxs) == 0 and len(v_idxs) == 0:
        return
    if isinstance(delay_steps, list):
        delays = [int(v) for v in delay_steps]
    else:
        delays = [int(delay_steps)]
    if len(delays) == 1 and len(phi_idxs) > 1:
        delays = list(data.get("motion_delay_steps_list", delays))
    if len(delays) < max(len(phi_idxs), len(v_idxs)):
        delays = delays + [delays[-1] if delays else 0] * (max(len(phi_idxs), len(v_idxs)) - len(delays))
    measured_phi = np.asarray(data["phi_scaled"], dtype=float)
    measured_v = np.asarray(data["v_est_scaled"], dtype=float)
    run_lo, _ = segment_for_index(data.get("run_segments"), len(measured_phi), idx)
    for i, col in enumerate(phi_idxs):
        ref = max(int(run_lo), int(idx) - int(delays[i]))
        phi_ref = float(pred_phi_s[ref]) if ref >= start_idx else float(measured_phi[ref])
        X_work[idx, int(col)] = phi_ref
    for i, col in enumerate(v_idxs):
        ref = max(int(run_lo), int(idx) - int(delays[i]))
        v_ref = float(pred_v_s[ref]) if ref >= start_idx else float(measured_v[ref])
        X_work[idx, int(col)] = v_ref


@torch.no_grad()
def _forecast_force_at(model: PINNLSTM, X_work: np.ndarray,
                       data: Dict[str, object], cfg: Dict[str, object], idx: int) -> float:
    seq_len = int(cfg["seq_len"])
    run_lo, _ = segment_for_index(data.get("run_segments"), len(X_work), idx)
    s = max(int(run_lo), int(idx) - seq_len + 1)
    e = int(idx) + 1
    window = torch.as_tensor(X_work[s:e], dtype=torch.float32, device=data["device"]).unsqueeze(0)
    _, _, force_without_turn, _ = model(window, window)
    turn = model.turn_moment(window)
    total_force = force_without_turn + turn
    return float(total_force[0, -1, 0].detach().cpu().item())


@torch.no_grad()
def direct_roll_prediction_scaled(model: PINNLSTM,
                                  data: Dict[str, object],
                                  cfg: Dict[str, object],
                                  start_idx: int,
                                  end_idx: int) -> Dict[str, object]:
    """Run the direct LSTM roll/rate forecast over a contiguous region."""
    model.eval()
    t = np.asarray(data["t"], dtype=float)
    X_work = np.asarray(data["X"], dtype=np.float32).copy()
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float)
    pred_phi_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_v_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_phi_s[:start_idx + 1] = measured_phi_s[:start_idx + 1]
    pred_v_s[:start_idx + 1] = measured_v_s[:start_idx + 1]

    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    delay_steps = list(data.get("motion_delay_steps_list", motion_feedback_delay_steps(cfg, dt_med)))
    motion_indices = data.get("motion_feature_indices", {})
    feedback_active = bool(
        isinstance(motion_indices, dict)
        and (motion_indices.get("phi_fb", []) or motion_indices.get("v_fb", []))
    )
    seq_len = int(cfg["seq_len"])
    for j in range(start_idx + 1, end_idx + 1):
        if feedback_active:
            _set_forecast_motion_feedback(
                X_work,
                j,
                start_idx,
                delay_steps,
                pred_phi_s,
                pred_v_s,
                data,
            )
        run_lo, _ = segment_for_index(data.get("run_segments"), len(t), j)
        s = max(int(run_lo), int(j) - seq_len + 1)
        window = torch.as_tensor(
            X_work[s:int(j) + 1],
            dtype=torch.float32,
            device=data["device"],
        ).unsqueeze(0)
        phi_p, v_p, _, _ = model(window, window)
        pred_phi_s[j] = float(phi_p[0, -1, 0].detach().cpu().item())
        pred_v_s[j] = float(v_p[0, -1, 0].detach().cpu().item())

    return {
        "pred_phi_scaled": pred_phi_s,
        "pred_v_scaled": pred_v_s,
        "X_work": X_work,
        "delay_steps": delay_steps,
        "feedback_active": feedback_active,
    }


def apply_direct_forecast_amplitude_gain(raw_deg: np.ndarray, gain: float) -> np.ndarray:
    raw = np.asarray(raw_deg, dtype=float)
    finite = np.isfinite(raw)
    if int(np.sum(finite)) < 2:
        return raw.copy()
    center = float(np.nanmean(raw[finite]))
    return center + float(gain) * (raw - center)


@torch.no_grad()
def select_direct_forecast_amplitude_calibration(model: PINNLSTM,
                                                 data: Dict[str, object],
                                                 cfg: Dict[str, object]) -> Dict[str, object]:
    """Select a scalar direct forecast amplitude gain using the validation block."""
    if not bool(cfg.get("direct_forecast_amplitude_calibration_enabled", False)):
        return {"enabled": False, "gain": 1.0, "reason": "direct forecast amplitude calibration is disabled."}
    val_idx = np.asarray(data.get("val_idx", []), dtype=int)
    if len(val_idx) < 5:
        return {"enabled": False, "gain": 1.0, "reason": "validation region too short."}
    start_idx = int(val_idx[0])
    end_idx = int(val_idx[-1])
    if start_idx < 1 or end_idx <= start_idx + 3:
        return {"enabled": False, "gain": 1.0, "reason": "validation handoff is invalid."}

    pred = direct_roll_prediction_scaled(model, data, cfg, start_idx, end_idx)
    idx = np.arange(start_idx, end_idx + 1, dtype=int)
    measured_deg = np.rad2deg(np.asarray(data["phi"], dtype=float)[idx])
    raw_deg = np.rad2deg(np.asarray(pred["pred_phi_scaled"], dtype=float)[idx] * float(data["phi_std"]))
    raw_metrics = regression_metrics(measured_deg, raw_deg) if len(idx) > 2 else {}
    if len(idx) > 2:
        raw_metrics.update(phase_metrics(measured_deg, raw_deg, np.asarray(data["t"], dtype=float)[idx]))

    gain_min = float(cfg.get("direct_forecast_amplitude_calibration_gain_min", 0.85))
    gain_max = float(cfg.get("direct_forecast_amplitude_calibration_gain_max", 1.25))
    steps = int(max(2, cfg.get("direct_forecast_amplitude_calibration_gain_steps", 41)))
    extrema_weight = float(cfg.get("direct_forecast_amplitude_calibration_extrema_weight", 0.35))
    gains = np.linspace(gain_min, gain_max, steps, dtype=float)
    best: Dict[str, object] = {
        "enabled": False,
        "gain": 1.0,
        "reason": "no finite validation calibration objective.",
        "validation_raw_metrics": raw_metrics,
    }
    best_objective = float("inf")
    for gain in gains:
        corrected = apply_direct_forecast_amplitude_gain(raw_deg, float(gain))
        if len(corrected) > 0:
            corrected[0] = raw_deg[0]
        metrics = regression_metrics(measured_deg, corrected) if len(idx) > 2 else {}
        if len(idx) > 2:
            metrics.update(phase_metrics(measured_deg, corrected, np.asarray(data["t"], dtype=float)[idx]))
        rmse = float(metrics.get("rmse_deg", float("nan")))
        extrema = float(metrics.get("extrema_rmse_deg", float("nan")))
        if not math.isfinite(rmse):
            continue
        objective = rmse + extrema_weight * (extrema if math.isfinite(extrema) else 0.0)
        if objective < best_objective:
            best_objective = float(objective)
            best = {
                "enabled": True,
                "gain": float(gain),
                "objective": float(objective),
                "objective_extrema_weight": float(extrema_weight),
                "validation_start_idx": int(start_idx),
                "validation_end_idx": int(end_idx),
                "validation_raw_metrics": raw_metrics,
                "validation_calibrated_metrics": metrics,
                "gain_min": float(gain_min),
                "gain_max": float(gain_max),
                "gain_steps": int(steps),
            }
    return best


@torch.no_grad()
def forecast_direct_roll_region(model: PINNLSTM,
                                data: Dict[str, object],
                                cfg: Dict[str, object],
                                region: Optional[Tuple[int, int]] = None) -> Dict[str, object]:
    """Direct LSTM roll forecast over the hidden region, without ODE integration."""
    model.eval()
    t = np.asarray(data["t"], dtype=float)
    resolved_region = tuple(int(v) for v in region) if region is not None else choose_forecast_region(t, cfg)
    if resolved_region is None:
        return {"enabled": False, "reason": "No valid forecast region."}
    start_idx, end_idx = resolved_region
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float)
    direct_pred = direct_roll_prediction_scaled(model, data, cfg, start_idx, end_idx)
    X_work = np.asarray(direct_pred["X_work"], dtype=np.float32)
    pred_phi_s = np.asarray(direct_pred["pred_phi_scaled"], dtype=np.float64)
    pred_v_s = np.asarray(direct_pred["pred_v_scaled"], dtype=np.float64)
    delay_steps = list(direct_pred.get("delay_steps", []))
    feedback_active = bool(direct_pred.get("feedback_active", False))

    idx = np.arange(start_idx, end_idx + 1, dtype=int)
    measured_deg = np.rad2deg(np.asarray(data["phi"], dtype=float)[idx])
    forecast_deg = np.rad2deg(pred_phi_s[idx] * float(data["phi_std"]))
    direct_metrics = regression_metrics(measured_deg, forecast_deg) if len(idx) > 2 else {}
    if len(idx) > 2:
        direct_metrics.update(phase_metrics(measured_deg, forecast_deg, t[idx]))
    uncorrected_forecast_deg = forecast_deg.copy()
    uncorrected_phi_s = pred_phi_s[idx].copy()
    uncorrected_v_s = pred_v_s[idx].copy()
    uncorrected_metrics = copy.deepcopy(direct_metrics)
    amplitude_calibration = select_direct_forecast_amplitude_calibration(model, data, cfg)
    if bool(amplitude_calibration.get("enabled", False)) and len(idx) > 2:
        gain = float(amplitude_calibration.get("gain", 1.0))
        calibrated_deg = apply_direct_forecast_amplitude_gain(forecast_deg, gain)
        if len(calibrated_deg) > 0:
            calibrated_deg[0] = forecast_deg[0]
        if len(calibrated_deg) == len(idx) and np.all(np.isfinite(calibrated_deg)):
            forecast_deg = calibrated_deg
            pred_phi_s[idx] = np.deg2rad(forecast_deg) / max(float(data["phi_std"]), 1.0e-12)
            direct_metrics = regression_metrics(measured_deg, forecast_deg)
            direct_metrics.update(phase_metrics(measured_deg, forecast_deg, t[idx]))
            amplitude_calibration["applied_to_forecast"] = True
            amplitude_calibration["forecast_uncalibrated_metrics"] = uncorrected_metrics
            amplitude_calibration["forecast_calibrated_metrics"] = copy.deepcopy(direct_metrics)
        else:
            amplitude_calibration = {
                **amplitude_calibration,
                "enabled": False,
                "applied_to_forecast": False,
                "reason": "calibrated direct forecast contained non-finite values.",
            }
    else:
        amplitude_calibration["applied_to_forecast"] = False
    diagnostic_force_s = np.full(len(idx), np.nan, dtype=np.float64)
    for pos, j in enumerate(idx):
        diagnostic_force_s[pos] = _forecast_force_at(model, X_work, data, cfg, int(j))
    wave_force_verification, pure_wave_force_s, wave_input_rms_s, orientation_gain_s, envelope_gate_s = (
        forecast_wave_force_verification(
            model,
            X_work,
            data,
            idx,
            diagnostic_force_s,
        )
    )
    wave_force_verification = copy.deepcopy(wave_force_verification)
    wave_force_verification["status"] = "direct_lstm_force_diagnostic_only"
    wave_force_verification["wave_force_applied_to_forecast_ode"] = False
    wave_force_verification["force_path_note"] = (
        "ODE rolling forecast is disabled. Force values in this forecast are "
        "diagnostic LSTM force-head outputs only; the primary roll "
        "forecast comes directly from the LSTM roll/rate heads."
    )
    handoff_phi_error_scaled = float(pred_phi_s[start_idx] - measured_phi_s[start_idx])
    handoff_v_error_scaled = float(pred_v_s[start_idx] - measured_v_s[start_idx])
    return {
        "enabled": True,
        "method": "direct_lstm_no_ode",
        "forecast_primary": "direct_lstm_no_ode",
        "forecast_ode_enabled": False,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_time_s": float(t[start_idx]),
        "end_time_s": float(t[end_idx]),
        "duration_s": float(t[end_idx] - t[start_idx]),
        "delay_steps": int(delay_steps[-1]) if delay_steps else 0,
        "feedback_delay_steps": [int(v) for v in delay_steps],
        "note": (
            "Direct roll-head forecast over the same hidden region. It starts "
            "from the measured handoff state; if delayed motion feedback is "
            "enabled, future feedback features are filled from previous direct "
            "predictions rather than hidden measured roll."
        ),
        "uses_predicted_motion_feedback": feedback_active,
        "training_feedback_rollout_window_s": (
            float(cfg.get("motion_feedback_rollout_window_s", cfg.get("rollout_window_s", 5.08)))
            if feedback_active else None
        ),
        "training_feedback_rollout_batch_size": (
            max(1, int(cfg.get("motion_feedback_rollout_batch_size", cfg.get("batch_size", 1))))
            if feedback_active else None
        ),
        "known_inputs_note": "The synchronised beam-sea wave, vessel-state, and wave-orientation gain channels remain in the model input tensor through every forecast step; roll labels are hidden.",
        "roll_dynamics_note": "ODE rolling forecast is disabled; the primary hidden-region forecast is produced directly by the LSTM roll/rate heads.",
        "wave_force_verification": wave_force_verification,
        "force_post_correction": {
            "enabled": False,
            "reason": "ODE rolling forecast is disabled.",
        },
        "direct_forecast_amplitude_calibration": amplitude_calibration,
        "forecast_force_lag_s": None,
        "forecast_force_lag_note": "Not used because the rolling ODE forecast is disabled.",
        "handoff_phi_error_scaled": handoff_phi_error_scaled,
        "handoff_phi_error_deg": float(np.rad2deg(handoff_phi_error_scaled * float(data["phi_std"]))),
        "handoff_v_error_scaled": handoff_v_error_scaled,
        "index": idx,
        "time_s": t[idx],
        "measured_roll_deg": measured_deg,
        "forecast_roll_deg": forecast_deg,
        "forecast_phi_scaled": pred_phi_s[idx],
        "forecast_v_scaled": pred_v_s[idx],
        "forecast_force_scaled": diagnostic_force_s,
        "forecast_raw_force_before_lag_scaled": diagnostic_force_s.copy(),
        "uncorrected_forecast_roll_deg": uncorrected_forecast_deg,
        "uncorrected_forecast_phi_scaled": uncorrected_phi_s,
        "uncorrected_forecast_v_scaled": uncorrected_v_s,
        "uncorrected_forecast_force_scaled": diagnostic_force_s.copy(),
        "uncorrected_metrics": uncorrected_metrics,
        "forecast_pure_wave_force_scaled": pure_wave_force_s,
        "forecast_wave_input_rms_scaled": wave_input_rms_s,
        "forecast_wave_orientation_gain": orientation_gain_s,
        "forecast_wave_envelope_gate": envelope_gate_s,
        "metrics": direct_metrics,
    }


def lag_force_by_segments(t: np.ndarray, force: np.ndarray,
                          segments: List[Tuple[int, int]], lag_s: float) -> np.ndarray:
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if len(force) != len(t):
        raise ValueError("Force lag input must match the time vector length.")
    lag = float(lag_s)
    if abs(lag) <= 1.0e-12:
        return force.copy()
    shifted = np.full(len(force), np.nan, dtype=np.float64)
    for lo, hi in segments:
        lo_i = int(max(0, lo))
        hi_i = int(min(len(force) - 1, hi))
        if hi_i < lo_i:
            continue
        sl = slice(lo_i, hi_i + 1)
        src_t = t[sl]
        src_f = force[sl]
        finite = np.isfinite(src_t) & np.isfinite(src_f)
        if int(np.sum(finite)) < 2:
            shifted[sl] = src_f
            continue
        # Positive lag delays the learned force: force(t) = raw_force(t-lag).
        shifted[sl] = np.interp(
            src_t - lag,
            src_t[finite],
            src_f[finite],
            left=float(src_f[finite][0]),
            right=float(src_f[finite][-1]),
        )
    return shifted


def lagged_force_value_from_history(t: np.ndarray, raw_force: np.ndarray,
                                    run_lo: int, idx: int, lag_s: float) -> float:
    lag = float(lag_s)
    if abs(lag) <= 1.0e-12:
        return float(raw_force[idx])
    lo_i = int(max(0, run_lo))
    hi_i = int(idx)
    src_t = np.asarray(t[lo_i:hi_i + 1], dtype=np.float64)
    src_f = np.asarray(raw_force[lo_i:hi_i + 1], dtype=np.float64)
    finite = np.isfinite(src_t) & np.isfinite(src_f)
    if int(np.sum(finite)) < 2:
        return float(raw_force[idx])
    query_t = float(t[idx]) - lag
    return float(np.interp(
        query_t,
        src_t[finite],
        src_f[finite],
        left=float(src_f[finite][0]),
        right=float(src_f[finite][-1]),
    ))


@torch.no_grad()
def forecast_wave_force_verification(
    model: PINNLSTM,
    X_work: np.ndarray,
    data: Dict[str, object],
    forecast_indices: np.ndarray,
    total_force_scaled: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Audit the explicit wave channels and force inside the forecast region."""
    idx = np.asarray(forecast_indices, dtype=int)
    total_force = np.asarray(total_force_scaled, dtype=float)
    raw_groups = data.get("wave_feature_indices", {})
    wave_groups = raw_groups if isinstance(raw_groups, dict) else {}
    wave_columns = sorted({
        int(column)
        for columns in wave_groups.values()
        if isinstance(columns, (list, tuple))
        for column in columns
    })
    wave_input_rms = np.full(len(idx), np.nan, dtype=np.float64)
    pure_wave_force = np.full(len(idx), np.nan, dtype=np.float64)
    orientation_gain = np.full(len(idx), np.nan, dtype=np.float64)
    envelope_gate = np.full(len(idx), np.nan, dtype=np.float64)
    counterfactual_deltas: List[float] = []
    state_groups = data.get("state_feature_indices", {})
    orientation_gain_columns = (
        state_groups.get("wave_orientation_effect_gain", [])
        if isinstance(state_groups, dict)
        else []
    )
    if len(idx) > 0 and orientation_gain_columns:
        orientation_values = np.asarray(
            X_work[np.ix_(idx, [int(c) for c in orientation_gain_columns])],
            dtype=np.float64,
        )
        orientation_gain = np.mean(orientation_values, axis=1)
    if len(idx) > 0 and wave_columns:
        wave_inputs = np.asarray(X_work[np.ix_(idx, wave_columns)], dtype=np.float64)
        wave_input_rms = np.sqrt(np.mean(wave_inputs ** 2, axis=1))
        forecast_X = torch.as_tensor(
            np.asarray(X_work[idx], dtype=np.float32),
            dtype=torch.float32,
            device=data["device"],
        ).unsqueeze(0)
        wave_breakdown = model.wave_forcing_breakdown(forecast_X)
        pure_wave_tensor = wave_breakdown["pure_wave_force"]
        pure_wave_force = (
            pure_wave_tensor[0, :, 0].detach().cpu().numpy().astype(np.float64)
        )
        wave_signal_force = (
            wave_breakdown["wave_signal_force"][0, :, 0].detach().cpu().numpy().astype(np.float64)
        )
        wave_slope_force = (
            wave_breakdown["wave_slope_force"][0, :, 0].detach().cpu().numpy().astype(np.float64)
        )
        wave_auxiliary_force = (
            wave_breakdown["wave_auxiliary_force"][0, :, 0].detach().cpu().numpy().astype(np.float64)
        )
        wave_shape_correction_force = (
            wave_breakdown["wave_shape_correction_force"][0, :, 0].detach().cpu().numpy().astype(np.float64)
        )
        envelope_gate = (
            wave_breakdown["wave_envelope_gate"][0, :, 0].detach().cpu().numpy().astype(np.float64)
        )
        wave_inputs_finite = bool(np.all(np.isfinite(wave_inputs)))
        # Sample the recursive forecast tensor and zero only its scaled wave
        # channels. The resulting total-force change proves that the trained
        # forecast path is wave-sensitive, without doubling every forecast step.
        sample_positions = np.unique(
            np.linspace(0, len(idx) - 1, min(16, len(idx)), dtype=int)
        )
        seq_len = int(model.cfg["seq_len"])
        for position in sample_positions:
            sample_idx = int(idx[int(position)])
            run_lo, _ = segment_for_index(
                data.get("run_segments"),
                len(X_work),
                sample_idx,
            )
            start = max(int(run_lo), sample_idx - seq_len + 1)
            zero_wave_window = np.asarray(
                X_work[start:sample_idx + 1],
                dtype=np.float32,
            ).copy()
            zero_wave_window[:, wave_columns] = 0.0
            zero_wave_tensor = torch.as_tensor(
                zero_wave_window,
                dtype=torch.float32,
                device=data["device"],
            ).unsqueeze(0)
            _, _, zero_wave_force_without_turn, _ = model(
                zero_wave_tensor,
                zero_wave_tensor,
            )
            zero_wave_total = (
                zero_wave_force_without_turn + model.turn_moment(zero_wave_tensor)
            )
            actual_total = float(total_force[int(position)])
            counterfactual_total = float(
                zero_wave_total[0, -1, 0].detach().cpu().item()
            )
            counterfactual_deltas.append(actual_total - counterfactual_total)
    else:
        wave_inputs_finite = False

    pure_wave_finite = bool(len(idx) > 0 and np.all(np.isfinite(pure_wave_force)))
    wave_signal_force_rms = (
        float(np.sqrt(np.mean(wave_signal_force ** 2)))
        if len(idx) > 0 and "wave_signal_force" in locals() and np.all(np.isfinite(wave_signal_force))
        else float("nan")
    )
    wave_slope_force_rms = (
        float(np.sqrt(np.mean(wave_slope_force ** 2)))
        if len(idx) > 0 and "wave_slope_force" in locals() and np.all(np.isfinite(wave_slope_force))
        else float("nan")
    )
    wave_auxiliary_force_rms = (
        float(np.sqrt(np.mean(wave_auxiliary_force ** 2)))
        if len(idx) > 0 and "wave_auxiliary_force" in locals() and np.all(np.isfinite(wave_auxiliary_force))
        else float("nan")
    )
    wave_shape_correction_force_rms = (
        float(np.sqrt(np.mean(wave_shape_correction_force ** 2)))
        if len(idx) > 0 and "wave_shape_correction_force" in locals()
        and np.all(np.isfinite(wave_shape_correction_force))
        else float("nan")
    )
    total_force_finite = bool(len(idx) > 0 and np.all(np.isfinite(total_force)))
    wave_input_rms_value = (
        float(np.sqrt(np.mean(wave_input_rms ** 2)))
        if np.any(np.isfinite(wave_input_rms))
        else float("nan")
    )
    pure_wave_rms_value = (
        float(np.sqrt(np.mean(pure_wave_force ** 2)))
        if pure_wave_finite
        else float("nan")
    )
    total_force_rms_value = (
        float(np.sqrt(np.mean(total_force ** 2)))
        if total_force_finite
        else float("nan")
    )
    wave_channels_present = bool(wave_columns and wave_inputs_finite)
    wave_force_nonzero = bool(
        pure_wave_finite
        and math.isfinite(pure_wave_rms_value)
        and pure_wave_rms_value > 1.0e-10
    )
    counterfactual_delta_rms = (
        float(np.sqrt(np.mean(np.asarray(counterfactual_deltas, dtype=float) ** 2)))
        if counterfactual_deltas
        else float("nan")
    )
    counterfactual_wave_effect_nonzero = bool(
        math.isfinite(counterfactual_delta_rms)
        and counterfactual_delta_rms > 1.0e-10
    )
    orientation_gain_finite = bool(len(idx) > 0 and np.all(np.isfinite(orientation_gain)))
    envelope_gate_finite = bool(len(idx) > 0 and np.all(np.isfinite(envelope_gate)))
    applied_to_ode = bool(
        wave_channels_present
        and pure_wave_finite
        and total_force_finite
        and len(total_force) == len(idx)
    )
    if applied_to_ode and wave_force_nonzero and counterfactual_wave_effect_nonzero:
        status = "verified_nonzero_wave_force"
    elif applied_to_ode:
        status = "wave_channels_verified_but_learned_wave_effect_is_negligible"
    else:
        status = "wave_force_verification_failed"
    verification = {
        "status": status,
        "wave_force_applied_to_forecast_ode": applied_to_ode,
        "wave_channels_present_in_forecast_tensor": wave_channels_present,
        "wave_feature_count": int(len(wave_columns)),
        "wave_feature_indices": wave_columns,
        "wave_input_all_finite": wave_inputs_finite,
        "wave_input_rms_scaled": wave_input_rms_value,
        "wave_orientation_gain_present": bool(len(orientation_gain_columns) > 0),
        "wave_orientation_gain_all_finite": orientation_gain_finite,
        "wave_orientation_gain_min": (
            float(np.min(orientation_gain)) if orientation_gain_finite else float("nan")
        ),
        "wave_orientation_gain_max": (
            float(np.max(orientation_gain)) if orientation_gain_finite else float("nan")
        ),
        "wave_orientation_gain_mean": (
            float(np.mean(orientation_gain)) if orientation_gain_finite else float("nan")
        ),
        "wave_orientation_gain_note": (
            "1.0 means high wave-normal encounter; lower values attenuate wave forcing "
            "when perpendicular vessel velocity is low."
        ),
        "wave_envelope_gate_enabled": bool(model.cfg.get("wave_envelope_gate_enabled", True)),
        "wave_envelope_gate_all_finite": envelope_gate_finite,
        "wave_envelope_gate_min": (
            float(np.min(envelope_gate)) if envelope_gate_finite else float("nan")
        ),
        "wave_envelope_gate_max": (
            float(np.max(envelope_gate)) if envelope_gate_finite else float("nan")
        ),
        "wave_envelope_gate_mean": (
            float(np.mean(envelope_gate)) if envelope_gate_finite else float("nan")
        ),
        "pure_wave_force_all_finite": pure_wave_finite,
        "pure_wave_force_nonzero": wave_force_nonzero,
        "pure_wave_force_rms_scaled": pure_wave_rms_value,
        "wave_signal_force_rms_scaled": wave_signal_force_rms,
        "wave_slope_force_rms_scaled": wave_slope_force_rms,
        "wave_auxiliary_force_rms_scaled": wave_auxiliary_force_rms,
        "wave_shape_correction_enabled": bool(model.cfg.get("wave_shape_correction_enabled", False)),
        "wave_shape_correction_gain": float(model.cfg.get("wave_shape_correction_gain", 0.0)),
        "wave_shape_correction_force_rms_scaled": wave_shape_correction_force_rms,
        "pure_wave_force_peak_abs_scaled": (
            float(np.max(np.abs(pure_wave_force))) if pure_wave_finite else float("nan")
        ),
        "counterfactual_samples": int(len(counterfactual_deltas)),
        "counterfactual_wave_effect_nonzero": counterfactual_wave_effect_nonzero,
        "counterfactual_force_delta_rms_scaled": counterfactual_delta_rms,
        "counterfactual_note": (
            "At sampled recursive forecast steps, only scaled wave channels were "
            "set to zero; this RMS is the resulting change in total forecast force."
        ),
        "total_force_all_finite": total_force_finite,
        "total_force_rms_scaled": total_force_rms_value,
        "pure_wave_to_total_force_rms_ratio": (
            pure_wave_rms_value / max(total_force_rms_value, 1.0e-12)
            if pure_wave_finite and total_force_finite
            else float("nan")
        ),
        "force_path_note": (
            "At every recursive forecast step, the unchanged synchronised wave "
            "features enter model(...); its gated wave-force output is included in "
            "forecast_force_scaled before damping and restoring terms are applied."
        ),
    }
    return verification, pure_wave_force, wave_input_rms, orientation_gain, envelope_gate


@torch.no_grad()
def forecast_roll_region(model: PINNLSTM, data: Dict[str, object],
                         cfg: Dict[str, object],
                         precomputed_exogenous_force: Optional[np.ndarray] = None) -> Dict[str, object]:
    model.eval()
    t = np.asarray(data["t"], dtype=float)
    prepared_region = data.get("forecast_region")
    region = tuple(int(v) for v in prepared_region) if prepared_region is not None else choose_forecast_region(t, cfg)
    if region is None:
        return {"enabled": False, "reason": "No valid forecast region."}
    start_idx, end_idx = region
    if not bool(cfg.get("forecast_use_ode", True)):
        direct_forecast = forecast_direct_roll_region(
            model,
            data,
            cfg,
            region=(start_idx, end_idx),
        )
        if not bool(direct_forecast.get("enabled", False)):
            return direct_forecast
        direct_summary = copy.deepcopy(direct_forecast)
        direct_forecast["direct_no_ode_forecast"] = direct_summary
        return direct_forecast
    X_work = np.asarray(data["X"], dtype=np.float32).copy()
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float)
    pred_phi_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_v_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_force_s = np.full(len(t), np.nan, dtype=np.float64)
    raw_forecast_force_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_phi_s[:start_idx + 1] = measured_phi_s[:start_idx + 1]
    pred_v_s[:start_idx + 1] = measured_v_s[:start_idx + 1]

    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    forecast_force_lag_s = float(cfg.get("forecast_force_lag_s", 0.0))
    delay_steps = list(data.get("motion_delay_steps_list", motion_feedback_delay_steps(cfg, dt_med)))
    motion_indices = data.get("motion_feature_indices", {})
    feedback_active = bool(
        isinstance(motion_indices, dict)
        and (motion_indices.get("phi_fb", []) or motion_indices.get("v_fb", []))
    )
    if not feedback_active:
        if precomputed_exogenous_force is None:
            precomputed_exogenous_force = precompute_full_ode_exogenous_force(
                model,
                X_work,
                data,
                cfg,
                normalise_segments(data.get("run_segments"), len(t)),
            )
        else:
            precomputed_exogenous_force = np.asarray(precomputed_exogenous_force, dtype=np.float64)
            if len(precomputed_exogenous_force) != len(t):
                raise ValueError("Precomputed exogenous force length must match the forecast timeline.")
        raw_forecast_force_s[:] = precomputed_exogenous_force
        precomputed_exogenous_force = lag_force_by_segments(
            t,
            precomputed_exogenous_force,
            normalise_segments(data.get("run_segments"), len(t)),
            forecast_force_lag_s,
        )
    c1 = float(model.c_roll.detach().cpu().item())
    c2 = float(model.c_quad.detach().cpu().item())
    k1 = float(model.k_roll.detach().cpu().item())

    for j in range(start_idx, end_idx):
        if feedback_active:
            _set_forecast_motion_feedback(X_work, j, start_idx, delay_steps, pred_phi_s, pred_v_s, data)
            raw_force_j = _forecast_force_at(model, X_work, data, cfg, j)
            raw_forecast_force_s[j] = raw_force_j
            run_lo, _ = segment_for_index(data.get("run_segments"), len(t), j)
            force_j = lagged_force_value_from_history(
                t,
                raw_forecast_force_s,
                int(run_lo),
                j,
                forecast_force_lag_s,
            )
        else:
            force_j = float(precomputed_exogenous_force[j])
        pred_force_s[j] = force_j
        dt = max(1.0e-8, float(t[j + 1] - t[j]))
        phi_j = float(pred_phi_s[j])
        v_j = float(pred_v_s[j])
        accel_j = force_j - c1 * v_j - c2 * abs(v_j) * v_j - k1 * phi_j
        pred_v_s[j + 1] = v_j + dt * accel_j
        pred_phi_s[j + 1] = phi_j + dt * pred_v_s[j + 1]
        if feedback_active:
            _set_forecast_motion_feedback(X_work, j + 1, start_idx, delay_steps, pred_phi_s, pred_v_s, data)
    if feedback_active:
        raw_forecast_force_s[end_idx] = _forecast_force_at(model, X_work, data, cfg, end_idx)
    pred_force_s[end_idx] = (
        lagged_force_value_from_history(
            t,
            raw_forecast_force_s,
            int(segment_for_index(data.get("run_segments"), len(t), end_idx)[0]),
            end_idx,
            forecast_force_lag_s,
        )
        if feedback_active
        else float(precomputed_exogenous_force[end_idx])
    )

    idx = np.arange(start_idx, end_idx + 1, dtype=int)
    measured_deg = np.rad2deg(np.asarray(data["phi"], dtype=float)[idx])
    forecast_deg = np.rad2deg(pred_phi_s[idx] * float(data["phi_std"]))
    forecast_metrics = regression_metrics(measured_deg, forecast_deg) if len(idx) > 2 else {}
    if len(idx) > 2:
        forecast_metrics.update(phase_metrics(measured_deg, forecast_deg, t[idx]))
    uncorrected_forecast_deg = forecast_deg.copy()
    uncorrected_phi_s = pred_phi_s[idx].copy()
    uncorrected_v_s = pred_v_s[idx].copy()
    uncorrected_force_s = pred_force_s[idx].copy()
    uncorrected_metrics = copy.deepcopy(forecast_metrics)
    post_correction: Dict[str, object] = {
        "enabled": False,
        "reason": "forecast_force_post_correction_enabled is false.",
    }
    if bool(cfg.get("forecast_force_post_correction_enabled", False)) and len(idx) > 2:
        post_gain = float(cfg.get("forecast_force_post_correction_gain", 1.0))
        post_lag_s = float(cfg.get("forecast_force_post_correction_lag_s", 0.0))
        replay = replay_forecast_state_with_force(
            t,
            measured_phi_s,
            measured_v_s,
            float(data["phi_std"]),
            measured_deg,
            idx,
            t[idx],
            uncorrected_force_s,
            c1,
            c2,
            k1,
            post_gain,
            post_lag_s,
        )
        corrected_phi_s = np.asarray(replay.get("phi_scaled", []), dtype=float)
        corrected_v_s = np.asarray(replay.get("v_scaled", []), dtype=float)
        corrected_force_s = np.asarray(replay.get("force_scaled", []), dtype=float)
        corrected_deg = np.asarray(replay.get("roll_deg", []), dtype=float)
        corrected_metrics = replay.get("metrics", {})
        if (
            len(corrected_phi_s) == len(idx)
            and len(corrected_v_s) == len(idx)
            and len(corrected_force_s) == len(idx)
            and len(corrected_deg) == len(idx)
            and isinstance(corrected_metrics, dict)
            and np.all(np.isfinite(corrected_force_s))
        ):
            pred_phi_s[idx] = corrected_phi_s
            pred_v_s[idx] = corrected_v_s
            pred_force_s[idx] = corrected_force_s
            forecast_deg = corrected_deg
            forecast_metrics = corrected_metrics
            post_correction = {
                "enabled": True,
                "lag_s": post_lag_s,
                "gain": post_gain,
                "note": (
                    "Diagnostic-only post correction: the learned forecast force "
                    "is shifted/scaled and the ODE is replayed from the same "
                    "measured handoff state. This is not yet a learned training path."
                ),
                "baseline_metrics": uncorrected_metrics,
                "corrected_metrics": corrected_metrics,
            }
        else:
            post_correction = {
                "enabled": False,
                "reason": "Corrected replay returned invalid arrays.",
                "lag_s": post_lag_s,
                "gain": post_gain,
            }
    wave_force_verification, pure_wave_force_s, wave_input_rms_s, orientation_gain_s, envelope_gate_s = (
        forecast_wave_force_verification(
            model,
            X_work,
            data,
            idx,
            uncorrected_force_s,
        )
    )
    direct_no_ode_forecast = forecast_direct_roll_region(
        model,
        data,
        cfg,
        region=(start_idx, end_idx),
    )
    handoff_phi_error_scaled = float(pred_phi_s[start_idx] - measured_phi_s[start_idx])
    handoff_v_error_scaled = float(pred_v_s[start_idx] - measured_v_s[start_idx])
    return {
        "enabled": True,
        "method": "ode_recursive_force",
        "forecast_primary": "ode_recursive_force",
        "forecast_ode_enabled": True,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_time_s": float(t[start_idx]),
        "end_time_s": float(t[end_idx]),
        "duration_s": float(t[end_idx] - t[start_idx]),
        "delay_steps": int(delay_steps[-1]) if delay_steps else 0,
        "feedback_delay_steps": [int(v) for v in delay_steps],
        "uses_predicted_motion_feedback": bool(len(data.get("motion_feature_indices", {}).get("phi_fb", [])) > 0),
        "training_feedback_rollout_window_s": (
            float(cfg.get("motion_feedback_rollout_window_s", cfg.get("rollout_window_s", 5.08)))
            if feedback_active else None
        ),
        "training_feedback_rollout_batch_size": (
            max(1, int(cfg.get("motion_feedback_rollout_batch_size", cfg.get("batch_size", 1))))
            if feedback_active else None
        ),
        "known_inputs_note": "The synchronised beam-sea wave, vessel-state, and wave-orientation gain channels remain in the model input tensor through every forecast step; roll labels are hidden.",
        "roll_dynamics_note": "Forecast starts exactly from measured roll/rate at the changeover sample, then advances recursively using total learned force (including the gated wave force) and c1/c2/k1 dynamics.",
        "direct_no_ode_forecast": direct_no_ode_forecast,
        "wave_force_verification": wave_force_verification,
        "force_post_correction": post_correction,
        "forecast_force_lag_s": forecast_force_lag_s,
        "forecast_force_lag_note": (
            "Positive forecast_force_lag_s delays the learned force before it is "
            "used by the recursive forecast ODE."
        ),
        "handoff_phi_error_scaled": handoff_phi_error_scaled,
        "handoff_phi_error_deg": float(np.rad2deg(handoff_phi_error_scaled * float(data["phi_std"]))),
        "handoff_v_error_scaled": handoff_v_error_scaled,
        "index": idx,
        "time_s": t[idx],
        "measured_roll_deg": measured_deg,
        "forecast_roll_deg": forecast_deg,
        "forecast_phi_scaled": pred_phi_s[idx],
        "forecast_v_scaled": pred_v_s[idx],
        "forecast_force_scaled": pred_force_s[idx],
        "forecast_raw_force_before_lag_scaled": raw_forecast_force_s[idx],
        "uncorrected_forecast_roll_deg": uncorrected_forecast_deg,
        "uncorrected_forecast_phi_scaled": uncorrected_phi_s,
        "uncorrected_forecast_v_scaled": uncorrected_v_s,
        "uncorrected_forecast_force_scaled": uncorrected_force_s,
        "uncorrected_metrics": uncorrected_metrics,
        "forecast_pure_wave_force_scaled": pure_wave_force_s,
        "forecast_wave_input_rms_scaled": wave_input_rms_s,
        "forecast_wave_orientation_gain": orientation_gain_s,
        "forecast_wave_envelope_gate": envelope_gate_s,
        "metrics": forecast_metrics,
    }


def replay_forecast_with_force(t: np.ndarray,
                               measured_phi_s: np.ndarray,
                               measured_v_s: np.ndarray,
                               phi_std: float,
                               measured_deg: np.ndarray,
                               forecast_idx: np.ndarray,
                               force_time: np.ndarray,
                               force_scaled: np.ndarray,
                               c1: float,
                               c2: float,
                               k1: float,
                               gain: float,
                               lag_s: float) -> Tuple[np.ndarray, Dict[str, float]]:
    replay = replay_forecast_state_with_force(
        t,
        measured_phi_s,
        measured_v_s,
        phi_std,
        measured_deg,
        forecast_idx,
        force_time,
        force_scaled,
        c1,
        c2,
        k1,
        gain,
        lag_s,
    )
    return (
        np.asarray(replay.get("roll_deg", []), dtype=float),
        replay.get("metrics", {}) if isinstance(replay.get("metrics", {}), dict) else {},
    )


def replay_forecast_state_with_force(t: np.ndarray,
                                     measured_phi_s: np.ndarray,
                                     measured_v_s: np.ndarray,
                                     phi_std: float,
                                     measured_deg: np.ndarray,
                                     forecast_idx: np.ndarray,
                                     force_time: np.ndarray,
                                     force_scaled: np.ndarray,
                                     c1: float,
                                     c2: float,
                                     k1: float,
                                     gain: float,
                                     lag_s: float) -> Dict[str, object]:
    idx = np.asarray(forecast_idx, dtype=int).reshape(-1)
    local_t = np.asarray(t, dtype=float)[idx]
    force = np.asarray(force_scaled, dtype=float).reshape(-1)
    f_time = np.asarray(force_time, dtype=float).reshape(-1)
    pred_phi_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_v_s = np.full(len(t), np.nan, dtype=np.float64)
    start_idx = int(idx[0])
    end_idx = int(idx[-1])
    pred_phi_s[start_idx] = float(measured_phi_s[start_idx])
    pred_v_s[start_idx] = float(measured_v_s[start_idx])
    finite = np.isfinite(f_time) & np.isfinite(force)
    if int(np.sum(finite)) < 2:
        return {
            "force_scaled": np.full(len(idx), np.nan, dtype=np.float64),
            "phi_scaled": np.full(len(idx), np.nan, dtype=np.float64),
            "v_scaled": np.full(len(idx), np.nan, dtype=np.float64),
            "roll_deg": np.full(len(idx), np.nan, dtype=np.float64),
            "metrics": {
                "rmse_deg": float("nan"),
                "phase_corr": float("nan"),
                "amplitude_ratio": float("nan"),
                "r2": float("nan"),
            },
        }
    # Positive lag_s delays the learned force; negative lag_s advances it.
    shifted_force = float(gain) * np.interp(
        local_t - float(lag_s),
        f_time[finite],
        force[finite],
        left=float(force[finite][0]),
        right=float(force[finite][-1]),
    )
    local_force = {int(i): float(f) for i, f in zip(idx, shifted_force)}
    for j in range(start_idx, end_idx):
        dt = max(1.0e-8, float(t[j + 1] - t[j]))
        phi_j = float(pred_phi_s[j])
        v_j = float(pred_v_s[j])
        force_j = float(local_force.get(j, shifted_force[-1]))
        accel_j = force_j - c1 * v_j - c2 * abs(v_j) * v_j - k1 * phi_j
        pred_v_s[j + 1] = v_j + dt * accel_j
        pred_phi_s[j + 1] = phi_j + dt * pred_v_s[j + 1]
    replay_deg = np.rad2deg(pred_phi_s[idx] * float(phi_std))
    metrics = regression_metrics(measured_deg, replay_deg) if len(idx) > 2 else {}
    if len(idx) > 2:
        metrics.update(phase_metrics(measured_deg, replay_deg, local_t))
    return {
        "force_scaled": shifted_force,
        "phi_scaled": pred_phi_s[idx],
        "v_scaled": pred_v_s[idx],
        "roll_deg": replay_deg,
        "metrics": metrics,
    }


def forecast_force_replay_sweep(model: PINNLSTM,
                                data: Dict[str, object],
                                forecast: Dict[str, object],
                                cfg: Dict[str, object]) -> Dict[str, object]:
    """Sweep lag/gain corrections to the learned forecast force and replay the ODE."""
    if not bool(forecast.get("enabled", False)):
        return {"enabled": False, "reason": "Forecast unavailable."}
    if not bool(forecast.get("forecast_ode_enabled", True)):
        return {
            "enabled": False,
            "reason": "ODE rolling forecast is disabled; force replay sweep is not applicable.",
        }
    idx = np.asarray(forecast.get("index", []), dtype=int).reshape(-1)
    if len(idx) < 5:
        return {"enabled": False, "reason": "Forecast window is too short."}
    local_t = np.asarray(forecast.get("time_s", []), dtype=float).reshape(-1)
    learned_force = np.asarray(
        forecast.get(
            "uncorrected_forecast_force_scaled",
            forecast.get("forecast_force_scaled", np.full(len(idx), np.nan)),
        ),
        dtype=float,
    ).reshape(-1)
    measured_deg = np.asarray(
        forecast.get("measured_roll_deg", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    if len(local_t) != len(idx) or len(learned_force) != len(idx) or len(measured_deg) != len(idx):
        return {"enabled": False, "reason": "Forecast arrays have inconsistent lengths."}
    if int(np.sum(np.isfinite(learned_force))) < 2 or not np.all(np.isfinite(measured_deg)):
        return {"enabled": False, "reason": "Forecast force or measured roll is unavailable."}

    lag_min = float(cfg.get("forecast_force_replay_lag_min_s", -1.0))
    lag_max = float(cfg.get("forecast_force_replay_lag_max_s", 1.0))
    lag_n = int(max(3, cfg.get("forecast_force_replay_lag_steps", 41)))
    gain_min = float(cfg.get("forecast_force_replay_gain_min", 0.4))
    gain_max = float(cfg.get("forecast_force_replay_gain_max", 1.4))
    gain_n = int(max(3, cfg.get("forecast_force_replay_gain_steps", 41)))
    lags = np.linspace(lag_min, lag_max, lag_n, dtype=np.float64)
    gains = np.linspace(gain_min, gain_max, gain_n, dtype=np.float64)

    t = np.asarray(data["t"], dtype=float).reshape(-1)
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float).reshape(-1)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float).reshape(-1)
    c1 = float(model.c_roll.detach().cpu().item())
    c2 = float(model.c_quad.detach().cpu().item())
    k1 = float(model.k_roll.detach().cpu().item())
    phi_std = float(data["phi_std"])
    rmse = np.full((len(gains), len(lags)), np.nan, dtype=np.float64)
    corr = np.full_like(rmse, np.nan)
    amp = np.full_like(rmse, np.nan)
    r2 = np.full_like(rmse, np.nan)
    best_metric: Dict[str, float] = {
        "rmse_deg": float("inf"),
        "lag_s": float("nan"),
        "gain": float("nan"),
        "phase_corr": float("nan"),
        "amplitude_ratio": float("nan"),
        "r2": float("nan"),
    }
    best_replay = np.full(len(idx), np.nan, dtype=np.float64)
    for gi, gain in enumerate(gains):
        for li, lag_s in enumerate(lags):
            replay_deg, replay_metrics = replay_forecast_with_force(
                t,
                measured_phi_s,
                measured_v_s,
                phi_std,
                measured_deg,
                idx,
                local_t,
                learned_force,
                c1,
                c2,
                k1,
                float(gain),
                float(lag_s),
            )
            rmse_val = float(replay_metrics.get("rmse_deg", float("nan")))
            rmse[gi, li] = rmse_val
            corr[gi, li] = float(replay_metrics.get("phase_corr", float("nan")))
            amp[gi, li] = float(replay_metrics.get("amplitude_ratio", float("nan")))
            r2[gi, li] = float(replay_metrics.get("r2", float("nan")))
            if math.isfinite(rmse_val) and rmse_val < float(best_metric["rmse_deg"]):
                best_metric = {
                    "rmse_deg": rmse_val,
                    "lag_s": float(lag_s),
                    "gain": float(gain),
                    "phase_corr": float(replay_metrics.get("phase_corr", float("nan"))),
                    "amplitude_ratio": float(replay_metrics.get("amplitude_ratio", float("nan"))),
                    "r2": float(replay_metrics.get("r2", float("nan"))),
                }
                best_replay = replay_deg

    baseline = forecast.get("uncorrected_metrics", forecast.get("metrics", {}))
    if not isinstance(baseline, dict):
        baseline = {}
    return {
        "enabled": True,
        "note": (
            "Diagnostic only: the learned forecast force is shifted/scaled, then "
            "the same c1/c2/k1 roll ODE is replayed from the measured handoff state. "
            "Positive force_lag_s delays the learned force; negative force_lag_s advances it."
        ),
        "lags_s": lags,
        "gains": gains,
        "rmse_deg": rmse,
        "phase_corr": corr,
        "amplitude_ratio": amp,
        "r2": r2,
        "best_by_rmse": best_metric,
        "baseline": {
            "rmse_deg": float(baseline.get("rmse_deg", float("nan"))),
            "phase_corr": float(baseline.get("phase_corr", float("nan"))),
            "amplitude_ratio": float(baseline.get("amplitude_ratio", float("nan"))),
            "r2": float(baseline.get("r2", float("nan"))),
        },
        "time_s": local_t,
        "measured_roll_deg": measured_deg,
        "best_replay_roll_deg": best_replay,
    }


def forecast_coefficient_replay_sweep(model: PINNLSTM,
                                      data: Dict[str, object],
                                      forecast: Dict[str, object],
                                      cfg: Dict[str, object]) -> Dict[str, object]:
    """Sweep ODE coefficient multipliers while keeping learned forecast force fixed."""
    if not bool(forecast.get("enabled", False)):
        return {"enabled": False, "reason": "Forecast unavailable."}
    if not bool(forecast.get("forecast_ode_enabled", True)):
        return {
            "enabled": False,
            "reason": "ODE rolling forecast is disabled; coefficient replay sweep is not applicable.",
        }
    idx = np.asarray(forecast.get("index", []), dtype=int).reshape(-1)
    if len(idx) < 5:
        return {"enabled": False, "reason": "Forecast window is too short."}
    local_t = np.asarray(forecast.get("time_s", []), dtype=float).reshape(-1)
    learned_force = np.asarray(
        forecast.get("forecast_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    measured_deg = np.asarray(
        forecast.get("measured_roll_deg", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    if len(local_t) != len(idx) or len(learned_force) != len(idx) or len(measured_deg) != len(idx):
        return {"enabled": False, "reason": "Forecast arrays have inconsistent lengths."}
    if int(np.sum(np.isfinite(learned_force))) < 2 or not np.all(np.isfinite(measured_deg)):
        return {"enabled": False, "reason": "Forecast force or measured roll is unavailable."}

    mult_min = float(cfg.get("forecast_coefficient_replay_multiplier_min", 0.25))
    mult_max = float(cfg.get("forecast_coefficient_replay_multiplier_max", 7.50))
    mult_n = int(max(3, cfg.get("forecast_coefficient_replay_multiplier_steps", 46)))
    if not math.isfinite(mult_min) or not math.isfinite(mult_max) or mult_max <= mult_min:
        mult_min, mult_max = 0.25, 7.50
    multipliers = np.linspace(mult_min, mult_max, mult_n, dtype=np.float64)

    t = np.asarray(data["t"], dtype=float).reshape(-1)
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float).reshape(-1)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float).reshape(-1)
    phi_std = float(data["phi_std"])
    base_coeffs = {
        "c_roll": float(model.c_roll.detach().cpu().item()),
        "c_quad": float(model.c_quad.detach().cpu().item()),
        "k_roll": float(model.k_roll.detach().cpu().item()),
    }
    baseline = forecast.get("metrics", {})
    if not isinstance(baseline, dict):
        baseline = {}

    series: Dict[str, Dict[str, object]] = {}
    selection_metric = str(cfg.get("forecast_coefficient_replay_selection_metric", "r2")).strip().lower()
    if selection_metric not in {"r2", "rmse"}:
        selection_metric = "r2"
    best_by_rmse: Dict[str, object] = {
        "rmse_deg": float("inf"),
        "coefficient": "",
        "multiplier": float("nan"),
        "phase_corr": float("nan"),
        "amplitude_ratio": float("nan"),
        "r2": float("nan"),
        "c_roll": base_coeffs["c_roll"],
        "c_quad": base_coeffs["c_quad"],
        "k_roll": base_coeffs["k_roll"],
    }
    best_by_r2: Dict[str, object] = {
        "r2": -float("inf"),
        "coefficient": "",
        "multiplier": float("nan"),
        "rmse_deg": float("nan"),
        "phase_corr": float("nan"),
        "amplitude_ratio": float("nan"),
        "c_roll": base_coeffs["c_roll"],
        "c_quad": base_coeffs["c_quad"],
        "k_roll": base_coeffs["k_roll"],
    }
    best_replay = np.full(len(idx), np.nan, dtype=np.float64)
    best_r2_replay = np.full(len(idx), np.nan, dtype=np.float64)
    best_rmse_replay = np.full(len(idx), np.nan, dtype=np.float64)
    for coeff_name in ("c_roll", "c_quad", "k_roll"):
        rmse_vals: List[float] = []
        corr_vals: List[float] = []
        amp_vals: List[float] = []
        r2_vals: List[float] = []
        for multiplier in multipliers:
            coeffs = dict(base_coeffs)
            coeffs[coeff_name] = float(base_coeffs[coeff_name] * multiplier)
            replay_deg, replay_metrics = replay_forecast_with_force(
                t,
                measured_phi_s,
                measured_v_s,
                phi_std,
                measured_deg,
                idx,
                local_t,
                learned_force,
                coeffs["c_roll"],
                coeffs["c_quad"],
                coeffs["k_roll"],
                1.0,
                0.0,
            )
            rmse_val = float(replay_metrics.get("rmse_deg", float("nan")))
            corr_val = float(replay_metrics.get("phase_corr", float("nan")))
            amp_val = float(replay_metrics.get("amplitude_ratio", float("nan")))
            r2_val = float(replay_metrics.get("r2", float("nan")))
            rmse_vals.append(rmse_val)
            corr_vals.append(corr_val)
            amp_vals.append(amp_val)
            r2_vals.append(r2_val)
            if math.isfinite(rmse_val) and rmse_val < float(best_by_rmse["rmse_deg"]):
                best_by_rmse = {
                    "rmse_deg": rmse_val,
                    "coefficient": coeff_name,
                    "multiplier": float(multiplier),
                    "phase_corr": corr_val,
                    "amplitude_ratio": amp_val,
                    "r2": r2_val,
                    "c_roll": coeffs["c_roll"],
                    "c_quad": coeffs["c_quad"],
                    "k_roll": coeffs["k_roll"],
                }
                best_rmse_replay = replay_deg
            if math.isfinite(r2_val) and (
                r2_val > float(best_by_r2["r2"])
                or (
                    math.isclose(r2_val, float(best_by_r2["r2"]), rel_tol=0.0, abs_tol=1.0e-12)
                    and math.isfinite(rmse_val)
                    and rmse_val < float(best_by_r2.get("rmse_deg", float("inf")))
                )
            ):
                best_by_r2 = {
                    "r2": r2_val,
                    "coefficient": coeff_name,
                    "multiplier": float(multiplier),
                    "rmse_deg": rmse_val,
                    "phase_corr": corr_val,
                    "amplitude_ratio": amp_val,
                    "c_roll": coeffs["c_roll"],
                    "c_quad": coeffs["c_quad"],
                    "k_roll": coeffs["k_roll"],
                }
                best_r2_replay = replay_deg
        series[coeff_name] = {
            "rmse_deg": rmse_vals,
            "phase_corr": corr_vals,
            "amplitude_ratio": amp_vals,
            "r2": r2_vals,
        }
    selected_metric = best_by_r2 if selection_metric == "r2" else best_by_rmse
    selected_replay = best_r2_replay if selection_metric == "r2" else best_rmse_replay
    best_replay = selected_replay

    return {
        "enabled": True,
        "note": (
            "Diagnostic only: the learned forecast force is held fixed and the "
            "forecast ODE is replayed from the same measured handoff state while "
            "one coefficient multiplier is swept at a time. A best multiplier far "
            "from 1.0 indicates the frozen/source coefficient may be biasing the ODE."
        ),
        "multipliers": multipliers,
        "base_coefficients": base_coeffs,
        "series": series,
        "selection_metric": selection_metric,
        "best_selected": selected_metric,
        "best_by_r2": best_by_r2,
        "best_by_rmse": best_by_rmse,
        "baseline": {
            "rmse_deg": float(baseline.get("rmse_deg", float("nan"))),
            "phase_corr": float(baseline.get("phase_corr", float("nan"))),
            "amplitude_ratio": float(baseline.get("amplitude_ratio", float("nan"))),
            "r2": float(baseline.get("r2", float("nan"))),
        },
        "time_s": local_t,
        "measured_roll_deg": measured_deg,
        "baseline_roll_deg": np.asarray(forecast.get("forecast_roll_deg", np.full(len(idx), np.nan)), dtype=float),
        "best_replay_roll_deg": best_replay,
        "best_r2_replay_roll_deg": best_r2_replay,
        "best_rmse_replay_roll_deg": best_rmse_replay,
    }


def forecast_orientation_gate_ablation(model: PINNLSTM,
                                       data: Dict[str, object],
                                       forecast: Dict[str, object],
                                       cfg: Dict[str, object]) -> Dict[str, object]:
    """Replay forecast forces while replacing only the wave-orientation multiplier."""
    if not bool(forecast.get("enabled", False)):
        return {"enabled": False, "reason": "Forecast unavailable."}
    if not bool(forecast.get("forecast_ode_enabled", True)):
        return {
            "enabled": False,
            "reason": "ODE rolling forecast is disabled; orientation-gate replay is not applicable.",
        }
    idx = np.asarray(forecast.get("index", []), dtype=int).reshape(-1)
    if len(idx) < 5:
        return {"enabled": False, "reason": "Forecast window is too short."}
    local_t = np.asarray(forecast.get("time_s", []), dtype=float).reshape(-1)
    measured_deg = np.asarray(
        forecast.get("measured_roll_deg", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    raw_force = np.asarray(
        forecast.get(
            "uncorrected_forecast_force_scaled",
            forecast.get("forecast_force_scaled", np.full(len(idx), np.nan)),
        ),
        dtype=float,
    ).reshape(-1)
    pure_force = np.asarray(
        forecast.get("forecast_pure_wave_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    inferred_force = np.asarray(
        forecast.get("inferred_measured_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    orientation_gain = np.asarray(
        forecast.get("forecast_wave_orientation_gain", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    arrays = [local_t, measured_deg, raw_force, pure_force, inferred_force, orientation_gain]
    if any(len(v) != len(idx) for v in arrays):
        return {"enabled": False, "reason": "Forecast arrays have inconsistent lengths."}
    if int(np.sum(np.isfinite(raw_force) & np.isfinite(pure_force) & np.isfinite(orientation_gain))) < 3:
        return {"enabled": False, "reason": "Forecast force or orientation gain is unavailable."}

    t = np.asarray(data["t"], dtype=float).reshape(-1)
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float).reshape(-1)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float).reshape(-1)
    phi_std = float(data["phi_std"])
    c1 = float(model.c_roll.detach().cpu().item())
    c2 = float(model.c_quad.detach().cpu().item())
    k1 = float(model.k_roll.detach().cpu().item())

    gate_floor = 1.0e-4
    gate_clean = np.where(
        np.isfinite(orientation_gain),
        np.clip(orientation_gain, gate_floor, None),
        1.0,
    )
    gate_is_active = bool(cfg.get("wave_orientation_gate_enabled", True))
    pure_wave_base = pure_force / np.maximum(gate_clean, gate_floor) if gate_is_active else pure_force
    non_pure_force = raw_force - pure_force
    finite_gate = np.isfinite(local_t) & np.isfinite(gate_clean)
    if int(np.sum(finite_gate)) < 2:
        return {"enabled": False, "reason": "Orientation gain has too few finite samples."}

    def shifted_gate(lag_s: float) -> np.ndarray:
        return np.interp(
            local_t - float(lag_s),
            local_t[finite_gate],
            gate_clean[finite_gate],
            left=float(gate_clean[finite_gate][0]),
            right=float(gate_clean[finite_gate][-1]),
        )

    def candidate_force(strength: float, lag_s: float) -> Tuple[np.ndarray, np.ndarray]:
        gate = shifted_gate(lag_s)
        # strength=0 removes the hard gate, strength=1 applies the configured
        # physical gate, and strength<0 tests inverted timing.
        candidate_gate = np.clip(1.0 + float(strength) * (gate - 1.0), 0.0, 2.0)
        force = non_pure_force + pure_wave_base * candidate_gate
        return force, candidate_gate

    def force_envelope(values: np.ndarray) -> np.ndarray:
        clean = np.where(np.isfinite(values), values, 0.0)
        return rolling_rms_time_by_segments(
            clean,
            local_t,
            float(cfg.get("force_envelope_window_s", 1.2)),
            [(0, len(local_t) - 1)],
        )

    inferred_env = force_envelope(inferred_force)

    def zero_lag_corr(reference: np.ndarray, candidate: np.ndarray) -> float:
        ref = np.asarray(reference, dtype=float)
        cand = np.asarray(candidate, dtype=float)
        finite = np.isfinite(ref) & np.isfinite(cand)
        if int(np.sum(finite)) < 3:
            return float("nan")
        ref_c = ref[finite] - float(np.mean(ref[finite]))
        cand_c = cand[finite] - float(np.mean(cand[finite]))
        denom = math.sqrt(float(np.sum(ref_c ** 2) * np.sum(cand_c ** 2)))
        if denom <= 1.0e-12:
            return float("nan")
        return float(np.sum(ref_c * cand_c) / denom)

    def replay_variant(name: str, strength: float, lag_s: float) -> Dict[str, object]:
        force, gate = candidate_force(strength, lag_s)
        replay = replay_forecast_state_with_force(
            t,
            measured_phi_s,
            measured_v_s,
            phi_std,
            measured_deg,
            idx,
            local_t,
            force,
            c1,
            c2,
            k1,
            1.0,
            0.0,
        )
        roll = np.asarray(replay.get("roll_deg", np.full(len(idx), np.nan)), dtype=float)
        env = force_envelope(force)
        metrics = replay.get("metrics", {}) if isinstance(replay.get("metrics", {}), dict) else {}
        return {
            "name": name,
            "strength": float(strength),
            "lag_s": float(lag_s),
            "force_scaled": force,
            "gate": gate,
            "roll_deg": roll,
            "force_envelope": env,
            "metrics": {
                **{str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))},
                "force_envelope_corr": zero_lag_corr(inferred_env, env),
                "force_envelope_amplitude_ratio": (
                    float(np.sqrt(np.nanmean(env ** 2)) / max(np.sqrt(np.nanmean(inferred_env ** 2)), 1.0e-12))
                    if np.any(np.isfinite(env)) and np.any(np.isfinite(inferred_env))
                    else float("nan")
                ),
            },
        }

    lag_min = float(cfg.get("forecast_orientation_ablation_lag_min_s", -2.0))
    lag_max = float(cfg.get("forecast_orientation_ablation_lag_max_s", 2.0))
    lag_n = int(max(3, cfg.get("forecast_orientation_ablation_lag_steps", 41)))
    strength_min = float(cfg.get("forecast_orientation_ablation_strength_min", -1.0))
    strength_max = float(cfg.get("forecast_orientation_ablation_strength_max", 1.5))
    strength_n = int(max(3, cfg.get("forecast_orientation_ablation_strength_steps", 51)))
    lags = np.linspace(lag_min, lag_max, lag_n, dtype=np.float64)
    strengths = np.linspace(strength_min, strength_max, strength_n, dtype=np.float64)
    rmse = np.full((len(strengths), len(lags)), np.nan, dtype=np.float64)
    corr = np.full_like(rmse, np.nan)
    amp = np.full_like(rmse, np.nan)
    env_corr = np.full_like(rmse, np.nan)
    best_by_rmse: Dict[str, float] = {
        "rmse_deg": float("inf"),
        "strength": float("nan"),
        "lag_s": float("nan"),
        "phase_corr": float("nan"),
        "amplitude_ratio": float("nan"),
        "force_envelope_corr": float("nan"),
    }
    best_by_envelope: Dict[str, float] = {
        "force_envelope_corr": -float("inf"),
        "strength": float("nan"),
        "lag_s": float("nan"),
        "rmse_deg": float("nan"),
        "phase_corr": float("nan"),
        "amplitude_ratio": float("nan"),
    }
    for si, strength in enumerate(strengths):
        for li, lag_s in enumerate(lags):
            variant = replay_variant("grid", float(strength), float(lag_s))
            metric = variant["metrics"]
            rmse_val = float(metric.get("rmse_deg", float("nan")))
            env_corr_val = float(metric.get("force_envelope_corr", float("nan")))
            rmse[si, li] = rmse_val
            corr[si, li] = float(metric.get("phase_corr", float("nan")))
            amp[si, li] = float(metric.get("amplitude_ratio", float("nan")))
            env_corr[si, li] = env_corr_val
            if math.isfinite(rmse_val) and rmse_val < float(best_by_rmse["rmse_deg"]):
                best_by_rmse = {
                    "rmse_deg": rmse_val,
                    "strength": float(strength),
                    "lag_s": float(lag_s),
                    "phase_corr": float(metric.get("phase_corr", float("nan"))),
                    "amplitude_ratio": float(metric.get("amplitude_ratio", float("nan"))),
                    "force_envelope_corr": env_corr_val,
                }
            if math.isfinite(env_corr_val) and env_corr_val > float(best_by_envelope["force_envelope_corr"]):
                best_by_envelope = {
                    "force_envelope_corr": env_corr_val,
                    "strength": float(strength),
                    "lag_s": float(lag_s),
                    "rmse_deg": rmse_val,
                    "phase_corr": float(metric.get("phase_corr", float("nan"))),
                    "amplitude_ratio": float(metric.get("amplitude_ratio", float("nan"))),
                }

    current_strength = 1.0 if gate_is_active else 0.0
    variants = [
        replay_variant("current model", current_strength, 0.0),
        replay_variant("no gate", 0.0, 0.0),
        replay_variant("soft gate", 0.5, 0.0),
        replay_variant("current gate", 1.0, 0.0),
        replay_variant("inverted diagnostic", -1.0, 0.0),
    ]
    if math.isfinite(float(best_by_rmse["strength"])) and math.isfinite(float(best_by_rmse["lag_s"])):
        variants.append(replay_variant("best RMSE gate", float(best_by_rmse["strength"]), float(best_by_rmse["lag_s"])))

    return {
        "enabled": True,
        "note": (
            "Diagnostic only: total forecast force is decomposed into non-pure "
            "force plus the learned pure-wave force. Only the pure-wave "
            "orientation multiplier is replaced before replaying the same roll ODE. "
            "strength=0 removes the hard gate; strength=1 applies the "
            "configured physical gate; negative strength tests inverted timing."
        ),
        "lags_s": lags,
        "strengths": strengths,
        "rmse_deg": rmse,
        "phase_corr": corr,
        "amplitude_ratio": amp,
        "force_envelope_corr": env_corr,
        "best_by_rmse": best_by_rmse,
        "best_by_envelope": best_by_envelope,
        "current_gate_strength": float(current_strength),
        "time_s": local_t,
        "measured_roll_deg": measured_deg,
        "inferred_force_envelope": inferred_env,
        "orientation_gain": gate_clean,
        "variants": variants,
        "metrics": {
            "best_by_rmse": best_by_rmse,
            "best_by_envelope": best_by_envelope,
            "named_variants": {
                str(v["name"]): v.get("metrics", {})
                for v in variants
                if isinstance(v, dict)
            },
        },
    }


def oracle_force_replay_forecast(model: PINNLSTM,
                                 data: Dict[str, object],
                                 forecast: Dict[str, object]) -> Dict[str, object]:
    """Replay the forecast ODE with force inferred from measured roll."""
    if not bool(forecast.get("enabled", False)):
        return {"enabled": False, "reason": "Forecast unavailable."}
    if not bool(forecast.get("forecast_ode_enabled", True)):
        return {
            "enabled": False,
            "reason": "ODE rolling forecast is disabled; oracle-force ODE replay is not applicable.",
        }
    idx = np.asarray(forecast.get("index", []), dtype=int).reshape(-1)
    if len(idx) < 3:
        return {"enabled": False, "reason": "Forecast window is too short."}

    t = np.asarray(data["t"], dtype=float).reshape(-1)
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float).reshape(-1)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float).reshape(-1)
    inferred_force = np.asarray(
        forecast.get("inferred_measured_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    if len(inferred_force) != len(idx):
        return {"enabled": False, "reason": "Inferred force length does not match forecast window."}

    finite_force = np.isfinite(inferred_force)
    if not np.any(finite_force):
        return {"enabled": False, "reason": "Inferred force is unavailable in the forecast window."}
    if not np.all(finite_force):
        local_t = t[idx]
        if int(np.sum(finite_force)) == 1:
            inferred_force = np.full_like(inferred_force, float(inferred_force[finite_force][0]))
        else:
            inferred_force = np.interp(
                local_t,
                local_t[finite_force],
                inferred_force[finite_force],
            )

    start_idx = int(idx[0])
    end_idx = int(idx[-1])
    pred_phi_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_v_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_phi_s[start_idx] = float(measured_phi_s[start_idx])
    pred_v_s[start_idx] = float(measured_v_s[start_idx])
    c1 = float(model.c_roll.detach().cpu().item())
    c2 = float(model.c_quad.detach().cpu().item())
    k1 = float(model.k_roll.detach().cpu().item())
    force_lookup = {int(i): float(f) for i, f in zip(idx, inferred_force)}
    for j in range(start_idx, end_idx):
        if j not in force_lookup:
            continue
        dt = max(1.0e-8, float(t[j + 1] - t[j]))
        phi_j = float(pred_phi_s[j])
        v_j = float(pred_v_s[j])
        force_j = float(force_lookup[j])
        accel_j = force_j - c1 * v_j - c2 * abs(v_j) * v_j - k1 * phi_j
        pred_v_s[j + 1] = v_j + dt * accel_j
        pred_phi_s[j + 1] = phi_j + dt * pred_v_s[j + 1]

    measured_deg = np.asarray(forecast.get("measured_roll_deg", []), dtype=float).reshape(-1)
    oracle_deg = np.rad2deg(pred_phi_s[idx] * float(data["phi_std"]))
    metrics = regression_metrics(measured_deg, oracle_deg) if len(idx) > 2 else {}
    if len(idx) > 2:
        metrics.update(phase_metrics(measured_deg, oracle_deg, t[idx]))
    learned_force = np.asarray(
        forecast.get("forecast_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    if len(learned_force) != len(idx):
        fixed = np.full(len(idx), np.nan, dtype=np.float64)
        fixed[:min(len(idx), len(learned_force))] = learned_force[:min(len(idx), len(learned_force))]
        learned_force = fixed
    return {
        "enabled": True,
        "note": (
            "Diagnostic replay only: the forecast ODE is advanced from the same "
            "measured handoff state, but force is replaced by force inferred "
            "from measured roll/rate over the hidden forecast window."
        ),
        "index": idx,
        "time_s": t[idx],
        "oracle_roll_deg": oracle_deg,
        "oracle_phi_scaled": pred_phi_s[idx],
        "oracle_v_scaled": pred_v_s[idx],
        "oracle_force_scaled": inferred_force,
        "metrics": metrics,
        "learned_force_vs_oracle_force": alignment_metrics(
            inferred_force,
            learned_force,
            t[idx],
            "scaled",
        ),
    }


@torch.no_grad()
def teacher_forced_force_alignment_diagnostic(model: PINNLSTM,
                                              data: Dict[str, object],
                                              cfg: Dict[str, object],
                                              force_without_turn_s: np.ndarray,
                                              turn_s: np.ndarray) -> Dict[str, object]:
    """Compare inferred measured force with teacher-forced learned force."""
    inferred = measured_force_implied_by_roll(model, data, cfg)
    total = np.asarray(force_without_turn_s, dtype=float).reshape(-1) + np.asarray(turn_s, dtype=float).reshape(-1)
    X_tensor = torch.as_tensor(
        np.asarray(data["X"], dtype=np.float32),
        dtype=torch.float32,
        device=data["device"],
    )
    pure = model.pure_wave_forcing(X_tensor).detach().cpu().numpy().astype(np.float64).reshape(-1)
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    n = int(min(len(t), len(inferred), len(total), len(pure)))
    t = t[:n]
    inferred = inferred[:n]
    total = total[:n]
    pure = pure[:n]

    center_period_s = None
    try:
        center_period_s = float(data.get("dominant_roll_period_s"))
    except (TypeError, ValueError):
        center_period_s = None

    def split_metrics(raw_idx: object) -> Dict[str, Dict[str, float]]:
        idx = np.asarray(raw_idx, dtype=int).reshape(-1)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) < 5:
            return {
                "total_force": alignment_metrics([], [], [], "scaled"),
                "pure_wave_force": alignment_metrics([], [], [], "scaled"),
                "spectral_total_force": force_spectral_metrics([], [], [], cfg, center_period_s),
                "spectral_pure_wave_force": force_spectral_metrics([], [], [], cfg, center_period_s),
            }
        return {
            "total_force": alignment_metrics(inferred[idx], total[idx], t[idx], "scaled"),
            "pure_wave_force": alignment_metrics(inferred[idx], pure[idx], t[idx], "scaled"),
            "spectral_total_force": force_spectral_metrics(inferred[idx], total[idx], t[idx], cfg, center_period_s),
            "spectral_pure_wave_force": force_spectral_metrics(inferred[idx], pure[idx], t[idx], cfg, center_period_s),
        }

    return {
        "note": (
            "Teacher-forced force diagnostic: model force is evaluated with measured "
            "history/inputs, not recursively forecast motion. Total force includes "
            "turn moment; pure wave force is the explicit measured-wave coupling path."
        ),
        "time_s": t,
        "inferred_measured_force_scaled": inferred,
        "teacher_forced_total_force_scaled": total,
        "teacher_forced_pure_wave_force_scaled": pure,
        "metrics": {
            "train": split_metrics(data.get("train_idx", [])),
            "validation": split_metrics(data.get("val_idx", [])),
        },
        "spectral_band_center_period_s": center_period_s,
    }


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def local_extrema_error_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(target) < 3 or len(pred) != len(target):
        return {
            "extrema_point_rmse_deg": float("nan"),
            "extrema_point_mae_deg": float("nan"),
            "peak_underfit_deg": float("nan"),
            "peak_underfit_max_deg": float("nan"),
            "trough_overshoot_deg": float("nan"),
            "trough_overshoot_max_deg": float("nan"),
        }
    mid = target[1:-1]
    is_peak = (mid >= target[:-2]) & (mid >= target[2:])
    is_trough = (mid <= target[:-2]) & (mid <= target[2:])
    peak_idx = np.flatnonzero(is_peak & (mid > 0.0)) + 1
    trough_idx = np.flatnonzero(is_trough & (mid < 0.0)) + 1
    extrema_idx = np.sort(np.concatenate([peak_idx, trough_idx])).astype(int)
    if extrema_idx.size == 0:
        peak_idx = np.asarray([int(np.argmax(target))], dtype=int) if float(np.max(target)) > 0.0 else np.asarray([], dtype=int)
        trough_idx = np.asarray([int(np.argmin(target))], dtype=int) if float(np.min(target)) < 0.0 else np.asarray([], dtype=int)
        extrema_idx = np.sort(np.concatenate([peak_idx, trough_idx])).astype(int)
    if extrema_idx.size > 0:
        err = pred[extrema_idx] - target[extrema_idx]
        extrema_rmse = float(np.sqrt(np.mean(err ** 2)))
        extrema_mae = float(np.mean(np.abs(err)))
    else:
        extrema_rmse = float("nan")
        extrema_mae = float("nan")
    if peak_idx.size > 0:
        peak_under = np.maximum(target[peak_idx] - pred[peak_idx], 0.0)
        peak_under_mean = float(np.mean(peak_under))
        peak_under_max = float(np.max(peak_under))
    else:
        peak_under_mean = float("nan")
        peak_under_max = float("nan")
    if trough_idx.size > 0:
        trough_over = np.maximum(target[trough_idx] - pred[trough_idx], 0.0)
        trough_over_mean = float(np.mean(trough_over))
        trough_over_max = float(np.max(trough_over))
    else:
        trough_over_mean = float("nan")
        trough_over_max = float("nan")
    return {
        "extrema_point_rmse_deg": extrema_rmse,
        "extrema_point_mae_deg": extrema_mae,
        "peak_underfit_deg": peak_under_mean,
        "peak_underfit_max_deg": peak_under_max,
        "trough_overshoot_deg": trough_over_mean,
        "trough_overshoot_max_deg": trough_over_max,
    }


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    target = np.asarray(y, dtype=float)
    pred = np.asarray(yhat, dtype=float)
    err = pred - target
    target_peak = float(np.max(target))
    pred_peak = float(np.max(pred))
    target_trough = float(np.min(target))
    pred_trough = float(np.min(pred))
    peak_error = pred_peak - target_peak
    trough_error = pred_trough - target_trough
    target_range = max(target_peak - target_trough, 1.0e-12)
    pred_range = pred_peak - pred_trough
    metrics = {
        "r2": r2_score(y, yhat),
        "measured_rms_deg": float(np.sqrt(np.mean(target ** 2))),
        "fit_rms_deg": float(np.sqrt(np.mean(pred ** 2))),
        "error_rms_deg": float(np.sqrt(np.mean(err ** 2))),
        "rmse_deg": float(np.sqrt(np.mean(err ** 2))),
        "mae_deg": float(np.mean(np.abs(err))),
        "max_abs_error_deg": float(np.max(np.abs(err))),
        "target_peak_abs_deg": float(np.max(np.abs(y))),
        "pred_peak_abs_deg": float(np.max(np.abs(yhat))),
        "peak_abs_error_deg": float(abs(np.max(np.abs(yhat)) - np.max(np.abs(y)))),
        "target_peak_deg": target_peak,
        "pred_peak_deg": pred_peak,
        "target_trough_deg": target_trough,
        "pred_trough_deg": pred_trough,
        "signed_peak_error_deg": float(peak_error),
        "signed_trough_error_deg": float(trough_error),
        "extrema_rmse_deg": float(math.sqrt(0.5 * (peak_error ** 2 + trough_error ** 2))),
        "extrema_relative_error": float(
            0.5 * (abs(peak_error) + abs(trough_error)) / target_range
        ),
        "extrema_range_ratio": float(pred_range / target_range),
    }
    metrics.update(local_extrema_error_metrics(target, pred))
    return metrics


def phase_metrics(y: np.ndarray, yhat: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    tt = np.asarray(t, dtype=float)
    if len(y) < 5 or len(yhat) != len(y) or len(tt) != len(y):
        return {
            "phase_lag_steps": 0.0,
            "phase_lag_s": 0.0,
            "phase_corr": float("nan"),
            "phase_aligned_rmse_deg": float("nan"),
            "amplitude_ratio": float("nan"),
            "amplitude_underfit_ratio": float("nan"),
        }

    y0 = y - np.mean(y)
    p0 = yhat - np.mean(yhat)
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 1.0
    max_lag_steps = int(min(max(1, round(0.35 * len(y))), max(1, round(0.5 / max(dt, 1.0e-8)))))
    best_corr = -float("inf")
    best_lag = 0
    best_target = y0
    best_pred = p0
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            target_seg = y0[-lag:]
            pred_seg = p0[:lag]
        elif lag > 0:
            target_seg = y0[:-lag]
            pred_seg = p0[lag:]
        else:
            target_seg = y0
            pred_seg = p0
        if len(target_seg) < 3:
            continue
        denom = float(np.linalg.norm(target_seg) * np.linalg.norm(pred_seg))
        corr = 0.0 if denom < 1.0e-12 else float(np.dot(target_seg, pred_seg) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
            best_target = target_seg
            best_pred = pred_seg

    aligned_err = best_pred - best_target
    target_rms = float(np.sqrt(np.mean(y ** 2)))
    pred_rms = float(np.sqrt(np.mean(yhat ** 2)))
    amp_ratio = pred_rms / max(target_rms, 1.0e-12)
    return {
        "phase_lag_steps": float(best_lag),
        "phase_lag_s": float(best_lag * dt),
        "phase_corr": float(best_corr),
        "phase_aligned_rmse_deg": float(np.sqrt(np.mean(aligned_err ** 2))),
        "amplitude_ratio": float(amp_ratio),
        "amplitude_underfit_ratio": float(max(0.0, 1.0 - amp_ratio)),
    }


def alignment_metrics(reference: np.ndarray, candidate: np.ndarray, t: np.ndarray,
                      value_suffix: str = "scaled") -> Dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(candidate, dtype=float)
    tt = np.asarray(t, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(pred) & np.isfinite(tt)
    if int(np.sum(mask)) < 5:
        return {
            "phase_lag_steps": 0.0,
            "phase_lag_s": 0.0,
            "phase_corr": float("nan"),
            f"phase_aligned_rmse_{value_suffix}": float("nan"),
            "amplitude_ratio": float("nan"),
            "reference_rms_scaled": float("nan"),
            "candidate_rms_scaled": float("nan"),
            f"rmse_{value_suffix}": float("nan"),
        }

    ref = ref[mask]
    pred = pred[mask]
    tt = tt[mask]
    ref0 = ref - np.mean(ref)
    pred0 = pred - np.mean(pred)
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 1.0
    max_lag_steps = int(min(max(1, round(0.35 * len(ref))), max(1, round(2.0 / max(dt, 1.0e-8)))))
    best_corr = -float("inf")
    best_lag = 0
    best_ref = ref0
    best_pred = pred0
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            ref_seg = ref0[-lag:]
            pred_seg = pred0[:lag]
        elif lag > 0:
            ref_seg = ref0[:-lag]
            pred_seg = pred0[lag:]
        else:
            ref_seg = ref0
            pred_seg = pred0
        if len(ref_seg) < 3:
            continue
        denom = float(np.linalg.norm(ref_seg) * np.linalg.norm(pred_seg))
        corr = 0.0 if denom < 1.0e-12 else float(np.dot(ref_seg, pred_seg) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
            best_ref = ref_seg
            best_pred = pred_seg

    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    pred_rms = float(np.sqrt(np.mean(pred ** 2)))
    err = pred - ref
    aligned_err = best_pred - best_ref
    return {
        "phase_lag_steps": float(best_lag),
        "phase_lag_s": float(best_lag * dt),
        "phase_corr": float(best_corr),
        f"phase_aligned_rmse_{value_suffix}": float(np.sqrt(np.mean(aligned_err ** 2))),
        "amplitude_ratio": float(pred_rms / max(ref_rms, 1.0e-12)),
        "reference_rms_scaled": ref_rms,
        "candidate_rms_scaled": pred_rms,
        f"rmse_{value_suffix}": float(np.sqrt(np.mean(err ** 2))),
    }


def one_sided_amplitude_spectrum(values: np.ndarray,
                                 t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float).reshape(-1)
    tt = np.asarray(t, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(tt)
    y = y[mask]
    tt = tt[mask]
    if len(y) < 8:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    order = np.argsort(tt)
    y = y[order] - float(np.mean(y[order]))
    tt = tt[order]
    dt = float(np.median(np.diff(tt))) if len(tt) > 1 else 1.0
    if not math.isfinite(dt) or dt <= 0.0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    window = np.hanning(len(y))
    if float(np.sum(window)) <= 1.0e-12:
        window = np.ones_like(y)
    spec = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(len(y), d=max(dt, 1.0e-8))
    amplitude = 2.0 * np.abs(spec) / max(float(np.sum(window)), 1.0e-12)
    power = np.abs(spec) ** 2
    if len(amplitude) > 0:
        amplitude[0] = 0.0
        power[0] = 0.0
    return freqs.astype(np.float64), amplitude.astype(np.float64), power.astype(np.float64)


def force_spectral_metrics(reference: np.ndarray,
                           candidate: np.ndarray,
                           t: np.ndarray,
                           cfg: Dict[str, object],
                           center_period_s: Optional[float]) -> Dict[str, float]:
    ref_freq, ref_amp, ref_power = one_sided_amplitude_spectrum(reference, t)
    cand_freq, cand_amp, cand_power = one_sided_amplitude_spectrum(candidate, t)
    if len(ref_freq) < 2 or len(cand_freq) < 2:
        return {
            "reference_dominant_frequency_hz": float("nan"),
            "reference_dominant_period_s": float("nan"),
            "candidate_dominant_frequency_hz": float("nan"),
            "candidate_dominant_period_s": float("nan"),
            "dominant_frequency_error_hz": float("nan"),
            "dominant_period_error_s": float("nan"),
            "band_low_hz": float("nan"),
            "band_high_hz": float("nan"),
            "reference_band_energy": float("nan"),
            "candidate_band_energy": float("nan"),
            "band_energy_ratio": float("nan"),
            "total_energy_ratio": float("nan"),
        }

    ref_dom_idx = int(np.argmax(ref_power[1:]) + 1)
    cand_dom_idx = int(np.argmax(cand_power[1:]) + 1)
    ref_dom_hz = float(ref_freq[ref_dom_idx])
    cand_dom_hz = float(cand_freq[cand_dom_idx])
    period = (
        float(center_period_s)
        if center_period_s is not None and math.isfinite(float(center_period_s)) and float(center_period_s) > 0.0
        else 1.0 / max(ref_dom_hz, 1.0e-8)
    )
    low_factor = float(cfg.get("bandpass_low_period_factor", 1.8))
    high_factor = float(cfg.get("bandpass_high_period_factor", 0.35))
    band_low_hz = 1.0 / max(period * low_factor, 1.0e-8)
    band_high_hz = 1.0 / max(period * high_factor, 1.0e-8)
    if band_low_hz > band_high_hz:
        band_low_hz, band_high_hz = band_high_hz, band_low_hz
    ref_band = (ref_freq >= band_low_hz) & (ref_freq <= band_high_hz)
    cand_band = (cand_freq >= band_low_hz) & (cand_freq <= band_high_hz)
    ref_band_energy = float(np.sum(ref_power[ref_band])) if np.any(ref_band) else float("nan")
    cand_band_energy = float(np.sum(cand_power[cand_band])) if np.any(cand_band) else float("nan")
    ref_total = float(np.sum(ref_power))
    cand_total = float(np.sum(cand_power))
    return {
        "reference_dominant_frequency_hz": ref_dom_hz,
        "reference_dominant_period_s": float(1.0 / max(ref_dom_hz, 1.0e-8)),
        "candidate_dominant_frequency_hz": cand_dom_hz,
        "candidate_dominant_period_s": float(1.0 / max(cand_dom_hz, 1.0e-8)),
        "dominant_frequency_error_hz": float(cand_dom_hz - ref_dom_hz),
        "dominant_period_error_s": float((1.0 / max(cand_dom_hz, 1.0e-8)) - (1.0 / max(ref_dom_hz, 1.0e-8))),
        "band_low_hz": float(band_low_hz),
        "band_high_hz": float(band_high_hz),
        "reference_band_energy": ref_band_energy,
        "candidate_band_energy": cand_band_energy,
        "band_energy_ratio": float(cand_band_energy / max(ref_band_energy, 1.0e-12)) if math.isfinite(ref_band_energy) and math.isfinite(cand_band_energy) else float("nan"),
        "total_energy_ratio": float(cand_total / max(ref_total, 1.0e-12)),
    }


@torch.no_grad()
def measured_force_implied_by_roll(model: PINNLSTM,
                                   data: Dict[str, object],
                                   cfg: Dict[str, object]) -> np.ndarray:
    t = np.asarray(data["t"], dtype=float).reshape(-1)
    phi = np.asarray(data["phi_scaled"], dtype=float).reshape(-1)
    v = np.asarray(data["v_est_scaled"], dtype=float).reshape(-1)
    X_np = np.asarray(data["X"], dtype=np.float32)
    n_points = int(min(t.size, phi.size, v.size, X_np.shape[0]))
    target = np.full(n_points, np.nan, dtype=np.float64)
    if n_points < 3:
        return target
    t = t[:n_points]
    phi = phi[:n_points]
    v = v[:n_points]
    X_np = X_np[:n_points]

    run_segments = normalise_segments(data.get("run_segments"), n_points)
    dvdt = np.full(n_points, np.nan, dtype=np.float64)
    for lo, hi in run_segments:
        lo_i = max(0, int(lo))
        hi_i = min(n_points - 1, int(hi))
        if hi_i < lo_i:
            continue
        sl = slice(lo_i, hi_i + 1)
        if hi_i - lo_i + 1 < 3:
            continue
        edge_order = 2 if hi_i - lo_i + 1 > 3 else 1
        dvdt[sl] = np.gradient(v[sl], t[sl], edge_order=edge_order)

    X_tensor = torch.as_tensor(
        X_np,
        dtype=torch.float32,
        device=data["device"],
    )
    turn_raw = model.turn_moment(X_tensor).detach().cpu().numpy().astype(np.float64)
    turn = np.full(n_points, 0.0, dtype=np.float64)
    turn_flat = np.asarray(turn_raw, dtype=np.float64).reshape(-1)
    turn[:min(n_points, turn_flat.size)] = turn_flat[:min(n_points, turn_flat.size)]
    c1 = float(model.c_roll.detach().cpu().item())
    c2 = float(model.c_quad.detach().cpu().item())
    k1 = float(model.k_roll.detach().cpu().item())
    target = dvdt + c1 * v + c2 * np.abs(v) * v + k1 * phi - turn
    smoothing_s = float(cfg.get("wave_force_target_smoothing_seconds", 0.0))
    if smoothing_s > 0.0:
        for lo, hi in run_segments:
            lo_i = max(0, int(lo))
            hi_i = min(n_points - 1, int(hi))
            if hi_i < lo_i:
                continue
            sl = slice(lo_i, hi_i + 1)
            if hi_i - lo_i + 1 >= 3:
                target[sl] = moving_average_time(target[sl], t[sl], smoothing_s)
    return target


def add_force_alignment_diagnostic(model: PINNLSTM,
                                   data: Dict[str, object],
                                   cfg: Dict[str, object],
                                   forecast: Dict[str, object]) -> Dict[str, object]:
    if not bool(forecast.get("enabled", False)):
        return forecast
    idx = np.asarray(forecast.get("index", []), dtype=int).reshape(-1)
    t_f = np.asarray(forecast.get("time_s", []), dtype=float).reshape(-1)
    if len(idx) == 0 or len(t_f) != len(idx):
        return forecast

    inferred_all = measured_force_implied_by_roll(model, data, cfg)
    inferred = np.full(len(idx), np.nan, dtype=np.float64)
    valid_idx = (idx >= 0) & (idx < len(inferred_all))
    if np.any(valid_idx):
        inferred[valid_idx] = inferred_all[idx[valid_idx]]
    pure = np.asarray(
        forecast.get("forecast_pure_wave_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    total = np.asarray(
        forecast.get("forecast_force_scaled", np.full(len(idx), np.nan)),
        dtype=float,
    ).reshape(-1)
    if len(pure) != len(idx):
        pure_fixed = np.full(len(idx), np.nan, dtype=np.float64)
        pure_fixed[:min(len(idx), len(pure))] = pure[:min(len(idx), len(pure))]
        pure = pure_fixed
    if len(total) != len(idx):
        total_fixed = np.full(len(idx), np.nan, dtype=np.float64)
        total_fixed[:min(len(idx), len(total))] = total[:min(len(idx), len(total))]
        total = total_fixed
    non_pure = total - pure
    roll_metrics = phase_metrics(
        np.asarray(forecast.get("measured_roll_deg", []), dtype=float),
        np.asarray(forecast.get("forecast_roll_deg", []), dtype=float),
        t_f,
    )
    diagnostic = {
        "note": (
            "Inferred measured force is dv/dt + c_roll*v + c_quad*|v|*v + "
            "k_roll*phi - turn_moment, using measured roll/rate and current "
            "learned coefficients in scaled coordinates."
        ),
        "smoothing_seconds": float(cfg.get("wave_force_target_smoothing_seconds", 0.0)),
        "roll": roll_metrics,
        "pure_wave_force": alignment_metrics(inferred, pure, t_f, "scaled"),
        "total_force": alignment_metrics(inferred, total, t_f, "scaled"),
        "non_pure_force_rms_scaled": float(
            np.sqrt(np.nanmean(non_pure ** 2)) if np.any(np.isfinite(non_pure)) else float("nan")
        ),
    }
    forecast["inferred_measured_force_scaled"] = inferred
    forecast["forecast_non_pure_force_scaled"] = non_pure
    forecast["force_alignment"] = diagnostic
    return forecast


def forecast_force_envelope_diagnostic(forecast: Dict[str, object],
                                       cfg: Dict[str, object]) -> Dict[str, object]:
    if not bool(forecast.get("enabled", False)):
        return {"enabled": False, "reason": "Forecast unavailable."}
    t_f = np.asarray(forecast.get("time_s", []), dtype=float).reshape(-1)
    if len(t_f) < 5:
        return {"enabled": False, "reason": "Forecast window too short for envelope diagnostic."}
    inferred = np.asarray(
        forecast.get("inferred_measured_force_scaled", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)
    raw_force = np.asarray(
        forecast.get("uncorrected_forecast_force_scaled", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)
    corrected_force = np.asarray(
        forecast.get("forecast_force_scaled", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)
    pure_force = np.asarray(
        forecast.get("forecast_pure_wave_force_scaled", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)
    wave_input_rms = np.asarray(
        forecast.get("forecast_wave_input_rms_scaled", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)
    orientation_gain = np.asarray(
        forecast.get("forecast_wave_orientation_gain", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)
    envelope_gate = np.asarray(
        forecast.get("forecast_wave_envelope_gate", np.full(len(t_f), np.nan)),
        dtype=float,
    ).reshape(-1)

    def fixed(values: np.ndarray) -> np.ndarray:
        out = np.full(len(t_f), np.nan, dtype=np.float64)
        n = min(len(out), len(values))
        if n > 0:
            out[:n] = values[:n]
        return out

    inferred = fixed(inferred)
    raw_force = fixed(raw_force)
    corrected_force = fixed(corrected_force)
    pure_force = fixed(pure_force)
    wave_input_rms = fixed(wave_input_rms)
    orientation_gain = fixed(orientation_gain)
    envelope_gate = fixed(envelope_gate)
    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    segments = [(0, len(t_f) - 1)]

    def env(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0)
        return rolling_rms_time_by_segments(clean, t_f, window_s, segments)

    inferred_env = env(inferred)
    raw_env = env(raw_force)
    corrected_env = env(corrected_force)
    pure_env = env(pure_force)
    geometry_proxy = wave_input_rms * orientation_gain
    geometry_env = env(geometry_proxy)
    finite_gain = np.isfinite(inferred_env) & np.isfinite(raw_env) & (raw_env > 1.0e-8)
    required_gain = np.full(len(t_f), np.nan, dtype=np.float64)
    required_gain[finite_gain] = inferred_env[finite_gain] / np.maximum(raw_env[finite_gain], 1.0e-8)
    finite_gate_compare = np.isfinite(required_gain) & np.isfinite(envelope_gate)
    learned_gate_required_corr = (
        float(np.corrcoef(required_gain[finite_gate_compare], envelope_gate[finite_gate_compare])[0, 1])
        if int(np.sum(finite_gate_compare)) > 2 else float("nan")
    )

    def envelope_metrics(candidate_env: np.ndarray) -> Dict[str, float]:
        metric = alignment_metrics(inferred_env, candidate_env, t_f, "scaled")
        finite = np.isfinite(inferred_env) & np.isfinite(candidate_env)
        metric["mean_absolute_envelope_error_scaled"] = (
            float(np.mean(np.abs(candidate_env[finite] - inferred_env[finite])))
            if int(np.sum(finite)) > 1 else float("nan")
        )
        return metric

    metrics = {
        "note": (
            "Force envelopes are rolling RMS curves in the configured force band. "
            "Geometry proxy is wave-channel RMS multiplied by wave_orientation_effect_gain; "
            "higher gain corresponds to stronger wave-normal/perpendicular encounter and "
            "lower gain to more nearly parallel encounter."
        ),
        "window_s": window_s,
        "raw_force_envelope": envelope_metrics(raw_env),
        "corrected_force_envelope": envelope_metrics(corrected_env),
        "pure_wave_force_envelope": envelope_metrics(pure_env),
        "geometry_proxy_envelope": envelope_metrics(geometry_env),
        "required_gain_median": (
            float(np.nanmedian(required_gain)) if np.any(np.isfinite(required_gain)) else float("nan")
        ),
        "required_gain_mean": (
            float(np.nanmean(required_gain)) if np.any(np.isfinite(required_gain)) else float("nan")
        ),
        "orientation_gain_min": (
            float(np.nanmin(orientation_gain)) if np.any(np.isfinite(orientation_gain)) else float("nan")
        ),
        "orientation_gain_max": (
            float(np.nanmax(orientation_gain)) if np.any(np.isfinite(orientation_gain)) else float("nan")
        ),
        "orientation_gain_mean": (
            float(np.nanmean(orientation_gain)) if np.any(np.isfinite(orientation_gain)) else float("nan")
        ),
        "learned_envelope_gate_min": (
            float(np.nanmin(envelope_gate)) if np.any(np.isfinite(envelope_gate)) else float("nan")
        ),
        "learned_envelope_gate_max": (
            float(np.nanmax(envelope_gate)) if np.any(np.isfinite(envelope_gate)) else float("nan")
        ),
        "learned_envelope_gate_mean": (
            float(np.nanmean(envelope_gate)) if np.any(np.isfinite(envelope_gate)) else float("nan")
        ),
        "learned_envelope_gate_required_gain_corr": learned_gate_required_corr,
    }
    return {
        "enabled": True,
        "time_s": t_f,
        "inferred_envelope": inferred_env,
        "raw_force_envelope": raw_env,
        "corrected_force_envelope": corrected_env,
        "pure_wave_force_envelope": pure_env,
        "geometry_proxy_envelope": geometry_env,
        "required_gain": required_gain,
        "orientation_gain": orientation_gain,
        "learned_envelope_gate": envelope_gate,
        "metrics": metrics,
    }


def orientation_envelope_lag_diagnostic(data: Dict[str, object],
                                        inferred_force: np.ndarray,
                                        forecast: Dict[str, object],
                                        cfg: Dict[str, object]) -> Dict[str, object]:
    t = np.asarray(data.get("t", []), dtype=float).reshape(-1)
    X = np.asarray(data.get("X", []), dtype=float)
    inferred = np.asarray(inferred_force, dtype=float).reshape(-1)
    if len(t) < 8 or X.ndim != 2 or X.shape[0] < len(t) or len(inferred) < len(t):
        return {"enabled": False, "reason": "Missing aligned vessel, feature, or inferred-force arrays."}
    n = int(min(len(t), X.shape[0], len(inferred)))
    t = t[:n]
    X = X[:n]
    inferred = inferred[:n]
    run_segments = normalise_segments(data.get("run_segments"), n)
    wave_groups = data.get("wave_feature_indices", {})
    if not isinstance(wave_groups, dict):
        wave_groups = {}
    wave_columns = sorted({
        int(column)
        for columns in wave_groups.values()
        if isinstance(columns, (list, tuple))
        for column in columns
        if 0 <= int(column) < X.shape[1]
    })
    state_groups = data.get("state_feature_indices", {})
    orientation_columns = (
        state_groups.get("wave_orientation_effect_gain", [])
        if isinstance(state_groups, dict)
        else []
    )
    orientation_columns = [
        int(column) for column in orientation_columns
        if 0 <= int(column) < X.shape[1]
    ]
    if not wave_columns:
        return {"enabled": False, "reason": "No wave input columns are available."}

    wave_input_rms = np.sqrt(np.nanmean(X[:, wave_columns] ** 2, axis=1))
    orientation_gain = (
        np.nanmean(X[:, orientation_columns], axis=1)
        if orientation_columns
        else np.ones(n, dtype=np.float64)
    )
    orientation_gain = np.where(np.isfinite(orientation_gain), orientation_gain, 1.0)
    geometry_proxy = wave_input_rms * orientation_gain
    dominant_period = float(data.get("dominant_roll_period_s", cfg.get("dominant_roll_period_s", 1.0)))
    if not math.isfinite(dominant_period) or dominant_period <= 0.0:
        dominant_period = 1.0
    force_bp = bandpass_time_by_segments(
        inferred,
        t,
        dominant_period,
        float(cfg.get("bandpass_low_period_factor", 1.8)),
        float(cfg.get("bandpass_high_period_factor", 0.35)),
        run_segments,
    )
    window_s = float(cfg.get("force_envelope_window_s", 1.2))
    inferred_env = rolling_rms_time_by_segments(force_bp, t, window_s, run_segments)
    raw_wave_env = rolling_rms_time_by_segments(wave_input_rms, t, window_s, run_segments)
    geometry_env = rolling_rms_time_by_segments(geometry_proxy, t, window_s, run_segments)

    max_lag = float(cfg.get("orientation_envelope_lag_max_s", 5.0))
    n_lags = int(max(5, cfg.get("orientation_envelope_lag_candidates", 151)))
    lags = np.linspace(-max_lag, max_lag, n_lags, dtype=np.float64)

    def shifted_by_segment(values: np.ndarray, lag_s: float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        shifted = np.full(n, np.nan, dtype=np.float64)
        for lo, hi in run_segments:
            lo_i = max(0, int(lo))
            hi_i = min(n - 1, int(hi))
            if hi_i - lo_i + 1 < 2:
                continue
            sl = slice(lo_i, hi_i + 1)
            shifted[sl] = np.interp(
                t[sl] - float(lag_s),
                t[sl],
                values[sl],
                left=float(values[lo_i]),
                right=float(values[hi_i]),
            )
        return shifted

    def corr_at_indices(reference: np.ndarray, candidate: np.ndarray, idx: np.ndarray) -> float:
        idx = np.asarray(idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        if len(idx) < 8:
            return float("nan")
        ref = np.asarray(reference, dtype=float)[idx]
        cand = np.asarray(candidate, dtype=float)[idx]
        finite = np.isfinite(ref) & np.isfinite(cand)
        if int(np.sum(finite)) < 8:
            return float("nan")
        ref_c = ref[finite] - float(np.mean(ref[finite]))
        cand_c = cand[finite] - float(np.mean(cand[finite]))
        denom = math.sqrt(float(np.sum(ref_c ** 2) * np.sum(cand_c ** 2)))
        if denom <= 1.0e-12:
            return float("nan")
        return float(np.sum(ref_c * cand_c) / denom)

    def lag_scores(candidate_env: np.ndarray, idx: np.ndarray) -> np.ndarray:
        scores = np.full(len(lags), np.nan, dtype=np.float64)
        for i, lag_s in enumerate(lags):
            scores[i] = corr_at_indices(inferred_env, shifted_by_segment(candidate_env, float(lag_s)), idx)
        return scores

    forecast_idx = np.asarray(forecast.get("index", []), dtype=int)
    split_indices = {
        "train": np.asarray(data.get("train_idx", []), dtype=int),
        "validation": np.asarray(data.get("val_idx", []), dtype=int),
        "forecast": forecast_idx,
    }
    split_metrics: Dict[str, Dict[str, object]] = {}
    for split_name, raw_idx in split_indices.items():
        idx = np.asarray(raw_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        raw_scores = lag_scores(raw_wave_env, idx)
        geometry_scores = lag_scores(geometry_env, idx)
        gate_scores = lag_scores(orientation_gain, idx)
        split_metrics[split_name] = {
            "indices": idx,
            "raw_wave_scores": raw_scores,
            "orientation_weighted_scores": geometry_scores,
            "orientation_gate_scores": gate_scores,
            "raw_wave": force_lag_scan_summary(lags, raw_scores, 0.0),
            "orientation_weighted_wave": force_lag_scan_summary(lags, geometry_scores, 0.0),
            "orientation_gate": force_lag_scan_summary(lags, gate_scores, 0.0),
        }

    return {
        "enabled": True,
        "note": (
            "Diagnostic only: compares inferred measured-force envelope with raw "
            "wave-input envelope, wave-orientation gain, and their product. "
            "Positive lag shifts the candidate envelope later relative to the "
            "inferred force envelope."
        ),
        "time_s": t,
        "lags_s": lags,
        "inferred_force_envelope": inferred_env,
        "raw_wave_envelope": raw_wave_env,
        "orientation_gain": orientation_gain,
        "orientation_weighted_wave_envelope": geometry_env,
        "window_s": window_s,
        "dominant_period_s": dominant_period,
        "splits": split_metrics,
        "metrics": {
            "note": (
                "Best lag/correlation values are reported separately for train, "
                "validation, and forecast. Consistent large best lags across all "
                "splits indicate a real encounter/envelope timing offset; a "
                "forecast-only large lag is more likely local compensation."
            ),
            "window_s": window_s,
            "dominant_period_s": dominant_period,
            "splits": {
                split_name: {
                    "raw_wave": split.get("raw_wave", {}),
                    "orientation_weighted_wave": split.get("orientation_weighted_wave", {}),
                    "orientation_gate": split.get("orientation_gate", {}),
                }
                for split_name, split in split_metrics.items()
            },
        },
    }


def resolve_full_ode_coefficients(model: PINNLSTM,
                                  cfg: Dict[str, object],
                                  coefficient_source: Optional[str] = None) -> Tuple[Dict[str, float], str]:
    source = str(
        coefficient_source
        if coefficient_source is not None
        else cfg.get("full_ode_coefficient_source", "source")
    ).strip().lower()
    if source == "source":
        raw = cfg.get("source_learned_physics_parameters", REVM_SOURCE_LEARNED_PHYSICS)
        if not isinstance(raw, dict):
            raise ValueError("source_learned_physics_parameters must be a dictionary.")
        coefficients = {
            "c_roll": float(raw["c_roll"]),
            "c_quad": float(raw["c_quad"]),
            "k_roll": float(raw["k_roll"]),
        }
        source_note = (
            f"embedded {cfg.get('source_revision', 'Rev H')} Trial "
            f"{REVM_SOURCE_TRIAL_INDEX} terminal coefficients used to seed Rev M"
        )
    elif source == "model":
        coefficients = {
            "c_roll": float(model.c_roll.detach().cpu().item()),
            "c_quad": float(model.c_quad.detach().cpu().item()),
            "k_roll": float(model.k_roll.detach().cpu().item()),
        }
        source_note = "current post-training model coefficients"
    else:
        raise ValueError("full ODE coefficient source must be 'source' or 'model'.")
    if not all(math.isfinite(v) and v >= 0.0 for v in coefficients.values()):
        raise ValueError(f"Full ODE coefficients must be finite and non-negative: {coefficients}")
    return coefficients, source_note


@torch.no_grad()
def precompute_full_ode_exogenous_force(model: PINNLSTM,
                                        X_work: np.ndarray,
                                        data: Dict[str, object],
                                        cfg: Dict[str, object],
                                        run_segments: List[Tuple[int, int]]) -> np.ndarray:
    """Evaluate the causal learned force for every sample without motion feedback."""
    model.eval()
    force = np.full(len(X_work), np.nan, dtype=np.float64)
    seq_len = int(max(2, cfg["seq_len"]))
    force_batch_size = int(max(1, int(cfg.get("inference_batch_size", 64))))
    for lo, hi in run_segments:
        lo_i, hi_i = int(lo), int(hi)
        prefix_end = min(hi_i, lo_i + seq_len - 2)
        for j in range(lo_i, prefix_end + 1):
            force[j] = _forecast_force_at(model, X_work, data, cfg, j)

        mature = np.arange(max(lo_i + seq_len - 1, lo_i), hi_i + 1, dtype=int)
        for batch_start in range(0, len(mature), force_batch_size):
            batch_idx = mature[batch_start:batch_start + force_batch_size]
            windows = np.stack(
                [X_work[int(j) - seq_len + 1:int(j) + 1] for j in batch_idx],
                axis=0,
            )
            batch = torch.as_tensor(windows, dtype=torch.float32, device=data["device"])
            _, _, force_without_turn, _ = model(batch, batch)
            total_force = force_without_turn + model.turn_moment(batch)
            force[batch_idx] = total_force[:, -1, 0].detach().cpu().numpy().astype(np.float64)
    return force


@torch.no_grad()
def integrate_full_ode_wave_train(model: PINNLSTM,
                                  data: Dict[str, object],
                                  cfg: Dict[str, object],
                                  coefficient_source: Optional[str] = None,
                                  precomputed_exogenous_force: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Integrate learned roll dynamics across every complete loaded run.

    Each independent worksheet/run is initialised once from its first measured
    roll and roll rate. No later measured motion is injected. Known exogenous
    features remain available over the complete run, matching the forecast
    assumptions used by Rev M.
    """
    model.eval()
    t = np.asarray(data["t"], dtype=float)
    measured_phi_s = np.asarray(data["phi_scaled"], dtype=float)
    measured_v_s = np.asarray(data["v_est_scaled"], dtype=float)
    X_work = np.asarray(data["X"], dtype=np.float32).copy()
    run_segments = normalise_segments(data.get("run_segments"), len(t))
    run_names = [str(v) for v in list(data.get("run_names", []))]
    if len(run_names) < len(run_segments):
        run_names.extend(f"run_{i + 1}" for i in range(len(run_names), len(run_segments)))
    run_names = run_names[:len(run_segments)]

    coefficients, coefficient_note = resolve_full_ode_coefficients(model, cfg, coefficient_source)
    c1 = float(coefficients["c_roll"])
    c2 = float(coefficients["c_quad"])
    k1 = float(coefficients["k_roll"])
    pred_phi_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_v_s = np.full(len(t), np.nan, dtype=np.float64)
    pred_force_s = np.full(len(t), np.nan, dtype=np.float64)

    dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    delay_steps = list(data.get("motion_delay_steps_list", motion_feedback_delay_steps(cfg, dt_med)))
    motion_indices = data.get("motion_feature_indices", {})
    feedback_active = bool(
        isinstance(motion_indices, dict)
        and (motion_indices.get("phi_fb", []) or motion_indices.get("v_fb", []))
    )
    if not feedback_active:
        if precomputed_exogenous_force is None:
            pred_force_s = precompute_full_ode_exogenous_force(model, X_work, data, cfg, run_segments)
        else:
            pred_force_s = np.asarray(precomputed_exogenous_force, dtype=np.float64).copy()
            if len(pred_force_s) != len(t):
                raise ValueError("Precomputed exogenous force length must match the integration timeline.")

    run_status: List[Dict[str, object]] = []
    for name, (lo, hi) in zip(run_names, run_segments):
        lo_i, hi_i = int(lo), int(hi)
        pred_phi_s[lo_i] = measured_phi_s[lo_i]
        pred_v_s[lo_i] = measured_v_s[lo_i]
        completed_to = lo_i
        for j in range(lo_i, hi_i):
            if feedback_active:
                _set_forecast_motion_feedback(X_work, j, lo_i, delay_steps, pred_phi_s, pred_v_s, data)
                pred_force_s[j] = _forecast_force_at(model, X_work, data, cfg, j)
            dt = max(1.0e-8, float(t[j + 1] - t[j]))
            phi_j = float(pred_phi_s[j])
            v_j = float(pred_v_s[j])
            force_j = float(pred_force_s[j])
            accel_j = force_j - c1 * v_j - c2 * abs(v_j) * v_j - k1 * phi_j
            pred_v_s[j + 1] = v_j + dt * accel_j
            pred_phi_s[j + 1] = phi_j + dt * pred_v_s[j + 1]
            if not math.isfinite(float(pred_phi_s[j + 1])) or not math.isfinite(float(pred_v_s[j + 1])):
                break
            completed_to = j + 1
            if feedback_active:
                _set_forecast_motion_feedback(X_work, j + 1, lo_i, delay_steps, pred_phi_s, pred_v_s, data)
        if feedback_active and completed_to == hi_i:
            pred_force_s[hi_i] = _forecast_force_at(model, X_work, data, cfg, hi_i)
        run_status.append({
            "name": name,
            "start_idx": lo_i,
            "end_idx": hi_i,
            "completed_to_idx": int(completed_to),
            "completed": bool(completed_to == hi_i),
            "initial_roll_error_scaled": float(pred_phi_s[lo_i] - measured_phi_s[lo_i]),
            "initial_rate_error_scaled": float(pred_v_s[lo_i] - measured_v_s[lo_i]),
        })

    measured_deg = np.rad2deg(np.asarray(data["phi"], dtype=float))
    integrated_deg = np.rad2deg(pred_phi_s * float(data["phi_std"]))
    valid = np.isfinite(measured_deg) & np.isfinite(integrated_deg)
    full_metrics = regression_metrics(measured_deg[valid], integrated_deg[valid]) if int(np.sum(valid)) > 2 else {}
    per_run: Dict[str, object] = {}
    for name, (lo, hi) in zip(run_names, run_segments):
        idx = np.arange(int(lo), int(hi) + 1, dtype=int)
        run_valid = valid[idx]
        if int(np.sum(run_valid)) > 2:
            per_run[name] = regression_metrics(measured_deg[idx][run_valid], integrated_deg[idx][run_valid])

    return {
        "enabled": True,
        "coefficient_source": str(coefficient_source or cfg.get("full_ode_coefficient_source", "source")),
        "coefficient_note": coefficient_note,
        "coefficients": coefficients,
        "force_source": "trained LSTM beam-sea wave coupling, wave-orientation attenuation, wave gate, and residual; turning forcing is disabled",
        "initialisation_note": "Each loaded run starts once from its first measured roll/rate; measured motion is not injected again.",
        "known_inputs_note": "The synchronised beam-sea wave and wave-orientation gain remain known for the entire integration.",
        "uses_predicted_motion_feedback": bool(feedback_active),
        "index": np.arange(len(t), dtype=int),
        "time_s": t,
        "measured_roll_deg": measured_deg,
        "integrated_roll_deg": integrated_deg,
        "integrated_phi_scaled": pred_phi_s,
        "integrated_v_scaled": pred_v_s,
        "force_scaled": pred_force_s,
        "valid_points": int(np.sum(valid)),
        "total_points": int(len(t)),
        "run_status": run_status,
        "metrics": {"full": full_metrics, "per_run": per_run},
    }


def add_forecast_uncertainty(forecast: Dict[str, object], data: Dict[str, object],
                             teacher_forced_pred_deg: np.ndarray,
                             cfg: Dict[str, object]) -> Dict[str, object]:
    """Add a widening empirical interval based on held-out teacher-forced error.

    This is an uncertainty visualisation, not a calibrated probabilistic model.
    It starts at zero at the measured forecast handoff and grows to the empirical
    validation RMSE at the end of the recursive horizon (before confidence scaling).
    """
    if not bool(cfg.get("forecast_uncertainty_enabled", True)) or not bool(forecast.get("enabled", False)):
        return forecast

    forecast_pred = np.asarray(forecast.get("forecast_roll_deg", []), dtype=float)
    forecast_time = np.asarray(forecast.get("time_s", []), dtype=float)
    measured_deg = np.rad2deg(np.asarray(data.get("phi", []), dtype=float))
    teacher_pred = np.asarray(teacher_forced_pred_deg, dtype=float)
    if len(forecast_pred) == 0 or len(forecast_time) != len(forecast_pred):
        return forecast

    source_name = "validation"
    source_idx = np.asarray(data.get("val_idx", []), dtype=int)
    if source_idx.size == 0:
        source_name = "training"
        source_idx = np.asarray(data.get("train_idx", []), dtype=int)
    valid_idx = source_idx[
        (source_idx >= 0)
        & (source_idx < len(measured_deg))
        & (source_idx < len(teacher_pred))
    ]
    if valid_idx.size:
        residual = teacher_pred[valid_idx] - measured_deg[valid_idx]
        residual = residual[np.isfinite(residual)]
    else:
        residual = np.asarray([], dtype=float)

    floor_deg = max(0.0, float(cfg.get("forecast_uncertainty_floor_deg", 0.05)))
    empirical_rmse = float(np.sqrt(np.mean(residual ** 2))) if residual.size else floor_deg
    base_sigma = max(empirical_rmse, floor_deg)
    duration = max(float(forecast_time[-1] - forecast_time[0]), 1.0e-12)
    progress = np.clip((forecast_time - forecast_time[0]) / duration, 0.0, 1.0)
    growth_power = max(1.0e-6, float(cfg.get("forecast_uncertainty_growth_power", 0.5)))
    sigma = base_sigma * np.power(progress, growth_power)
    confidence = float(np.clip(float(cfg.get("forecast_uncertainty_confidence", 0.95)), 0.50, 0.999))
    z_value = float(NormalDist().inv_cdf(0.5 + 0.5 * confidence))
    half_width = z_value * sigma

    forecast["uncertainty_std_deg"] = sigma
    forecast["uncertainty_lower_deg"] = forecast_pred - half_width
    forecast["uncertainty_upper_deg"] = forecast_pred + half_width
    forecast["uncertainty"] = {
        "method": "empirical_validation_rmse_with_sqrt_horizon_growth",
        "calibrated_probabilistic_interval": False,
        "source_split": source_name,
        "source_points": int(valid_idx.size),
        "source_teacher_forced_rmse_deg": float(empirical_rmse),
        "confidence": confidence,
        "z_value": z_value,
        "growth_power": growth_power,
        "end_standard_deviation_deg": float(sigma[-1]),
        "end_half_width_deg": float(half_width[-1]),
    }
    return forecast


def add_full_ode_forecast_uncertainty(integration: Dict[str, object],
                                      forecast: Dict[str, object],
                                      cfg: Dict[str, object]) -> Dict[str, object]:
    """Apply the forecast uncertainty width to the full-ODE curve after handoff.

    The uncertainty calibration remains unchanged: it comes from held-out
    teacher-forced residuals and grows from zero at the configured forecast
    start. Only the centre curve changes from the short rolling forecast to the
    uninterrupted full-wave ODE integration.
    """
    if not bool(integration.get("enabled", False)) or not bool(forecast.get("enabled", False)):
        return integration
    full_pred = np.asarray(integration.get("integrated_roll_deg", []), dtype=float)
    forecast_idx = np.asarray(forecast.get("index", []), dtype=int)
    forecast_std = np.asarray(forecast.get("uncertainty_std_deg", []), dtype=float)
    if len(full_pred) == 0 or len(forecast_idx) == 0 or len(forecast_std) != len(forecast_idx):
        return integration
    valid = (
        (forecast_idx >= 0)
        & (forecast_idx < len(full_pred))
        & np.isfinite(forecast_std)
    )
    if not np.any(valid):
        return integration
    idx = forecast_idx[valid]
    std = forecast_std[valid]
    confidence = float(np.clip(float(cfg.get("forecast_uncertainty_confidence", 0.95)), 0.50, 0.999))
    z_value = float(NormalDist().inv_cdf(0.5 + 0.5 * confidence))
    half_width = z_value * std
    lower_full = np.full(len(full_pred), np.nan, dtype=np.float64)
    upper_full = np.full(len(full_pred), np.nan, dtype=np.float64)
    std_full = np.full(len(full_pred), np.nan, dtype=np.float64)
    lower_full[idx] = full_pred[idx] - half_width
    upper_full[idx] = full_pred[idx] + half_width
    std_full[idx] = std
    integration["uncertainty_std_deg"] = std_full
    integration["uncertainty_lower_deg"] = lower_full
    integration["uncertainty_upper_deg"] = upper_full
    integration["uncertainty_forecast_index"] = idx
    integration["uncertainty"] = {
        **copy.deepcopy(forecast.get("uncertainty", {})),
        "center_curve": "uninterrupted_full_ode_integration",
        "start_idx": int(idx[0]),
        "end_idx": int(idx[-1]),
        "start_time_s": float(np.asarray(integration["time_s"], dtype=float)[idx[0]]),
        "end_time_s": float(np.asarray(integration["time_s"], dtype=float)[idx[-1]]),
        "confidence": confidence,
        "z_value": z_value,
    }
    return integration


def format_fit_metrics(metrics: Dict[str, object], key: str = "full") -> str:
    m = metrics.get(key, {})
    if not isinstance(m, dict):
        return ""
    return (
        f"R2 = {float(m.get('r2', float('nan'))):.4f}\n"
        f"RMS data = {float(m.get('measured_rms_deg', float('nan'))):.3f} deg\n"
        f"RMS fit = {float(m.get('fit_rms_deg', float('nan'))):.3f} deg\n"
        f"RMS error = {float(m.get('error_rms_deg', float('nan'))):.3f} deg"
    )


def format_fit_forecast_plot_metrics(fit_metrics: Dict[str, object],
                                     forecast_metrics: Optional[Dict[str, object]] = None,
                                     direct_metrics: Optional[Dict[str, object]] = None,
                                     forecast_label: str = "ODE forecast",
                                     direct_label: str = "No-ODE direct") -> str:
    fit_text = format_fit_metrics({"fit": fit_metrics}, "fit")
    if not forecast_metrics:
        return f"Fit\n{fit_text}"
    text = (
        f"Fit (forecast excluded)\n{fit_text}\n\n"
        f"{forecast_label}\n"
        f"R2 = {float(forecast_metrics.get('r2', float('nan'))):.4f}\n"
        f"RMS error = {float(forecast_metrics.get('error_rms_deg', forecast_metrics.get('rmse_deg', float('nan')))):.3f} deg"
    )
    if direct_metrics:
        text += (
            f"\n\n{direct_label}\n"
            f"R2 = {float(direct_metrics.get('r2', float('nan'))):.4f}\n"
            f"RMS error = {float(direct_metrics.get('error_rms_deg', direct_metrics.get('rmse_deg', float('nan')))):.3f} deg"
        )
    return text


def choose_zoom_windows(t: np.ndarray, n_windows: int = 3) -> List[Tuple[float, float]]:
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return []
    start = float(t[0])
    end = float(t[-1])
    span = end - start
    if span <= 0.0:
        return []
    width = min(span / max(n_windows + 1, 2), 30.0)
    width = max(width, min(span, 5.0))
    width = min(width, span)
    centres = np.linspace(start + 0.18 * span, end - 0.18 * span, n_windows)
    windows: List[Tuple[float, float]] = []
    for centre in centres:
        lo = max(start, float(centre) - width / 2.0)
        hi = min(end, float(centre) + width / 2.0)
        if hi > lo:
            windows.append((lo, hi))
    return windows


def add_fit_lines(ax, t: np.ndarray, measured_deg: np.ndarray, pred_deg: np.ndarray,
                  title: str, split_t: Optional[float] = None) -> None:
    ax.plot(t, measured_deg, label="Measured roll", linewidth=1.0)
    ax.plot(t, pred_deg, label="Rev M fitted roll", linewidth=1.2)
    if split_t is not None:
        ax.axvline(split_t, linestyle="--", linewidth=1.0, label="validation/forecast boundary")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def mask_forecast_from_fit_curve(pred_deg: np.ndarray,
                                 forecast: Dict[str, object]) -> np.ndarray:
    """Hide teacher-forced predictions wherever recursive forecast is shown."""
    fit_only = np.asarray(pred_deg, dtype=float).copy()
    if not bool(forecast.get("enabled", False)):
        return fit_only
    forecast_idx = np.asarray(forecast.get("index", []), dtype=int)
    valid = forecast_idx[(forecast_idx >= 0) & (forecast_idx < len(fit_only))]
    fit_only[valid] = np.nan
    return fit_only


def add_forecast_plot_lines(ax, forecast_time_s: np.ndarray,
                            forecast_pred_deg: np.ndarray,
                            uncertainty_lower_deg: np.ndarray,
                            uncertainty_upper_deg: np.ndarray,
                            cfg: Dict[str, object], mark_start: bool = False,
                            direct_no_ode_deg: Optional[np.ndarray] = None,
                            forecast_label: Optional[str] = None) -> None:
    """Draw forecast and uncertainty beside, rather than on top of, fitted data."""
    forecast_time_s = np.asarray(forecast_time_s, dtype=float)
    forecast_pred_deg = np.asarray(forecast_pred_deg, dtype=float)
    lower = np.asarray(uncertainty_lower_deg, dtype=float)
    upper = np.asarray(uncertainty_upper_deg, dtype=float)
    if len(forecast_time_s) == 0 or len(forecast_pred_deg) != len(forecast_time_s):
        return
    if len(lower) == len(forecast_time_s) and len(upper) == len(forecast_time_s):
        ax.fill_between(
            forecast_time_s,
            lower,
            upper,
            color="0.55",
            alpha=0.28,
            linewidth=0.0,
            label=f"{100.0 * float(cfg.get('forecast_uncertainty_confidence', 0.95)):.0f}% empirical uncertainty",
        )
    ax.plot(
        forecast_time_s,
        forecast_pred_deg,
        color="tab:red",
        linewidth=1.6,
        label=forecast_label or f"ODE rolling forecast ({forecast_window_description(cfg)})",
    )
    if direct_no_ode_deg is not None:
        direct = np.asarray(direct_no_ode_deg, dtype=float)
        if len(direct) == len(forecast_time_s):
            ax.plot(
                forecast_time_s,
                direct,
                color="tab:purple",
                linewidth=1.25,
                alpha=0.90,
                linestyle="--",
                label="Direct forecast (no ODE)",
            )
    if mark_start:
        ax.axvline(
            float(forecast_time_s[0]),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label="forecast begins",
        )


def safe_filename_token(text: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return token.strip("_") or "run"


def compute_all_metrics(data: Dict[str, object], pred_phi_scaled: np.ndarray, cfg: Dict[str, object]) -> Dict[str, object]:
    t = np.asarray(data["t"], dtype=float)
    measured_deg = np.rad2deg(np.asarray(data["phi"], dtype=float))
    pred_deg = np.rad2deg(np.asarray(pred_phi_scaled, dtype=float) * float(data["phi_std"]))
    metrics: Dict[str, object] = {"full": regression_metrics(measured_deg, pred_deg)}

    train_idx = np.asarray(data["train_idx"], dtype=int)
    val_idx = np.asarray(data["val_idx"], dtype=int)
    exclude_s = float(cfg.get("metrics_exclude_initial_s", 0.0))
    if exclude_s > 0.0:
        train_idx = train_idx[t[train_idx] >= t[train_idx[0]] + exclude_s]
    if len(train_idx) > 2:
        metrics["train"] = regression_metrics(measured_deg[train_idx], pred_deg[train_idx])
    if len(val_idx) > 2:
        metrics["validation"] = regression_metrics(measured_deg[val_idx], pred_deg[val_idx])
    run_segments = normalise_segments(data.get("run_segments"), len(t))
    run_names = [str(v) for v in list(data.get("run_names", []))]
    if len(run_names) < len(run_segments):
        run_names.extend(f"run_{i + 1}" for i in range(len(run_names), len(run_segments)))
    per_run: Dict[str, object] = {}
    for name, (lo, hi) in zip(run_names, run_segments):
        idx = np.arange(int(lo), int(hi) + 1, dtype=int)
        if len(idx) > 2:
            per_run[str(name)] = regression_metrics(measured_deg[idx], pred_deg[idx])
    if per_run:
        metrics["per_run"] = per_run
    metrics["physics_parameters"] = {
        "note": "c1, c2, k1 are in scaled model coordinates",
    }
    return metrics


def orientation_gain_series_from_data(data: Dict[str, object]) -> np.ndarray:
    X = np.asarray(data.get("X", []), dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    state_groups = data.get("state_feature_indices", {})
    cols = (
        state_groups.get("wave_orientation_effect_gain", [])
        if isinstance(state_groups, dict)
        else []
    )
    if not cols:
        return np.ones(X.shape[0], dtype=np.float64)
    valid_cols = [int(c) for c in cols if 0 <= int(c) < X.shape[1]]
    if not valid_cols:
        return np.ones(X.shape[0], dtype=np.float64)
    gain = np.nanmean(X[:, valid_cols], axis=1)
    gain = np.where(np.isfinite(gain), gain, 1.0)
    return np.clip(gain, 0.0, 1.0)


def encounter_regime_thresholds(cfg: Dict[str, object]) -> Tuple[float, float]:
    parallel_max = float(np.clip(float(cfg.get("force_regime_parallel_max_gain", 0.45)), 0.0, 1.0))
    side_min = float(np.clip(float(cfg.get("force_regime_side_min_gain", 0.75)), parallel_max, 1.0))
    return parallel_max, side_min


def encounter_regime_summary(orientation_gain: np.ndarray, indices: np.ndarray,
                             cfg: Dict[str, object]) -> Dict[str, object]:
    gain = np.asarray(orientation_gain, dtype=np.float64).reshape(-1)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    idx = idx[(idx >= 0) & (idx < len(gain))]
    parallel_max, side_min = encounter_regime_thresholds(cfg)
    if len(idx) == 0:
        return {
            "count": 0,
            "parallel_count": 0,
            "oblique_count": 0,
            "side_on_count": 0,
            "parallel_fraction": float("nan"),
            "oblique_fraction": float("nan"),
            "side_on_fraction": float("nan"),
            "mean_orientation_gain": float("nan"),
        }
    local_gain = gain[idx]
    parallel = local_gain <= parallel_max
    side_on = local_gain >= side_min
    oblique = ~(parallel | side_on)
    denom = float(len(idx))
    return {
        "count": int(len(idx)),
        "parallel_count": int(np.sum(parallel)),
        "oblique_count": int(np.sum(oblique)),
        "side_on_count": int(np.sum(side_on)),
        "parallel_fraction": float(np.sum(parallel) / denom),
        "oblique_fraction": float(np.sum(oblique) / denom),
        "side_on_fraction": float(np.sum(side_on) / denom),
        "mean_orientation_gain": float(np.nanmean(local_gain)),
        "min_orientation_gain": float(np.nanmin(local_gain)),
        "max_orientation_gain": float(np.nanmax(local_gain)),
    }


def write_outputs(model: PINNLSTM, data: Dict[str, object], cfg: Dict[str, object],
                  train_result: Dict[str, object], lg: logging.Logger,
                  extra_metrics: Optional[Dict[str, object]] = None,
                  include_full_ode: bool = False,
                  full_ode_coefficient_source: str = "model") -> Dict[str, Path]:
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_phi_s, _, pred_force_s, pred_turn_s, pred_force_residual_s = predict_full_teacher_forced(
        model, data, cfg
    )
    t = np.asarray(data["t"], dtype=float)
    measured_deg = np.rad2deg(np.asarray(data["phi"], dtype=float))
    pred_deg = np.rad2deg(pred_phi_s * float(data["phi_std"]))
    orientation_gain_all = orientation_gain_series_from_data(data)
    motion_indices = data.get("motion_feature_indices", {})
    feedback_active = bool(
        isinstance(motion_indices, dict)
        and (motion_indices.get("phi_fb", []) or motion_indices.get("v_fb", []))
    )
    exogenous_force = None
    if bool(cfg.get("forecast_use_ode", True)) and not feedback_active:
        exogenous_force = precompute_full_ode_exogenous_force(
            model,
            np.asarray(data["X"], dtype=np.float32),
            data,
            cfg,
            normalise_segments(data.get("run_segments"), len(t)),
        )
    forecast = forecast_roll_region(model, data, cfg, precomputed_exogenous_force=exogenous_force)
    forecast = add_forecast_uncertainty(forecast, data, pred_deg, cfg)
    forecast = add_force_alignment_diagnostic(model, data, cfg, forecast)
    oracle_replay = oracle_force_replay_forecast(model, data, forecast)
    force_replay_sweep = forecast_force_replay_sweep(model, data, forecast, cfg)
    coefficient_replay_sweep = forecast_coefficient_replay_sweep(model, data, forecast, cfg)
    forecast_force_envelope = forecast_force_envelope_diagnostic(forecast, cfg)
    orientation_gate_ablation = forecast_orientation_gate_ablation(model, data, forecast, cfg)
    teacher_force_alignment = teacher_forced_force_alignment_diagnostic(
        model,
        data,
        cfg,
        pred_force_s,
        pred_turn_s,
    )
    inferred_force_all = np.asarray(
        teacher_force_alignment.get("inferred_measured_force_scaled", []),
        dtype=float,
    )
    orientation_envelope_lag = orientation_envelope_lag_diagnostic(
        data,
        inferred_force_all,
        forecast,
        cfg,
    )
    force_lag_lags_train, force_lag_scores_train = wave_to_force_lag_scores(
        np.asarray(data["t"], dtype=float),
        inferred_force_all,
        np.asarray(data.get("wave_time_raw", []), dtype=float),
        np.asarray(data.get("wave_signal_raw", []), dtype=float),
        np.asarray(data.get("train_idx", []), dtype=int),
        cfg,
        normalise_segments(data.get("run_segments"), len(data["t"])),
        normalise_segments(data.get("wave_run_segments"), len(np.asarray(data.get("wave_time_raw", []), dtype=float))),
        np.asarray(data.get("encounter_time_shift_s", np.zeros(len(data["t"]))), dtype=float),
        float(data.get("dominant_roll_period_s", cfg.get("dominant_roll_period_s", 1.0))),
    )
    force_lag_lags_val, force_lag_scores_val = wave_to_force_lag_scores(
        np.asarray(data["t"], dtype=float),
        inferred_force_all,
        np.asarray(data.get("wave_time_raw", []), dtype=float),
        np.asarray(data.get("wave_signal_raw", []), dtype=float),
        np.asarray(data.get("val_idx", []), dtype=int),
        cfg,
        normalise_segments(data.get("run_segments"), len(data["t"])),
        normalise_segments(data.get("wave_run_segments"), len(np.asarray(data.get("wave_time_raw", []), dtype=float))),
        np.asarray(data.get("encounter_time_shift_s", np.zeros(len(data["t"]))), dtype=float),
        float(data.get("dominant_roll_period_s", cfg.get("dominant_roll_period_s", 1.0))),
        force_lag_lags_train,
    )
    current_wave_lag_s = float(data.get("best_wave_lag_s", 0.0))
    force_lag_scan = {
        "note": "Band-passed wave-slope proxy scanned directly against inferred measured force.",
        "current_wave_lag_s": current_wave_lag_s,
        "lags_s": force_lag_lags_train,
        "train_scores": force_lag_scores_train,
        "validation_scores": force_lag_scores_val,
        "train": force_lag_scan_summary(force_lag_lags_train, force_lag_scores_train, current_wave_lag_s),
        "validation": force_lag_scan_summary(force_lag_lags_val, force_lag_scores_val, current_wave_lag_s),
    }
    multi_force_lag_scan = multi_wave_proxy_force_lag_scan(
        data,
        inferred_force_all,
        cfg,
        current_wave_lag_s,
        force_lag_lags_train,
    )
    metrics = compute_all_metrics(data, pred_phi_s, cfg)
    metrics["physics_parameters"]["c_roll"] = float(model.c_roll.detach().cpu().item())
    metrics["physics_parameters"]["c_quad"] = float(model.c_quad.detach().cpu().item())
    metrics["physics_parameters"]["k_roll"] = float(model.k_roll.detach().cpu().item())
    metrics["roll_amplitude_calibration"] = {
        "enabled": bool(cfg.get("roll_amplitude_calibration_enabled", False)),
        "gain": float(model.roll_amplitude_gain.detach().cpu().item()),
        "gain_max": float(cfg.get("roll_amplitude_calibration_gain_max", 0.0)),
        "threshold_scaled": float(cfg.get("roll_amplitude_calibration_threshold_scaled", 0.0)),
        "softness_scaled": float(cfg.get("roll_amplitude_calibration_softness_scaled", 0.0)),
        "rate_scale": bool(cfg.get("roll_amplitude_calibration_rate_scale", True)),
    }
    pred_wave_force_s = np.asarray(pred_force_s, dtype=float) - np.asarray(
        pred_force_residual_s, dtype=float
    )
    total_force_rms = float(np.sqrt(np.mean(np.asarray(pred_force_s, dtype=float) ** 2)))
    wave_force_rms = float(np.sqrt(np.mean(pred_wave_force_s ** 2)))
    residual_force_rms = float(
        np.sqrt(np.mean(np.asarray(pred_force_residual_s, dtype=float) ** 2))
    )
    with torch.no_grad():
        diagnostic_X = torch.as_tensor(
            np.asarray(data["X"], dtype=np.float32),
            dtype=torch.float32,
            device=data["device"],
        )
        pure_wave_base, state_force_base = model.wave_forcing_components(diagnostic_X)
        wave_breakdown_base = model.wave_forcing_breakdown(diagnostic_X)
        pure_wave_base_rms = float(
            torch.sqrt(torch.mean(pure_wave_base ** 2)).detach().cpu().item()
        )
        state_force_base_rms = float(
            torch.sqrt(torch.mean(state_force_base ** 2)).detach().cpu().item()
        )
        wave_signal_force_rms = float(
            torch.sqrt(torch.mean(wave_breakdown_base["wave_signal_force"] ** 2)).detach().cpu().item()
        )
        wave_slope_force_rms = float(
            torch.sqrt(torch.mean(wave_breakdown_base["wave_slope_force"] ** 2)).detach().cpu().item()
        )
        wave_auxiliary_force_rms = float(
            torch.sqrt(torch.mean(wave_breakdown_base["wave_auxiliary_force"] ** 2)).detach().cpu().item()
        )
        wave_envelope_gate_base = (
            wave_breakdown_base["wave_envelope_gate"].detach().cpu().numpy().astype(np.float64)
        )
        pure_wave_gate_base = (
            wave_breakdown_base["pure_wave_gate"].detach().cpu().numpy().astype(np.float64)
        )
    metrics["wave_forcing"] = {
        "wave_forcing_gain": float(cfg.get("wave_forcing_gain", 0.0)),
        "wave_quadrature_basis_enabled": bool(
            cfg.get("wave_quadrature_basis_enabled", True)
        ),
        "wave_quadrature_signal_init": float(
            cfg.get("wave_quadrature_signal_init", -0.08)
        ),
        "wave_quadrature_slope_init": float(
            cfg.get("wave_quadrature_slope_init", 0.18)
        ),
        "wave_auxiliary_force_gain": float(
            cfg.get("wave_auxiliary_force_gain", 0.0)
        ),
        "wave_shape_correction_enabled": bool(
            cfg.get("wave_shape_correction_enabled", False)
        ),
        "wave_shape_correction_hidden": int(
            cfg.get("wave_shape_correction_hidden", 0)
        ),
        "wave_shape_correction_gain": float(
            cfg.get("wave_shape_correction_gain", 0.0)
        ),
        "force_residual_scale": float(cfg.get("force_residual_scale", 0.0)),
        "lambda_wave_force_target": float(
            cfg.get("lambda_wave_force_target", 0.0)
        ),
        "lambda_pure_wave_force_target": float(
            cfg.get("lambda_pure_wave_force_target", 0.0)
        ),
        "lambda_total_force_target": float(
            cfg.get("lambda_total_force_target", 0.0)
        ),
        "total_force_target_phase_weight": float(
            cfg.get("total_force_target_phase_weight", 0.0)
        ),
        "total_force_target_underforce_weight": float(
            cfg.get("total_force_target_underforce_weight", 0.0)
        ),
        "lambda_total_force_shape": float(
            cfg.get("lambda_total_force_shape", 0.0)
        ),
        "total_force_shape_overshoot_weight": float(
            cfg.get("total_force_shape_overshoot_weight", 0.0)
        ),
        "lambda_total_force_tail": float(
            cfg.get("lambda_total_force_tail", 0.0)
        ),
        "lambda_total_force_event": float(
            cfg.get("lambda_total_force_event", 0.0)
        ),
        "total_force_event_quantile": float(
            cfg.get("total_force_event_quantile", 0.0)
        ),
        "total_force_event_weight": float(
            cfg.get("total_force_event_weight", 0.0)
        ),
        "lambda_total_force_band_shape": float(
            cfg.get("lambda_total_force_band_shape", 0.0)
        ),
        "total_force_band_shape_freqs_hz": config_float_list(
            cfg, "total_force_band_shape_freqs_hz", [0.45, 1.70, 3.20]
        ),
        "total_force_band_shape_weights": config_float_list(
            cfg, "total_force_band_shape_weights", [1.0, 2.0, 0.75]
        ),
        "total_force_band_shape_phase_weight": float(
            cfg.get("total_force_band_shape_phase_weight", 0.0)
        ),
        "total_force_band_shape_event_weight": float(
            cfg.get("total_force_band_shape_event_weight", 0.0)
        ),
        "total_force_band_shape_envelope_weight": float(
            cfg.get("total_force_band_shape_envelope_weight", 0.0)
        ),
        "lambda_total_force_spectral_shape": float(
            cfg.get("lambda_total_force_spectral_shape", 0.0)
        ),
        "total_force_spectral_shape_band_lows_hz": config_float_list(
            cfg, "total_force_spectral_shape_band_lows_hz", [0.02, 1.45, 2.90]
        ),
        "total_force_spectral_shape_band_highs_hz": config_float_list(
            cfg, "total_force_spectral_shape_band_highs_hz", [0.55, 1.85, 3.40]
        ),
        "total_force_spectral_shape_weights": config_float_list(
            cfg, "total_force_spectral_shape_weights", [1.50, 6.0, 3.0]
        ),
        "total_force_spectral_shape_excess_only": config_float_list(
            cfg, "total_force_spectral_shape_excess_only", [1.0, 0.0, 0.0]
        ),
        "total_force_spectral_shape_underfit_weight": float(
            cfg.get("total_force_spectral_shape_underfit_weight", 0.0)
        ),
        "total_force_spectral_shape_complex_weight": float(
            cfg.get("total_force_spectral_shape_complex_weight", 0.0)
        ),
        "total_force_spectral_shape_low_excess_margin": float(
            cfg.get("total_force_spectral_shape_low_excess_margin", 0.0)
        ),
        "lambda_total_force_regime_balance": float(
            cfg.get("lambda_total_force_regime_balance", 0.0)
        ),
        "force_regime_parallel_max_gain": float(
            cfg.get("force_regime_parallel_max_gain", 0.45)
        ),
        "force_regime_side_min_gain": float(
            cfg.get("force_regime_side_min_gain", 0.75)
        ),
        "force_regime_parallel_weight": float(
            cfg.get("force_regime_parallel_weight", 1.0)
        ),
        "force_regime_oblique_weight": float(
            cfg.get("force_regime_oblique_weight", 1.0)
        ),
        "force_regime_side_weight": float(
            cfg.get("force_regime_side_weight", 2.0)
        ),
        "force_regime_envelope_weight": float(
            cfg.get("force_regime_envelope_weight", 0.0)
        ),
        "lambda_force_residual": float(cfg.get("lambda_force_residual", 0.0)),
        "motion_feedback_backbone_gain": float(
            cfg.get("motion_feedback_backbone_gain", 1.0)
        ),
        "motion_feedback_gate_gain": float(
            cfg.get("motion_feedback_gate_gain", 1.0)
        ),
        "wave_orientation_gate_enabled": bool(cfg.get("wave_orientation_gate_enabled", True)),
        "wave_parallel_gate_state_force": bool(cfg.get("wave_parallel_gate_state_force", True)),
        "wave_parallel_min_gain": float(cfg.get("wave_parallel_min_gain", 0.0)),
        "wave_parallel_velocity_blend": float(cfg.get("wave_parallel_velocity_blend", 0.0)),
        "wave_parallel_velocity_reference_quantile": float(
            cfg.get("wave_parallel_velocity_reference_quantile", 0.0)
        ),
        "wave_parallel_obliquity_power": float(cfg.get("wave_parallel_obliquity_power", 0.0)),
        "wave_parallel_gate_smoothing_seconds": float(
            cfg.get("wave_parallel_gate_smoothing_seconds", 0.0)
        ),
        "wave_envelope_gate_enabled": bool(cfg.get("wave_envelope_gate_enabled", True)),
        "wave_envelope_gate_gain": float(cfg.get("wave_envelope_gate_gain", 0.0)),
        "wave_envelope_gate_min_config": float(cfg.get("wave_envelope_gate_min", 0.0)),
        "wave_envelope_gate_max_config": float(cfg.get("wave_envelope_gate_max", 0.0)),
        "wave_envelope_gate_input_dim": int(
            getattr(model, "wave_envelope_gate_input_dim", 0)
        ),
        "wave_envelope_gate_min": float(np.nanmin(wave_envelope_gate_base)),
        "wave_envelope_gate_max": float(np.nanmax(wave_envelope_gate_base)),
        "wave_envelope_gate_mean": float(np.nanmean(wave_envelope_gate_base)),
        "pure_wave_gate_min": float(np.nanmin(pure_wave_gate_base)),
        "pure_wave_gate_max": float(np.nanmax(pure_wave_gate_base)),
        "pure_wave_gate_mean": float(np.nanmean(pure_wave_gate_base)),
        "lambda_wave_envelope_gate_reg": float(
            cfg.get("lambda_wave_envelope_gate_reg", 0.0)
        ),
        "lambda_wave_envelope_gate_target": float(
            cfg.get("lambda_wave_envelope_gate_target", 0.0)
        ),
        "lambda_wave_envelope_gate_shape": float(
            cfg.get("lambda_wave_envelope_gate_shape", 0.0)
        ),
        "wave_envelope_gate_target_huber_delta": float(
            cfg.get("wave_envelope_gate_target_huber_delta", 0.0)
        ),
        "wave_envelope_gate_target_base_floor_ratio": float(
            cfg.get("wave_envelope_gate_target_base_floor_ratio", 0.0)
        ),
        "wave_force_target_smoothing_seconds": float(
            cfg.get("wave_force_target_smoothing_seconds", 0.0)
        ),
        "force_target_band_mse_weight": float(
            cfg.get("force_target_band_mse_weight", 0.0)
        ),
        "force_target_raw_weight": float(
            cfg.get("force_target_raw_weight", 0.0)
        ),
        "force_target_huber_delta": float(
            cfg.get("force_target_huber_delta", 0.0)
        ),
        "force_target_band_huber_delta": float(
            cfg.get("force_target_band_huber_delta", 0.0)
        ),
        "wave_force_target_phase_weight": float(
            cfg.get("wave_force_target_phase_weight", 0.0)
        ),
        "wave_force_target_amplitude_weight": float(
            cfg.get("wave_force_target_amplitude_weight", 0.0)
        ),
        "wave_force_target_underforce_weight": float(
            cfg.get("wave_force_target_underforce_weight", 0.0)
        ),
        "lambda_wave_force_highpass": float(
            cfg.get("lambda_wave_force_highpass", 0.0)
        ),
        "wave_force_highpass_window_s": float(
            cfg.get("wave_force_highpass_window_s", 0.0)
        ),
        "lambda_force_band_amplitude": float(
            cfg.get("lambda_force_band_amplitude", 0.0)
        ),
        "lambda_force_band_phase": float(
            cfg.get("lambda_force_band_phase", 0.0)
        ),
        "lambda_force_lag_penalty": float(
            cfg.get("lambda_force_lag_penalty", 0.0)
        ),
        "force_lag_penalty_max_s": float(
            cfg.get("force_lag_penalty_max_s", 0.0)
        ),
        "force_lag_penalty_steps": int(
            cfg.get("force_lag_penalty_steps", 0)
        ),
        "force_band_amplitude_underfit_weight": float(
            cfg.get("force_band_amplitude_underfit_weight", 0.0)
        ),
        "lambda_force_envelope": float(
            cfg.get("lambda_force_envelope", 0.0)
        ),
        "force_envelope_window_s": float(
            cfg.get("force_envelope_window_s", 0.0)
        ),
        "force_envelope_huber_delta": float(
            cfg.get("force_envelope_huber_delta", 0.0)
        ),
        "force_band_amplitude_center_period_s": (
            None
            if cfg.get("force_band_amplitude_center_period_s") is None
            else float(cfg.get("force_band_amplitude_center_period_s", 0.0))
        ),
        "dominant_roll_period_s": (
            None
            if cfg.get("dominant_roll_period_s") is None
            else float(cfg.get("dominant_roll_period_s", 0.0))
        ),
        "total_force_rms_scaled": total_force_rms,
        "wave_driven_force_rms_scaled": wave_force_rms,
        "residual_force_rms_scaled": residual_force_rms,
        "pure_wave_base_rms_scaled": pure_wave_base_rms,
        "wave_signal_force_rms_scaled": wave_signal_force_rms,
        "wave_slope_force_rms_scaled": wave_slope_force_rms,
        "wave_auxiliary_force_rms_scaled": wave_auxiliary_force_rms,
        "state_force_base_rms_scaled": state_force_base_rms,
        "wave_to_total_rms_ratio": (
            wave_force_rms / max(total_force_rms, 1.0e-12)
        ),
        "learned_couplings": {
            "wave_signal": model.a_wave.detach().cpu().tolist(),
            "wave_slope": model.a_wave_slope.detach().cpu().tolist(),
            "wave_abs_slope": model.a_wave_abs_slope.detach().cpu().tolist(),
            "wave_envelope": model.a_wave_env.detach().cpu().tolist(),
            "wave_envelope_slow": model.a_wave_env_slow.detach().cpu().tolist(),
            "wave_cross_beam_gradient": (
                model.a_wave_cross_beam_gradient.detach().cpu().tolist()
            ),
            "wave_shape_correction_parameter_count": int(
                sum(p.numel() for p in model.wave_shape_correction.parameters())
            ),
        },
        "learned_state_couplings": {
            "speed": model.a_speed.detach().cpu().tolist(),
            "yawrate": model.a_yawrate.detach().cpu().tolist(),
            "x_position": model.a_x_position.detach().cpu().tolist(),
            "y_position": model.a_y_position.detach().cpu().tolist(),
            "track_speed_xy": model.a_track_speed_xy.detach().cpu().tolist(),
            "wave_perpendicular_velocity": model.a_wave_perpendicular_velocity.detach().cpu().tolist(),
            "wave_parallel_velocity": model.a_wave_parallel_velocity.detach().cpu().tolist(),
            "heading_perpendicular_to_waves": model.a_heading_perpendicular_to_waves.detach().cpu().tolist(),
            "heading_parallel_to_waves": model.a_heading_parallel_to_waves.detach().cpu().tolist(),
            "heading_obliquity_abs": model.a_heading_obliquity_abs.detach().cpu().tolist(),
        },
    }
    full_ode_integration: Optional[Dict[str, object]] = None
    if bool(include_full_ode):
        full_ode_integration = integrate_full_ode_wave_train(
            model,
            data,
            cfg,
            coefficient_source=str(full_ode_coefficient_source),
            precomputed_exogenous_force=exogenous_force,
        )
        full_ode_integration = add_full_ode_forecast_uncertainty(full_ode_integration, forecast, cfg)
        metrics["full_ode"] = {
            "coefficient_source": full_ode_integration.get("coefficient_source"),
            "coefficient_note": full_ode_integration.get("coefficient_note"),
            "coefficients": full_ode_integration.get("coefficients"),
            "force_source": full_ode_integration.get("force_source"),
            "initialisation_note": full_ode_integration.get("initialisation_note"),
            "known_inputs_note": full_ode_integration.get("known_inputs_note"),
            "uncertainty": full_ode_integration.get("uncertainty", {}),
            "valid_points": full_ode_integration.get("valid_points"),
            "total_points": full_ode_integration.get("total_points"),
            "run_status": full_ode_integration.get("run_status"),
            "metrics": full_ode_integration.get("metrics"),
        }
    history = list(train_result.get("history", []))
    final_losses: Dict[str, object] = {}
    if history and isinstance(history[-1], dict):
        final_losses = {
            str(k): v for k, v in history[-1].items()
            if str(k).startswith("train_") or str(k).startswith("val_") or str(k).startswith("monitor_") or str(k) in {"epoch", "lr"}
        }
    metrics["training"] = {
        "best_epoch": train_result.get("best_epoch"),
        "best_metric": train_result.get("best_metric"),
        "best_metric_name": train_result.get("best_metric_name"),
        "best_checkpoint": train_result.get("best_checkpoint"),
        "final_checkpoint": train_result.get("final_checkpoint"),
        "final_losses": final_losses,
    }
    metrics["extrema_emphasis"] = {
        "lambda_peak_data": float(cfg.get("lambda_peak_data", 0.0)),
        "lambda_peak_trough": float(cfg.get("lambda_peak_trough", 0.0)),
        "lambda_global_extrema": float(cfg.get("lambda_global_extrema", 0.0)),
        "lambda_local_prominence": float(cfg.get("lambda_local_prominence", 0.0)),
        "lambda_amplitude_underfit": float(cfg.get("lambda_amplitude_underfit", 0.0)),
        "lambda_extrema_window_underfit": float(cfg.get("lambda_extrema_window_underfit", 0.0)),
        "extrema_window_radius": int(cfg.get("extrema_window_radius", 0)),
        "extrema_window_quantile": float(cfg.get("extrema_window_quantile", 0.0)),
        "extrema_window_overshoot_weight": float(cfg.get("extrema_window_overshoot_weight", 0.0)),
        "lambda_high_amplitude_underfit": float(cfg.get("lambda_high_amplitude_underfit", 0.0)),
        "high_amplitude_underfit_quantile": float(cfg.get("high_amplitude_underfit_quantile", 0.0)),
        "high_amplitude_underfit_overshoot_weight": float(
            cfg.get("high_amplitude_underfit_overshoot_weight", 0.0)
        ),
        "lambda_rollout_peak_trough": float(cfg.get("lambda_rollout_peak_trough", 0.0)),
        "lambda_rollout_global_extrema": float(cfg.get("lambda_rollout_global_extrema", 0.0)),
        "lambda_rollout_extrema_window_underfit": float(cfg.get("lambda_rollout_extrema_window_underfit", 0.0)),
        "lambda_rollout_high_amplitude_underfit": float(
            cfg.get("lambda_rollout_high_amplitude_underfit", 0.0)
        ),
        "lambda_turning_point": float(cfg.get("lambda_turning_point", 0.0)),
        "lambda_rollout_local_prominence": float(cfg.get("lambda_rollout_local_prominence", 0.0)),
        "direct_forecast_loss_enabled": bool(cfg.get("direct_forecast_loss_enabled", True)),
        "direct_forecast_loss_window_s": float(cfg.get("direct_forecast_loss_window_s", 0.0)),
        "direct_forecast_loss_horizons_s": config_float_list(
            cfg,
            "direct_forecast_loss_horizons_s",
            [float(cfg.get("direct_forecast_loss_window_s", 0.0))],
        ),
        "direct_forecast_loss_horizon_weights": config_float_list(
            cfg,
            "direct_forecast_loss_horizon_weights",
            [1.0],
        ),
        "direct_forecast_tail_fraction": float(cfg.get("direct_forecast_tail_fraction", 0.0)),
        "direct_forecast_tail_multiplier": float(cfg.get("direct_forecast_tail_multiplier", 1.0)),
        "direct_forecast_loss_batch_size": int(cfg.get("direct_forecast_loss_batch_size", 0)),
        "direct_forecast_turning_point_sampling_enabled": bool(
            cfg.get("direct_forecast_turning_point_sampling_enabled", False)
        ),
        "direct_forecast_turning_point_weight": float(
            cfg.get("direct_forecast_turning_point_weight", 0.0)
        ),
        "direct_forecast_turning_point_max_tasks": int(
            cfg.get("direct_forecast_turning_point_max_tasks", 0)
        ),
        "direct_forecast_turning_point_min_prominence_deg": float(
            cfg.get("direct_forecast_turning_point_min_prominence_deg", 0.0)
        ),
        "direct_forecast_turning_point_neighbourhood": int(
            cfg.get("direct_forecast_turning_point_neighbourhood", 0)
        ),
        "direct_forecast_turning_point_window_fracs": config_float_list(
            cfg, "direct_forecast_turning_point_window_fracs", [0.5]
        ),
        "lambda_direct_forecast": float(cfg.get("lambda_direct_forecast", 0.0)),
        "lambda_direct_forecast_amplitude": float(cfg.get("lambda_direct_forecast_amplitude", 0.0)),
        "lambda_direct_forecast_phase": float(cfg.get("lambda_direct_forecast_phase", 0.0)),
        "lambda_direct_forecast_peak_trough": float(cfg.get("lambda_direct_forecast_peak_trough", 0.0)),
        "lambda_direct_forecast_global_extrema": float(cfg.get("lambda_direct_forecast_global_extrema", 0.0)),
        "lambda_direct_forecast_extrema_window_underfit": float(
            cfg.get("lambda_direct_forecast_extrema_window_underfit", 0.0)
        ),
        "lambda_direct_forecast_high_amplitude_underfit": float(
            cfg.get("lambda_direct_forecast_high_amplitude_underfit", 0.0)
        ),
        "lambda_direct_forecast_asymmetric_extrema": float(
            cfg.get("lambda_direct_forecast_asymmetric_extrema", 0.0)
        ),
        "direct_forecast_asym_extrema_radius": int(
            cfg.get("direct_forecast_asym_extrema_radius", 0)
        ),
        "direct_forecast_asym_extrema_quantile": float(
            cfg.get("direct_forecast_asym_extrema_quantile", 0.0)
        ),
        "direct_forecast_peak_underfit_weight": float(
            cfg.get("direct_forecast_peak_underfit_weight", 0.0)
        ),
        "direct_forecast_peak_overshoot_weight": float(
            cfg.get("direct_forecast_peak_overshoot_weight", 0.0)
        ),
        "direct_forecast_trough_overshoot_weight": float(
            cfg.get("direct_forecast_trough_overshoot_weight", 0.0)
        ),
        "direct_forecast_trough_underfit_weight": float(
            cfg.get("direct_forecast_trough_underfit_weight", 0.0)
        ),
        "direct_forecast_asym_extrema_scale_floor_ratio": float(
            cfg.get("direct_forecast_asym_extrema_scale_floor_ratio", 0.0)
        ),
        "lambda_direct_forecast_local_prominence": float(
            cfg.get("lambda_direct_forecast_local_prominence", 0.0)
        ),
        "lambda_direct_forecast_high_pass_residual": float(
            cfg.get("lambda_direct_forecast_high_pass_residual", 0.0)
        ),
        "lambda_direct_forecast_roll_spectral_shape": float(
            cfg.get("lambda_direct_forecast_roll_spectral_shape", 0.0)
        ),
        "lambda_direct_forecast_roll_curvature": float(
            cfg.get("lambda_direct_forecast_roll_curvature", 0.0)
        ),
        "direct_forecast_amplitude_calibration": {
            "enabled": bool(cfg.get("direct_forecast_amplitude_calibration_enabled", False)),
            "gain_min": float(cfg.get("direct_forecast_amplitude_calibration_gain_min", 0.0)),
            "gain_max": float(cfg.get("direct_forecast_amplitude_calibration_gain_max", 0.0)),
            "gain_steps": int(cfg.get("direct_forecast_amplitude_calibration_gain_steps", 0)),
            "extrema_weight": float(cfg.get("direct_forecast_amplitude_calibration_extrema_weight", 0.0)),
        },
        "direct_forecast_selection": {
            "r2_weight": float(cfg.get("direct_forecast_selection_r2_weight", 1.0)),
            "loss_weight": float(cfg.get("direct_forecast_selection_loss_weight", 0.35)),
            "peak_weight": float(cfg.get("direct_forecast_selection_peak_weight", 1.0)),
            "extrema_weight": float(cfg.get("direct_forecast_selection_extrema_weight", 0.75)),
            "asym_extrema_weight": float(cfg.get("direct_forecast_selection_asym_extrema_weight", 1.0)),
            "spectral_weight": float(cfg.get("direct_forecast_selection_spectral_weight", 0.25)),
        },
        "peak_data_quantile": float(cfg.get("peak_data_quantile", 0.0)),
        "peak_trough_quantile": float(cfg.get("peak_trough_quantile", 0.0)),
        "training_rollout_window_s": float(
            cfg.get("motion_feedback_rollout_window_s", cfg.get("rollout_window_s", 0.0))
        ),
        "bayes_extrema_objective_weight": float(
            cfg.get("bayes_extrema_objective_weight", 0.0)
        ),
    }
    metrics["wave_alignment"] = {
        "mode": (
            "fixed"
            if cfg.get("fixed_wave_lag_s") is not None
            else "training-only band-limited correlation scan"
        ),
        "wave_lag_source": data.get("wave_lag_source"),
        "source_file": cfg.get("wave_alignment_source_file"),
        "fixed_wave_lag_s": cfg.get("fixed_wave_lag_s"),
        "data_time_window": data.get("data_time_window"),
        "validation_split": data.get("validation_split"),
        "best_wave_lag_s": data.get("best_wave_lag_s"),
        "wave_lags_s": data.get("wave_lags_s"),
        "dominant_roll_period_s": data.get("dominant_roll_period_s"),
        "encounter_stats": data.get("encounter_stats"),
        "orientation_stats": data.get("orientation_stats"),
        "run_names": data.get("run_names"),
        "run_segments": data.get("run_segments"),
        "wave_run_segments": data.get("wave_run_segments"),
    }
    if extra_metrics:
        metrics.update(copy.deepcopy(extra_metrics))
    metrics["teacher_forced_force_alignment"] = {
        "note": teacher_force_alignment.get("note"),
        "spectral_band_center_period_s": teacher_force_alignment.get("spectral_band_center_period_s"),
        "metrics": teacher_force_alignment.get("metrics", {}),
    }
    metrics["wave_to_inferred_force_lag_scan"] = {
        "note": force_lag_scan.get("note"),
        "current_wave_lag_s": force_lag_scan.get("current_wave_lag_s"),
        "train": force_lag_scan.get("train", {}),
        "validation": force_lag_scan.get("validation", {}),
    }
    metrics["wave_to_inferred_force_multifeature_lag_scan"] = {
        "note": multi_force_lag_scan.get("note"),
        "current_wave_lag_s": multi_force_lag_scan.get("current_wave_lag_s"),
        "proxies": {
            str(name): {
                "train": scan.get("train", {}),
                "validation": scan.get("validation", {}),
            }
            for name, scan in (
                multi_force_lag_scan.get("proxies", {})
                if isinstance(multi_force_lag_scan.get("proxies", {}), dict)
                else {}
            ).items()
            if isinstance(scan, dict)
        },
    }
    metrics["orientation_envelope_lag_diagnostic"] = (
        orientation_envelope_lag.get("metrics", {})
        if bool(orientation_envelope_lag.get("enabled", False))
        else {
            "enabled": False,
            "reason": orientation_envelope_lag.get(
                "reason",
                "Orientation envelope lag diagnostic unavailable.",
            ),
        }
    )
    if bool(forecast.get("enabled", False)):
        metrics["forecast"] = {
            "method": forecast.get("method"),
            "forecast_primary": forecast.get("forecast_primary"),
            "forecast_ode_enabled": bool(forecast.get("forecast_ode_enabled", True)),
            "start_idx": forecast.get("start_idx"),
            "end_idx": forecast.get("end_idx"),
            "start_time_s": forecast.get("start_time_s"),
            "end_time_s": forecast.get("end_time_s"),
            "duration_s": forecast.get("duration_s"),
            "window_target_s": (
                float(cfg["forecast_window_s"])
                if cfg.get("forecast_window_s") is not None else None
            ),
            "window_target_fraction": (
                float(cfg.get("forecast_frac", 0.10))
                if cfg.get("forecast_window_s") is None else None
            ),
            "delay_steps": forecast.get("delay_steps"),
            "feedback_delay_steps": forecast.get("feedback_delay_steps"),
            "uses_predicted_motion_feedback": forecast.get("uses_predicted_motion_feedback"),
            "training_feedback_rollout_window_s": forecast.get("training_feedback_rollout_window_s"),
            "training_feedback_rollout_batch_size": forecast.get("training_feedback_rollout_batch_size"),
            "handoff_phi_error_scaled": forecast.get("handoff_phi_error_scaled"),
            "handoff_phi_error_deg": forecast.get("handoff_phi_error_deg"),
            "handoff_v_error_scaled": forecast.get("handoff_v_error_scaled"),
            "known_inputs_note": forecast.get("known_inputs_note"),
            "roll_dynamics_note": forecast.get("roll_dynamics_note"),
            "direct_no_ode_forecast": (
                {
                    "enabled": bool(forecast.get("direct_no_ode_forecast", {}).get("enabled", False)),
                    "method": forecast.get("direct_no_ode_forecast", {}).get("method"),
                    "note": forecast.get("direct_no_ode_forecast", {}).get("note"),
                    "uses_predicted_motion_feedback": forecast.get("direct_no_ode_forecast", {}).get(
                        "uses_predicted_motion_feedback"
                    ),
                    "amplitude_calibration": forecast.get("direct_no_ode_forecast", {}).get(
                        "direct_forecast_amplitude_calibration", {}
                    ),
                    "uncorrected_metrics": forecast.get("direct_no_ode_forecast", {}).get(
                        "uncorrected_metrics", {}
                    ),
                    "metrics": forecast.get("direct_no_ode_forecast", {}).get("metrics", {}),
                }
                if isinstance(forecast.get("direct_no_ode_forecast", {}), dict)
                else {"enabled": False, "reason": "Direct no-ODE forecast unavailable."}
            ),
            "wave_force_verification": forecast.get("wave_force_verification", {}),
            "force_post_correction": forecast.get("force_post_correction", {}),
            "forecast_force_lag_s": forecast.get("forecast_force_lag_s"),
            "forecast_force_lag_note": forecast.get("forecast_force_lag_note"),
            "uncorrected_metrics": forecast.get("uncorrected_metrics", {}),
            "force_alignment": forecast.get("force_alignment", {}),
            "oracle_force_replay": {
                "note": oracle_replay.get("note"),
                "metrics": oracle_replay.get("metrics", {}),
                "learned_force_vs_oracle_force": oracle_replay.get(
                    "learned_force_vs_oracle_force",
                    {},
                ),
            } if bool(oracle_replay.get("enabled", False)) else {
                "enabled": False,
                "reason": oracle_replay.get("reason", "Oracle replay unavailable."),
            },
            "force_replay_sweep": {
                "note": force_replay_sweep.get("note"),
                "lag_min_s": float(np.nanmin(force_replay_sweep.get("lags_s", [float("nan")]))),
                "lag_max_s": float(np.nanmax(force_replay_sweep.get("lags_s", [float("nan")]))),
                "gain_min": float(np.nanmin(force_replay_sweep.get("gains", [float("nan")]))),
                "gain_max": float(np.nanmax(force_replay_sweep.get("gains", [float("nan")]))),
                "best_by_rmse": force_replay_sweep.get("best_by_rmse", {}),
                "baseline": force_replay_sweep.get("baseline", {}),
            } if bool(force_replay_sweep.get("enabled", False)) else {
                "enabled": False,
                "reason": force_replay_sweep.get("reason", "Force replay sweep unavailable."),
            },
            "coefficient_replay_sweep": {
                "note": coefficient_replay_sweep.get("note"),
                "multiplier_min": float(np.nanmin(coefficient_replay_sweep.get("multipliers", [float("nan")]))),
                "multiplier_max": float(np.nanmax(coefficient_replay_sweep.get("multipliers", [float("nan")]))),
                "base_coefficients": coefficient_replay_sweep.get("base_coefficients", {}),
                "selection_metric": coefficient_replay_sweep.get("selection_metric", "r2"),
                "best_selected": coefficient_replay_sweep.get("best_selected", {}),
                "best_by_r2": coefficient_replay_sweep.get("best_by_r2", {}),
                "best_by_rmse": coefficient_replay_sweep.get("best_by_rmse", {}),
                "baseline": coefficient_replay_sweep.get("baseline", {}),
            } if bool(coefficient_replay_sweep.get("enabled", False)) else {
                "enabled": False,
                "reason": coefficient_replay_sweep.get(
                    "reason",
                    "Coefficient replay sweep unavailable.",
                ),
            },
            "force_envelope_diagnostic": (
                forecast_force_envelope.get("metrics", {})
                if bool(forecast_force_envelope.get("enabled", False))
                else {
                    "enabled": False,
                    "reason": forecast_force_envelope.get(
                        "reason",
                        "Force envelope diagnostic unavailable.",
                    ),
                }
            ),
            "orientation_gate_ablation": (
                orientation_gate_ablation.get("metrics", {})
                if bool(orientation_gate_ablation.get("enabled", False))
                else {
                    "enabled": False,
                    "reason": orientation_gate_ablation.get(
                        "reason",
                        "Orientation-gate ablation unavailable.",
                    ),
                }
            ),
            "uncertainty": forecast.get("uncertainty", {}),
            "metrics": forecast.get("metrics", {}),
        }
    else:
        metrics["forecast"] = {"enabled": False, "reason": forecast.get("reason", "Forecast unavailable.")}

    parallel_max, side_min = encounter_regime_thresholds(cfg)
    metrics["encounter_regimes"] = {
        "note": (
            "Regimes are derived from wave_orientation_effect_gain: lower values "
            "are closer to parallel wave travel, higher values are side-on/wave-normal."
        ),
        "parallel_max_gain": parallel_max,
        "side_on_min_gain": side_min,
        "lambda_total_force_regime_balance": float(
            cfg.get("lambda_total_force_regime_balance", 0.0)
        ),
        "parallel_weight": float(cfg.get("force_regime_parallel_weight", 1.0)),
        "oblique_weight": float(cfg.get("force_regime_oblique_weight", 1.0)),
        "side_on_weight": float(cfg.get("force_regime_side_weight", 2.0)),
        "full": encounter_regime_summary(
            orientation_gain_all,
            np.arange(len(orientation_gain_all), dtype=int),
            cfg,
        ),
        "train": encounter_regime_summary(
            orientation_gain_all,
            np.asarray(data.get("train_idx", []), dtype=int),
            cfg,
        ),
        "validation": encounter_regime_summary(
            orientation_gain_all,
            np.asarray(data.get("val_idx", []), dtype=int),
            cfg,
        ),
        "forecast": encounter_regime_summary(
            orientation_gain_all,
            np.asarray(forecast.get("index", []), dtype=int),
            cfg,
        ) if bool(forecast.get("enabled", False)) else {
            "enabled": False,
            "reason": forecast.get("reason", "Forecast unavailable."),
        },
    }

    metrics_path = output_dir / "metrics_revm.json"
    cfg_path = output_dir / "config_revm.json"
    plot_path = output_dir / "roll_fit_revm.png"
    fit_csv_path = output_dir / "roll_fit_revm.csv"
    forecast_plot_path = output_dir / "roll_forecast_zoom_revm.png"
    forecast_zoom_csv_path = output_dir / "roll_forecast_zoom_revm.csv"
    wave_force_plot_path = output_dir / "wave_force_forecast_revm.png"
    force_alignment_plot_path = output_dir / "force_alignment_forecast_revm.png"
    oracle_replay_plot_path = output_dir / "oracle_force_replay_revm.png"
    force_replay_sweep_plot_path = output_dir / "forecast_force_replay_sweep_revm.png"
    coefficient_replay_sweep_plot_path = output_dir / "forecast_coefficient_replay_sweep_revm.png"
    force_post_correction_plot_path = output_dir / "forecast_force_post_correction_revm.png"
    force_envelope_plot_path = output_dir / "forecast_force_envelope_diagnostic_revm.png"
    orientation_gate_ablation_plot_path = output_dir / "forecast_orientation_gate_ablation_revm.png"
    vessel_motion_regime_plot_path = output_dir / "vessel_motion_encounter_regimes_revm.png"
    teacher_force_alignment_plot_path = output_dir / "teacher_forced_force_alignment_revm.png"
    teacher_force_spectrum_plot_path = output_dir / "teacher_forced_force_spectrum_revm.png"
    teacher_force_band_amplitude_plot_path = output_dir / "teacher_forced_force_band_amplitude_revm.png"
    wave_force_lag_plot_path = output_dir / "wave_to_inferred_force_lag_scan_revm.png"
    wave_force_multifeature_lag_plot_path = output_dir / "wave_to_inferred_force_multifeature_lag_scan_revm.png"
    orientation_envelope_lag_plot_path = output_dir / "orientation_envelope_lag_diagnostic_revm.png"
    forecast_csv_path = output_dir / "roll_forecast_region_revm.csv"
    lag_plot_path = output_dir / "wave_lag_scan_revm.png"
    lag_csv_path = output_dir / "wave_lag_scan_revm.csv"
    loss_plot_path = output_dir / "loss_curves_revm.png"
    loss_csv_path = output_dir / "loss_curves_revm.csv"
    run_fit_plot_paths: Dict[str, Path] = {}
    run_fit_csv_paths: Dict[str, Path] = {}
    removed_diagnostic_plot_paths = (
        wave_force_plot_path,
        force_alignment_plot_path,
        oracle_replay_plot_path,
        force_replay_sweep_plot_path,
        coefficient_replay_sweep_plot_path,
        force_post_correction_plot_path,
        force_envelope_plot_path,
        orientation_gate_ablation_plot_path,
        vessel_motion_regime_plot_path,
        teacher_force_alignment_plot_path,
        teacher_force_spectrum_plot_path,
        teacher_force_band_amplitude_plot_path,
        wave_force_lag_plot_path,
        wave_force_multifeature_lag_plot_path,
        orientation_envelope_lag_plot_path,
    )
    for stale_plot_path in removed_diagnostic_plot_paths:
        try:
            if stale_plot_path.exists():
                stale_plot_path.unlink()
        except OSError as exc:
            lg.warning("Could not remove stale diagnostic plot %s: %s", stale_plot_path, exc)
    removed_diagnostic_file_paths = (
        output_dir / "forecast_force_post_correction_revm.csv",
        output_dir / "full_ode_integration_revm.csv",
        output_dir / "full_ode_integration_metrics_revm.json",
        output_dir / "full_ode_integration_revm.png",
    )
    for stale_file_path in removed_diagnostic_file_paths:
        try:
            if stale_file_path.exists():
                stale_file_path.unlink()
        except OSError as exc:
            lg.warning("Could not remove stale diagnostic output %s: %s", stale_file_path, exc)
    forecast_ode_enabled = bool(forecast.get("forecast_ode_enabled", True))
    forecast_line_label = (
        f"Direct LSTM forecast ({forecast_window_description(cfg)})"
        if not forecast_ode_enabled
        else f"ODE rolling forecast ({forecast_window_description(cfg)})"
    )
    forecast_metrics_label = "LSTM forecast" if not forecast_ode_enabled else "ODE forecast"

    fit_plot_pred_deg = mask_forecast_from_fit_curve(pred_deg, forecast)
    forecast_enabled = bool(forecast.get("enabled", False))
    if forecast_enabled:
        forecast_indices = np.asarray(forecast.get("index", []), dtype=int)
        forecast_time = np.asarray(forecast.get("time_s", []), dtype=float)
        forecast_pred = np.asarray(forecast.get("forecast_roll_deg", []), dtype=float)
        forecast_lower = np.asarray(
            forecast.get("uncertainty_lower_deg", np.full(len(forecast_pred), np.nan)),
            dtype=float,
        )
        forecast_upper = np.asarray(
            forecast.get("uncertainty_upper_deg", np.full(len(forecast_pred), np.nan)),
            dtype=float,
        )
        direct_no_ode = forecast.get("direct_no_ode_forecast", {})
        if not isinstance(direct_no_ode, dict):
            direct_no_ode = {}
        show_direct_overlay = bool(
            forecast_ode_enabled and direct_no_ode.get("enabled", False)
        )
        direct_forecast_pred = (
            np.asarray(
                direct_no_ode.get("forecast_roll_deg", np.full(len(forecast_pred), np.nan)),
                dtype=float,
            )
            if show_direct_overlay
            else np.asarray([], dtype=float)
        )
        direct_forecast_metrics = direct_no_ode.get("metrics", {}) if show_direct_overlay else {}
        if not isinstance(direct_forecast_metrics, dict):
            direct_forecast_metrics = {}
    else:
        forecast_indices = np.asarray([], dtype=int)
        forecast_time = np.asarray([], dtype=float)
        forecast_pred = np.asarray([], dtype=float)
        forecast_lower = np.asarray([], dtype=float)
        forecast_upper = np.asarray([], dtype=float)
        direct_forecast_pred = np.asarray([], dtype=float)
        direct_forecast_metrics = {}
    forecast_plot_metrics = forecast.get("metrics", {}) if forecast_enabled else {}
    if not isinstance(forecast_plot_metrics, dict):
        forecast_plot_metrics = {}
    fit_metric_mask = np.isfinite(fit_plot_pred_deg) & np.isfinite(measured_deg)
    fit_plot_metrics = (
        regression_metrics(measured_deg[fit_metric_mask], fit_plot_pred_deg[fit_metric_mask])
        if int(np.sum(fit_metric_mask)) > 2
        else {}
    )

    def forecast_column(values: object) -> np.ndarray:
        column = np.full(len(t), np.nan, dtype=np.float64)
        if not forecast_enabled or len(forecast_indices) == 0:
            return column
        arr = np.asarray(values, dtype=float).reshape(-1)
        n = min(len(arr), len(forecast_indices))
        if n <= 0:
            return column
        idx = forecast_indices[:n]
        valid = (idx >= 0) & (idx < len(column))
        column[idx[valid]] = arr[:n][valid]
        return column

    fit_csv_data: Dict[str, object] = {
        "index": np.arange(len(t), dtype=int),
        "time_s": t,
        "measured_roll_deg": measured_deg,
        "fit_roll_deg": fit_plot_pred_deg,
        "teacher_forced_roll_deg": pred_deg,
        "forecast_roll_deg": forecast_column(forecast_pred),
        "forecast_uncertainty_lower_deg": forecast_column(forecast_lower),
        "forecast_uncertainty_upper_deg": forecast_column(forecast_upper),
    }
    if len(direct_forecast_pred) == len(forecast_indices):
        fit_csv_data["direct_no_ode_forecast_roll_deg"] = forecast_column(direct_forecast_pred)
    pd.DataFrame(fit_csv_data).to_csv(fit_csv_path, index=False)

    if forecast_enabled:
        forecast_zoom_data: Dict[str, object] = {
            "index": forecast_indices,
            "time_s": forecast_time,
            "measured_roll_deg": np.asarray(forecast.get("measured_roll_deg", np.full(len(forecast_indices), np.nan)), dtype=float),
            "forecast_roll_deg": forecast_pred,
            "uncertainty_lower_deg": forecast_lower,
            "uncertainty_upper_deg": forecast_upper,
        }
        if len(direct_forecast_pred) == len(forecast_indices):
            forecast_zoom_data["direct_no_ode_forecast_roll_deg"] = direct_forecast_pred
        pd.DataFrame(forecast_zoom_data).to_csv(forecast_zoom_csv_path, index=False)

    if "lag_grid_s" in data and "lag_scores" in data:
        lag_grid = np.asarray(data.get("lag_grid_s", []), dtype=float)
        lag_scores = np.asarray(data.get("lag_scores", []), dtype=float)
        if len(lag_grid) == len(lag_scores) and len(lag_grid) > 0:
            pd.DataFrame({
                "lag_s": lag_grid,
                "band_limited_correlation": lag_scores,
                "best_wave_lag_s": float(data.get("best_wave_lag_s", float("nan"))),
            }).to_csv(lag_csv_path, index=False)

    if history:
        pd.DataFrame(history).to_csv(loss_csv_path, index=False)

    if bool(forecast.get("enabled", False)):
        pd.DataFrame({
            "index": np.asarray(forecast["index"], dtype=int),
            "time_s": np.asarray(forecast["time_s"], dtype=float),
            "measured_roll_deg": np.asarray(forecast["measured_roll_deg"], dtype=float),
            "forecast_roll_deg": np.asarray(forecast["forecast_roll_deg"], dtype=float),
            "forecast_phi_scaled": np.asarray(forecast["forecast_phi_scaled"], dtype=float),
            "forecast_v_scaled": np.asarray(forecast["forecast_v_scaled"], dtype=float),
            "forecast_force_scaled": np.asarray(forecast["forecast_force_scaled"], dtype=float),
            "forecast_raw_force_before_lag_scaled": np.asarray(
                forecast.get(
                    "forecast_raw_force_before_lag_scaled",
                    np.asarray(forecast["forecast_force_scaled"], dtype=float),
                ),
                dtype=float,
            ),
            "uncorrected_forecast_roll_deg": np.asarray(
                forecast.get(
                    "uncorrected_forecast_roll_deg",
                    np.asarray(forecast["forecast_roll_deg"], dtype=float),
                ),
                dtype=float,
            ),
            "uncorrected_forecast_phi_scaled": np.asarray(
                forecast.get(
                    "uncorrected_forecast_phi_scaled",
                    np.asarray(forecast["forecast_phi_scaled"], dtype=float),
                ),
                dtype=float,
            ),
            "uncorrected_forecast_v_scaled": np.asarray(
                forecast.get(
                    "uncorrected_forecast_v_scaled",
                    np.asarray(forecast["forecast_v_scaled"], dtype=float),
                ),
                dtype=float,
            ),
            "uncorrected_forecast_force_scaled": np.asarray(
                forecast.get(
                    "uncorrected_forecast_force_scaled",
                    np.asarray(forecast["forecast_force_scaled"], dtype=float),
                ),
                dtype=float,
            ),
            "direct_no_ode_forecast_roll_deg": np.asarray(
                forecast.get("direct_no_ode_forecast", {}).get(
                    "forecast_roll_deg",
                    np.full(len(forecast["index"]), np.nan),
                )
                if isinstance(forecast.get("direct_no_ode_forecast", {}), dict)
                else np.full(len(forecast["index"]), np.nan),
                dtype=float,
            ),
            "direct_no_ode_forecast_phi_scaled": np.asarray(
                forecast.get("direct_no_ode_forecast", {}).get(
                    "forecast_phi_scaled",
                    np.full(len(forecast["index"]), np.nan),
                )
                if isinstance(forecast.get("direct_no_ode_forecast", {}), dict)
                else np.full(len(forecast["index"]), np.nan),
                dtype=float,
            ),
            "direct_no_ode_forecast_v_scaled": np.asarray(
                forecast.get("direct_no_ode_forecast", {}).get(
                    "forecast_v_scaled",
                    np.full(len(forecast["index"]), np.nan),
                )
                if isinstance(forecast.get("direct_no_ode_forecast", {}), dict)
                else np.full(len(forecast["index"]), np.nan),
                dtype=float,
            ),
            "forecast_pure_wave_force_scaled": np.asarray(
                forecast.get(
                    "forecast_pure_wave_force_scaled",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "inferred_measured_force_scaled": np.asarray(
                forecast.get(
                    "inferred_measured_force_scaled",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "forecast_non_pure_force_scaled": np.asarray(
                forecast.get(
                    "forecast_non_pure_force_scaled",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "oracle_force_replay_roll_deg": np.asarray(
                oracle_replay.get(
                    "oracle_roll_deg",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "oracle_force_replay_phi_scaled": np.asarray(
                oracle_replay.get(
                    "oracle_phi_scaled",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "oracle_force_replay_force_scaled": np.asarray(
                oracle_replay.get(
                    "oracle_force_scaled",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "forecast_wave_input_rms_scaled": np.asarray(
                forecast.get(
                    "forecast_wave_input_rms_scaled",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "forecast_wave_orientation_gain": np.asarray(
                forecast.get(
                    "forecast_wave_orientation_gain",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "forecast_wave_envelope_gate": np.asarray(
                forecast.get(
                    "forecast_wave_envelope_gate",
                    np.full(len(forecast["index"]), np.nan),
                ),
                dtype=float,
            ),
            "uncertainty_std_deg": np.asarray(forecast.get("uncertainty_std_deg", np.full(len(forecast["index"]), np.nan)), dtype=float),
            "uncertainty_lower_deg": np.asarray(forecast.get("uncertainty_lower_deg", np.full(len(forecast["index"]), np.nan)), dtype=float),
            "uncertainty_upper_deg": np.asarray(forecast.get("uncertainty_upper_deg", np.full(len(forecast["index"]), np.nan)), dtype=float),
        }).to_csv(forecast_csv_path, index=False)

    safe_json_dump(metrics, metrics_path)
    safe_json_dump(cfg, cfg_path)

    if HAS_PLT and bool(cfg.get("save_plots", True)):
        emit_diagnostic_plots = False
        fit_plot_pred_deg = mask_forecast_from_fit_curve(pred_deg, forecast)
        forecast_enabled = bool(forecast.get("enabled", False))
        if forecast_enabled:
            forecast_indices = np.asarray(forecast.get("index", []), dtype=int)
            forecast_time = np.asarray(forecast.get("time_s", []), dtype=float)
            forecast_pred = np.asarray(forecast.get("forecast_roll_deg", []), dtype=float)
            forecast_lower = np.asarray(
                forecast.get("uncertainty_lower_deg", np.full(len(forecast_pred), np.nan)),
                dtype=float,
            )
            forecast_upper = np.asarray(
                forecast.get("uncertainty_upper_deg", np.full(len(forecast_pred), np.nan)),
                dtype=float,
            )
            direct_no_ode = forecast.get("direct_no_ode_forecast", {})
            if not isinstance(direct_no_ode, dict):
                direct_no_ode = {}
            show_direct_overlay = bool(
                forecast_ode_enabled and direct_no_ode.get("enabled", False)
            )
            direct_forecast_pred = (
                np.asarray(
                    direct_no_ode.get("forecast_roll_deg", np.full(len(forecast_pred), np.nan)),
                    dtype=float,
                )
                if show_direct_overlay
                else np.asarray([], dtype=float)
            )
            direct_forecast_metrics = direct_no_ode.get("metrics", {}) if show_direct_overlay else {}
            if not isinstance(direct_forecast_metrics, dict):
                direct_forecast_metrics = {}
        else:
            forecast_indices = np.asarray([], dtype=int)
            forecast_time = np.asarray([], dtype=float)
            forecast_pred = np.asarray([], dtype=float)
            forecast_lower = np.asarray([], dtype=float)
            forecast_upper = np.asarray([], dtype=float)
            direct_forecast_pred = np.asarray([], dtype=float)
            direct_forecast_metrics = {}
        forecast_plot_metrics = forecast.get("metrics", {}) if forecast_enabled else {}
        if not isinstance(forecast_plot_metrics, dict):
            forecast_plot_metrics = {}
        fit_metric_mask = np.isfinite(fit_plot_pred_deg) & np.isfinite(measured_deg)
        fit_plot_metrics = (
            regression_metrics(measured_deg[fit_metric_mask], fit_plot_pred_deg[fit_metric_mask])
            if int(np.sum(fit_metric_mask)) > 2
            else {}
        )

        split_t = None
        forecast_region = data.get("forecast_region")
        if isinstance(forecast_region, (tuple, list, np.ndarray)) and len(forecast_region) >= 1:
            forecast_start_idx = int(forecast_region[0])
            if 0 <= forecast_start_idx < len(t):
                split_t = float(t[forecast_start_idx])
        validation_split = data.get("validation_split", {})
        if split_t is None and isinstance(validation_split, dict):
            split_time = validation_split.get("forecast_start_time_s")
            if split_time is not None and np.isfinite(float(split_time)):
                split_t = float(split_time)
        if (
            split_t is None
            and isinstance(validation_split, dict)
            and str(validation_split.get("mode", "")).startswith("contiguous")
            and len(data["val_idx"]) > 0
        ):
            val_end_idx = int(np.asarray(data["val_idx"], dtype=int)[-1])
            boundary_idx = min(val_end_idx + 1, len(t) - 1)
            split_t = float(t[boundary_idx])
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.36, wspace=0.22)
        ax_full = fig.add_subplot(gs[0, :])
        add_fit_lines(ax_full, t, measured_deg, fit_plot_pred_deg, "PINN LSTM Rev M Comparison - fit and rolling forecast", split_t)
        if forecast_enabled:
            add_forecast_plot_lines(
                ax_full,
                forecast_time,
                forecast_pred,
                forecast_lower,
                forecast_upper,
                cfg,
                mark_start=True,
                direct_no_ode_deg=direct_forecast_pred,
                forecast_label=forecast_line_label,
            )
        ax_full.set_xlabel("Time [s]")
        ax_full.set_ylabel("Roll [deg]")
        ax_full.legend(loc="upper right")
        ax_full.text(
            0.015,
            0.97,
            format_fit_forecast_plot_metrics(
                fit_plot_metrics,
                forecast_plot_metrics,
                direct_forecast_metrics,
                forecast_label=forecast_metrics_label,
            ),
            transform=ax_full.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
        )

        for idx, (lo, hi) in enumerate(choose_zoom_windows(t, 3)):
            ax = fig.add_subplot(gs[1, idx])
            mask = (t >= lo) & (t <= hi)
            add_fit_lines(ax, t[mask], measured_deg[mask], fit_plot_pred_deg[mask], f"Zoom {idx + 1}: {lo:.1f}-{hi:.1f} s")
            forecast_zoom_mask = (forecast_time >= lo) & (forecast_time <= hi)
            if np.any(forecast_zoom_mask):
                add_forecast_plot_lines(
                    ax,
                    forecast_time[forecast_zoom_mask],
                    forecast_pred[forecast_zoom_mask],
                    forecast_lower[forecast_zoom_mask],
                    forecast_upper[forecast_zoom_mask],
                    cfg,
                    direct_no_ode_deg=(
                        direct_forecast_pred[forecast_zoom_mask]
                        if len(direct_forecast_pred) == len(forecast_time)
                        else None
                    ),
                    forecast_label=forecast_line_label,
                )
            ax.set_xlabel("Time [s]")
            if idx == 0:
                ax.set_ylabel("Roll [deg]")

        fig.savefig(plot_path, dpi=180)
        plt.close()

        if emit_diagnostic_plots and len(orientation_gain_all) == len(t):
            parallel_max, side_min = encounter_regime_thresholds(cfg)
            parallel_mask = orientation_gain_all <= parallel_max
            side_on_mask = orientation_gain_all >= side_min
            oblique_mask = ~(parallel_mask | side_on_mask)
            regime_specs = (
                ("Parallel", parallel_mask, "tab:blue", 0.08),
                ("Oblique", oblique_mask, "tab:orange", 0.08),
                ("Side-on", side_on_mask, "tab:green", 0.10),
            )

            def add_regime_background(ax) -> None:
                for label, mask, color, alpha in regime_specs:
                    truth = np.asarray(mask, dtype=bool) & np.isfinite(t)
                    if not np.any(truth):
                        continue
                    edges = np.flatnonzero(np.diff(np.r_[False, truth, False].astype(int)) != 0)
                    labelled = False
                    for start, stop in zip(edges[0::2], edges[1::2]):
                        if stop <= start:
                            continue
                        ax.axvspan(
                            float(t[start]),
                            float(t[stop - 1]),
                            color=color,
                            alpha=alpha,
                            linewidth=0.0,
                            label=f"{label} encounter" if not labelled else None,
                        )
                        labelled = True

            state_groups = data.get("state_feature_indices", {})
            x_cols = (
                state_groups.get("x_position", [])
                if isinstance(state_groups, dict)
                else []
            )
            y_cols = (
                state_groups.get("y_position", [])
                if isinstance(state_groups, dict)
                else []
            )
            X_np = np.asarray(data.get("X", []), dtype=np.float64)
            has_track = (
                X_np.ndim == 2
                and len(x_cols) > 0
                and len(y_cols) > 0
                and max(max(int(c) for c in x_cols), max(int(c) for c in y_cols)) < X_np.shape[1]
            )
            nrows = 3 if has_track else 2
            fig, axes = plt.subplots(
                nrows,
                1,
                figsize=(14, 3.7 * nrows),
                sharex=not has_track,
                constrained_layout=True,
            )
            axes_arr = np.atleast_1d(axes)
            ax_roll = axes_arr[0]
            add_regime_background(ax_roll)
            add_fit_lines(
                ax_roll,
                t,
                measured_deg,
                fit_plot_pred_deg,
                "Rev M vessel motion and wave-encounter regimes",
                split_t,
            )
            if forecast_enabled:
                add_forecast_plot_lines(
                    ax_roll,
                    forecast_time,
                    forecast_pred,
                    forecast_lower,
                    forecast_upper,
                    cfg,
                    mark_start=True,
                    forecast_label=forecast_line_label,
                )
            ax_roll.set_ylabel("Roll [deg]")
            ax_roll.grid(True, alpha=0.3)
            ax_roll.legend(loc="upper right")

            regime_metrics = metrics.get("encounter_regimes", {})
            forecast_regimes = (
                regime_metrics.get("forecast", {})
                if isinstance(regime_metrics, dict) and isinstance(regime_metrics.get("forecast", {}), dict)
                else {}
            )
            ax_roll.text(
                0.015,
                0.97,
                (
                    f"Parallel <= {parallel_max:.2f}\n"
                    f"Side-on >= {side_min:.2f}\n"
                    "Forecast fractions\n"
                    f"  parallel: {float(forecast_regimes.get('parallel_fraction', float('nan'))):.3f}\n"
                    f"  oblique: {float(forecast_regimes.get('oblique_fraction', float('nan'))):.3f}\n"
                    f"  side-on: {float(forecast_regimes.get('side_on_fraction', float('nan'))):.3f}"
                ),
                transform=ax_roll.transAxes,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
            )

            if has_track:
                ax_track = axes_arr[1]
                x_track = np.nanmean(X_np[:, [int(c) for c in x_cols]], axis=1)
                y_track = np.nanmean(X_np[:, [int(c) for c in y_cols]], axis=1)
                ax_track.plot(x_track, y_track, color="0.65", linewidth=0.8, label="Vessel track")
                for label, mask, color, _ in regime_specs:
                    finite = (
                        np.asarray(mask, dtype=bool)
                        & np.isfinite(x_track)
                        & np.isfinite(y_track)
                    )
                    if np.any(finite):
                        ax_track.scatter(
                            x_track[finite],
                            y_track[finite],
                            s=7,
                            color=color,
                            alpha=0.65,
                            label=label,
                        )
                if forecast_enabled and len(forecast_indices) > 0:
                    idx = forecast_indices[
                        (forecast_indices >= 0)
                        & (forecast_indices < len(x_track))
                    ]
                    if len(idx) > 0:
                        ax_track.scatter(
                            x_track[idx],
                            y_track[idx],
                            s=18,
                            facecolors="none",
                            edgecolors="tab:red",
                            linewidths=0.8,
                            label="Forecast region",
                        )
                ax_track.set_title("Scaled vessel track by encounter regime")
                ax_track.set_xlabel("Scaled x position")
                ax_track.set_ylabel("Scaled y position")
                ax_track.grid(True, alpha=0.3)
                ax_track.legend(loc="best")
                ax_gain = axes_arr[2]
            else:
                ax_gain = axes_arr[1]

            add_regime_background(ax_gain)
            ax_gain.plot(t, orientation_gain_all, color="black", linewidth=1.1, label="Wave-orientation effect gain")
            ax_gain.axhline(parallel_max, color="tab:blue", linestyle="--", linewidth=0.9, label="parallel/oblique threshold")
            ax_gain.axhline(side_min, color="tab:green", linestyle="--", linewidth=0.9, label="oblique/side-on threshold")
            if split_t is not None:
                ax_gain.axvline(split_t, color="tab:blue", linestyle="--", linewidth=1.0, label="validation/forecast boundary")
            if forecast_enabled and len(forecast_time) > 0:
                ax_gain.axvline(
                    float(forecast_time[0]),
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    label="forecast begins",
                )
            ax_gain.set_xlabel("Time [s]")
            ax_gain.set_ylabel("Encounter gain")
            ax_gain.set_ylim(-0.03, 1.03)
            ax_gain.grid(True, alpha=0.3)
            ax_gain.legend(loc="best")
            plt.close(fig)

        run_segments = normalise_segments(data.get("run_segments"), len(t))
        run_names = [str(v) for v in list(data.get("run_names", []))]
        if len(run_names) < len(run_segments):
            run_names.extend(f"run_{i + 1}" for i in range(len(run_names), len(run_segments)))
        local_t = np.asarray(data.get("run_local_time", t), dtype=float)
        if local_t.shape[:1] != t.shape[:1]:
            local_t = t
        per_run_metrics = metrics.get("per_run", {}) if isinstance(metrics.get("per_run", {}), dict) else {}
        for name, (lo, hi) in zip(run_names, run_segments):
            sl = slice(int(lo), int(hi) + 1)
            run_forecast_mask = (forecast_indices >= int(lo)) & (forecast_indices <= int(hi))
            run_path = output_dir / f"roll_fit_{safe_filename_token(name)}_revm.png"
            run_csv_path = output_dir / f"roll_fit_{safe_filename_token(name)}_revm.csv"
            run_csv_data: Dict[str, object] = {
                "index": np.arange(int(lo), int(hi) + 1, dtype=int),
                "run_local_time_s": local_t[sl],
                "time_s": t[sl],
                "measured_roll_deg": measured_deg[sl],
                "fit_roll_deg": fit_plot_pred_deg[sl],
                "teacher_forced_roll_deg": pred_deg[sl],
                "forecast_roll_deg": forecast_column(forecast_pred)[sl],
                "forecast_uncertainty_lower_deg": forecast_column(forecast_lower)[sl],
                "forecast_uncertainty_upper_deg": forecast_column(forecast_upper)[sl],
            }
            if len(direct_forecast_pred) == len(forecast_indices):
                run_csv_data["direct_no_ode_forecast_roll_deg"] = forecast_column(direct_forecast_pred)[sl]
            pd.DataFrame(run_csv_data).to_csv(run_csv_path, index=False)
            fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
            add_fit_lines(
                ax,
                local_t[sl],
                measured_deg[sl],
                fit_plot_pred_deg[sl],
                f"PINN LSTM Rev M Comparison fit{' and rolling forecast' if np.any(run_forecast_mask) else ''} - {name}",
            )
            if np.any(run_forecast_mask):
                run_forecast_indices = forecast_indices[run_forecast_mask]
                add_forecast_plot_lines(
                    ax,
                    local_t[run_forecast_indices],
                    forecast_pred[run_forecast_mask],
                    forecast_lower[run_forecast_mask],
                    forecast_upper[run_forecast_mask],
                    cfg,
                    mark_start=True,
                    direct_no_ode_deg=(
                        direct_forecast_pred[run_forecast_mask]
                        if len(direct_forecast_pred) == len(forecast_indices)
                        else None
                    ),
                    forecast_label=forecast_line_label,
                )
            ax.set_xlabel("Run-local time [s]")
            ax.set_ylabel("Roll [deg]")
            ax.legend(loc="best")
            run_metric = per_run_metrics.get(str(name), {})
            if isinstance(run_metric, dict):
                if np.any(run_forecast_mask):
                    run_fit_mask = np.isfinite(fit_plot_pred_deg[sl]) & np.isfinite(measured_deg[sl])
                    if int(np.sum(run_fit_mask)) > 2:
                        run_metric = regression_metrics(
                            measured_deg[sl][run_fit_mask],
                            fit_plot_pred_deg[sl][run_fit_mask],
                        )
                ax.text(
                    0.015,
                    0.97,
                    format_fit_forecast_plot_metrics(
                        run_metric,
                        forecast_plot_metrics if np.any(run_forecast_mask) else None,
                        direct_forecast_metrics if np.any(run_forecast_mask) else None,
                        forecast_label=forecast_metrics_label,
                    ),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
                )
            fig.savefig(run_path, dpi=180)
            plt.close(fig)
            run_fit_plot_paths[str(name)] = run_path
            run_fit_csv_paths[str(name)] = run_csv_path

        tf_t = np.asarray(teacher_force_alignment.get("time_s", []), dtype=float)
        tf_inferred = np.asarray(
            teacher_force_alignment.get("inferred_measured_force_scaled", []),
            dtype=float,
        )
        tf_total = np.asarray(
            teacher_force_alignment.get("teacher_forced_total_force_scaled", []),
            dtype=float,
        )
        tf_pure = np.asarray(
            teacher_force_alignment.get("teacher_forced_pure_wave_force_scaled", []),
            dtype=float,
        )
        tf_metrics = teacher_force_alignment.get("metrics", {})
        if not isinstance(tf_metrics, dict):
            tf_metrics = {}
        split_specs = [
            ("Train", np.asarray(data.get("train_idx", []), dtype=int), tf_metrics.get("train", {})),
            ("Validation", np.asarray(data.get("val_idx", []), dtype=int), tf_metrics.get("validation", {})),
        ]
        split_specs = [
            (name, idx, metric)
            for name, idx, metric in split_specs
            if len(idx) > 0 and len(tf_t) > 0
        ]
        if emit_diagnostic_plots and split_specs and len(tf_t) == len(tf_inferred) == len(tf_total) == len(tf_pure):
            fig, axes = plt.subplots(
                len(split_specs),
                1,
                figsize=(13, 4.5 * len(split_specs)),
                sharex=False,
                constrained_layout=True,
            )
            axes_arr = np.atleast_1d(axes)

            def force_metric_text(split_metric: object) -> str:
                metric = split_metric if isinstance(split_metric, dict) else {}
                total_metric = metric.get("total_force", {}) if isinstance(metric.get("total_force", {}), dict) else {}
                pure_metric = metric.get("pure_wave_force", {}) if isinstance(metric.get("pure_wave_force", {}), dict) else {}
                return (
                    "Total learned vs inferred\n"
                    f"  lag: {float(total_metric.get('phase_lag_s', float('nan'))):.4f} s\n"
                    f"  corr: {float(total_metric.get('phase_corr', float('nan'))):.3f}\n"
                    f"  amp ratio: {float(total_metric.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  RMSE: {float(total_metric.get('rmse_scaled', float('nan'))):.4g}\n"
                    "Pure wave vs inferred\n"
                    f"  lag: {float(pure_metric.get('phase_lag_s', float('nan'))):.4f} s\n"
                    f"  corr: {float(pure_metric.get('phase_corr', float('nan'))):.3f}\n"
                    f"  amp ratio: {float(pure_metric.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  RMSE: {float(pure_metric.get('rmse_scaled', float('nan'))):.4g}"
                )

            for ax, (split_name, raw_idx, split_metric) in zip(axes_arr, split_specs):
                idx = raw_idx[(raw_idx >= 0) & (raw_idx < len(tf_t))]
                ax.plot(
                    tf_t[idx],
                    tf_inferred[idx],
                    color="black",
                    linewidth=1.15,
                    label="Inferred measured force",
                )
                ax.plot(
                    tf_t[idx],
                    tf_total[idx],
                    color="tab:red",
                    linewidth=1.05,
                    alpha=0.9,
                    label="Teacher-forced total learned force",
                )
                ax.plot(
                    tf_t[idx],
                    tf_pure[idx],
                    color="tab:cyan",
                    linewidth=1.0,
                    alpha=0.9,
                    label="Teacher-forced pure wave force",
                )
                ax.axhline(0.0, color="0.4", linewidth=0.8)
                ax.set_title(f"Rev M teacher-forced force alignment - {split_name}")
                ax.set_ylabel("Scaled force")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower left")
                ax.text(
                    0.015,
                    0.97,
                    force_metric_text(split_metric),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
            axes_arr[-1].set_xlabel("Time [s]")
            plt.close(fig)

            fig, axes = plt.subplots(
                len(split_specs),
                1,
                figsize=(13, 4.5 * len(split_specs)),
                sharex=False,
                constrained_layout=True,
            )
            axes_arr = np.atleast_1d(axes)

            def plot_spectrum_line(ax, freq: np.ndarray, amp: np.ndarray,
                                   color: str, label: str, linewidth: float) -> None:
                mask = np.isfinite(freq) & np.isfinite(amp) & (freq > 0.0) & (amp > 0.0)
                if np.any(mask):
                    ax.semilogy(freq[mask], amp[mask], color=color, linewidth=linewidth, label=label)

            def spectral_metric_text(split_metric: object) -> Tuple[str, float, float]:
                metric = split_metric if isinstance(split_metric, dict) else {}
                total_metric = metric.get("spectral_total_force", {}) if isinstance(metric.get("spectral_total_force", {}), dict) else {}
                pure_metric = metric.get("spectral_pure_wave_force", {}) if isinstance(metric.get("spectral_pure_wave_force", {}), dict) else {}
                band_low = float(total_metric.get("band_low_hz", float("nan")))
                band_high = float(total_metric.get("band_high_hz", float("nan")))
                text = (
                    "Total learned spectrum\n"
                    f"  ref dom: {float(total_metric.get('reference_dominant_period_s', float('nan'))):.3f} s\n"
                    f"  learned dom: {float(total_metric.get('candidate_dominant_period_s', float('nan'))):.3f} s\n"
                    f"  band energy ratio: {float(total_metric.get('band_energy_ratio', float('nan'))):.3f}\n"
                    f"  total energy ratio: {float(total_metric.get('total_energy_ratio', float('nan'))):.3f}\n"
                    "Pure wave spectrum\n"
                    f"  learned dom: {float(pure_metric.get('candidate_dominant_period_s', float('nan'))):.3f} s\n"
                    f"  band energy ratio: {float(pure_metric.get('band_energy_ratio', float('nan'))):.3f}\n"
                    f"  total energy ratio: {float(pure_metric.get('total_energy_ratio', float('nan'))):.3f}"
                )
                return text, band_low, band_high

            for ax, (split_name, raw_idx, split_metric) in zip(axes_arr, split_specs):
                idx = raw_idx[(raw_idx >= 0) & (raw_idx < len(tf_t))]
                ref_freq, ref_amp, _ = one_sided_amplitude_spectrum(tf_inferred[idx], tf_t[idx])
                total_freq, total_amp, _ = one_sided_amplitude_spectrum(tf_total[idx], tf_t[idx])
                pure_freq, pure_amp, _ = one_sided_amplitude_spectrum(tf_pure[idx], tf_t[idx])
                plot_spectrum_line(ax, ref_freq, ref_amp, "black", "Inferred measured force", 1.25)
                plot_spectrum_line(ax, total_freq, total_amp, "tab:red", "Teacher-forced total learned force", 1.15)
                plot_spectrum_line(ax, pure_freq, pure_amp, "tab:cyan", "Teacher-forced pure wave force", 1.05)
                text, band_low, band_high = spectral_metric_text(split_metric)
                if math.isfinite(band_low) and math.isfinite(band_high):
                    ax.axvspan(band_low, band_high, color="0.65", alpha=0.18, label="configured roll/wave band")
                    x_high = min(
                        max(float(np.nanmax(ref_freq)) if len(ref_freq) else band_high, band_high * 2.5),
                        band_high * 2.5,
                    )
                    ax.set_xlim(0.0, max(x_high, band_high * 1.05))
                ax.set_title(f"Rev M teacher-forced force spectrum - {split_name}")
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("One-sided amplitude")
                ax.grid(True, alpha=0.3, which="both")
                ax.legend(loc="lower left")
                ax.text(
                    0.985,
                    0.03,
                    text,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="right",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
            plt.close(fig)

            band_specs = [
                (0.45, "0.45 Hz"),
                (1.70, "1.70 Hz"),
                (3.20, "3.20 Hz"),
            ]
            n_rows = max(1, len(split_specs)) * 2
            fig, axes = plt.subplots(
                n_rows,
                len(band_specs),
                figsize=(16, 3.2 * n_rows),
                sharex=False,
                constrained_layout=True,
            )
            axes_arr = np.asarray(axes, dtype=object)
            if axes_arr.ndim == 1:
                axes_arr = axes_arr.reshape(n_rows, len(band_specs))

            def _band_env(values: np.ndarray, tt: np.ndarray, center_hz: float) -> np.ndarray:
                period_s = 1.0 / max(float(center_hz), 1.0e-8)
                band = bandpass_time(
                    values,
                    tt,
                    period_s,
                    low_period_factor=1.25,
                    high_period_factor=0.80,
                )
                return rolling_rms_time(band, tt, max(1.2, 2.0 * period_s))

            def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
                mask = np.isfinite(a) & np.isfinite(b)
                if int(np.sum(mask)) < 3:
                    return float("nan")
                aa = a[mask] - float(np.mean(a[mask]))
                bb = b[mask] - float(np.mean(b[mask]))
                denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
                return float(np.dot(aa, bb) / denom) if denom > 1.0e-12 else float("nan")

            for split_i, (split_name, raw_idx, _) in enumerate(split_specs):
                idx = raw_idx[(raw_idx >= 0) & (raw_idx < len(tf_t))]
                tt = tf_t[idx]
                ref = tf_inferred[idx]
                total = tf_total[idx]
                pure = tf_pure[idx]
                for band_i, (center_hz, label) in enumerate(band_specs):
                    ax_env = axes_arr[2 * split_i, band_i]
                    ax_ratio = axes_arr[2 * split_i + 1, band_i]
                    if len(idx) < 8:
                        ax_env.set_axis_off()
                        ax_ratio.set_axis_off()
                        continue
                    ref_env = _band_env(ref, tt, center_hz)
                    total_env = _band_env(total, tt, center_hz)
                    pure_env = _band_env(pure, tt, center_hz)
                    denom = np.maximum(ref_env, 1.0e-8)
                    total_ratio = total_env / denom
                    pure_ratio = pure_env / denom
                    finite_total = total_ratio[np.isfinite(total_ratio)]
                    finite_pure = pure_ratio[np.isfinite(pure_ratio)]
                    total_med = float(np.median(finite_total)) if len(finite_total) else float("nan")
                    pure_med = float(np.median(finite_pure)) if len(finite_pure) else float("nan")
                    total_corr = _safe_corr(ref_env, total_env)
                    pure_corr = _safe_corr(ref_env, pure_env)

                    ax_env.plot(tt, ref_env, color="black", linewidth=1.2, label="Inferred envelope")
                    ax_env.plot(tt, total_env, color="tab:red", linewidth=1.0, label="Total learned envelope")
                    ax_env.plot(tt, pure_env, color="tab:cyan", linewidth=0.95, label="Pure wave envelope")
                    ax_env.set_title(f"{split_name} band envelope: {label}")
                    ax_env.set_ylabel("RMS envelope")
                    ax_env.grid(True, alpha=0.3)
                    ax_env.text(
                        0.015,
                        0.95,
                        (
                            f"total med ratio: {total_med:.3f}\n"
                            f"total env corr: {total_corr:.3f}\n"
                            f"pure med ratio: {pure_med:.3f}\n"
                            f"pure env corr: {pure_corr:.3f}"
                        ),
                        transform=ax_env.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.86,
                            "boxstyle": "round,pad=0.25",
                        },
                    )
                    if split_i == 0 and band_i == len(band_specs) - 1:
                        ax_env.legend(loc="upper right", fontsize=8)

                    ax_ratio.plot(tt, total_ratio, color="tab:red", linewidth=1.0, label="Total / inferred")
                    ax_ratio.plot(tt, pure_ratio, color="tab:cyan", linewidth=0.95, label="Pure / inferred")
                    ax_ratio.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
                    ax_ratio.set_title(f"{split_name} local amplitude ratio: {label}")
                    ax_ratio.set_xlabel("Time [s]")
                    ax_ratio.set_ylabel("Amplitude ratio")
                    ax_ratio.set_ylim(0.0, min(4.0, max(1.5, float(np.nanpercentile(total_ratio, 95)) * 1.1)))
                    ax_ratio.grid(True, alpha=0.3)
                    if split_i == 0 and band_i == len(band_specs) - 1:
                        ax_ratio.legend(loc="upper right", fontsize=8)

            fig.suptitle("Rev M teacher-forced force band-specific amplitude diagnostic", fontsize=14)
            plt.close(fig)

        if bool(forecast.get("enabled", False)):
            f_t = np.asarray(forecast["time_s"], dtype=float)
            f_measured = np.asarray(forecast["measured_roll_deg"], dtype=float)
            f_pred = np.asarray(forecast["forecast_roll_deg"], dtype=float)
            direct_forecast_payload = (
                forecast.get("direct_no_ode_forecast", {})
                if isinstance(forecast.get("direct_no_ode_forecast", {}), dict)
                else {}
            )
            show_direct_overlay = bool(
                forecast_ode_enabled and direct_forecast_payload.get("enabled", False)
            )
            f_direct_no_ode = (
                np.asarray(
                    direct_forecast_payload.get("forecast_roll_deg", np.full(len(f_pred), np.nan)),
                    dtype=float,
                )
                if show_direct_overlay
                else np.asarray([], dtype=float)
            )
            f_direct_metrics = direct_forecast_payload.get("metrics", {}) if show_direct_overlay else {}
            if not isinstance(f_direct_metrics, dict):
                f_direct_metrics = {}
            f_uncorrected_pred = np.asarray(
                forecast.get("uncorrected_forecast_roll_deg", f_pred),
                dtype=float,
            )
            f_start = float(forecast["start_time_s"])
            f_end = float(forecast["end_time_s"])
            context_s = max(0.0, float(cfg.get("forecast_plot_context_s", 1.0)))
            context_mask = (t >= f_start - context_s) & (t <= f_end + 0.25 * context_s)

            fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
            ax.plot(t[context_mask], measured_deg[context_mask], color="0.35", linewidth=1.0, label="Measured roll outside forecast")
            ax.plot(f_t, f_measured, color="tab:blue", linewidth=1.2, label="Measured roll hidden during forecast")
            if "uncertainty_lower_deg" in forecast and "uncertainty_upper_deg" in forecast:
                ax.fill_between(
                    f_t,
                    np.asarray(forecast["uncertainty_lower_deg"], dtype=float),
                    np.asarray(forecast["uncertainty_upper_deg"], dtype=float),
                    color="0.55",
                    alpha=0.28,
                    linewidth=0.0,
                    label=f"{100.0 * float(cfg.get('forecast_uncertainty_confidence', 0.95)):.0f}% empirical uncertainty",
                )
            ax.plot(
                f_t,
                f_pred,
                color="tab:red",
                linewidth=1.6,
                label=forecast_line_label,
            )
            if len(f_direct_no_ode) == len(f_t):
                ax.plot(
                    f_t,
                    f_direct_no_ode,
                    color="tab:purple",
                    linewidth=1.25,
                    alpha=0.90,
                    linestyle="--",
                    label="Direct forecast (no ODE)",
                )
            ax.axvspan(f_start, f_end, color="tab:red", alpha=0.08, label="forecast window")
            ax.axvline(f_start, color="tab:red", linestyle="--", linewidth=1.0)
            ax.axvline(f_end, color="tab:red", linestyle="--", linewidth=1.0)
            ax.set_title(f"Rev M rolling vessel-motion forecast: {f_start:.2f}-{f_end:.2f} s")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Roll [deg]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            f_metrics = forecast.get("metrics", {})
            if isinstance(f_metrics, dict):
                ax.text(
                    0.015,
                    0.97,
                    (
                        f"{forecast_metrics_label}\n"
                        f"R2 = {float(f_metrics.get('r2', float('nan'))):.4f}\n"
                        f"RMSE = {float(f_metrics.get('rmse_deg', float('nan'))):.3f} deg\n"
                        f"MAE = {float(f_metrics.get('mae_deg', float('nan'))):.3f} deg"
                    )
                    + (
                        f"\n\nNo ODE direct\n"
                        f"R2 = {float(f_direct_metrics.get('r2', float('nan'))):.4f}\n"
                        f"RMSE = {float(f_direct_metrics.get('rmse_deg', float('nan'))):.3f} deg"
                        if f_direct_metrics else ""
                    ),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
                )
            fig.savefig(forecast_plot_path, dpi=180)
            plt.close(fig)

            f_wave_force = np.asarray(
                forecast.get(
                    "forecast_pure_wave_force_scaled",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            f_total_force = np.asarray(
                forecast.get("forecast_force_scaled", np.full(len(f_t), np.nan)),
                dtype=float,
            )
            f_raw_force_before_lag = np.asarray(
                forecast.get(
                    "forecast_raw_force_before_lag_scaled",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            f_wave_input_rms = np.asarray(
                forecast.get(
                    "forecast_wave_input_rms_scaled",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            f_orientation_gain = np.asarray(
                forecast.get(
                    "forecast_wave_orientation_gain",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            f_envelope_gate = np.asarray(
                forecast.get(
                    "forecast_wave_envelope_gate",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 7),
                sharex=True,
                constrained_layout=True,
            )
            axes[0].plot(
                f_t,
                f_wave_input_rms,
                color="tab:blue",
                linewidth=1.3,
                label="Wave-channel RMS in model input",
            )
            has_orientation_gain = bool(np.any(np.isfinite(f_orientation_gain)))
            has_envelope_gate = bool(np.any(np.isfinite(f_envelope_gate)))
            if has_orientation_gain or has_envelope_gate:
                gain_ax = axes[0].twinx()
                if has_orientation_gain:
                    gain_ax.plot(
                        f_t,
                        f_orientation_gain,
                        color="tab:green",
                        linewidth=1.2,
                        alpha=0.80,
                        label="Wave-orientation force gain",
                    )
                if has_envelope_gate:
                    gain_ax.plot(
                        f_t,
                        f_envelope_gate,
                        color="tab:purple",
                        linewidth=1.2,
                        alpha=0.90,
                        label="Learned envelope gate",
                    )
                gain_ax.set_ylabel("Force gain")
                finite_gains = np.concatenate([
                    f_orientation_gain[np.isfinite(f_orientation_gain)],
                    f_envelope_gate[np.isfinite(f_envelope_gate)],
                ])
                if len(finite_gains):
                    gain_ax.set_ylim(
                        0.0,
                        max(1.05, min(3.0, float(np.nanmax(finite_gains)) * 1.10)),
                    )
                lines, labels = axes[0].get_legend_handles_labels()
                gain_lines, gain_labels = gain_ax.get_legend_handles_labels()
                axes[0].legend(lines + gain_lines, labels + gain_labels, loc="best")
            else:
                axes[0].legend(loc="best")
            axes[0].set_ylabel("Scaled wave input")
            axes[0].set_title("Rev M forecast-region wave-force verification")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(
                f_t,
                f_wave_force,
                color="tab:cyan",
                linewidth=1.3,
                label="Explicit pure wave-force base",
            )
            if (
                np.any(np.isfinite(f_raw_force_before_lag))
                and len(f_raw_force_before_lag) == len(f_t)
                and not np.allclose(
                    np.nan_to_num(f_raw_force_before_lag, nan=0.0),
                    np.nan_to_num(f_total_force, nan=0.0),
                    atol=1.0e-10,
                    rtol=1.0e-10,
                )
            ):
                axes[1].plot(
                    f_t,
                    f_raw_force_before_lag,
                    color="tab:orange",
                    linewidth=1.0,
                    alpha=0.75,
                    linestyle="--",
                    label="Raw total force before lag",
                )
            axes[1].plot(
                f_t,
                f_total_force,
                color="tab:red",
                linewidth=1.2,
                alpha=0.85,
                label=(
                    "Total force used by forecast ODE"
                    if forecast_ode_enabled
                    else "LSTM force diagnostic"
                ),
            )
            axes[1].axhline(0.0, color="0.4", linewidth=0.8)
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Scaled force")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best")
            verification = forecast.get("wave_force_verification", {})
            if isinstance(verification, dict):
                raw_lag_value = forecast.get("forecast_force_lag_s", None)
                lag_text = "disabled" if raw_lag_value is None else f"{float(raw_lag_value):.3f} s"
                axes[1].text(
                    0.015,
                    0.97,
                    f"status: {verification.get('status', 'unknown')}\n"
                    f"wave RMS: {float(verification.get('pure_wave_force_rms_scaled', float('nan'))):.5g}\n"
                    f"orientation gain: {float(verification.get('wave_orientation_gain_min', float('nan'))):.3f}-"
                    f"{float(verification.get('wave_orientation_gain_max', float('nan'))):.3f}\n"
                    f"envelope gate: {float(verification.get('wave_envelope_gate_min', float('nan'))):.3f}-"
                    f"{float(verification.get('wave_envelope_gate_max', float('nan'))):.3f}\n"
                    f"shape corr RMS: {float(verification.get('wave_shape_correction_force_rms_scaled', float('nan'))):.5g}\n"
                    f"forecast force lag: {lag_text}\n"
                    f"zero-wave ΔRMS: {float(verification.get('counterfactual_force_delta_rms_scaled', float('nan'))):.5g}\n"
                    f"total RMS: {float(verification.get('total_force_rms_scaled', float('nan'))):.5g}",
                    transform=axes[1].transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
            plt.close(fig)

            inferred_force = np.asarray(
                forecast.get(
                    "inferred_measured_force_scaled",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            non_pure_force = np.asarray(
                forecast.get(
                    "forecast_non_pure_force_scaled",
                    np.full(len(f_t), np.nan),
                ),
                dtype=float,
            )
            f_measured_roll = np.asarray(
                forecast.get("measured_roll_deg", np.full(len(f_t), np.nan)),
                dtype=float,
            )
            force_alignment = forecast.get("force_alignment", {})
            if not isinstance(force_alignment, dict):
                force_alignment = {}
            roll_diag = force_alignment.get("roll", {})
            pure_diag = force_alignment.get("pure_wave_force", {})
            total_diag = force_alignment.get("total_force", {})
            if not isinstance(roll_diag, dict):
                roll_diag = {}
            if not isinstance(pure_diag, dict):
                pure_diag = {}
            if not isinstance(total_diag, dict):
                total_diag = {}

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
                constrained_layout=True,
            )
            axes[0].plot(
                f_t,
                f_measured_roll,
                color="tab:blue",
                linewidth=1.2,
                label="Measured roll",
            )
            axes[0].plot(
                f_t,
                f_pred,
                color="tab:red",
                linewidth=1.3,
                label="Forecast roll",
            )
            axes[0].axvline(float(f_t[0]), color="0.45", linestyle="--", linewidth=0.9)
            axes[0].set_ylabel("Roll [deg]")
            axes[0].set_title("Rev M forecast force/roll alignment diagnostic")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="best")
            axes[0].text(
                0.015,
                0.97,
                f"Roll lag: {float(roll_diag.get('phase_lag_s', float('nan'))):.4f} s\n"
                f"Roll corr: {float(roll_diag.get('phase_corr', float('nan'))):.3f}\n"
                f"Roll amp ratio: {float(roll_diag.get('amplitude_ratio', float('nan'))):.3f}\n"
                f"Roll RMSE: {float(forecast_plot_metrics.get('rmse_deg', float('nan'))):.3f} deg",
                transform=axes[0].transAxes,
                va="top",
                ha="left",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.88,
                    "boxstyle": "round,pad=0.35",
                },
            )

            axes[1].plot(
                f_t,
                inferred_force,
                color="black",
                linewidth=1.4,
                label="Inferred measured force",
            )
            axes[1].plot(
                f_t,
                f_wave_force,
                color="tab:cyan",
                linewidth=1.2,
                label="Pure learned wave force",
            )
            axes[1].plot(
                f_t,
                f_total_force,
                color="tab:red",
                linewidth=1.1,
                alpha=0.85,
                label="Total learned force",
            )
            if np.any(np.isfinite(non_pure_force)):
                axes[1].plot(
                    f_t,
                    non_pure_force,
                    color="tab:purple",
                    linewidth=1.0,
                    alpha=0.75,
                    linestyle="--",
                    label="Total minus pure wave",
                )
            axes[1].axhline(0.0, color="0.4", linewidth=0.8)
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Scaled force")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best")
            axes[1].text(
                0.015,
                0.97,
                "Pure wave vs inferred\n"
                f"  lag: {float(pure_diag.get('phase_lag_s', float('nan'))):.4f} s\n"
                f"  corr: {float(pure_diag.get('phase_corr', float('nan'))):.3f}\n"
                f"  amp ratio: {float(pure_diag.get('amplitude_ratio', float('nan'))):.3f}\n"
                f"  RMSE: {float(pure_diag.get('rmse_scaled', float('nan'))):.4g}\n"
                "Total force vs inferred\n"
                f"  lag: {float(total_diag.get('phase_lag_s', float('nan'))):.4f} s\n"
                f"  corr: {float(total_diag.get('phase_corr', float('nan'))):.3f}\n"
                f"  amp ratio: {float(total_diag.get('amplitude_ratio', float('nan'))):.3f}\n"
                f"  RMSE: {float(total_diag.get('rmse_scaled', float('nan'))):.4g}",
                transform=axes[1].transAxes,
                va="top",
                ha="left",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.88,
                    "boxstyle": "round,pad=0.35",
                },
            )
            plt.close(fig)

            if bool(forecast_force_envelope.get("enabled", False)):
                env_t = np.asarray(forecast_force_envelope.get("time_s", []), dtype=float)
                inferred_env = np.asarray(
                    forecast_force_envelope.get("inferred_envelope", []),
                    dtype=float,
                )
                raw_env = np.asarray(
                    forecast_force_envelope.get("raw_force_envelope", []),
                    dtype=float,
                )
                corrected_env = np.asarray(
                    forecast_force_envelope.get("corrected_force_envelope", []),
                    dtype=float,
                )
                pure_env = np.asarray(
                    forecast_force_envelope.get("pure_wave_force_envelope", []),
                    dtype=float,
                )
                geometry_env = np.asarray(
                    forecast_force_envelope.get("geometry_proxy_envelope", []),
                    dtype=float,
                )
                required_gain = np.asarray(
                    forecast_force_envelope.get("required_gain", []),
                    dtype=float,
                )
                orientation_gain = np.asarray(
                    forecast_force_envelope.get("orientation_gain", []),
                    dtype=float,
                )
                learned_envelope_gate = np.asarray(
                    forecast_force_envelope.get("learned_envelope_gate", []),
                    dtype=float,
                )
                env_metrics = forecast_force_envelope.get("metrics", {})
                if not isinstance(env_metrics, dict):
                    env_metrics = {}

                def scale_for_overlay(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
                    signal = np.asarray(signal, dtype=float)
                    reference = np.asarray(reference, dtype=float)
                    finite_signal = np.isfinite(signal)
                    finite_reference = np.isfinite(reference)
                    if int(np.sum(finite_signal)) < 2 or int(np.sum(finite_reference)) < 2:
                        return np.full_like(signal, np.nan, dtype=np.float64)
                    signal_level = float(np.nanmedian(np.abs(signal[finite_signal])))
                    reference_level = float(np.nanmedian(np.abs(reference[finite_reference])))
                    if signal_level <= 1.0e-12 or not math.isfinite(signal_level):
                        return np.full_like(signal, np.nan, dtype=np.float64)
                    return signal * (reference_level / signal_level)

                geometry_overlay = scale_for_overlay(geometry_env, inferred_env)
                raw_metric = env_metrics.get("raw_force_envelope", {})
                corrected_metric = env_metrics.get("corrected_force_envelope", {})
                geometry_metric = env_metrics.get("geometry_proxy_envelope", {})
                if not isinstance(raw_metric, dict):
                    raw_metric = {}
                if not isinstance(corrected_metric, dict):
                    corrected_metric = {}
                if not isinstance(geometry_metric, dict):
                    geometry_metric = {}

                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(12, 8),
                    sharex=True,
                    constrained_layout=True,
                )
                axes[0].plot(
                    env_t,
                    inferred_env,
                    color="black",
                    linewidth=1.4,
                    label="Inferred measured-force envelope",
                )
                axes[0].plot(
                    env_t,
                    raw_env,
                    color="tab:orange",
                    linewidth=1.1,
                    linestyle="--",
                    label="Raw learned-force envelope",
                )
                axes[0].plot(
                    env_t,
                    corrected_env,
                    color="tab:red",
                    linewidth=1.2,
                    label="Corrected-force envelope",
                )
                axes[0].plot(
                    env_t,
                    pure_env,
                    color="tab:cyan",
                    linewidth=1.0,
                    alpha=0.85,
                    label="Pure wave-force envelope",
                )
                if np.any(np.isfinite(geometry_overlay)):
                    axes[0].plot(
                        env_t,
                        geometry_overlay,
                        color="tab:green",
                        linewidth=1.0,
                        alpha=0.85,
                        linestyle=":",
                        label="Orientation-weighted wave proxy",
                    )
                axes[0].set_title("Rev M forecast force-envelope and vessel-encounter diagnostic")
                axes[0].set_ylabel("Scaled force envelope")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="best")
                axes[0].text(
                    0.015,
                    0.97,
                    "Envelope alignment\n"
                    f"  raw corr: {float(raw_metric.get('phase_corr', float('nan'))):.3f}, "
                    f"amp: {float(raw_metric.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  corrected corr: {float(corrected_metric.get('phase_corr', float('nan'))):.3f}, "
                    f"amp: {float(corrected_metric.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  geometry corr: {float(geometry_metric.get('phase_corr', float('nan'))):.3f}, "
                    f"amp: {float(geometry_metric.get('amplitude_ratio', float('nan'))):.3f}",
                    transform=axes[0].transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
                axes[1].plot(
                    env_t,
                    required_gain,
                    color="tab:purple",
                    linewidth=1.2,
                    label="Local gain required: inferred/raw envelope",
                )
                has_orientation_gain = bool(np.any(np.isfinite(orientation_gain)))
                has_learned_gate = bool(np.any(np.isfinite(learned_envelope_gate)))
                if has_orientation_gain or has_learned_gate:
                    gain_ax = axes[1].twinx()
                    if has_orientation_gain:
                        gain_ax.plot(
                            env_t,
                            orientation_gain,
                            color="tab:green",
                            linewidth=1.1,
                            alpha=0.75,
                            label="Wave-orientation force gain",
                        )
                    if has_learned_gate:
                        gain_ax.plot(
                            env_t,
                            learned_envelope_gate,
                            color="tab:orange",
                            linewidth=1.2,
                            alpha=0.90,
                            label="Learned envelope gate",
                        )
                    gain_ax.set_ylabel("Applied gain")
                    finite_gains = np.concatenate([
                        orientation_gain[np.isfinite(orientation_gain)],
                        learned_envelope_gate[np.isfinite(learned_envelope_gate)],
                    ])
                    if len(finite_gains):
                        gain_ax.set_ylim(
                            0.0,
                            max(1.05, min(3.0, float(np.nanmax(finite_gains)) * 1.10)),
                        )
                    lines, labels = axes[1].get_legend_handles_labels()
                    gain_lines, gain_labels = gain_ax.get_legend_handles_labels()
                    axes[1].legend(lines + gain_lines, labels + gain_labels, loc="best")
                else:
                    axes[1].legend(loc="best")
                axes[1].axhline(1.0, color="0.4", linewidth=0.8, linestyle=":")
                axes[1].set_xlabel("Time [s]")
                axes[1].set_ylabel("Required force gain")
                axes[1].grid(True, alpha=0.3)
                axes[1].text(
                    0.015,
                    0.97,
                    f"Envelope window: {float(env_metrics.get('window_s', float('nan'))):.3f} s\n"
                    f"Required gain median: {float(env_metrics.get('required_gain_median', float('nan'))):.3f}\n"
                    f"Required gain mean: {float(env_metrics.get('required_gain_mean', float('nan'))):.3f}\n"
                    "Orientation gain\n"
                    f"  min: {float(env_metrics.get('orientation_gain_min', float('nan'))):.3f}\n"
                    f"  mean: {float(env_metrics.get('orientation_gain_mean', float('nan'))):.3f}\n"
                    f"  max: {float(env_metrics.get('orientation_gain_max', float('nan'))):.3f}\n"
                    "Learned gate\n"
                    f"  min: {float(env_metrics.get('learned_envelope_gate_min', float('nan'))):.3f}\n"
                    f"  mean: {float(env_metrics.get('learned_envelope_gate_mean', float('nan'))):.3f}\n"
                    f"  max: {float(env_metrics.get('learned_envelope_gate_max', float('nan'))):.3f}\n"
                    f"  req corr: {float(env_metrics.get('learned_envelope_gate_required_gain_corr', float('nan'))):.3f}",
                    transform=axes[1].transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
                plt.close(fig)

            if emit_diagnostic_plots and bool(orientation_gate_ablation.get("enabled", False)):
                gate_lags = np.asarray(orientation_gate_ablation.get("lags_s", []), dtype=float)
                gate_strengths = np.asarray(orientation_gate_ablation.get("strengths", []), dtype=float)
                gate_rmse = np.asarray(orientation_gate_ablation.get("rmse_deg", []), dtype=float)
                gate_env_corr = np.asarray(orientation_gate_ablation.get("force_envelope_corr", []), dtype=float)
                gate_time = np.asarray(orientation_gate_ablation.get("time_s", []), dtype=float)
                gate_measured = np.asarray(orientation_gate_ablation.get("measured_roll_deg", []), dtype=float)
                gate_inferred_env = np.asarray(
                    orientation_gate_ablation.get("inferred_force_envelope", []),
                    dtype=float,
                )
                gate_variants = [
                    v for v in orientation_gate_ablation.get("variants", [])
                    if isinstance(v, dict)
                ]
                best_gate = orientation_gate_ablation.get("best_by_rmse", {})
                best_env_gate = orientation_gate_ablation.get("best_by_envelope", {})
                current_gate_strength = float(
                    orientation_gate_ablation.get("current_gate_strength", 1.0)
                )
                if not isinstance(best_gate, dict):
                    best_gate = {}
                if not isinstance(best_env_gate, dict):
                    best_env_gate = {}

                if (
                    len(gate_lags) > 0
                    and len(gate_strengths) > 0
                    and gate_rmse.shape == (len(gate_strengths), len(gate_lags))
                    and gate_env_corr.shape == gate_rmse.shape
                ):
                    fig, axes = plt.subplots(
                        2,
                        2,
                        figsize=(13, 9),
                        constrained_layout=True,
                    )
                    extent = [
                        float(gate_lags[0]),
                        float(gate_lags[-1]),
                        float(gate_strengths[0]),
                        float(gate_strengths[-1]),
                    ]

                    def draw_gate_heatmap(ax, values: np.ndarray, title: str,
                                          cmap: str, best: Dict[str, object]) -> None:
                        im = ax.imshow(
                            values,
                            origin="lower",
                            aspect="auto",
                            extent=extent,
                            cmap=cmap,
                        )
                        ax.axvline(0.0, color="white", linewidth=0.8, linestyle=":")
                        ax.axhline(1.0, color="white", linewidth=0.8, linestyle=":")
                        ax.axhline(current_gate_strength, color="black", linewidth=0.7, linestyle=":")
                        ax.plot(0.0, current_gate_strength, marker="+", markersize=10, color="white", mew=2)
                        ax.plot(0.0, current_gate_strength, marker="+", markersize=8, color="black", mew=1)
                        best_lag = float(best.get("lag_s", float("nan")))
                        best_strength = float(best.get("strength", float("nan")))
                        if math.isfinite(best_lag) and math.isfinite(best_strength):
                            ax.plot(best_lag, best_strength, marker="x", markersize=9, color="white", mew=2)
                            ax.plot(best_lag, best_strength, marker="x", markersize=7, color="black", mew=1)
                        ax.set_title(title)
                        ax.set_xlabel("Orientation-gate lag applied [s]")
                        ax.set_ylabel("Orientation-gate strength")
                        ax.grid(False)
                        fig.colorbar(im, ax=ax, shrink=0.88)

                    draw_gate_heatmap(
                        axes[0, 0],
                        gate_rmse,
                        "Replay RMSE [deg]",
                        "viridis_r",
                        best_gate,
                    )
                    draw_gate_heatmap(
                        axes[0, 1],
                        gate_env_corr,
                        "Force-envelope zero-lag correlation",
                        "coolwarm",
                        best_env_gate,
                    )

                    def variant_by_name(name: str) -> Optional[Dict[str, object]]:
                        for variant in gate_variants:
                            if str(variant.get("name", "")) == name:
                                return variant
                        return None

                    selected_variants = [
                        ("current model", "tab:orange", "--"),
                        ("no gate", "tab:green", "-"),
                        ("soft gate", "tab:purple", "-"),
                        ("current gate", "tab:brown", "-."),
                        ("inverted diagnostic", "0.45", ":"),
                        ("best RMSE gate", "tab:red", "-"),
                    ]
                    ax_roll = axes[1, 0]
                    if len(gate_time) == len(gate_measured):
                        ax_roll.plot(
                            gate_time,
                            gate_measured,
                            color="tab:blue",
                            linewidth=1.2,
                            label="Measured roll",
                        )
                    for name, color, linestyle in selected_variants:
                        variant = variant_by_name(name)
                        if not isinstance(variant, dict):
                            continue
                        roll = np.asarray(variant.get("roll_deg", []), dtype=float)
                        if len(roll) == len(gate_time):
                            ax_roll.plot(
                                gate_time,
                                roll,
                                color=color,
                                linewidth=1.05 if name != "best RMSE gate" else 1.35,
                                linestyle=linestyle,
                                alpha=0.88,
                                label=name,
                            )
                    ax_roll.set_title("Roll replay under orientation-gate ablations")
                    ax_roll.set_xlabel("Time [s]")
                    ax_roll.set_ylabel("Roll [deg]")
                    ax_roll.grid(True, alpha=0.3)
                    ax_roll.legend(loc="best", fontsize=8)
                    ax_roll.text(
                        0.015,
                        0.97,
                        f"Current marker: + at lag 0, strength {current_gate_strength:.1f}\n"
                        "Best RMSE marker: x\n"
                        f"Best RMSE: {float(best_gate.get('rmse_deg', float('nan'))):.3f} deg\n"
                        f"  lag: {float(best_gate.get('lag_s', float('nan'))):.3f} s\n"
                        f"  strength: {float(best_gate.get('strength', float('nan'))):.3f}\n"
                        f"  corr: {float(best_gate.get('phase_corr', float('nan'))):.3f}\n"
                        f"Best envelope corr: {float(best_env_gate.get('force_envelope_corr', float('nan'))):.3f}",
                        transform=ax_roll.transAxes,
                        va="top",
                        ha="left",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.88,
                            "boxstyle": "round,pad=0.35",
                        },
                    )

                    ax_env = axes[1, 1]
                    if len(gate_time) == len(gate_inferred_env):
                        ax_env.plot(
                            gate_time,
                            gate_inferred_env,
                            color="black",
                            linewidth=1.3,
                            label="Inferred measured-force envelope",
                        )
                    for name, color, linestyle in selected_variants:
                        variant = variant_by_name(name)
                        if not isinstance(variant, dict):
                            continue
                        env = np.asarray(variant.get("force_envelope", []), dtype=float)
                        if len(env) == len(gate_time):
                            ax_env.plot(
                                gate_time,
                                env,
                                color=color,
                                linewidth=1.0 if name != "best RMSE gate" else 1.25,
                                linestyle=linestyle,
                                alpha=0.86,
                                label=name,
                            )
                    ax_env.set_title("Force-envelope replay under orientation-gate ablations")
                    ax_env.set_xlabel("Time [s]")
                    ax_env.set_ylabel("Scaled force envelope")
                    ax_env.grid(True, alpha=0.3)
                    ax_env.legend(loc="best", fontsize=8)
                    fig.suptitle("Rev M orientation-gate ablation diagnostic")
                    plt.close(fig)

            if emit_diagnostic_plots and bool(oracle_replay.get("enabled", False)):
                oracle_roll = np.asarray(
                    oracle_replay.get("oracle_roll_deg", np.full(len(f_t), np.nan)),
                    dtype=float,
                )
                oracle_force = np.asarray(
                    oracle_replay.get("oracle_force_scaled", np.full(len(f_t), np.nan)),
                    dtype=float,
                )
                oracle_metrics = oracle_replay.get("metrics", {})
                learned_force_diag = oracle_replay.get("learned_force_vs_oracle_force", {})
                if not isinstance(oracle_metrics, dict):
                    oracle_metrics = {}
                if not isinstance(learned_force_diag, dict):
                    learned_force_diag = {}

                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(12, 8),
                    sharex=True,
                    constrained_layout=True,
                )
                axes[0].plot(
                    f_t,
                    f_measured_roll,
                    color="tab:blue",
                    linewidth=1.2,
                    label="Measured roll",
                )
                axes[0].plot(
                    f_t,
                    f_pred,
                    color="tab:red",
                    linewidth=1.2,
                    label="Learned-force forecast",
                )
                axes[0].plot(
                    f_t,
                    oracle_roll,
                    color="tab:green",
                    linewidth=1.4,
                    label="Oracle-force replay",
                )
                axes[0].axvline(float(f_t[0]), color="0.45", linestyle="--", linewidth=0.9)
                axes[0].set_ylabel("Roll [deg]")
                axes[0].set_title("Rev M oracle-force replay diagnostic")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="best")
                axes[0].text(
                    0.015,
                    0.97,
                    "Learned-force forecast\n"
                    f"  lag: {float(roll_diag.get('phase_lag_s', float('nan'))):.4f} s\n"
                    f"  amp ratio: {float(roll_diag.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  RMSE: {float(forecast_plot_metrics.get('rmse_deg', float('nan'))):.3f} deg\n"
                    "Oracle-force replay\n"
                    f"  lag: {float(oracle_metrics.get('phase_lag_s', float('nan'))):.4f} s\n"
                    f"  amp ratio: {float(oracle_metrics.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  RMSE: {float(oracle_metrics.get('rmse_deg', float('nan'))):.3f} deg",
                    transform=axes[0].transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )

                axes[1].plot(
                    f_t,
                    oracle_force,
                    color="black",
                    linewidth=1.4,
                    label="Oracle inferred measured force",
                )
                axes[1].plot(
                    f_t,
                    f_total_force,
                    color="tab:red",
                    linewidth=1.15,
                    alpha=0.9,
                    label="Learned total forecast force",
                )
                axes[1].plot(
                    f_t,
                    f_wave_force,
                    color="tab:cyan",
                    linewidth=1.1,
                    alpha=0.9,
                    label="Learned pure wave force",
                )
                axes[1].axhline(0.0, color="0.4", linewidth=0.8)
                axes[1].set_xlabel("Time [s]")
                axes[1].set_ylabel("Scaled force")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc="best")
                axes[1].text(
                    0.015,
                    0.97,
                    "Learned force vs oracle force\n"
                    f"  lag: {float(learned_force_diag.get('phase_lag_s', float('nan'))):.4f} s\n"
                    f"  corr: {float(learned_force_diag.get('phase_corr', float('nan'))):.3f}\n"
                    f"  amp ratio: {float(learned_force_diag.get('amplitude_ratio', float('nan'))):.3f}\n"
                    f"  RMSE: {float(learned_force_diag.get('rmse_scaled', float('nan'))):.4g}",
                    transform=axes[1].transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
                plt.close(fig)

            if emit_diagnostic_plots and bool(force_replay_sweep.get("enabled", False)):
                sweep_lags = np.asarray(force_replay_sweep.get("lags_s", []), dtype=float)
                sweep_gains = np.asarray(force_replay_sweep.get("gains", []), dtype=float)
                sweep_rmse = np.asarray(force_replay_sweep.get("rmse_deg", []), dtype=float)
                sweep_corr = np.asarray(force_replay_sweep.get("phase_corr", []), dtype=float)
                sweep_amp = np.asarray(force_replay_sweep.get("amplitude_ratio", []), dtype=float)
                best_sweep = force_replay_sweep.get("best_by_rmse", {})
                if not isinstance(best_sweep, dict):
                    best_sweep = {}
                if (
                    len(sweep_lags) > 1
                    and len(sweep_gains) > 1
                    and sweep_rmse.shape == (len(sweep_gains), len(sweep_lags))
                    and sweep_corr.shape == sweep_rmse.shape
                    and sweep_amp.shape == sweep_rmse.shape
                ):
                    fig, axes = plt.subplots(
                        2,
                        2,
                        figsize=(13, 9),
                        constrained_layout=True,
                    )
                    extent = [
                        float(sweep_lags[0]),
                        float(sweep_lags[-1]),
                        float(sweep_gains[0]),
                        float(sweep_gains[-1]),
                    ]

                    def draw_sweep_heatmap(ax, values: np.ndarray, title: str, cmap: str) -> None:
                        im = ax.imshow(
                            values,
                            origin="lower",
                            aspect="auto",
                            extent=extent,
                            cmap=cmap,
                        )
                        ax.set_title(title)
                        ax.set_xlabel("Force lag applied [s]")
                        ax.set_ylabel("Force gain")
                        ax.grid(False)
                        best_lag = float(best_sweep.get("lag_s", float("nan")))
                        best_gain = float(best_sweep.get("gain", float("nan")))
                        if math.isfinite(best_lag) and math.isfinite(best_gain):
                            ax.plot(best_lag, best_gain, marker="x", markersize=9, color="white", mew=2)
                            ax.plot(best_lag, best_gain, marker="x", markersize=7, color="black", mew=1)
                        fig.colorbar(im, ax=ax, shrink=0.88)

                    draw_sweep_heatmap(axes[0, 0], sweep_rmse, "Replay RMSE [deg]", "viridis_r")
                    draw_sweep_heatmap(axes[0, 1], sweep_corr, "Replay phase correlation", "coolwarm")
                    draw_sweep_heatmap(axes[1, 0], sweep_amp, "Replay amplitude ratio", "magma")

                    best_time = np.asarray(force_replay_sweep.get("time_s", []), dtype=float)
                    best_measured = np.asarray(force_replay_sweep.get("measured_roll_deg", []), dtype=float)
                    best_replay = np.asarray(force_replay_sweep.get("best_replay_roll_deg", []), dtype=float)
                    ax = axes[1, 1]
                    if len(best_time) == len(best_measured):
                        ax.plot(best_time, best_measured, color="tab:blue", linewidth=1.2, label="Measured roll")
                    if len(f_t) == len(f_pred):
                        ax.plot(f_t, f_pred, color="tab:red", linewidth=1.1, alpha=0.75, label="Original forecast")
                    if len(best_time) == len(best_replay):
                        ax.plot(best_time, best_replay, color="tab:green", linewidth=1.3, label="Best force replay")
                    ax.set_title("Best replay from lag/gain sweep")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Roll [deg]")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                    baseline = force_replay_sweep.get("baseline", {})
                    if not isinstance(baseline, dict):
                        baseline = {}
                    ax.text(
                        0.015,
                        0.97,
                        "Original forecast\n"
                        f"  RMSE: {float(baseline.get('rmse_deg', float('nan'))):.3f} deg\n"
                        f"  corr: {float(baseline.get('phase_corr', float('nan'))):.3f}\n"
                        f"  amp: {float(baseline.get('amplitude_ratio', float('nan'))):.3f}\n"
                        "Best replay\n"
                        f"  lag: {float(best_sweep.get('lag_s', float('nan'))):.3f} s\n"
                        f"  gain: {float(best_sweep.get('gain', float('nan'))):.3f}\n"
                        f"  RMSE: {float(best_sweep.get('rmse_deg', float('nan'))):.3f} deg\n"
                        f"  corr: {float(best_sweep.get('phase_corr', float('nan'))):.3f}\n"
                        f"  amp: {float(best_sweep.get('amplitude_ratio', float('nan'))):.3f}",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.88,
                            "boxstyle": "round,pad=0.35",
                        },
                    )
                    fig.suptitle("Rev M forecast force replay lag/gain sweep")
                    plt.close(fig)

            if emit_diagnostic_plots and bool(coefficient_replay_sweep.get("enabled", False)):
                coeff_multipliers = np.asarray(
                    coefficient_replay_sweep.get("multipliers", []),
                    dtype=float,
                )
                coeff_series = coefficient_replay_sweep.get("series", {})
                if not isinstance(coeff_series, dict):
                    coeff_series = {}
                best_coeff = coefficient_replay_sweep.get("best_selected", {})
                if not isinstance(best_coeff, dict):
                    best_coeff = {}
                best_rmse_coeff = coefficient_replay_sweep.get("best_by_rmse", {})
                if not isinstance(best_rmse_coeff, dict):
                    best_rmse_coeff = {}
                coeff_selection_metric = str(coefficient_replay_sweep.get("selection_metric", "r2")).upper()
                baseline_coeff = coefficient_replay_sweep.get("baseline", {})
                if not isinstance(baseline_coeff, dict):
                    baseline_coeff = {}
                if len(coeff_multipliers) > 1 and coeff_series:
                    fig, axes = plt.subplots(
                        2,
                        2,
                        figsize=(13, 8.5),
                        constrained_layout=True,
                    )
                    coeff_labels = {
                        "c_roll": "linear damping c_roll",
                        "c_quad": "quadratic damping c_quad",
                        "k_roll": "restoring k_roll",
                    }
                    coeff_colors = {
                        "c_roll": "tab:blue",
                        "c_quad": "tab:orange",
                        "k_roll": "tab:green",
                    }

                    def plot_coeff_metric(ax, metric_key: str, title: str, ylabel: str,
                                          baseline_key: str, target_line: Optional[float] = None) -> None:
                        for coeff_name in ("c_roll", "c_quad", "k_roll"):
                            values = np.asarray(
                                coeff_series.get(coeff_name, {}).get(metric_key, []),
                                dtype=float,
                            )
                            if len(values) != len(coeff_multipliers):
                                continue
                            ax.plot(
                                coeff_multipliers,
                                values,
                                linewidth=1.4,
                                color=coeff_colors.get(coeff_name),
                                label=coeff_labels.get(coeff_name, coeff_name),
                            )
                        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.0, label="current coefficients")
                        baseline_value = float(baseline_coeff.get(baseline_key, float("nan")))
                        if math.isfinite(baseline_value):
                            ax.axhline(
                                baseline_value,
                                color="0.55",
                                linestyle=":",
                                linewidth=1.0,
                                label="current forecast",
                            )
                        if target_line is not None:
                            ax.axhline(float(target_line), color="0.2", linestyle="-.", linewidth=0.8)
                        best_name = str(best_coeff.get("coefficient", ""))
                        best_mult = float(best_coeff.get("multiplier", float("nan")))
                        best_value = float(best_coeff.get(baseline_key, float("nan")))
                        if math.isfinite(best_mult) and math.isfinite(best_value):
                            ax.plot(best_mult, best_value, marker="x", markersize=9, color="black", mew=2)
                        ax.set_title(title)
                        ax.set_xlabel("Coefficient multiplier")
                        ax.set_ylabel(ylabel)
                        ax.grid(True, alpha=0.3)
                        if best_name:
                            ax.text(
                                0.015,
                                0.97,
                                f"Best {coeff_selection_metric}: {best_name} x {best_mult:.3f}\n"
                                f"R2: {float(best_coeff.get('r2', float('nan'))):.3f}\n"
                                f"RMSE: {float(best_coeff.get('rmse_deg', float('nan'))):.3f} deg\n"
                                f"corr: {float(best_coeff.get('phase_corr', float('nan'))):.3f}\n"
                                f"amp: {float(best_coeff.get('amplitude_ratio', float('nan'))):.3f}",
                                transform=ax.transAxes,
                                va="top",
                                ha="left",
                                bbox={
                                    "facecolor": "white",
                                    "edgecolor": "0.75",
                                    "alpha": 0.88,
                                    "boxstyle": "round,pad=0.35",
                                },
                            )

                    plot_coeff_metric(axes[0, 0], "rmse_deg", "Replay RMSE", "RMSE [deg]", "rmse_deg")
                    plot_coeff_metric(
                        axes[0, 1],
                        "amplitude_ratio",
                        "Replay amplitude ratio",
                        "Amplitude ratio",
                        "amplitude_ratio",
                        target_line=1.0,
                    )
                    plot_coeff_metric(
                        axes[1, 0],
                        "r2",
                        "Replay R2",
                        "R2",
                        "r2",
                    )
                    axes[0, 0].legend(loc="best")

                    coeff_time = np.asarray(coefficient_replay_sweep.get("time_s", []), dtype=float)
                    coeff_measured = np.asarray(
                        coefficient_replay_sweep.get("measured_roll_deg", []),
                        dtype=float,
                    )
                    coeff_baseline_roll = np.asarray(
                        coefficient_replay_sweep.get("baseline_roll_deg", []),
                        dtype=float,
                    )
                    coeff_best_roll = np.asarray(
                        coefficient_replay_sweep.get("best_replay_roll_deg", []),
                        dtype=float,
                    )
                    coeff_best_rmse_roll = np.asarray(
                        coefficient_replay_sweep.get("best_rmse_replay_roll_deg", []),
                        dtype=float,
                    )
                    ax = axes[1, 1]
                    if len(coeff_time) == len(coeff_measured):
                        ax.plot(coeff_time, coeff_measured, color="tab:blue", linewidth=1.2, label="Measured roll")
                    if len(coeff_time) == len(coeff_baseline_roll):
                        ax.plot(coeff_time, coeff_baseline_roll, color="tab:red", linewidth=1.1, alpha=0.75, label="Current coefficients")
                    if len(coeff_time) == len(coeff_best_roll):
                        ax.plot(coeff_time, coeff_best_roll, color="tab:green", linewidth=1.3, label=f"Best {coeff_selection_metric} replay")
                    if coeff_selection_metric != "RMSE" and len(coeff_time) == len(coeff_best_rmse_roll):
                        ax.plot(coeff_time, coeff_best_rmse_roll, color="0.35", linewidth=1.0, linestyle=":", label="Best RMSE replay")
                    ax.set_title("Best replay from coefficient sweep")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Roll [deg]")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                    base_coeffs = coefficient_replay_sweep.get("base_coefficients", {})
                    if not isinstance(base_coeffs, dict):
                        base_coeffs = {}
                    ax.text(
                        0.015,
                        0.97,
                        "Current coefficients\n"
                        f"  c_roll: {float(base_coeffs.get('c_roll', float('nan'))):.5g}\n"
                        f"  c_quad: {float(base_coeffs.get('c_quad', float('nan'))):.5g}\n"
                        f"  k_roll: {float(base_coeffs.get('k_roll', float('nan'))):.5g}\n"
                        "Best replay coefficients\n"
                        f"  c_roll: {float(best_coeff.get('c_roll', float('nan'))):.5g}\n"
                        f"  c_quad: {float(best_coeff.get('c_quad', float('nan'))):.5g}\n"
                        f"  k_roll: {float(best_coeff.get('k_roll', float('nan'))):.5g}\n"
                        "RMSE-only replay\n"
                        f"  {str(best_rmse_coeff.get('coefficient', ''))} x {float(best_rmse_coeff.get('multiplier', float('nan'))):.3f}\n"
                        f"  R2: {float(best_rmse_coeff.get('r2', float('nan'))):.3f}\n"
                        f"  amp: {float(best_rmse_coeff.get('amplitude_ratio', float('nan'))):.3f}",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.88,
                            "boxstyle": "round,pad=0.35",
                        },
                    )
                    fig.suptitle("Rev M forecast ODE coefficient replay sweep")
                    plt.close(fig)

        plt.figure(figsize=(10, 4))
        plt.plot(data["lag_grid_s"], data["lag_scores"])
        plt.axvline(float(data["best_wave_lag_s"]), linestyle="--", linewidth=1.0)
        plt.xlabel("Lag [s]")
        plt.ylabel("Band-limited correlation")
        plt.title("Wave-to-roll lag scan")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(lag_plot_path, dpi=180)
        plt.close()

        force_lag_lags = np.asarray(force_lag_scan.get("lags_s", []), dtype=float)
        force_lag_train = np.asarray(force_lag_scan.get("train_scores", []), dtype=float)
        force_lag_val = np.asarray(force_lag_scan.get("validation_scores", []), dtype=float)
        if emit_diagnostic_plots and len(force_lag_lags) > 0 and len(force_lag_train) == len(force_lag_lags):
            train_summary = force_lag_scan.get("train", {})
            val_summary = force_lag_scan.get("validation", {})
            if not isinstance(train_summary, dict):
                train_summary = {}
            if not isinstance(val_summary, dict):
                val_summary = {}
            fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
            ax.plot(
                force_lag_lags,
                force_lag_train,
                color="tab:blue",
                linewidth=1.4,
                label="Train: wave slope vs inferred force",
            )
            if len(force_lag_val) == len(force_lag_lags):
                ax.plot(
                    force_lag_lags,
                    force_lag_val,
                    color="tab:orange",
                    linewidth=1.4,
                    label="Validation: wave slope vs inferred force",
                )
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.axvline(0.0, color="0.35", linewidth=0.8, linestyle=":")
            current_lag = float(force_lag_scan.get("current_wave_lag_s", float("nan")))
            if math.isfinite(current_lag):
                ax.axvline(
                    current_lag,
                    color="tab:red",
                    linewidth=1.2,
                    linestyle="--",
                    label="current wave lag",
                )
            for summary, color, label in (
                (train_summary, "tab:blue", "best train force lag"),
                (val_summary, "tab:orange", "best validation force lag"),
            ):
                best_lag = float(summary.get("best_lag_s", float("nan")))
                if math.isfinite(best_lag):
                    ax.axvline(best_lag, color=color, linewidth=1.0, linestyle="-.", alpha=0.85, label=label)
            ax.set_title("Rev M wave-to-inferred-force lag scan")
            ax.set_xlabel("Wave lag applied to wave proxy [s]")
            ax.set_ylabel("Band-limited correlation")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            ax.text(
                0.015,
                0.97,
                "Train\n"
                f"  current lag: {float(train_summary.get('current_lag_s', float('nan'))):.3f} s, corr: {float(train_summary.get('current_corr', float('nan'))):.3f}\n"
                f"  best lag: {float(train_summary.get('best_lag_s', float('nan'))):.3f} s, corr: {float(train_summary.get('best_corr', float('nan'))):.3f}\n"
                f"  delta: {float(train_summary.get('lag_difference_s', float('nan'))):.3f} s\n"
                "Validation\n"
                f"  current lag: {float(val_summary.get('current_lag_s', float('nan'))):.3f} s, corr: {float(val_summary.get('current_corr', float('nan'))):.3f}\n"
                f"  best lag: {float(val_summary.get('best_lag_s', float('nan'))):.3f} s, corr: {float(val_summary.get('best_corr', float('nan'))):.3f}\n"
                f"  delta: {float(val_summary.get('lag_difference_s', float('nan'))):.3f} s",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.88,
                    "boxstyle": "round,pad=0.35",
                },
            )
            plt.close(fig)

        multi_force_lag_lags = np.asarray(multi_force_lag_scan.get("lags_s", []), dtype=float)
        multi_force_lag_proxies = multi_force_lag_scan.get("proxies", {})
        if (
            emit_diagnostic_plots
            and
            len(multi_force_lag_lags) > 0
            and isinstance(multi_force_lag_proxies, dict)
            and len(multi_force_lag_proxies) > 0
        ):
            def format_top_force_lag_proxies(split_name: str) -> str:
                rows: List[Tuple[float, str, Dict[str, float]]] = []
                for proxy_name, scan in multi_force_lag_proxies.items():
                    if not isinstance(scan, dict):
                        continue
                    summary = scan.get(split_name, {})
                    if not isinstance(summary, dict):
                        continue
                    score = float(summary.get("best_abs_corr", float("nan")))
                    if math.isfinite(score):
                        rows.append((score, str(proxy_name), summary))
                rows.sort(key=lambda row: row[0], reverse=True)
                lines = [split_name.capitalize()]
                for _, proxy_name, summary in rows[:4]:
                    lines.append(
                        f"  {proxy_name}: lag {float(summary.get('best_lag_s', float('nan'))):.3f} s, "
                        f"corr {float(summary.get('best_corr', float('nan'))):.3f}"
                    )
                if len(lines) == 1:
                    lines.append("  no finite proxy scores")
                return "\n".join(lines)

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
                constrained_layout=True,
            )
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
            if not color_cycle:
                color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            for i, (proxy_name, scan) in enumerate(multi_force_lag_proxies.items()):
                if not isinstance(scan, dict):
                    continue
                color = color_cycle[i % len(color_cycle)]
                train_scores = np.asarray(scan.get("train_scores", []), dtype=float)
                val_scores = np.asarray(scan.get("validation_scores", []), dtype=float)
                if len(train_scores) == len(multi_force_lag_lags):
                    axes[0].plot(
                        multi_force_lag_lags,
                        train_scores,
                        linewidth=1.2,
                        alpha=0.9,
                        color=color,
                        label=str(proxy_name),
                    )
                if len(val_scores) == len(multi_force_lag_lags):
                    axes[1].plot(
                        multi_force_lag_lags,
                        val_scores,
                        linewidth=1.2,
                        alpha=0.9,
                        color=color,
                        label=str(proxy_name),
                    )
            for ax, split_name in zip(axes, ("train", "validation")):
                ax.axhline(0.0, color="0.35", linewidth=0.8)
                ax.axvline(0.0, color="0.35", linewidth=0.8, linestyle=":")
                if math.isfinite(current_wave_lag_s):
                    ax.axvline(
                        current_wave_lag_s,
                        color="tab:red",
                        linewidth=1.1,
                        linestyle="--",
                        label="current wave lag" if split_name == "train" else None,
                    )
                ax.set_ylabel("Band-limited correlation")
                ax.grid(True, alpha=0.3)
                ax.text(
                    0.015,
                    0.97,
                    format_top_force_lag_proxies(split_name),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.75",
                        "alpha": 0.88,
                        "boxstyle": "round,pad=0.35",
                    },
                )
            axes[0].set_title("Rev M multi-proxy wave-to-inferred-force lag scan - Train")
            axes[1].set_title("Rev M multi-proxy wave-to-inferred-force lag scan - Validation")
            axes[1].set_xlabel("Wave lag applied to candidate proxy [s]")
            axes[0].legend(loc="upper right", ncol=2, fontsize=8)
            plt.close(fig)

        if emit_diagnostic_plots and bool(orientation_envelope_lag.get("enabled", False)):
            env_t = np.asarray(orientation_envelope_lag.get("time_s", []), dtype=float)
            env_lags = np.asarray(orientation_envelope_lag.get("lags_s", []), dtype=float)
            inferred_env = np.asarray(
                orientation_envelope_lag.get("inferred_force_envelope", []),
                dtype=float,
            )
            raw_wave_env = np.asarray(
                orientation_envelope_lag.get("raw_wave_envelope", []),
                dtype=float,
            )
            weighted_wave_env = np.asarray(
                orientation_envelope_lag.get("orientation_weighted_wave_envelope", []),
                dtype=float,
            )
            orientation_gain = np.asarray(
                orientation_envelope_lag.get("orientation_gain", []),
                dtype=float,
            )
            env_splits = orientation_envelope_lag.get("splits", {})
            if not isinstance(env_splits, dict):
                env_splits = {}
            split_names = [
                name for name in ("train", "validation", "forecast")
                if isinstance(env_splits.get(name), dict)
            ]
            if (
                split_names
                and len(env_t) == len(inferred_env) == len(raw_wave_env) == len(weighted_wave_env) == len(orientation_gain)
                and len(env_lags) > 0
            ):
                fig, axes = plt.subplots(
                    len(split_names),
                    2,
                    figsize=(14, 3.7 * len(split_names)),
                    constrained_layout=True,
                    squeeze=False,
                )

                def overlay_scale(candidate: np.ndarray, reference: np.ndarray, idx: np.ndarray) -> np.ndarray:
                    idx = idx[(idx >= 0) & (idx < len(reference))]
                    if len(idx) < 3:
                        return np.full_like(candidate, np.nan, dtype=np.float64)
                    finite_c = np.isfinite(candidate[idx])
                    finite_r = np.isfinite(reference[idx])
                    finite = finite_c & finite_r
                    if int(np.sum(finite)) < 3:
                        return np.full_like(candidate, np.nan, dtype=np.float64)
                    cand_level = float(np.nanmedian(np.abs(candidate[idx][finite])))
                    ref_level = float(np.nanmedian(np.abs(reference[idx][finite])))
                    if not math.isfinite(cand_level) or cand_level <= 1.0e-12:
                        return np.full_like(candidate, np.nan, dtype=np.float64)
                    return candidate * (ref_level / cand_level)

                def split_text(split: Dict[str, object]) -> str:
                    raw = split.get("raw_wave", {}) if isinstance(split.get("raw_wave", {}), dict) else {}
                    weighted = (
                        split.get("orientation_weighted_wave", {})
                        if isinstance(split.get("orientation_weighted_wave", {}), dict)
                        else {}
                    )
                    gate = (
                        split.get("orientation_gate", {})
                        if isinstance(split.get("orientation_gate", {}), dict)
                        else {}
                    )
                    return (
                        "Best envelope lag\n"
                        f"  weighted: {float(weighted.get('best_lag_s', float('nan'))):.3f} s, "
                        f"corr {float(weighted.get('best_corr', float('nan'))):.3f}\n"
                        f"  raw wave: {float(raw.get('best_lag_s', float('nan'))):.3f} s, "
                        f"corr {float(raw.get('best_corr', float('nan'))):.3f}\n"
                        f"  gate only: {float(gate.get('best_lag_s', float('nan'))):.3f} s, "
                        f"corr {float(gate.get('best_corr', float('nan'))):.3f}"
                    )

                for row, split_name in enumerate(split_names):
                    split = env_splits.get(split_name, {})
                    idx = np.asarray(split.get("indices", []), dtype=int)
                    idx = idx[(idx >= 0) & (idx < len(env_t))]
                    ax_env = axes[row, 0]
                    ax_scan = axes[row, 1]
                    raw_overlay = overlay_scale(raw_wave_env, inferred_env, idx)
                    weighted_overlay = overlay_scale(weighted_wave_env, inferred_env, idx)
                    if len(idx) > 0:
                        ax_env.plot(
                            env_t[idx],
                            inferred_env[idx],
                            color="black",
                            linewidth=1.25,
                            label="Inferred measured-force envelope",
                        )
                        ax_env.plot(
                            env_t[idx],
                            raw_overlay[idx],
                            color="tab:blue",
                            linewidth=1.0,
                            alpha=0.85,
                            linestyle="--",
                            label="Raw wave envelope (scaled)",
                        )
                        ax_env.plot(
                            env_t[idx],
                            weighted_overlay[idx],
                            color="tab:green",
                            linewidth=1.05,
                            alpha=0.9,
                            label="Orientation-weighted wave envelope (scaled)",
                        )
                        gain_ax = ax_env.twinx()
                        gain_ax.plot(
                            env_t[idx],
                            orientation_gain[idx],
                            color="tab:purple",
                            linewidth=0.95,
                            alpha=0.72,
                            linestyle=":",
                            label="Orientation gain",
                        )
                        gain_ax.set_ylabel("Orientation gain")
                        gain_ax.set_ylim(0.0, 1.05)
                        lines, labels = ax_env.get_legend_handles_labels()
                        gain_lines, gain_labels = gain_ax.get_legend_handles_labels()
                        ax_env.legend(lines + gain_lines, labels + gain_labels, loc="best", fontsize=8)
                    ax_env.set_title(f"Envelope timing - {split_name.capitalize()}")
                    ax_env.set_xlabel("Time [s]")
                    ax_env.set_ylabel("Scaled envelope")
                    ax_env.grid(True, alpha=0.3)
                    ax_env.text(
                        0.015,
                        0.97,
                        split_text(split),
                        transform=ax_env.transAxes,
                        va="top",
                        ha="left",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "0.75",
                            "alpha": 0.88,
                            "boxstyle": "round,pad=0.35",
                        },
                    )

                    raw_scores = np.asarray(split.get("raw_wave_scores", []), dtype=float)
                    weighted_scores = np.asarray(split.get("orientation_weighted_scores", []), dtype=float)
                    gate_scores = np.asarray(split.get("orientation_gate_scores", []), dtype=float)
                    if len(raw_scores) == len(env_lags):
                        ax_scan.plot(env_lags, raw_scores, color="tab:blue", linewidth=1.2, label="Raw wave envelope")
                    if len(weighted_scores) == len(env_lags):
                        ax_scan.plot(env_lags, weighted_scores, color="tab:green", linewidth=1.3, label="Orientation-weighted wave envelope")
                    if len(gate_scores) == len(env_lags):
                        ax_scan.plot(env_lags, gate_scores, color="tab:purple", linewidth=1.1, linestyle=":", label="Orientation gate only")
                    ax_scan.axhline(0.0, color="0.35", linewidth=0.8)
                    ax_scan.axvline(0.0, color="0.35", linewidth=0.8, linestyle=":")
                    ax_scan.set_title(f"Envelope lag scan - {split_name.capitalize()}")
                    ax_scan.set_xlabel("Lag applied to candidate envelope [s]")
                    ax_scan.set_ylabel("Envelope correlation")
                    ax_scan.grid(True, alpha=0.3)
                    ax_scan.legend(loc="best", fontsize=8)
                fig.suptitle("Rev M orientation-weighted wave envelope lag diagnostic")
                plt.close(fig)

        if history:
            epochs = np.asarray([row.get("epoch", np.nan) for row in history], dtype=float)
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

            def plot_loss_series(ax, key: str, label: str) -> None:
                values = np.asarray([row.get(key, np.nan) for row in history], dtype=float)
                mask = np.isfinite(epochs) & np.isfinite(values) & (values > 0.0)
                if np.any(mask):
                    ax.semilogy(epochs[mask], values[mask], label=label, linewidth=1.4)

            plot_loss_series(axes[0], "train_total", "Train total")
            plot_loss_series(axes[0], "val_total", "Validation total")
            axes[0].set_title("Training and validation loss")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3, which="both")
            axes[0].legend()

            plot_loss_series(axes[1], "train_data", "Data")
            plot_loss_series(axes[1], "train_r2_data", "R2 objective")
            plot_loss_series(axes[1], "train_peak_trough", "Peak/trough")
            plot_loss_series(axes[1], "train_global_extrema", "Global extrema")
            plot_loss_series(axes[1], "train_extrema_window_underfit", "Extrema window underfit")
            plot_loss_series(axes[1], "train_high_amplitude_underfit", "High-amplitude underfit")
            plot_loss_series(axes[1], "train_high_pass_residual", "High-pass residual")
            plot_loss_series(axes[1], "train_low_pass_residual", "Low-pass residual")
            plot_loss_series(axes[1], "train_roll_spectral_shape", "Roll spectral shape")
            plot_loss_series(axes[1], "train_amplitude_underfit", "Amplitude underfit")
            plot_loss_series(axes[1], "train_rollout", "Rollout")
            plot_loss_series(axes[1], "train_rollout_peak_trough", "Rollout peak/trough")
            plot_loss_series(axes[1], "train_rollout_global_extrema", "Rollout global extrema")
            plot_loss_series(axes[1], "train_rollout_extrema_window_underfit", "Rollout extrema window")
            plot_loss_series(axes[1], "train_rollout_high_amplitude_underfit", "Rollout high-amplitude")
            plot_loss_series(axes[1], "train_rollout_local_prominence", "Rollout prominence")
            plot_loss_series(axes[1], "train_rollout_high_pass_residual", "Rollout high-pass")
            plot_loss_series(axes[1], "train_rollout_roll_spectral_shape", "Rollout spectrum")
            plot_loss_series(axes[1], "train_rollout_phase", "Rollout phase")
            plot_loss_series(axes[1], "train_direct_forecast", "Direct forecast")
            plot_loss_series(axes[1], "train_direct_forecast_peak_trough", "Direct peak/trough")
            plot_loss_series(axes[1], "train_direct_forecast_extrema_window_underfit", "Direct extrema window")
            plot_loss_series(axes[1], "train_direct_forecast_high_amplitude_underfit", "Direct high-amplitude")
            plot_loss_series(axes[1], "train_direct_forecast_asymmetric_extrema", "Direct asym extrema")
            plot_loss_series(axes[1], "train_direct_forecast_roll_spectral_shape", "Direct spectrum")
            plot_loss_series(axes[1], "train_roll_slope", "Roll slope")
            plot_loss_series(axes[1], "train_rate", "Rate")
            plot_loss_series(axes[1], "train_physics", "Physics")
            plot_loss_series(axes[1], "train_kinematic", "Kinematic")
            plot_loss_series(axes[1], "train_total_force_target", "Total force target")
            plot_loss_series(axes[1], "train_total_force_event", "Total force event")
            plot_loss_series(axes[1], "train_total_force_band_shape", "Total force band shape")
            plot_loss_series(axes[1], "train_total_force_spectral_shape", "Total force spectral shape")
            plot_loss_series(axes[1], "train_force_envelope", "Force envelope")
            axes[1].set_title("Training loss components")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True, alpha=0.3, which="both")
            axes[1].legend()

            fig.savefig(loss_plot_path, dpi=180)
            plt.close(fig)

    lg.info("Outputs written to: %s", output_dir)
    lg.info("Full fit R2=%.5f RMSE=%.5f deg", metrics["full"]["r2"], metrics["full"]["rmse_deg"])
    if "train" in metrics:
        lg.info("Train fit R2=%.5f RMSE=%.5f deg", metrics["train"]["r2"], metrics["train"]["rmse_deg"])
    if "validation" in metrics:
        lg.info("Validation fit R2=%.5f RMSE=%.5f deg", metrics["validation"]["r2"], metrics["validation"]["rmse_deg"])
    tf_alignment_log = metrics.get("teacher_forced_force_alignment", {})
    if isinstance(tf_alignment_log, dict):
        tf_metric_log = tf_alignment_log.get("metrics", {})
        if isinstance(tf_metric_log, dict):
            for split_name in ("train", "validation"):
                split_metric = tf_metric_log.get(split_name, {})
                if not isinstance(split_metric, dict):
                    continue
                total_metric = split_metric.get("total_force", {})
                pure_metric = split_metric.get("pure_wave_force", {})
                if isinstance(total_metric, dict) and isinstance(pure_metric, dict):
                    lg.info(
                        "Teacher-forced force alignment %s | total corr=%.5f lag=%.5f s amp=%.5f rmse=%.5g | pure corr=%.5f lag=%.5f s amp=%.5f rmse=%.5g",
                        split_name,
                        float(total_metric.get("phase_corr", float("nan"))),
                        float(total_metric.get("phase_lag_s", float("nan"))),
                        float(total_metric.get("amplitude_ratio", float("nan"))),
                        float(total_metric.get("rmse_scaled", float("nan"))),
                        float(pure_metric.get("phase_corr", float("nan"))),
                        float(pure_metric.get("phase_lag_s", float("nan"))),
                        float(pure_metric.get("amplitude_ratio", float("nan"))),
                        float(pure_metric.get("rmse_scaled", float("nan"))),
                    )
    force_lag_log = metrics.get("wave_to_inferred_force_lag_scan", {})
    if isinstance(force_lag_log, dict):
        for split_name in ("train", "validation"):
            split_metric = force_lag_log.get(split_name, {})
            if isinstance(split_metric, dict):
                lg.info(
                    "Wave-to-inferred-force lag %s | current=%.5f s corr=%.5f | best=%.5f s corr=%.5f | delta=%.5f s",
                    split_name,
                    float(split_metric.get("current_lag_s", float("nan"))),
                    float(split_metric.get("current_corr", float("nan"))),
                    float(split_metric.get("best_lag_s", float("nan"))),
                    float(split_metric.get("best_corr", float("nan"))),
                    float(split_metric.get("lag_difference_s", float("nan"))),
                )
    multi_force_lag_log = metrics.get("wave_to_inferred_force_multifeature_lag_scan", {})
    if isinstance(multi_force_lag_log, dict):
        proxy_metrics = multi_force_lag_log.get("proxies", {})
        if isinstance(proxy_metrics, dict):
            for split_name in ("train", "validation"):
                best_proxy = None
                best_summary: Dict[str, float] = {}
                best_score = -np.inf
                for proxy_name, proxy_metric in proxy_metrics.items():
                    if not isinstance(proxy_metric, dict):
                        continue
                    summary = proxy_metric.get(split_name, {})
                    if not isinstance(summary, dict):
                        continue
                    score = float(summary.get("best_abs_corr", float("nan")))
                    if math.isfinite(score) and score > best_score:
                        best_score = score
                        best_proxy = str(proxy_name)
                        best_summary = summary
                if best_proxy is not None:
                    lg.info(
                        "Multi-proxy wave-to-force lag %s | best_proxy=%s lag=%.5f s corr=%.5f | current_corr=%.5f",
                        split_name,
                        best_proxy,
                        float(best_summary.get("best_lag_s", float("nan"))),
                        float(best_summary.get("best_corr", float("nan"))),
                        float(best_summary.get("current_corr", float("nan"))),
                    )
    per_run_log = metrics.get("per_run", {})
    if isinstance(per_run_log, dict):
        for name, run_metric in per_run_log.items():
            if isinstance(run_metric, dict):
                lg.info("Run %s fit R2=%.5f RMSE=%.5f deg",
                        name, float(run_metric.get("r2", float("nan"))), float(run_metric.get("rmse_deg", float("nan"))))
    if bool(forecast.get("enabled", False)) and isinstance(metrics.get("forecast"), dict):
        f_metrics = metrics["forecast"].get("metrics", {})
        if isinstance(f_metrics, dict):
            lg.info(
                "Forecast %.3f-%.3f s | R2=%.5f RMSE=%.5f deg phase_lag=%.5f s amp_ratio=%.5f",
                float(metrics["forecast"].get("start_time_s", float("nan"))),
                float(metrics["forecast"].get("end_time_s", float("nan"))),
                float(f_metrics.get("r2", float("nan"))),
                float(f_metrics.get("rmse_deg", float("nan"))),
                float(f_metrics.get("phase_lag_s", float("nan"))),
                float(f_metrics.get("amplitude_ratio", float("nan"))),
            )
        verification = metrics["forecast"].get("wave_force_verification", {})
        if isinstance(verification, dict):
            lg.info(
                "Forecast wave-force verification: status=%s features=%d input_rms=%.6g pure_wave_force_rms=%.6g counterfactual_delta_rms=%.6g total_force_rms=%.6g",
                verification.get("status", "unknown"),
                int(verification.get("wave_feature_count", 0)),
                float(verification.get("wave_input_rms_scaled", float("nan"))),
                float(verification.get("pure_wave_force_rms_scaled", float("nan"))),
                float(verification.get("counterfactual_force_delta_rms_scaled", float("nan"))),
                float(verification.get("total_force_rms_scaled", float("nan"))),
            )
            if (
                bool(metrics["forecast"].get("forecast_ode_enabled", True))
                and not bool(verification.get("wave_force_applied_to_forecast_ode", False))
            ):
                lg.error("Forecast wave-force verification failed: %s", verification)
            elif (
                not bool(verification.get("pure_wave_force_nonzero", False))
                or not bool(verification.get("counterfactual_wave_effect_nonzero", False))
            ):
                lg.warning(
                    "Forecast wave channels are present, but their learned force effect is negligible; "
                    "the forecast may resemble unforced roll decay."
                )
        if bool(oracle_replay.get("enabled", False)):
            oracle_metrics = oracle_replay.get("metrics", {})
            force_metrics = oracle_replay.get("learned_force_vs_oracle_force", {})
            if isinstance(oracle_metrics, dict) and isinstance(force_metrics, dict):
                lg.info(
                    "Oracle-force replay | R2=%.5f RMSE=%.5f deg phase_lag=%.5f s amp_ratio=%.5f | learned_force_corr=%.5f learned_force_lag=%.5f s",
                    float(oracle_metrics.get("r2", float("nan"))),
                    float(oracle_metrics.get("rmse_deg", float("nan"))),
                    float(oracle_metrics.get("phase_lag_s", float("nan"))),
                    float(oracle_metrics.get("amplitude_ratio", float("nan"))),
                    float(force_metrics.get("phase_corr", float("nan"))),
                    float(force_metrics.get("phase_lag_s", float("nan"))),
                )
        if bool(force_replay_sweep.get("enabled", False)):
            best_sweep = force_replay_sweep.get("best_by_rmse", {})
            baseline_sweep = force_replay_sweep.get("baseline", {})
            if isinstance(best_sweep, dict) and isinstance(baseline_sweep, dict):
                lg.info(
                    "Forecast force replay sweep | baseline RMSE=%.5f corr=%.5f amp=%.5f | best lag=%.5f s gain=%.5f RMSE=%.5f corr=%.5f amp=%.5f",
                    float(baseline_sweep.get("rmse_deg", float("nan"))),
                    float(baseline_sweep.get("phase_corr", float("nan"))),
                    float(baseline_sweep.get("amplitude_ratio", float("nan"))),
                    float(best_sweep.get("lag_s", float("nan"))),
                    float(best_sweep.get("gain", float("nan"))),
                    float(best_sweep.get("rmse_deg", float("nan"))),
                    float(best_sweep.get("phase_corr", float("nan"))),
                    float(best_sweep.get("amplitude_ratio", float("nan"))),
                )
        if bool(coefficient_replay_sweep.get("enabled", False)):
            best_coeff = coefficient_replay_sweep.get("best_selected", {})
            best_rmse_coeff = coefficient_replay_sweep.get("best_by_rmse", {})
            baseline_coeff = coefficient_replay_sweep.get("baseline", {})
            if isinstance(best_coeff, dict) and isinstance(best_rmse_coeff, dict) and isinstance(baseline_coeff, dict):
                lg.info(
                    "Forecast coefficient replay sweep | baseline R2=%.5f RMSE=%.5f corr=%.5f amp=%.5f | best %s %s x %.5f R2=%.5f RMSE=%.5f corr=%.5f amp=%.5f | RMSE-only %s x %.5f R2=%.5f RMSE=%.5f amp=%.5f",
                    float(baseline_coeff.get("r2", float("nan"))),
                    float(baseline_coeff.get("rmse_deg", float("nan"))),
                    float(baseline_coeff.get("phase_corr", float("nan"))),
                    float(baseline_coeff.get("amplitude_ratio", float("nan"))),
                    str(coefficient_replay_sweep.get("selection_metric", "r2")).upper(),
                    str(best_coeff.get("coefficient", "")),
                    float(best_coeff.get("multiplier", float("nan"))),
                    float(best_coeff.get("r2", float("nan"))),
                    float(best_coeff.get("rmse_deg", float("nan"))),
                    float(best_coeff.get("phase_corr", float("nan"))),
                    float(best_coeff.get("amplitude_ratio", float("nan"))),
                    str(best_rmse_coeff.get("coefficient", "")),
                    float(best_rmse_coeff.get("multiplier", float("nan"))),
                    float(best_rmse_coeff.get("r2", float("nan"))),
                    float(best_rmse_coeff.get("rmse_deg", float("nan"))),
                    float(best_rmse_coeff.get("amplitude_ratio", float("nan"))),
                )

    paths = {"metrics": metrics_path, "config": cfg_path}
    if fit_csv_path.exists():
        paths["fit_csv"] = fit_csv_path
    if forecast_csv_path.exists():
        paths["forecast_csv"] = forecast_csv_path
    if forecast_zoom_csv_path.exists():
        paths["forecast_zoom_csv"] = forecast_zoom_csv_path
    if lag_csv_path.exists():
        paths["lag_csv"] = lag_csv_path
    if loss_csv_path.exists():
        paths["loss_csv"] = loss_csv_path
    if HAS_PLT and bool(cfg.get("save_plots", True)):
        paths["fit_plot"] = plot_path
        for name, path in run_fit_plot_paths.items():
            paths[f"fit_plot_{safe_filename_token(name)}"] = path
        for name, path in run_fit_csv_paths.items():
            if path.exists():
                paths[f"fit_csv_{safe_filename_token(name)}"] = path
        if forecast_plot_path.exists():
            paths["forecast_zoom_plot"] = forecast_plot_path
        if lag_plot_path.exists():
            paths["lag_plot"] = lag_plot_path
        if loss_plot_path.exists():
            paths["loss_plot"] = loss_plot_path
    return paths


# =============================================================================
# EXPLORATORY DATA/FORECAST LENGTH ASSESSMENT
# =============================================================================

def run_standard_training_once(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                               cfg: Dict[str, object], lg: logging.Logger,
                               resume: Optional[str] = None,
                               include_full_ode: bool = False,
                               full_ode_coefficient_source: str = "source") -> Tuple[Dict[str, Path], Dict[str, object], Dict[str, object]]:
    set_global_seed(int(cfg["seed"]))
    data = prepare_data(t, phi, wave_pack, cfg, lg)
    train_loader, val_loader = build_loaders(data, cfg, lg)
    model = build_model(cfg, data)
    log_model_summary(model, lg)
    train_result = train_model(model, train_loader, val_loader, data, cfg, lg, resume=resume)
    paths = write_outputs(
        model,
        data,
        cfg,
        train_result,
        lg,
        include_full_ode=bool(include_full_ode),
        full_ode_coefficient_source=str(full_ode_coefficient_source),
    )
    best_checkpoint = train_result.get("best_checkpoint")
    final_checkpoint = train_result.get("final_checkpoint")
    if best_checkpoint:
        paths["best_checkpoint"] = Path(str(best_checkpoint))
    if final_checkpoint and final_checkpoint != best_checkpoint:
        paths["final_checkpoint"] = Path(str(final_checkpoint))
    metrics = read_json_dict(Path(str(paths["metrics"]))) or {}
    split_info = data.get("validation_split", {})
    if not isinstance(split_info, dict):
        split_info = {}
    del model, train_loader, val_loader, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paths, metrics, split_info


def _metric_group(metrics: Dict[str, object], key: str) -> Dict[str, object]:
    group = metrics.get(key, {})
    return group if isinstance(group, dict) else {}


def _rmse_deg(metrics: Dict[str, object]) -> float:
    return finite_or(metrics.get("rmse_deg"), finite_or(metrics.get("error_rms_deg"), float("nan")))


def _path_string(paths: Dict[str, Path], key: str) -> Optional[str]:
    path = paths.get(key)
    return str(path) if path is not None else None


def exploratory_assessment_record(run_index: int, split_label: str, is_baseline: bool,
                                  train_pct: int, val_pct: int, forecast_pct: int,
                                  paths: Dict[str, Path], metrics: Dict[str, object],
                                  split_info: Dict[str, object],
                                  duration_s: float) -> Dict[str, object]:
    forecast_payload = _metric_group(metrics, "forecast")
    forecast_metrics = (
        forecast_payload.get("metrics", {})
        if bool(forecast_payload.get("enabled", True)) and isinstance(forecast_payload.get("metrics", {}), dict)
        else {}
    )
    full_metrics = _metric_group(metrics, "full")
    train_metrics = _metric_group(metrics, "train")
    validation_metrics = _metric_group(metrics, "validation")
    training = _metric_group(metrics, "training")
    forecast_points = 0
    if bool(split_info.get("forecast_holdout_active", False)):
        forecast_points = int(split_info.get("forecast_end_idx", -1)) - int(split_info.get("forecast_start_idx", 0)) + 1
        forecast_points = max(0, forecast_points)
    return {
        "status": "ok",
        "run_index": int(run_index),
        "split_label": split_label,
        "baseline_75_15_10": bool(is_baseline),
        "train_fraction_pct": int(train_pct),
        "validation_fraction_pct": int(val_pct),
        "forecast_fraction_pct": int(forecast_pct),
        "train_fraction": float(train_pct / 100.0),
        "validation_fraction": float(val_pct / 100.0),
        "forecast_fraction": float(forecast_pct / 100.0),
        "actual_train_point_fraction": finite_or(split_info.get("actual_train_point_frac"), float("nan")),
        "actual_validation_point_fraction": finite_or(split_info.get("actual_val_point_frac"), float("nan")),
        "actual_forecast_point_fraction": finite_or(split_info.get("actual_forecast_point_frac"), float("nan")),
        "total_points": int(
            int(split_info.get("train_points", 0))
            + int(split_info.get("validation_points", 0))
            + int(forecast_points)
        ),
        "data_points": int(split_info.get("train_points", 0)),
        "train_points": int(split_info.get("train_points", 0)),
        "validation_points": int(split_info.get("validation_points", 0)),
        "forecast_points": int(forecast_points),
        "preforecast_points": int(split_info.get("preforecast_points", 0)),
        "train_windows": int(split_info.get("train_windows", 0)),
        "validation_windows": int(split_info.get("validation_windows", 0)),
        "forecast_start_time_s": split_info.get("forecast_start_time_s"),
        "forecast_end_time_s": split_info.get("forecast_end_time_s"),
        "forecast_duration_s": forecast_payload.get("duration_s"),
        "forecast_r2": finite_or(forecast_metrics.get("r2"), float("nan")),
        "forecast_rmse_deg": _rmse_deg(forecast_metrics),
        "forecast_rms_error_deg": finite_or(forecast_metrics.get("error_rms_deg"), _rmse_deg(forecast_metrics)),
        "validation_r2": finite_or(validation_metrics.get("r2"), float("nan")),
        "validation_rmse_deg": _rmse_deg(validation_metrics),
        "train_r2": finite_or(train_metrics.get("r2"), float("nan")),
        "train_rmse_deg": _rmse_deg(train_metrics),
        "fit_r2": finite_or(full_metrics.get("r2"), float("nan")),
        "fit_rmse_deg": _rmse_deg(full_metrics),
        "best_epoch": training.get("best_epoch"),
        "best_metric": training.get("best_metric"),
        "duration_seconds": float(duration_s),
        "output_dir": str(Path(str(paths["metrics"])).parent) if "metrics" in paths else None,
        "metrics_path": _path_string(paths, "metrics"),
        "config_path": _path_string(paths, "config"),
        "fit_csv_path": _path_string(paths, "fit_csv"),
        "forecast_csv_path": _path_string(paths, "forecast_csv"),
        "forecast_zoom_csv_path": _path_string(paths, "forecast_zoom_csv"),
        "loss_csv_path": _path_string(paths, "loss_csv"),
        "best_checkpoint": _path_string(paths, "best_checkpoint"),
    }


def run_exploratory_assessment(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                               cfg: Dict[str, object], lg: logging.Logger,
                               include_full_ode: bool = False,
                               full_ode_coefficient_source: str = "source") -> Dict[str, object]:
    base_output_dir = Path(str(cfg["output_dir"]))
    study_dir = base_output_dir / str(cfg.get("exploratory_assessment_dir", "exploratory_data_forecast_lengths"))
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = study_dir / "exploratory_data_forecast_lengths_revm.csv"
    summary_json = study_dir / "exploratory_data_forecast_lengths_revm.json"
    val_pct = 15
    records: List[Dict[str, object]] = []
    split_specs = [
        (train_pct, val_pct, 100 - val_pct - train_pct)
        for train_pct in range(35, 81, 5)
    ]
    lg.info(
        "Starting exploratory data/forecast length assessment: %d runs from 35/15/50 through 80/15/5; baseline is 75/15/10.",
        len(split_specs),
    )
    for run_index, (train_pct, validation_pct, forecast_pct) in enumerate(split_specs, start=1):
        split_label = f"train_{train_pct:02d}_val_{validation_pct:02d}_forecast_{forecast_pct:02d}"
        run_dir = study_dir / split_label
        run_cfg = copy.deepcopy(cfg)
        run_cfg["val_frac"] = float(validation_pct / 100.0)
        run_cfg["forecast_frac"] = float(forecast_pct / 100.0)
        run_cfg["validation_window_s"] = None
        run_cfg["forecast_window_s"] = None
        run_cfg["forecast_start_s"] = None
        run_cfg["output_dir"] = str(run_dir / "outputs")
        run_cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
        run_cfg["log_dir"] = str(run_dir / "logs")
        is_baseline = (train_pct, validation_pct, forecast_pct) == (75, 15, 10)
        lg.info(
            "[EXP] Run %02d/%02d | split=%s%s",
            run_index,
            len(split_specs),
            split_label,
            " | baseline" if is_baseline else "",
        )
        start = time.time()
        try:
            paths, metrics, split_info = run_standard_training_once(
                t,
                phi,
                wave_pack,
                run_cfg,
                lg,
                resume=None,
                include_full_ode=bool(include_full_ode),
                full_ode_coefficient_source=str(full_ode_coefficient_source),
            )
            record = exploratory_assessment_record(
                run_index,
                split_label,
                is_baseline,
                train_pct,
                validation_pct,
                forecast_pct,
                paths,
                metrics,
                split_info,
                time.time() - start,
            )
            lg.info(
                "[EXP] Run %02d/%02d complete | forecast R2=%.6f RMSE=%.6f deg | points train/val/forecast=%d/%d/%d",
                run_index,
                len(split_specs),
                finite_or(record.get("forecast_r2"), float("nan")),
                finite_or(record.get("forecast_rmse_deg"), float("nan")),
                int(record.get("train_points", 0)),
                int(record.get("validation_points", 0)),
                int(record.get("forecast_points", 0)),
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            lg.exception("[EXP] Run %02d/%02d failed | split=%s", run_index, len(split_specs), split_label)
            record = {
                "status": "failed",
                "run_index": int(run_index),
                "split_label": split_label,
                "baseline_75_15_10": bool(is_baseline),
                "train_fraction_pct": int(train_pct),
                "validation_fraction_pct": int(validation_pct),
                "forecast_fraction_pct": int(forecast_pct),
                "train_fraction": float(train_pct / 100.0),
                "validation_fraction": float(validation_pct / 100.0),
                "forecast_fraction": float(forecast_pct / 100.0),
                "duration_seconds": float(time.time() - start),
                "output_dir": str(run_dir / "outputs"),
                "error": str(exc),
            }
        records.append(record)
        pd.DataFrame(records).to_csv(summary_csv, index=False)
        safe_json_dump({
            "study_type": "exploratory_data_forecast_lengths",
            "validation_fraction_pct": int(val_pct),
            "baseline_split": "75/15/10",
            "range": "35/15/50 through 80/15/5",
            "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
            "summary_csv": str(summary_csv),
            "runs": records,
        }, summary_json)
    successful = [record for record in records if record.get("status") == "ok"]
    best_r2 = max(
        successful,
        key=lambda record: finite_or(record.get("forecast_r2"), -1.0e9),
        default=None,
    )
    best_rmse = min(
        successful,
        key=lambda record: finite_or(record.get("forecast_rmse_deg"), float("inf")),
        default=None,
    )
    summary = {
        "study_type": "exploratory_data_forecast_lengths",
        "validation_fraction_pct": int(val_pct),
        "baseline_split": "75/15/10",
        "range": "35/15/50 through 80/15/5",
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "completed_runs": int(len(successful)),
        "total_runs": int(len(records)),
        "best_forecast_r2_run": copy.deepcopy(best_r2),
        "best_forecast_rmse_run": copy.deepcopy(best_rmse),
        "runs": records,
    }
    pd.DataFrame(records).to_csv(summary_csv, index=False)
    safe_json_dump(summary, summary_json)
    return summary


# =============================================================================
# BAYESIAN HYPERPARAMETER OPTIMISATION
# =============================================================================

def finite_or(value: object, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def bayes_forecast_metric(record: Dict[str, object], key: str, fallback: float) -> float:
    """Return a forecast-window metric only; never fall back to teacher-forced fit."""
    return finite_or(record.get(key), fallback)


def bayes_fit_rank_key(record: Dict[str, object]) -> Tuple[float, float, float, float]:
    forecast_r2 = bayes_forecast_metric(record, "forecast_r2", -1.0e9)
    fit_r2 = finite_or(record.get("fit_r2"), -1.0e9)
    forecast_rms = bayes_forecast_metric(record, "forecast_rms_error_deg", float("inf"))
    fit_rms = finite_or(record.get("fit_rms_error_deg"), float("inf"))
    # The scalar acquisition objective is primary so neither fit nor forecast
    # can be ignored; component metrics provide deterministic tie-breaks.
    return -bayes_objective_from_record(record), forecast_r2 + fit_r2, -forecast_rms, -fit_rms


def update_best_forecast_metric_records(summary: Dict[str, object],
                                        records: List[Dict[str, object]]) -> None:
    successful = [record for record in records if record.get("status") == "ok"]
    finite_r2 = [
        record for record in successful
        if math.isfinite(finite_or(record.get("forecast_r2"), float("nan")))
    ]
    finite_rmse = [
        record for record in successful
        if math.isfinite(finite_or(record.get("forecast_rms_error_deg"), float("nan")))
    ]
    best_r2 = max(
        finite_r2,
        key=lambda record: (
            finite_or(record.get("forecast_r2"), -1.0e9),
            -finite_or(record.get("forecast_rms_error_deg"), float("inf")),
        ),
        default=None,
    )
    best_rmse = min(
        finite_rmse,
        key=lambda record: (
            finite_or(record.get("forecast_rms_error_deg"), float("inf")),
            -finite_or(record.get("forecast_r2"), -1.0e9),
        ),
        default=None,
    )
    summary["best_forecast_r2_trial"] = copy.deepcopy(best_r2)
    summary["best_forecast_rmse_trial"] = copy.deepcopy(best_rmse)


def bayes_objective_from_record(record: Dict[str, object]) -> float:
    forecast_r2 = bayes_forecast_metric(record, "forecast_r2", -1.0e9)
    forecast_rms = bayes_forecast_metric(record, "forecast_rms_error_deg", float("inf"))
    fit_r2 = finite_or(record.get("fit_r2"), -1.0e9)
    fit_rms = finite_or(record.get("fit_rms_error_deg"), float("inf"))
    phase_lag = abs(finite_or(record.get("forecast_phase_lag_s"), 0.0))
    amp_underfit = finite_or(record.get("forecast_amplitude_underfit_ratio"), 0.0)
    extrema_relative_error = finite_or(
        record.get("forecast_extrema_relative_error"),
        0.0,
    )
    extrema_point_rmse = finite_or(record.get("forecast_extrema_point_rmse_deg"), 0.0)
    peak_underfit = finite_or(record.get("forecast_peak_underfit_deg"), 0.0)
    trough_overshoot = finite_or(record.get("forecast_trough_overshoot_deg"), 0.0)
    physics = finite_or(record.get("final_train_physics_loss"), float("inf"))
    kinematic = finite_or(record.get("final_train_kinematic_loss"), float("inf"))
    forecast_r2_weight = finite_or(record.get("bayes_r2_objective_weight"), 10.0)
    forecast_rmse_weight = finite_or(record.get("bayes_rmse_objective_weight"), 0.0)
    fit_r2_weight = finite_or(record.get("bayes_fit_r2_objective_weight"), 10.0)
    fit_rmse_weight = finite_or(record.get("bayes_fit_rmse_objective_weight"), 0.0)
    phase_weight = finite_or(record.get("bayes_phase_objective_weight"), 0.0)
    amplitude_weight = finite_or(record.get("bayes_amplitude_objective_weight"), 0.0)
    extrema_weight = finite_or(record.get("bayes_extrema_objective_weight"), 0.0)
    extrema_rmse_weight = finite_or(record.get("bayes_forecast_extrema_rmse_objective_weight"), 0.0)
    peak_underfit_weight = finite_or(record.get("bayes_forecast_peak_underfit_objective_weight"), 0.0)
    trough_overshoot_weight = finite_or(record.get("bayes_forecast_trough_overshoot_objective_weight"), 0.0)
    physics_weight = finite_or(record.get("bayes_physics_objective_weight"), 0.0)
    kinematic_weight = finite_or(record.get("bayes_kinematic_objective_weight"), 0.0)
    if not math.isfinite(forecast_rms):
        forecast_rms = 1.0e9
    if not math.isfinite(fit_rms):
        fit_rms = 1.0e9
    if not math.isfinite(physics):
        physics = 1.0e9
    if not math.isfinite(kinematic):
        kinematic = 1.0e9
    if not math.isfinite(extrema_point_rmse):
        extrema_point_rmse = 1.0e9
    if not math.isfinite(peak_underfit):
        peak_underfit = 1.0e9
    if not math.isfinite(trough_overshoot):
        trough_overshoot = 1.0e9
    fit_forecast_objective = (
        -forecast_r2_weight * forecast_r2
        + forecast_rmse_weight * math.log1p(max(forecast_rms, 0.0))
        - fit_r2_weight * fit_r2
        + fit_rmse_weight * math.log1p(max(fit_rms, 0.0))
        + extrema_weight * math.log1p(max(extrema_relative_error, 0.0))
        + extrema_rmse_weight * math.log1p(max(extrema_point_rmse, 0.0))
        + peak_underfit_weight * math.log1p(max(peak_underfit, 0.0))
        + trough_overshoot_weight * math.log1p(max(trough_overshoot, 0.0))
    )
    auxiliary_objective = (
        phase_weight * math.log1p(max(phase_lag, 0.0))
        + amplitude_weight * math.log1p(max(amp_underfit, 0.0))
        + physics_weight * math.log1p(max(physics, 0.0))
        + kinematic_weight * math.log1p(max(kinematic, 0.0))
    )
    return float(fit_forecast_objective + 0.05 * auxiliary_objective)


def final_history_value(train_result: Dict[str, object], key: str, fallback: float = float("nan")) -> float:
    history = train_result.get("history", [])
    if isinstance(history, list) and history:
        return finite_or(history[-1].get(key), fallback)
    return float(fallback)


def bayes_target_reached(record: Dict[str, object], target_r2: float, target_rms_error_deg: float,
                         target_fit_r2: float = -float("inf"),
                         target_fit_rms_error_deg: float = float("inf")) -> bool:
    forecast_r2 = bayes_forecast_metric(record, "forecast_r2", -1.0e9)
    forecast_rms = bayes_forecast_metric(record, "forecast_rms_error_deg", float("inf"))
    fit_r2 = finite_or(record.get("fit_r2"), -1.0e9)
    fit_rms = finite_or(record.get("fit_rms_error_deg"), float("inf"))
    return (
        forecast_r2 > float(target_r2)
        and forecast_rms < float(target_rms_error_deg)
        and fit_r2 > float(target_fit_r2)
        and fit_rms < float(target_fit_rms_error_deg)
    )


def bayes_full_ode_target_reached(record: Dict[str, object], target_r2: float,
                                  target_rms_error_deg: float) -> bool:
    full_ode_r2 = finite_or(record.get("full_ode_r2"), -1.0e9)
    full_ode_rms = finite_or(record.get("full_ode_rms_error_deg"), float("inf"))
    return full_ode_r2 > float(target_r2) or full_ode_rms < float(target_rms_error_deg)


def bayes_stop_reason(record: Dict[str, object], target_r2: float, target_rms_error_deg: float,
                      target_fit_r2: float, target_fit_rms_error_deg: float,
                      target_full_ode_r2: float,
                      target_full_ode_rms_error_deg: float) -> Optional[str]:
    reasons: List[str] = []
    if bayes_target_reached(
        record, target_r2, target_rms_error_deg, target_fit_r2, target_fit_rms_error_deg,
    ):
        reasons.append("joint_fit_forecast")
    if finite_or(record.get("full_ode_r2"), -1.0e9) > float(target_full_ode_r2):
        reasons.append("full_ode_r2")
    if finite_or(record.get("full_ode_rms_error_deg"), float("inf")) < float(target_full_ode_rms_error_deg):
        reasons.append("full_ode_rmse")
    return "+".join(reasons) if reasons else None


def update_bayes_stop_status(summary: Dict[str, object], records: List[Dict[str, object]],
                             target_r2: float, target_rms_error_deg: float,
                             target_fit_r2: float, target_fit_rms_error_deg: float,
                             target_full_ode_r2: float,
                             target_full_ode_rms_error_deg: float,
                             ) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    for record in sorted(records, key=lambda item: int_or(item.get("trial_index"), 0)):
        if record.get("status") != "ok":
            continue
        reason = bayes_stop_reason(
            record,
            target_r2,
            target_rms_error_deg,
            target_fit_r2,
            target_fit_rms_error_deg,
            target_full_ode_r2,
            target_full_ode_rms_error_deg,
        )
        if reason is not None:
            summary["target_reached"] = True
            summary["stop_reason"] = reason
            summary["stop_trial_index"] = int_or(record.get("trial_index"), 0)
            return record, reason
    summary["target_reached"] = False
    summary["stop_reason"] = None
    summary["stop_trial_index"] = None
    return None, None


def nearest_categorical_value(value: object, values: List[object]) -> object:
    if value in values:
        return value
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if numeric_values and isinstance(value, (int, float)):
        return min(numeric_values, key=lambda v: abs(float(v) - float(value)))
    return values[0]


def current_search_params(cfg: Dict[str, object], space: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for name, spec in space.items():
        kind = str(spec["type"])
        if kind == "categorical":
            params[name] = nearest_categorical_value(cfg.get(name), list(spec["values"]))
        elif kind == "int":
            params[name] = int(round(finite_or(cfg.get(name), float(spec["low"]))))
        else:
            params[name] = float(finite_or(cfg.get(name), float(spec["low"])))
    return params


def sample_search_params(rng: np.random.Generator, space: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for name, spec in space.items():
        kind = str(spec["type"])
        if kind == "categorical":
            values = list(spec["values"])
            value = rng.choice(values)
            params[name] = value.item() if hasattr(value, "item") else value
        elif kind == "int":
            params[name] = int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
        else:
            low = float(spec["low"])
            high = float(spec["high"])
            if bool(spec.get("log", False)):
                params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                params[name] = float(rng.uniform(low, high))
    return params


def encode_search_params(params: Dict[str, object], space: Dict[str, Dict[str, object]]) -> np.ndarray:
    encoded: List[float] = []
    for name, spec in space.items():
        kind = str(spec["type"])
        value = params[name]
        if kind == "categorical":
            values = list(spec["values"])
            try:
                idx = values.index(value)
            except ValueError:
                idx = values.index(nearest_categorical_value(value, values))
            encoded.append(0.0 if len(values) == 1 else float(idx) / float(len(values) - 1))
        elif kind == "int":
            low = float(spec["low"])
            high = float(spec["high"])
            encoded.append((float(value) - low) / max(high - low, 1.0))
        else:
            low = float(spec["low"])
            high = float(spec["high"])
            if bool(spec.get("log", False)):
                encoded.append((np.log(float(value)) - np.log(low)) / max(np.log(high) - np.log(low), 1.0e-12))
            else:
                encoded.append((float(value) - low) / max(high - low, 1.0e-12))
    return np.asarray(encoded, dtype=np.float64)


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(diff * diff, axis=-1)
    return np.exp(-0.5 * sqdist / max(float(length_scale) ** 2, 1.0e-12))


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def gp_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_mean = float(np.mean(train_y))
    y_std = float(np.std(train_y))
    if y_std < 1.0e-8:
        y_std = 1.0
    y_scaled = (train_y - y_mean) / y_std
    length_scale = max(0.18, 0.85 / math.sqrt(max(train_x.shape[1], 1)))
    k_base = rbf_kernel(train_x, train_x, length_scale)

    chol = None
    for jitter in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3):
        try:
            chol = np.linalg.cholesky(k_base + jitter * np.eye(len(train_x)))
            break
        except np.linalg.LinAlgError:
            continue
    if chol is None:
        mean = np.full(test_x.shape[0], y_mean, dtype=np.float64)
        sigma = np.full(test_x.shape[0], y_std, dtype=np.float64)
        return mean, sigma

    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))
    k_xs = rbf_kernel(train_x, test_x, length_scale)
    mean_scaled = k_xs.T @ alpha
    v = np.linalg.solve(chol, k_xs)
    var_scaled = np.clip(1.0 - np.sum(v * v, axis=0), 1.0e-10, None)
    return mean_scaled * y_std + y_mean, np.sqrt(var_scaled) * y_std


def propose_bayes_candidate(rng: np.random.Generator, space: Dict[str, Dict[str, object]],
                            observed_params: List[Dict[str, object]], observed_vals: List[float],
                            candidate_pool: int, jitter: float) -> Tuple[Dict[str, object], str]:
    if len(observed_params) < 2:
        return sample_search_params(rng, space), "random"
    train_x = np.vstack([encode_search_params(p, space) for p in observed_params])
    train_y = np.asarray(observed_vals, dtype=np.float64)
    candidates = [sample_search_params(rng, space) for _ in range(max(int(candidate_pool), 32))]
    cand_x = np.vstack([encode_search_params(p, space) for p in candidates])
    mu, sigma = gp_predict(train_x, train_y, cand_x)
    best_y = float(np.min(train_y))
    improvement = best_y - mu - float(jitter)
    z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 1.0e-12)
    ei = np.where(sigma > 1.0e-12, improvement * normal_cdf(z) + sigma * normal_pdf(z), 0.0)
    return candidates[int(np.argmax(ei))], "expected_improvement"


def clamp_bayes_trial_cfg(cfg: Dict[str, object], t: np.ndarray) -> Dict[str, object]:
    n_points = len(t)
    val_frac = min(max(float(cfg.get("val_frac", 0.0)), 0.0), 0.9)
    forecast_region = choose_forecast_region(np.asarray(t, dtype=float), cfg)
    forecast_points = (
        int(forecast_region[1] - forecast_region[0] + 1)
        if forecast_region is not None else 0
    )
    validation_points = int(round(n_points * val_frac))
    train_points = int(max(1, n_points - validation_points - forecast_points))
    max_seq = max(8, min(n_points - 4, train_points - 1))
    cfg["seq_len"] = int(min(max(8, int(cfg["seq_len"])), max_seq))
    cfg["stride"] = int(max(1, int(cfg["stride"])))
    cfg["turning_point_sampling_repeats"] = int(
        max(0, int(cfg.get("turning_point_sampling_repeats", 0)))
    )
    cfg["turning_point_sampling_neighbourhood"] = int(
        max(1, int(cfg.get("turning_point_sampling_neighbourhood", 4)))
    )
    cfg["turning_point_sampling_min_prominence_deg"] = float(
        max(0.0, float(cfg.get("turning_point_sampling_min_prominence_deg", 0.0)))
    )
    cfg["prediction_stride"] = int(min(max(1, int(cfg.get("prediction_stride", cfg["stride"]))), cfg["seq_len"]))
    cfg["batch_size"] = int(max(1, int(cfg["batch_size"])))
    if str(cfg.get("runtime_profile", "auto")).strip().lower() == "cpu_large_memory":
        cfg["batch_size"] = int(max(cfg["batch_size"], int(cfg.get("cpu_large_memory_min_batch_size", 64))))

    cfg["lstm_hidden_size"] = int(max(8, int(cfg["lstm_hidden_size"])))
    cfg["lstm_layers"] = int(max(1, int(cfg["lstm_layers"])))
    cfg["fc_hidden"] = int(max(8, int(cfg["fc_hidden"])))
    cfg["lstm_dropout"] = float(np.clip(float(cfg["lstm_dropout"]), 0.0, 0.95))

    cfg["c_roll_min"] = float(max(float(cfg.get("c_roll_min", 0.0)), 1.0e-8))
    cfg["c_quad_min"] = float(max(float(cfg.get("c_quad_min", 0.0)), 1.0e-8))
    cfg["k_roll_min"] = float(max(float(cfg.get("k_roll_min", 0.0)), 1.0e-8))
    cfg["c_quad_max"] = float(max(float(cfg.get("c_quad_max", 0.02)), cfg["c_quad_min"] * 1.1))
    cfg["c_roll_init"] = float(max(float(cfg["c_roll_init"]), cfg["c_roll_min"] * 1.01))
    cfg["c_quad_init"] = float(min(max(float(cfg["c_quad_init"]), cfg["c_quad_min"] * 1.01), cfg["c_quad_max"]))
    cfg["k_roll_init"] = float(max(float(cfg["k_roll_init"]), cfg["k_roll_min"] * 1.01))
    for key in [
        "lambda_data", "lambda_r2_data", "lambda_rate_data", "lambda_roll_slope", "lambda_kinematic", "lambda_physics", "lambda_boundary",
        "lambda_force_reg", "lambda_force_smooth",
        "wave_force_target_smoothing_seconds",
        "lambda_total_force_target",
        "total_force_target_phase_weight",
        "lambda_total_force_band_shape",
        "total_force_band_shape_phase_weight",
        "total_force_band_shape_event_weight",
        "total_force_band_shape_envelope_weight",
        "lambda_total_force_spectral_shape",
        "total_force_spectral_shape_underfit_weight",
        "total_force_spectral_shape_complex_weight",
        "total_force_spectral_shape_low_excess_margin",
        "total_force_target_underforce_weight",
        "force_target_raw_weight",
        "wave_force_target_phase_weight", "wave_force_target_amplitude_weight",
        "wave_force_target_underforce_weight", "lambda_wave_force_highpass",
        "wave_force_highpass_window_s",
        "force_band_amplitude_underfit_weight",
        "lambda_total_force_event", "total_force_event_weight",
        "lambda_force_envelope", "force_envelope_window_s", "force_envelope_huber_delta",
        "wave_envelope_gate_gain",
        "lambda_peak_data", "lambda_peak_trough", "force_residual_scale", "wave_forcing_gain",
        "state_forcing_gain", "wave_gate_bias", "wave_gate_gain",
        "wave_parallel_min_gain", "wave_parallel_velocity_blend",
        "wave_parallel_obliquity_power", "wave_parallel_gate_smoothing_seconds",
        "motion_feedback_backbone_gain", "motion_feedback_gate_gain",
        "turn_moment_scale", "peak_trough_underfit_weight", "lambda_amplitude_underfit",
        "lambda_rollout", "lambda_rollout_rate", "lambda_rollout_amplitude", "lambda_rollout_phase",
        "lambda_rollout_peak_trough", "lambda_rollout_global_extrema",
        "lambda_global_extrema", "lambda_high_pass_residual",
        "lambda_low_pass_residual", "lambda_roll_spectral_shape",
        "roll_spectral_shape_underfit_weight", "roll_spectral_shape_complex_weight",
        "roll_spectral_shape_mag_floor_ratio",
        "curvature_high_pass_window_s", "curvature_scale_floor_ratio",
        "lambda_rollout_local_prominence", "lambda_rollout_high_pass_residual",
        "lambda_rollout_roll_spectral_shape",
        "direct_forecast_loss_window_s",
        "direct_forecast_tail_multiplier",
        "direct_forecast_turning_point_weight",
        "direct_forecast_turning_point_min_prominence_deg",
        "lambda_direct_forecast", "lambda_direct_forecast_rate",
        "lambda_direct_forecast_amplitude", "lambda_direct_forecast_phase",
        "lambda_direct_forecast_peak_trough", "lambda_direct_forecast_global_extrema",
        "lambda_direct_forecast_extrema_window_underfit",
        "lambda_direct_forecast_high_amplitude_underfit",
        "lambda_direct_forecast_asymmetric_extrema",
        "direct_forecast_peak_underfit_weight",
        "direct_forecast_peak_overshoot_weight",
        "direct_forecast_trough_overshoot_weight",
        "direct_forecast_trough_underfit_weight",
        "direct_forecast_asym_extrema_scale_floor_ratio",
        "lambda_direct_forecast_local_prominence",
        "lambda_direct_forecast_high_pass_residual",
        "lambda_direct_forecast_roll_spectral_shape",
        "lambda_direct_forecast_roll_curvature",
        "direct_forecast_amplitude_calibration_gain_min",
        "direct_forecast_amplitude_calibration_gain_max",
        "direct_forecast_amplitude_calibration_extrema_weight",
        "direct_forecast_selection_r2_weight",
        "direct_forecast_selection_loss_weight",
        "direct_forecast_selection_peak_weight",
        "direct_forecast_selection_extrema_weight",
        "direct_forecast_selection_asym_extrema_weight",
        "direct_forecast_selection_spectral_weight",
        "global_extrema_location_weight", "global_extrema_softmax_beta",
        "amplitude_window_s", "amplitude_underfit_margin", "amplitude_underfit_power",
        "lambda_extrema_window_underfit", "extrema_window_overshoot_weight",
        "extrema_window_peak_weight", "extrema_window_trough_weight",
        "extrema_window_scale_floor_ratio", "lambda_rollout_extrema_window_underfit",
        "lambda_high_amplitude_underfit", "high_amplitude_underfit_overshoot_weight",
        "high_amplitude_underfit_scale_floor_ratio", "lambda_rollout_high_amplitude_underfit",
        "roll_amplitude_calibration_gain_init", "roll_amplitude_calibration_gain_max",
        "roll_amplitude_calibration_threshold_scaled", "roll_amplitude_calibration_softness_scaled",
        "local_prominence_window_s", "local_prominence_scale_floor_ratio", "local_prominence_value_weight",
        "local_prominence_shape_weight", "high_pass_window_s", "high_pass_scale_floor_ratio",
        "low_pass_window_s", "low_pass_scale_floor_ratio",
        "rollout_window_s", "kinematic_weight_alpha",
    ]:
        cfg[key] = float(max(0.0, float(cfg[key])))
    cfg["rollout_min_steps"] = int(max(2, int(cfg.get("rollout_min_steps", 8))))
    cfg["direct_forecast_loss_min_steps"] = int(
        max(2, int(cfg.get("direct_forecast_loss_min_steps", 8)))
    )
    cfg["direct_forecast_loss_batch_size"] = int(
        max(1, int(cfg.get("direct_forecast_loss_batch_size", 1)))
    )
    cfg["direct_forecast_turning_point_max_tasks"] = int(
        max(0, int(cfg.get("direct_forecast_turning_point_max_tasks", 0)))
    )
    cfg["direct_forecast_turning_point_neighbourhood"] = int(
        max(1, int(cfg.get("direct_forecast_turning_point_neighbourhood", 4)))
    )
    cfg["direct_forecast_tail_fraction"] = float(
        np.clip(float(cfg.get("direct_forecast_tail_fraction", 0.0)), 0.0, 1.0)
    )
    cfg["direct_forecast_tail_multiplier"] = float(
        max(1.0, float(cfg.get("direct_forecast_tail_multiplier", 1.0)))
    )
    cfg["direct_forecast_amplitude_calibration_gain_min"] = float(
        max(0.0, float(cfg.get("direct_forecast_amplitude_calibration_gain_min", 0.85)))
    )
    cfg["direct_forecast_amplitude_calibration_gain_max"] = float(
        max(
            cfg["direct_forecast_amplitude_calibration_gain_min"],
            float(cfg.get("direct_forecast_amplitude_calibration_gain_max", 1.25)),
        )
    )
    cfg["direct_forecast_amplitude_calibration_gain_steps"] = int(
        max(2, int(cfg.get("direct_forecast_amplitude_calibration_gain_steps", 41)))
    )
    cfg["direct_forecast_asym_extrema_radius"] = int(
        max(0, int(cfg.get("direct_forecast_asym_extrema_radius", 4)))
    )
    cfg["direct_forecast_asym_extrema_quantile"] = float(
        np.clip(float(cfg.get("direct_forecast_asym_extrema_quantile", 0.45)), 0.01, 0.99)
    )
    cfg["rollout_every_n_batches"] = int(max(1, int(cfg.get("rollout_every_n_batches", 1))))
    cfg["rollout_warmup_epochs"] = int(max(0, int(cfg.get("rollout_warmup_epochs", 0))))
    cfg["peak_data_quantile"] = float(np.clip(float(cfg["peak_data_quantile"]), 0.01, 0.99))
    cfg["peak_data_alpha"] = float(max(0.0, float(cfg["peak_data_alpha"])))
    cfg["peak_trough_quantile"] = float(np.clip(float(cfg["peak_trough_quantile"]), 0.01, 0.99))
    cfg["peak_trough_neighbourhood"] = int(max(1, int(cfg["peak_trough_neighbourhood"])))
    cfg["total_force_event_quantile"] = float(
        np.clip(float(cfg.get("total_force_event_quantile", 0.72)), 0.0, 1.0)
    )
    cfg["extrema_window_radius"] = int(max(0, int(cfg.get("extrema_window_radius", 3))))
    cfg["extrema_window_quantile"] = float(np.clip(float(cfg.get("extrema_window_quantile", 0.55)), 0.01, 0.99))
    cfg["high_amplitude_underfit_quantile"] = float(
        np.clip(float(cfg.get("high_amplitude_underfit_quantile", 0.68)), 0.01, 0.99)
    )
    cfg["roll_amplitude_calibration_gain_max"] = float(
        max(0.0, float(cfg.get("roll_amplitude_calibration_gain_max", 0.40)))
    )
    cfg["roll_amplitude_calibration_gain_init"] = float(
        np.clip(
            float(cfg.get("roll_amplitude_calibration_gain_init", 0.08)),
            0.0,
            cfg["roll_amplitude_calibration_gain_max"],
        )
    )
    cfg["amplitude_quantile"] = float(np.clip(float(cfg["amplitude_quantile"]), 0.01, 0.99))
    cfg["local_prominence_quantile"] = float(np.clip(float(cfg["local_prominence_quantile"]), 0.01, 0.99))
    cfg["local_prominence_neighbourhood"] = int(max(1, int(cfg["local_prominence_neighbourhood"])))
    cfg["kinematic_weight_quantile"] = float(np.clip(float(cfg["kinematic_weight_quantile"]), 0.01, 0.99))
    cfg["wave_parallel_min_gain"] = float(np.clip(float(cfg["wave_parallel_min_gain"]), 0.0, 1.0))
    cfg["wave_parallel_velocity_blend"] = float(np.clip(float(cfg["wave_parallel_velocity_blend"]), 0.0, 1.0))
    cfg["wave_parallel_velocity_reference_quantile"] = float(np.clip(float(cfg["wave_parallel_velocity_reference_quantile"]), 0.01, 0.99))
    cfg["wave_parallel_obliquity_power"] = float(max(1.0e-6, float(cfg["wave_parallel_obliquity_power"])))
    cfg["wave_envelope_gate_min"] = float(max(0.0, float(cfg.get("wave_envelope_gate_min", 0.35))))
    cfg["wave_envelope_gate_max"] = float(
        max(
            cfg["wave_envelope_gate_min"] + 1.0e-6,
            float(cfg.get("wave_envelope_gate_max", 2.20)),
        )
    )
    return cfg


def build_bayes_trial_cfg(base_cfg: Dict[str, object], trial_params: Dict[str, object],
                          trial_idx: int, t: np.ndarray) -> Dict[str, object]:
    cfg = copy.deepcopy(base_cfg)
    cfg.update(trial_params)
    cfg["epochs"] = int(base_cfg.get("bayes_trial_epochs", 150))
    cfg["checkpoint_dir"] = str(Path(str(base_cfg["bayes_opt_dir"])) / f"trial_{trial_idx:05d}" / "checkpoints")
    cfg["output_dir"] = str(Path(str(base_cfg["bayes_opt_dir"])) / f"trial_{trial_idx:05d}" / "outputs")
    # Rev M never writes tensor checkpoints during Bayes. Trial parameters and
    # metrics remain in compact JSON files, which are sufficient for acquisition
    # and trial-boundary resume.
    cfg["save_checkpoints"] = False
    cfg["checkpoint_policy"] = "disabled"
    cfg["save_best_checkpoint"] = False
    cfg["save_periodic_checkpoints"] = False
    cfg["save_last_checkpoint"] = False
    cfg["save_final_checkpoint"] = False
    cfg["checkpoint_every"] = 0
    cfg["freeze_physics_coefficients"] = bool(
        base_cfg.get("bayes_freeze_physics_coefficients", True)
    )
    cfg["save_plots"] = bool(base_cfg.get("bayes_save_trial_plots", False))
    progress_every = max(1, int(base_cfg.get("bayes_log_every", 50)))
    cfg["bayes_compact_logging"] = True
    cfg["bayes_trial_index"] = int(trial_idx)
    cfg["bayes_total_trials"] = int(base_cfg.get("bayes_trials", trial_idx))
    cfg["log_every"] = min(progress_every, max(int(cfg["epochs"]), 1))
    cfg["val_every"] = max(1, min(int(cfg.get("val_every", progress_every)), max(int(cfg["epochs"]), 1)))
    return clamp_bayes_trial_cfg(cfg, t)


@torch.no_grad()
def evaluate_full_fit_metrics(model: PINNLSTM, data: Dict[str, object],
                              cfg: Dict[str, object]) -> Dict[str, object]:
    pred_phi_s, _, _, _, _ = predict_full_teacher_forced(model, data, cfg)
    metrics = compute_all_metrics(data, pred_phi_s, cfg)
    metrics["physics_parameters"]["c_roll"] = float(model.c_roll.detach().cpu().item())
    metrics["physics_parameters"]["c_quad"] = float(model.c_quad.detach().cpu().item())
    metrics["physics_parameters"]["k_roll"] = float(model.k_roll.detach().cpu().item())
    metrics["roll_amplitude_calibration"] = {
        "enabled": bool(cfg.get("roll_amplitude_calibration_enabled", False)),
        "gain": float(model.roll_amplitude_gain.detach().cpu().item()),
        "gain_max": float(cfg.get("roll_amplitude_calibration_gain_max", 0.0)),
        "threshold_scaled": float(cfg.get("roll_amplitude_calibration_threshold_scaled", 0.0)),
        "softness_scaled": float(cfg.get("roll_amplitude_calibration_softness_scaled", 0.0)),
        "rate_scale": bool(cfg.get("roll_amplitude_calibration_rate_scale", True)),
    }
    return metrics


def run_bayes_training_trial(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                             base_cfg: Dict[str, object], trial_params: Dict[str, object],
                             trial_idx: int, lg: logging.Logger,
                             data_cache: Optional[BayesDataCache] = None,
                             progress: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    cfg = build_bayes_trial_cfg(base_cfg, trial_params, trial_idx, t)
    if progress:
        cfg.update(copy.deepcopy(progress))
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("save_checkpoints", False)):
        Path(str(cfg["checkpoint_dir"])).mkdir(parents=True, exist_ok=True)
    set_global_seed(int(base_cfg["seed"]))

    if data_cache is not None and bool(cfg.get("bayes_reuse_prepared_data", True)):
        data = data_cache.get_prepared(cfg, lg)
        shared_cache = data_cache.loader_shared_cache(cfg)
    else:
        data = prepare_data(t, phi, wave_pack, cfg, lg)
        shared_cache = None
    train_loader, val_loader = build_loaders(data, cfg, lg, shared_cache=shared_cache)
    model = build_model(cfg, data)
    resume_path: Optional[Path] = None
    if bool(base_cfg.get("bayes_resume", True)) and bool(cfg.get("save_checkpoints", False)):
        checkpoint_dir = Path(str(cfg["checkpoint_dir"]))
        for candidate in (checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"):
            if candidate.exists():
                resume_path = candidate
                break
        if resume_path is not None:
            lg.info("[BO] Resuming trial %05d from checkpoint: %s", trial_idx, resume_path)

    try:
        train_result = train_model(
            model, train_loader, val_loader, data, cfg, lg,
            resume=str(resume_path) if resume_path is not None else None,
        )
    except Exception as exc:
        if resume_path is None:
            raise
        lg.warning("[BO] Could not resume trial %05d from %s (%s). Restarting this trial from epoch 0.",
                   trial_idx, resume_path, exc)
        train_result = train_model(model, train_loader, val_loader, data, cfg, lg, resume=None)

    if not train_result.get("history"):
        train_metrics = {f"train_{k}": v for k, v in evaluate_loader(model, train_loader, cfg).items()}
        val_metrics = {f"val_{k}": v for k, v in evaluate_loader(model, val_loader, cfg).items()} if val_loader is not None else {}
        train_result["history"] = [{
            "epoch": float(cfg["epochs"]),
            **train_metrics,
            **val_metrics,
        }]
    bayes_meta = {
        "trial_index": int(trial_idx),
        "params": copy.deepcopy(trial_params),
        "epochs": int(cfg["epochs"]),
    }
    paths = write_outputs(
        model,
        data,
        cfg,
        train_result,
        lg,
        extra_metrics={"bayes_trial": bayes_meta},
        include_full_ode=bool(cfg.get("bayes_full_ode_outputs", True)),
        full_ode_coefficient_source="model",
    )

    metrics = read_json_dict(Path(str(paths["metrics"]))) or {}
    full = metrics.get("full", {})
    if not isinstance(full, dict):
        full = {}
    forecast = metrics.get("forecast", {})
    if not isinstance(forecast, dict):
        forecast = {}
    forecast_metrics = forecast.get("metrics", {}) if bool(forecast.get("enabled", True)) else {}
    if not isinstance(forecast_metrics, dict):
        forecast_metrics = {}
    full_ode_payload = metrics.get("full_ode", {})
    if not isinstance(full_ode_payload, dict):
        full_ode_payload = {}
    full_ode_metric_groups = full_ode_payload.get("metrics", {})
    if not isinstance(full_ode_metric_groups, dict):
        full_ode_metric_groups = {}
    full_ode_metrics = full_ode_metric_groups.get("full", {})
    if not isinstance(full_ode_metrics, dict):
        full_ode_metrics = {}
    record = {
        "status": "ok",
        "trial_index": int(trial_idx),
        "fit_r2": float(full["r2"]),
        "fit_rms_error_deg": float(full["error_rms_deg"]),
        "fit_rmse_deg": float(full["rmse_deg"]),
        "fit_measured_rms_deg": float(full["measured_rms_deg"]),
        "fit_curve_rms_deg": float(full["fit_rms_deg"]),
        "forecast_r2": finite_or(forecast_metrics.get("r2"), -1.0e9),
        "forecast_rms_error_deg": finite_or(forecast_metrics.get("error_rms_deg"), finite_or(forecast_metrics.get("rmse_deg"), float("inf"))),
        "forecast_rmse_deg": finite_or(forecast_metrics.get("rmse_deg"), float("inf")),
        "forecast_mae_deg": finite_or(forecast_metrics.get("mae_deg"), float("inf")),
        "forecast_phase_lag_s": finite_or(forecast_metrics.get("phase_lag_s"), 0.0),
        "forecast_phase_corr": finite_or(forecast_metrics.get("phase_corr"), float("nan")),
        "forecast_phase_aligned_rmse_deg": finite_or(forecast_metrics.get("phase_aligned_rmse_deg"), float("inf")),
        "forecast_amplitude_ratio": finite_or(forecast_metrics.get("amplitude_ratio"), float("nan")),
        "forecast_amplitude_underfit_ratio": finite_or(forecast_metrics.get("amplitude_underfit_ratio"), 1.0),
        "forecast_extrema_rmse_deg": finite_or(forecast_metrics.get("extrema_rmse_deg"), float("inf")),
        "forecast_extrema_relative_error": finite_or(forecast_metrics.get("extrema_relative_error"), 1.0),
        "forecast_extrema_range_ratio": finite_or(forecast_metrics.get("extrema_range_ratio"), float("nan")),
        "forecast_extrema_point_rmse_deg": finite_or(forecast_metrics.get("extrema_point_rmse_deg"), float("inf")),
        "forecast_extrema_point_mae_deg": finite_or(forecast_metrics.get("extrema_point_mae_deg"), float("inf")),
        "forecast_peak_underfit_deg": finite_or(forecast_metrics.get("peak_underfit_deg"), float("inf")),
        "forecast_peak_underfit_max_deg": finite_or(forecast_metrics.get("peak_underfit_max_deg"), float("inf")),
        "forecast_trough_overshoot_deg": finite_or(forecast_metrics.get("trough_overshoot_deg"), float("inf")),
        "forecast_trough_overshoot_max_deg": finite_or(forecast_metrics.get("trough_overshoot_max_deg"), float("inf")),
        "forecast_start_time_s": forecast.get("start_time_s"),
        "forecast_end_time_s": forecast.get("end_time_s"),
        "forecast_duration_s": forecast.get("duration_s"),
        "full_ode_r2": finite_or(full_ode_metrics.get("r2"), -1.0e9),
        "full_ode_rms_error_deg": finite_or(full_ode_metrics.get("error_rms_deg"), finite_or(full_ode_metrics.get("rmse_deg"), float("inf"))),
        "full_ode_rmse_deg": finite_or(full_ode_metrics.get("rmse_deg"), float("inf")),
        "full_ode_mae_deg": finite_or(full_ode_metrics.get("mae_deg"), float("inf")),
        "best_epoch": int(train_result.get("best_epoch", 0)),
        "best_metric": float(train_result.get("best_metric", float("nan"))),
        "final_train_physics_loss": final_history_value(train_result, "train_physics"),
        "final_train_total_force_target_loss": final_history_value(
            train_result, "train_total_force_target"
        ),
        "final_train_total_force_event_loss": final_history_value(
            train_result, "train_total_force_event"
        ),
        "final_train_total_force_band_shape_loss": final_history_value(
            train_result, "train_total_force_band_shape"
        ),
        "final_train_total_force_spectral_shape_loss": final_history_value(
            train_result, "train_total_force_spectral_shape"
        ),
        "final_train_force_envelope_loss": final_history_value(
            train_result, "train_force_envelope"
        ),
        "final_train_kinematic_loss": final_history_value(train_result, "train_kinematic"),
        "final_train_data_loss": final_history_value(train_result, "train_data"),
        "final_train_global_extrema_loss": final_history_value(train_result, "train_global_extrema"),
        "final_train_extrema_window_underfit_loss": final_history_value(train_result, "train_extrema_window_underfit"),
        "final_train_high_amplitude_underfit_loss": final_history_value(train_result, "train_high_amplitude_underfit"),
        "final_train_high_pass_residual_loss": final_history_value(train_result, "train_high_pass_residual"),
        "final_train_low_pass_residual_loss": final_history_value(train_result, "train_low_pass_residual"),
        "final_train_roll_spectral_shape_loss": final_history_value(train_result, "train_roll_spectral_shape"),
        "final_train_amplitude_underfit_loss": final_history_value(train_result, "train_amplitude_underfit"),
        "final_train_rollout_loss": final_history_value(train_result, "train_rollout"),
        "final_train_rollout_peak_trough_loss": final_history_value(
            train_result, "train_rollout_peak_trough"
        ),
        "final_train_rollout_global_extrema_loss": final_history_value(
            train_result, "train_rollout_global_extrema"
        ),
        "final_train_rollout_extrema_window_underfit_loss": final_history_value(
            train_result, "train_rollout_extrema_window_underfit"
        ),
        "final_train_rollout_high_amplitude_underfit_loss": final_history_value(
            train_result, "train_rollout_high_amplitude_underfit"
        ),
        "final_train_rollout_local_prominence_loss": final_history_value(train_result, "train_rollout_local_prominence"),
        "final_train_rollout_high_pass_residual_loss": final_history_value(train_result, "train_rollout_high_pass_residual"),
        "final_train_rollout_roll_spectral_shape_loss": final_history_value(train_result, "train_rollout_roll_spectral_shape"),
        "final_train_rollout_phase_loss": final_history_value(train_result, "train_rollout_phase"),
        "bayes_r2_objective_weight": float(base_cfg.get("bayes_r2_objective_weight", 10.0)),
        "bayes_rmse_objective_weight": float(base_cfg.get("bayes_rmse_objective_weight", 0.0)),
        "bayes_fit_r2_objective_weight": float(base_cfg.get("bayes_fit_r2_objective_weight", 10.0)),
        "bayes_fit_rmse_objective_weight": float(base_cfg.get("bayes_fit_rmse_objective_weight", 0.0)),
        "bayes_phase_objective_weight": float(base_cfg.get("bayes_phase_objective_weight", 0.0)),
        "bayes_amplitude_objective_weight": float(base_cfg.get("bayes_amplitude_objective_weight", 0.0)),
        "bayes_extrema_objective_weight": float(base_cfg.get("bayes_extrema_objective_weight", 0.0)),
        "bayes_forecast_extrema_rmse_objective_weight": float(
            base_cfg.get("bayes_forecast_extrema_rmse_objective_weight", 0.0)
        ),
        "bayes_forecast_peak_underfit_objective_weight": float(
            base_cfg.get("bayes_forecast_peak_underfit_objective_weight", 0.0)
        ),
        "bayes_forecast_trough_overshoot_objective_weight": float(
            base_cfg.get("bayes_forecast_trough_overshoot_objective_weight", 0.0)
        ),
        "bayes_turning_objective_weight": float(base_cfg.get("bayes_turning_objective_weight", 0.0)),
        "bayes_physics_objective_weight": float(base_cfg.get("bayes_physics_objective_weight", 0.0)),
        "bayes_kinematic_objective_weight": float(base_cfg.get("bayes_kinematic_objective_weight", 0.0)),
        "bo_objective": 0.0,
        "params": copy.deepcopy(trial_params),
        "trial_epochs": int(cfg["epochs"]),
        "output_dir": str(output_dir),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    record["bo_objective"] = bayes_objective_from_record(record)

    del model, train_loader, val_loader, data, metrics, forecast, train_result
    if data_cache is None:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def write_bayes_summary(path: Path, summary: Dict[str, object]) -> None:
    safe_json_dump(summary, path)


def read_json_dict(path: Path) -> Optional[Dict[str, object]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def int_or(value: object, fallback: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(fallback)


def bayes_run_label(record: object) -> str:
    if not isinstance(record, dict):
        return "none"
    trial_idx = int_or(record.get("trial_index"), 0)
    return f"{trial_idx:05d}" if trial_idx > 0 else "none"


def bayes_trial_dir(bayes_dir: Path, trial_idx: int) -> Path:
    return bayes_dir / f"trial_{int(trial_idx):05d}"


def bayes_trial_state_path(trial_dir: Path) -> Path:
    return trial_dir / "bayes_trial_state_revm.json"


def read_bayes_trial_state(trial_dir: Path) -> Optional[Dict[str, object]]:
    return (
        read_json_dict(bayes_trial_state_path(trial_dir))
        or read_json_dict(trial_dir / "bayes_trial_state_revh.json")
        or read_json_dict(trial_dir / "bayes_trial_state_revg.json")
        or read_json_dict(trial_dir / "bayes_trial_state_reve.json")
    )


def write_bayes_trial_state(trial_dir: Path, updates: Dict[str, object]) -> None:
    state = read_bayes_trial_state(trial_dir) or {}
    state.update(copy.deepcopy(updates))
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    safe_json_dump(state, bayes_trial_state_path(trial_dir))


def bayes_trial_index_from_dir(trial_dir: Path) -> int:
    m = re.match(r"trial_(\d+)$", trial_dir.name)
    return int(m.group(1)) if m else 0


def grid_values(low: float, high: float, step: float) -> List[float]:
    values: List[float] = []
    x = float(low)
    high_f = float(high)
    step_f = max(float(step), 1.0e-12)
    while x <= high_f + 1.0e-10:
        values.append(round(float(x), 10))
        x += step_f
    if not values or abs(values[-1] - high_f) > 1.0e-9:
        values.append(round(high_f, 10))
    return values


def build_grid_search_values(cfg: Dict[str, object]) -> Dict[str, List[float]]:
    step = float(cfg.get("grid_step", 0.25))
    values: Dict[str, List[float]] = {}
    for key, (low, high) in GRID_SEARCH_SPACE.items():
        key_step = float(GRID_SEARCH_STEPS.get(key, step))
        raw_values = grid_values(float(low), float(high), key_step)
        current = float(cfg.get(key, raw_values[0]))
        values[key] = sorted(raw_values, key=lambda item: (abs(float(item) - current), float(item)))
    return values


def grid_trial_params_from_index(values: Dict[str, List[float]], index_zero_based: int) -> Dict[str, object]:
    keys = list(values.keys())
    lengths = [len(values[key]) for key in keys]
    total = int(math.prod(lengths))
    if index_zero_based < 0 or index_zero_based >= total:
        raise IndexError(f"Grid index {index_zero_based} outside 0..{total - 1}")
    remainder = int(index_zero_based)
    params: Dict[str, object] = {}
    for key, length in reversed(list(zip(keys, lengths))):
        choice_idx = remainder % length
        remainder //= length
        params[key] = values[key][choice_idx]
    return params


def write_grid_best_payload(grid_dir: Path, summary: Dict[str, object]) -> None:
    best = summary.get("best_trial")
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_trials": summary.get("completed_trials", 0),
        "total_grid_trials": summary.get("total_grid_trials", 0),
        "best_trial": copy.deepcopy(best),
        "best_params": copy.deepcopy(best.get("params")) if isinstance(best, dict) else None,
    }
    safe_json_dump(payload, grid_dir / "best_grid_hyperparameters_revm.json")


def run_grid_search(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                    cfg: Dict[str, object], lg: logging.Logger) -> Dict[str, object]:
    grid_dir = resolve_runtime_path(str(cfg.get("grid_opt_dir", "Grid_RevM_SpectralRollout")))
    grid_dir.mkdir(parents=True, exist_ok=True)
    summary_path = grid_dir / "grid_search_results_revm.json"
    resume_enabled = bool(cfg.get("grid_resume", True))
    previous_summary = read_json_dict(summary_path) if resume_enabled else None
    records = list(previous_summary.get("trials", [])) if isinstance(previous_summary, dict) else []
    records = [record for record in records if isinstance(record, dict)]
    completed_indices = {int_or(record.get("trial_index"), 0) for record in records}
    grid_value_map = build_grid_search_values(cfg)
    total_grid_trials = int(math.prod([len(v) for v in grid_value_map.values()]))
    max_new_trials = int(cfg.get("grid_max_trials", 0))

    best_record = max(
        [record for record in records if record.get("status") == "ok"],
        key=bayes_fit_rank_key,
        default=None,
    )
    summary = {
        "study_type": "deterministic_grid",
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "same composite fit/forecast rank used for Rev M search summaries",
        "grid_step": float(cfg.get("grid_step", 0.25)),
        "grid_step_overrides": copy.deepcopy(GRID_SEARCH_STEPS),
        "grid_value_order": "nearest-current-first per parameter, Cartesian product",
        "search_space": copy.deepcopy(GRID_SEARCH_SPACE),
        "grid_values": copy.deepcopy(grid_value_map),
        "locked_parameters_source": "active CONFIG outside GRID_SEARCH_SPACE",
        "total_grid_trials": total_grid_trials,
        "max_new_trials_this_run": max_new_trials,
        "completed_trials": len(records),
        "best_trial": copy.deepcopy(best_record),
        "trials": records,
    }
    update_best_forecast_metric_records(summary, records)
    write_bayes_summary(summary_path, summary)
    write_grid_best_payload(grid_dir, summary)

    lg.warning(
        "[GRID] Full 0.25 grid contains %d trials. At 150 epochs each this is a long-running search; "
        "use --grid-max-trials to run it in chunks.",
        total_grid_trials,
    )

    base_cfg = copy.deepcopy(cfg)
    base_cfg["bayes_opt_dir"] = str(grid_dir)
    base_cfg["bayes_trials"] = total_grid_trials
    base_cfg["bayes_resume"] = bool(cfg.get("grid_resume", True))
    base_cfg["bayes_save_trial_plots"] = bool(cfg.get("grid_save_trial_plots", True))
    data_cache = (
        BayesDataCache(t, phi, wave_pack)
        if bool(base_cfg.get("bayes_reuse_prepared_data", True)) else None
    )
    new_trials_run = 0

    for trial_idx in range(1, total_grid_trials + 1):
        if trial_idx in completed_indices:
            continue
        if max_new_trials > 0 and new_trials_run >= max_new_trials:
            break
        params = grid_trial_params_from_index(grid_value_map, trial_idx - 1)
        trial_dir = bayes_trial_dir(grid_dir, trial_idx)
        write_bayes_trial_state(trial_dir, {
            "status": "running",
            "trial_index": int(trial_idx),
            "selection_method": "deterministic_grid",
            "params": copy.deepcopy(params),
            "trial_epochs": int(base_cfg.get("bayes_trial_epochs", 150)),
            "started_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
        })
        lg.info("[GRID] Run %05d/%05d | params=%s", trial_idx, total_grid_trials, params)
        start = time.time()
        try:
            record = run_bayes_training_trial(
                t, phi, wave_pack, base_cfg, params, trial_idx, lg,
                data_cache=data_cache,
                progress={
                    "bayes_total_trials": int(total_grid_trials),
                    "bayes_best_composite_run": bayes_run_label(best_record),
                    "bayes_best_r2_run": bayes_run_label(summary.get("best_forecast_r2_trial")),
                    "bayes_best_rmse_run": bayes_run_label(summary.get("best_forecast_rmse_trial")),
                },
            )
        except KeyboardInterrupt:
            write_bayes_trial_state(trial_dir, {
                "status": "interrupted",
                "interrupted_at": datetime.now().isoformat(timespec="seconds"),
                "duration_seconds": float(time.time() - start),
            })
            summary["interrupted"] = True
            summary["interrupted_at_trial"] = int(trial_idx)
            write_bayes_summary(summary_path, summary)
            write_grid_best_payload(grid_dir, summary)
            raise
        except Exception as exc:
            lg.exception("[GRID] Run %05d/%05d | status=failed", trial_idx, total_grid_trials)
            record = {
                "status": "failed",
                "trial_index": int(trial_idx),
                "bo_objective": 1.0e9,
                "params": copy.deepcopy(params),
                "trial_epochs": int(base_cfg.get("bayes_trial_epochs", 150)),
                "error": str(exc),
            }

        record["selection_method"] = "deterministic_grid"
        record["duration_seconds"] = float(time.time() - start)
        records.append(record)
        completed_indices.add(trial_idx)
        new_trials_run += 1
        write_bayes_trial_state(trial_dir, {
            "status": str(record.get("status", "unknown")),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": float(record["duration_seconds"]),
            "forecast_r2": record.get("forecast_r2"),
            "forecast_rms_error_deg": record.get("forecast_rms_error_deg"),
            "fit_r2": record.get("fit_r2"),
            "fit_rms_error_deg": record.get("fit_rms_error_deg"),
            "params": copy.deepcopy(params),
            "error": record.get("error"),
        })
        if data_cache is not None:
            data_cache.release_between_trials(base_cfg, lg)
        else:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if record.get("status") == "ok" and (best_record is None or bayes_fit_rank_key(record) > bayes_fit_rank_key(best_record)):
            best_record = copy.deepcopy(record)
        summary.update({
            "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
            "completed_trials": len(records),
            "best_trial": copy.deepcopy(best_record),
            "trials": records,
        })
        update_best_forecast_metric_records(summary, records)
        write_bayes_summary(summary_path, summary)
        write_grid_best_payload(grid_dir, summary)
        if record.get("status") == "ok":
            lg.info(
                "[GRID] Run %05d/%05d complete | forecast R2=%.6f RMSE=%.6f deg | fit R2=%.6f | best=%s",
                trial_idx,
                total_grid_trials,
                finite_or(record.get("forecast_r2"), float("nan")),
                finite_or(record.get("forecast_rms_error_deg"), float("nan")),
                finite_or(record.get("fit_r2"), float("nan")),
                bayes_run_label(best_record),
            )

    summary.update({
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_trials": len(records),
        "best_trial": copy.deepcopy(best_record),
        "trials": records,
        "exhaustive_complete": len(completed_indices) >= total_grid_trials,
    })
    update_best_forecast_metric_records(summary, records)
    write_bayes_summary(summary_path, summary)
    write_grid_best_payload(grid_dir, summary)
    return summary


def existing_bayes_output_paths(output_dir: Path) -> Dict[str, str]:
    candidates = {
        "metrics": (output_dir / "metrics_revm.json", output_dir / "metrics_revh.json", output_dir / "metrics_revg.json", output_dir / "metrics_reve.json"),
        "config": (output_dir / "config_revm.json", output_dir / "config_revh.json", output_dir / "config_revg.json", output_dir / "config_reve.json"),
        "fit_plot": (output_dir / "roll_fit_revm.png", output_dir / "roll_fit_revh.png", output_dir / "roll_fit_revg.png", output_dir / "roll_fit_reve.png"),
        "fit_csv": (output_dir / "roll_fit_revm.csv",),
        "forecast_zoom_plot": (output_dir / "roll_forecast_zoom_revm.png", output_dir / "roll_forecast_zoom_revh.png", output_dir / "roll_forecast_zoom_revg.png", output_dir / "roll_forecast_zoom_reve.png"),
        "forecast_zoom_csv": (output_dir / "roll_forecast_zoom_revm.csv",),
        "forecast_csv": (output_dir / "roll_forecast_region_revm.csv", output_dir / "roll_forecast_region_revh.csv", output_dir / "roll_forecast_region_revg.csv", output_dir / "roll_forecast_region_reve.csv"),
        "lag_plot": (output_dir / "wave_lag_scan_revm.png", output_dir / "wave_lag_scan_revh.png", output_dir / "wave_lag_scan_revg.png", output_dir / "wave_lag_scan_reve.png"),
        "lag_csv": (output_dir / "wave_lag_scan_revm.csv",),
        "loss_plot": (output_dir / "loss_curves_revm.png", output_dir / "loss_curves_revh.png", output_dir / "loss_curves_revg.png", output_dir / "loss_curves_reve.png"),
        "loss_csv": (output_dir / "loss_curves_revm.csv",),
    }
    paths: Dict[str, str] = {}
    for name, options in candidates.items():
        for path in options:
            if path.exists():
                paths[name] = str(path)
                break
    return paths


def normalise_bayes_params(params: Dict[str, object], base_cfg: Dict[str, object]) -> Dict[str, object]:
    cfg = copy.deepcopy(base_cfg)
    cfg.update(params)
    return current_search_params(cfg, BAYES_SEARCH_SPACE)


def final_training_loss_from_metrics(training: Dict[str, object], key: str, fallback: float = 0.0) -> float:
    final_losses = training.get("final_losses", {})
    if isinstance(final_losses, dict) and key in final_losses:
        return finite_or(final_losses.get(key), fallback)
    legacy_key = f"final_{key}"
    if legacy_key in training:
        return finite_or(training.get(legacy_key), fallback)
    return float(fallback)


def record_from_bayes_trial_metrics(trial_dir: Path, base_cfg: Dict[str, object],
                                    lg: logging.Logger) -> Optional[Dict[str, object]]:
    output_dir = trial_dir / "outputs"
    metrics_path = output_dir / "metrics_revm.json"
    metrics = read_json_dict(metrics_path)
    if metrics is None:
        metrics_path = output_dir / "metrics_revh.json"
        metrics = read_json_dict(metrics_path)
    if metrics is None:
        metrics_path = output_dir / "metrics_revg.json"
        metrics = read_json_dict(metrics_path)
    if metrics is None:
        metrics_path = output_dir / "metrics_reve.json"
        metrics = read_json_dict(metrics_path)
    if metrics is None:
        return None

    full = metrics.get("full", {})
    if not isinstance(full, dict) or "r2" not in full:
        lg.warning("[BO] Ignoring incomplete trial metrics: %s", metrics_path)
        return None

    meta = metrics.get("bayes_trial", {})
    if not isinstance(meta, dict):
        meta = {}

    trial_idx = int_or(meta.get("trial_index"), bayes_trial_index_from_dir(trial_dir))
    if trial_idx <= 0:
        lg.warning("[BO] Could not identify trial index from %s", metrics_path)
        return None

    cfg_payload = (
        read_json_dict(output_dir / "config_revm.json")
        or read_json_dict(output_dir / "config_revh.json")
        or read_json_dict(output_dir / "config_revg.json")
        or read_json_dict(output_dir / "config_reve.json")
        or {}
    )
    params = meta.get("params")
    if isinstance(params, dict):
        trial_params = normalise_bayes_params(params, base_cfg)
    elif isinstance(cfg_payload, dict) and cfg_payload:
        trial_params = normalise_bayes_params(cfg_payload, base_cfg)
    else:
        trial_params = current_search_params(base_cfg, BAYES_SEARCH_SPACE)

    training = metrics.get("training", {})
    if not isinstance(training, dict):
        training = {}
    final_losses = training.get("final_losses", {})
    missing_loss_components = not (isinstance(final_losses, dict) and "train_physics" in final_losses)
    forecast_payload = metrics.get("forecast", {})
    if not isinstance(forecast_payload, dict):
        forecast_payload = {}
    forecast_metrics = forecast_payload.get("metrics", {})
    if not isinstance(forecast_metrics, dict):
        forecast_metrics = {}
    full_ode_payload = metrics.get("full_ode", {})
    if not isinstance(full_ode_payload, dict):
        full_ode_payload = {}
    full_ode_metric_groups = full_ode_payload.get("metrics", {})
    if not isinstance(full_ode_metric_groups, dict):
        full_ode_metric_groups = {}
    full_ode_metrics = full_ode_metric_groups.get("full", {})
    if not isinstance(full_ode_metrics, dict):
        full_ode_metrics = {}

    record: Dict[str, object] = {
        "status": "ok",
        "trial_index": int(trial_idx),
        "fit_r2": finite_or(full.get("r2"), -1.0e9),
        "fit_rms_error_deg": finite_or(full.get("error_rms_deg"), float("inf")),
        "fit_rmse_deg": finite_or(full.get("rmse_deg"), float("inf")),
        "fit_measured_rms_deg": finite_or(full.get("measured_rms_deg"), float("nan")),
        "fit_curve_rms_deg": finite_or(full.get("fit_rms_deg"), float("nan")),
        "forecast_r2": finite_or(forecast_metrics.get("r2"), -1.0e9),
        "forecast_rms_error_deg": finite_or(forecast_metrics.get("error_rms_deg"), finite_or(forecast_metrics.get("rmse_deg"), float("inf"))),
        "forecast_rmse_deg": finite_or(forecast_metrics.get("rmse_deg"), float("inf")),
        "forecast_mae_deg": finite_or(forecast_metrics.get("mae_deg"), float("inf")),
        "forecast_phase_lag_s": finite_or(forecast_metrics.get("phase_lag_s"), 0.0),
        "forecast_phase_corr": finite_or(forecast_metrics.get("phase_corr"), float("nan")),
        "forecast_phase_aligned_rmse_deg": finite_or(forecast_metrics.get("phase_aligned_rmse_deg"), float("inf")),
        "forecast_amplitude_ratio": finite_or(forecast_metrics.get("amplitude_ratio"), float("nan")),
        "forecast_amplitude_underfit_ratio": finite_or(forecast_metrics.get("amplitude_underfit_ratio"), 1.0),
        "forecast_extrema_rmse_deg": finite_or(forecast_metrics.get("extrema_rmse_deg"), float("inf")),
        "forecast_extrema_relative_error": finite_or(forecast_metrics.get("extrema_relative_error"), 1.0),
        "forecast_extrema_range_ratio": finite_or(forecast_metrics.get("extrema_range_ratio"), float("nan")),
        "forecast_extrema_point_rmse_deg": finite_or(forecast_metrics.get("extrema_point_rmse_deg"), float("inf")),
        "forecast_extrema_point_mae_deg": finite_or(forecast_metrics.get("extrema_point_mae_deg"), float("inf")),
        "forecast_peak_underfit_deg": finite_or(forecast_metrics.get("peak_underfit_deg"), float("inf")),
        "forecast_peak_underfit_max_deg": finite_or(forecast_metrics.get("peak_underfit_max_deg"), float("inf")),
        "forecast_trough_overshoot_deg": finite_or(forecast_metrics.get("trough_overshoot_deg"), float("inf")),
        "forecast_trough_overshoot_max_deg": finite_or(forecast_metrics.get("trough_overshoot_max_deg"), float("inf")),
        "forecast_start_time_s": forecast_payload.get("start_time_s"),
        "forecast_end_time_s": forecast_payload.get("end_time_s"),
        "forecast_duration_s": forecast_payload.get("duration_s"),
        "full_ode_r2": finite_or(full_ode_metrics.get("r2"), -1.0e9),
        "full_ode_rms_error_deg": finite_or(full_ode_metrics.get("error_rms_deg"), finite_or(full_ode_metrics.get("rmse_deg"), float("inf"))),
        "full_ode_rmse_deg": finite_or(full_ode_metrics.get("rmse_deg"), float("inf")),
        "full_ode_mae_deg": finite_or(full_ode_metrics.get("mae_deg"), float("inf")),
        "best_epoch": int_or(training.get("best_epoch"), 0),
        "best_metric": finite_or(training.get("best_metric"), float("nan")),
        "final_train_physics_loss": final_training_loss_from_metrics(training, "train_physics", 0.0),
        "final_train_total_force_target_loss": final_training_loss_from_metrics(
            training, "train_total_force_target", 0.0
        ),
        "final_train_total_force_event_loss": final_training_loss_from_metrics(
            training, "train_total_force_event", 0.0
        ),
        "final_train_total_force_band_shape_loss": final_training_loss_from_metrics(
            training, "train_total_force_band_shape", 0.0
        ),
        "final_train_total_force_spectral_shape_loss": final_training_loss_from_metrics(
            training, "train_total_force_spectral_shape", 0.0
        ),
        "final_train_force_envelope_loss": final_training_loss_from_metrics(
            training, "train_force_envelope", 0.0
        ),
        "final_train_kinematic_loss": final_training_loss_from_metrics(training, "train_kinematic", 0.0),
        "final_train_data_loss": final_training_loss_from_metrics(training, "train_data", 0.0),
        "final_train_global_extrema_loss": final_training_loss_from_metrics(training, "train_global_extrema", 0.0),
        "final_train_extrema_window_underfit_loss": final_training_loss_from_metrics(
            training, "train_extrema_window_underfit", 0.0
        ),
        "final_train_high_amplitude_underfit_loss": final_training_loss_from_metrics(
            training, "train_high_amplitude_underfit", 0.0
        ),
        "final_train_high_pass_residual_loss": final_training_loss_from_metrics(training, "train_high_pass_residual", 0.0),
        "final_train_low_pass_residual_loss": final_training_loss_from_metrics(training, "train_low_pass_residual", 0.0),
        "final_train_roll_spectral_shape_loss": final_training_loss_from_metrics(training, "train_roll_spectral_shape", 0.0),
        "final_train_amplitude_underfit_loss": final_training_loss_from_metrics(training, "train_amplitude_underfit", 0.0),
        "final_train_rollout_loss": final_training_loss_from_metrics(training, "train_rollout", 0.0),
        "final_train_rollout_peak_trough_loss": final_training_loss_from_metrics(
            training, "train_rollout_peak_trough", 0.0
        ),
        "final_train_rollout_global_extrema_loss": final_training_loss_from_metrics(
            training, "train_rollout_global_extrema", 0.0
        ),
        "final_train_rollout_extrema_window_underfit_loss": final_training_loss_from_metrics(
            training, "train_rollout_extrema_window_underfit", 0.0
        ),
        "final_train_rollout_high_amplitude_underfit_loss": final_training_loss_from_metrics(
            training, "train_rollout_high_amplitude_underfit", 0.0
        ),
        "final_train_rollout_local_prominence_loss": final_training_loss_from_metrics(training, "train_rollout_local_prominence", 0.0),
        "final_train_rollout_high_pass_residual_loss": final_training_loss_from_metrics(training, "train_rollout_high_pass_residual", 0.0),
        "final_train_rollout_roll_spectral_shape_loss": final_training_loss_from_metrics(training, "train_rollout_roll_spectral_shape", 0.0),
        "final_train_rollout_phase_loss": final_training_loss_from_metrics(training, "train_rollout_phase", 0.0),
        "bayes_r2_objective_weight": float(base_cfg.get("bayes_r2_objective_weight", 10.0)),
        "bayes_rmse_objective_weight": float(base_cfg.get("bayes_rmse_objective_weight", 0.0)),
        "bayes_fit_r2_objective_weight": float(base_cfg.get("bayes_fit_r2_objective_weight", 10.0)),
        "bayes_fit_rmse_objective_weight": float(base_cfg.get("bayes_fit_rmse_objective_weight", 0.0)),
        "bayes_phase_objective_weight": float(base_cfg.get("bayes_phase_objective_weight", 0.0)),
        "bayes_amplitude_objective_weight": float(base_cfg.get("bayes_amplitude_objective_weight", 0.0)),
        "bayes_extrema_objective_weight": float(base_cfg.get("bayes_extrema_objective_weight", 0.0)),
        "bayes_forecast_extrema_rmse_objective_weight": float(
            base_cfg.get("bayes_forecast_extrema_rmse_objective_weight", 0.0)
        ),
        "bayes_forecast_peak_underfit_objective_weight": float(
            base_cfg.get("bayes_forecast_peak_underfit_objective_weight", 0.0)
        ),
        "bayes_forecast_trough_overshoot_objective_weight": float(
            base_cfg.get("bayes_forecast_trough_overshoot_objective_weight", 0.0)
        ),
        "bayes_turning_objective_weight": float(base_cfg.get("bayes_turning_objective_weight", 0.0)),
        "bayes_physics_objective_weight": float(base_cfg.get("bayes_physics_objective_weight", 0.0)),
        "bayes_kinematic_objective_weight": float(base_cfg.get("bayes_kinematic_objective_weight", 0.0)),
        "bo_objective": 0.0,
        "params": copy.deepcopy(trial_params),
        "trial_epochs": int_or(meta.get("epochs"), int_or(cfg_payload.get("epochs"), int(base_cfg.get("bayes_trial_epochs", 150)))),
        "output_dir": str(output_dir),
        "paths": existing_bayes_output_paths(output_dir),
        "selection_method": "resumed_from_trial_metrics",
        "resume_missing_loss_components": bool(missing_loss_components),
    }
    record["bo_objective"] = bayes_objective_from_record(record)
    return record


def load_bayes_resume_records(bayes_dir: Path, summary_path: Path, base_cfg: Dict[str, object],
                              lg: logging.Logger) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    by_idx: Dict[int, Dict[str, object]] = {}
    previous_summary = (
        read_json_dict(summary_path)
        or read_json_dict(summary_path.with_name("bayes_search_results_revh.json"))
        or read_json_dict(summary_path.with_name("bayes_search_results_revg.json"))
        or read_json_dict(summary_path.with_name("bayes_search_results_reve.json"))
        or {}
    )
    max_trials = int(base_cfg.get("bayes_trials", 100000))
    current_version = str(base_cfg.get("bayes_study_version", "revm_unversioned"))
    previous_version = str(previous_summary.get("study_version", "revm_legacy")) if previous_summary else current_version
    configured_versions = base_cfg.get("bayes_compatible_study_versions", [current_version])
    compatible_versions = (
        {str(version) for version in configured_versions}
        if isinstance(configured_versions, (list, tuple, set))
        else {current_version}
    )
    compatible_versions.add(current_version)
    if previous_summary and previous_version not in compatible_versions:
        raise ValueError(
            f"Bayesian output directory contains incompatible study {previous_version!r}; "
            f"current study is {current_version!r}. Use a new --bayes-output-dir."
        )
    if previous_summary and previous_version != current_version:
        lg.info(
            "[BO] Continuing compatible study %r as Rev M study %r.",
            previous_version,
            current_version,
        )

    raw_trials = previous_summary.get("trials", []) if isinstance(previous_summary, dict) else []
    if isinstance(raw_trials, list):
        for raw in raw_trials:
            if not isinstance(raw, dict):
                continue
            trial_idx = int_or(raw.get("trial_index"), 0)
            if trial_idx <= 0 or trial_idx > max_trials:
                continue
            record = copy.deepcopy(raw)
            # Re-score resumed trials with the current forecast-first objective.
            record["bayes_r2_objective_weight"] = float(base_cfg.get("bayes_r2_objective_weight", 10.0))
            record["bayes_rmse_objective_weight"] = float(base_cfg.get("bayes_rmse_objective_weight", 0.0))
            record["bayes_fit_r2_objective_weight"] = float(base_cfg.get("bayes_fit_r2_objective_weight", 10.0))
            record["bayes_fit_rmse_objective_weight"] = float(base_cfg.get("bayes_fit_rmse_objective_weight", 0.0))
            record["bayes_phase_objective_weight"] = float(base_cfg.get("bayes_phase_objective_weight", 0.0))
            record["bayes_amplitude_objective_weight"] = float(base_cfg.get("bayes_amplitude_objective_weight", 0.0))
            record["bayes_turning_objective_weight"] = float(base_cfg.get("bayes_turning_objective_weight", 0.0))
            record["bayes_physics_objective_weight"] = float(base_cfg.get("bayes_physics_objective_weight", 0.0))
            record["bayes_kinematic_objective_weight"] = float(base_cfg.get("bayes_kinematic_objective_weight", 0.0))
            if record.get("status") == "ok":
                params = record.get("params")
                if isinstance(params, dict):
                    record["params"] = normalise_bayes_params(params, base_cfg)
                record["bo_objective"] = bayes_objective_from_record(record)
            by_idx[trial_idx] = record

    for trial_dir in sorted(bayes_dir.glob("trial_*")):
        trial_idx = bayes_trial_index_from_dir(trial_dir)
        if trial_idx <= 0 or trial_idx > max_trials:
            continue
        reconstructed = record_from_bayes_trial_metrics(trial_dir, base_cfg, lg)
        if reconstructed is None:
            continue
        existing = by_idx.get(trial_idx)
        if existing is None or existing.get("status") != "ok":
            by_idx[trial_idx] = reconstructed
        else:
            existing.setdefault("paths", reconstructed.get("paths", {}))
            existing.setdefault("output_dir", reconstructed.get("output_dir"))
            for key in (
                "forecast_r2", "forecast_rms_error_deg", "forecast_rmse_deg", "forecast_mae_deg",
                "forecast_phase_lag_s", "forecast_phase_corr", "forecast_phase_aligned_rmse_deg",
                "forecast_amplitude_ratio", "forecast_amplitude_underfit_ratio",
                "forecast_extrema_rmse_deg", "forecast_extrema_relative_error", "forecast_extrema_range_ratio",
                "forecast_extrema_point_rmse_deg", "forecast_extrema_point_mae_deg",
                "forecast_peak_underfit_deg", "forecast_peak_underfit_max_deg",
                "forecast_trough_overshoot_deg", "forecast_trough_overshoot_max_deg",
                "forecast_start_time_s", "forecast_end_time_s", "forecast_duration_s",
                "full_ode_r2", "full_ode_rms_error_deg", "full_ode_rmse_deg", "full_ode_mae_deg",
            ):
                if key in reconstructed and math.isfinite(finite_or(reconstructed.get(key), float("nan"))):
                    existing[key] = reconstructed[key]
            if (
                "bo_objective" not in existing
                or not math.isfinite(finite_or(existing.get("bo_objective"), float("nan")))
                or math.isfinite(finite_or(existing.get("forecast_r2"), float("nan")))
            ):
                existing["bo_objective"] = bayes_objective_from_record(existing)

    return [by_idx[idx] for idx in sorted(by_idx)], previous_summary


def run_bayesian_optimisation(t: np.ndarray, phi: np.ndarray, wave_pack: Dict[str, np.ndarray],
                              base_cfg: Dict[str, object], lg: logging.Logger,
                              retrain_best: bool = True) -> Dict[str, object]:
    bayes_dir = Path(str(base_cfg["bayes_opt_dir"]))
    bayes_dir.mkdir(parents=True, exist_ok=True)
    summary_path = bayes_dir / "bayes_search_results_revm.json"
    n_trials = int(base_cfg.get("bayes_trials", 100000))
    init_points = int(max(1, base_cfg.get("bayes_init_points", 8)))
    trial_epochs = int(base_cfg.get("bayes_trial_epochs", 150))
    candidate_pool = int(base_cfg.get("bayes_candidate_pool", 512))
    jitter = float(base_cfg.get("bayes_ei_jitter", 0.01))
    target_r2 = float(base_cfg.get("bayes_target_r2", 0.95))
    target_rms = float(base_cfg.get("bayes_target_rms_error_deg", 0.05))
    target_fit_r2 = float(base_cfg.get("bayes_target_fit_r2", 0.70))
    target_fit_rms = float(base_cfg.get("bayes_target_fit_rms_error_deg", 1.25))
    target_full_ode_r2 = float(base_cfg.get("bayes_target_full_ode_r2", 0.95))
    target_full_ode_rms = float(base_cfg.get("bayes_target_full_ode_rms_error_deg", 0.15))
    stop_on_target = bool(base_cfg.get("bayes_stop_on_target", True))
    resume_enabled = bool(base_cfg.get("bayes_resume", True))
    study_version = str(base_cfg.get("bayes_study_version", "revm_unversioned"))
    data_cache = BayesDataCache(t, phi, wave_pack) if bool(base_cfg.get("bayes_reuse_prepared_data", True)) else None

    lg.info("Starting Rev M Bayesian hyperparameter optimisation")
    lg.info(
        "  Objective: jointly maximise hidden-window forecast and full-fit R2 while minimising both RMS errors; "
        "physics/kinematic residuals remain light penalties | trial_epochs=%d | "
        "targets: (forecast R2>%.4f RMS<%.6f deg and fit R2>%.4f RMS<%.6f deg) "
        "OR full ODE R2>%.4f OR full ODE RMS<%.6f deg",
        trial_epochs, target_r2, target_rms, target_fit_r2, target_fit_rms,
        target_full_ode_r2, target_full_ode_rms,
    )
    lg.info(
        "[BO] Search ready | dimensions=%d | progress interval=%d epochs",
        len(BAYES_SEARCH_SPACE),
        max(1, int(base_cfg.get("bayes_log_every", 50))),
    )
    training_start_params = current_search_params(base_cfg, BAYES_SEARCH_SPACE)
    training_start_weights = {
        key: training_start_params[key]
        for key in REVM_TRAINING_WEIGHT_KEYS
        if key in training_start_params
    }
    source_trial_label = f"source_trial_{int(base_cfg.get('source_trial_index', 0)):05d}"
    lg.info(
        "[BO] Trial 1 seed: %s from %s",
        source_trial_label,
        str(base_cfg.get("source_metrics_file", "embedded configuration")),
    )
    lg.info("[BO] First-trial loss weightings from Training component: %s", training_start_weights)
    if data_cache is not None:
        lg.info("[BO] Prepared-data RAM cache enabled; loader tensor cache=%s prepared_cache_max_entries=%d tensor_cache_max_entries=%d window_cache_max_entries=%d",
                bool(base_cfg.get("bayes_reuse_loader_cache", True)),
                int(base_cfg.get("bayes_prepared_data_cache_max_entries", 1)),
                int(base_cfg.get("bayes_tensor_cache_max_entries", 1)),
                int(base_cfg.get("bayes_window_cache_max_entries", 2)))

    observed_params: List[Dict[str, object]] = []
    observed_vals: List[float] = []
    records: List[Dict[str, object]] = []
    best_record: Optional[Dict[str, object]] = None
    previous_summary: Dict[str, object] = {}

    if resume_enabled:
        records, previous_summary = load_bayes_resume_records(bayes_dir, summary_path, base_cfg, lg)
        for record in records:
            if record.get("status") != "ok":
                continue
            params = record.get("params")
            if not isinstance(params, dict):
                continue
            objective = finite_or(record.get("bo_objective"), float("nan"))
            if math.isfinite(objective):
                observed_params.append(copy.deepcopy(params))
                observed_vals.append(float(objective))
            if best_record is None or bayes_fit_rank_key(record) > bayes_fit_rank_key(best_record):
                best_record = copy.deepcopy(record)
        if records:
            lg.info("[BO] Resume loaded %d previous trial records (%d usable for acquisition).",
                    len(records), len(observed_params))
    else:
        lg.info("[BO] Resume disabled; existing Bayes summaries and trial folders will not seed this search.")

    start_trial_idx = 1
    if records:
        start_trial_idx = max(int_or(record.get("trial_index"), 0) for record in records) + 1
    rng = np.random.default_rng(int(base_cfg["seed"]) + max(0, start_trial_idx - 1))

    summary: Dict[str, object] = {
        "study_version": study_version,
        "baseline_run_id": base_cfg.get("baseline_run_id"),
        "baseline_run_metrics": copy.deepcopy(base_cfg.get("baseline_run_metrics", {})),
        "source_metrics_file": base_cfg.get("source_metrics_file"),
        "source_trial_index": int(base_cfg.get("source_trial_index", 0)),
        "source_active_parameters": copy.deepcopy(training_start_params),
        "objective": "jointly_optimize_hidden_forecast_and_full_teacher_forced_fit_with_light_auxiliary_penalties",
        "objective_metric_source": "forecast_window_and_full_fit",
        "forecast_window_s": (
            float(base_cfg["forecast_window_s"])
            if base_cfg.get("forecast_window_s") is not None else None
        ),
        "forecast_fraction": (
            float(base_cfg.get("forecast_frac", 0.10))
            if base_cfg.get("forecast_window_s") is None else None
        ),
        "initial_training_weightings": copy.deepcopy(training_start_weights),
        "r2_objective_weight": float(base_cfg.get("bayes_r2_objective_weight", 10.0)),
        "rmse_objective_weight": float(base_cfg.get("bayes_rmse_objective_weight", 0.0)),
        "fit_r2_objective_weight": float(base_cfg.get("bayes_fit_r2_objective_weight", 10.0)),
        "fit_rmse_objective_weight": float(base_cfg.get("bayes_fit_rmse_objective_weight", 0.0)),
        "phase_objective_weight": float(base_cfg.get("bayes_phase_objective_weight", 0.0)),
        "amplitude_objective_weight": float(base_cfg.get("bayes_amplitude_objective_weight", 0.0)),
        "turning_objective_weight": float(base_cfg.get("bayes_turning_objective_weight", 0.0)),
        "forecast_extrema_rmse_objective_weight": float(
            base_cfg.get("bayes_forecast_extrema_rmse_objective_weight", 0.0)
        ),
        "forecast_peak_underfit_objective_weight": float(
            base_cfg.get("bayes_forecast_peak_underfit_objective_weight", 0.0)
        ),
        "forecast_trough_overshoot_objective_weight": float(
            base_cfg.get("bayes_forecast_trough_overshoot_objective_weight", 0.0)
        ),
        "physics_objective_weight": float(base_cfg.get("bayes_physics_objective_weight", 0.0)),
        "kinematic_objective_weight": float(base_cfg.get("bayes_kinematic_objective_weight", 0.0)),
        "target_r2": target_r2,
        "target_rms_error_deg": target_rms,
        "target_fit_r2": target_fit_r2,
        "target_fit_rms_error_deg": target_fit_rms,
        "target_full_ode_r2": target_full_ode_r2,
        "target_full_ode_rms_error_deg": target_full_ode_rms,
        "target_reached": False,
        "trial_epochs": trial_epochs,
        "max_trials": n_trials,
        "resume_enabled": resume_enabled,
        "resumed_trials": len(records) if resume_enabled else 0,
        "interrupted": False,
        "search_space": copy.deepcopy(BAYES_SEARCH_SPACE),
        "trials": records,
        "best_trial": None,
    }
    if isinstance(previous_summary.get("best_retrain"), dict):
        summary["best_retrain"] = copy.deepcopy(previous_summary["best_retrain"])
    summary["completed_trials"] = len(records)
    summary["best_trial"] = copy.deepcopy(best_record)
    update_best_forecast_metric_records(summary, records)
    stop_record, stop_reason = update_bayes_stop_status(
        summary, records,
        target_r2, target_rms, target_fit_r2, target_fit_rms,
        target_full_ode_r2, target_full_ode_rms,
    )
    write_bayes_summary(summary_path, summary)

    if stop_on_target and summary["target_reached"]:
        lg.info(
            "[BO] Existing Bayes target already satisfied by trial %05d (%s); no new trials required.",
            int_or(stop_record.get("trial_index"), 0) if stop_record is not None else 0,
            stop_reason or "target",
        )
    elif start_trial_idx > n_trials:
        lg.info("[BO] Existing records already cover max_trials=%d; no new trials required.", n_trials)
    else:
        for trial_idx in range(start_trial_idx, n_trials + 1):
            trial_dir = bayes_trial_dir(bayes_dir, trial_idx)
            state = read_bayes_trial_state(trial_dir) if resume_enabled else None
            output_metrics = trial_dir / "outputs" / "metrics_revm.json"
            legacy_revh_output_metrics = trial_dir / "outputs" / "metrics_revh.json"
            legacy_revg_output_metrics = trial_dir / "outputs" / "metrics_revg.json"
            legacy_output_metrics = trial_dir / "outputs" / "metrics_reve.json"
            if (
                resume_enabled
                and isinstance(state, dict)
                and not output_metrics.exists()
                and not legacy_revh_output_metrics.exists()
                and not legacy_revg_output_metrics.exists()
                and not legacy_output_metrics.exists()
                and isinstance(state.get("params"), dict)
            ):
                params = normalise_bayes_params(state["params"], base_cfg)
                previous_method = str(state.get("selection_method", "pending"))
                method = previous_method if previous_method.startswith("resume_") else f"resume_{previous_method}"
                lg.info("[BO] Resuming pending trial %05d with stored parameters; restarting training from epoch 0.", trial_idx)
            elif trial_idx == 1:
                params = copy.deepcopy(training_start_params)
                method = source_trial_label
            elif len(observed_params) < init_points:
                params = sample_search_params(rng, BAYES_SEARCH_SPACE)
                method = "random_initial"
            else:
                params, method = propose_bayes_candidate(
                    rng, BAYES_SEARCH_SPACE, observed_params, observed_vals,
                    candidate_pool=candidate_pool, jitter=jitter,
                )

            write_bayes_trial_state(trial_dir, {
                "status": "running",
                "trial_index": int(trial_idx),
                "selection_method": method,
                "params": copy.deepcopy(params),
                "trial_epochs": int(trial_epochs),
                "started_or_resumed_at": datetime.now().isoformat(timespec="seconds"),
            })

            best_composite_label = bayes_run_label(best_record)
            best_r2_label = bayes_run_label(summary.get("best_forecast_r2_trial"))
            best_rmse_label = bayes_run_label(summary.get("best_forecast_rmse_trial"))
            lg.info(
                "[BO] Run %05d/%05d | status=starting | method=%s | best composite=%s R2=%s RMSE=%s",
                trial_idx,
                n_trials,
                method,
                best_composite_label,
                best_r2_label,
                best_rmse_label,
            )
            progress = {
                "bayes_total_trials": int(n_trials),
                "bayes_best_composite_run": best_composite_label,
                "bayes_best_r2_run": best_r2_label,
                "bayes_best_rmse_run": best_rmse_label,
            }
            start = time.time()
            try:
                record = run_bayes_training_trial(
                    t, phi, wave_pack, base_cfg, params, trial_idx, lg,
                    data_cache=data_cache,
                    progress=progress,
                )
            except KeyboardInterrupt:
                duration = float(time.time() - start)
                write_bayes_trial_state(trial_dir, {
                    "status": "interrupted",
                    "interrupted_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": duration,
                })
                summary["completed_trials"] = len(records)
                summary["best_trial"] = copy.deepcopy(best_record)
                update_best_forecast_metric_records(summary, records)
                update_bayes_stop_status(
                    summary, records,
                    target_r2, target_rms, target_fit_r2, target_fit_rms,
                    target_full_ode_r2, target_full_ode_rms,
                )
                summary["interrupted"] = True
                summary["interrupted_at_trial"] = int(trial_idx)
                summary["message"] = (
                    "Interrupted during a Bayes trial. Re-run with the same --bayes-output-dir "
                    "to restart the same trial from its saved parameter state."
                )
                write_bayes_summary(summary_path, summary)
                lg.warning(
                    "[BO] Run %05d/%05d | status=interrupted | summary saved; restart with the same Bayes output directory to rerun from epoch 0.",
                    trial_idx,
                    n_trials,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if data_cache is not None:
                    data_cache.release_between_trials(base_cfg, lg)
                else:
                    gc.collect()
                return summary
            except Exception as exc:
                record = {
                    "status": "failed",
                    "trial_index": int(trial_idx),
                    "fit_r2": -1.0e9,
                    "fit_rms_error_deg": float("inf"),
                    "forecast_r2": -1.0e9,
                    "forecast_rms_error_deg": float("inf"),
                    "bo_objective": 1.0e9,
                    "params": copy.deepcopy(params),
                    "trial_epochs": int(trial_epochs),
                    "error": str(exc),
                }
                lg.exception("[BO] Run %05d/%05d | status=failed", trial_idx, n_trials)
            if data_cache is not None:
                data_cache.release_between_trials(base_cfg, lg)
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            record["selection_method"] = method
            record["duration_seconds"] = float(time.time() - start)
            records.append(record)

            write_bayes_trial_state(trial_dir, {
                "status": str(record.get("status", "unknown")),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "duration_seconds": float(record["duration_seconds"]),
                "fit_r2": record.get("fit_r2"),
                "fit_rms_error_deg": record.get("fit_rms_error_deg"),
                "forecast_r2": record.get("forecast_r2"),
                "forecast_rms_error_deg": record.get("forecast_rms_error_deg"),
                "forecast_phase_lag_s": record.get("forecast_phase_lag_s"),
                "forecast_amplitude_ratio": record.get("forecast_amplitude_ratio"),
                "forecast_amplitude_underfit_ratio": record.get("forecast_amplitude_underfit_ratio"),
                "full_ode_r2": record.get("full_ode_r2"),
                "full_ode_rms_error_deg": record.get("full_ode_rms_error_deg"),
                "error": record.get("error"),
            })

            if record.get("status") == "ok" and math.isfinite(float(record["bo_objective"])):
                record_params = record.get("params")
                observed_params.append(copy.deepcopy(record_params if isinstance(record_params, dict) else params))
                observed_vals.append(float(record["bo_objective"]))
                if best_record is None or bayes_fit_rank_key(record) > bayes_fit_rank_key(best_record):
                    best_record = copy.deepcopy(record)

            update_best_forecast_metric_records(summary, records)
            if record.get("status") == "ok":
                lg.info(
                    "[BO] Run %05d/%05d | status=complete | forecast R2=%.6f | forecast RMS error=%.6f deg | full ODE R2=%.6f | full ODE RMS error=%.6f deg | phase lag=%.5f s | amp ratio=%.5f | fit R2=%.6f | fit RMS error=%.6f deg | best_epoch=%s | best composite=%s R2=%s RMSE=%s",
                    trial_idx,
                    n_trials,
                    finite_or(record.get("forecast_r2"), float("nan")),
                    finite_or(record.get("forecast_rms_error_deg"), float("nan")),
                    finite_or(record.get("full_ode_r2"), float("nan")),
                    finite_or(record.get("full_ode_rms_error_deg"), float("nan")),
                    finite_or(record.get("forecast_phase_lag_s"), float("nan")),
                    finite_or(record.get("forecast_amplitude_ratio"), float("nan")),
                    finite_or(record.get("fit_r2"), float("nan")),
                    finite_or(record.get("fit_rms_error_deg"), float("nan")),
                    record.get("best_epoch"),
                    bayes_run_label(best_record),
                    bayes_run_label(summary.get("best_forecast_r2_trial")),
                    bayes_run_label(summary.get("best_forecast_rmse_trial")),
                )

            summary["completed_trials"] = len(records)
            summary["best_trial"] = copy.deepcopy(best_record)
            update_best_forecast_metric_records(summary, records)
            update_bayes_stop_status(
                summary, records,
                target_r2, target_rms, target_fit_r2, target_fit_rms,
                target_full_ode_r2, target_full_ode_rms,
            )
            summary["interrupted"] = False
            write_bayes_summary(summary_path, summary)

            record_stop_reason = (
                bayes_stop_reason(
                    record,
                    target_r2,
                    target_rms,
                    target_fit_r2,
                    target_fit_rms,
                    target_full_ode_r2,
                    target_full_ode_rms,
                )
                if record.get("status") == "ok"
                else None
            )
            if stop_on_target and record_stop_reason is not None:
                lg.info(
                    "[BO] Stop target reached by trial %05d (%s): forecast R2=%.6f RMS=%.6f deg; "
                    "fit R2=%.6f RMS=%.6f deg; full ODE R2=%.6f RMS=%.6f deg. Stopping search.",
                    trial_idx,
                    record_stop_reason,
                    finite_or(record.get("forecast_r2"), float("nan")),
                    finite_or(record.get("forecast_rms_error_deg"), float("nan")),
                    finite_or(record.get("fit_r2"), float("nan")),
                    finite_or(record.get("fit_rms_error_deg"), float("nan")),
                    finite_or(record.get("full_ode_r2"), float("nan")),
                    finite_or(record.get("full_ode_rms_error_deg"), float("nan")),
                )
                break

    if retrain_best and best_record is not None:
        existing_retrain = summary.get("best_retrain")
        existing_retrain_done = False
        if resume_enabled and isinstance(existing_retrain, dict):
            output_dir = Path(str(existing_retrain.get("output_dir", "")))
            existing_retrain_done = (
                (output_dir / "metrics_revm.json").exists()
                or (output_dir / "metrics_revh.json").exists()
                or (output_dir / "metrics_revg.json").exists()
                or (output_dir / "metrics_reve.json").exists()
            )

        if existing_retrain_done:
            lg.info("[BO] Existing best retrain outputs found; keeping: %s", existing_retrain.get("output_dir"))
        else:
            lg.info("[BO] Retraining best Rev M forecast-focused hyperparameter set for %d epochs", trial_epochs)
            final_cfg = copy.deepcopy(base_cfg)
            final_cfg.update(copy.deepcopy(best_record["params"]))
            final_cfg["epochs"] = trial_epochs
            final_cfg["checkpoint_dir"] = str(bayes_dir / "best_run_checkpoints")
            final_cfg["output_dir"] = str(bayes_dir / "best_run_outputs")
            # The final Bayes retrain is checkpoint-free as well. Standard
            # non-Bayes training still uses CONFIG's final-only policy.
            final_cfg["save_checkpoints"] = False
            final_cfg["checkpoint_policy"] = "disabled"
            final_cfg["save_best_checkpoint"] = False
            final_cfg["save_periodic_checkpoints"] = False
            final_cfg["save_last_checkpoint"] = False
            final_cfg["save_final_checkpoint"] = False
            final_cfg["final_checkpoint_name"] = "final.pt"
            final_cfg["checkpoint_every"] = 0
            final_cfg = clamp_bayes_trial_cfg(final_cfg, t)
            set_global_seed(int(final_cfg["seed"]))
            if data_cache is not None and bool(final_cfg.get("bayes_reuse_prepared_data", True)):
                data = data_cache.get_prepared(final_cfg, lg)
                shared_cache = data_cache.loader_shared_cache(final_cfg)
            else:
                data = prepare_data(t, phi, wave_pack, final_cfg, lg)
                shared_cache = None
            train_loader, val_loader = build_loaders(data, final_cfg, lg, shared_cache=shared_cache)
            model = build_model(final_cfg, data)
            log_model_summary(model, lg)

            try:
                train_result = train_model(
                    model, train_loader, val_loader, data, final_cfg, lg,
                    resume=None,
                )
            except KeyboardInterrupt:
                summary["best_retrain_interrupted"] = True
                summary["interrupted"] = True
                summary["message"] = (
                    "Interrupted during the final Bayes best retrain. Re-run with the same "
                    "--bayes-output-dir to restart the final retrain from the beginning."
                )
                write_bayes_summary(summary_path, summary)
                lg.warning("[BO] Interrupted during best retrain. Summary saved; restart with the same Bayes output directory to rerun the final retrain from epoch 0.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if data_cache is not None:
                    data_cache.release_between_trials(final_cfg, lg)
                else:
                    gc.collect()
                return summary

            if not train_result.get("history"):
                train_metrics = {f"train_{k}": v for k, v in evaluate_loader(model, train_loader, final_cfg).items()}
                val_metrics = {f"val_{k}": v for k, v in evaluate_loader(model, val_loader, final_cfg).items()} if val_loader is not None else {}
                train_result["history"] = [{
                    "epoch": float(final_cfg["epochs"]),
                    **train_metrics,
                    **val_metrics,
                }]
            paths = write_outputs(
                model,
                data,
                final_cfg,
                train_result,
                lg,
                include_full_ode=bool(final_cfg.get("bayes_full_ode_outputs", True)),
                full_ode_coefficient_source="model",
            )
            best_checkpoint = train_result.get("best_checkpoint")
            final_checkpoint = train_result.get("final_checkpoint")
            if best_checkpoint:
                paths["best_checkpoint"] = Path(str(best_checkpoint))
            if final_checkpoint and final_checkpoint != best_checkpoint:
                paths["final_checkpoint"] = Path(str(final_checkpoint))
            summary["best_retrain"] = {
                "output_dir": final_cfg["output_dir"],
                "checkpoint_dir": final_cfg["checkpoint_dir"],
                "final_checkpoint": train_result.get("final_checkpoint"),
                "paths": {k: str(v) for k, v in paths.items()},
            }
            summary["best_retrain_interrupted"] = False
            summary["interrupted"] = False
            write_bayes_summary(summary_path, summary)
            del model, train_loader, val_loader, data, train_result
            if data_cache is not None:
                data_cache.release_between_trials(final_cfg, lg)
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    lg.info("Bayesian optimisation summary written to: %s", summary_path)
    return summary


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestPINNLSTMRevM(unittest.TestCase):
    def test_column_matching(self):
        cols = ["Wave_Time", "SS5-seed1", "Ship_Time", "Roll", "Speed", "Yaw rate", "X", "Y"]
        self.assertEqual(match_column(cols, ["wave_time"]), "Wave_Time")
        self.assertEqual(match_column(cols, ["ship_time"]), "Ship_Time")
        self.assertEqual(match_column(cols, ["yawrate", "yaw rate"]), "Yaw rate")

    def test_combined_active_layout_uses_independent_ship_and_wave_clocks(self):
        df = pd.DataFrame({
            "Ship_Time": ["[sec]", 0.0, 0.04, 0.08, None],
            "Rudder Feedback": ["[deg]", -0.1, 0.2, 0.3, None],
            "RPS Feedback": ["[/sec]", 17.4, 17.4, 17.4, None],
            "X": ["[m]", 0.0, 0.04, 0.08, None],
            "Y": ["[m]", 0.0, 0.01, 0.02, None],
            "Roll": ["[deg]", 0.0, 30.0, -45.0, None],
            "Pitch": ["[deg]", 1.0, 2.0, 3.0, None],
            "Yaw": ["[deg]", 4.0, 5.0, 6.0, None],
            "Speed": ["[m/sec]", 0.8, 0.81, 0.82, None],
            "Yaw rate": ["[deg/sec]", -0.1, -0.2, -0.3, None],
            "Unnamed: 10": [None, None, None, None, None],
            "Wave_Time": ["[sec]", 0.0, 0.02, 0.04, 0.06],
            "SS5-seed1": ["[m]", 0.10, 0.20, 0.30, 0.40],
        })
        lg = logging.getLogger("unit_test_excel_layout_revm")
        lg.addHandler(logging.NullHandler())
        with mock.patch.object(pd, "read_excel", return_value=df):
            t, phi, pack = load_excel_sheet(Path("synthetic.xlsx"), object(), "Active", lg)

        self.assertTrue(np.allclose(t, [0.0, 0.04, 0.08]))
        self.assertTrue(np.allclose(phi, np.deg2rad([0.0, 30.0, -45.0])))
        self.assertTrue(np.allclose(pack["wave_time_raw"], [0.0, 0.02, 0.04, 0.06]))
        self.assertTrue(np.allclose(pack["wave_signal_raw"], [0.10, 0.20, 0.30, 0.40]))
        self.assertTrue(np.allclose(pack["wave_cross_beam_raw"], 0.0))
        self.assertNotIn("wave_2_raw", pack)
        self.assertTrue(np.allclose(pack["x_position_raw"], [0.0, 0.04, 0.08]))
        self.assertTrue(np.allclose(pack["y_position_raw"], [0.0, 0.01, 0.02]))
        self.assertTrue(np.allclose(pack["speed_raw"], [0.8, 0.81, 0.82]))
        self.assertTrue(np.allclose(pack["yawrate_raw"], np.deg2rad([-0.1, -0.2, -0.3])))
        self.assertTrue(np.allclose(pack["rudder_feedback_raw"], np.deg2rad([-0.1, 0.2, 0.3])))
        self.assertTrue(np.allclose(pack["rps_feedback_raw"], [17.4, 17.4, 17.4]))
        self.assertTrue(np.allclose(pack["pitch_angle_raw"], np.deg2rad([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(pack["yaw_angle_raw"], np.deg2rad([4.0, 5.0, 6.0])))

    def test_excel_positional_layout_uses_two_probes_positions_and_degree_roll(self):
        df = pd.DataFrame({
            "Time": [0.0, 0.1, 0.2],
            "Wave Probe 1": [0.10, 0.20, 0.30],
            "Wave Probe 2": [0.30, 0.40, 0.50],
            "X": [1.0, 1.1, 1.2],
            "Y": [2.0, 2.1, 2.2],
            "Unused 6": [0.0, 0.0, 0.0],
            "Unused 7": [0.0, 0.0, 0.0],
            "Unused 8": [0.0, 0.0, 0.0],
            "Unused 9": [0.0, 0.0, 0.0],
            "Unused 10": [0.0, 0.0, 0.0],
            "Unused 11": [0.0, 0.0, 0.0],
            "Unused 12": [0.0, 0.0, 0.0],
            "Roll": [0.0, 30.0, -45.0],
        })
        lg = logging.getLogger("unit_test_excel_layout_revm")
        lg.addHandler(logging.NullHandler())
        with mock.patch.object(pd, "read_excel", return_value=df):
            t, phi, pack = load_excel_sheet(Path("synthetic.xlsx"), object(), "Sheet1", lg)

        self.assertTrue(np.allclose(t, [0.0, 0.1, 0.2]))
        self.assertTrue(np.allclose(phi, np.deg2rad([0.0, 30.0, -45.0])))
        self.assertTrue(np.allclose(pack["wave_1_raw"], [0.10, 0.20, 0.30]))
        self.assertTrue(np.allclose(pack["wave_2_raw"], [0.30, 0.40, 0.50]))
        self.assertTrue(np.allclose(pack["wave_signal_raw"], [0.20, 0.30, 0.40]))
        self.assertTrue(np.allclose(pack["wave_cross_beam_raw"], [0.10, 0.10, 0.10]))
        self.assertTrue(np.allclose(pack["x_position_raw"], [1.0, 1.1, 1.2]))
        self.assertTrue(np.allclose(pack["y_position_raw"], [2.0, 2.1, 2.2]))

    def test_single_sheet_run_is_rebased_to_zero_without_truncation(self):
        lg = logging.getLogger("unit_test_single_run_zero_origin_revm")
        lg.addHandler(logging.NullHandler())
        t = np.asarray([50.0, 50.1, 50.2], dtype=float)
        phi = np.asarray([0.0, 0.1, 0.2], dtype=float)
        pack = {
            "wave_time_raw": np.asarray([50.0, 50.05, 50.1, 50.15], dtype=float),
            "wave_signal_raw": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
        }
        t_out, phi_out, pack_out = combine_excel_runs([("Active", t, phi, pack)], CONFIG, lg)

        self.assertTrue(np.allclose(t_out, [0.0, 0.1, 0.2]))
        self.assertTrue(np.allclose(phi_out, phi))
        self.assertTrue(np.allclose(pack_out["wave_time_raw"], [0.0, 0.05, 0.1, 0.15]))
        self.assertEqual(len(t_out), len(t))
        self.assertEqual(len(pack_out["wave_time_raw"]), len(pack["wave_time_raw"]))


    def test_model_forward_shapes_and_no_embedding(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "lstm_hidden_size": 32,
            "lstm_layers": 1,
            "fc_hidden": 32,
            "wave_shape_correction_enabled": True,
            "wave_shape_correction_gain": 0.35,
            "motion_feedback_backbone_gain": 0.0,
        })
        wave_idx = {"wave_signal": [3], "wave_slope": [4], "wave_abs_slope": [], "wave_envelope": [], "wave_envelope_slow": []}
        state_idx = {"speed": [5], "yawrate": [6], "x_position": [], "y_position": [], "encounter_frequency_ratio": [], "encounter_time_shift": []}
        motion_idx = {"phi_fb": [7], "v_fb": [8]}
        model = PINNLSTM(cfg, 9, wave_idx, state_idx, motion_idx)
        self.assertEqual(sum(1 for m in model.modules() if isinstance(m, nn.Embedding)), 0)
        x = torch.randn(4, 12, 9)
        phi, v, f, fr = model(x, x)
        self.assertEqual(tuple(phi.shape), (4, 12, 1))
        self.assertEqual(tuple(v.shape), (4, 12, 1))
        self.assertEqual(tuple(f.shape), (4, 12, 1))
        self.assertEqual(tuple(fr.shape), (4, 12, 1))
        self.assertFalse(model.c_roll_param.requires_grad)
        self.assertFalse(model.c_quad_param.requires_grad)
        self.assertFalse(model.k_roll_param.requires_grad)
        self.assertAlmostEqual(float(model.c_roll.detach().cpu().item()), float(cfg["c_roll_init"]), places=6)
        self.assertAlmostEqual(float(model.c_quad.detach().cpu().item()), float(cfg["c_quad_init"]), places=8)
        self.assertAlmostEqual(float(model.k_roll.detach().cpu().item()), float(cfg["k_roll_init"]), places=6)
        self.assertEqual(model.wave_shape_correction_input_dim, 2)
        model.eval()
        x_feedback_changed = x.clone()
        x_feedback_changed[..., motion_idx["phi_fb"]] += 10.0
        x_feedback_changed[..., motion_idx["v_fb"]] -= 10.0
        with torch.no_grad():
            phi_base, v_base, _, fr_base = model(x, x)
            phi_changed, v_changed, _, fr_changed = model(
                x_feedback_changed, x_feedback_changed
            )
        self.assertTrue(torch.allclose(phi_base, phi_changed, atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(v_base, v_changed, atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(fr_base, fr_changed, atol=1.0e-6, rtol=1.0e-6))
        with torch.no_grad():
            initial_correction = model.wave_forcing_breakdown(x)["wave_shape_correction_force"]
            final = model.wave_shape_correction[-1]
            self.assertIsInstance(final, nn.Linear)
            final.bias.fill_(1.0)
            active_correction = model.wave_forcing_breakdown(x)["wave_shape_correction_force"]
        self.assertTrue(torch.allclose(initial_correction, torch.zeros_like(initial_correction), atol=1.0e-7))
        self.assertTrue(torch.all(torch.isfinite(active_correction)))
        self.assertGreater(float(torch.mean(torch.abs(active_correction)).detach().cpu().item()), 0.0)

    def test_roll_amplitude_calibration_boosts_high_roll_only(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "roll_amplitude_calibration_enabled": True,
            "roll_amplitude_calibration_gain_init": 0.20,
            "roll_amplitude_calibration_gain_max": 0.40,
            "roll_amplitude_calibration_threshold_scaled": 0.75,
            "roll_amplitude_calibration_softness_scaled": 0.10,
            "roll_amplitude_calibration_rate_scale": True,
            "lstm_hidden_size": 32,
            "lstm_layers": 1,
            "fc_hidden": 32,
        })
        model = PINNLSTM(cfg, 3, {"wave_signal": [0]}, {}, {})
        phi_raw = torch.tensor([[[0.10], [1.50], [-1.50]]], dtype=torch.float32)
        v_raw = torch.ones_like(phi_raw)
        phi_cal, v_cal, boost = model.calibrate_roll_amplitude(phi_raw, v_raw)
        self.assertLess(float(boost[0, 0, 0].detach().cpu().item()), 1.01)
        self.assertGreater(float(boost[0, 1, 0].detach().cpu().item()), 1.19)
        self.assertLessEqual(float(boost[0, 1, 0].detach().cpu().item()), 1.40 + 1.0e-6)
        self.assertGreater(float(phi_cal[0, 1, 0].detach().cpu().item()), float(phi_raw[0, 1, 0].item()))
        self.assertLess(float(phi_cal[0, 2, 0].detach().cpu().item()), float(phi_raw[0, 2, 0].item()))
        self.assertGreater(float(v_cal[0, 1, 0].detach().cpu().item()), 1.19)

    def test_data_time_window_truncates_vessel_and_wave_channels(self):
        lg = logging.getLogger("unit_test_window_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg["disable_data_time_window"] = False
        cfg["data_time_start_s"] = 20.0
        cfg["data_time_end_s"] = 280.0
        cfg["data_trim_final_s"] = 0.0
        t = np.linspace(0.0, 300.0, 301)
        phi = np.sin(t)
        pack = {
            "wave_time_raw": t.copy(),
            "wave_signal_raw": np.cos(t),
            "speed_raw": 1.0 + 0.01 * t,
            "yawrate_raw": 0.001 * t,
        }
        t_win, phi_win, pack_win, info = apply_data_time_window(t, phi, pack, cfg, lg)
        self.assertEqual(float(t_win[0]), 20.0)
        self.assertEqual(float(t_win[-1]), 280.0)
        self.assertEqual(len(t_win), 261)
        self.assertEqual(len(phi_win), len(t_win))
        self.assertEqual(len(pack_win["speed_raw"]), len(t_win))
        self.assertEqual(float(pack_win["wave_time_raw"][0]), 20.0)
        self.assertEqual(float(pack_win["wave_time_raw"][-1]), 280.0)
        self.assertTrue(info["active"])

    def test_revm_disables_data_time_window_even_if_start_is_configured(self):
        lg = logging.getLogger("unit_test_window_disabled_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg["data_trim_final_s"] = 0.0
        cfg["data_time_start_s"] = 50.0
        cfg["data_time_end_s"] = 80.0
        t = np.linspace(0.0, 100.0, 101)
        phi = np.sin(t)
        pack = {
            "wave_time_raw": t.copy(),
            "wave_signal_raw": np.cos(t),
        }
        t_out, phi_out, pack_out, info = apply_data_time_window(t, phi, pack, cfg, lg)

        self.assertFalse(bool(info["active"]))
        self.assertTrue(bool(info["disabled"]))
        self.assertEqual(len(t_out), len(t))
        self.assertEqual(len(pack_out["wave_time_raw"]), len(pack["wave_time_raw"]))
        self.assertEqual(float(t_out[0]), 0.0)
        self.assertEqual(float(t_out[-1]), 100.0)

    def test_final_data_trim_applies_when_manual_window_is_disabled(self):
        lg = logging.getLogger("unit_test_final_trim_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg["disable_data_time_window"] = True
        cfg["data_trim_final_s"] = 2.0
        t = np.linspace(0.0, 100.0, 101)
        phi = np.sin(t)
        pack = {
            "wave_time_raw": t.copy(),
            "wave_signal_raw": np.cos(t),
        }
        t_out, phi_out, pack_out, info = apply_data_time_window(t, phi, pack, cfg, lg)

        self.assertTrue(bool(info["active"]))
        self.assertTrue(bool(info["disabled_manual_window"]))
        self.assertEqual(len(phi_out), len(t_out))
        self.assertEqual(float(t_out[-1]), 98.0)
        self.assertEqual(float(pack_out["wave_time_raw"][-1]), 98.0)
        self.assertAlmostEqual(float(info["trim_final_s"]), 2.0, places=12)

    def test_prepare_data_runs_training_only_wave_lag_search(self):
        lg = logging.getLogger("unit_test_wave_lag_search_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "device": "cpu",
            "seq_len": 16,
            "stride": 8,
            "t_end_demo": 4.0,
            "dt_demo": 0.04,
            "val_frac": 0.0,
            "validation_window_s": None,
            "forecast_window_s": 0.64,
            "fixed_wave_lag_s": None,
            "fixed_wave_lag_source": None,
            "data_time_start_s": None,
            "data_time_end_s": None,
            "data_trim_final_s": 0.0,
            "apply_position_encounter_shift": True,
        })
        t, phi, pack = generate_internal_demo(cfg, lg)
        data = prepare_data(t, phi, pack, cfg, lg)

        scores = np.asarray(data["lag_scores"], dtype=float)
        self.assertEqual(len(np.asarray(data["lag_grid_s"])), 81)
        self.assertTrue(np.any(np.isfinite(scores)))
        self.assertGreaterEqual(float(data["best_wave_lag_s"]), 0.0)
        self.assertLessEqual(float(data["best_wave_lag_s"]), 4.0)
        self.assertEqual(
            data["wave_lag_source"],
            "../LSTMPINNS/PINNSLSTM_REAL_KRISO_RevR.py",
        )
        self.assertTrue(bool(data["encounter_stats"]["active"]))
        self.assertTrue(bool(data["orientation_stats"]["active"]))
        self.assertTrue(np.allclose(data["encounter_stats"]["direction_xy"], [-1.0, 0.0]))
        self.assertTrue(np.allclose(data["orientation_stats"]["wave_direction_xy"], [-1.0, 0.0]))
        for name in [
            "x_position",
            "y_position",
            "encounter_frequency_ratio",
            "encounter_time_shift",
            "track_speed_xy",
            "wave_perpendicular_velocity",
            "wave_parallel_velocity",
            "heading_perpendicular_to_waves",
            "heading_parallel_to_waves",
            "heading_obliquity_abs",
            "wave_orientation_effect_gain",
        ]:
            with self.subTest(positional_feature=name):
                self.assertTrue(data["state_feature_indices"][name])
        self.assertFalse(bool(data["orientation_stats"].get("wave_orientation_gate_active", True)))
        self.assertGreaterEqual(float(data["orientation_stats"]["wave_orientation_gain_min"]), 0.0)
        self.assertLessEqual(float(data["orientation_stats"]["wave_orientation_gain_max"]), 1.0)

    def test_orientation_features_follow_configured_negative_x_wave_direction(self):
        t = np.arange(0.0, 2.04, 0.04)
        x = t.copy()
        y = 0.5 * t
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "wave_direction_xy": [-1.0, 0.0],
            "wave_orientation_gate_enabled": True,
            "orientation_smoothing_seconds": 0.0,
            "wave_parallel_gate_smoothing_seconds": 0.0,
        })

        features, info = vessel_wave_orientation_features_from_xy(t, x, y, cfg)

        speed = math.sqrt(1.25)
        self.assertTrue(bool(info["active"]))
        self.assertTrue(np.allclose(info["wave_direction_xy"], [-1.0, 0.0]))
        self.assertTrue(np.allclose(info["crest_direction_xy"], [0.0, -1.0]))
        self.assertTrue(np.allclose(features["track_speed_xy"], speed))
        self.assertTrue(np.allclose(features["wave_perpendicular_velocity"], -1.0))
        self.assertTrue(np.allclose(features["wave_parallel_velocity"], -0.5))
        self.assertTrue(np.allclose(features["heading_perpendicular_to_waves"], -1.0 / speed))
        self.assertTrue(np.allclose(features["heading_parallel_to_waves"], -0.5 / speed))
        self.assertTrue(np.all(features["wave_orientation_effect_gain"] > 0.90))
        self.assertTrue(bool(info["wave_orientation_gate_active"]))

        parallel_features, _ = vessel_wave_orientation_features_from_xy(
            t,
            np.zeros_like(t),
            t,
            cfg,
        )
        self.assertTrue(
            np.allclose(
                parallel_features["wave_orientation_effect_gain"],
                float(cfg["wave_parallel_min_gain"]),
            )
        )

    def test_wave_orientation_gain_attenuates_pure_wave_force(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "wave_forcing_gain": 1.0,
            "state_forcing_gain": 0.0,
            "wave_parallel_min_gain": 0.20,
            "wave_orientation_gate_enabled": True,
            "lstm_hidden_size": 32,
            "lstm_layers": 1,
            "fc_hidden": 32,
        })
        wave_idx = {"wave_signal": [0]}
        state_idx = {"wave_orientation_effect_gain": [1]}
        model = PINNLSTM(cfg, 2, wave_idx, state_idx, {})
        with torch.no_grad():
            model.a_wave.fill_(1.0)
        high_gain = torch.ones(1, 4, 2)
        high_gain[..., 0] = 2.0
        high_gain[..., 1] = 1.0
        low_gain = high_gain.clone()
        low_gain[..., 1] = 0.20
        high_force = model.pure_wave_forcing(high_gain)
        low_force = model.pure_wave_forcing(low_gain)
        self.assertTrue(
            torch.allclose(
                low_force,
                0.20 * high_force,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )

    def test_wave_envelope_gate_starts_as_identity_and_can_modulate(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "wave_forcing_gain": 1.0,
            "state_forcing_gain": 0.0,
            "wave_orientation_gate_enabled": False,
            "wave_envelope_gate_enabled": True,
            "wave_envelope_gate_gain": 0.50,
            "wave_envelope_gate_min": 0.35,
            "wave_envelope_gate_max": 2.20,
            "lstm_hidden_size": 32,
            "lstm_layers": 1,
            "fc_hidden": 32,
        })
        wave_idx = {"wave_signal": [0], "wave_envelope": [1], "wave_envelope_slow": [2]}
        state_idx = {"x_position": [3], "y_position": [4], "wave_orientation_effect_gain": [5]}
        model = PINNLSTM(cfg, 6, wave_idx, state_idx, {})
        x = torch.randn(1, 8, 6)
        identity_gate = model.learned_wave_envelope_gate(x)
        self.assertTrue(torch.allclose(identity_gate, torch.ones_like(identity_gate), atol=1.0e-6))

        with torch.no_grad():
            model.wave_envelope_gate[-1].bias.fill_(2.0)
        boosted_gate = model.learned_wave_envelope_gate(x)
        self.assertTrue(torch.all(boosted_gate > identity_gate).item())
        self.assertLessEqual(float(torch.max(boosted_gate).item()), float(cfg["wave_envelope_gate_max"]))

    def test_multi_run_split_holds_out_73_and_keeps_windows_inside_runs(self):
        lg = logging.getLogger("unit_test_multi_run_split_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "seq_len": 4,
            "stride": 1,
            "val_frac": 0.0,
            "validation_window_s": None,
            "forecast_window_s": 2.0,
            "forecast_sheet": "KTHTest73",
            "training_sheets": ["KTHTest73", "KTHTest74", "KTHTest75"],
        })
        t = np.concatenate([
            np.arange(0.0, 10.0),
            np.arange(100.0, 110.0),
            np.arange(200.0, 210.0),
        ])
        segments = [(0, 9), (10, 19), (20, 29)]
        names = ["KTHTest73", "KTHTest74", "KTHTest75"]
        split = split_preforecast_windows(t, cfg, lg, segments, names)
        self.assertEqual(tuple(split["forecast_region"]), (7, 9))
        starts = np.asarray(split["train_window_starts"], dtype=int)
        self.assertTrue(np.all(((starts + int(cfg["seq_len"]) - 1) < 7) | (starts >= 10)))
        for start in starts:
            end = int(start) + int(cfg["seq_len"]) - 1
            self.assertTrue(any(lo <= int(start) <= end <= hi for lo, hi in segments))
        self.assertTrue(np.any((starts >= 10) & (starts <= 16)))
        self.assertTrue(np.any((starts >= 20) & (starts <= 26)))

    def test_default_split_is_contiguous_75_train_15_validation_10_forecast(self):
        lg = logging.getLogger("unit_test_chronological_split_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "seq_len": 10,
            "stride": 1,
            "forecast_sheet": "Run1",
            "training_sheets": ["Run1"],
        })
        t = np.arange(1000, dtype=float)
        split = split_preforecast_windows(t, cfg, lg, [(0, 999)], ["Run1"])

        self.assertEqual(tuple(split["forecast_region"]), (900, 999))
        self.assertTrue(np.array_equal(split["train_idx"], np.arange(0, 750, dtype=int)))
        self.assertTrue(np.array_equal(split["val_idx"], np.arange(750, 900, dtype=int)))
        self.assertEqual(int(split["val_idx"][-1]) + 1, int(split["forecast_region"][0]))
        self.assertLessEqual(int(np.max(split["train_window_starts"])) + cfg["seq_len"] - 1, 749)
        self.assertGreaterEqual(int(np.min(split["val_window_starts"])), 750)
        info = split["validation_split"]
        self.assertIsNone(info["validation_window_s"])
        self.assertAlmostEqual(float(info["actual_train_point_frac"]), 0.75, places=12)
        self.assertAlmostEqual(float(info["actual_val_point_frac"]), 0.15, places=12)
        self.assertAlmostEqual(float(info["actual_forecast_point_frac"]), 0.10, places=12)

    def test_turning_point_sampling_duplicates_training_windows_only(self):
        t = np.arange(120, dtype=float) * 0.04
        phi = np.deg2rad(2.0 * np.sin(2.0 * math.pi * 1.0 * t))
        seq_len = 16
        train_idx = np.arange(0, 80, dtype=int)
        base = np.arange(0, 65, 8, dtype=int)
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "turning_point_sampling_enabled": True,
            "turning_point_sampling_repeats": 2,
            "turning_point_sampling_min_prominence_deg": 0.02,
            "turning_point_sampling_neighbourhood": 3,
            "turning_point_sampling_window_fracs": [0.5],
        })

        augmented, info = augment_window_starts_for_turning_points(
            t, phi, base, train_idx, seq_len, cfg
        )

        self.assertTrue(bool(info["enabled"]))
        self.assertGreater(int(info["selected_turning_points"]), 0)
        self.assertGreater(int(info["extra_windows"]), 0)
        self.assertGreater(augmented.size, base.size)
        counts = collections.Counter(int(s) for s in augmented)
        self.assertTrue(any(count > 1 for count in counts.values()))
        train_mask = np.zeros_like(t, dtype=bool)
        train_mask[train_idx] = True
        for start in augmented:
            self.assertTrue(np.all(train_mask[int(start):int(start) + seq_len]))

    def test_direct_forecast_turning_point_tasks_penalise_internal_missed_extrema(self):
        class DummyNoFeedbackModel:
            motion_feature_indices: Dict[str, List[int]] = {}

        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "direct_forecast_loss_horizons_s": [0.40],
            "direct_forecast_loss_horizon_weights": [1.0],
            "direct_forecast_loss_min_steps": 3,
            "direct_forecast_turning_point_max_tasks": 2,
            "direct_forecast_turning_point_min_prominence_deg": 0.01,
            "direct_forecast_turning_point_neighbourhood": 3,
            "direct_forecast_turning_point_window_fracs": [0.5],
        })
        t = torch.arange(80, dtype=torch.float32).view(1, -1, 1) * 0.04
        target_phi = torch.sin(2.0 * math.pi * t)
        target_v = torch.cos(2.0 * math.pi * t)
        pred_phi = target_phi.clone()
        pred_phi[:, 1:22, :] = 0.0
        pred_v = target_v.clone()
        X = torch.zeros(1, 80, 1, dtype=torch.float32)

        cfg["direct_forecast_turning_point_sampling_enabled"] = False
        without_tasks = direct_lstm_rollout_losses(
            DummyNoFeedbackModel(), pred_phi, pred_v, X, target_phi, target_v, t, cfg
        )
        cfg["direct_forecast_turning_point_sampling_enabled"] = True
        cfg["direct_forecast_turning_point_weight"] = 1.0
        with_tasks = direct_lstm_rollout_losses(
            DummyNoFeedbackModel(), pred_phi, pred_v, X, target_phi, target_v, t, cfg
        )

        self.assertGreater(
            float(with_tasks["direct_forecast"].detach().cpu().item()),
            float(without_tasks["direct_forecast"].detach().cpu().item()) + 1.0e-4,
        )

    def test_amplitude_underfit_loss_penalises_shortfall(self):
        cfg = copy.deepcopy(CONFIG)
        t = torch.linspace(0.0, 2.0, 51).view(1, -1, 1)
        target = torch.sin(2.0 * math.pi * t)
        low_pred = 0.45 * target
        full_pred = target.clone()
        low_loss = amplitude_underfit_loss(low_pred, target, t, cfg)
        full_loss = amplitude_underfit_loss(full_pred, target, t, cfg)
        self.assertGreater(float(low_loss.detach().cpu().item()), float(full_loss.detach().cpu().item()) + 1.0e-4)

    def test_high_amplitude_underfit_penalises_compressed_shoulders(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "high_amplitude_underfit_quantile": 0.60,
            "high_amplitude_underfit_overshoot_weight": 0.05,
        })
        t = torch.linspace(0.0, 2.0, 201).view(1, -1, 1)
        target = torch.sin(2.0 * math.pi * t)
        compressed = 0.70 * target
        full = target.clone()
        compressed_loss = high_amplitude_underfit_loss(compressed, target, cfg)
        full_loss = high_amplitude_underfit_loss(full, target, cfg)
        self.assertGreater(
            float(compressed_loss.detach().cpu().item()),
            float(full_loss.detach().cpu().item()) + 1.0e-2,
        )

    def test_asymmetric_extrema_forecast_loss_targets_peak_underfit_and_trough_overshoot(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "peak_trough_neighbourhood": 1,
            "direct_forecast_asym_extrema_radius": 2,
            "direct_forecast_asym_extrema_quantile": 0.10,
            "direct_forecast_peak_underfit_weight": 4.0,
            "direct_forecast_peak_overshoot_weight": 0.25,
            "direct_forecast_trough_overshoot_weight": 4.0,
            "direct_forecast_trough_underfit_weight": 0.25,
        })
        t = torch.linspace(0.0, 2.0, 201).view(1, -1, 1)
        target = torch.sin(2.0 * math.pi * t)
        matched = target.clone()
        weak_peaks = torch.where(target > 0.0, 0.70 * target, target)
        deep_troughs = torch.where(target < 0.0, 1.25 * target, target)
        matched_loss = asymmetric_extrema_forecast_loss(matched, target, cfg)
        weak_peak_loss = asymmetric_extrema_forecast_loss(weak_peaks, target, cfg)
        deep_trough_loss = asymmetric_extrema_forecast_loss(deep_troughs, target, cfg)
        self.assertGreater(
            float(weak_peak_loss.detach().cpu().item()),
            float(matched_loss.detach().cpu().item()) + 1.0e-2,
        )
        self.assertGreater(
            float(deep_trough_loss.detach().cpu().item()),
            float(matched_loss.detach().cpu().item()) + 1.0e-2,
        )

    def test_total_force_underforce_loss_penalises_collapsed_burst(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "total_force_target_underforce_weight": 3.0,
            "total_force_event_quantile": 0.50,
            "force_envelope_window_s": 0.12,
            "force_target_band_mse_weight": 0.0,
            "force_target_raw_weight": 1.0,
        })
        t = torch.arange(12, dtype=torch.float32).view(1, -1, 1) * 0.04
        target = torch.tensor(
            [[[0.0], [0.2], [1.0], [2.0], [1.0], [0.2], [0.0], [-0.2], [-1.0], [-2.0], [-1.0], [-0.2]]],
            dtype=torch.float32,
        )
        collapsed = 0.35 * target
        matched = target.clone()
        collapsed_loss = total_force_underforce_loss(collapsed, target, t, cfg)
        matched_loss = total_force_underforce_loss(matched, target, t, cfg)
        self.assertGreater(
            float(collapsed_loss.detach().cpu().item()),
            float(matched_loss.detach().cpu().item()) + 1.0e-6,
        )

    def test_extrema_window_underfit_tolerates_small_peak_shift(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "peak_trough_neighbourhood": 1,
            "extrema_window_radius": 3,
            "extrema_window_quantile": 0.40,
            "extrema_window_overshoot_weight": 0.05,
        })
        t = torch.linspace(0.0, 2.0, 201).view(1, -1, 1)
        target = torch.sin(2.0 * math.pi * t)
        shifted_full_amplitude = torch.sin(2.0 * math.pi * (t - 0.02))
        shifted_underfit = 0.70 * shifted_full_amplitude
        full_loss = extrema_window_underfit_loss(shifted_full_amplitude, target, cfg)
        underfit_loss = extrema_window_underfit_loss(shifted_underfit, target, cfg)
        self.assertGreater(
            float(underfit_loss.detach().cpu().item()),
            float(full_loss.detach().cpu().item()) + 1.0e-2,
        )

    def test_peak_data_weight_strength_increases_extrema_penalty(self):
        target = torch.full((1, 21, 1), 0.5, dtype=torch.float32)
        target[:, 10, :] = 4.0
        pred = target + 0.25
        pred[:, 10, :] = 0.0
        weak_cfg = {
            "lambda_peak_data": 0.25,
            "peak_data_quantile": 0.70,
            "peak_data_alpha": 6.0,
        }
        strong_cfg = {
            **weak_cfg,
            "lambda_peak_data": 4.0,
        }
        weak = mse_with_peak_weight(pred, target, weak_cfg)
        strong = mse_with_peak_weight(pred, target, strong_cfg)
        self.assertGreater(
            float(strong.detach().cpu().item()),
            float(weak.detach().cpu().item()) + 1.0e-4,
        )

    def test_global_extrema_loss_penalises_peak_shift(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "global_extrema_softmax_beta": 18.0,
        })
        t = torch.linspace(0.0, 1.0, 101).view(1, -1, 1)
        target = torch.sin(2.0 * math.pi * t)
        aligned = target.clone()
        shifted = torch.sin(2.0 * math.pi * (t - 0.06))

        global_aligned = global_extrema_loss(aligned, target, t, cfg)
        global_shifted = global_extrema_loss(shifted, target, t, cfg)

        self.assertGreater(float(global_shifted.detach().cpu().item()), float(global_aligned.detach().cpu().item()) + 1.0e-4)

    def test_small_scale_losses_penalise_missing_secondary_extrema(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "local_prominence_quantile": 0.10,
            "local_prominence_window_s": 0.6,
            "high_pass_window_s": 0.6,
            "low_pass_window_s": 2.0,
            "curvature_high_pass_window_s": 0.5,
        })
        t = torch.linspace(0.0, 4.0, 201).view(1, -1, 1)
        main = torch.sin(2.0 * math.pi * 0.5 * t)
        ripple = 0.18 * torch.sin(2.0 * math.pi * 3.0 * t)
        target = main + ripple
        perfect = target.clone()
        missing_small_peaks = main

        prom_perfect = local_prominence_extrema_loss(perfect, target, t, cfg)
        prom_missing = local_prominence_extrema_loss(missing_small_peaks, target, t, cfg)
        hp_perfect = high_pass_residual_loss(perfect, target, t, cfg)
        hp_missing = high_pass_residual_loss(missing_small_peaks, target, t, cfg)
        spec_perfect = roll_spectral_shape_loss(perfect, target, t, cfg)
        spec_missing = roll_spectral_shape_loss(missing_small_peaks, target, t, cfg)
        curv_perfect = roll_curvature_loss(perfect, target, t, cfg)
        curv_missing = roll_curvature_loss(missing_small_peaks, target, t, cfg)
        slow_target = target + 0.35 * torch.sin(2.0 * math.pi * 0.12 * t)
        lp_perfect = low_pass_residual_loss(slow_target, slow_target, t, cfg)
        lp_missing = low_pass_residual_loss(target, slow_target, t, cfg)

        self.assertGreater(float(prom_missing.detach().cpu().item()), float(prom_perfect.detach().cpu().item()) + 1.0e-4)
        self.assertGreater(float(hp_missing.detach().cpu().item()), float(hp_perfect.detach().cpu().item()) + 1.0e-4)
        self.assertGreater(float(spec_missing.detach().cpu().item()), float(spec_perfect.detach().cpu().item()) + 1.0e-4)
        self.assertAlmostEqual(float(curv_perfect.detach().cpu().item()), 0.0, places=12)
        self.assertAlmostEqual(float(curv_missing.detach().cpu().item()), 0.0, places=12)
        self.assertGreater(float(lp_missing.detach().cpu().item()), float(lp_perfect.detach().cpu().item()) + 1.0e-4)

    def test_rollout_schedule_always_selects_first_batch(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({"rollout_warmup_epochs": 2, "rollout_every_n_batches": 16})
        self.assertFalse(scheduled_rollout_enabled(cfg, epoch=2, batch_idx=1))
        self.assertTrue(scheduled_rollout_enabled(cfg, epoch=3, batch_idx=1))
        self.assertFalse(scheduled_rollout_enabled(cfg, epoch=3, batch_idx=2))

    def test_lstm_recurrence_blocks_future_exogenous_samples(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "lstm_hidden_size": 16,
            "lstm_layers": 1,
            "fc_hidden": 16,
            "lstm_dropout": 0.0,
        })
        model = PINNLSTM(cfg, 5, {"wave_signal": [3]}, {}, {})
        model.eval()
        x = torch.randn(1, 12, 5)
        changed_future = x.clone()
        changed_future[:, 7:, :] += 100.0
        with torch.no_grad():
            phi_a, _, force_a, _ = model(x, x)
            phi_b, _, force_b, _ = model(changed_future, changed_future)
        self.assertTrue(torch.allclose(phi_a[:, :7, :], phi_b[:, :7, :], atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(force_a[:, :7, :], force_b[:, :7, :], atol=1.0e-6, rtol=1.0e-6))

    def test_context_stitching_keeps_prediction_with_longest_history(self):
        stitched = {"phi": np.zeros(6, dtype=float)}
        best_context = np.full(6, -1, dtype=int)
        stitch_window_by_causal_context(
            stitched,
            best_context,
            np.arange(0, 4, dtype=int),
            {"phi": np.asarray([10.0, 11.0, 12.0, 13.0])},
        )
        stitch_window_by_causal_context(
            stitched,
            best_context,
            np.arange(2, 6, dtype=int),
            {"phi": np.asarray([20.0, 21.0, 22.0, 23.0])},
        )
        self.assertTrue(np.array_equal(stitched["phi"], np.asarray([10.0, 11.0, 12.0, 13.0, 22.0, 23.0])))
        self.assertTrue(np.array_equal(best_context, np.asarray([0, 1, 2, 3, 2, 3])))

    def test_phase_metrics_detect_forecast_lag(self):
        t = np.linspace(0.0, 4.0, 101)
        measured = np.sin(2.0 * np.pi * t)
        delayed = np.sin(2.0 * np.pi * (t - 0.16))
        metrics = phase_metrics(measured, delayed, t)
        self.assertTrue(math.isfinite(metrics["phase_lag_s"]))
        self.assertGreater(abs(metrics["phase_lag_s"]), 0.0)
        self.assertLess(metrics["phase_aligned_rmse_deg"], regression_metrics(measured, delayed)["rmse_deg"])

    def test_regression_metrics_report_directional_local_extrema_errors(self):
        target = np.asarray([0.0, 3.0, 0.0, -4.0, 0.0, 2.0, 0.0, -2.0, 0.0], dtype=float)
        pred = np.asarray([0.0, 2.0, 0.0, -5.0, 0.0, 1.5, 0.0, -2.5, 0.0], dtype=float)
        metrics = regression_metrics(target, pred)
        self.assertGreater(float(metrics["extrema_point_rmse_deg"]), 0.0)
        self.assertAlmostEqual(float(metrics["peak_underfit_deg"]), 0.75, places=12)
        self.assertAlmostEqual(float(metrics["peak_underfit_max_deg"]), 1.0, places=12)
        self.assertAlmostEqual(float(metrics["trough_overshoot_deg"]), 0.75, places=12)
        self.assertAlmostEqual(float(metrics["trough_overshoot_max_deg"]), 1.0, places=12)

    def test_direct_forecast_amplitude_gain_expands_about_mean(self):
        raw = np.asarray([1.0, 2.0, 3.0], dtype=float)
        corrected = apply_direct_forecast_amplitude_gain(raw, 1.20)
        self.assertTrue(np.allclose(corrected, np.asarray([0.8, 2.0, 3.2], dtype=float)))
        self.assertAlmostEqual(float(np.mean(corrected)), float(np.mean(raw)), places=12)

    def test_lag_force_by_segments_delays_without_crossing_runs(self):
        t = np.arange(8, dtype=float)
        force = np.arange(8, dtype=float)
        delayed = lag_force_by_segments(t, force, [(0, 3), (4, 7)], 1.0)
        self.assertTrue(np.allclose(delayed[:4], np.asarray([0.0, 0.0, 1.0, 2.0])))
        self.assertTrue(np.allclose(delayed[4:], np.asarray([4.0, 4.0, 5.0, 6.0])))

    def test_revm_baseline_uses_full_pure_seas_15pct_validation_10pct_forecast(self):
        self.assertEqual(REVM_SOURCE_TRIAL_INDEX, 19)
        self.assertEqual(
            CONFIG["baseline_run_id"],
            "revm_metrics2_trial19_direct_lstm_no_ode",
        )
        self.assertEqual(CONFIG["source_metrics_file"], "metrics2.json")
        for key, expected in REVM_SOURCE_TRIAL_PARAMS.items():
            if key in {
                "state_forcing_gain",
                "turn_moment_scale",
                "force_residual_scale",
                "seq_len",
                "batch_size",
                "rollout_window_s",
                "c_roll_init",
                "c_quad_init",
                "k_roll_init",
            } or key in REVM_TRAINING_WEIGHT_KEYS:
                continue
            with self.subTest(attached_trial_parameter=key):
                self.assertEqual(CONFIG[key], expected)
        for key, expected in REVM_TRAINING_WEIGHTINGS.items():
            with self.subTest(training_weighting=key):
                self.assertEqual(CONFIG[key], expected)
        self.assertEqual(CONFIG["seq_len"], 192)
        self.assertEqual(CONFIG["stride"], 2)
        self.assertEqual(CONFIG["prediction_stride"], 128)
        self.assertEqual(CONFIG["batch_size"], 16)
        self.assertEqual(int(CONFIG["early_stopping_patience"]), 18)
        self.assertTrue(bool(CONFIG["turning_point_sampling_enabled"]))
        self.assertEqual(int(CONFIG["turning_point_sampling_repeats"]), 2)
        self.assertAlmostEqual(float(CONFIG["turning_point_sampling_min_prominence_deg"]), 0.08, places=12)
        self.assertEqual(int(CONFIG["turning_point_sampling_neighbourhood"]), 4)
        self.assertEqual(CONFIG["turning_point_sampling_window_fracs"], [0.35, 0.50, 0.65])
        expected_bayes_space = {
            "seq_len": {"type": "categorical", "values": [128, 160, 192, 224, 256]},
            "prediction_stride": {"type": "categorical", "values": [32, 64, 96, 128, 192]},
            "batch_size": {"type": "categorical", "values": [16, 32, 64]},
            "learning_rate": {"type": "float", "low": 2.0e-5, "high": 1.0e-2, "log": True},
            "weight_decay": {"type": "float", "low": 1.0e-8, "high": 2.0e-2, "log": True},
            "lstm_hidden_size": {"type": "categorical", "values": [32, 48, 64, 96, 128]},
            "lstm_layers": {"type": "categorical", "values": [1, 2, 3, 4]},
            "lstm_dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "fc_hidden": {"type": "categorical", "values": [64, 96, 128, 192, 256]},
            "c_roll_init": {"type": "float", "low": 0.12, "high": 0.22},
            "c_quad_init": {"type": "float", "low": 1.0e-6, "high": 5.0e-4, "log": True},
            "k_roll_init": {"type": "float", "low": 5.0, "high": 5.6},
        }
        self.assertEqual(set(BAYES_SEARCH_SPACE), set(expected_bayes_space))
        for key, expected_spec in expected_bayes_space.items():
            with self.subTest(architecture_dynamics_bayes_space=key):
                self.assertEqual(BAYES_SEARCH_SPACE[key]["type"], expected_spec["type"])
                if expected_spec["type"] == "categorical":
                    self.assertEqual(BAYES_SEARCH_SPACE[key]["values"], expected_spec["values"])
                else:
                    self.assertAlmostEqual(float(BAYES_SEARCH_SPACE[key]["low"]), float(expected_spec["low"]), places=12)
                    self.assertAlmostEqual(float(BAYES_SEARCH_SPACE[key]["high"]), float(expected_spec["high"]), places=12)
        for locked_key in REVM_TRAINING_WEIGHT_KEYS:
            with self.subTest(training_weight_bayes_locked=locked_key):
                self.assertNotIn(locked_key, BAYES_SEARCH_SPACE)
        self.assertAlmostEqual(float(CONFIG["fixed_wave_lag_s"]), 3.0, places=12)
        self.assertEqual(CONFIG["fixed_wave_lag_source"], "validation_force_lag_scan")
        self.assertEqual(
            CONFIG["wave_alignment_source_file"],
            "../LSTMPINNS/PINNSLSTM_REAL_KRISO_RevR.py",
        )
        self.assertEqual(CONFIG["wave_lag_feature_offsets_s"], [-0.96, -0.48, 0.0, 0.48, 0.96])
        self.assertAlmostEqual(float(CONFIG["max_wave_lag_s"]), 4.0, places=12)
        self.assertEqual(int(CONFIG["n_wave_lag_candidates"]), 81)
        self.assertAlmostEqual(float(CONFIG["bandpass_low_period_factor"]), 1.8, places=12)
        self.assertAlmostEqual(float(CONFIG["bandpass_high_period_factor"]), 0.35, places=12)
        self.assertEqual(CONFIG["default_data_file"], "KTHTest73Excel.xlsx")
        self.assertEqual(CONFIG["excel_run_sheets"], ["Active"])
        self.assertEqual(CONFIG["forecast_sheet"], "KTHTest73")
        self.assertEqual(CONFIG["training_sheets"], ["KTHTest73"])
        self.assertTrue(bool(CONFIG["disable_data_time_window"]))
        self.assertIsNone(CONFIG["data_time_start_s"])
        self.assertIsNone(CONFIG["data_time_end_s"])
        self.assertAlmostEqual(float(CONFIG["data_trim_final_s"]), 2.0, places=12)
        self.assertTrue(bool(CONFIG["use_position_features"]))
        self.assertFalse(bool(CONFIG["apply_position_encounter_shift"]))
        self.assertTrue(bool(CONFIG["use_orientation_features"]))
        self.assertFalse(bool(CONFIG["wave_orientation_gate_enabled"]))
        self.assertAlmostEqual(float(CONFIG["wave_parallel_min_gain"]), 0.3173611847161887, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_parallel_velocity_blend"]), 0.5855481030715662, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_parallel_velocity_reference_quantile"]), 0.8800043044906735, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_parallel_obliquity_power"]), 2.459497372056859, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_parallel_gate_smoothing_seconds"]), 0.60, places=12)
        self.assertFalse(bool(CONFIG["wave_parallel_gate_state_force"]))
        self.assertFalse(bool(CONFIG["use_motion_feedback"]))
        self.assertEqual(CONFIG["motion_feedback_delay_offsets_s"], [0.20, 0.40, 0.80])
        self.assertAlmostEqual(
            float(CONFIG["motion_feedback_rollout_window_s"]), 5.08, places=12
        )
        self.assertEqual(int(CONFIG["motion_feedback_rollout_batch_size"]), 1)
        self.assertEqual(float(CONFIG["state_forcing_gain"]), 0.0)
        self.assertAlmostEqual(float(CONFIG["wave_forcing_gain"]), 4.0, places=12)
        self.assertFalse(bool(CONFIG["wave_shape_correction_enabled"]))
        self.assertEqual(int(CONFIG["wave_shape_correction_hidden"]), 8)
        self.assertAlmostEqual(float(CONFIG["wave_shape_correction_gain"]), 0.0, places=12)
        self.assertEqual(float(CONFIG["wave_gate_bias"]), 1.0)
        self.assertEqual(float(CONFIG["wave_gate_gain"]), 0.0)
        self.assertAlmostEqual(float(CONFIG["motion_feedback_backbone_gain"]), 0.0, places=12)
        self.assertEqual(float(CONFIG["motion_feedback_gate_gain"]), 0.0)
        self.assertEqual(float(CONFIG["turn_moment_scale"]), 0.0)
        self.assertNotIn("state_forcing_gain", BAYES_SEARCH_SPACE)
        self.assertNotIn("turn_moment_scale", BAYES_SEARCH_SPACE)
        self.assertEqual(CONFIG["val_every"], 5)
        self.assertEqual(CONFIG["training_selection_metric"], "forecast_r2")
        self.assertAlmostEqual(float(CONFIG["direct_forecast_selection_r2_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_selection_loss_weight"]), 0.35, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_selection_peak_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_selection_extrema_weight"]), 0.75, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_selection_asym_extrema_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_selection_spectral_weight"]), 0.25, places=12)
        self.assertIsNone(CONFIG["validation_window_s"])
        self.assertAlmostEqual(float(CONFIG["val_frac"]), 0.15, places=12)
        self.assertAlmostEqual(float(CONFIG["forecast_frac"]), 0.10, places=12)
        self.assertIsNone(CONFIG["forecast_window_s"])
        self.assertIsNone(CONFIG["forecast_start_s"])
        self.assertAlmostEqual(float(CONFIG["lambda_data"]), 1.60, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_r2_data"]), 17.028917094784482, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rate_data"]), 0.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_roll_slope"]), 0.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_kinematic"]), 0.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_physics"]), 0.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_boundary"]), 0.0, places=12)
        for pinn_weight in [
            "lambda_rate_data",
            "lambda_roll_slope",
            "lambda_kinematic",
            "lambda_physics",
            "lambda_boundary",
        ]:
            with self.subTest(nopinn_training_weighting=pinn_weight):
                self.assertNotIn(pinn_weight, REVM_TRAINING_WEIGHTINGS)
        self.assertAlmostEqual(float(CONFIG["lambda_total_force_target"]), 0.35, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_target_phase_weight"]), 2.0, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_target_underforce_weight"]), 3.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_total_force_event"]), 1.5, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_event_quantile"]), 0.72, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_event_weight"]), 6.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_total_force_band_shape"]), 0.90, places=12)
        self.assertEqual(CONFIG["total_force_band_shape_freqs_hz"], [0.45, 1.70, 3.20])
        self.assertEqual(CONFIG["total_force_band_shape_weights"], [0.25, 5.0, 6.0])
        self.assertAlmostEqual(float(CONFIG["total_force_band_shape_phase_weight"]), 0.45, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_band_shape_event_weight"]), 10.0, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_band_shape_envelope_weight"]), 1.20, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_total_force_spectral_shape"]), 1.60, places=12)
        self.assertEqual(CONFIG["total_force_spectral_shape_band_lows_hz"], [0.02, 0.95, 1.55])
        self.assertEqual(CONFIG["total_force_spectral_shape_band_highs_hz"], [0.95, 1.55, 3.20])
        self.assertEqual(CONFIG["total_force_spectral_shape_weights"], [1.0, 3.0, 18.0])
        self.assertEqual(CONFIG["total_force_spectral_shape_excess_only"], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(CONFIG["total_force_spectral_shape_underfit_weight"]), 16.00, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_spectral_shape_complex_weight"]), 0.60, places=12)
        self.assertAlmostEqual(float(CONFIG["total_force_spectral_shape_low_excess_margin"]), 1.02, places=12)
        self.assertAlmostEqual(float(CONFIG["force_regime_parallel_max_gain"]), 0.45, places=12)
        self.assertAlmostEqual(float(CONFIG["force_regime_side_min_gain"]), 0.75, places=12)
        self.assertAlmostEqual(float(CONFIG["force_regime_parallel_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["force_regime_oblique_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["force_regime_side_weight"]), 2.0, places=12)
        self.assertAlmostEqual(float(CONFIG["force_regime_envelope_weight"]), 0.25, places=12)
        self.assertAlmostEqual(
            float(CONFIG["wave_force_target_smoothing_seconds"]), 0.06, places=12
        )
        self.assertAlmostEqual(float(CONFIG["force_target_band_mse_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["force_target_raw_weight"]), 0.25, places=12)
        self.assertAlmostEqual(float(CONFIG["force_target_huber_delta"]), 1.50, places=12)
        self.assertAlmostEqual(float(CONFIG["force_target_band_huber_delta"]), 1.25, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_force_target_phase_weight"]), 3.0, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_force_target_amplitude_weight"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_force_target_underforce_weight"]), 3.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_wave_force_highpass"]), 1.5, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_force_highpass_window_s"]), 0.45, places=12)
        self.assertAlmostEqual(float(CONFIG["force_band_amplitude_underfit_weight"]), 4.0, places=12)
        self.assertFalse(bool(CONFIG["forecast_use_ode"]))
        self.assertTrue(bool(CONFIG["direct_forecast_amplitude_calibration_enabled"]))
        self.assertAlmostEqual(float(CONFIG["direct_forecast_amplitude_calibration_gain_min"]), 0.85, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_amplitude_calibration_gain_max"]), 1.25, places=12)
        self.assertEqual(int(CONFIG["direct_forecast_amplitude_calibration_gain_steps"]), 41)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_amplitude_calibration_extrema_weight"]), 0.35, places=12)
        self.assertAlmostEqual(float(CONFIG["forecast_force_lag_s"]), -0.18, places=12)
        self.assertFalse(bool(CONFIG["forecast_force_post_correction_enabled"]))
        self.assertAlmostEqual(float(CONFIG["lambda_force_envelope"]), 0.4, places=12)
        self.assertAlmostEqual(float(CONFIG["force_envelope_window_s"]), 0.80, places=12)
        self.assertAlmostEqual(float(CONFIG["force_envelope_huber_delta"]), 0.60, places=12)
        self.assertFalse(bool(CONFIG["wave_envelope_gate_enabled"]))
        self.assertAlmostEqual(float(CONFIG["wave_envelope_gate_gain"]), 0.60, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_envelope_gate_min"]), 0.35, places=12)
        self.assertAlmostEqual(float(CONFIG["wave_envelope_gate_max"]), 1.25, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_force_reg"]), 3.0e-5, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_force_smooth"]), 2.5e-5, places=12)
        self.assertAlmostEqual(float(CONFIG["force_residual_scale"]), 0.0, places=12)
        self.assertNotIn("force_residual_scale", BAYES_SEARCH_SPACE)
        self.assertAlmostEqual(float(CONFIG["lambda_peak_data"]), 4.50, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_peak_trough"]), 4.196567737966396, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_amplitude_underfit"]), 7.584956872866984, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_extrema_window_underfit"]), 12.00, places=12)
        self.assertEqual(int(CONFIG["extrema_window_radius"]), 3)
        self.assertAlmostEqual(float(CONFIG["extrema_window_quantile"]), 0.60, places=12)
        self.assertAlmostEqual(float(CONFIG["extrema_window_overshoot_weight"]), 0.12, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_high_amplitude_underfit"]), 6.00, places=12)
        self.assertAlmostEqual(float(CONFIG["high_amplitude_underfit_quantile"]), 0.68, places=12)
        self.assertAlmostEqual(float(CONFIG["high_amplitude_underfit_overshoot_weight"]), 0.08, places=12)
        self.assertTrue(bool(CONFIG["roll_amplitude_calibration_enabled"]))
        self.assertAlmostEqual(float(CONFIG["roll_amplitude_calibration_gain_init"]), 0.0742194652557373, places=12)
        self.assertAlmostEqual(float(CONFIG["roll_amplitude_calibration_gain_max"]), 0.40, places=12)
        self.assertAlmostEqual(float(CONFIG["roll_amplitude_calibration_threshold_scaled"]), 0.75, places=12)
        self.assertAlmostEqual(float(CONFIG["roll_amplitude_calibration_softness_scaled"]), 0.25, places=12)
        self.assertTrue(bool(CONFIG["roll_amplitude_calibration_rate_scale"]))
        self.assertAlmostEqual(float(CONFIG["lambda_global_extrema"]), 2.1055894117187552, places=12)
        self.assertEqual(CONFIG["wave_feature_set"], [
            "wave_signal",
            "wave_slope",
            "wave_abs_slope",
            "wave_curvature",
            "wave_abs_curvature",
            "wave_signal_slope_product",
            "wave_envelope",
            "wave_envelope_slow",
        ])
        self.assertAlmostEqual(float(CONFIG["lambda_high_pass_residual"]), 0.70, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_low_pass_residual"]), 1.80, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_roll_spectral_shape"]), 1.90, places=12)
        self.assertEqual(CONFIG["roll_spectral_shape_band_lows_hz"], [0.00, 0.85, 1.55])
        self.assertEqual(CONFIG["roll_spectral_shape_band_highs_hz"], [0.28, 1.15, 3.20])
        self.assertEqual(CONFIG["roll_spectral_shape_weights"], [2.5, 4.0, 3.5])
        self.assertEqual(CONFIG["roll_spectral_shape_raw_bands"], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(CONFIG["roll_spectral_shape_underfit_weight"]), 3.25, places=12)
        self.assertAlmostEqual(float(CONFIG["roll_spectral_shape_complex_weight"]), 0.18, places=12)
        self.assertAlmostEqual(float(CONFIG["roll_spectral_shape_mag_floor_ratio"]), 0.025, places=12)
        self.assertAlmostEqual(float(CONFIG["curvature_high_pass_window_s"]), 0.22, places=12)
        self.assertAlmostEqual(float(CONFIG["curvature_scale_floor_ratio"]), 0.04, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout"]), 4.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_rate"]), 0.15, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_amplitude"]), 8.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_phase"]), 3.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_peak_trough"]), 4.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_global_extrema"]), 1.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_extrema_window_underfit"]), 3.00, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_high_amplitude_underfit"]), 1.50, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_local_prominence"]), 0.5, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_high_pass_residual"]), 0.70, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_rollout_roll_spectral_shape"]), 0.45, places=12)
        self.assertTrue(bool(CONFIG["direct_forecast_loss_enabled"]))
        self.assertAlmostEqual(float(CONFIG["direct_forecast_loss_window_s"]), 7.60, places=12)
        self.assertEqual(CONFIG["direct_forecast_loss_horizons_s"], [1.52, 3.04, 5.08, 7.60])
        self.assertEqual(CONFIG["direct_forecast_loss_horizon_weights"], [1.1, 1.2, 1.3, 1.6])
        self.assertAlmostEqual(float(CONFIG["direct_forecast_tail_fraction"]), 0.0, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_tail_multiplier"]), 1.0, places=12)
        self.assertEqual(int(CONFIG["direct_forecast_loss_min_steps"]), 8)
        self.assertEqual(int(CONFIG["direct_forecast_loss_batch_size"]), 3)
        self.assertTrue(bool(CONFIG["direct_forecast_turning_point_sampling_enabled"]))
        self.assertAlmostEqual(float(CONFIG["direct_forecast_turning_point_weight"]), 0.85, places=12)
        self.assertEqual(int(CONFIG["direct_forecast_turning_point_max_tasks"]), 3)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_turning_point_min_prominence_deg"]), 0.08, places=12)
        self.assertEqual(int(CONFIG["direct_forecast_turning_point_neighbourhood"]), 4)
        self.assertEqual(CONFIG["direct_forecast_turning_point_window_fracs"], [0.35, 0.50, 0.65])
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast"]), 6.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_rate"]), 0.20, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_amplitude"]), 24.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_phase"]), 4.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_peak_trough"]), 26.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_global_extrema"]), 2.5, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_extrema_window_underfit"]), 13.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_high_amplitude_underfit"]), 8.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_asymmetric_extrema"]), 10.0, places=12)
        self.assertEqual(int(CONFIG["direct_forecast_asym_extrema_radius"]), 4)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_asym_extrema_quantile"]), 0.45, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_peak_underfit_weight"]), 4.0, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_peak_overshoot_weight"]), 0.35, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_trough_overshoot_weight"]), 4.5, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_trough_underfit_weight"]), 0.75, places=12)
        self.assertAlmostEqual(float(CONFIG["direct_forecast_asym_extrema_scale_floor_ratio"]), 0.06, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_local_prominence"]), 2.0, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_high_pass_residual"]), 1.20, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_roll_spectral_shape"]), 2.50, places=12)
        self.assertAlmostEqual(float(CONFIG["lambda_direct_forecast_roll_curvature"]), 0.25, places=12)
        self.assertAlmostEqual(float(CONFIG["bayes_extrema_objective_weight"]), 20.0, places=12)
        self.assertAlmostEqual(float(CONFIG["bayes_forecast_extrema_rmse_objective_weight"]), 1.5, places=12)
        self.assertAlmostEqual(float(CONFIG["bayes_forecast_peak_underfit_objective_weight"]), 6.0, places=12)
        self.assertAlmostEqual(float(CONFIG["bayes_forecast_trough_overshoot_objective_weight"]), 6.0, places=12)
        for locked_key in [
            "lambda_total_force_target",
            "lambda_total_force_event",
            "lambda_wave_force_highpass",
            "lambda_force_envelope",
            "wave_force_target_amplitude_weight",
            "wave_force_target_underforce_weight",
            "force_band_amplitude_underfit_weight",
            "total_force_target_phase_weight",
            "total_force_target_underforce_weight",
            "total_force_event_quantile",
            "total_force_event_weight",
            "total_force_band_shape_event_weight",
            "total_force_spectral_shape_low_excess_margin",
            "force_regime_parallel_max_gain",
            "force_regime_side_min_gain",
            "force_regime_side_weight",
            "force_regime_envelope_weight",
            "wave_shape_correction_gain",
            "wave_envelope_gate_gain",
            "lambda_rollout_rate",
            "lambda_rollout_extrema_window_underfit",
            "lambda_extrema_window_underfit",
            "extrema_window_radius",
            "extrema_window_quantile",
            "extrema_window_overshoot_weight",
            "extrema_window_peak_weight",
            "extrema_window_trough_weight",
            "extrema_window_scale_floor_ratio",
            "lambda_high_amplitude_underfit",
            "high_amplitude_underfit_quantile",
            "high_amplitude_underfit_overshoot_weight",
            "high_amplitude_underfit_scale_floor_ratio",
            "lambda_rollout_high_amplitude_underfit",
            "roll_amplitude_calibration_gain_init",
            "roll_amplitude_calibration_gain_max",
            "roll_amplitude_calibration_threshold_scaled",
            "roll_amplitude_calibration_softness_scaled",
            "force_envelope_window_s",
            "wave_force_highpass_window_s",
            "wave_parallel_min_gain",
            "wave_parallel_velocity_blend",
            "wave_parallel_velocity_reference_quantile",
            "wave_parallel_obliquity_power",
            "roll_spectral_shape_band_lows_hz",
            "roll_spectral_shape_band_highs_hz",
            "roll_spectral_shape_weights",
            "roll_spectral_shape_raw_bands",
            "roll_spectral_shape_mag_floor_ratio",
            "curvature_high_pass_window_s",
            "lambda_rollout_roll_spectral_shape",
            "curvature_scale_floor_ratio",
        ]:
            with self.subTest(locked_out_of_bayes=locked_key):
                self.assertNotIn(locked_key, BAYES_SEARCH_SPACE)
        self.assertAlmostEqual(float(CONFIG["rollout_window_s"]), 5.08, places=12)
        self.assertAlmostEqual(float(CONFIG["learning_rate"]), 0.0004986359006149152, places=15)
        self.assertAlmostEqual(float(CONFIG["lstm_dropout"]), 0.10453660952184797, places=15)
        self.assertAlmostEqual(float(CONFIG["weight_decay"]), 4.326508084466064e-06, places=15)
        self.assertFalse(bool(CONFIG["freeze_physics_coefficients"]))
        self.assertFalse(bool(CONFIG["bayes_freeze_physics_coefficients"]))
        self.assertAlmostEqual(float(CONFIG["c_roll_init"]), 0.20610502092309096, places=15)
        self.assertAlmostEqual(float(CONFIG["c_quad_init"]), 4.4562293397215795e-05, places=15)
        self.assertAlmostEqual(float(CONFIG["k_roll_init"]), 5.286357774176246, places=15)
        self.assertAlmostEqual(float(CONFIG["source_learned_physics_parameters"]["c_roll"]), 0.3853868842124939, places=15)
        self.assertIn("c_roll_init", BAYES_SEARCH_SPACE)
        self.assertIn("c_quad_init", BAYES_SEARCH_SPACE)
        self.assertIn("k_roll_init", BAYES_SEARCH_SPACE)
        trial_one_params = current_search_params(CONFIG, BAYES_SEARCH_SPACE)
        for key in REVM_TRAINING_WEIGHT_KEYS:
            if key in BAYES_SEARCH_SPACE:
                with self.subTest(first_bayes_trial_weighting=key):
                    self.assertEqual(trial_one_params[key], CONFIG[key])
        t = np.arange(2001, dtype=float) * 0.04
        start_idx, end_idx = choose_forecast_region(t, CONFIG)
        self.assertEqual((start_idx, end_idx), (1801, 2000))
        self.assertEqual(end_idx - start_idx + 1, round(len(t) * 0.10))

    def test_forecast_uncertainty_widens_from_handoff(self):
        measured_deg = 2.0 * np.sin(np.linspace(0.0, 8.0, 200))
        teacher_pred_deg = measured_deg + 0.5
        forecast_time = np.linspace(0.0, 5.08, 128)
        forecast_pred_deg = np.cos(forecast_time)
        forecast = {
            "enabled": True,
            "index": np.arange(72, 200, dtype=int),
            "time_s": forecast_time,
            "forecast_roll_deg": forecast_pred_deg,
        }
        data = {
            "phi": np.deg2rad(measured_deg),
            "val_idx": np.arange(120, 200, dtype=int),
            "train_idx": np.arange(0, 120, dtype=int),
        }
        result = add_forecast_uncertainty(forecast, data, teacher_pred_deg, CONFIG)
        lower = np.asarray(result["uncertainty_lower_deg"], dtype=float)
        upper = np.asarray(result["uncertainty_upper_deg"], dtype=float)
        width = upper - lower
        self.assertAlmostEqual(float(width[0]), 0.0, places=12)
        self.assertTrue(np.all(np.diff(width) >= -1.0e-12))
        self.assertGreater(float(width[-1]), float(width[len(width) // 2]))
        self.assertEqual(result["uncertainty"]["source_split"], "validation")
        self.assertFalse(result["uncertainty"]["calibrated_probabilistic_interval"])
        fit_curve = np.arange(200, dtype=float)
        fit_only_curve = mask_forecast_from_fit_curve(fit_curve, result)
        self.assertTrue(np.array_equal(fit_only_curve[:72], fit_curve[:72]))
        self.assertTrue(np.all(np.isnan(fit_only_curve[72:])))

    def test_direct_forecast_selection_objective_blends_validation_shape_terms(self):
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "direct_forecast_selection_r2_weight": 1.0,
            "direct_forecast_selection_loss_weight": 0.35,
            "direct_forecast_selection_peak_weight": 1.0,
            "direct_forecast_selection_extrema_weight": 0.75,
            "direct_forecast_selection_asym_extrema_weight": 1.25,
            "direct_forecast_selection_spectral_weight": 0.25,
        })
        metrics = {
            "val_r2_data": 0.08,
            "val_direct_forecast": 0.20,
            "val_direct_forecast_peak_trough": 0.30,
            "val_direct_forecast_extrema_window_underfit": 0.40,
            "val_direct_forecast_asymmetric_extrema": 0.20,
            "val_direct_forecast_roll_spectral_shape": 0.50,
        }
        objective, components = direct_forecast_selection_objective(metrics, "val", cfg)
        expected = 0.08 + 0.35 * 0.20 + 0.30 + 0.75 * 0.40 + 1.25 * 0.20 + 0.25 * 0.50
        self.assertAlmostEqual(float(objective), expected, places=12)
        self.assertAlmostEqual(float(components["direct_forecast_peak_trough"]), 0.30, places=12)
        self.assertAlmostEqual(float(components["direct_forecast_asymmetric_extrema"]), 0.20, places=12)

    def test_full_ode_uncertainty_starts_at_forecast_handoff(self):
        n = 200
        integration = {
            "enabled": True,
            "time_s": np.arange(n, dtype=float) * 0.04,
            "integrated_roll_deg": np.sin(np.arange(n, dtype=float) * 0.04),
        }
        forecast_idx = np.arange(120, 200, dtype=int)
        forecast = {
            "enabled": True,
            "index": forecast_idx,
            "uncertainty_std_deg": np.linspace(0.0, 1.5, len(forecast_idx)),
            "uncertainty": {"method": "unit_test", "confidence": 0.95},
        }
        result = add_full_ode_forecast_uncertainty(integration, forecast, CONFIG)
        lower = np.asarray(result["uncertainty_lower_deg"], dtype=float)
        upper = np.asarray(result["uncertainty_upper_deg"], dtype=float)
        center = np.asarray(result["integrated_roll_deg"], dtype=float)
        self.assertTrue(np.all(np.isnan(lower[:120])))
        self.assertAlmostEqual(float(lower[120]), float(center[120]), places=12)
        self.assertAlmostEqual(float(upper[120]), float(center[120]), places=12)
        self.assertGreater(float(upper[-1] - lower[-1]), 0.0)
        self.assertEqual(int(result["uncertainty"]["start_idx"]), 120)

    def test_bayes_objective_requires_both_fit_and_forecast(self):
        weights = {
            "bayes_r2_objective_weight": 20.0,
            "bayes_rmse_objective_weight": 2.0,
            "bayes_fit_r2_objective_weight": 20.0,
            "bayes_fit_rmse_objective_weight": 2.0,
        }
        fit_only = {
            **weights,
            "fit_r2": 0.999,
            "fit_rms_error_deg": 0.001,
        }
        forecast_only = {
            **weights,
            "forecast_r2": 0.999,
            "forecast_rms_error_deg": 0.001,
        }
        balanced = {
            **weights,
            "fit_r2": 0.80,
            "fit_rms_error_deg": 0.90,
            "forecast_r2": 0.80,
            "forecast_rms_error_deg": 0.80,
        }
        worse_fit = {
            **balanced,
            "fit_r2": 0.20,
            "fit_rms_error_deg": 2.0,
        }
        self.assertLess(bayes_objective_from_record(balanced), bayes_objective_from_record(fit_only))
        self.assertLess(bayes_objective_from_record(balanced), bayes_objective_from_record(forecast_only))
        self.assertLess(bayes_objective_from_record(balanced), bayes_objective_from_record(worse_fit))
        self.assertGreater(bayes_fit_rank_key(balanced), bayes_fit_rank_key(fit_only))
        self.assertTrue(bayes_target_reached(balanced, 0.75, 1.0, 0.75, 1.0))
        self.assertFalse(bayes_target_reached(fit_only, 0.75, 1.0, 0.75, 1.0))
        self.assertFalse(bayes_target_reached(forecast_only, 0.75, 1.0, 0.75, 1.0))

    def test_full_ode_stop_targets_are_independent_alternatives(self):
        misses_all = {
            "status": "ok",
            "trial_index": 1,
            "fit_r2": 0.1,
            "fit_rms_error_deg": 3.0,
            "forecast_r2": 0.1,
            "forecast_rms_error_deg": 3.0,
            "full_ode_r2": 0.95,
            "full_ode_rms_error_deg": 0.15,
        }
        r2_pass = {**misses_all, "trial_index": 2, "full_ode_r2": 0.951}
        rmse_pass = {**misses_all, "trial_index": 3, "full_ode_rms_error_deg": 0.149}
        self.assertFalse(bayes_full_ode_target_reached(misses_all, 0.95, 0.15))
        self.assertTrue(bayes_full_ode_target_reached(r2_pass, 0.95, 0.15))
        self.assertTrue(bayes_full_ode_target_reached(rmse_pass, 0.95, 0.15))

        summary: Dict[str, object] = {}
        stop_record, reason = update_bayes_stop_status(
            summary,
            [misses_all, r2_pass, rmse_pass],
            0.75, 0.15, 0.80, 0.96, 0.95, 0.15,
        )
        self.assertTrue(bool(summary["target_reached"]))
        self.assertEqual(int(summary["stop_trial_index"]), 2)
        self.assertEqual(reason, "full_ode_r2")
        self.assertEqual(int(stop_record["trial_index"]), 2)

    def test_stationary_beam_sea_rejects_moving_vessel_bayes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            bayes_dir = Path(tmp)
            legacy_summary_path = bayes_dir / "bayes_search_results_revg.json"
            safe_json_dump({
                "study_version": "revg_metrics_revg128_seed_seq128_5p08_balanced_fit_forecast_v1",
                "trials": [],
            }, legacy_summary_path)
            cfg = copy.deepcopy(CONFIG)
            with self.assertRaisesRegex(ValueError, "incompatible study"):
                load_bayes_resume_records(
                    bayes_dir,
                    bayes_dir / "bayes_search_results_revm.json",
                    cfg,
                    logging.getLogger("revm_resume_test"),
                )

    def test_current_study_resumes_matching_70_15_15_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            bayes_dir = Path(tmp)
            summary_path = bayes_dir / "bayes_search_results_revm.json"
            safe_json_dump({
                "study_version": CONFIG["bayes_study_version"],
                "trials": [],
            }, summary_path)
            records, previous_summary = load_bayes_resume_records(
                bayes_dir,
                summary_path,
                copy.deepcopy(CONFIG),
                logging.getLogger("revm_resume_test"),
            )
            self.assertEqual(records, [])
            self.assertEqual(
                previous_summary["study_version"],
                CONFIG["bayes_study_version"],
            )

    def test_bayes_objective_rewards_forecast_quality_with_fit_held_constant(self):
        high_r2_bad_aux = {
            "fit_r2": 0.80,
            "fit_rms_error_deg": 0.90,
            "forecast_r2": 0.60,
            "forecast_rms_error_deg": 0.30,
            "forecast_phase_lag_s": 10.0,
            "forecast_amplitude_underfit_ratio": 10.0,
            "final_train_physics_loss": 1.0e4,
            "final_train_kinematic_loss": 1.0e4,
            "bayes_r2_objective_weight": 50.0,
            "bayes_rmse_objective_weight": 10.0,
            "bayes_fit_r2_objective_weight": 50.0,
            "bayes_fit_rmse_objective_weight": 10.0,
            "bayes_phase_objective_weight": 0.5,
            "bayes_amplitude_objective_weight": 0.5,
            "bayes_physics_objective_weight": 0.25,
            "bayes_kinematic_objective_weight": 0.25,
        }
        lower_r2_good_aux = {
            **high_r2_bad_aux,
            "forecast_r2": 0.55,
            "forecast_rms_error_deg": 0.30,
            "forecast_phase_lag_s": 0.04000091000000339,
            "forecast_amplitude_underfit_ratio": 0.0,
            "final_train_physics_loss": 0.0,
            "final_train_kinematic_loss": 0.0,
        }
        same_r2_lower_rmse = {
            **high_r2_bad_aux,
            "forecast_rms_error_deg": 0.10,
        }
        self.assertLess(bayes_objective_from_record(high_r2_bad_aux), bayes_objective_from_record(lower_r2_good_aux))
        self.assertLess(bayes_objective_from_record(same_r2_lower_rmse), bayes_objective_from_record(high_r2_bad_aux))

    def test_bayes_objective_rewards_forecast_extrema_fidelity(self):
        common = {
            "fit_r2": 0.80,
            "fit_rms_error_deg": 0.90,
            "forecast_r2": 0.70,
            "forecast_rms_error_deg": 0.30,
            "bayes_r2_objective_weight": 50.0,
            "bayes_rmse_objective_weight": 10.0,
            "bayes_fit_r2_objective_weight": 50.0,
            "bayes_fit_rmse_objective_weight": 10.0,
            "bayes_extrema_objective_weight": 20.0,
        }
        accurate_extrema = {**common, "forecast_extrema_relative_error": 0.05}
        collapsed_extrema = {**common, "forecast_extrema_relative_error": 0.80}
        self.assertLess(
            bayes_objective_from_record(accurate_extrema),
            bayes_objective_from_record(collapsed_extrema),
        )

    def test_bayes_objective_rewards_directional_forecast_extrema_errors(self):
        common = {
            "fit_r2": 0.80,
            "fit_rms_error_deg": 0.90,
            "forecast_r2": 0.70,
            "forecast_rms_error_deg": 0.30,
            "bayes_r2_objective_weight": 50.0,
            "bayes_rmse_objective_weight": 10.0,
            "bayes_fit_r2_objective_weight": 50.0,
            "bayes_fit_rmse_objective_weight": 10.0,
            "bayes_forecast_extrema_rmse_objective_weight": 1.5,
            "bayes_forecast_peak_underfit_objective_weight": 6.0,
            "bayes_forecast_trough_overshoot_objective_weight": 6.0,
        }
        accurate_extrema = {
            **common,
            "forecast_extrema_point_rmse_deg": 0.10,
            "forecast_peak_underfit_deg": 0.05,
            "forecast_trough_overshoot_deg": 0.05,
        }
        poor_extrema = {
            **common,
            "forecast_extrema_point_rmse_deg": 1.10,
            "forecast_peak_underfit_deg": 0.85,
            "forecast_trough_overshoot_deg": 0.85,
        }
        self.assertLess(
            bayes_objective_from_record(accurate_extrema),
            bayes_objective_from_record(poor_extrema),
        )

    def test_best_forecast_r2_and_rmse_are_reported_independently(self):
        records = [
            {"status": "ok", "trial_index": 1, "forecast_r2": 0.95, "forecast_rms_error_deg": 0.50},
            {"status": "ok", "trial_index": 2, "forecast_r2": 0.90, "forecast_rms_error_deg": 0.30},
        ]
        summary: Dict[str, object] = {}
        update_best_forecast_metric_records(summary, records)
        self.assertEqual(summary["best_forecast_r2_trial"]["trial_index"], 1)
        self.assertEqual(summary["best_forecast_rmse_trial"]["trial_index"], 2)

    def test_physics_loss_finite_on_demo_batch(self):
        lg = logging.getLogger("unit_test_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "device": "cpu", "seq_len": 32, "stride": 16, "batch_size": 2,
            "lstm_hidden_size": 32, "lstm_layers": 1, "fc_hidden": 32,
            "t_end_demo": 8.0, "dt_demo": 0.04, "val_frac": 0.0, "validation_window_s": None,
            "forecast_window_s": 1.0,
            "data_time_start_s": None, "data_time_end_s": None,
            "data_trim_final_s": 0.0,
            "dataloader_num_workers": 0,
            "motion_feedback_rollout_window_s": 0.32,
            "motion_feedback_rollout_batch_size": 1,
            "lambda_total_force_band_shape": 1.0,
            "lambda_total_force_spectral_shape": 1.0,
        })
        t, phi, pack = generate_internal_demo(cfg, lg)
        data = prepare_data(t, phi, pack, cfg, lg)
        train_loader, _ = build_loaders(data, cfg, lg)
        model = build_model(cfg, data)
        batch = next(iter(train_loader))
        loss, pieces = compute_loss(model, batch, cfg)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(pieces["physics"]).item())
        self.assertTrue(torch.isfinite(pieces["total_force_target"]).item())
        self.assertTrue(torch.isfinite(pieces["total_force_event"]).item())
        self.assertTrue(torch.isfinite(pieces["total_force_band_shape"]).item())
        self.assertTrue(torch.isfinite(pieces["total_force_spectral_shape"]).item())
        self.assertTrue(torch.isfinite(pieces["force_envelope"]).item())
        self.assertTrue(torch.isfinite(pieces["rollout_peak_trough"]).item())
        self.assertTrue(torch.isfinite(pieces["rollout_global_extrema"]).item())
        self.assertTrue(torch.isfinite(pieces["extrema_window_underfit"]).item())
        self.assertTrue(torch.isfinite(pieces["rollout_extrema_window_underfit"]).item())
        self.assertTrue(torch.isfinite(pieces["high_amplitude_underfit"]).item())
        self.assertTrue(torch.isfinite(pieces["rollout_high_amplitude_underfit"]).item())
        self.assertTrue(torch.isfinite(pieces["direct_forecast"]).item())
        self.assertTrue(torch.isfinite(pieces["direct_forecast_peak_trough"]).item())
        self.assertTrue(torch.isfinite(pieces["direct_forecast_extrema_window_underfit"]).item())
        self.assertTrue(torch.isfinite(pieces["direct_forecast_high_amplitude_underfit"]).item())
        self.assertTrue(torch.isfinite(pieces["direct_forecast_asymmetric_extrema"]).item())
        self.assertTrue(torch.isfinite(pieces["direct_forecast_roll_spectral_shape"]).item())
        model.zero_grad(set_to_none=True)
        _, _, force_pred, _ = model(batch["src"], batch["tgt"])
        total_force_loss = total_force_target_loss(
            model,
            force_pred,
            batch["y_phi"],
            batch["y_v"],
            batch["t"],
            batch["tgt"],
            cfg,
        )
        total_force_loss.backward()
        self.assertIsNotNone(model.a_wave.grad)
        self.assertGreater(float(torch.sum(torch.abs(model.a_wave.grad)).item()), 0.0)

        model.zero_grad(set_to_none=True)
        _, _, force_pred, _ = model(batch["src"], batch["tgt"])
        total_event_loss = total_force_event_loss(
            model,
            force_pred,
            batch["y_phi"],
            batch["y_v"],
            batch["t"],
            batch["tgt"],
            cfg,
        )
        total_event_loss.backward()
        self.assertIsNotNone(model.a_wave.grad)
        self.assertGreater(float(torch.sum(torch.abs(model.a_wave.grad)).item()), 0.0)

        model.zero_grad(set_to_none=True)
        _, _, force_pred, _ = model(batch["src"], batch["tgt"])
        total_band_shape_loss = total_force_band_shape_loss(
            model,
            force_pred,
            batch["y_phi"],
            batch["y_v"],
            batch["t"],
            batch["tgt"],
            cfg,
        )
        total_band_shape_loss.backward()
        self.assertIsNotNone(model.a_wave.grad)
        self.assertGreater(float(torch.sum(torch.abs(model.a_wave.grad)).item()), 0.0)

        model.zero_grad(set_to_none=True)
        _, _, force_pred, _ = model(batch["src"], batch["tgt"])
        total_spectral_shape_loss = total_force_spectral_shape_loss(
            model,
            force_pred,
            batch["y_phi"],
            batch["y_v"],
            batch["t"],
            batch["tgt"],
            cfg,
        )
        total_spectral_shape_loss.backward()
        self.assertIsNotNone(model.a_wave.grad)
        self.assertGreater(float(torch.sum(torch.abs(model.a_wave.grad)).item()), 0.0)

    def test_forecast_handoff_matches_measured_motion(self):
        lg = logging.getLogger("unit_test_forecast_revm")
        lg.addHandler(logging.NullHandler())
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "device": "cpu", "seq_len": 16, "stride": 8, "batch_size": 2,
            "lstm_hidden_size": 32, "lstm_layers": 1, "fc_hidden": 32,
            "t_end_demo": 8.0, "dt_demo": 0.04, "val_frac": 0.0, "validation_window_s": None,
            "forecast_window_s": 1.0, "use_motion_feedback": True,
            "data_time_start_s": None, "data_time_end_s": None,
            "data_trim_final_s": 0.0,
            "forecast_use_ode": True,
        })
        t, phi, pack = generate_internal_demo(cfg, lg)
        data = prepare_data(t, phi, pack, cfg, lg)
        model = build_model(cfg, data)
        forecast = forecast_roll_region(model, data, cfg)
        self.assertTrue(forecast.get("enabled", False))
        self.assertTrue(forecast.get("uses_predicted_motion_feedback", False))
        start_idx = int(forecast["start_idx"])
        self.assertEqual(int(forecast["index"][0]), start_idx)
        self.assertAlmostEqual(float(forecast["forecast_phi_scaled"][0]), float(data["phi_scaled"][start_idx]), places=12)
        self.assertAlmostEqual(float(forecast["forecast_v_scaled"][0]), float(data["v_est_scaled"][start_idx]), places=12)
        self.assertAlmostEqual(float(forecast["handoff_phi_error_scaled"]), 0.0, places=12)
        self.assertAlmostEqual(float(forecast["handoff_v_error_scaled"]), 0.0, places=12)
        verification = forecast.get("wave_force_verification", {})
        self.assertTrue(bool(verification.get("wave_force_applied_to_forecast_ode", False)))
        self.assertTrue(bool(verification.get("wave_channels_present_in_forecast_tensor", False)))
        self.assertTrue(bool(verification.get("pure_wave_force_nonzero", False)))
        self.assertTrue(bool(verification.get("counterfactual_wave_effect_nonzero", False)))
        self.assertTrue(bool(verification.get("wave_orientation_gain_present", False)))
        self.assertGreater(int(verification.get("wave_feature_count", 0)), 0)
        self.assertGreater(float(verification.get("wave_input_rms_scaled", 0.0)), 0.0)
        self.assertGreater(
            float(verification.get("counterfactual_force_delta_rms_scaled", 0.0)),
            0.0,
        )
        self.assertEqual(
            len(np.asarray(forecast["forecast_pure_wave_force_scaled"])),
            len(np.asarray(forecast["index"])),
        )
        self.assertEqual(
            len(np.asarray(forecast["forecast_wave_orientation_gain"])),
            len(np.asarray(forecast["index"])),
        )
        direct_forecast = forecast.get("direct_no_ode_forecast", {})
        self.assertTrue(bool(direct_forecast.get("enabled", False)))
        self.assertTrue(bool(direct_forecast.get("uses_predicted_motion_feedback", False)))
        self.assertEqual(
            len(np.asarray(direct_forecast["forecast_roll_deg"])),
            len(np.asarray(forecast["index"])),
        )
        self.assertIn("metrics", direct_forecast)

        wave_columns = sorted({
            int(column)
            for columns in data["wave_feature_indices"].values()
            for column in columns
        })
        zero_wave_data = dict(data)
        zero_wave_X = np.asarray(data["X"], dtype=np.float32).copy()
        zero_wave_X[:, wave_columns] = 0.0
        zero_wave_data["X"] = zero_wave_X
        zero_wave_forecast = forecast_roll_region(model, zero_wave_data, cfg)
        self.assertFalse(
            np.allclose(
                np.asarray(forecast["forecast_force_scaled"], dtype=float),
                np.asarray(zero_wave_forecast["forecast_force_scaled"], dtype=float),
                atol=1.0e-8,
                rtol=1.0e-8,
            )
        )

    def test_direct_lstm_forecast_is_primary_when_ode_disabled(self):
        lg = logging.getLogger("unit_test_direct_forecast_revm")
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "device": "cpu",
            "seq_len": 16,
            "stride": 8,
            "batch_size": 2,
            "lstm_hidden_size": 32,
            "lstm_layers": 1,
            "fc_hidden": 32,
            "t_end_demo": 8.0,
            "dt_demo": 0.04,
            "val_frac": 0.0,
            "validation_window_s": None,
            "forecast_window_s": 1.0,
            "use_motion_feedback": True,
            "forecast_use_ode": False,
            "data_time_start_s": None,
            "data_time_end_s": None,
            "data_trim_final_s": 0.0,
        })
        t, phi, pack = generate_internal_demo(cfg, lg)
        data = prepare_data(t, phi, pack, cfg, lg)
        model = build_model(cfg, data)
        forecast = forecast_roll_region(model, data, cfg)
        self.assertTrue(forecast.get("enabled", False))
        self.assertEqual(forecast.get("method"), "direct_lstm_no_ode")
        self.assertFalse(bool(forecast.get("forecast_ode_enabled", True)))
        start_idx = int(forecast["start_idx"])
        self.assertAlmostEqual(float(forecast["forecast_phi_scaled"][0]), float(data["phi_scaled"][start_idx]), places=12)
        self.assertAlmostEqual(float(forecast["forecast_v_scaled"][0]), float(data["v_est_scaled"][start_idx]), places=12)
        verification = forecast.get("wave_force_verification", {})
        self.assertFalse(bool(verification.get("wave_force_applied_to_forecast_ode", True)))
        self.assertEqual(
            len(np.asarray(forecast["forecast_roll_deg"], dtype=float)),
            len(np.asarray(forecast["index"], dtype=int)),
        )
        self.assertEqual(
            len(np.asarray(forecast["forecast_force_scaled"], dtype=float)),
            len(np.asarray(forecast["index"], dtype=int)),
        )
        self.assertIn("direct_no_ode_forecast", forecast)

    def test_full_ode_integration_uses_source_coefficients_and_first_state_only(self):
        lg = logging.getLogger("unit_test_full_ode_revm")
        cfg = copy.deepcopy(CONFIG)
        cfg.update({
            "device": "cpu",
            "t_end_demo": 4.0,
            "dt_demo": 0.04,
            "seq_len": 16,
            "stride": 16,
            "prediction_stride": 16,
            "batch_size": 4,
            "lstm_hidden_size": 16,
            "lstm_layers": 1,
            "fc_hidden": 16,
            "val_frac": 0.0,
            "validation_window_s": None,
            "forecast_window_s": 0.64,
            "rollout_window_s": 0.64,
            "use_motion_feedback": False,
            "data_time_start_s": None,
            "data_time_end_s": None,
            "data_trim_final_s": 0.0,
            "source_learned_physics_parameters": {
                "c_roll": 0.16,
                "c_quad": 3.0e-05,
                "k_roll": 0.30,
            },
        })
        t, phi, pack = generate_internal_demo(cfg, lg)
        data = prepare_data(t, phi, pack, cfg, lg)
        model = build_model(cfg, data)
        result = integrate_full_ode_wave_train(model, data, cfg, coefficient_source="source")
        coefficients = result["coefficients"]
        self.assertEqual(int(result["total_points"]), len(t))
        self.assertEqual(len(np.asarray(result["integrated_roll_deg"])), len(t))
        expected_source = cfg["source_learned_physics_parameters"]
        self.assertAlmostEqual(float(coefficients["c_roll"]), expected_source["c_roll"], places=12)
        self.assertAlmostEqual(float(coefficients["c_quad"]), expected_source["c_quad"], places=12)
        self.assertAlmostEqual(float(coefficients["k_roll"]), expected_source["k_roll"], places=12)
        first = int(np.asarray(data["run_segments"], dtype=int)[0, 0])
        self.assertAlmostEqual(
            float(np.asarray(result["integrated_phi_scaled"])[first]),
            float(np.asarray(data["phi_scaled"])[first]),
            places=12,
        )
        self.assertTrue(bool(result["run_status"][0]["completed"]))

    def test_best_only_checkpoint_policy_writes_one_lean_checkpoint(self):
        lg = logging.getLogger("unit_test_best_checkpoint_revm")
        lg.addHandler(logging.NullHandler())
        with tempfile.TemporaryDirectory(prefix="revm_checkpoint_test_") as tmp:
            cfg = copy.deepcopy(CONFIG)
            cfg.update({
                "device": "cpu",
                "epochs": 1,
                "t_end_demo": 4.0,
                "dt_demo": 0.04,
                "seq_len": 16,
                "stride": 16,
                "prediction_stride": 16,
                "batch_size": 4,
                "lstm_hidden_size": 16,
                "lstm_layers": 1,
                "fc_hidden": 16,
                "val_frac": 0.0,
                "validation_window_s": None,
                "forecast_window_s": 0.64,
                "rollout_window_s": 0.64,
                "rollout_warmup_epochs": 5,
                "dataloader_num_workers": 0,
                "data_time_start_s": None,
                "data_time_end_s": None,
                "data_trim_final_s": 0.0,
                "checkpoint_dir": tmp,
                "save_checkpoints": True,
                "save_best_checkpoint": True,
                "save_periodic_checkpoints": False,
                "checkpoint_every": 0,
                "save_last_checkpoint": False,
                "save_final_checkpoint": False,
                "final_checkpoint_name": "best.pt",
            })
            t, phi, pack = generate_internal_demo(cfg, lg)
            data = prepare_data(t, phi, pack, cfg, lg)
            train_loader, val_loader = build_loaders(data, cfg, lg)
            model = build_model(cfg, data)
            result = train_model(model, train_loader, val_loader, data, cfg, lg)
            checkpoint_files = sorted(Path(tmp).glob("*.pt"))
            self.assertEqual([path.name for path in checkpoint_files], ["best.pt"])
            self.assertEqual(Path(str(result["best_checkpoint"])).name, "best.pt")
            self.assertEqual(Path(str(result["final_checkpoint"])).name, "best.pt")
            payload = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)
            self.assertEqual(payload.get("checkpoint_kind"), "best_epoch_model_weights")
            self.assertIsNone(payload.get("optimizer"))
            self.assertIsNone(payload.get("scheduler"))

    def test_bayes_trials_are_checkpoint_free_and_default_directory_is_simple(self):
        cfg = build_bayes_trial_cfg(CONFIG, REVM_SOURCE_TRIAL_PARAMS, 1, np.arange(300, dtype=float) * 0.04)
        self.assertEqual(CONFIG["bayes_opt_dir"], "Bayes_RevM_LSTM_ArchitectureDynamics_v1")
        self.assertTrue(bool(CONFIG["bayes_resume"]))
        self.assertIn(
            "revm_lstm_architecture_dynamics_v1",
            CONFIG["bayes_compatible_study_versions"],
        )
        self.assertEqual(CONFIG["bayes_compatible_study_versions"], ["revm_lstm_architecture_dynamics_v1"])
        self.assertFalse(bool(cfg["freeze_physics_coefficients"]))
        self.assertFalse(bool(cfg["save_checkpoints"]))
        self.assertFalse(bool(cfg["save_final_checkpoint"]))
        self.assertEqual(cfg["checkpoint_policy"], "disabled")
        self.assertEqual(int(cfg["checkpoint_every"]), 0)
        self.assertTrue(bool(cfg["bayes_compact_logging"]))
        self.assertEqual(int(cfg["log_every"]), 50)
        self.assertEqual(int(cfg["val_every"]), 5)
        self.assertEqual(int(cfg["bayes_trial_index"]), 1)
        self.assertTrue(bool(cfg["save_plots"]))

    def test_32vcpu_profile_preserves_training_semantics(self):
        self.assertEqual(int(CONFIG["vcpu_limit"]), 32)
        self.assertEqual(int(CONFIG["torch_num_threads"]), 32)
        self.assertEqual(int(CONFIG["torch_num_interop_threads"]), 1)
        self.assertEqual(int(CONFIG["dataloader_num_workers"]), 0)
        self.assertTrue(bool(CONFIG["precompute_windows"]))
        self.assertTrue(bool(CONFIG["precompute_windows_cpu_only"]))
        self.assertEqual(int(CONFIG["inference_batch_size"]), 64)
        self.assertFalse(bool(CONFIG["cpu_large_memory_auto"]))
        self.assertEqual(int(CONFIG["batch_size"]), 16)
        self.assertEqual(int(CONFIG["rollout_every_n_batches"]), 1)
        self.assertEqual(int(CONFIG["rollout_warmup_epochs"]), 0)
        self.assertFalse(bool(CONFIG["use_motion_feedback"]))
        self.assertEqual(
            effective_rollout_batch_size(CONFIG, int(CONFIG["batch_size"]), True),
            1,
        )
        self.assertAlmostEqual(
            effective_rollout_window_s(CONFIG, True), 5.08, places=12
        )
        self.assertEqual(int(CONFIG["bayes_candidate_pool"]), 2048)
        self.assertEqual(int(CONFIG["bayes_prepared_data_cache_max_entries"]), 4)
        self.assertEqual(int(CONFIG["bayes_tensor_cache_max_entries"]), 4)
        self.assertEqual(int(CONFIG["bayes_window_cache_max_entries"]), 8)
        self.assertTrue(bool(CONFIG["bayes_keep_window_cache_between_trials"]))


def run_unit_tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPINNLSTMRevM)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NoPINN LSTM Rev M Comparison: KRISO roll fitting, forecast, and full-wave ODE diagnostics")
    parser.add_argument("--data", type=str, default=None, help="Excel file path. Default: KTHTest73Excel.xlsx.")
    parser.add_argument("--demo", action="store_true", help="Use the internal synthetic demo instead of the default Excel data.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--stride", type=int, default=None, help="Override training stride.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None, help="Override device.")
    parser.add_argument("--vcpu-limit", type=int, default=None, help="Cap CPU use at this many allocated vCPUs. Default: 32.")
    parser.add_argument("--torch-threads", type=int, default=None, help="Set torch CPU intraop threads. Use 0 for auto/all allocated CPUs.")
    parser.add_argument("--torch-interop-threads", type=int, default=None, help="Set torch CPU interop threads. Use 0 for auto.")
    parser.add_argument("--dataloader-workers", type=int, default=None, help="Set DataLoader worker processes. Auto uses CPU workers only for CPU-resident datasets.")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=None, help="Batches prefetched by each DataLoader worker when workers > 0.")
    parser.add_argument("--precompute-windows", action="store_true", help="Cache all sequence windows in RAM before training.")
    parser.add_argument("--cpu-large-memory", action="store_true", help="CPU-only high-RAM profile: larger batches, in-RAM windows, low-overhead loaders, no CUDA-specific optimisers.")
    parser.add_argument("--disable-tf32", action="store_true", help="Disable TF32 matmul/convolution acceleration on supported NVIDIA GPUs.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default=None, help="CUDA AMP dtype. Default is float16.")
    parser.add_argument("--no-fused-adamw", action="store_true", help="Disable fused AdamW even when CUDA supports it.")
    parser.add_argument("--rollout-every-n-batches", type=int, default=None, help="Compute expensive closed-loop rollout loss every N batches. Default is 4.")
    parser.add_argument("--rollout-warmup-epochs", type=int, default=None, help="Skip closed-loop rollout loss for this many initial epochs. Default is 30.")
    parser.add_argument("--rollout-eval", action="store_true", help="Include expensive rollout losses during validation/evaluation.")
    parser.add_argument("--data-time-start-s", type=float, default=None, help="Optional start time for truncating data before training. Default uses the file start.")
    parser.add_argument("--data-time-end-s", type=float, default=None, help="Optional end time for truncating data before training. Default uses the file end.")
    parser.add_argument("--data-trim-final-s", type=float, default=None, help="Trim this many seconds from the final loaded sample before splitting. Default is 2.0.")
    parser.add_argument("--excel-run-sheets", type=str, default=None,
                        help="Comma-separated workbook sheets to load as independent runs. Default: Active.")
    parser.add_argument("--forecast-sheet", type=str, default=None, help="Sheet/run name used for the rolling forecast holdout. Default: Active.")
    parser.add_argument("--training-sheets", type=str, default=None,
                        help="Comma-separated sheet/run names allowed to contribute training windows. Default: Active.")
    parser.add_argument(
        "--validation-window-s",
        type=float,
        default=None,
        help="Optional seconds-based validation-window override immediately before the forecast. Default uses --val-frac.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Contiguous validation fraction of the complete data when --validation-window-s is unset. Default is 0.15.",
    )
    parser.add_argument(
        "--forecast-frac",
        type=float,
        default=None,
        help="Fraction at the end of the data hidden for rolling forecast. Default is 0.10.",
    )
    parser.add_argument(
        "--forecast-window-s",
        type=float,
        default=None,
        help="Optional seconds-based forecast-window override; supersedes --forecast-frac.",
    )
    parser.add_argument(
        "--forecast-start-s",
        type=float,
        default=None,
        help="Optional explicit forecast start time; by default the last 15%% of data is used.",
    )
    parser.add_argument("--training-selection-metric", choices=["forecast_r2", "r2", "loss", "direct_forecast"], default=None,
                        help="Metric used to select best checkpoints. forecast_r2 uses the hidden forecast window.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument(
        "--exploratory-assessment",
        action="store_true",
        help="Run the 35/15/50 through 80/15/5 data/forecast-length assessment and export a CSV summary.",
    )
    parser.add_argument(
        "--seed-sweep-5pct",
        action="store_true",
        help=f"Run {len(SEED_SWEEP_5PCT_SEEDS)} standard trainings with unchanged input data and varied starting/training seeds.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from a compatible Rev G/Rev H/Rev M checkpoint.")
    parser.add_argument(
        "--integrate-full-ode",
        action="store_true",
        help="After the normal training/resume path, integrate the learned-force roll ODE over every complete loaded wave train.",
    )
    parser.add_argument(
        "--full-ode-coefficients",
        choices=["source", "model"],
        default=None,
        help="Coefficients for --integrate-full-ode: source uses embedded source-trial terminal values (default); model uses current post-training values.",
    )
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests and exit.")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run: fewer epochs and smaller model.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint writing.")
    parser.add_argument("--no-position-features", action="store_true", help="Disable X/Y position features, but keep encounter shift if positions are present.")
    parser.add_argument("--no-encounter-shift", action="store_true", help="Disable XY encounter-time correction.")
    parser.add_argument("--use-motion-feedback", action="store_true", help="Include delayed measured roll/rate as input features for comparison runs.")
    parser.add_argument("--no-motion-feedback", action="store_true", help="Disable delayed roll/rate history input features.")
    parser.add_argument(
        "--motion-feedback-rollout-window-s",
        type=float,
        default=None,
        help="Bound the gradient-retaining closed-loop feedback rollout horizon; does not shorten the final forecast.",
    )
    parser.add_argument(
        "--motion-feedback-rollout-batch-size",
        type=int,
        default=None,
        help="Sub-batch used only for memory-intensive feedback rollout losses; ordinary fitting keeps --batch-size.",
    )
    parser.add_argument("--bayes-opt", action="store_true", help="Run Bayesian hyperparameter optimisation instead of one standard training run.")
    parser.add_argument("--bayes-trials", type=int, default=None, help="Maximum Bayes trials. Default is a high cap so the target metric controls stopping.")
    parser.add_argument("--bayes-init-points", type=int, default=None, help="Random initial Bayes trials before expected improvement is used.")
    parser.add_argument("--bayes-trial-epochs", type=int, default=None, help="Epochs per Bayes trial. Default is 150.")
    parser.add_argument("--bayes-candidate-pool", type=int, default=None, help="Candidate pool size for expected-improvement proposals.")
    parser.add_argument("--bayes-ei-jitter", type=float, default=None, help="Expected-improvement exploration jitter.")
    parser.add_argument("--bayes-r2-objective-weight", type=float, default=None, help="Bayes acquisition weight on forecast R2.")
    parser.add_argument("--bayes-rmse-objective-weight", type=float, default=None, help="Bayes acquisition penalty on forecast RMS/RMSE error.")
    parser.add_argument("--bayes-fit-r2-objective-weight", type=float, default=None, help="Bayes acquisition weight on full teacher-forced fit R2.")
    parser.add_argument("--bayes-fit-rmse-objective-weight", type=float, default=None, help="Bayes acquisition penalty on full-fit RMS/RMSE error.")
    parser.add_argument("--bayes-phase-objective-weight", type=float, default=None, help="Bayes acquisition penalty on absolute forecast phase lag.")
    parser.add_argument("--bayes-amplitude-objective-weight", type=float, default=None, help="Bayes acquisition penalty when forecast amplitude underfits measured amplitude.")
    parser.add_argument("--bayes-extrema-objective-weight", type=float, default=None,
                        help="Bayes acquisition penalty on forecast peak/trough relative error.")
    parser.add_argument("--bayes-turning-objective-weight", type=float, default=None, help="Bayes acquisition penalty on rollout turning-point timing loss.")
    parser.add_argument("--bayes-physics-objective-weight", type=float, default=None, help="Penalty weight for train physics residual in Bayes acquisition.")
    parser.add_argument("--bayes-kinematic-objective-weight", type=float, default=None, help="Penalty weight for train kinematic residual in Bayes acquisition.")
    parser.add_argument("--bayes-target-r2", type=float, default=None, help="Forecast R2 component of the joint fit/forecast stop target.")
    parser.add_argument("--bayes-target-rms-error-deg", "--bayes-target-rmse-deg", dest="bayes_target_rms_error_deg",
                        type=float, default=None,
                        help="Forecast RMSE/RMS-error component of the joint fit/forecast stop target.")
    parser.add_argument("--bayes-target-fit-r2", type=float, default=None,
                        help="Full-fit R2 target that must also be satisfied before Bayes stops.")
    parser.add_argument("--bayes-target-fit-rms-error-deg", type=float, default=None,
                        help="Full-fit RMS-error target that must also be satisfied before Bayes stops.")
    parser.add_argument("--bayes-target-full-ode-r2", type=float, default=None,
                        help="Alternative full-wave ODE R2 stop target (default: 0.95).")
    parser.add_argument("--bayes-target-full-ode-rms-error-deg", "--bayes-target-full-ode-rmse-deg",
                        dest="bayes_target_full_ode_rms_error_deg", type=float, default=None,
                        help="Alternative full-wave ODE RMSE/RMS-error stop target in degrees (default: 0.15).")
    parser.add_argument("--bayes-output-dir", type=str, default=None, help="Directory for Bayes trial outputs and final best run.")
    parser.add_argument("--bayes-no-resume", action="store_true", help="Do not seed/resume Bayes from an existing output directory.")
    parser.add_argument("--bayes-checkpoint-every", type=int, default=None,
                        help="Deprecated compatibility option; Rev M does not write intermediate Bayes checkpoints.")
    parser.add_argument("--bayes-save-trial-checkpoints", action="store_true",
                        help="Deprecated compatibility option; individual Rev M Bayes trials remain checkpoint-free.")
    parser.add_argument("--bayes-no-retrain", action="store_true", help="Skip retraining the best Bayes configuration at the end.")
    parser.add_argument("--bayes-no-data-cache", action="store_true", help="Disable shared prepared-data/tensor caches between Bayes trials.")
    parser.add_argument("--grid-opt", action="store_true",
                        help="Run deterministic grid search over the locked Rev M spectral/rollout parameter subset.")
    parser.add_argument("--grid-output-dir", type=str, default=None,
                        help="Directory for grid-search trial outputs and best-grid hyperparameter JSON.")
    parser.add_argument("--grid-max-trials", type=int, default=None,
                        help="Maximum new grid trials to run this invocation. Default 0 means exhaustive.")
    parser.add_argument("--grid-step", type=float, default=None,
                        help="Grid step for all unlocked parameters. Default is 0.25.")
    parser.add_argument("--grid-no-resume", action="store_true",
                        help="Ignore any existing grid summary in --grid-output-dir.")
    parser.add_argument("--grid-no-trial-plots", action="store_true",
                        help="Disable per-trial plot generation during grid search.")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(cfg)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.seq_len is not None:
        cfg["seq_len"] = int(args.seq_len)
    if args.stride is not None:
        cfg["stride"] = int(args.stride)
    if args.lr is not None:
        cfg["learning_rate"] = float(args.lr)
    if args.device is not None:
        cfg["device"] = args.device
    if args.vcpu_limit is not None:
        cfg["vcpu_limit"] = int(max(1, args.vcpu_limit))
    if args.torch_threads is not None:
        cfg["torch_num_threads"] = int(args.torch_threads)
    if args.torch_interop_threads is not None:
        cfg["torch_num_interop_threads"] = int(args.torch_interop_threads)
    if args.dataloader_workers is not None:
        cfg["dataloader_num_workers"] = int(args.dataloader_workers)
    if args.dataloader_prefetch_factor is not None:
        cfg["dataloader_prefetch_factor"] = int(args.dataloader_prefetch_factor)
    if args.precompute_windows:
        cfg["precompute_windows"] = True
    if args.disable_tf32:
        cfg["cuda_allow_tf32"] = False
        cfg["float32_matmul_precision"] = "highest"
    if args.no_amp:
        cfg["use_amp"] = False
    if args.amp_dtype is not None:
        cfg["amp_dtype"] = args.amp_dtype
    if args.no_fused_adamw:
        cfg["use_fused_adamw"] = False
    if args.rollout_every_n_batches is not None:
        cfg["rollout_every_n_batches"] = int(max(1, args.rollout_every_n_batches))
    if args.rollout_warmup_epochs is not None:
        cfg["rollout_warmup_epochs"] = int(max(0, args.rollout_warmup_epochs))
    if args.rollout_eval:
        cfg["rollout_eval"] = True
    detected_memory_gb = physical_memory_gb()
    if detected_memory_gb is not None:
        cfg["physical_memory_gb"] = float(detected_memory_gb)
    requested_device = str(cfg.get("device", "auto")).strip().lower()
    auto_cpu_large_memory = (
        not args.cpu_large_memory
        and bool(cfg.get("cpu_large_memory_auto", True))
        and str(cfg.get("runtime_profile", "auto")).strip().lower() == "auto"
        and sys.platform.startswith("linux")
        and requested_device in {"auto", "cpu"}
        and not torch.cuda.is_available()
        and detected_memory_gb is not None
        and detected_memory_gb >= float(cfg.get("cpu_large_memory_min_ram_gb", 48.0))
    )
    if args.cpu_large_memory or auto_cpu_large_memory:
        cfg["runtime_profile"] = "cpu_large_memory"
        cfg["cpu_large_memory_auto_applied"] = bool(auto_cpu_large_memory)
        cfg["device"] = "cpu"
        cfg["use_amp"] = False
        cfg["use_fused_adamw"] = False
        cfg["precompute_windows"] = True
        cfg["cuda_benchmark"] = False
        cfg["cuda_allow_tf32"] = False
        if args.torch_interop_threads is None:
            cfg["torch_num_interop_threads"] = 1
        if args.batch_size is None:
            cfg["batch_size"] = int(max(int(cfg.get("batch_size", 32)), int(cfg.get("cpu_large_memory_min_batch_size", 64))))
        if args.dataloader_workers is None:
            cfg["dataloader_num_workers"] = 0
        if args.dataloader_prefetch_factor is None:
            cfg["dataloader_prefetch_factor"] = 2
        if args.rollout_every_n_batches is None:
            cfg["rollout_every_n_batches"] = 16
        if args.rollout_warmup_epochs is None:
            cfg["rollout_warmup_epochs"] = 30
        if args.bayes_candidate_pool is None:
            cfg["bayes_candidate_pool"] = int(min(int(cfg.get("bayes_candidate_pool", 2048)), 1024))
    if args.data_time_start_s is not None:
        cfg["data_time_start_s"] = float(args.data_time_start_s)
    if args.data_time_end_s is not None:
        cfg["data_time_end_s"] = float(args.data_time_end_s)
    if args.data_trim_final_s is not None:
        cfg["data_trim_final_s"] = float(max(0.0, args.data_trim_final_s))
    if bool(cfg.get("disable_data_time_window", False)):
        cfg["data_time_start_s"] = None
        cfg["data_time_end_s"] = None
    if args.excel_run_sheets is not None:
        cfg["excel_run_sheets"] = [part.strip() for part in args.excel_run_sheets.split(",") if part.strip()]
    if args.forecast_sheet is not None:
        cfg["forecast_sheet"] = str(args.forecast_sheet).strip()
    if args.training_sheets is not None:
        cfg["training_sheets"] = [part.strip() for part in args.training_sheets.split(",") if part.strip()]
    if args.validation_window_s is not None:
        cfg["validation_window_s"] = float(args.validation_window_s)
    if args.val_frac is not None:
        cfg["val_frac"] = float(args.val_frac)
    if args.forecast_frac is not None:
        cfg["forecast_frac"] = float(args.forecast_frac)
    if args.forecast_window_s is not None:
        cfg["forecast_window_s"] = float(args.forecast_window_s)
    if args.forecast_start_s is not None:
        cfg["forecast_start_s"] = float(args.forecast_start_s)
    if args.training_selection_metric is not None:
        cfg["training_selection_metric"] = args.training_selection_metric
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.full_ode_coefficients is not None:
        cfg["full_ode_coefficient_source"] = str(args.full_ode_coefficients)
    if args.no_checkpoints:
        cfg["save_checkpoints"] = False
    if args.no_position_features:
        cfg["use_position_features"] = False
    if args.no_encounter_shift:
        cfg["apply_position_encounter_shift"] = False
    if args.use_motion_feedback:
        cfg["use_motion_feedback"] = True
    if args.no_motion_feedback:
        cfg["use_motion_feedback"] = False
    if args.motion_feedback_rollout_window_s is not None:
        cfg["motion_feedback_rollout_window_s"] = float(
            max(0.0, args.motion_feedback_rollout_window_s)
        )
    if args.motion_feedback_rollout_batch_size is not None:
        cfg["motion_feedback_rollout_batch_size"] = int(
            max(1, args.motion_feedback_rollout_batch_size)
        )
    if args.bayes_trials is not None:
        cfg["bayes_trials"] = int(args.bayes_trials)
    if args.bayes_init_points is not None:
        cfg["bayes_init_points"] = int(args.bayes_init_points)
    if args.bayes_trial_epochs is not None:
        cfg["bayes_trial_epochs"] = int(args.bayes_trial_epochs)
    if args.bayes_candidate_pool is not None:
        cfg["bayes_candidate_pool"] = int(args.bayes_candidate_pool)
    if args.bayes_ei_jitter is not None:
        cfg["bayes_ei_jitter"] = float(args.bayes_ei_jitter)
    if args.bayes_r2_objective_weight is not None:
        cfg["bayes_r2_objective_weight"] = float(args.bayes_r2_objective_weight)
    if args.bayes_rmse_objective_weight is not None:
        cfg["bayes_rmse_objective_weight"] = float(args.bayes_rmse_objective_weight)
    if args.bayes_fit_r2_objective_weight is not None:
        cfg["bayes_fit_r2_objective_weight"] = float(args.bayes_fit_r2_objective_weight)
    if args.bayes_fit_rmse_objective_weight is not None:
        cfg["bayes_fit_rmse_objective_weight"] = float(args.bayes_fit_rmse_objective_weight)
    if args.bayes_phase_objective_weight is not None:
        cfg["bayes_phase_objective_weight"] = float(args.bayes_phase_objective_weight)
    if args.bayes_amplitude_objective_weight is not None:
        cfg["bayes_amplitude_objective_weight"] = float(args.bayes_amplitude_objective_weight)
    if args.bayes_extrema_objective_weight is not None:
        cfg["bayes_extrema_objective_weight"] = float(args.bayes_extrema_objective_weight)
    if args.bayes_turning_objective_weight is not None:
        cfg["bayes_turning_objective_weight"] = float(args.bayes_turning_objective_weight)
    if args.bayes_physics_objective_weight is not None:
        cfg["bayes_physics_objective_weight"] = float(args.bayes_physics_objective_weight)
    if args.bayes_kinematic_objective_weight is not None:
        cfg["bayes_kinematic_objective_weight"] = float(args.bayes_kinematic_objective_weight)
    if args.bayes_target_r2 is not None:
        cfg["bayes_target_r2"] = float(args.bayes_target_r2)
    if args.bayes_target_rms_error_deg is not None:
        cfg["bayes_target_rms_error_deg"] = float(args.bayes_target_rms_error_deg)
    if args.bayes_target_fit_r2 is not None:
        cfg["bayes_target_fit_r2"] = float(args.bayes_target_fit_r2)
    if args.bayes_target_fit_rms_error_deg is not None:
        cfg["bayes_target_fit_rms_error_deg"] = float(args.bayes_target_fit_rms_error_deg)
    if args.bayes_target_full_ode_r2 is not None:
        cfg["bayes_target_full_ode_r2"] = float(args.bayes_target_full_ode_r2)
    if args.bayes_target_full_ode_rms_error_deg is not None:
        cfg["bayes_target_full_ode_rms_error_deg"] = float(args.bayes_target_full_ode_rms_error_deg)
    if args.bayes_output_dir is not None:
        cfg["bayes_opt_dir"] = args.bayes_output_dir
    if args.bayes_no_resume:
        cfg["bayes_resume"] = False
    # Retain these parsed options for CLI compatibility, but keep Rev M Bayes
    # trials checkpoint-free regardless of their values.
    cfg["bayes_checkpoint_every"] = 0
    cfg["bayes_save_trial_checkpoints"] = False
    if args.bayes_no_retrain:
        cfg["bayes_retrain_best"] = False
    if args.bayes_no_data_cache:
        cfg["bayes_reuse_prepared_data"] = False
        cfg["bayes_reuse_loader_cache"] = False
    if args.grid_output_dir is not None:
        cfg["grid_opt_dir"] = args.grid_output_dir
    if args.grid_max_trials is not None:
        cfg["grid_max_trials"] = int(max(0, args.grid_max_trials))
    if args.grid_step is not None:
        cfg["grid_step"] = float(max(1.0e-12, args.grid_step))
    if args.grid_no_resume:
        cfg["grid_resume"] = False
    if args.grid_no_trial_plots:
        cfg["grid_save_trial_plots"] = False
    if args.bayes_opt:
        cfg["log_dir"] = os.path.join(str(cfg["bayes_opt_dir"]), "logs")
    if args.grid_opt:
        cfg["log_dir"] = os.path.join(str(cfg["grid_opt_dir"]), "logs")
    if args.exploratory_assessment:
        cfg["log_dir"] = os.path.join(str(cfg["output_dir"]), str(cfg["exploratory_assessment_dir"]), "logs")
    if args.seed_sweep_5pct:
        cfg["seed_sweep_source_file"] = Path(__file__).name
        cfg["log_dir"] = os.path.join(str(cfg["output_dir"]), "seed_sweep_5pct", Path(__file__).stem, "logs")
    if args.quick:
        cfg.update({
            "epochs": 2,
            "seq_len": 32,
            "stride": 32,
            "batch_size": 4,
            "t_end_demo": 12.0,
            "lstm_hidden_size": 32,
            "lstm_layers": 1,
            "fc_hidden": 48,
            "prediction_stride": 32,
            "log_every": 1,
            "checkpoint_every": 1,
            "rollout_every_n_batches": 1,
            "rollout_warmup_epochs": 0,
        })
        if args.demo:
            cfg["data_time_start_s"] = None
            cfg["data_time_end_s"] = None
    for key in ("output_dir", "checkpoint_dir", "log_dir", "bayes_opt_dir", "grid_opt_dir"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = str(resolve_runtime_path(str(cfg[key])))
    return cfg


def main() -> None:
    args = parse_args()
    if args.run_tests:
        run_unit_tests()
        return

    cfg = apply_cli_overrides(CONFIG, args)
    set_global_seed(int(cfg["seed"]))
    configure_torch_threads(cfg, None)
    lg = setup_logging(str(cfg["log_dir"]))
    configure_torch_threads(cfg, lg)
    lg.info("Starting NoPINN LSTM Rev M Comparison training run")
    lg.info(
        "Runtime profile: %s | cpu_large_memory_auto_applied=%s | physical_memory_gb=%s",
        cfg.get("runtime_profile", "auto"),
        bool(cfg.get("cpu_large_memory_auto_applied", False)),
        f"{float(cfg['physical_memory_gb']):.2f}" if "physical_memory_gb" in cfg else "unknown",
    )
    lg.info("Post-training rolling forecast window: %s", forecast_window_description(cfg))

    if args.demo:
        t, phi, pack = generate_internal_demo(cfg, lg)
    else:
        data_file = args.data or str(cfg.get("default_data_file", "KTHTest73Excel.xlsx"))
        t, phi, pack = load_excel(data_file, lg, cfg)

    search_modes = [bool(args.bayes_opt), bool(args.grid_opt), bool(args.seed_sweep_5pct)]
    if sum(search_modes) > 1:
        raise ValueError("--bayes-opt, --grid-opt, and --seed-sweep-5pct are mutually exclusive.")
    if args.exploratory_assessment and any(search_modes):
        raise ValueError("--exploratory-assessment cannot be combined with --bayes-opt, --grid-opt, or --seed-sweep-5pct.")

    if args.seed_sweep_5pct:
        if args.resume:
            raise ValueError("--resume is not supported during --seed-sweep-5pct; each seed is trained as an independent run.")
        summary = run_seed_sweep_5pct(
            t,
            phi,
            pack,
            cfg,
            lg,
            run_standard_training_once,
            include_full_ode=bool(args.integrate_full_ode),
            full_ode_coefficient_source=str(cfg.get("full_ode_coefficient_source", "source")),
        )
        lg.info("Finished Rev M seed-only sweep with unchanged input data.")
        lg.info("  Seeds: %s", summary.get("seeds", list(SEED_SWEEP_5PCT_SEEDS)))
        lg.info("  Summary CSV: %s", summary.get("summary_csv"))
        lg.info("  Summary JSON: %s", summary.get("summary_json"))
        return

    if args.grid_opt:
        if args.integrate_full_ode:
            raise ValueError("--integrate-full-ode is available on the standard train/resume path, not during --grid-opt.")
        summary = run_grid_search(t, phi, pack, cfg, lg)
        lg.info("Finished Rev M deterministic grid search.")
        best = summary.get("best_trial")
        if isinstance(best, dict):
            lg.info("Best grid trial: %s", best.get("trial_index"))
            lg.info("  Params: %s", best.get("params"))
            lg.info("  Forecast R2: %.6f", finite_or(best.get("forecast_r2"), float("nan")))
            lg.info("  Forecast RMSE: %.6f deg", finite_or(best.get("forecast_rms_error_deg"), float("nan")))
            lg.info("  Fit R2: %.6f", finite_or(best.get("fit_r2"), float("nan")))
            lg.info("  Fit RMSE: %.6f deg", finite_or(best.get("fit_rms_error_deg"), float("nan")))
        return

    if args.bayes_opt:
        if args.integrate_full_ode:
            raise ValueError("--integrate-full-ode is available on the standard train/resume path, not during --bayes-opt.")
        summary = run_bayesian_optimisation(
            t, phi, pack, cfg, lg,
            retrain_best=bool(cfg.get("bayes_retrain_best", True)),
        )
        lg.info("Finished Rev M Bayesian optimisation.")
        best = summary.get("best_trial")
        if isinstance(best, dict):
            lg.info("Best composite Bayes trial: %s", best.get("trial_index"))
            lg.info("  Composite-trial forecast R2: %.6f", finite_or(best.get("forecast_r2"), float("nan")))
            lg.info("  Composite-trial forecast RMSE: %.6f deg", finite_or(best.get("forecast_rms_error_deg"), float("nan")))
            lg.info("  Composite-trial fit R2: %.6f", finite_or(best.get("fit_r2"), float("nan")))
            lg.info("  Composite-trial fit RMSE: %.6f deg", finite_or(best.get("fit_rms_error_deg"), float("nan")))
        best_r2 = summary.get("best_forecast_r2_trial")
        if isinstance(best_r2, dict):
            lg.info(
                "Best forecast R2: trial %s | R2=%.6f | RMSE=%.6f deg",
                best_r2.get("trial_index"),
                finite_or(best_r2.get("forecast_r2"), float("nan")),
                finite_or(best_r2.get("forecast_rms_error_deg"), float("nan")),
            )
        best_rmse = summary.get("best_forecast_rmse_trial")
        if isinstance(best_rmse, dict):
            lg.info(
                "Best forecast RMSE: trial %s | RMSE=%.6f deg | R2=%.6f",
                best_rmse.get("trial_index"),
                finite_or(best_rmse.get("forecast_rms_error_deg"), float("nan")),
                finite_or(best_rmse.get("forecast_r2"), float("nan")),
            )
        return

    if args.exploratory_assessment:
        if args.resume:
            raise ValueError("--resume is not supported during --exploratory-assessment; each split is trained as an independent run.")
        summary = run_exploratory_assessment(
            t,
            phi,
            pack,
            cfg,
            lg,
            include_full_ode=bool(args.integrate_full_ode),
            full_ode_coefficient_source=str(cfg.get("full_ode_coefficient_source", "source")),
        )
        lg.info("Finished Rev M exploratory assessment.")
        lg.info("  Summary CSV: %s", summary.get("summary_csv"))
        lg.info("  Summary JSON: %s", summary.get("summary_json"))
        best_r2 = summary.get("best_forecast_r2_run")
        if isinstance(best_r2, dict):
            lg.info(
                "  Best forecast R2: %s | R2=%.6f | RMSE=%.6f deg",
                best_r2.get("split_label"),
                finite_or(best_r2.get("forecast_r2"), float("nan")),
                finite_or(best_r2.get("forecast_rmse_deg"), float("nan")),
            )
        best_rmse = summary.get("best_forecast_rmse_run")
        if isinstance(best_rmse, dict):
            lg.info(
                "  Best forecast RMSE: %s | RMSE=%.6f deg | R2=%.6f",
                best_rmse.get("split_label"),
                finite_or(best_rmse.get("forecast_rmse_deg"), float("nan")),
                finite_or(best_rmse.get("forecast_r2"), float("nan")),
            )
        return

    resume_path = str(resolve_runtime_path(args.resume)) if args.resume else None
    paths, _, _ = run_standard_training_once(
        t,
        phi,
        pack,
        cfg,
        lg,
        resume=resume_path,
        include_full_ode=bool(args.integrate_full_ode),
        full_ode_coefficient_source=str(cfg.get("full_ode_coefficient_source", "source")),
    )
    lg.info("Finished Rev M run.")
    for name, path in paths.items():
        lg.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()
