# 10-fold cross-evaluation of the translucent IMf variants (IMtf / IMts).
# Model is discovered on each train split; all metrics are computed on the held-out test split.
import warnings
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import pandas as pd
import os
import random
import math
from pathlib import Path
import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.util import constants
import time
import sys
import traceback

if os.name != 'nt':
    sys.path.append("/home/eliasmullers/Desktop/thesis/eval_fall")
else:
    sys.path.append(
        "C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\"
        "TranslucentActivityRelationships-main"
    )

from new_eval.utils.make_rooted import (
    add_artificial_start_and_end_activities_translucent,
    get_alignment_fitness_with_processtree,
)
from translucent_precision.main import translucent_precision_score_eval_version as translucent_precision_score
from translucent_fitness.fitness import calculate_log_fitness
from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
constants.DEFAULT_LP_SOLVER = "gurobi"

LOG_NAMES = ["Sepsis", "road_traffic_fine", "hospital_billing"]
TIMEOUT   = 7200
N_FOLDS   = 10
N_CPUS    = mp.cpu_count()
N_WORKERS = N_CPUS // 2 + 4

# ---------------------------------------------------------------------------
# Error file helper
# ---------------------------------------------------------------------------

def _write_error_file(context: str, exc: BaseException) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    err_path = Path(f"error_{context}_{ts}.txt")
    with open(err_path, "w") as fh:
        fh.write(f"=== CRASH REPORT ===\n")
        fh.write(f"Context : {context}\n")
        fh.write(f"Time    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Error   : {type(exc).__name__}: {exc}\n\n")
        fh.write("--- Traceback ---\n")
        fh.write(traceback.format_exc())
    print(f"[ERROR] Crash report written to: {err_path}", flush=True)

# ---------------------------------------------------------------------------
# Parameter-config generators  (unchanged from original)
# ---------------------------------------------------------------------------

def generate_parameter_configs():
    parameter_list = []
    translucent_self_loops            = [False]
    strict_end_activities             = [True]
    parallel_end_activities_heuristic = [False]
    remove_arcs_heuristic = ["dependency_score", "support", "confidence", "exclusive_choice_frequency"]
    add_arcs_heuristic    = ["dependency_score", "support", "confidence", "parallel_relationship_frequency"]
    for self_loops in translucent_self_loops:
        for strict_end in strict_end_activities:
            for parallel_heuristic in parallel_end_activities_heuristic:
                for remove_heuristic in remove_arcs_heuristic:
                    for add_heuristic in add_arcs_heuristic:
                        if remove_heuristic == add_heuristic and remove_heuristic is not False:
                            continue
                        parameter_list.append({
                            "translucent_self_loops": self_loops,
                            "strict_end_activities": strict_end,
                            "parallel_end_activities_heuristic": parallel_heuristic,
                            "remove_arcs_heuristics": remove_heuristic,
                            "add_arcs_heuristics": add_heuristic,
                        })
    return parameter_list

def generate_minor_heuristics_parameter_configs():
    parameter_list = []
    for self_loops in [True, False]:
        for strict_end in [True, False]:
            for parallel_heuristic in [True, False]:
                parameter_list.append({
                    "translucent_self_loops": self_loops,
                    "strict_end_activities": strict_end,
                    "parallel_end_activities_heuristic": parallel_heuristic,
                    "remove_arcs_heuristics": False,
                    "add_arcs_heuristics": False,
                })
    return parameter_list

def prettify_config_for_file_name(config):
    return "_".join([f"{k}_{v}" for k, v in config.items()])

# ---------------------------------------------------------------------------
# Timeout infrastructure
# ---------------------------------------------------------------------------

def _run_fn_in_process(target, args, result_value: Value) -> bool:
    def _wrapper(rv, fn, fn_args):
        rv.value = fn(*fn_args)
    p = Process(target=_wrapper, args=(result_value, target, args))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(); return False
    p.close(); return True

def _run_returning_pair_in_process(target, args) -> "tuple | None":
    q = Queue()
    def _wrapper(out_q, fn, fn_args):
        out_q.put(fn(*fn_args))
    p = Process(target=_wrapper, args=(q, target, args))
    p.start(); p.join(TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join(); return None
    p.close(); return q.get()

def _precision_alignments_wrapper(log, net, im, fm) -> float:
    return pm4py.conformance.precision_alignments(log, net, im, fm)

def _calculate_log_fitness_wrapper(log, net, im, fm) -> float:
    return calculate_log_fitness(log, net, im, fm)

def _translucent_precision_score_wrapper(log, net, im, fm):
    return translucent_precision_score(log, net, im, fm)

# ---------------------------------------------------------------------------
# PNML export helpers  (fold-aware)
# ---------------------------------------------------------------------------

def _pnml_path(log_name: str, noise_label: str, config: dict,
               variant: str, f: float, fold_idx: int) -> Path:
    folder = Path(
        f"new_eval/translucent_datasets/{log_name}/pnml/translucent_delta_10fold/{noise_label}/"
        f"{prettify_config_for_file_name(config)}/fold{fold_idx}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    f_str = f"{f:.2f}".replace(".", "_")
    return folder / f"net_{variant}_f{f_str}.pnml"

def _write_pnml(net, im, fm, path: Path) -> None:
    try:
        pm4py.write_pnml(net, im, fm, str(path))
    except Exception as exc:
        print(f" [WARN] Could not write PNML to {path}: {exc}")

# ---------------------------------------------------------------------------
# 10-fold splitter
# ---------------------------------------------------------------------------

def split_log_into_folds(log, n_folds: int = N_FOLDS):
    """Return list of (fold_idx, train_log, test_log). Shuffle is seeded (42)."""
    cases = list(log)
    random.Random(42).shuffle(cases)
    fold_size = math.ceil(len(cases) / n_folds)
    kw = dict(attributes=log.attributes, extensions=log.extensions,
              classifiers=log.classifiers, omni_present=log.omni_present,
              properties=log.properties)
    result = []
    for k in range(n_folds):
        test_cases  = cases[k * fold_size : (k + 1) * fold_size]
        train_cases = cases[:k * fold_size] + cases[(k + 1) * fold_size:]
        result.append((k, EventLog(train_cases, **kw), EventLog(test_cases, **kw)))
    return result

# ---------------------------------------------------------------------------
# Single-fold + single-filter evaluation
# ---------------------------------------------------------------------------

def evaluate_single_filter(f: float, train_log, test_log,
                            log_name: str, noise_type: str,
                            fold_idx: int, parameters: dict = None) -> tuple:
    """Discovers on train_log with filter f; all metrics evaluated on test_log."""
    parameters = parameters or {}

    # --- Discovery (on train split) ---------------------------------------
    start = time.time()
    net_tf, im_tf, fm_tf, fallthrough_count_tf = discover_petri_net(
        train_log,
        {"translucent_variant": "IMtf", "tDFG_fall_through": True} | parameters,
        noise_threshold=f,
    )
    duration_tf = time.time() - start

    start = time.time()
    net_ts, im_ts, fm_ts, fallthrough_count_ts = discover_petri_net(
        train_log,
        {"translucent_variant": "IMts", "tDFG_fall_through": False} | parameters,
        noise_threshold=f,
    )
    duration_ts = time.time() - start

    # --- Write Petri nets to PNML (train-derived nets) --------------------
    _write_pnml(net_tf, im_tf, fm_tf, _pnml_path(log_name, noise_type, parameters, "IMtf", f, fold_idx))
    _write_pnml(net_ts, im_ts, fm_ts, _pnml_path(log_name, noise_type, parameters, "IMts", f, fold_idx))
    """
    # --- precision_alignments on test_log ---------------------------------
    _prec_tf = Value("d", -1.0)
    if not _run_fn_in_process(_precision_alignments_wrapper, (test_log, net_tf, im_tf, fm_tf), _prec_tf):
        print(f" [TIMEOUT] precision_alignments IMtf | log={log_name}, noise={noise_type}, fold={fold_idx}, f={f}")
    precision_tf = _prec_tf.value

    _prec_ts = Value("d", -1.0)
    if not _run_fn_in_process(_precision_alignments_wrapper, (test_log, net_ts, im_ts, fm_ts), _prec_ts):
        print(f" [TIMEOUT] precision_alignments IMts | log={log_name}, noise={noise_type}, fold={fold_idx}, f={f}")
    precision_ts = _prec_ts.value

    # --- translucent_precision_score on test_log --------------------------
    pair_tf = _run_returning_pair_in_process(
        _translucent_precision_score_wrapper, (test_log, net_tf, im_tf, fm_tf))
    if pair_tf is None:
        print(f" [TIMEOUT] translucent_precision_score IMtf | log={log_name}, noise={noise_type}, fold={fold_idx}, f={f}")
        translucent_precision_tf = -1.0
        fitness_tf = get_alignment_fitness_with_processtree(test_log, net_tf, im_tf, fm_tf)
    else:
        translucent_precision_tf, fitness_tf = pair_tf

    pair_ts = _run_returning_pair_in_process(
        _translucent_precision_score_wrapper, (test_log, net_ts, im_ts, fm_ts))
    if pair_ts is None:
        print(f" [TIMEOUT] translucent_precision_score IMts | log={log_name}, noise={noise_type}, fold={fold_idx}, f={f}")
        translucent_precision_ts = -1.0
        fitness_ts = get_alignment_fitness_with_processtree(test_log, net_ts, im_ts, fm_ts)
    else:
        translucent_precision_ts, fitness_ts = pair_ts
    """
    # normal fitness
    fitness_tf = get_alignment_fitness_with_processtree(test_log, net_tf, im_tf, fm_tf)
    fitness_ts = get_alignment_fitness_with_processtree(test_log, net_ts, im_ts, fm_ts)
    
    # --- calculate_log_fitness on test_log --------------------------------
    _tfit_tf = Value("d", -1.0)
    if not _run_fn_in_process(_calculate_log_fitness_wrapper, (test_log, net_tf, im_tf, fm_tf), _tfit_tf):
        print(f" [TIMEOUT] calculate_log_fitness IMtf | log={log_name}, noise={noise_type}, fold={fold_idx}, f={f}")
    translucent_fitness_tf = _tfit_tf.value

    _tfit_ts = Value("d", -1.0)
    if not _run_fn_in_process(_calculate_log_fitness_wrapper, (test_log, net_ts, im_ts, fm_ts), _tfit_ts):
        print(f" [TIMEOUT] calculate_log_fitness IMts | log={log_name}, noise={noise_type}, fold={fold_idx}, f={f}")
    translucent_fitness_ts = _tfit_ts.value

    # --- F1 scores --------------------------------------------------------
    denom_f1_tf = 0
    f_1_score_tf = 0

    denom_f1_ts = 0
    f_1_score_ts = 0

    denom_tf1_tf = 0
    translucent_f_1_score_tf = 0

    denom_tf1_ts = 0
    translucent_f_1_score_ts = 0

    simplicity_tf = len(net_tf.places) + len(net_tf.transitions) + len(net_tf.arcs)
    simplicity_ts = len(net_ts.places) + len(net_ts.transitions) + len(net_ts.arcs)

    return f, {
        "fold": fold_idx,
        "time_tf": duration_tf, "time_ts": duration_ts, "RAM": 0,
        "fitness_tf": fitness_tf, "precision_tf": 0, "f_1_score_tf": f_1_score_tf,
        "fitness_ts": fitness_ts, "precision_ts": 0, "f_1_score_ts": f_1_score_ts,
        "translucent_fitness_tf": translucent_fitness_tf, "translucent_precision_tf": 0,
        "translucent_fitness_ts": translucent_fitness_ts, "translucent_precision_ts": 0,
        "translucent_f_1_score_tf": translucent_f_1_score_tf, "translucent_f_1_score_ts": translucent_f_1_score_ts,
        "simplicity_tf": simplicity_tf, "simplicity_ts": simplicity_ts, "failed": False,
    }

# ---------------------------------------------------------------------------
# CSV writer worker — one per (log_name, noise_label, config_key)
# ---------------------------------------------------------------------------

def _store_results_worker(result_queue: Queue, results_path: Path) -> None:
    try:
        first_write = not results_path.exists()
        while True:
            item = result_queue.get()
            if item == "Ende":
                break
            f, result_data = item
            row = {"threshold": f, **result_data}
            df = pd.DataFrame([row])
            df.to_csv(results_path, mode="a", header=first_write, index=False)
            first_write = False
    except Exception as exc:
        _write_error_file(f"store_results_{results_path.stem}", exc); raise

# ---------------------------------------------------------------------------
# Flat task dispatch
# ---------------------------------------------------------------------------

def _build_task_list(log_names, noise_types, fold_map, parameter_configs, filter_values):
    tasks = []
    for log_name in log_names:
        for noise_type in noise_types:
            noise_label = noise_type if noise_type else "base"
            for fold_idx, train_log, test_log in fold_map[(log_name, noise_label)]:
                for config in parameter_configs:
                    for f in filter_values:
                        tasks.append((log_name, noise_label, config, f, fold_idx, train_log, test_log))
    return tasks

def _central_task_worker(task_queue: Queue, out_queue: Queue) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            break
        log_name, noise_label, config, f, fold_idx, train_log, test_log = item
        try:
            f_out, result_data = evaluate_single_filter(
                f, train_log, test_log, log_name, noise_label, fold_idx, config)
            out_queue.put((log_name, noise_label, prettify_config_for_file_name(config), f_out, result_data))
        except Exception as exc:
            _write_error_file(f"task_{log_name}_{noise_label}_fold{fold_idx}_f{f}", exc); raise

# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def _load_log_worker(log_name: str, noise_type, path_to_log_fn, out_queue: Queue) -> None:
    try:
        noise_label = noise_type if noise_type else "base"
        path_to_log = path_to_log_fn(log_name)
        log_path = path_to_log / (f"{log_name}_{noise_type}.csv" if noise_type else f"{log_name}_base.csv")
        log = pd.read_csv(log_path)
        log = pm4py.format_dataframe(log, case_id="case:concept:name",
                                     activity_key="concept:name",
                                     timestamp_key="time:timestamp",
                                     timest_format="%Y-%m-%d %H:%M:%S%z")
        log = pm4py.convert_to_event_log(log)
        log = add_artificial_start_and_end_activities_translucent(log)
        out_queue.put((log_name, noise_label, log))
    except Exception as exc:
        _write_error_file(f"load_log_{log_name}_{noise_type or 'base'}", exc); raise

# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(log_names, noise_types, path_to_log_fn, filter_values,
                   parameter_configs, n_folds: int = N_FOLDS):
    total_logs = len(log_names) * len(noise_types)

    print(f"[INIT] Loading {total_logs} event log(s) in parallel...")
    load_queue = Queue(); loaders = []
    for log_name in log_names:
        for noise_type in noise_types:
            p = Process(target=_load_log_worker, args=(log_name, noise_type, path_to_log_fn, load_queue))
            p.start(); loaders.append(p)

    log_map = {}
    for _ in range(total_logs):
        log_name, noise_label, log = load_queue.get()
        log_map[(log_name, noise_label)] = log
        print(f" [{len(log_map)}/{total_logs}] Loaded: {log_name} / {noise_label}")
    for p in loaders:
        p.join()
    print("[INIT] All logs loaded.\n")

    fold_map = {}
    for (log_name, noise_label), log in log_map.items():
        fold_map[(log_name, noise_label)] = split_log_into_folds(log, n_folds)
        print(f"[FOLD] {log_name}/{noise_label}: {n_folds} folds, "
              f"~{len(list(log)) // n_folds} test cases/fold")

    result_queues = {}; writer_processes = {}
    for log_name in log_names:
        for noise_type in noise_types:
            noise_label = noise_type if noise_type else "base"
            for config in parameter_configs:
                key = (log_name, noise_label, prettify_config_for_file_name(config))
                results_folder = Path(
                    f"new_eval/translucent_datasets/{log_name}/results/"
                    f"translucent_delta_10fold/{noise_label}"
                )
                results_folder.mkdir(parents=True, exist_ok=True)
                results_path = results_folder / f"IMf_10fold_{prettify_config_for_file_name(config)}.csv"
                q = Queue(); result_queues[key] = q
                writer = Process(target=_store_results_worker, args=(q, results_path))
                writer.start(); writer_processes[key] = writer

    tasks = _build_task_list(log_names, noise_types, fold_map, parameter_configs, filter_values)
    total_tasks = len(tasks)
    print(f"[INIT] {total_tasks} jobs queued across {N_WORKERS} workers "
          f"({len(log_names)} log(s) x {len(noise_types)} noise type(s) x "
          f"{len(parameter_configs)} config(s) x {n_folds} folds x {len(filter_values)} filter value(s))\n")

    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(N_WORKERS):
        task_queue.put(None)

    central_out = Queue(); workers = []
    for _ in range(N_WORKERS):
        p = Process(target=_central_task_worker, args=(task_queue, central_out))
        p.start(); workers.append(p)

    completed = 0; width = len(str(total_tasks))
    while completed < total_tasks:
        log_name, noise_label, config_key, f, result_data = central_out.get()
        result_queues[(log_name, noise_label, config_key)].put((f, result_data))
        completed += 1
        print(f"[{completed:{width}}/{total_tasks}] log={log_name}, noise={noise_label}, "
              f"fold={result_data['fold']}, f={f}", flush=True)

    for worker in workers:
        worker.join()
    for q in result_queues.values():
        q.put("Ende")
    for writer in writer_processes.values():
        writer.join()

    print(f"\n[DONE] IMf_translucent 10-fold evaluation complete — {total_tasks} jobs finished.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    noise_types       = [False]
    filter_values     = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parameter_configs = generate_parameter_configs()

    def path_to_log_fn(log_name: str) -> Path:
        if os.name == "nt":
            return Path(
                f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\"
                f"TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{log_name}"
            )
        return Path(f"new_eval/translucent_datasets/{log_name}")

    try:
        run_evaluation(LOG_NAMES, noise_types, path_to_log_fn, filter_values, parameter_configs)
    except Exception as exc:
        _write_error_file("main", exc)
        sys.exit(1)