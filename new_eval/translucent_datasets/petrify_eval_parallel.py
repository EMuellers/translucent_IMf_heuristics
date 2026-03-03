# Conducts in parallel the evaluation of the petrify method for the different noise types
import warnings
import traceback
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import pandas as pd
import os
from pathlib import Path
import pm4py
from pm4py.util import constants
from subprocess import CalledProcessError
import time
import sys

if os.name != 'nt':
    sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")
else:
    sys.path.append(
        "C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\"
        "TranslucentActivityRelationships-main"
    )

from new_eval.utils.discover_model_with_regions import discover_net_with_regions_from_rooted_log
from new_eval.utils.make_rooted import (
    add_artificial_start_and_end_activities_translucent,

)
from translucent_precision.main import translucent_precision_score
from translucent_fitness.fitness import calculate_log_fitness
from translucent_discovery.translucent_inductive_miner.translucent_datatype import translucent_log_to_tcl
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

constants.DEFAULT_LP_SOLVER = "gurobi"

LOG_NAMES = ["Sepsis", "hospital_billing", "road_traffic_fine"]

TIMEOUT = 7200  # seconds per individual metric computation
N_CPUS = mp.cpu_count()
N_WORKERS = N_CPUS // 2 + 3

# ---------------------------------------------------------------------------
# Error file helper
# ---------------------------------------------------------------------------

def _write_error_file(context: str, exc: BaseException) -> None:
    """Write a crash report to a timestamped error file."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    err_path = Path(f"error_{context}_{ts}.txt")
    with open(err_path, "w") as fh:
        fh.write("=== CRASH REPORT ===\n")
        fh.write(f"Context : {context}\n")
        fh.write(f"Time    : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Error   : {type(exc).__name__}: {exc}\n\n")
        fh.write("--- Traceback ---\n")
        fh.write(traceback.format_exc())
    print(f"[ERROR] Crash report written to: {err_path}", flush=True)

# ---------------------------------------------------------------------------
# Timeout infrastructure
# ---------------------------------------------------------------------------

def _run_fn_in_process(target, args, result_value: Value) -> bool:
    """Runs target(*args) in a child Process. Returns True if finished within TIMEOUT."""
    def _wrapper(rv, fn, fn_args):
        rv.value = fn(*fn_args)

    p = Process(target=_wrapper, args=(result_value, target, args))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    p.close()
    return True

def _run_returning_value_in_process(target, args) -> object:
    """Runs target(*args) in a child Process via Queue. Returns None on timeout."""
    q = Queue()

    def _wrapper(out_q, fn, fn_args):
        out_q.put(fn(*fn_args))

    p = Process(target=_wrapper, args=(q, target, args))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    p.close()
    return q.get()

def _fitness_alignments_wrapper(log, net, im, fm) -> float:
    return pm4py.conformance.fitness_alignments(log, net, im, fm)["log_fitness"]

def _precision_alignments_wrapper(log, net, im, fm) -> float:
    return pm4py.conformance.precision_alignments(log, net, im, fm)

def _calculate_log_fitness_wrapper(log, net, im, fm) -> float:
    return calculate_log_fitness(log, net, im, fm)

def _translucent_precision_score_wrapper(log, net, im, fm):
    return translucent_precision_score(log, net, im, fm)

# ---------------------------------------------------------------------------
# Single noise-type evaluation
# ---------------------------------------------------------------------------

def evaluate_single_noise_type(log, log_name: str, noise_label: str) -> dict:
    """Evaluates petrify for one (log_name, noise_label). Returns result_dict."""

    tcl_log = translucent_log_to_tcl(log)

    # --- Discovery --------------------------------------------------------
    try:
        start = time.time()
        if os.name == "nt":
            net, im, fm = discover_net_with_regions_from_rooted_log(
                tcl_log, log_name=f"{log_name}_{noise_label}"
            )
            max_memory = 0
        else:
            net, im, fm, max_memory = discover_net_with_regions_from_rooted_log(
                tcl_log, log_name=f"{log_name}_{noise_label}"
            )
        duration = time.time() - start
        print(f"  [DISC] log={log_name}, noise={noise_label} — discovered in {duration:.1f}s", flush=True)
    except CalledProcessError as exc:
        print(f"  [FAIL] Petrify failed for log={log_name}, noise={noise_label}: {exc}", flush=True)
        _write_error_file(f"petrify_{log_name}_{noise_label}", exc)
        return {
            "time": 0, "RAM": 0, "fitness": 0, "precision": 0,
            "translucent_fitness": 0, "translucent_precision": 0,
            "f_1_score": 0, "translucent_f_1_score": 0,
            "simplicity": 0, "failed": True,
        }

    # --- fitness_alignments -----------------------------------------------
    _fit = Value("d", -1.0)
    if not _run_fn_in_process(_fitness_alignments_wrapper, (log, net, im, fm), _fit):
        print(f"  [TIMEOUT] fitness_alignments | log={log_name}, noise={noise_label}", flush=True)
    fitness = _fit.value

    # --- precision_alignments ---------------------------------------------
    _prec = Value("d", -1.0)
    if not _run_fn_in_process(_precision_alignments_wrapper, (log, net, im, fm), _prec):
        print(f"  [TIMEOUT] precision_alignments | log={log_name}, noise={noise_label}", flush=True)
    precision = _prec.value

    # --- translucent_precision_score --------------------------------------
    translucent_precision = _run_returning_value_in_process(
        _translucent_precision_score_wrapper, (log, net, im, fm)
    )
    if translucent_precision is None:
        print(f"  [TIMEOUT] translucent_precision_score | log={log_name}, noise={noise_label}", flush=True)
        translucent_precision = -1.0

    # --- calculate_log_fitness --------------------------------------------
    _tfit = Value("d", -1.0)
    if not _run_fn_in_process(_calculate_log_fitness_wrapper, (log, net, im, fm), _tfit):
        print(f"  [TIMEOUT] calculate_log_fitness | log={log_name}, noise={noise_label}", flush=True)
    translucent_fitness = _tfit.value

    # --- F1 scores --------------------------------------------------------
    denom_f1 = fitness + precision
    f_1_score = (2 * fitness * precision) / denom_f1 if denom_f1 > 0 else 0

    denom_tf1 = translucent_fitness + translucent_precision
    translucent_f_1_score = (
        (2 * translucent_fitness * translucent_precision) / denom_tf1
        if denom_tf1 > 0 else 0
    )

    simplicity = len(net.places) + len(net.transitions) + len(net.arcs)

    print(f"  [DONE ] log={log_name}, noise={noise_label}", flush=True)
    return {
        "time": duration,
        "RAM": max_memory / 1024,  # Convert to MB
        "fitness": fitness,
        "precision": precision,
        "translucent_fitness": translucent_fitness,
        "translucent_precision": translucent_precision,
        "f_1_score": f_1_score,
        "translucent_f_1_score": translucent_f_1_score,
        "simplicity": simplicity,
        "failed": False,
    }

# ---------------------------------------------------------------------------
# CSV writer worker — one per (log_name, noise_label) combination
# ---------------------------------------------------------------------------

def _store_results_worker(result_queue: Queue, results_path: Path) -> None:
    try:
        first_write = not results_path.exists()
        while True:
            item = result_queue.get()
            if item == "Ende":
                break
            result_data = item
            df = pd.DataFrame([result_data])
            df.to_csv(results_path, mode="a", header=first_write, index=False)
            first_write = False
    except Exception as exc:
        _write_error_file(f"store_results_{results_path.stem}", exc)
        raise

# ---------------------------------------------------------------------------
# Central task worker
# ---------------------------------------------------------------------------

def _central_task_worker(task_queue: Queue, out_queue: Queue) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            break
        log_name, noise_label, log = item
        try:
            result_data = evaluate_single_noise_type(log, log_name, noise_label)
            out_queue.put((log_name, noise_label, result_data))
        except Exception as exc:
            _write_error_file(f"task_{log_name}_{noise_label}", exc)
            raise

# ---------------------------------------------------------------------------
# Log loading — runs in parallel
# ---------------------------------------------------------------------------

def _load_log_worker(log_name: str, noise_type, path_to_log_fn, out_queue: Queue) -> None:
    try:
        noise_label = noise_type if noise_type else "base"
        path_to_log = path_to_log_fn(log_name)
        if not noise_type:
            log_path = path_to_log / f"{log_name}_base.csv"
        else:
            log_path = path_to_log / f"{log_name}_{noise_type}.csv"

        log = pd.read_csv(log_path)
        log = pm4py.format_dataframe(
            log,
            case_id="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
            timest_format="%Y-%m-%d %H:%M:%S%z",
        )
        log = pm4py.convert_to_event_log(log)
        log = add_artificial_start_and_end_activities_translucent(log)
        out_queue.put((log_name, noise_label, log))
    except Exception as exc:
        _write_error_file(f"load_log_{log_name}_{noise_type or 'base'}", exc)
        raise

# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

def run_evaluation(log_names, noise_types, path_to_log_fn):
    total_logs = len(log_names) * len(noise_types)

    # --- Load all logs in parallel ----------------------------------------
    print(f"[INIT] Loading {total_logs} event log(s) in parallel...")
    load_queue = Queue()
    loaders = []
    for log_name in log_names:
        for noise_type in noise_types:
            p = Process(target=_load_log_worker, args=(log_name, noise_type, path_to_log_fn, load_queue))
            p.start()
            loaders.append(p)

    log_map = {}
    for _ in range(total_logs):
        log_name, noise_label, log = load_queue.get()
        log_map[(log_name, noise_label)] = log
        print(f"  [{len(log_map)}/{total_logs}] Loaded: {log_name} / {noise_label}")

    for p in loaders:
        p.join()
    print("[INIT] All logs loaded.\n")

    # --- Build result queues and writer processes -------------------------
    result_queues = {}
    writer_processes = {}
    for log_name in log_names:
        for noise_type in noise_types:
            noise_label = noise_type if noise_type else "base"
            key = (log_name, noise_label)
            results_folder = Path(f"new_eval/translucent_datasets/{log_name}/results/petrify/{noise_label}")
            results_folder.mkdir(parents=True, exist_ok=True)
            results_path = results_folder / "petrify_results.csv"
            q = Queue()
            result_queues[key] = q
            writer = Process(target=_store_results_worker, args=(q, results_path))
            writer.start()
            writer_processes[key] = writer

    # --- Build flat task queue --------------------------------------------
    tasks = []
    for log_name in log_names:
        for noise_type in noise_types:
            noise_label = noise_type if noise_type else "base"
            log = log_map[(log_name, noise_label)]
            tasks.append((log_name, noise_label, log))

    total_tasks = len(tasks)
    print(
        f"[INIT] {total_tasks} jobs queued across {N_WORKERS} workers "
        f"({len(log_names)} log(s) x {len(noise_types)} noise type(s))\n"
    )

    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(N_WORKERS):
        task_queue.put(None)  # sentinel per worker

    # --- Launch workers ---------------------------------------------------
    central_out = Queue()
    workers = []
    for _ in range(N_WORKERS):
        p = Process(target=_central_task_worker, args=(task_queue, central_out))
        p.start()
        workers.append(p)

    # --- Collect results & print progress ---------------------------------
    completed = 0
    width = len(str(total_tasks))
    while completed < total_tasks:
        log_name, noise_label, result_data = central_out.get()
        key = (log_name, noise_label)
        result_queues[key].put(result_data)
        completed += 1
        print(f"[{completed:{width}}/{total_tasks}] log={log_name}, noise={noise_label}", flush=True)

    for worker in workers:
        worker.join()

    # --- Shut down writers ------------------------------------------------
    for q in result_queues.values():
        q.put("Ende")
    for writer in writer_processes.values():
        writer.join()

    print(f"\n[DONE] Petrify evaluation complete — {total_tasks} jobs finished.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    noise_types = [False, "remove_enabled"]

    def path_to_log_fn(log_name: str) -> Path:
        if os.name == "nt":
            return Path(
                f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\"
                f"TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{log_name}"
            )
        return Path(f"new_eval/translucent_datasets/{log_name}")

    try:
        run_evaluation(LOG_NAMES, noise_types, path_to_log_fn)
    except Exception as exc:
        _write_error_file("main", exc)
        sys.exit(1)
