import warnings
import traceback
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
import pandas as pd
import os
from pathlib import Path
import pm4py
from pm4py.util import constants
import time
import sys

if os.name != 'nt':
    sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")
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

def _run_returning_pair_in_process(target, args) -> tuple | None:
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

def _precision_alignments_wrapper(log, net, im, fm) -> float:
    return pm4py.conformance.precision_alignments(log, net, im, fm)

def _calculate_log_fitness_wrapper(log, net, im, fm) -> float:
    return calculate_log_fitness(log, net, im, fm)

def _translucent_precision_score_wrapper(log, net, im, fm):
    # Returns (translucent_precision, fitness) tuple
    return translucent_precision_score(log, net, im, fm)

# ---------------------------------------------------------------------------
# Single-filter evaluation
# ---------------------------------------------------------------------------

def evaluate_single_filter(f: float, log, log_name: str, noise_type: str) -> tuple:
    """Evaluates a single filter value 'f'. Returns (f, result_dict)."""

    # --- Discovery --------------------------------------------------------
    start = time.time()
    net, im, fm = pm4py.discovery.discover_petri_net_inductive(log, noise_threshold=f)
    duration = time.time() - start
    print(f"  [DISC] log={log_name}, noise={noise_type}, f={f} — discovered in {duration:.1f}s", flush=True)

    # --- precision_alignments ---------------------------------------------
    _prec = Value("d", -1.0)
    if not _run_fn_in_process(_precision_alignments_wrapper, (log, net, im, fm), _prec):
        print(f"  [TIMEOUT] precision_alignments | log={log_name}, noise={noise_type}, f={f}", flush=True)
    precision = _prec.value

    # --- translucent_precision_score — returns (translucent_precision, fitness) ---
    pair = _run_returning_pair_in_process(
        _translucent_precision_score_wrapper, (log, net, im, fm)
    )
    if pair is None:
        print(f"  [TIMEOUT] translucent_precision_score | log={log_name}, noise={noise_type}, f={f}", flush=True)
        translucent_precision = -1.0
        fitness = get_alignment_fitness_with_processtree(log, net, im, fm)
    else:
        translucent_precision, fitness = pair

    # --- calculate_log_fitness --------------------------------------------
    _tfit = Value("d", -1.0)
    if not _run_fn_in_process(_calculate_log_fitness_wrapper, (log, net, im, fm), _tfit):
        print(f"  [TIMEOUT] calculate_log_fitness | log={log_name}, noise={noise_type}, f={f}", flush=True)
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

    result_data = {
        "time": duration,
        "RAM": 0,
        "fitness": fitness,
        "precision": precision,
        "translucent_fitness": translucent_fitness,
        "translucent_precision": translucent_precision,
        "f_1_score": f_1_score,
        "translucent_f_1_score": translucent_f_1_score,
        "simplicity": simplicity,
        "failed": False,
    }

    print(f"  [DONE ] log={log_name}, noise={noise_type}, f={f}", flush=True)
    return f, result_data

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
            f, result_data = item
            row = {"threshold": f, **result_data}
            df = pd.DataFrame([row])
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
        log_name, noise_label, f, log = item
        try:
            f_out, result_data = evaluate_single_filter(f, log, log_name, noise_label)
            out_queue.put((log_name, noise_label, f_out, result_data))
        except Exception as exc:
            _write_error_file(f"task_{log_name}_{noise_label}_f{f}", exc)
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

def run_evaluation(log_names, noise_types, path_to_log_fn, filter_values):
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
            results_folder = Path(f"new_eval/translucent_datasets/{log_name}/results/IMf/{noise_label}")
            results_folder.mkdir(parents=True, exist_ok=True)
            results_path = results_folder / "IMf_results.csv"
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
            for f in filter_values:
                tasks.append((log_name, noise_label, f, log))

    total_tasks = len(tasks)
    print(
        f"[INIT] {total_tasks} jobs queued across {N_WORKERS} workers "
        f"({len(log_names)} log(s) x {len(noise_types)} noise type(s) x {len(filter_values)} filter value(s))\n"
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
        log_name, noise_label, f, result_data = central_out.get()
        key = (log_name, noise_label)
        result_queues[key].put((f, result_data))
        completed += 1
        print(f"[{completed:{width}}/{total_tasks}] log={log_name}, noise={noise_label}, f={f}", flush=True)

    for worker in workers:
        worker.join()

    # --- Shut down writers ------------------------------------------------
    for q in result_queues.values():
        q.put("Ende")
    for writer in writer_processes.values():
        writer.join()

    print(f"\n[DONE] IMf evaluation complete — {total_tasks} jobs finished.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    noise_types = [False, "remove_enabled"]

    filter_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def path_to_log_fn(log_name: str) -> Path:
        if os.name == "nt":
            return Path(
                f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\"
                f"TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{log_name}"
            )
        return Path(f"new_eval/translucent_datasets/{log_name}")

    try:
        run_evaluation(LOG_NAMES, noise_types, path_to_log_fn, filter_values)
    except Exception as exc:
        _write_error_file("main", exc)
        sys.exit(1)
