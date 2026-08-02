"""
Script to discover a process model with the region based method described by van der Aalst using translucent event information.
"""

from new_eval.utils.read_in_petrify import parse_petrify_net_to_pm4py, load_apt_to_pm4py
from pm4py.objects.transition_system.obj import TransitionSystem
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.objects.transition_system.obj import TransitionSystem
from translucent_discovery.translucent_inductive_miner.translucent_datatype import TCL, translucent_log_to_tcl
import os
import subprocess    

def discover_net_with_regions_from_rooted_log(log: TCL, log_name= '', parameters=None):
    """
    Discovers a process model using the region based method by van der Aalst with translucent event information.
    Expects the log to be rooted, meaning the enabled activities of the first event in each trace should be the same!!!
    Translucent log is a TCL object.

    Parameters
    ----------
    log: : TCL
        Input event log
    parameters: dict
        Parameters of the algorithm

    Returns
    -------
    net : :class:`pm4py.objects.petri_net.petrinet.PetriNet`
        Discovered Petri net
    im : :class:`pm4py.objects.petri_net.marking.Marking`
        Initial marking of the Petri net
    fm : :class:`pm4py.objects.petri_net.marking.Marking`
        Final marking of the Petri net
    """
    # 1. Discover the automaton (Def. 5.1 in vdA paper)
    ts = _discover_automaton(log)
    
    # 2. Create petrify compatible .g file from the automaton
    path = os.getcwd()
    print("Current working directory:", path)
    path = os.path.join(path.split("new_eval")[0], r"new_eval",r"utils")
    rel_path = os.path.join(path, "temp_files")
    file_path = os.path.join(rel_path, log_name + "_temp.g")
    export_to_petrify(ts, file_path)
    
    # 3. Call petrify to discover the net
    if os.name == 'nt':  # Windows
        petrify_path = os.path.join(path, r"petrify",r"windows",r"bin",r"petrify.exe")
        output_path = path + "\\petrify_nets\\" + log_name + "_net.g"
    else:  # Linux
        import resource
        petrify_path = path + r"/petrify/linux/bin/petrify"
        output_path = path + r"/petrify_nets/" + log_name + "_net.g"
    
    
    print("Start Petrify, beginning net discovery...")
    result = subprocess.run([petrify_path, "-dead","-ip", file_path, "-o", output_path], check=True)
    if os.name != 'nt':  # Linux: Check for memory usage
        max_memory = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss # in KB
        print(f"Petrify max memory usage: {max_memory} KB")  
    print("Petrify finished net discovery.")
    # 4. Read in the petrify output as pm4py Petri net
    net, im, fm = parse_petrify_net_to_pm4py(output_path)
    
    if os.name != 'nt':
        return net, im, fm, max_memory 
    
    return net, im, fm

def _discover_automaton(log: TCL):
    """
    Discovers the automaton as described in Definition 5.1 of van der Aalst's paper. Returns a pm4py TransitionSystem object.
    """
    ts = TransitionSystem()
    existing_states = {}
    bottom_state =TransitionSystem.State("⊥")
    existing_states["⊥"] = bottom_state
    ts.states.add(bottom_state)
    
    # Create all states
    for variant in log.keys():
        for i in range(len(variant)):
            enabled_activities = variant[i][1]
            if enabled_activities not in existing_states:
                new_state = TransitionSystem.State(enabled_activities)
                existing_states[enabled_activities] = new_state
                ts.states.add(new_state)
    
    # Add the transitions
    for variant in log.keys():
        for i in range(len(variant)):
            current_event = variant[i]
            current_activity = current_event[0].replace(" ", "_")  # Petrify does not allow spaces in event names
            current_enabled_activities = current_event[1]
            current_state = existing_states[current_enabled_activities]
            if i < len(variant) - 1:
                next_event = variant[i+1]
                next_enabled_activities = next_event[1]
                next_state = existing_states[next_enabled_activities]
            else:
                next_state = bottom_state
            # Add transition
            new_transition = TransitionSystem.Transition(current_activity, current_state, next_state)
            ts.transitions.add(new_transition)
    
    return ts

def _rename_ts(ts: TransitionSystem):
    """
    Takes a TransitionSystem and renames its states to a standardized format (s0, s1, ...). Also reflects these changes in the transitions. The initial state is always s0.
    
    :param ts: Description
    :type ts: TransitionSystem
    """
    state_list = list(ts.states)
    state_map = {}
    
    to_state_set = set()
    for t in ts.transitions:
        to_state_set.add(t.to_state)
    
    # Find the initial state (no incoming transitions)
    initial_state = list(ts.states.difference(to_state_set))[0]
    # Remove initial state from list
    state_list.remove(initial_state)
    
    initial_state.name = "s0"
    
    for idx, state in enumerate(state_list):
        new_name = f"s{idx+1}"
        state_map[state] = new_name
        state.name = new_name  # Rename the state in place
        
def export_to_petrify(transition_system: TransitionSystem, file_path: str):
    _rename_ts(transition_system)
    with open(file_path, 'w') as f:
        # 1. Write Model Name
        model_name = "ts_model"
        if transition_system.name:
            model_name = str(transition_system.name)
        f.write(f".model {model_name}\n")

        # 2. Collect and Write Outputs (Events)
        events = set()
        for t in transition_system.transitions:
            events.add(t.name)
        
        if events:
            f.write(f".outputs {' '.join(sorted(events))}\n")

        # 3. Define State Graph Header
        f.write(".state graph\n")

        # 4. Write Transitions using the state names
        for t in transition_system.transitions:
            src = t.from_state.name
            tgt = t.to_state.name
            evt = t.name
            f.write(f"{src} {evt} {tgt}\n")

        # 5. Write Initial Marking
        f.write(".marking {s0}\n")

        f.write(".end\n")

    print(f"Successfully exported to {file_path}")       



def save_ts_to_apt(ts: TransitionSystem, output_file: str):
    """
    Converts a PM4Py Transition System to the official APT .type LTS format.
    Ref: APT_format.pdf section 6.3.2
    """
    
    # 1. Map complex PM4Py states to simple IDs (s0, s1...)
    # Sorting ensures deterministic output
    sorted_states = sorted(list(ts.states), key=lambda s: str(s))
    state_map = {state: f"s{i}" for i, state in enumerate(sorted_states)}
    
    # 2. Identify the Initial State object
    initial_state_obj = sorted_states[0] # Desperate fallback

    # 3. Collect Labels and Arcs
    labels = set()
    arcs_list = []
    
    for t in ts.transitions:
        src_id = state_map[t.from_state]
        tgt_id = state_map[t.to_state]
        
        # Clean label: remove spaces as they break the format
        label = t.name.replace(" ", "_") if t.name else "tau"
        
        labels.add(label)
        arcs_list.append((src_id, label, tgt_id))
        
    # 4. Write to File using strict APT LTS syntax
    with open(output_file, 'w') as f:
        # Header [cite: 23, 29]
        f.write('.name "pm4py_export"\n')
        f.write('.type LTS\n') 
        
        # States Section [cite: 31, 50]
        f.write('\n.states\n')
        for state_obj in sorted_states:
            s_id = state_map[state_obj]
            if state_obj == initial_state_obj:
                # The doc requires the initial state to be marked with [initial] [cite: 32, 50]
                f.write(f'{s_id} [initial]\n')
            else:
                f.write(f'{s_id}\n')
        
        # Labels Section (instead of .events) 
        f.write('\n.labels\n')
        for label in sorted(list(labels)):
            f.write(f'{label}\n')
            
        # Arcs Section (instead of .transitions) 
        f.write('\n.arcs\n')
        for src, label, tgt in sorted(arcs_list):
            f.write(f'{src} {label} {tgt}\n')

    print(f"Exported to {output_file} in APT LTS format.")
    split_labels_in_apt(output_file, output_file + "_edited")  # Handle non-deterministic transitions by splitting labels
    #print(f"States: {len(sorted_states)}, Labels: {len(labels)}, Arcs: {len(arcs_list)}")


def split_labels_in_apt(input_file, output_file):
    """
    Reads an APT file, detects non-deterministic transitions 
    (same source + same label -> diff target), and splits the labels.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    # Track existing arcs to detect conflicts: { (source, label): target }
    # If we see (source, label) again with a DIFFERENT target, we split.
    seen_transitions = {} 
    
    # We also need to track all used labels to update the .labels section later
    all_labels = set()
    newly_created_labels = set()
    
    parsing_arcs = False
    parsing_labels = False
    
    # First pass: Identify all labels currently in the file
    for line in lines:
        clean = line.strip()
        if clean.startswith('.labels'):
            parsing_labels = True
            continue
        if clean.startswith('.'):
            parsing_labels = False
        
        if parsing_labels and clean:
            all_labels.add(clean)

    # Second pass: Process lines and split arcs
    for line in lines:
        clean = line.strip()
        
        # Detect sections
        if clean.startswith('.arcs'):
            parsing_arcs = True
            new_lines.append(line)
            continue
        if clean.startswith('.') and not clean.startswith('.arcs'):
            parsing_arcs = False
            
        if parsing_arcs and clean:
            # Parse arc: "s0 label s1"
            parts = clean.split()
            if len(parts) == 3:
                src, label, tgt = parts
                
                key = (src, label)
                
                if key in seen_transitions:
                    existing_tgt = seen_transitions[key]
                    
                    if existing_tgt != tgt:
                        # CONFLICT DETECTED!
                        # Create a new label
                        base_label = label
                        counter = 2
                        new_label = f"{base_label}_{counter}"
                        
                        # Ensure we don't collide with existing labels
                        while new_label in all_labels or new_label in newly_created_labels:
                            counter += 1
                            new_label = f"{base_label}_{counter}"
                        
                        print(f"Splitting: {src} -> {label} -> {tgt}  ==>  {new_label}")
                        
                        newly_created_labels.add(new_label)
                        # Write the modified line
                        new_lines.append(f"{src} {new_label} {tgt}\n")
                        continue
                
                # No conflict, or first time seeing this source/label pair
                seen_transitions[key] = tgt
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        # Handle the .labels section update
        elif clean.startswith('.labels'):
            new_lines.append(line)
            # We will append new labels after reading the whole file to be safe, 
            # or we can insert them here if we buffer. 
            # Easier strategy: Just write the file, and we inject new labels at the end of the .labels block.
        else:
            new_lines.append(line)

    # Re-write the file, injecting the new labels
    with open(output_file, 'w') as f:
        in_labels_section = False
        for line in new_lines:
            if line.strip().startswith('.labels'):
                in_labels_section = True
                f.write(line)
                continue
            
            # If we hit the next section, dump our new labels first
            if in_labels_section and line.strip().startswith('.'):
                for nl in newly_created_labels:
                    f.write(f"{nl}\n")
                in_labels_section = False
            
            f.write(line)

    
if __name__ == "__main__":
    # Testing some functionality
    import pandas as pd
    import pm4py
    import time
    from pm4py.objects.conversion.log import converter as log_converter
    from new_eval.translucent_datasets.generate_noisy_datasets import get_noisy_log
    from new_eval.utils.make_rooted import add_artificial_start_and_end_activities_translucent
    from translucent_precision.main import translucent_precision_score_eval_version as translucent_precision_score
   
    from translucent_fitness.fitness import calculate_log_fitness
    
    
    #df = pd.read_csv(r"C:\eval_fall\new_eval\translucent_datasets\Sepsis\Sepsis_remove_events.csv")
    df = pd.read_csv(r"C:\eval_fall\new_eval\translucent_datasets\road_traffic_fine\road_traffic_fine_base.csv")
    #log = get_noisy_log(df, "change_events")
    #log =log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters={'pm4py:param:case_id_key': "case:concept:name", "activity_key": "concept:name"})
    #df = pd.read_csv(r"C:\Users\elias\Desktop\Results_22.02\TranslucentActivityRelationships-main\new_eval\translucent_datasets\Sepsis\Sepsis_0.2_add_events.csv")
    #df = pd.read_csv(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\Sepsis\Sepsis_0.2.csv") #!Very slow, good benchmark
    all_activities = set(df['concept:name'].unique())
    log = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp", timest_format="%Y-%m-%d %H:%M:%S%z")
    log = pm4py.convert_to_event_log(log)
    log = add_artificial_start_and_end_activities_translucent(log)
    #tcl_log = translucent_log_to_tcl(log)
    #net, im, fm = discover_net_with_regions_from_rooted_log(tcl_log, log_name="hospital_billing")
    #net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    #pm4py.write_pnml(net, im, fm, r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\utils\petrify_nets\road_traffic_fine_0.2_test.pnml")
    #net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    net, im, fm = pm4py.read_pnml(r"C:\eval_fall\new_eval\translucent_datasets\Sepsis\pnml\petrify\remove_events\petrify.pnml")
    #net, im, fm = pm4py.read_pnml(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\utils\petrify_nets\Sepsis_0.2_test.pnml")
    #fitness = pm4py.conformance.fitness_alignments(log, net, im, fm)["log_fitness"]
    #precision = pm4py.conformance.precision_alignments(log, net, im, fm)
    
    #precision, classic_fitness = translucent_precision_score(log, net, im, fm)
    start = time.time()
    fitness = pm4py.conformance.fitness_alignments(log, net, im, fm)["log_fitness"]
    print(f"Fitness: {fitness}")
    print(f"Time taken for fitness calculation: {time.time() - start} seconds")
    pm4py.view_petri_net(net, im, fm)
    