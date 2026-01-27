"""
Script to discover a process model with the region based method described by van der Aalst using translucent event information.
"""

from new_eval.utils.read_in_petrify import parse_petrify_net_to_pm4py
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
    path = path.split("new_eval")[0] + "new_eval\\utils"
    rel_path = path + "\\temp_files"
    file_path = os.path.join(rel_path, log_name + "_temp.g")
    export_to_petrify(ts, file_path)
    
    # 3. Call petrify to discover the net
    if os.name == 'nt':  # Windows
        petrify_path = path +"\\petrify\\windows\\bin\\petrify.exe"
    else:  # Linux
        petrify_path = path + "\\petrify\\linux\\bin\\petrify"
    
    output_path = os.path.join(path + "\\petrify_nets", log_name + "_net.g")
    
    subprocess.run([petrify_path, "-dead","-ip", file_path, "-o", output_path], check=True)  
    
    # 4. Read in the petrify output as pm4py Petri net
    net, im, fm = parse_petrify_net_to_pm4py(output_path) 
    
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
            current_activity = current_event[0]
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
    
if __name__ == "__main__":
    # Testing some functionality
    import pandas as pd
    from pm4py.objects.conversion.log import converter as log_converter
    df = pd.read_csv(r"C:\Users\elias\Downloads\log_E.csv")
    log =log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters={'pm4py:param:case_id_key': "case_id", "activity_key": "activity"})
    tcl_log = translucent_log_to_tcl(log)
    net, im, fm = discover_net_with_regions_from_rooted_log(tcl_log, log_name="test_log")
    import pm4py
    pm4py.view_petri_net(net, im, fm)
    