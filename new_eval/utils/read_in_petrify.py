import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
import re

def remove_label_splitting_for_conformance_checking(net: PetriNet, original_transitions: list[str]) -> PetriNet:
    for transition in net.transitions:
        if transition.label not in original_transitions:
            transition.label = transition.label.rsplit('/', 1)[0]

def add_back_spaces_in_labels(net: PetriNet) -> PetriNet:
    for transition in net.transitions:
        if transition.label != '__end__' and transition.label != '__start__':
            transition.label = transition.label.replace('_', ' ')


def parse_petrify_net_to_pm4py_old(file_path: str) -> tuple[PetriNet, Marking, Marking]:
    """
    Parses a Petri net from a Petrify .sg or .g file and converts it to a pm4py Petri net.

    :param file_path: Path to the .sg or .g file.
    :return: A tuple containing the pm4py Petri net, initial marking, and final marking.
    """
    net = PetriNet("Synthesized_Net")
    places = {}
    transitions = {}
    
    # Read in the Petrify net
    with open(file_path, 'r') as file:
        for line in file:
            if line.lstrip().startswith('#') or line.lstrip().startswith('.model') or line.lstrip().startswith('.graph') or line.lstrip().startswith('.end'):
                continue
            elif line.lstrip().startswith('.inputs'): # These are the transitions
                # Remove .inputs and split by spaces
                inputs = line.lstrip()[7:].strip().split()
                for t_name in inputs:
                    transition = PetriNet.Transition(t_name)
                    transition.label = t_name #TODO: Do we need to set the label here?
                    net.transitions.add(transition)
                    transitions[t_name] = transition
                continue
            elif line.lstrip().startswith('.outputs'): # These are the transitions
                # Remove .inputs and split by spaces
                inputs = line.lstrip()[8:].strip().split()
                for t_name in inputs:
                    transition = PetriNet.Transition(t_name)
                    transition.label = t_name #TODO: Do we need to set the label here?
                    net.transitions.add(transition)
                    transitions[t_name] = transition
                continue
            elif line.lstrip().startswith('.marking'): # This is the place that contains the initial marking, e.g.: .marking { p4 }
                # The place is already created at this point
                p_name = line.lstrip()[8:].strip().replace('{', '').replace('}', '').strip()
                im = Marking()
                im[places[p_name]] = 1
                continue           
            else: # arcs follow, everything that is not in the .inputs line is a place, arcs are given in the form: place transition or transition place
                arcs = line.strip().split()
                if arcs[0] in transitions: # arc from transtion to place(s)
                    for i in range (1, len(arcs)):
                        place_name = arcs[i]
                        if place_name not in places:
                            place = PetriNet.Place(place_name)
                            net.places.add(place)
                            places[place_name] = place
                        arc = PetriNet.Arc(transitions[arcs[0]], places[place_name])
                        net.arcs.add(arc)
                        transitions[arcs[0]].out_arcs.add(arc)
                        places[place_name].in_arcs.add(arc)
                else: # arc from place to transition(s)
                    place_name = arcs[0]
                    if place_name not in places:
                        place = PetriNet.Place(place_name)
                        net.places.add(place)
                        places[place_name] = place
                    for i in range (1, len(arcs)):
                        transition_name = arcs[i]
                        arc = PetriNet.Arc(places[place_name], transitions[transition_name])
                        net.arcs.add(arc)
                        places[place_name].out_arcs.add(arc)
                        transitions[transition_name].in_arcs.add(arc)
    
    # Check whether a final place exists (places with no outgoing arcs), if not create one and connect activitys with no outgoing arcs to it
    found_final_place = False
    fm = Marking()
    for place in net.places:
        if len(place.out_arcs) == 0:
            found_final_place = True
            fm[place] = 1
    if not found_final_place:
        counter = 0
        for trans in net.transitions:
            if len(trans.out_arcs) == 0:
                final_place = PetriNet.Place("final_place")
                net.places.add(final_place)
                arc = PetriNet.Arc(trans, final_place)
                net.arcs.add(arc)
                trans.out_arcs.add(arc)
                final_place.in_arcs.add(arc)
                fm[final_place] = 1
                counter += 1
    if counter > 1:
        raise Exception("Multiple final places were created.")
    
    remove_label_splitting_for_conformance_checking(net, inputs)
    
    # Debug: Display the Petri net
    #pm4py.view_petri_net(net, im, fm)
    
    return net, im, fm

def parse_petrify_net_to_pm4py(file_path: str) -> tuple[PetriNet, Marking, Marking]:
    """
    Parses a Petri net from a Petrify .sg or .g file and converts it to a pm4py Petri net.

    :param file_path: Path to the .sg or .g file.
    :return: A tuple containing the pm4py Petri net, initial marking, and final marking.
    """
    net = PetriNet("Synthesized_Net")
    places = {}
    transitions = {}
    is_place = re.compile(r'p\d+')
    
    # Read in the Petrify net
    with open(file_path, 'r') as file:
        for line in file:
            if line.lstrip().startswith('#') or line.lstrip().startswith('.model') or line.lstrip().startswith('.graph') or line.lstrip().startswith('.end'):
                continue
            elif line.lstrip().startswith('.inputs'): # These are the transitions
                # Remove .inputs and split by spaces
                inputs = line.lstrip()[7:].strip().split() # needed for having original labels
                continue
            elif line.lstrip().startswith('.outputs'): # These are the transitions
                # Remove .inputs and split by spaces
                inputs = line.lstrip()[8:].strip().split() # needed for having original labels
                continue
            elif line.lstrip().startswith('.marking'): # This is the place that contains the initial marking, e.g.: .marking { p4 }
                # The place is already created at this point
                p_name = line.lstrip()[8:].strip().replace('{', '').replace('}', '').strip()
                im = Marking()
                im[places[p_name]] = 1
                continue           
            else: # arcs follow
                arcs = line.strip().split()
                if not is_place.match(arcs[0]): # arc from transition to place(s)
                    transition = PetriNet.Transition(arcs[0])
                    transition.label = arcs[0] #TODO: Do we need to set the label here?
                    net.transitions.add(transition)
                    transitions[arcs[0]] = transition
                    for i in range (1, len(arcs)):
                        place_name = arcs[i]
                        if place_name not in places:
                            place = PetriNet.Place(place_name)
                            net.places.add(place)
                            places[place_name] = place
                        arc = PetriNet.Arc(transitions[arcs[0]], places[place_name])
                        net.arcs.add(arc)
                        transitions[arcs[0]].out_arcs.add(arc)
                        places[place_name].in_arcs.add(arc)
                else: # arc from place to transition(s)
                    place_name = arcs[0]
                    if place_name not in places: # Should only be the start place
                        place = PetriNet.Place(place_name)
                        net.places.add(place)
                        places[place_name] = place
                    for i in range (1, len(arcs)):
                        transition_name = arcs[i]
                        if transition_name not in transitions:
                            transition = PetriNet.Transition(transition_name)
                            transition.label = transition_name #TODO: Do we need to set the label here?
                            net.transitions.add(transition)
                            transitions[transition_name] = transition
                        arc = PetriNet.Arc(places[place_name], transitions[transition_name])
                        net.arcs.add(arc)
                        places[place_name].out_arcs.add(arc)
                        transitions[transition_name].in_arcs.add(arc)
    
    # Check whether a final place exists (places with no outgoing arcs), if not create one and connect activitys with no outgoing arcs to it
    found_final_place = False
    fm = Marking()
    for place in net.places:
        if len(place.out_arcs) == 0:
            found_final_place = True
            fm[place] = 1
    if not found_final_place:
        counter = 0
        for trans in net.transitions:
            if len(trans.out_arcs) == 0:
                final_place = PetriNet.Place("final_place")
                net.places.add(final_place)
                arc = PetriNet.Arc(trans, final_place)
                net.arcs.add(arc)
                trans.out_arcs.add(arc)
                final_place.in_arcs.add(arc)
                fm[final_place] = 1
                counter += 1
    if counter > 1:
        raise Exception("Multiple final places were created.")
    
     
    remove_label_splitting_for_conformance_checking(net, inputs)
    #add_back_spaces_in_labels(net) #! If activities contain underscores in original log this needs to be changed! Even more headache if both spaces and underscores are used...
    
    # Debug: Display the Petri net
    #pm4py.view_petri_net(net, im, fm)
    
    return net, im, fm

def parse_apt_multiset(multiset_str):
    """
    Parses APT multiset strings like "{2*s1, s2, 5*s3}" or "{}"
    Returns a dictionary: {'s1': 2, 's2': 1, 's3': 5}
    Ref: APT_petri.pdf [cite: 141-146]
    """
    content = multiset_str.strip().strip('{}')
    if not content:
        return {}
    
    items = [x.strip() for x in content.split(',')]
    result = {}
    
    for item in items:
        if not item: continue
        
        # Check for multiplier "number * id"
        if '*' in item:
            parts = item.split('*')
            count = int(parts[0])
            p_id = parts[1].strip()
        else:
            count = 1
            p_id = item
            
        if p_id in result:
            result[p_id] += count
        else:
            result[p_id] = count
            
    return result

def load_apt_to_pm4py(file_path):
    """
    Reads an APT format Petri Net file and converts it to PM4Py objects.
    Ref: APT_petri.pdf Section 6.3.1 [cite: 104]
    """
    net = PetriNet("imported_from_apt")
    initial_marking = Marking()
    final_marking = Marking()
    
    places_map = {}      # ID -> Place Object
    transitions_map = {} # ID -> Transition Object
    
    current_section = None
    
    # Regex for parsing attributes: [key="value", label="b"]
    # Captures the label value specifically
    label_regex = re.compile(r'label\s*=\s*"([^"]+)"')
    
    # Regex for the flows line: t1: {pre} -> {post}
    # 
    flow_regex = re.compile(r'^([^:]+):\s*(\{.*?\})\s*->\s*(\{.*?\})')

    with open(file_path, 'r') as f:
        # Read lines, strip whitespace
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        # 1. Skip Comments [cite: 152]
        if line.startswith('//') or line.startswith('/*'):
            continue
            
        # 2. Detect Sections [cite: 119, 128, 139]
        if line.startswith('.'):
            header = line.split()[0].lower()
            if header == '.places':
                current_section = 'places'
                continue
            elif header == '.transitions':
                current_section = 'transitions'
                continue
            elif header == '.flows':
                current_section = 'flows'
                continue
            elif header == '.initial_marking':
                current_section = 'initial_marking'
                # initial_marking might be on the same line or next lines
                # If content follows the header on same line: .initial_marking {s1}
                content = line[len(header):].strip()
                if content:
                    # Parse immediately if on same line
                    im_dict = parse_apt_multiset(content)
                    for p_id, count in im_dict.items():
                        if p_id in places_map:
                            initial_marking[places_map[p_id]] += count
                continue
            else:
                current_section = None # Unknown or metadata section (.name, .type)
                continue

        # 3. Parse Content based on section
        
        # --- PLACES [cite: 120] ---
        if current_section == 'places':
            # Format: s1 [attributes]
            # We just need the ID (first token)
            parts = line.split()
            p_id = parts[0]
            
            if p_id not in places_map:
                p = PetriNet.Place(p_id)
                net.places.add(p)
                places_map[p_id] = p
                
        # --- TRANSITIONS [cite: 129] ---
        elif current_section == 'transitions':
            continue
            # Format: t1 [label="b"]
            parts = line.split(maxsplit=1)
            t_id = parts[0]
            t_label = t_id # Default label is the ID [cite: 132]
            
            # Check for explicitly defined label in attributes
            if len(parts) > 1:
                attrs = parts[1]
                match = label_regex.search(attrs)
                if match:
                    t_label = match.group(1)
            
            # Create transition
            if t_id not in transitions_map:
                t = PetriNet.Transition(t_id, label=t_label)
                net.transitions.add(t)
                transitions_map[t_id] = t
                
        # --- FLOWS  ---
        elif current_section == 'flows':
            # Format: t1: {s1, s2} -> {2*s3}
            match = flow_regex.match(line)
            if match:
                t_id = match.group(1).strip()
                pre_set_str = match.group(2)
                post_set_str = match.group(3)
                
                trans_obj = transitions_map.get(t_id)
                if not trans_obj:
                    # Auto-create if missing definition (safeguard)
                    trans_obj = PetriNet.Transition(t_id, label=t_id)
                    net.transitions.add(trans_obj)
                    transitions_map[t_id] = trans_obj
                
                # Process Pre-set (Places -> Transition)
                pre_dict = parse_apt_multiset(pre_set_str)
                for p_id, weight in pre_dict.items():
                    if p_id in places_map:
                        petri_utils.add_arc_from_to(places_map[p_id], trans_obj, net, weight=weight)
                        
                # Process Post-set (Transition -> Places)
                post_dict = parse_apt_multiset(post_set_str)
                for p_id, weight in post_dict.items():
                    if p_id in places_map:
                        petri_utils.add_arc_from_to(trans_obj, places_map[p_id], net, weight=weight)

        # --- INITIAL MARKING (continuation)  ---
        elif current_section == 'initial_marking':
            # If marking was defined on valid lines following the header
            im_dict = parse_apt_multiset(line)
            for p_id, count in im_dict.items():
                if p_id in places_map:
                    initial_marking[places_map[p_id]] += count

    # Check whether a final place exists (places with no outgoing arcs), if not create one and connect activitys with no outgoing arcs to it
    found_final_place = False
    fm = Marking()
    for place in net.places:
        if len(place.out_arcs) == 0:
            found_final_place = True
            fm[place] = 1
    if not found_final_place:
        counter = 0
        for trans in net.transitions:
            if len(trans.out_arcs) == 0:
                final_place = PetriNet.Place("final_place")
                net.places.add(final_place)
                arc = PetriNet.Arc(trans, final_place)
                net.arcs.add(arc)
                trans.out_arcs.add(arc)
                final_place.in_arcs.add(arc)
                fm[final_place] = 1
                counter += 1
    if counter > 1:
        raise Exception("Multiple final places were created.")
    
    return net, initial_marking, final_marking



if __name__ == "__main__":
    file_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\utils\petrify_nets\Sepsis_add_enabled_net.g"
    net, im, fm = parse_petrify_net_to_pm4py(file_path)
    pm4py.view_petri_net(net, im, fm)
    import pandas as pd
    from pm4py.objects.conversion.log import converter as log_converter
    df = pd.read_csv(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\Sepsis\Sepsis_0.2.csv")
    log =log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters={'pm4py:param:case_id_key': "case_id", "activity_key": "activity"})
    fitness = pm4py.conformance.fitness_alignments(log, net, im, fm, activity_key="activity")
    precision = pm4py.conformance.precision_alignments(log, net, im, fm, activity_key="activity")
    print("Done.")          