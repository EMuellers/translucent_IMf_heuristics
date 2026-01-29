import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
import re

def remove_label_splitting_for_conformance_checking(net: PetriNet, original_transitions: list[str]) -> PetriNet:
    #Todo: Check whether only changing the label and not the name is sufficient
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
    add_back_spaces_in_labels(net)
    
    # Debug: Display the Petri net
    #pm4py.view_petri_net(net, im, fm)
    
    return net, im, fm

if __name__ == "__main__":
    file_path = r"C:\Users\elias\Desktop\petrify\bin\log_Epetri.g"
    net, im, fm = parse_petrify_net_to_pm4py(file_path)
    pm4py.view_petri_net(net, im, fm)
    import pandas as pd
    from pm4py.objects.conversion.log import converter as log_converter
    df = pd.read_csv(r"C:\Users\elias\Downloads\log_E.csv")
    log =log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters={'pm4py:param:case_id_key': "case_id", "activity_key": "activity"})
    fitness = pm4py.conformance.fitness_alignments(log, net, im, fm, activity_key="activity")
    precision = pm4py.conformance.precision_alignments(log, net, im, fm, activity_key="activity")
    print("Done.")          