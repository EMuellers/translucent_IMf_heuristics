import os
from abc import abstractmethod, ABC
from typing import Optional, Tuple, List, TypeVar, Generic, Dict, Any

from pm4py.objects.dfg.obj import DFG
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from translucent_discovery.translucent_inductive_miner.base_case.factory import BaseCaseFactory
from translucent_discovery.translucent_inductive_miner.cuts.factory import CutFactory
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructure
from translucent_discovery.translucent_inductive_miner.fall_through.factory import FallThroughFactory
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree
from enum import Enum
from pm4py.util import exec_utils, constants
from copy import copy
from copy import deepcopy
from translucent_discovery.translucent_inductive_miner.fall_through.empty_traces import EmptyTracesTranslucent
from translucent_discovery.translucent_inductive_miner.utils import get_delta_arcs, get_sorted_delta_arcs
from translucent_discovery.translucent_inductive_miner.tDFG import discover_frequent_dfg_tcl

T = TypeVar('T', bound=IMDataStructure)


class Parameters(Enum):
    MULTIPROCESSING = "multiprocessing"


class InductiveMinerFrequentFrameworkTranslucent(ABC, Generic[T]):
    """
    Base Class Implementing the Inductive Miner Framework.
    How to Extend:
    1. Create a dedicated IMDataStructure class (see pm4py.algo.discovery.inductive.dtypes.im_ds.py)
    2. Create dedicated Base Cases, Cuts and Fall Throughs for the newly constructed IMDataStructure
    3. Extend the BaseCaseFactory, CutFactory and FallThroughFactory with the newly created functions
    4. Create a subclass of this class indicating the type on which it is defined and the corresponding IMInstance.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        if parameters is None:
            parameters = {}

        enable_multiprocessing = exec_utils.get_param_value(Parameters.MULTIPROCESSING, parameters, constants.ENABLE_MULTIPROCESSING_DEFAULT)

        if enable_multiprocessing:
            from multiprocessing import Pool, Manager

            self._pool = Pool(os.cpu_count() - 1)
            self._manager = Manager()
            self._manager.support_list = []
        else:
            self._pool = None
            self._manager = None

    def apply_base_cases(self, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[ProcessTree]:
        return BaseCaseFactory.apply_base_cases(obj, self.instance(), parameters=parameters)

    def find_cut(self, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[ProcessTree, List[T]]]:
        return CutFactory.find_cut(obj, self.instance(), parameters=parameters)

    def fall_through(self, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Tuple[ProcessTree, List[T]]:
        return FallThroughFactory.fall_through(obj, self.instance(), self._pool, self._manager, parameters=parameters)

    def apply(self, obj: T, parameters: Optional[Dict[str, Any]] = None, second_iteration_translucent=False, second_iteration_normal = False) -> ProcessTree:
        noise_threshold = parameters["noise_threshold"]

        empty_traces = EmptyTracesTranslucent.apply(obj, parameters=parameters)
        if empty_traces is not None:
            number_original_traces = sum(y for y in obj.data_structure.values())
            number_filtered_traces = sum(y for y in empty_traces[1][1].data_structure.values())

            if number_original_traces - number_filtered_traces > noise_threshold * number_original_traces:
                return self._recurse(empty_traces[0], empty_traces[1], parameters)
            else:
                obj = empty_traces[1][1]

        tree = self.apply_base_cases(obj, parameters=parameters)
        if tree is None:
            # First tDFG, then filtered tDFG, then DFG, then filtered DFG
            if parameters["translucent_variant"] == "IMtf":
                if not second_iteration_normal:
                    parameters["tDFG"] = True
                cut = self.find_cut(obj, parameters=parameters) # called for filtered DFG and unfiltered tDFG
                if cut is not None:
                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                if tree is None:
                    if not second_iteration_translucent: # Hier arc removal heuristik vor Filter
                        if parameters.get("remove_arcs_heuristics", False): # Apply Arc Removal Heuristics before filtering
                                cut = self.apply_arc_removal_heuristics(obj, parameters)
                                if cut is not None:
                                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                        if tree is None:
                            filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, True, parameters=parameters)
                            tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True)
                    if second_iteration_translucent: # Hier Heuristik nach Filter arc removal
                        if parameters.get("remove_arcs_heuristics", False) and not second_iteration_normal: # Apply Arc Removal Heuristics after filtering
                            cut = self.apply_arc_removal_heuristics(obj, parameters) 
                            if cut is not None:
                                tree = self._recurse(cut[0], cut[1], parameters=parameters)
                        if tree is None:   # Heuristic did not yield a cut     
                            # From here: Work on normal DFG
                            parameters["tDFG"] = False
                            # First we need to get the unfiltered DFG and tDFG back if it is not the second iteration
                            if not second_iteration_normal:
                                obj = IMDataStructureTranslucent(obj.data_structure, obj.tcl, parameters=parameters, self_loop_info = obj._translucent_self_loops)
                                cut = self.find_cut(obj, parameters) # performed on unfiltered DFG
                            if cut is not None:
                                parameters["tDFG"] = True
                                tree = self._recurse(cut[0], cut[1], parameters=parameters)
                            if tree is None:
                                if not second_iteration_normal:
                                    # Apply arc addition heuristics before filtering
                                    if parameters.get("add_arcs_heuristics", False):
                                        cut = self.apply_arc_addition_heuristics(obj, parameters)
                                        if cut is not None:
                                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
                                    if tree is None: # heuristic did not yield a cut
                                        filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, False, parameters=parameters)
                                        tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True, second_iteration_normal=True) # Try filtered DFG
                                        if tree is None:
                                            if parameters["tDFG_fall_through"]: #TODO: Werden fallthroughs nicht sowieso auf dem log performed?
                                                parameters["tDFG"] = True
                                            else:
                                                parameters["tDFG"] = False
                                            ft = self.fall_through(obj, parameters)
                                            tree = self._recurse(ft[0], ft[1], parameters=parameters)
                                if second_iteration_normal: # no cut found on filtered DFG, try heuristic
                                    # Apply arc addition heuristics on filtered DFG and filtered tDFG
                                    if parameters.get("add_arcs_heuristics", False):
                                        cut = self.apply_arc_addition_heuristics(obj, parameters)
                                        if cut is not None:
                                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
            elif parameters["translucent_variant"] == "IM":
                parameters["tDFG"] = False
                if tree is None:
                    cut = self.find_cut(obj, parameters)
                    if cut is not None:
                        tree = self._recurse(cut[0], cut[1], parameters=parameters)
                    if tree is None:
                        if not second_iteration_normal:
                            filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, False, parameters=parameters)
                            tree = self.apply(filtered_ds, parameters=parameters, second_iteration_normal=True)
                            if tree is None:
                                if parameters["tDFG_fall_through"]:
                                    parameters["tDFG"] = True
                                else:
                                    parameters["tDFG"] = False
                                ft = self.fall_through(obj, parameters)
                                tree = self._recurse(ft[0], ft[1], parameters=parameters)
            elif parameters["translucent_variant"] == "IMto":
                parameters["tDFG"] = True
                if tree is None:
                    cut = self.find_cut(obj, parameters)
                    if cut is not None:
                        tree = self._recurse(cut[0], cut[1], parameters=parameters)
                    if tree is None:
                        if not second_iteration_translucent: # Hier bevor filter heuristic
                            if parameters.get("remove_arcs_heuristics", False) and parameters["delta_heuristic_frequent_before"]: # Apply Arc Removal Heuristics before filtering
                                cut = self.apply_arc_removal_heuristics(obj, parameters)
                                if cut is not None:
                                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                            if tree is None:
                                filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, translucent=True, parameters=parameters)
                                tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True)
                                if tree is None:
                                    # Get the filtered dfg 
                                    filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, translucent=False, parameters=parameters)
                                    obj._dfg = filtered_ds.dfg # Update the dfg to the filtered one for the fallthrough
                                    if parameters.get("remove_arcs_heuristics", False) and parameters["delta_heuristic_frequent_after"]: # Apply Arc Removal Heuristics after filtering
                                        cut = self.apply_arc_removal_heuristics(obj, parameters)
                                        if cut is not None:
                                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
                                    if tree is None:        
                                        if parameters["tDFG_fall_through"]:
                                            parameters["tDFG"] = True
                                        else:
                                            parameters["tDFG"] = False
                                        ft = self.fall_through(obj, parameters)
                                        tree = self._recurse(ft[0], ft[1], parameters=parameters)
            elif parameters["translucent_variant"] == "IMts": #TODO: Fix this
                if not second_iteration_translucent:
                    parameters["tDFG"] = False
                cut = self.find_cut(obj, parameters) # performed on normal DFG, filtered DFG and filtered tDFG
                if cut is not None:
                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                if tree is None:
                    if not second_iteration_normal:
                        if parameters.get("add_arcs_heuristics", False): # Arc addition heuristic before filtering
                                        cut = self.apply_arc_addition_heuristics(obj, parameters)
                                        if cut is not None:
                                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
                        if tree is None:
                            filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, False, parameters=parameters)
                            tree = self.apply(filtered_ds, parameters=parameters, second_iteration_normal=True)
                    if second_iteration_normal:
                        if parameters.get("add_arcs_heuristics", False) and not second_iteration_translucent: # Arc addition heuristic after filtering
                                        cut = self.apply_arc_addition_heuristics(obj, parameters)
                                        if cut is not None:
                                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
                        if tree is None:
                            parameters["tDFG"] = True
                            # We need to get the unfiltered DFG and tDFG back
                            if not second_iteration_translucent:
                                obj = IMDataStructureTranslucent(obj.data_structure, obj.tcl, parameters=parameters, self_loop_info = obj._translucent_self_loops)
                                cut = self.find_cut(obj, parameters)
                            if cut is not None:
                                parameters["tDFG"] = False
                                tree = self._recurse(cut[0], cut[1], parameters=parameters)
                            elif parameters.get("remove_arcs_heuristics", False): # Apply Arc Removal Heuristics before filtering, also called after filtering
                                cut = self.apply_arc_removal_heuristics(obj, parameters)
                                if cut is not None:
                                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                            if tree is None:
                                if not second_iteration_translucent:
                                    filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, True, parameters=parameters)
                                    tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True,
                                                    second_iteration_normal=True)
                                    if tree is None:
                                        if parameters["tDFG_fall_through"]:
                                            parameters["tDFG"] = True
                                        else:
                                            parameters["tDFG"] = False
                                        ft = self.fall_through(obj, parameters)
                                        tree = self._recurse(ft[0], ft[1], parameters=parameters)
            else:
                print("Variant not set!!!")
        return tree

    def _recurse(self, tree: ProcessTree, objs: List[T], parameters: Optional[Dict[str, Any]] = None):
        children = [self.apply(obj, parameters=parameters) for obj in objs]
        for c in children:
            c.parent = tree
        tree.children.extend(children)
        return tree

    @abstractmethod
    def instance(self) -> IMInstance:
        pass
    
    """
    #TODO: Shouldn't dfg and tdfg be filtered here? -> Yes!
    def __filter_dfg_noise(self, obj, noise_threshold, translucent, parameters={}):
        if translucent:
            start_activities = copy(obj.tdfg.start_activities)
            end_activities = copy(obj.tdfg.end_activities)
            dfg = copy(obj.tdfg.graph)
        else:
            start_activities = copy(obj.dfg.start_activities)
            end_activities = copy(obj.dfg.end_activities)
            dfg = copy(obj.dfg.graph)
        outgoing_max_occ = {}
        for x, y in dfg.items():
            act = x[0]
            if act not in outgoing_max_occ:
                outgoing_max_occ[act] = y
            else:
                outgoing_max_occ[act] = max(y, outgoing_max_occ[act])
            if act in end_activities:
                outgoing_max_occ[act] = max(outgoing_max_occ[act], end_activities[act])
        dfg_list = sorted([(x, y) for x, y in dfg.items()], key=lambda x: (x[1], x[0]), reverse=True)
        dfg_list = [x for x in dfg_list if x[1] > noise_threshold * outgoing_max_occ[x[0][0]]]
        dfg_list = [x[0] for x in dfg_list]
        # filter the elements in the DFG
        graph = {x: y for x, y in dfg.items() if x in dfg_list}
        
        # apply filtering to start activities
        start_max_occ = max(start_activities.values())
        start_activities = {x: y for x, y in start_activities.items()
             if y >= start_max_occ * noise_threshold
        }
        
        # apply filtering to end activities only if translucent
        # TODO: Check if this is necessary
        if translucent:
            end_max_occ = max(end_activities.values())
            end_activities = {x: y for x, y in end_activities.items()
                 if y >= end_max_occ * noise_threshold
            }
        
        dfg = DFG()
        for sa in start_activities:
            dfg.start_activities[sa] = start_activities[sa]
        for ea in end_activities:
            dfg.end_activities[ea] = end_activities[ea]
        for act in graph:
            dfg.graph[act] = graph[act]

        # Fix: Hand over the correct (t)dfg, frequent flag and parameters
        if translucent:
            return IMDataStructureTranslucent(obj.data_structure, obj.tcl, tdfg = dfg, frequent=obj.frequent, parameters=parameters, self_loop_info = obj._translucent_self_loops)
        else:
            return IMDataStructureTranslucent(obj.data_structure, obj.tcl, dfg = dfg, frequent=obj.frequent, parameters=parameters, self_loop_info = obj._translucent_self_loops)
    """
    def __filter_dfg_noise(self, obj, noise_threshold, translucent, parameters={}):
        # First filter the DFG
        start_activities = copy(obj.dfg.start_activities)
        end_activities = copy(obj.dfg.end_activities)
        dfg = copy(obj.dfg.graph)
        outgoing_max_occ = {}
        for x, y in dfg.items():
            act = x[0]
            if act not in outgoing_max_occ:
                outgoing_max_occ[act] = y
            else:
                outgoing_max_occ[act] = max(y, outgoing_max_occ[act])
            if act in end_activities:
                outgoing_max_occ[act] = max(outgoing_max_occ[act], end_activities[act])
        dfg_list = sorted([(x, y) for x, y in dfg.items()], key=lambda x: (x[1], x[0]), reverse=True)
        dfg_list = [x for x in dfg_list if x[1] > noise_threshold * outgoing_max_occ[x[0][0]]]
        dfg_list = [x[0] for x in dfg_list]
        # filter the elements in the DFG
        graph = {x: y for x, y in dfg.items() if x in dfg_list}
        
        # apply filtering to start activities
        start_max_occ = max(start_activities.values())
        start_activities = {x: y for x, y in start_activities.items()
             if y >= start_max_occ * noise_threshold
        }
        
        dfg = DFG()
        for sa in start_activities:
            dfg.start_activities[sa] = start_activities[sa]
        for ea in end_activities:
            dfg.end_activities[ea] = end_activities[ea]
        for act in graph:
            dfg.graph[act] = graph[act]
        
        # Create filtered tDFG
        # First discover plain tfdfg
        tfdfg = discover_frequent_dfg_tcl(obj.tcl, parameters=parameters, self_loops=obj._translucent_self_loops)
        
        #now apply filtering threshold to tfdfg
        start_activities_tdfg = copy(tfdfg.start_activities)
        end_activities_tdfg = copy(tfdfg.end_activities)
        tdfg = copy(tfdfg.graph)
        outgoing_max_occ_tdfg = {}
        for x, y in tfdfg.items():
            act = x[0]
            if act not in outgoing_max_occ_tdfg:
                outgoing_max_occ_tdfg[act] = y
            else:
                outgoing_max_occ_tdfg[act] = max(y, outgoing_max_occ_tdfg[act])
            if act in end_activities_tdfg:
                outgoing_max_occ_tdfg[act] = max(outgoing_max_occ_tdfg[act], end_activities_tdfg[act])
        tdfg_list = sorted([(x, y) for x, y in tdfg.items()], key=lambda x: (x[1], x[0]), reverse=True)
        tdfg_list = [x for x in tdfg_list if x[1] > noise_threshold * outgoing_max_occ_tdfg[x[0][0]]]
        tdfg_list = [x[0] for x in tdfg_list]
        # filter the elements in the DFG
        graph_tdfg = {x: y for x, y in tdfg.items() if x in tdfg_list}
        
        # apply filtering to start activities
        start_max_occ_tdfg = max(start_activities_tdfg.values())
        start_activities_tdfg = {x: y for x, y in start_activities_tdfg.items()
             if y >= start_max_occ_tdfg * noise_threshold
        }
        
        tdfg = DFG()
        for sa in start_activities_tdfg:
            tdfg.start_activities[sa] = start_activities_tdfg[sa]
        for ea in end_activities_tdfg:
            tdfg.end_activities[ea] = end_activities_tdfg[ea]
        for act in graph_tdfg:
            tdfg.graph[act] = graph_tdfg[act]
        
        """
        # apply filtering to end activities only if translucent
        # TODO: Check if this is necessary
        if translucent:
            end_max_occ = max(end_activities.values())
            end_activities = {x: y for x, y in end_activities.items()
                 if y >= end_max_occ * noise_threshold
            }
        """
        # Return datastructure with filtered DFG and tDFG
        return IMDataStructureTranslucent(obj.data_structure, obj.tcl, dfg = dfg, tdfg = tdfg, frequent=True, parameters=parameters, self_loop_info = obj._translucent_self_loops) # Frequent set to true so that handed tdfg is kept
        
        
    def apply_arc_removal_heuristics(self, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[ProcessTree, List[T]]]:
        """Applies arc removal heuristics to try to find a cut in the modified tDFG. If no cut is found, None is returned."""
        candidate_arcs = get_delta_arcs(obj.tdfg, obj.dfg)
        original_tdfg = deepcopy(obj.tdfg)
        cut = None
        if len(candidate_arcs) > 0:
            sorted_arcs = get_sorted_delta_arcs(candidate_arcs, obj, criterion=parameters["remove_arcs_heuristics"])
            # Remove worst arcs one by one and try to find a cut
            for arc, score in sorted_arcs:
                del obj.tdfg.graph[arc]
                cut = self.find_cut(obj, parameters)
                if cut is not None:
                    return cut
        obj._tdfg = original_tdfg
        return cut
    
    def apply_arc_addition_heuristics(self, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[ProcessTree, List[T]]]:
        """Applies arc addition heuristics to try to find a cut in the modified DFG. If no cut is found, None is returned."""
        candidate_arcs = get_delta_arcs(obj.tdfg, obj.dfg)
        original_dfg = deepcopy(obj.dfg)
        cut = None
        if len(candidate_arcs) > 0:
            sorted_arcs = get_sorted_delta_arcs(candidate_arcs, obj, criterion=parameters["add_arcs_heuristics"])
            sorted_arcs.reverse() # We want to add the best arcs first, function returns worst to best
            # Add best arcs one by one and try to find a cut
            for arc, score in sorted_arcs:
                obj.dfg.graph.update({arc: obj.tdfg.graph[arc]}) #TODO: Check if this works correctly
                cut = self.find_cut(obj, parameters)
                if cut is not None:
                    return cut
        obj._dfg = original_dfg
        return cut