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
from translucent_discovery.translucent_inductive_miner.fall_through.empty_traces import EmptyTracesTranslucent

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

        empty_traces = EmptyTracesTranslucent.apply(obj, parameters)
        if empty_traces is not None:
            number_original_traces = sum(y for y in obj.data_structure.values())
            number_filtered_traces = sum(y for y in empty_traces[1][1].data_structure.values())

            if number_original_traces - number_filtered_traces > noise_threshold * number_original_traces:
                return self._recurse(empty_traces[0], empty_traces[1], parameters)
            else:
                obj = empty_traces[1][1]

        tree = self.apply_base_cases(obj, parameters)
        if tree is None:
            # First tDFG, then filtered tDFG, then DFG, then filtered DFG
            if parameters["translucent_variant"] == "IMtf":
                parameters["tDFG"] = True
                cut = self.find_cut(obj, parameters)
                if cut is not None:
                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                if tree is None:
                    if not second_iteration_translucent:
                        filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, True)
                        tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True)
                    if second_iteration_translucent:
                        parameters["tDFG"] = False
                        cut = self.find_cut(obj, parameters)
                        if cut is not None:
                            parameters["tDFG"] = True
                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
                        if tree is None:
                            if not second_iteration_normal:
                                filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, False)
                                tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True, second_iteration_normal=True)
                                if tree is None:
                                    if parameters["tDFG_fall_through"]:
                                        parameters["tDFG"] = True
                                    else:
                                        parameters["tDFG"] = False
                                    ft = self.fall_through(obj, parameters)
                                    tree = self._recurse(ft[0], ft[1], parameters=parameters)
            elif parameters["translucent_variant"] == "IM":
                parameters["tDFG"] = False
                if tree is None:
                    cut = self.find_cut(obj, parameters)
                    if cut is not None:
                        tree = self._recurse(cut[0], cut[1], parameters=parameters)
                    if tree is None:
                        if not second_iteration_normal:
                            filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, False)
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
                        if not second_iteration_translucent:
                            filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, translucent=True)
                            tree = self.apply(filtered_ds, parameters=parameters, second_iteration_translucent=True)
                            if tree is None:
                                if parameters["tDFG_fall_through"]:
                                    parameters["tDFG"] = True
                                else:
                                    parameters["tDFG"] = False
                                ft = self.fall_through(obj, parameters)
                                tree = self._recurse(ft[0], ft[1], parameters=parameters)
            elif parameters["translucent_variant"] == "IMts":
                parameters["tDFG"] = False
                cut = self.find_cut(obj, parameters)
                if cut is not None:
                    tree = self._recurse(cut[0], cut[1], parameters=parameters)
                if tree is None:
                    if not second_iteration_normal:
                        filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, False)
                        tree = self.apply(filtered_ds, parameters=parameters, second_iteration_normal=True)
                    if second_iteration_normal:
                        parameters["tDFG"] = True
                        cut = self.find_cut(obj, parameters)
                        if cut is not None:
                            parameters["tDFG"] = False
                            tree = self._recurse(cut[0], cut[1], parameters=parameters)
                        if tree is None:
                            if not second_iteration_translucent:
                                filtered_ds = self.__filter_dfg_noise(obj, noise_threshold, True)
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


    def __filter_dfg_noise(self, obj, noise_threshold, translucent):
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

        dfg = DFG()
        for sa in start_activities:
            dfg.start_activities[sa] = start_activities[sa]
        for ea in end_activities:
            dfg.end_activities[ea] = end_activities[ea]
        for act in graph:
            dfg.graph[act] = graph[act]

        return IMDataStructureTranslucent(obj.data_structure, obj.log, tdfg = dfg)