'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import itertools
import sys
from abc import ABC
from collections import Counter
from itertools import product
from typing import Collection, Any, List, Optional, Generic, Dict
from typing import Tuple

from pm4py.algo.discovery.inductive.cuts import utils as cut_util
from pm4py.algo.discovery.inductive.cuts.abc import Cut
from pm4py.algo.discovery.inductive.cuts.abc import T
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.objects.dfg import util as dfu
from pm4py.objects.dfg.obj import DFG
from pm4py.objects.process_tree.obj import Operator, ProcessTree


class SequenceCut(Cut[T], ABC, Generic[T]):

    @classmethod
    def operator(cls, parameters: Optional[Dict[str, Any]] = None) -> ProcessTree:
        return ProcessTree(operator=Operator.SEQUENCE)

    #Elias: Dieser Code ist in neuer Version von pm4py anders, sollte aber das Gleiche tun
    @classmethod
    def holds(cls, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[List[Collection[Any]]]:
        '''
        This method finds a sequence cut in the dfg.
        Implementation follows function sequence on page 188 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)

        Basic Steps:
        1. create a group per activity
        2. merge pairwise reachable nodes (based on transitive relations)
        3. merge pairwise unreachable nodes (based on transitive relations)
        4. sort the groups based on their reachability
        '''
        if parameters["tDFG"]:
            dfg = obj.tdfg
        else:
            dfg = obj.dfg
        alphabet = dfu.get_vertices(dfg)
        transitive_predecessors, transitive_successors = dfu.get_transitive_relations(dfg)
        groups = [{a} for a in alphabet]
        if len(groups) == 0:
            return None
        for a, b in product(alphabet, alphabet):
            if (b in transitive_successors[a] and a in transitive_successors[b]) or (
                    b not in transitive_successors[a] and a not in transitive_successors[b]):
                groups = cut_util.merge_groups_based_on_activities(a, b, groups)

        groups = list(sorted(groups, key=lambda g: len(
            transitive_predecessors[next(iter(g))]) + (len(alphabet) - len(transitive_successors[next(iter(g))]))))
        return groups if len(groups) > 1 else None


class StrictSequenceCut(SequenceCut[T], ABC, Generic[T]):

    @classmethod
    def _skippable(cls, p: int, dfg: DFG, start: Collection[Any], end: Collection[Any],
                   groups: List[Collection[Any]], parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        This method implements the function SKIPPABLE as defined on page 233 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)
        The function is used as a helper function for the strict sequence cut detection mechanism, which detects
        larger groups of skippable activities.
        """
        for i, j in itertools.product(range(0, p), range(p + 1, len(groups))):
            for a, b in itertools.product(groups[i], groups[j]):
                if (a, b) in dfg.graph:
                    return True
        for i in range(p + 1, len(groups)):
            for a in groups[i]:
                if a in start:
                    return True
        for i in range(0, p):
            for a in groups[i]:
                if a in end:
                    return True
        return False

    @classmethod
    def holds(cls, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[List[Collection[Any]]]:
        """
        This method implements the strict sequence cut as defined on page 233 of
        "Robust Process Mining with Guarantees" by Sander J.J. Leemans (ISBN: 978-90-386-4257-4)
        The function merges groups that together can be skipped.
        """
        if parameters["tDFG"]:
            dfg = obj.tdfg
        else:
            dfg = obj.dfg
        c = SequenceCut.holds(obj, parameters)
        start = set(dfg.start_activities.keys())
        end = set(dfg.end_activities.keys())
        if c is not None:
            mf = [-1 * sys.maxsize if len(set(G).intersection(start)) > 0 else sys.maxsize for G in c]
            mt = [sys.maxsize if len(set(G).intersection(end)) > 0 else -1 * sys.maxsize for G in c]
            cmap = cls._construct_alphabet_cluster_map(c)
            for (a, b) in dfg.graph:
                mf[cmap[b]] = min(mf[cmap[b]], cmap[a])
                mt[cmap[a]] = max(mt[cmap[a]], cmap[b])

            for p in range(0, len(c)):
                if cls._skippable(p, dfg, start, end, c):
                    q = p - 1
                    while q >= 0 and mt[q] <= p:
                        c[p] = c[p].union(c[q])
                        c[q] = set()
                        q -= 1
                    q = p + 1
                    while q < len(mf) and mf[q] >= p:
                        c[p] = c[p].union(c[q])
                        c[q] = set()
                        q += 1
            return list(filter(lambda g: len(g) > 0, c))
        return None

    @classmethod
    def _construct_alphabet_cluster_map(cls, c: List[Collection[Any]], parameters: Optional[Dict[str, Any]] = None):
        map = dict()
        for i in range(0, len(c)):
            for a in c[i]:
                map[a] = i
        return map


class SequenceCutTranslucent(SequenceCut[IMDataStructureTranslucent]):

    @classmethod
    def project(cls, obj: IMDataStructureTranslucent, groups: List[Collection[Any]], parameters: Optional[Dict[str, Any]] = None) -> List[IMDataStructureTranslucent]:
        logs = [Counter() for g in groups]
        for t in obj.data_structure:
            i = 0
            split_point = 0
            act_union = set()
            while i < len(groups):
                new_split_point = cls._find_split_point(
                    t, groups[i], split_point, act_union)
                trace_i = tuple()
                j = split_point
                while j < new_split_point:
                    if t[j] in groups[i]:
                        trace_i = trace_i + (t[j],)
                    j = j + 1
                logs[i].update({trace_i: obj.data_structure[t]})
                split_point = new_split_point
                act_union = act_union.union(set(groups[i]))
                i = i + 1
        return list(map(lambda l: IMDataStructureTranslucent(l, obj.log, frequent=obj.frequent), logs))

    @classmethod
    def _find_split_point(cls, t: Tuple[Any], group: Collection[Any], start: int, ignore: Collection[Any], parameters: Optional[Dict[str, Any]] = None) -> int:
        least_cost = 0
        position_with_least_cost = start
        cost = 0
        i = start
        while i < len(t):
            if t[i] in group:
                cost = cost - 1
            elif t[i] not in ignore:
                cost = cost + 1

            if cost < least_cost:
                least_cost = cost
                position_with_least_cost = i + 1

            i = i + 1

        return position_with_least_cost


class StrictSequenceCutTranslucent(StrictSequenceCut[IMDataStructureTranslucent], SequenceCutTranslucent):

    @classmethod
    def holds(cls, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[List[Collection[Any]]]:
        return StrictSequenceCut.holds(obj, parameters)
    
#TODO: Make compatible with TCL
#TODO: Check if this is correct
class SequenceCutTranslucentTCL(SequenceCut[IMDataStructureTranslucent]):

    @classmethod
    def project(cls, obj: IMDataStructureTranslucent, groups: List[Collection[Any]], parameters: Optional[Dict[str, Any]] = None) -> List[IMDataStructureTranslucent]:
        logs = [Counter() for g in groups]
        for t in obj.tcl:
            i = 0
            split_point = 0
            act_union = set()
            while i < len(groups):
                new_split_point = cls._find_split_point(
                    t, groups[i], split_point, act_union)
                trace_i = tuple()
                j = split_point
                while j < new_split_point:
                    if t[j][0] in groups[i]:
                        trace_i = trace_i + (t[j],)
                    j = j + 1
                logs[i].update({trace_i: obj.tcl[t]})
                split_point = new_split_point
                act_union = act_union.union(set(groups[i]))
                i = i + 1
        return list(map(lambda l: IMDataStructureTranslucent(None, l, obj.log, frequent=obj.frequent, parameters=parameters), logs))

    @classmethod
    def _find_split_point(cls, t: Tuple[Any], group: Collection[Any], start: int, ignore: Collection[Any], parameters: Optional[Dict[str, Any]] = None) -> int:
        least_cost = 0
        position_with_least_cost = start
        cost = 0
        i = start
        while i < len(t):
            if t[i][0] in group:
                cost = cost - 1
            elif t[i][0] not in ignore:
                cost = cost + 1

            if cost < least_cost:
                least_cost = cost
                position_with_least_cost = i + 1

            i = i + 1

        return position_with_least_cost


#TODO: Make compatible with TCL
#! TODO: Add PARAMETERS TO DATASTRUCTURE CREATION!!! 
# Implements sequence cut for TCL logs
class StrictSequenceCutTranslucentTCL(StrictSequenceCut[IMDataStructureTranslucent], SequenceCutTranslucentTCL):

    @classmethod
    def holds(cls, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[List[Collection[Any]]]:
        return StrictSequenceCut.holds(obj, parameters)

