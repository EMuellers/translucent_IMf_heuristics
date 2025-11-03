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
from abc import ABC
from collections import Counter
from itertools import product
from typing import List, Collection, Any, Optional, Generic, Dict

from pm4py.algo.discovery.inductive.cuts import utils as cut_util
from pm4py.algo.discovery.inductive.cuts.abc import Cut, T
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.objects.dfg import util as dfu
from pm4py.objects.dfg.obj import DFG
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.util.compression import util as comut
from pm4py.util.compression.dtypes import UVCL


class ConcurrencyCut(Cut[T], ABC, Generic[T]):

    @classmethod
    def operator(cls, parameters: Optional[Dict[str, Any]] = None) -> ProcessTree:
        return ProcessTree(operator=Operator.PARALLEL)

    @classmethod
    def holds(cls, obj: T, parameters: Optional[Dict[str, Any]] = None) -> Optional[List[Collection[Any]]]:
        if parameters["tDFG"]:
            dfg = obj.tdfg
        else:
            dfg = obj.dfg
        alphabet = dfu.get_vertices(dfg)
        # TODO: We found a bug in PM4Py, alternating the results each time we execute it. Therefore, we fix the order here
        alphabet = sorted(list(alphabet))
        msdw = comut.msdw(obj, comut.msd(obj)) if obj is not None and type(obj) is UVCL else None
        groups = [{a} for a in alphabet]
        if len(groups) == 0:
            return None
        edges = dfu.get_edges(dfg)
        # TODO: We found a bug in PM4Py, alternating the results each time we execute it. Therefore, we fix the order here
        edges = sorted(list(edges))
        for a, b in product(alphabet, alphabet):
            if (a, b) not in edges or (b, a) not in edges:
                groups = cut_util.merge_groups_based_on_activities(a, b, groups)
            elif msdw is not None:
                if (a in msdw and b in msdw[a]) or (b in msdw and a in msdw[b]):
                    groups = cut_util.merge_groups_based_on_activities(a, b, groups)

        groups = list(sorted(groups, key=lambda g: len(g)))
        i = 0
        while i < len(groups) and len(groups) > 1:
            if len(groups[i].intersection(set(dfg.start_activities.keys()))) > 0 and len(
                    groups[i].intersection(set(dfg.end_activities.keys()))) > 0:
                i += 1
                continue
            group = groups[i]
            del groups[i]
            if i == 0:
                groups[i].update(group)
            else:
                groups[i - 1].update(group)

        return groups if len(groups) > 1 else None


class ConcurrencyCutTranslucent(ConcurrencyCut[IMDataStructureTranslucent]):

    @classmethod
    def project(cls, obj: IMDataStructureTranslucent, groups: List[Collection[Any]], parameters: Optional[Dict[str, Any]] = None) -> List[IMDataStructureTranslucent]:
        r = list()
        for g in groups:
            c = Counter()
            for t in obj.data_structure:
                c[tuple(filter(lambda e: e in g, t))] = obj.data_structure[t]
            r.append(c)
        return list(map(lambda l: IMDataStructureTranslucent(l, obj.log, frequent=obj.frequent), r))
