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
from typing import List, Optional, Tuple, TypeVar, Dict, Any

from pm4py.algo.discovery.inductive.cuts.abc import Cut
from translucent_discovery.translucent_inductive_miner.cuts.concurrency import ConcurrencyCutTranslucent, ConcurrencyCutTranslucentTCL
from translucent_discovery.translucent_inductive_miner.cuts.loop import LoopCutTranslucent, LoopCutTranslucentTCL
from translucent_discovery.translucent_inductive_miner.cuts.sequence import StrictSequenceCutTranslucent, SequenceCutTranslucent, StrictSequenceCutTranslucentTCL
from translucent_discovery.translucent_inductive_miner.cuts.xor import ExclusiveChoiceCutTranslucent, ExclusiveChoiceCutTranslucentTCL
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructure
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util import exec_utils
from enum import Enum


T = TypeVar('T', bound=IMDataStructure)
S = TypeVar('S', bound=Cut)


class Parameters(Enum):
    DISABLE_STRICT_SEQUENCE_CUT = "disable_strict_sequence_cut"


class CutFactory:

    #Elias: Gibt an welche Cuts erlaubt sind und in welcher Reihenfolge sie geprüft werden
    
    @classmethod
    def get_cuts(cls, obj: T, inst: IMInstance, parameters: Optional[Dict[str, Any]] = None, tcl=True) -> List[S]:
        if parameters is None:
            parameters = {}
        disable_strict_sequence_cut = exec_utils.get_param_value(Parameters.DISABLE_STRICT_SEQUENCE_CUT, parameters, False)
        if tcl:
            sequence_cut = StrictSequenceCutTranslucentTCL
        else:
            sequence_cut = StrictSequenceCutTranslucent
        if disable_strict_sequence_cut:
            sequence_cut = SequenceCutTranslucent
        if tcl:
            return[ExclusiveChoiceCutTranslucentTCL, sequence_cut, ConcurrencyCutTranslucentTCL, LoopCutTranslucentTCL]
        else:
            return [ExclusiveChoiceCutTranslucent, sequence_cut, ConcurrencyCutTranslucent, LoopCutTranslucent]

    @classmethod
    def find_cut(cls, obj: IMDataStructure, inst: IMInstance, parameters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[ProcessTree, List[T]]]:
        for c in CutFactory.get_cuts(obj, inst, parameters):
            r = c.apply(obj, parameters)
            if r is not None:
                return r
        return None
