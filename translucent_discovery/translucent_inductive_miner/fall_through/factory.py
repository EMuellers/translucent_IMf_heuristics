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

from typing import List, TypeVar, Tuple, Optional, Dict, Any

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructure
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough
from translucent_discovery.translucent_inductive_miner.fall_through.activity_once_per_trace import ActivityOncePerTraceTranslucent
from translucent_discovery.translucent_inductive_miner.fall_through.activity_once_per_trace import ActivityConcurrentTranslucent
from translucent_discovery.translucent_inductive_miner.fall_through.empty_traces import EmptyTracesTranslucent
from translucent_discovery.translucent_inductive_miner.fall_through.flower import FlowerModelTranslucent
from translucent_discovery.translucent_inductive_miner.fall_through.strict_tau_loop import StrictTauLoopTranslucent
from translucent_discovery.translucent_inductive_miner.fall_through.tau_loop import TauLoopTranslucent
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util import exec_utils
from enum import Enum


T = TypeVar('T', bound=IMDataStructure)
S = TypeVar('S', bound=FallThrough)


class Parameters(Enum):
    DISABLE_FALLTHROUGHS = "disable_fallthroughs"


class FallThroughFactory:

    @classmethod
    def get_fall_throughs(cls, obj: T, inst: IMInstance, parameters: Optional[Dict[str, Any]] = None) -> List[S]:
        if parameters is None:
            parameters = {}
        if type(obj) is IMDataStructureTranslucent:
            return [EmptyTracesTranslucent, ActivityOncePerTraceTranslucent, ActivityConcurrentTranslucent, StrictTauLoopTranslucent, TauLoopTranslucent, FlowerModelTranslucent]

    @classmethod
    def fall_through(cls, obj: T, inst: IMInstance, pool, manager, parameters: Optional[Dict[str, Any]] = None) -> Tuple[ProcessTree, List[T]]:
        for f in FallThroughFactory.get_fall_throughs(obj, inst, parameters):
            r = f.apply(obj, pool, manager, parameters)
            if r is not None:
                return r
        return None
