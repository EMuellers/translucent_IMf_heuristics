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

from typing import Optional, Tuple, List, Dict, Any

from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough
from translucent_discovery.translucent_inductive_miner.fall_through.empty_traces import EmptyTracesTranslucent
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.util.compression import util as comut
from pm4py.util.compression.dtypes import UVCL
from pm4py.objects.dfg.obj import DFG
from pm4py.algo.discovery.inductive.dtypes.im_dfg import InductiveDFG


class FlowerModelTranslucent(FallThrough[IMDataStructureTranslucent]):

    @classmethod
    def holds(cls, obj: IMDataStructureTranslucent, parameters: Optional[Dict[str, Any]] = None) -> bool:
        return not EmptyTracesTranslucent.holds(obj, parameters)

    @classmethod
    def apply(cls, obj: IMDataStructureTranslucent, pool=None, manager=None, parameters: Optional[Dict[str, Any]] = None) -> Optional[
        Tuple[ProcessTree, List[IMDataStructureTranslucent]]]:
        log = obj.data_structure
        uvcl_do = UVCL()
        for a in comut.get_alphabet(log):
            uvcl_do[(a,)] = 1
        uvcl_redo = UVCL()
        im_uvcl_do = IMDataStructureTranslucent(uvcl_do, obj.log, frequent=obj.frequent)
        im_uvcl_redo = IMDataStructureTranslucent(uvcl_redo, obj.log, frequent=obj.frequent)
        return ProcessTree(operator=Operator.LOOP), [im_uvcl_do, im_uvcl_redo]

