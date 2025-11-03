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
from collections import Counter
from typing import Tuple, List, Optional, Dict, Any
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.algo.discovery.inductive.fall_through.abc import FallThrough
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from copy import copy


class EmptyTracesTranslucent(FallThrough[IMDataStructureTranslucent]):

    @classmethod
    def apply(cls, obj: IMDataStructureTranslucent, pool=None, manager=None, parameters: Optional[Dict[str, Any]] = None) -> Optional[
        Tuple[ProcessTree, List[IMDataStructureTranslucent]]]:
        if cls.holds(obj, parameters):
            data_structure = copy(obj.data_structure)
            del data_structure[()]
            return ProcessTree(operator=Operator.XOR), [IMDataStructureTranslucent(Counter(), obj.log, frequent=obj.frequent),
                                                        IMDataStructureTranslucent(data_structure, obj.log, frequent=obj.frequent)]
        else:
            return None

    @classmethod
    def holds(cls, obj: IMDataStructureTranslucent, parameters: Optional[Dict[str, Any]] = None) -> bool:
        return len(list(filter(lambda t: len(t) == 0, obj.data_structure))) > 0