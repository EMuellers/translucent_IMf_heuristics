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
from pm4py.algo.discovery.inductive.base_case.abc import BaseCase
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from pm4py.objects.process_tree.obj import ProcessTree
from typing import Optional, Dict, Any


class SingleActivityBaseCaseTranslucent(BaseCase[IMDataStructureTranslucent]):
    @classmethod
    def holds(cls, obj=IMDataStructureTranslucent, parameters: Optional[Dict[str, Any]] = None) -> bool:
        if len(obj.data_structure.keys()) != 1:
            return False
        if len(list(obj.data_structure.keys())[0]) > 1:
            return False
        return True

    @classmethod
    def leaf(cls, obj=IMDataStructureTranslucent, parameters: Optional[Dict[str, Any]] = None) -> ProcessTree:
        for t in obj.data_structure:
            return ProcessTree(label=t[0])
