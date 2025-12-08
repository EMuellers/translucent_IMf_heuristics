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
import copy
from typing import Any, Optional, Dict
from collections import Counter
from translucent_discovery.translucent_inductive_miner.data_structure import IMDataStructureTranslucent
from translucent_discovery.translucent_inductive_miner.fall_through.activity_concurrent import ActivityConcurrentTranslucent
from pm4py.util.compression import util as comut

# No projection performed here, so it can stay this way for TCL
#TODO: In the future, we could use translucent information to see if an activity can really only occur once per trace
class ActivityOncePerTraceTranslucent(ActivityConcurrentTranslucent):

    @classmethod
    def _get_candidate(cls, obj: IMDataStructureTranslucent, pool=None, manager=None, parameters: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        candidates = set(comut.get_alphabet(obj.data_structure)) 
        
        for t in obj.data_structure:
            # Use a Counter to count occurrences of each activity in the trace
            activity_counts = Counter(t)
            # Create a set of activities that occur exactly once in the trace
            activities_once = {
                activity
                for activity, count in activity_counts.items()
                if count == 1
            }
            # Intersect with the existing candidates
            candidates &= activities_once
            # Early exit if no candidates remain
            if not candidates:
                return None

        # Deterministic behavior
        candidates = sorted(list(candidates))  # more deterministic behavior
        return candidates[0] if candidates else None
