from typing import TypeVar, Generic, Dict, Any, Optional

from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureUVCL, IMDataStructureLog
from pm4py.algo.discovery.inductive.fall_through.empty_traces import EmptyTracesUVCL
from pm4py.algo.discovery.inductive.variants.abc import InductiveMinerFramework
from translucent_discovery.translucent_inductive_miner.framework_frequent import InductiveMinerFrequentFrameworkTranslucent
from pm4py.algo.discovery.inductive.variants.instances import IMInstance
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.dfg.obj import DFG
from copy import copy
from enum import Enum
from pm4py.util import exec_utils


T = TypeVar('T', bound=IMDataStructureLog)


class IMFParameters(Enum):
    NOISE_THRESHOLD = "noise_threshold"


class IMF(Generic[T], InductiveMinerFrequentFrameworkTranslucent[T]):

    def instance(self) -> IMInstance:
        return IMInstance.IMf