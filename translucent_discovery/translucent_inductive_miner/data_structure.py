import copy

from pandas import DataFrame
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructure
from pm4py.algo.discovery.inductive.dtypes.im_ds import IMDataStructureLog
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.dfg.obj import DFG
from typing import TypeVar, Generic, Optional, Union
from translucent_discovery.translucent_inductive_miner.tDFG import discover_dfg, discover_frequent_dfg, discover_dfg_tcl, discover_frequent_dfg_tcl
from pm4py.objects.dfg.obj import DFG
from pm4py.util.compression import util as comut
from pm4py.util.compression.dtypes import UVCL
from translucent_discovery.translucent_inductive_miner.utils import get_translucent_trace_variants
from translucent_discovery.translucent_inductive_miner.translucent_datatype import TCL, tcl_to_uvcl
import pm4py


#Elias: pm4py Äquivalent: IMDataStructureUVCL in algo/discovery/inductive/dtypes/im_ds.py
class IMDataStructureTranslucent(IMDataStructureLog[UVCL]):
    def __init__(self, obj: UVCL, tcl: TCL, dfg: Optional[DFG] = None, frequent=False, tdfg = None, parameters = {}):
        if obj is None:
            obj = tcl_to_uvcl(tcl)
        super().__init__(obj)
        if dfg is None:
            self._dfg = comut.discover_dfg_uvcl(self._obj)
        else:
            self._dfg = dfg
        self._tcl = tcl
        #Elias: #debug: display dfg
        #pm4py.view_dfg(self._dfg.graph, self._dfg.start_activities, self._dfg.end_activities)
        self._frequent = frequent
        if not frequent:
            self._tdfg = discover_dfg_tcl(tcl, parameters=parameters)
        else:
            if tdfg is None:
                self._tdfg = discover_frequent_dfg_tcl(tcl, parameters=parameters)
            else:
                self._tdfg = tdfg
        #Elias: #debug: display tdfg
        #pm4py.view_dfg(self._tdfg.graph, self._tdfg.start_activities, self._tdfg.end_activities)

    @property
    def dfg(self) -> DFG:
        return self._dfg

    @property
    def tdfg(self) -> DFG:
        return self._tdfg

    @property
    def frequent(self):
        return self._frequent
    
    @property
    def tcl(self):
        return self._tcl

