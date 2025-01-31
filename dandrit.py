# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

from pymonntorch import Behavior
import torch


class InpSyn(Behavior):
    def initialize(self, ng):
        ng.I = ng.I_inp.clone().detach()

    def forward(self, ng):
        ng.I = ng.I_inp.clone().detach()
        for syn in ng.afferent_synapses["All"]:
            ng.I += syn.I
