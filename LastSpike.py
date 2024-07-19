import pymonntorch as pmt


class LastSpike(pmt.Behavior):  # reset network memory
    def initialize(self, ng):
        ng.last_spike = ng.vector(-1000)

    def forward(self, ng):
        ng.last_spike[ng.spike] = ng.network.iteration * ng.network.dt
