import pymonntorch as pmt
import torch


class Reward(pmt.Behavior):
    def initialize(self, ng):
        self.reward = self.parameter("reward", 0.02)
        self.punishment = self.parameter("punishment", -0.01)
        self.tau = self.parameter("tau", 30)
        ng.spike_counter = ng.vector(0)
        ng.network.winner = 0
        ng.network.dopamine = ng.vector()

    def forward(self, ng):
        ng.spike_counter += ng.spike.byte()
        if ng.network.iteration % ng.network.input_period == 0:
            result = ng.spike_counter == ng.spike_counter.max()
            ng.network.dopamine = ng.vector(0)
            ng.network.dopamine[result] = self.punishment
            ng.network.dopamine[ng.network.current_inp_indx] = self.reward
            ng.network.dopamine /= ng.size
            ng.spike_counter = ng.vector(0)
