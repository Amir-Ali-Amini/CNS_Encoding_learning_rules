from pymonntorch import Behavior
import torch

class dI(Behavior):
	def initialize(self, ng):
		ng.di = ng.vector(0)
		ng.prev_I = torch.tensor(ng.I)

	def forward(self, ng):
		ng.di = (ng.I - ng.prev_I) / ng.network.dt
		ng.prev_I = torch.tensor(ng.I)

