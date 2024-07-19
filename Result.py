import pymonntorch as pmt


class Result(pmt.Behavior):
    def initialize(self, ng):
        ng.spike_counter = ng.vector(0)
        ng.network.winner = 0

    def forward(self, ng):
        ng.spike_counter += ng.spike.byte()
        if ng.network.iteration % ng.network.input_period == 0:
            result = ng.spike_counter == ng.spike_counter.max()
            print(
                f"correct label: {ng.network.current_inp_indx}, predicted label: {((result == True).nonzero(as_tuple=True)[0])}"
            )
            print(ng.spike_counter)
            ng.spike_counter = ng.vector(0)
