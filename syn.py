# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

from pymonntorch import Behavior
import torch


class Test(Behavior):
    def initialize(self, sg):
        sg.j0 = self.parameter("j0", 1)
        self.tau = self.parameter("tau", 2)
        sg.W = self.parameter("W", required=True)
        self.N = sg.src.size
        self.C = self.N
        sg.C = self.C
        sg.I = sg.dst.vector()

    def forward(self, sg):
        sg.I += (
            -sg.I / self.tau * sg.network.dt
            + torch.sum(sg.W[sg.src.spike], axis=0) / sg.network.dt
        )


class RandomWeight(Behavior):
    def initialize(self, sg):
        sg.j0 = self.parameter("j0", 1)
        self.tau = self.parameter("tau", 10)
        sg.W = sg.matrix(mode="normal(0.5, 0.3)")
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        self.N = sg.src.size
        self.C = self.N
        sg.C = self.C
        sg.I = sg.dst.vector()

    def forward(self, sg):
        sg.I += (
            -sg.I / self.tau * sg.network.dt
            + torch.sum(sg.W[sg.src.spike], axis=0) / sg.network.dt
        )


class FullyConnected(Behavior):
    def initialize(self, sg):
        sg.j0 = self.parameter("j0", 10)
        self.tau = self.parameter("tau", 10)
        self.variation = self.parameter("variation", 0)
        self.N = sg.src.size
        self.ignore_last_column = self.parameter("ignore_last_column", False)
        self.C = self.N
        sg.C = self.C
        sg.W = sg.matrix(
            mode=f"normal({(sg.j0 / self.N)},{((sg.j0 / self.N )* self.variation )/ 100})"
        )
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector(0)
        if self.ignore_last_column:
            sg.W[:, -1] = 0

    def forward(self, sg):
        sg.I += (
            -sg.I / self.tau * sg.network.dt
            + torch.sum(sg.W[sg.src.spike], axis=0) / sg.network.dt
        )


class RandomConnectivity(Behavior):
    def initialize(self, sg):
        sg.j0 = self.parameter("j0", 10)
        self.tau = self.parameter("tau", 10)
        self.p = self.parameter("p", 20) / 100
        self.variation = self.parameter("variation", 0) / 100
        self.N = sg.src.size
        self.C = self.N * self.p
        sg.C = self.C
        base_val = sg.j0 / self.C
        sg.W = sg.matrix(mode=f"normal({base_val},{abs(base_val* self.variation) })")
        prob_matrix = torch.rand_like(sg.W) > self.p
        sg.W[prob_matrix] = 0
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector(0)

    def forward(self, sg):
        sg.I += (
            -sg.I / self.tau * sg.network.dt
            + torch.sum(sg.W[sg.src.spike], axis=0) / sg.network.dt
        )


class RandomConnectivityFix(Behavior):
    def initialize(self, sg):
        sg.j0 = self.parameter("j0", 10)
        self.tau = self.parameter("tau", 10)
        self.C = self.parameter("connection_number", 20)
        sg.C = self.C
        self.variation = self.parameter("variation", 0) / 100
        self.N = sg.src.size
        base_val = sg.j0 / self.C
        sg.W = sg.matrix(mode=f"normal({base_val},{abs(base_val* self.variation )})")

        prob_matrix = torch.zeros_like(sg.W)
        for i in range(sg.dst.size):
            prm = torch.randperm(sg.src.size)
            prob_matrix[:, i] = prm

        prob_matrix = prob_matrix >= self.C

        sg.W[prob_matrix] = 0
        if sg.src == sg.dst:
            sg.W.fill_diagonal_(0)
        sg.I = sg.dst.vector(0)

    def forward(self, sg):
        sg.I += (-sg.I / self.tau) * sg.network.dt + torch.sum(
            sg.W[sg.src.spike], axis=0
        ) / sg.network.dt


class STDP(Behavior):
    def initialize(self, sg):
        self.AP = self.parameter("AP", 0.1) / sg.C
        self.AM = self.parameter("AM", -0.03) / sg.C
        self.tau_src = self.parameter("tau_src", 3)
        self.tau_dst = self.parameter("tau_dst", 2)
        self.hight_limit = self.parameter("hight_limit", 0) or (sg.j0 / sg.C) * 25
        self.low_limit = self.parameter("low_limit", 0) or (-sg.j0 / sg.C) * 25
        self.normalize = self.parameter("normalize", True)

        sg.src.X = sg.src.vector(dtype=torch.float64)  # source
        sg.dst.X = sg.dst.vector(dtype=torch.float64)  # destination
        # print(sg.dst.X.shape, sg.src.X.shape)
        # print(sg.dst.spike.shape, sg.src.spike.shape)

    def forward(self, sg):
        src = sg.src
        dst = sg.dst
        absW = abs(sg.W)
        dW = (
            self.AP
            * sg.W
            * (self.hight_limit - absW)
            * (
                src.X.reshape(-1, 1)
                @ (dst.spike.byte().to(torch.float64).reshape(1, -1))
            )
        ) + (
            self.AM
            * absW
            * (-self.low_limit + sg.W)
            * (src.spike.byte().to(torch.float64).reshape(-1, 1) @ dst.X.reshape(1, -1))
        )
        if self.normalize:
            dW -= dW.sum(axis=0) / sg.src.size
        sg.W += dW

        src.X += -(src.X / self.tau_src) + src.spike.byte()
        dst.X += -(dst.X / self.tau_dst) + dst.spike.byte()


# class STDP(Behavior):
#     def initialize(self, sg):
#         self.AP = self.parameter("AP", 0.1)
#         self.AM = self.parameter("AM", -0.03)
#         self.tau_src = self.parameter("tau_src", 3)
#         self.tau_dst = self.parameter("tau_dst", 2)
#         sg.src.X = sg.src.vector(dtype=torch.float64)  # source
#         sg.dst.X = sg.dst.vector(dtype=torch.float64)  # destination
#         # print(sg.dst.X.shape, sg.src.X.shape)
#         # print(sg.dst.spike.shape, sg.src.spike.shape)

#     def forward(self, sg):
#         src = sg.src
#         dst = sg.dst
#         sg.W += (
#             self.AP
#             * sg.W
#             * (
#                 src.X.reshape(-1, 1)
#                 @ (dst.spike.byte().to(torch.float64).reshape(1, -1))
#             )
#         ) + (
#             self.AM
#             * sg.W
#             * (src.spike.byte().to(torch.float64).reshape(-1, 1) @ dst.X.reshape(1, -1))
#         )

#         src.X += -(src.X / self.tau_src) + src.spike.byte() / sg.C
#         dst.X += -(dst.X / self.tau_dst) + dst.spike.byte() / sg.C


class RSTDP(Behavior):
    def initialize(self, sg):
        self.AP = self.parameter("AP", 0.1) / sg.C
        self.AM = self.parameter("AM", -0.03) / sg.C
        self.reset = self.parameter("reset", False) / sg.C
        self.normalize = self.parameter("normalize", True)
        self.tau_src = self.parameter("tau_src", 3)
        self.tau_dst = self.parameter("tau_dst", 2)
        self.tau_c = self.parameter("tau_c", 10)
        self.hight_limit = self.parameter("hight_limit", 0) or (sg.j0 / sg.C) * 25
        self.low_limit = self.parameter("low_limit", 0) or (-sg.j0 / sg.C) * 25

        sg.C_learning = sg.matrix(0)

        sg.src.X = sg.src.vector(dtype=torch.float64)  # source
        sg.dst.X = sg.dst.vector(dtype=torch.float64)  # destination

    def forward(self, sg):
        src = sg.src
        dst = sg.dst
        absW = abs(sg.W)

        src.X += -(src.X / self.tau_src) + src.spike.byte()
        dst.X += -(dst.X / self.tau_dst) + dst.spike.byte()
        sg.C_learning += (
            -(sg.C_learning / self.tau_c)
            + (src.X.reshape(-1, 1) @ dst.spike.byte().to(torch.float64).reshape(1, -1))
            # - (src.spike.byte().to(torch.float64).reshape(-1, 1) @ dst.X.reshape(1, -1))
        )
        if sg.network.iteration % sg.network.input_period == 0:
            # sg.W += (
            #     (
            #         sg.C_learning
            #         - (
            #             (sg.C_learning.sum(axis=0) / (sg.src.size))
            #             if self.normalize
            #             else 0
            #         )
            #     )
            #     @ torch.diag(sg.network.dopamine)
            #     * abs((self.hight_limit - sg.W) * ((self.low_limit - sg.W)))
            #     # * abs((self.hight_limit - sg.W) * (-self.low_limit + sg.W))
            # )
            dW = (
                (
                    sg.C_learning
                    - (
                        (sg.C_learning.sum(axis=0) / (sg.src.size))
                        if self.normalize
                        else 0
                    )
                )
                @ torch.diag(sg.network.dopamine)
                * abs((self.hight_limit - sg.W) * ((self.low_limit - sg.W)))
                # * abs((self.hight_limit - sg.W) * (-self.low_limit + sg.W))
            )

            if self.normalize:
                dW -= dW.sum(axis=0) / sg.src.size
            sg.W += dW
            sg.C_learning *= 0
