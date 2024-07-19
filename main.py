# %%
import pymonntorch as pmt
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine

import model as mdl
import syn
import dandrit as dnd
import current as cnt
import activity as act
from getDevice import get_device
from dt import TimeResolution
from plot1 import plot
import InputData
import readImage


def LIF(**arg):
    tag = []
    for key in arg.keys():
        tag += [f"{key}={arg[key]}"]
    return mdl.LIF(**arg, tag="|".join(tag))


def ELIF(**arg):
    tag = []
    for key in arg.keys():
        tag += [f"{key}={arg[key]}"]
    return mdl.ELIF(**arg, tag="|".join(tag))


def AELIF(**arg):
    tag = []
    for key in arg.keys():
        tag += [f"{key}={arg[key]}"]
    return mdl.AELIF(**arg, tag="|".join(tag))


def simulate_two_neuron_group(
    input_data,
    title="",
    model_ex=mdl.LIF(),
    model_inh=LIF(tau_m=3, R=10),
    syn_model_ex_ex=syn.RandomConnectivityFix(
        j0=0, connection_number=100, tau=1, variation=30
    ),
    syn_model_ex_inh=syn.FullyConnected(j0=10, variation=50, tau=2),
    syn_model_inh_ex=syn.RandomConnectivityFix(j0=0, connection_number=30, tau=1),
    current_ex=cnt.SteadyCurrent(value=6 and 0),
    current_inh=cnt.SteadyCurrent(value=0.2 and 0),
    DEVICE=get_device(force_cpu=True)[0],
    dt=1,
    iteration=100,
    n_size=1000,
    print_plots=True,
    ex_size=0,
    in_size=0,
    training_rule=syn.STDP(AP=3, AM=-5),
):
    net = pmt.Network(
        device=DEVICE, dtype=torch.float32, behavior={1: TimeResolution(dt=dt)}
    )

    ng1_ex1 = pmt.NeuronGroup(
        size=ex_size or len(input_data.init_kwargs["data"][0]) or int(n_size * 0.8),
        net=net,
        tag="ng1_ex",
        behavior={
            2: current_ex,
            3: InputData.ResetMemory(),
            4: dnd.InpSyn(),
            5: model_ex,
            6: input_data,
            7: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T", "X"],
                tag="ng1_ex1_rec, ng1_ex1_recorder",
            ),
            10: pmt.EventRecorder("spike", tag="ng1_ex1_evrec"),
        },
    )

    ng2_inh = pmt.NeuronGroup(
        size=in_size or int(n_size * 0.2),
        net=net,
        tag="ng2_inh",
        behavior={
            2: current_inh,
            4: dnd.InpSyn(),
            5: model_inh,
            7: act.Activity(),
            9: pmt.Recorder(
                variables=["u", "I", "I_inp", "T", "X"],
                tag="ng2_inh_rec, ng2_inh_recorder",
            ),
            10: pmt.EventRecorder("spike", tag="ng2_inh_evrec"),
        },
    )

    sg_ex_ex = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng1_ex1,
        tag="ex1_ex1",
        behavior={3: syn_model_ex_ex},
    )
    sg_ex_inh = pmt.SynapseGroup(
        net=net,
        src=ng1_ex1,
        dst=ng2_inh,
        tag="ex1_inh",
        behavior={
            3: syn_model_ex_inh,
            8: training_rule,
            9: pmt.Recorder(variables=["W"], tag="sg_ex_inh"),
        },
    )
    sg_inh_ex = pmt.SynapseGroup(
        net=net,
        src=ng2_inh,
        dst=ng1_ex1,
        tag="inh_ex1",
        behavior={3: syn_model_inh_ex},
    )

    net.initialize(info=False)

    net.simulate_iterations(iteration)

    plot_title = f""

    print_plots and plot(
        net,
        plot_title,
        [ng1_ex1],
        # [ng1_ex1, ng2_inh],
        print_sum_activities=True,
        scaling_factor=2,
        # slice=-80,
    )

    # plt.plot(sg_ex_inh[9, 0].variables["W"][:, :, 0].cpu())
    # plt.show()

    # plt.plot(sg_ex_inh[9, 0].variables["W"][:, :, 1].cpu())
    # plt.show()

    # plt.plot(ng1_ex1[9, 0].variables["X"].cpu())
    # plt.show()
    # similarity = []
    # for i in range(iteration):
    #     similarity.append(
    #         1
    #         - cosine(
    #             sg_ex_inh[9, 0].variables["W"][i, :, 0].cpu(),
    #             sg_ex_inh[9, 0].variables["W"][i, :, 1].cpu(),
    #         )
    #     )
    # plt.plot(similarity)
    # plt.title("similarity")
    # plt.show()
    return net


# size = 50
# iteration = 1000
# time = 30
# encoded_matrix = torch.zeros(2, time, 100) != 0
# for j in range(time):
#     encoded_matrix[0][j][:size] = torch.randperm(size) > size * 0.9
# for j in range(time):
#     encoded_matrix[1][j][100 - size :] = torch.randperm(size) > size * 0.9

# net = simulate_two_neuron_group(
#     iteration=iteration,
#     in_size=2,
#     ex_size=100,
#     current_inh=cnt.SinCurrent(value=0),
#     syn_model_ex_inh=syn.FullyConnected(j0=12, variation=50, tau=2),
#     input_data=InputData.InputMatrix(
#         encoded_matrix=encoded_matrix,
#         time=time,
#         input_period=time + 1,
#         method="lin",
#     ),
# )


# iteration = 25
# net = simulate_two_neuron_group(
#     iteration=iteration,
#     in_size=2,
#     input_data=InputData.Encode(
#         data=torch.tensor([[103, 100, 15, 3, 35, 56, 77, 88, 89, 10]]),
#         # range=255,
#         time=20,
#         input_period=20,
#         method="poisson",
#     ),
# )
iteration = 25
net = simulate_two_neuron_group(
    iteration=iteration,
    in_size=2,
    syn_model_ex_inh=syn.FullyConnected(j0=1.2, variation=50, tau=2),
    input_data=InputData.Encode(
        data=torch.tensor(
            [
                readImage.readImage("1.tif", prefix="./images/"),
                readImage.readImage("2.tif", prefix="./images/"),
            ]
        ),
        range=255,
        time=30,
        method="poisson",
    ),
)
print(net)
# %%
iteration = 25
net = simulate_two_neuron_group(
    iteration=iteration,
    in_size=100,
    input_data=InputData.TTFS(
        data=torch.cat([i * torch.randperm(50).reshape(1, -1) for i in range(1, 3)]),
        time=30,
        method="exp",
    ),
)


# simulate_two_neuron_group(
#     iteration=40,
#     in_size=1,
#     input_data=InputData.TTFS(
#         data=torch.tensor([100, 3]),
#         time=int(40),
#         theta=10000,
#         method="exp",
#     ),
# )
# simulate_two_neuron_group(
#     iteration=40,
#     in_size=1,
#     input_data=InputData.TTFS(
#         data=torch.tensor([100, 103, 88, 89, 78, 56, 34, 15, 10, 3]),
#         time=int(40),
#         theta=10000,
#         method="exp",
#     ),
# )
# iteration = 1000
# simulate_two_neuron_group(
#     iteration=iteration,
#     in_size=30,
#     input_data=InputData.TTFS(
#         data=[torch.randperm(1000), torch.randperm(1000)],
#         time=100,
#         # theta=10000,
#         method="exp",
#     ),
# )
# for i in range(8):
#     simulate_two_neuron_group(
#         iteration=40,
#         in_size=1,
#         input_data=InputData.Number(
#             data=5 + i / 10,
#             time=int(5),
#         ),
#         ex_size=10,
#     )

# %%
