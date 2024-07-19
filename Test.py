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
from plotTest import plot as plot1
import Result


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


def Test(
    input_data,
    parameters,
    title="",
    model_inp=mdl.LIF(),
    model_out=LIF(tau_m=3, R=10),
    DEVICE=get_device(force_cpu=True)[0],
    dt=1,
    iteration=100,
    print_plots=True,
    tau=1,
):
    net = pmt.Network(
        device=DEVICE, dtype=torch.float32, behavior={1: TimeResolution(dt=dt)}
    )

    ng1_inp1 = pmt.NeuronGroup(
        size=len(input_data.init_kwargs["data"][0]),
        net=net,
        tag="ng-input",
        behavior={
            2: parameters["current_inp"],
            4: dnd.InpSyn(),
            5: model_inp,
            6: input_data,
            7: act.Activity(),
            9: pmt.Recorder(
                variables=[
                    "u",
                    "I",
                    "I_inp",
                    "T",
                ],
                tag="ng1_inp1_rec, ng1_inp1_recorder",
            ),
            10: pmt.EventRecorder("spike", tag="ng1_inp1_evrec"),
        },
    )

    ng2_out = pmt.NeuronGroup(
        size=parameters["out_size"],
        net=net,
        tag="ng-output",
        behavior={
            2: parameters["current_out"],
            4: dnd.InpSyn(),
            5: model_out,
            7: act.Activity(),
            8: Result.Result(),
            9: pmt.Recorder(
                variables=[
                    "u",
                    "I",
                    "I_inp",
                    "T",
                ],
                tag="ng2_out_rec, ng2_out_recorder",
            ),
            10: pmt.EventRecorder("spike", tag="ng2_out_evrec"),
        },
    )
    sg_inp_out = pmt.SynapseGroup(
        net=net,
        src=ng1_inp1,
        dst=ng2_out,
        tag="ex1_out",
        behavior={
            3: syn.Test(W=parameters["W"], tau=tau),
        },
    )

    net.initialize(info=False)

    net.simulate_iterations(iteration)
    print_plots and plot1(
        net,
        ngs=[ng1_inp1, ng2_out],
        print_sum_activities=True,
        scaling_factor=parameters["scaling_factor"],
        title=title + "\n",
    )


# %%
