from matplotlib import pyplot as plt
import torch
import numpy as np


def memoized_number():
    dic = [0]

    def inside():
        dic[0] += 1
        return f"plot({dic[0]}):"

    return inside


def plot(
    net,
    title=None,
    ngs=[],
    scaling_factor=3,
    label_font_size=8,
    print_activities=True,
    recorder_index=9,
    env_recorder_index=10,
):
    n = len(ngs)
    fig_counter = memoized_number()
    fig = plt.figure(
        figsize=(
            12 * scaling_factor,
            (6 if print_activities else 2) * scaling_factor,
        ),
    )

    fig.suptitle(
        title or "Plot",
        fontsize=(label_font_size + 4) * scaling_factor,
    )
    if not print_activities:
        gs = fig.add_gridspec(1, n * 2)
    else:
        gs = fig.add_gridspec(3, n * 2)

    for i in range(n):
        if not print_activities:
            ax1 = fig.add_subplot(gs[:, :])
        else:
            ax1 = fig.add_subplot(gs[:-1, i * 2 : i * 2 + 2])
            ax2 = fig.add_subplot(gs[-1, i * 2 : i * 2 + 2])
        ax1.scatter(
            ngs[i][env_recorder_index, 0].variables["spike"][:, 0].cpu(),
            ngs[i][env_recorder_index, 0].variables["spike"][:, 1].cpu(),
            label=f"{ngs[i].tag}",
        )
        ax1.set_ylabel(
            "spike (neuron number)", fontsize=label_font_size * scaling_factor
        )
        ax1.xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
        ax1.yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
        ax1.set_xlim(0, ngs[0].network.iteration)
        ax1.set_ylim(-1, ngs[0].size)

        ax1.set_title("Scatter-Plot", fontsize=(label_font_size + 1) * scaling_factor)
        ax1.set_xlabel(
            f"{fig_counter()} time ({ngs[i].tag})",
            fontsize=label_font_size * scaling_factor,
        )
        if print_activities:
            ax2.plot(ngs[i][recorder_index, 0].variables["T"].cpu())
            ax2.set_ylabel(f"activity", fontsize=label_font_size * scaling_factor)

            ax2.xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
            ax2.yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
            ax2.set_xlim(0, ngs[0].network.iteration)
            ax2.set_title("Activity", fontsize=(label_font_size + 1) * scaling_factor)
            ax2.set_xlabel(
                f"{fig_counter()} time ({ngs[i].tag})",
                fontsize=label_font_size * scaling_factor,
            )

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    fig.show()
