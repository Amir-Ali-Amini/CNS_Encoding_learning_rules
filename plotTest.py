from matplotlib import pyplot as plt
import torch
import numpy as np
from scipy.spatial.distance import cosine


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
    env_recorder_index=461,
):
    n = len(ngs)
    fig_counter = memoized_number()
    fig, axd = plt.subplot_mosaic(
        """
        BC
        """,
        layout="constrained",
        # "image" will contain a square image. We fine-tune the width so that
        # there is no excess horizontal or vertical margin around the image.
        figsize=(12 * scaling_factor, 3 * scaling_factor),
    )

    fig.suptitle(
        title or "Plot",
        fontsize=(label_font_size + 4) * scaling_factor,
    )
    axd["B"].scatter(
        ngs[0][env_recorder_index, 0].variables["spike"][:, 0].cpu(),
        ngs[0][env_recorder_index, 0].variables["spike"][:, 1].cpu(),
        label=f"{ngs[0].tag}",
    )
    axd["B"].set_ylabel(
        "spike (neuron number)", fontsize=label_font_size * scaling_factor
    )
    axd["B"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["B"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["B"].set_xlim(0, net.iteration)
    axd["B"].set_ylim(-1, ngs[0].size)

    # axd["B"].set_title(
    #     "1st Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    # )
    axd["B"].set_xlabel(
        f"{fig_counter()} time ({ngs[0].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["B"].grid()

    axd["C"].scatter(
        ngs[1][env_recorder_index, 0].variables["spike"][:, 0].cpu(),
        ngs[1][env_recorder_index, 0].variables["spike"][:, 1].cpu(),
        label=f"{ngs[1].tag}",
    )
    axd["C"].set_ylabel(
        "spike (neuron number)", fontsize=label_font_size * scaling_factor
    )
    axd["C"].xaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["C"].yaxis.set_tick_params(labelsize=(label_font_size - 2) * scaling_factor)
    axd["C"].set_xlim(0, net.iteration)
    axd["C"].set_ylim(-1, ngs[1].size)
    # axd["C"].set_title(
    #     "2nd Neuron Group", fontsize=(label_font_size + 1) * scaling_factor
    # )
    axd["C"].set_xlabel(
        f"{fig_counter()} time ({ngs[1].tag})",
        fontsize=label_font_size * scaling_factor,
    )
    axd["C"].grid()

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    fig.show()
