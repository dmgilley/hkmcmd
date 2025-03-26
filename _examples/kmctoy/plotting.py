#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import matplotlib.font_manager as mplfont_manager
import matplotlib.cm as mplcm
from matplotlib.lines import Line2D


def formatmpl0(
    figsize=(12, 8),
    labels=[],
    alpha=0.7,
    xstyle="sci",
    xscilimits=(-2, 2),
    ystyle="sci",
    yscilimits=(-2, 2),
    xlabel="",
    ylabel="",
):

    # create figure and axis
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    # create a colormap
    colorlist = ["#cd6155", "#eb984e", "#f4d03f", "#52be80", "#5499c7", "#af7ac5"]
    cmap = mplcm.get_cmap("nipy_spectral")
    if len(labels) > len(colorlist):
        colormap = {
            _: cmap(idx / len(labels), alpha=alpha) for idx, _ in enumerate(labels)
        }
    else:
        colormap = {_: colorlist[idx] for idx, _ in enumerate(labels)}

    # set the style of the axes and the text color
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["text.color"] = "black"

    # format the x- and y-axes
    tickfont = {"fontsize": 24, "weight": "bold"}
    plt.xticks(**tickfont)
    plt.yticks(**tickfont)
    ax.ticklabel_format(axis="y", style=ystyle, scilimits=yscilimits)
    ax.ticklabel_format(axis="x", style=xstyle, scilimits=xscilimits)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        pad=10,
        direction="out",
        width=3,
        length=6,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=24,
        pad=10,
        direction="out",
        width=2,
        length=4,
    )
    [j.set_linewidth(3) for j in ax.spines.values()]

    # generate the legend
    legendprop = {
        "handles": [mplpatches.Patch(color=colormap[_], alpha=alpha) for _ in labels],
        "fontsize": 24,
        "loc": 1,
        "handlelength": 2.5,
        "frameon": False,
        "shadow": False,
        "framealpha": 1.0,
        "edgecolor": "#FFFFFF",
    }

    # set the x and y axis
    # ax.set_xlabel(xlabel,fontsize=26,fontweight='black',color='black',labelpad=10)
    # ax.set_ylabel(ylabel,fontsize=26,fontweight='black',color='black',labelpad=10)
    ax.set_xlabel(xlabel, fontsize=32, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=32, fontweight="bold", labelpad=10)

    return fig, ax, colormap, legendprop


def formatmpl(
    figsize=(12, 8), xstyle="sci", xscilimits=(-2, 2), ystyle="sci", yscilimits=(-2, 2)
):

    # create figure and axis
    fig = mpl.figure.Figure(figsize=figsize)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["font.size"] = 24
    mpl.rcParams["axes.labelsize"] = 24
    mpl.rcParams["axes.linewidth"] = 2.0

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))

    ax.tick_params(axis="both", which="major", length=10, width=2, labelsize=24, pad=10)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 4), useMathText=True)

    ax.set_xlabel("", labelpad=10, fontsize=24)
    ax.set_ylabel("", labelpad=10, fontsize=24)

    legendprop = {
        "fontsize": 36,
        "loc": 1,
        "handlelength": 2.5,
        "frameon": False,
        "shadow": False,
        "framealpha": 1.0,
        "edgecolor": "#FFFFFF",
    }

    return fig, ax, legendprop


colors = [
    "#D3582C",  # Savoie red
    "#E9AEAB",  # Savoie light pink
    "#FE7378",  # strong pink
    "#ee7272",  # pink
    "#ffb9b9",  # light pink
    "#D8A4A7",  # light pink
    "#C39C9C",  # purple-pink
    "#F2B4B5",  # pink
    "#F4D0CF",  # pink
    "#FFEAF3",  # light pink
    "#FAEFEE",  # pale pink
    "#FFC004",  # yellow
    "#FDFFEA",  # light yellow
    "#5FC89F",  # Savoie green
    "#72D9AC",  # green - neon
    "#ABBA97",  # green - washed out
    "#9CB7AB",  # green - washed out
    "#87B085",  # green - darker
    "#9BC49A",  # green
    "#B6E1C6",  # green - light
    "#E3FFE4",  # green - light
    "#74D5D5",  # teal - neon
    "#3e9b96",  # teal
    "#9ed9cc",  # light teal
    "#DDEEE6",  # teal - lightest
    "#2656B8",  # Savoie blue
    "#55A0B5",  # blue - pure
    "#94CFDD",  # blue
    "#E4F8FF",  # blue - light
    "#B2D7E0",  # blue - steel
    "#372f53",  # navy
    "#8B95D9",  # violet - blue/violet
    "#B8CBDF",  # violet - blue/violet
    "#c0A8cb",  # violet
    "#F0EDFE",  # violet
    "#6d0202",  # burgundy
    "#8D8B8B",  # Savoie dark grey
    "#C3C2C2",  # Savoie light grey
]
