# This script opens the JSON results in this directory and displays them as
# CDFs.
import argparse
import itertools
import json
import matplotlib.pyplot as plot
import numpy

def convert_values_to_cdf(values):
    """Takes a 1-D list of values and converts it to a CDF representation. The
    CDF consists of a vector of times and a vector of percentages of 100."""
    if len(values) == 0:
        return [[], []]
    values.sort()
    total_size = float(len(values))
    current_min = values[0]
    count = 0.0
    data_list = [values[0]]
    ratio_list = [0.0]
    for v in values:
        count += 1.0
        if v > current_min:
            data_list.append(v)
            ratio_list.append((count / total_size) * 100.0)
            current_min = v
    data_list.append(values[-1])
    ratio_list.append(100)
    # Convert seconds to milliseconds
    for i in range(len(data_list)):
        data_list[i] *= 1000.0
    return [data_list, ratio_list]

def add_plot_padding(axes):
    """Takes matplotlib axes, and adds some padding so that lines close to
    edges aren't obscured by tickmarks or the plot border."""
    y_limits = axes.get_ybound()
    y_range = y_limits[1] - y_limits[0]
    y_pad = y_range * 0.05
    x_limits = axes.get_xbound()
    x_range = x_limits[1] - x_limits[0]
    x_pad = x_range * 0.05
    axes.set_ylim(y_limits[0] - y_pad, y_limits[1] + y_pad)
    axes.set_xlim(x_limits[0] - x_pad, x_limits[1] + x_pad)
    axes.xaxis.set_ticks(numpy.arange(x_limits[0], x_limits[1] + x_pad,
        x_range / 5.0))
    axes.yaxis.set_ticks(numpy.arange(y_limits[0], y_limits[1] + y_pad,
        y_range / 5.0))

all_styles = None
def get_line_styles():
    """Returns a list of line style possibilities, that includes more options
    than matplotlib's default set that includes only a few solid colors."""
    global all_styles
    if all_styles is not None:
        return all_styles
    color_options = ["black"]
    dashes_options = [
        [1, 0],
        [4, 1, 4, 1],
        [4, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    marker_options = [
        None,
        "o",
        "v",
        "s",
        "*",
        "+",
        "D"
    ]
    # Build a combined list containing every style combination.
    all_styles = []
    for m in marker_options:
        for d in dashes_options:
            for c in color_options:
                to_add = {}
                if m is not None:
                    to_add["marker"] = m
                    to_add["markevery"] = 0.1
                to_add["c"] = c
                to_add["dashes"] = d
                all_styles.append(to_add)
    return all_styles

def get_plot(datasets, data_keys_and_titles):
    cdfs = []
    titles = []
    for v in data_keys_and_titles:
        title = v[0]
        dataset_key = v[1]
        titles.append(title)
        with open(datasets[dataset_key]) as f:
            data = json.loads(f.read())
            cdf_data = convert_values_to_cdf(data[1:])
            cdfs.append(cdf_data)
    figure = plot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.autoscale(enable=True, axis="both", tight=True)
    style_cycler = itertools.cycle(get_line_styles())
    for i in range(len(cdfs)):
        cdf = cdfs[i]
        title = titles[i]
        axes.plot(cdf[0], cdf[1], label=title, lw=1.5, **(next(style_cycler)))
    add_plot_padding(axes)
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("% <= X")
    legend = plot.legend()
    legend.set_draggable(True)
    return figure

def print_stats_latex(filename, title):
    """ Prints some statistics about the data to stdout. """
    data = []
    with open(filename) as f:
        data = json.loads(f.read())
    data = data[1:]
    for i in range(len(data)):
        data[i] *= 1000.0
    min_time = min(data)
    max_time = max(data)
    median = numpy.median(data)
    mean = numpy.average(data)
    std_dev = numpy.std(data)
    fmt = "%s & %.02f & %.02f & %.02f & %.02f & %.02f " + r'\\'
    print(fmt % (title, min_time, max_time, median, mean, std_dev))

def print_stats_human_readable(filename, title):
    """ Prints some statistics about the data to stdout. """
    data = []
    with open(filename) as f:
        data = json.loads(f.read())
    data = data[1:]
    for i in range(len(data)):
        data[i] *= 1000.0
    min_time = min(data)
    max_time = max(data)
    median = numpy.median(data)
    mean = numpy.average(data)
    std_dev = numpy.std(data)
    fmt = "%s \t %.02f \t %.02f \t %.02f \t %.02f \t %.02f"
    print(fmt % (title, min_time, max_time, median, mean, std_dev))

if __name__ == "__main__":
    result_dir = "./results/"
    datasets = {
        "Titan V": "pytorch_times_gpu_titan_v.json",
        "GTX 1060": "pytorch_times_gtx_1060.json",
        "GTX 970": "pytorch_times_gtx_970.json",
        "RX 570 (32 CUs, isolated)": "pytorch_times_gpu.json",
        "RX 570 (16 CUs, isolated)": "pytorch_times_gpu_16_CUs.json",
        "RX 570 (8 CUs, isolated)": "pytorch_times_gpu_8_CUs.json",
        "RX 570 (4 CUs, isolated)": "pytorch_times_gpu_4_CUs.json",
        "RX 570 (2 CUs, isolated)": "pytorch_times_gpu_2_CUs.json",
        "RX 570 (1 CU, isolated)": "pytorch_times_gpu_1_CU.json",
        "Running on CPU": "pytorch_times_cpu.json",
        "RX 570 (fully shared with competitor)": "pytorch_times_gpu_full_shared.json",
        "RX 570 (partitioned to 16 CUs apart from competitor)": "pytorch_times_gpu_partitioned_16_CUs.json",
    }
    for k in datasets:
        datasets[k] = result_dir + datasets[k]
    plot_contents = [
        [
            ["Full GPU (32 CUs)", "RX 570 (32 CUs, isolated)"],
            ["16 CUs", "RX 570 (16 CUs, isolated)"],
            #["8 CUs", "RX 570 (8 CUs, isolated)"],
            ["4 CUs", "RX 570 (4 CUs, isolated)"],
            #["2 CUs", "RX 570 (2 CUs, isolated)"],
            ["1 CU", "RX 570 (1 CU, isolated)"],
        ],
        [
            ["No Competitor, Full GPU", "RX 570 (32 CUs, isolated)"],
            ["No Competitor, Limited to 16 CUs", "RX 570 (16 CUs, isolated)"],
            ["With Competitor, No Partitioning", "RX 570 (fully shared with competitor)"],
            ["With Competitor, Partitioned to 16 CUs", "RX 570 (partitioned to 16 CUs apart from competitor)"],
        ],
        [
            ["Titan V", "Titan V"],
            ["GTX 1060", "GTX 1060"],
            ["GTX 970", "GTX 970"],
            ["RX 570", "RX 570 (32 CUs, isolated)"],
            ["Running on CPU", "Running on CPU"],
        ],
    ]
    figures = []
    print(" Performance on Different Devices ".center(80, "="))
    print(" & Min & Max & Median & Mean & Std. Deviation" + r'\\')
    print(r'\hline')
    for v in plot_contents[2]:
        print_stats_latex(datasets[v[1]], v[0])
    print(" Performance with \"partitioning\" ".center(80, "="))
    print("\t\t\t Min \t Max \t Med. \t Mean \t Std.Dev.")
    for v in plot_contents[1]:
        print_stats_human_readable(datasets[v[1]], v[0])
    for p in plot_contents:
        figures.append(get_plot(datasets, p))
    plot.show()

