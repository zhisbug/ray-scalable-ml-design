import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime


def parse_csv(path):
    """Read in the way that dict[algorithm][world_size] --> list"""
    results = dict()
    with open(path) as csvfile:
        iterator = csv.reader(csvfile, delimiter=',')
        for row in iterator:
            print(row)
            algorithm, world_size, object_size, mean, std = row[0].split('_')[-1], int(row[1]), int(row[2]), float(row[3]), float(row[4])
            if algorithm not in results:
                results[algorithm] = {}
            if world_size not in results[algorithm]:
                results[algorithm][world_size] = []
            results[algorithm][world_size].append([object_size, mean, std])
    return results

def read_data(setting, backend):
    if setting == 'multigpu':
        assert backend == 'gpu'
        ray_path = 'multigpu/ray-microbenchmark-gpu.csv'
        pytorch_path = 'multigpu/pytorch-microbenchmark-nccl.csv'
    else:
        if backend == 'cpu':
            ray_path = 'distributed/ray-microbenchmark-cpu.csv'
            pytorch_path = 'distributed/pytorch-microbenchmark-gloo.csv'
        else:
            ray_path = 'distributed/ray-microbenchmark-gpu.csv'
            pytorch_path = 'distributed/pytorch-microbenchmark-nccl.csv'
    ray_data = parse_csv(ray_path)
    pytorch_data = parse_csv(pytorch_path)
    return ray_data, pytorch_data




def draw(ray, other, fig_name):
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    width = 0.1       # the width of the bars
    margin = width * 3 + 0.1
    start = 0.1
    ind  = np.array([start + m * margin for m in range(ray.shape[0])])

    labelfont = {
            'color':  'black',
            'weight': 'normal',
            'size': 12}

    colors = ['darkorange', 'c', 'lightgreen', 'lightskyblue']
    hatches = ['/', '\\', '+', 'x', '+',]
    rects = []

    data = np.stack((ray[:,1], other[:,1]))
    for i in range(2):
        m = ax.bar(ind + width * i, np.log10(data[i] * 1e6), width, color=colors[i], edgecolor='black', hatch=hatches[i])
        rects.append(m)

    # ax.set_xlim(xmin=0.0, xmax=2)
    xs = ind + width / 2
    ax.set_xticks(xs)
    ax.set_xticklabels(('1KB', '0.5MB', '1MB', '0.5GB', '1GB'), fontdict = labelfont)
    ax.set_ylabel('Latency (micro sec)', fontdict = labelfont)
    ax.set_ylim(ymax=7, ymin=1)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$'])
    if 'cpu' in fig_name:
        other_name = 'gloo'
    else:
        other_name = 'nccl'
    ax.legend((rects[0][0], rects[1][0]), ('ray', other_name), loc='upper left', ncol=1, prop={'size':12})
    # set the grid lines to dotted
    ax.grid(True)
    gridlines = ax.get_ygridlines() + ax.get_xgridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    # # add the imporvement
    # improvement1 = round(min(data[0][0], data[1][0]) / data[2][0], 2)
    # ax.text(0.26, data[2][0] + 0.005, str(improvement1) + 'x', size=10, weight='bold')
    #
    # # improvement1 = round(max(data[0][0], data[1][0]) / data[3][0], 2)
    # ax.text(0.36, data[3][0] + 0.005, '1.2x', size=10, weight='bold')
    #
    # improvement2 = round(min(data[0][1], data[1][1]) / data[2][1], 2)
    # ax.text(0.76, data[2][1] + 0.005, str(improvement2) + 'x', size=10, weight='bold')
    #
    # improvement2 = round(min(data[0][1], data[1][1]) / data[3][1], 2)
    # ax.text(0.86, data[3][1] + 0.005, str(improvement2) + 'x', size=10, weight='bold')
    #
    # # one bar is out ot range
    # ax.text(0.65, 0.38, str(round(data[1][1], 2)), size=10)


    ax.set_title(fig_name, fontsize=12, weight='bold')
    plt.show()
    #ax.text(rects12[0].get_x()+rects12[0].get_width()/2, rects12[0].get_y()+rects12[0].get_height() + 0.2, 'TF', ha='center', va='bottom', fontsize=9)
    #ax.text(rects12[1].get_x()+rects12[1].get_width()/2, rects12[1].get_y()+rects12[1].get_height() + 0.2, 'TF-P-wf', ha='center', va='bottom', fontsize=9)
    #ax.text(rects12[2].get_x()+rects12[2].get_width()/2, rects12[2].get_y()+rects12[2].get_height() + 0.2, 'TF-P', ha='center', va='bottom', fontsize=9)

    #ax.text(rects22[0].get_x()+rects22[0].get_width()/2, rects22[0].get_y()+rects22[0].get_height() + 0.2, 'TF', ha='center', va='bottom', fontsize=9)
    #ax.text(rects22[1].get_x()+rects22[1].get_width()/2, rects22[1].get_y()+rects22[1].get_height() + 0.2, 'TF-P-wf', ha='center', va='bottom', fontsize=9)
    #ax.text(rects22[2].get_x()+rects22[2].get_width()/2, rects22[2].get_y()+rects22[2].get_height() + 0.2, 'TF-P', ha='center', va='bottom', fontsize=9)

    #ax.text(rects32[0].get_x()+rects32[0].get_width()/2, rects32[0].get_y()+rects32[0].get_height() + 0.2, 'TF', ha='center', va='bottom', fontsize=9)
    #ax.text(rects32[1].get_x()+rects32[1].get_width()/2, rects32[1].get_y()+rects32[1].get_height() + 0.2, 'TF-P-wf', ha='center', va='bottom', fontsize=9)
    #ax.text(rects32[2].get_x()+rects32[2].get_width()/2, rects32[2].get_y()+rects32[2].get_height() + 0.2, 'TF-P', ha='center', va='bottom', fontsize=9)

    #def autolabel(rectss):
        # attach some text labels
    #    n = len(rectss)
    #    for j in range(len(rectss[0])):
    #      height = 0
    #      for i in range(n):
    #        height = height + rectss[i][j].get_height()
    #      ax.text(rectss[0][j].get_x()+rectss[0][j].get_width()/2, height + 0.6, '%0.0f'% height,
    #            ha='center', va='bottom', fontsize=8)

    # autolabel([rects1, rects12, rects3])
    # autolabel([rects12, rects122, rects32])

    plt.tight_layout()
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    #plt.show()
    save_dir = os.path.join('plots/', fig_name)
    # fig.savefig(save_dir + '.pdf', transparent = True, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig(save_dir + '.png', transparent = True, bbox_inches = 'tight', pad_inches = 0)


backends = ['cpu', 'gpu']
settings = ['multigpu', 'distributed']
for setting in settings:
    for backend in backends:
        if setting == 'multigpu' and backend == 'cpu':
            continue

        ray_data, pytorch_data = read_data(setting, backend)

        if backend == 'gpu':
            algorithms = ['reduce', 'broadcast', 'allgather', 'allreduce']
        else:
            algorithms = ['reduce', 'gather', 'broadcast', 'allgather', 'allreduce', 'sendrecv']
        if setting == 'multigpu':
            world_sizes = [2]
        else:
            world_sizes = [2, 4, 8, 16]

        for algorithm in algorithms:
            for world_size in world_sizes:
                fig_name = '{}-{}-{}-{}'.format(setting, backend, algorithm, world_size)
                draw(np.array(ray_data[algorithm][world_size]), np.array(pytorch_data[algorithm][world_size]), fig_name)

