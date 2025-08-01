"""
This script uses the calculated bin statistics to plot prob_change v.s. lat/lon figure
Author: Junjie Yang
Date: 2025-03-01
"""

import pandas as pd
import matplotlib.pyplot as plt


bin_deg = 1
flag = 'lat'
out_format = 'pdf' # 'jpg', 'svg', 'pdf'
font_size = 7
line_width_aux = 0.5
line_width_plot = 1
y_axis_pos = 'left'     # 'left', 'right'
alpha = 0.2
colors = ['tab:blue', 'tab:green', 'tab:red']
fill_colors = colors


# reset defaults
plt.rcdefaults()
# set default font
plt.rcParams['pdf.fonttype'] = 42  # 42 means that the output is a TrueType font rather than a path font
plt.rcParams['ps.fonttype'] = 42   # Same configuration to PostScript
plt.rcParams['font.family'] = 'Arial'

# load bin statistics
if flag == 'lat':
    if bin_deg == 1:
        bin_stats_585 = pd.read_parquet('/path/to/SSP585/lat/bin-statistics/bin_stats_585_lat_bin1.parquet')
        bin_stats_370 = pd.read_parquet('/path/to/SSP370/lat/bin-statistics/bin_stats_370_lat_bin1.parquet')
        bin_stats_126 = pd.read_parquet('/path/to/SSP126/lat/bin-statistics/bin_stats_126_lat_bin1.parquet')
    elif bin_deg == 0.1:
        raise NotImplementedError
elif flag == 'lon':
    if bin_deg == 1:
        bin_stats_585 = pd.read_parquet('/path/to/SSP585/lon/bin-statistics/bin_stats_585_lon_bin1.parquet')
        bin_stats_370 = pd.read_parquet('/path/to/SSP370/lon/bin-statistics/bin_stats_370_lon_bin1.parquet')
        bin_stats_126 = pd.read_parquet('/path/to/SSP126/lon/bin-statistics/bin_stats_126_lon_bin1.parquet')
    elif bin_deg == 0.1:
        raise NotImplementedError


if flag == 'lon':
    # configure the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 1.5)

    # control the size of the plot
    desired_height_in_inches = 0.62
    desired_width_in_inches = 5.35
    # get dpi
    dpi = fig.dpi
    # calculate needed with and height in pixels
    width_in_pixels = desired_width_in_inches * dpi
    height_in_pixels = desired_height_in_inches * dpi
    # set the position and size of current axis, where position_list is: (left, bottom, width, height)
    position_list = [0.1, 0.4, width_in_pixels / fig.get_figwidth() / dpi, height_in_pixels / fig.get_figheight() / dpi]
    ax.set_position(position_list)

    # plot the shading which represents +/- 1 standard deviation interval
    plt.fill_between(bin_stats_585['longitude_mid'],
                     bin_stats_585['mean'] - bin_stats_585['std'],
                     bin_stats_585['mean'] + bin_stats_585['std'],
                     color=fill_colors[2], alpha=alpha, edgecolor='none')
    plt.fill_between(bin_stats_370['longitude_mid'],
                     bin_stats_370['mean'] - bin_stats_370['std'],
                     bin_stats_370['mean'] + bin_stats_370['std'],
                     color=fill_colors[1], alpha=alpha, edgecolor='none')
    plt.fill_between(bin_stats_126['longitude_mid'],
                     bin_stats_126['mean'] - bin_stats_126['std'],
                     bin_stats_126['mean'] + bin_stats_126['std'],
                     color=fill_colors[0], alpha=alpha, edgecolor='none')

    # plot the line which represent mean value
    plt.plot(bin_stats_126['longitude_mid'], bin_stats_126['mean'], label='SSP126', color=colors[0], linewidth=line_width_plot)
    plt.plot(bin_stats_370['longitude_mid'], bin_stats_370['mean'], label='SSP370', color=colors[1], linewidth=line_width_plot)
    plt.plot(bin_stats_585['longitude_mid'], bin_stats_585['mean'], label='SSP585', color=colors[2], linewidth=line_width_plot)
    # plot the "y=0" line
    plt.axhline(y=0, color='black', linestyle='--', alpha=1, linewidth=line_width_aux)

    # set the labels
    plt.xlabel('Longitude', fontsize=font_size)
    plt.ylabel('Probability change', fontsize=font_size)

    plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(line_width_aux)
    # plt.gca().spines['left'].set_linewidth(line_width_aux)
    plt.gca().spines['right'].set_linewidth(line_width_aux)

    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')


    plt.xlim(-180, 180)
    plt.ylim(-0.3, 0.3)
    plt.gca().set_xticks(range(-180, 181, 30))
    xtick_labels = ['180°', '150°W', '120°W', '90°W', '60°W', '30°W', '0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°']
    plt.gca().set_xticklabels(xtick_labels, fontsize=font_size)

    plt.tick_params(direction='out', labelsize=font_size, width=line_width_aux)
    plt.legend(loc='upper left', ncol=3, frameon=False, fontsize=font_size)
elif flag == 'lat':
    # configure the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 1.5)

    # control the size of the plot
    desired_height_in_inches = 0.62
    desired_width_in_inches = 2.767
    # get dpi
    dpi = fig.dpi
    # calculate needed with and height in pixels
    width_in_pixels = desired_width_in_inches * dpi
    height_in_pixels = desired_height_in_inches * dpi
    # set the position and size of current axis, where position_list is: (left, bottom, width, height)
    position_list = [0.1, 0.4, width_in_pixels / fig.get_figwidth() / dpi, height_in_pixels / fig.get_figheight() / dpi]
    ax.set_position(position_list)

    # plot the shading which represents +/- 1 standard deviation interval
    plt.fill_between(bin_stats_585['latitude_mid'],
                     bin_stats_585['mean'] - bin_stats_585['std'],
                     bin_stats_585['mean'] + bin_stats_585['std'],
                     # label='SSP585',
                     color=fill_colors[2], alpha=alpha-0.05, edgecolor='none', zorder=-0)
    plt.fill_between(bin_stats_370['latitude_mid'],
                     bin_stats_370['mean'] - bin_stats_370['std'],
                     bin_stats_370['mean'] + bin_stats_370['std'],
                     # label='SSP370',
                     color=fill_colors[1], alpha=alpha, edgecolor='none', zorder=-2)
    plt.fill_between(bin_stats_126['latitude_mid'],
                     bin_stats_126['mean'] - bin_stats_126['std'],
                     bin_stats_126['mean'] + bin_stats_126['std'],
                     # label='SSP126',
                     color=fill_colors[0], alpha=alpha+0.2, edgecolor='none', zorder=-1)

    # plot the line which represent mean value
    plt.plot(bin_stats_126['latitude_mid'], bin_stats_126['mean'], label='SSP126', color=colors[0], linewidth=line_width_plot)
    plt.plot(bin_stats_370['latitude_mid'], bin_stats_370['mean'], label='SSP370', color=colors[1],
             linewidth=line_width_plot)
    plt.plot(bin_stats_585['latitude_mid'], bin_stats_585['mean'], label='SSP585', color=colors[2], linewidth=line_width_plot)
    # plot the "y=0" line
    plt.axhline(y=0, color='black', linestyle='--', alpha=1, linewidth=line_width_aux)

    # set the labels
    plt.xlabel('Latitude', rotation=180, fontsize=font_size)
    plt.ylabel('Probability change', fontsize=font_size)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_linewidth(line_width_aux)
    plt.gca().spines['bottom'].set_linewidth(line_width_aux)
    plt.xlim(81, -60)
    plt.ylim(-0.35, 0.25)
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')

    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.yticks([-0.2, 0.0, 0.2], ['-0.2', '0', '0.2'], va='center')

    plt.gca().set_xticks(range(-60, 91, 30))
    xtick_labels = ['60°S', '30°S', '0°', '30°N', '60°N', '90°N']
    plt.gca().set_xticklabels(xtick_labels, fontsize=font_size)
    plt.tick_params(direction='out', labelsize=font_size, width=line_width_aux)

path = '/output/directory/'
plt.savefig(path + 'prob_change_vs_' + flag + '_bin' + str(bin_deg) + '_rgb.' + out_format, dpi=400)