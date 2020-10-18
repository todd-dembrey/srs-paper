import math
import os

from adjustText import adjust_text
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


plt.style.use('classic')
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans'], 'size': 8})
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


all_cci_data = []
all_dci_data = []
all_radius_data = []


def print_stats(processed):
    volumes = processed.volumes_cm3

    def output(measure, value, stat=''):
        print(f'{measure} {stat}: {value}')

    def output_group(measure, data):
        for stat, calc in [
            ('max', max),
            ('min', min),
            ('mean', np.mean),
            ('std', np.std),
        ]:
            output(measure, calc(data), stat)

    for user, volume in zip(processed.data.users, volumes):
        output(user, volume)

    output_group('Volume', volumes)

    output('av50', processed.av_50/1000)
    output('ev', processed.ev/1000)

    output_group('CCI', processed.cci)
    output_group('DCI', processed.dci)


def plot_results(processed, plot_type, save, resub, compare):
    if save:
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(16, 12))

    plots[plot_type](processed, fig=ax)

    if save:
        if compare:
            processed = processed[0]
        file_parts = [processed.name, str(processed.data.RESOLUTION), plot_type]
        if processed.suffix:
            file_parts.append(processed.suffix)
        if compare:
            file_parts.append('compare')
        file_name = '-'.join(file_parts)
        save_path = os.path.join('images', f'{file_name}.png')
        fig.savefig(format='png', fname=save_path, dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

    fig.clf()


def plot_cci_dci(datas, fig):
    if not isinstance(datas, list):
        datas = [datas]

    # title = f'{datas[0].name}\nCCI against DCI'
    # fig.set_title(title)

    fig.set_xlim(0, 0.5)
    fig.set_xlabel('DCI')
    fig.set_ylim(0.5, 1)
    fig.set_ylabel('CCI')

    plt.setp(fig.spines.values(), linewidth=.75)
    fig.tick_params(direction='inout')

    for i, data in enumerate(datas):
        all_cci, all_dci = data.cci, data.dci
        outliers = [
            i for i, user in enumerate(data.data.users)
            if user.endswith('outlier')
        ]

        resub = [
            i for i, user in enumerate(data.data.users)
            if user.endswith('R')
        ]

        non_outliers = list(set(range(data.num_users)) - set(outliers) - set(resub))

        marker_style = {
            's': 18,
            'color': 'k',
            'linewidth': 0.5,
        }
        fig.scatter(all_dci[non_outliers], all_cci[non_outliers], label='Submissions', marker='o', **marker_style, facecolor='w')
        fig.scatter(all_dci[outliers], all_cci[outliers], label='ERG Outliers', marker='x', **marker_style)
        fig.scatter(all_dci[resub], all_cci[resub], label='Re-submissions', marker='+', **marker_style)

        if i == 0 :
            pass
            # texts = [
            #     fig.text(x, y, label)
            #     for label, x, y in zip(data.data.users, all_dci, all_cci)
            # ]
        else:
            previous = datas[i-1]
            previous_users = list(previous.data.users)
            for x, y, user in zip(all_dci, all_cci, data.data.users):
                try:
                    previous_index = previous_users.index(user)
                except ValueError:
                    if user.endswith('R'):
                        user = user[:-1] + ' outlier'
                        previous_index = previous_users.index(user)

                x2 = previous.dci[previous_index]
                y2 = previous.cci[previous_index]
                fig.annotate("", xy=(x, y), xytext=(x2, y2), arrowprops=dict(arrowstyle="->"))
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    fig.fill_between([0,1], [1, 0], 1, alpha=0.2, facecolor='grey')

    combined_limit = 0.2
    factor = 1/(1+combined_limit)
    x = [0, combined_limit, combined_limit,  0]
    y = [1, 1-combined_limit, factor - factor * combined_limit, factor]

    fig.plot(x, y, color='k', linewidth=0.5, ls='dashed', label=f'{combined_limit*100:.0f}% error margin')

    # fig.legend(fontsize='xx-small')


def plot_radii(processed, fig):
    radii = processed.radii
    std = np.std(radii)

    def plot_stats(method, data, label, style):
        value = method(data)
        upper = value + std
        lower = value - std
        plt.plot([value, value], [0, 1], linestyle=style, label=label)
        plt.plot([upper, upper], [0, 1], linestyle=style, label='upper')
        plt.plot([lower, lower], [0, 1], linestyle=style, label='lower')

    def calc_radius(volume):
        return (volume / np.pi) ** (1/3)

    n, bins, patches = plt.hist(processed.radii, processed.num_users, normed=1, cumulative=True)
    plot_stats(np.mean, radii, 'mean', '-')
    plot_stats(calc_radius, processed.av_50, 'av50', '--')
    fig.legend()


def plot_3d_shell(processed, fig):
    fig = plt.figure(figsize=plt.figaspect(1/3))
    for i, data in enumerate(
            [processed.ev_matrix, processed.av_50_matrix, processed.av_100_matrix]
    ):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.voxels(data, edgecolor='k')


def plot_density(processed, fig):
    *mesh, zs = processed.data.mesh_matrix
    x = mesh[0]
    y = mesh[1]
    total_data = processed.results.sum(3)

    diff = processed.data.RESOLUTION/2
    # Offset the grid so that the pixel is the center point
    extent = [x[0][0] - diff, x[0][-1] + diff, y[0][0] - diff, y[-1][0] + diff]

    number_of_plots = len(zs)
    required_rows = math.floor(number_of_plots ** 0.5)
    required_cols = math.ceil(number_of_plots / required_rows)

    title = f'{processed.name}\n' + '    '.join(
        [f'{av}: {getattr(processed, av)/1000:.2f}$cm^3$' for av in ['ev', 'av_50', 'av_100']]
    )
    fig.suptitle(title)

    grid = ImageGrid(
        fig,
        '111',
        nrows_ncols=(required_rows, required_cols),
        axes_pad=0.1,
        aspect=True,
        cbar_mode='single',
        cbar_location='top',
        cbar_size='2%',
    )

    # Scale up the AV50 so it is super simple to contour
    av_50 = processed.av_50_matrix
    shape = np.asarray(av_50.shape)
    scale_factor = [10, 10, 1]
    edge = np.empty(shape * scale_factor)
    for i in range(shape[2]):
        edge[:, :, i] = np.kron(av_50[:, :, i], np.ones(scale_factor[0:2]))

    def draw(i, height):
        ax = grid[i]
        # Make a new finer matrix for calculating the contour

        ax.imshow(np.fliplr(np.flipud(total_data[:, :, i])), extent=extent, vmin=0, vmax=processed.num_users, origin='top')
        ax.contour(np.fliplr(np.flipud(edge[:, :, i])), 0.5, extent=extent,
                   vmin=0, vmax=1, colors=['k'], linewidths=1.5, linestyles='--')

    for i, height in enumerate(zs):
        draw(i, height)

    grid.cbar_axes[0].colorbar(grid[0].images[0])
    grid.cbar_axes[0].set_xlabel('Number of outlines')

    for ax in grid[number_of_plots:]:
        fig.delaxes(ax)


plots = {
    'cci_dci': plot_cci_dci,
    'density': plot_density,
    'radii': plot_radii,
    '3d': plot_3d_shell,
}
