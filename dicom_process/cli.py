import click

from .read import process_one_file
from .output import plots, print_stats, plot_results


@click.command()
@click.argument('plot_type', nargs=1)
@click.argument('--filename', type=click.Path(exists=True))
@click.option('--stats', default=False, is_flag=True, help='display stats')
@click.option('--save', default=False, is_flag=True, help='save image')
@click.option('--clear', default=False, is_flag=True, help='clear the cache')
@click.option('--resub', default=False, is_flag=True, help='analyse resubmissions')
@click.option('--compare', default=False, is_flag=True, help='compare both submissions')
@click.option('--good', default=False, is_flag=True, help='compare against the good av50')
def run(plot_type, filename, stats, save, clear, resub, compare, good):
    plot_type = [k for k in plots.keys() if k.startswith(plot_type)][0]
    processed_data = process_one_file(
        filename,
        clear,
        resub,
        compare,
        good,
    )
    if stats:
        print_stats(processed_data)
    plot_results(processed_data, plot_type, save, resub, compare)


if __name__ == '__main__':
    run()
