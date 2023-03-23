import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # usage="%(prog)s [OPTION] [FILE]...",
        description="Plot speedup of run files."
    )
    parser.add_argument(
        "-n", "--no-show", action="store_false",
        help="Don't show the figures.",
        dest="show", default=True,
    )
    parser.add_argument(
        "-s", "--save", action="store_true",
        help="Save figures", default=False,
    )
    parser.add_argument(
        "-t", "--title", action="store",
        help="Set title of figure", default="",
    )
    parser.add_argument('files', nargs='+')
    return parser


def parse_data(file):
    columns = ['seed', 'n_particles', 'n_grid', 'n_threads', 'total_mass', 'runtime']
    with open(file) as in_file:
        line_iter = iter(in_file)
        data = []
        while True:
            try:
                # Skip first two lines of each block
                next(line_iter)
                next(line_iter)
            except StopIteration:
                break
            row = []
            for column, line in zip(columns, line_iter):
                print(line)
                row.append(line.split(f"{column} = ")[1].strip())
            data.append(row)
    df = pd.DataFrame(data, columns=columns)

    # Parse dtypes.
    for column in columns:
        if column in ['runtime']:
            df[column] = pd.to_timedelta(df[column]).values.astype(np.int64)
        elif column in ['seed', 'n_particles', 'n_grid', 'n_threads', 'total_mass']:
            df[column] = pd.to_numeric(df[column])
        else:
            raise "Encountered unknown column"

    return df


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    for file in args.files:
        data = parse_data(file)
        # print(data)
        # print(data.dtypes)

        fig, ax = plt.subplots(1)
        print(data.dtypes)
        print(data.head())
        # exit()
        d = data[['n_threads', 'runtime']].groupby('n_threads')
        means = d.mean(numeric_only=True).reset_index()
        # mins = d.min(numeric_only=True).reset_index()
        # # stds = d.std(numeric_only=True).reset_index()
        # # stderr = stds['runtime'].values / np.sqrt(d.count()['runtime'].values)
        # # stderr = stderr['runtime'].values
        # # print(stds)
        # # print(d.count())
        # # print(stderr)
        # # exit()
        speedup = means['runtime'][0] / means['runtime']
        # speedup_mins = means['runtime'][0] / mins['runtime']
        # relerr = stderr / means['runtime'].values + stderr[0] / means['runtime'].values[0]
        # speedup_err = relerr * speedup
        ax.plot(data['n_threads'], data['runtime'][0]/data['runtime'], 'o', label="all")
        ax.plot(means['n_threads'], speedup, 'o', label="mean")
        # ax.plot(mins['n_threads'], speedup_mins,'o', label="fastest")
        ax.plot(data['n_threads'], data['n_threads'], label='ideal')
        ax.set_xlabel('# threads')
        ax.set_ylabel('speedup')
        ax.title.set_text('strong scaling')
        fig.suptitle(args.title)
        ax.legend()
        if args.save:
            fig.savefig(f"{file}_plot.svg")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
