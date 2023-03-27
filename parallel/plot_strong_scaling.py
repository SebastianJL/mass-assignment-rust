import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it


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
            next_n_lines = list(it.islice(line_iter, len(columns)))
            if not next_n_lines:
                break
            row = []
            for column, line in zip(columns, next_n_lines):
                # print(line)
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
        fig, ax = plt.subplots(1)
        d = data[['n_threads', 'runtime']].groupby('n_threads')
        mins = d.min(numeric_only=True).reset_index()
        speedup = mins['runtime'][0] / mins['runtime']
        ax.plot(mins['n_threads'], speedup,'o', label="fastest run of 10")
        ax.plot(data['n_threads'], data['n_threads'], label='ideal')
        ax.set_xlabel('# threads')
        ax.set_ylabel('speedup')
        ax.title.set_text('strong scaling')
        fig.suptitle(args.title)
        ax.legend()
        if args.save:
            fig.savefig(f"{file}_plot.png", dpi=300)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
