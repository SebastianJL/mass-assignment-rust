import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from plot_strong_scaling import init_argparse, parse_data


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
        d = data[["n_threads", "runtime"]].groupby("n_threads")
        means = d.mean(numeric_only=True).reset_index()
        mins = d.min(numeric_only=True).reset_index()
        efficiency = means["runtime"][0] / means["runtime"]
        efficiency = mins["runtime"][0] / mins["runtime"]

        # ax.plot(means["n_threads"], efficiency, "o", label="mean")
        ax.plot(mins["n_threads"], efficiency, "o", label="fastest run of 10")
        ax.plot(means["n_threads"], np.ones(len(means)), label="ideal")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("# threads")
        ax.set_ylabel("efficiency")
        ax.title.set_text("weak scaling")
        fig.suptitle(args.title)
        ax.legend()

        if args.save:
            fig.savefig(f"{file}_plot.png", dpi=300)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
