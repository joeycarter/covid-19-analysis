#!/usr/bin/env python3

"""
Predict COVID-19 Vaccine Coverage
=================================

Predict the COVID-19 vaccine coverage in a given country or jurisdiction.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

from uncertainties import ufloat

import ROOT as root

import atlasplots as aplt


def _docstring(docstring):
    """Return summary of docstring."""
    return " ".join(docstring.split("\n")[4:]) if docstring else ""


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=_docstring(__doc__))
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose messages; multiple -v options increase the verbosity.",
    )
    parser.add_argument(
        "location",
        help="Perform the analysis for this location. "
        "Input as ISO code, e.g. 'CAN' for Canada.",
    )
    parser.add_argument("-n", "--n-days", type=int, help="Fit the last N days only.")
    parser.add_argument(
        "-o", "--outdir", default="output", help="Directory where the plots are saved."
    )

    args = parser.parse_args()

    return args


def days_to_seconds(days):
    return days * 86400  # = 24 * 60 * 60


def read_data():
    return pd.read_csv("data/owid-covid-data.csv")


def fit_and_plot(df_main, args):
    """Fit data and plot.

    Parameters
    ----------
    df_main : pd.DataFrame
        Main data frame with the "Our World in Data" COVID-19 dataset.
    args : argparse namespace
        Global input arguments from argparse.
    """
    # Select data for the desired location
    location = args.location.upper()
    df = df_main[df_main["iso_code"].str.match(location)]

    # Drop missing vaccination data
    df = df[df["people_vaccinated_per_hundred"].notnull()]

    # Compute timestamps
    df["timestamp"] = pd.to_datetime(df["date"], format="%Y-%m-%d").astype(int) / 1e9

    # Fit
    # ---
    graph = aplt.root_helpers.graph(
        df["timestamp"], df["people_vaccinated_per_hundred"]
    )

    if args.n_days:
        fit_xmin = df["timestamp"].max() - days_to_seconds(args.n_days)
    else:
        fit_xmin = df["timestamp"].min()

    fit_xmax = df["timestamp"].max() + days_to_seconds(60)

    # Fit function
    func = root.TF1(
        "func", "[0] / (1 + TMath::Exp(-[1] * (x - [2])))", fit_xmin, fit_xmax
    )
    func.SetParameter(0, 70)
    func.SetParameter(1, 1e-8)
    func.SetParameter(2, df["timestamp"].min())

    # Fit the histogram with the original distribution; store graphics func but do not draw
    fit_result = graph.Fit("func", "0RS")

    # Plot
    # ----
    aplt.set_atlas_style()

    fig, ax = aplt.subplots()

    # Display x-axis as date; do this before plotting
    ax.frame.GetXaxis().SetTimeDisplay(2)
    ax.frame.GetXaxis().SetTimeFormat("%b %d, %Y")
    ax.frame.GetXaxis().SetTimeOffset(0)
    ax.frame.GetXaxis().SetNdivisions(-403)

    # Plot data
    ax.plot(
        graph,
        "P",
        markerstyle=20,
        markercolor=root.kAzure - 2,
        markersize=0.7,
        label=df["location"].iloc[0],
        labelfmt="P",
    )

    # Draw the fit function
    func.SetNpx(1000)
    ax.plot(func, linecolor=root.kRed + 1, label="Fit (logistic)", labelfmt="L")

    ax.set_xlim(right=fit_xmax)

    ax.add_margins(bottom=0.05, top=0.15, left=0.05)

    # Draw line at fitted p0
    line = root.TLine(
        ax.get_xlim()[0],
        fit_result.Parameter(0),
        ax.get_xlim()[1],
        fit_result.Parameter(0),
    )
    ax.plot(line, linestyle=2)

    ax.text(0.2, 0.88, "#it{Vaccine Coverage:} Data & Predictions")

    last_updated = datetime.datetime.strptime(df["date"].iloc[-1], "%Y-%m-%d").strftime(
        "%b %d, %Y"
    )
    ax.text(0.2, 0.80, f"Last Updated: {last_updated}", size=22)
    ax.text(0.2, 0.75, f"Fit last {args.n_days} days", size=22)

    # Predicting maximum vaccination coverage (from fit)
    pred_max = ufloat(fit_result.Parameter(0), fit_result.ParError(0))
    ax.text(
        0.2, 0.70, f"Predicted max: {pred_max:.2uL} %".replace("\pm", "#pm"), size=22
    )

    ax.text(
        0.91,
        0.2,
        "#it{Source: ourworldindata.org/covid-vaccinations}",
        size=14,
        align=31,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Share of people with #geq 1 dose [%]")

    ax.legend(loc=(0.20, 0.55, 0.35, 0.65), textsize=22)

    # Save the plot
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    fig.savefig(os.path.join(args.outdir, f"predict_vaccine_coverage.{location}.pdf"))


def main():
    try:
        args = parse_args()
        df_main = read_data()
        fit_and_plot(df_main, args)

    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    # Stops ROOT from hijacking --help
    root.PyConfig.IgnoreCommandLineOptions = True

    # Batch mode
    root.gROOT.SetBatch(True)

    sys.exit(main())
