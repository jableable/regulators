from __future__ import annotations
from typing import Optional, Union, List, Tuple
from concurrent import futures
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataprep import Dataprep, IsinArg_t
from .base import BaseSynth


class HoldoutSplitter:
    """Iterator that prepares the time series for cross‐validation by
    progressively removing blocks of length `holdout_len`.
    """

    def __init__(self, df: pd.DataFrame, ser: pd.Series, holdout_len: int = 1):
        if df.shape[0] != ser.shape[0]:
            raise ValueError("`df` and `ser` must have the same number of rows.")
        if holdout_len < 1:
            raise ValueError("`holdout_len` must be at least 1.")
        if holdout_len >= df.shape[0]:
            raise ValueError("`holdout_len` must be less than df.shape[0]")
        self.df = df
        self.ser = ser
        self.holdout_len = holdout_len
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (self.idx + self.holdout_len) > self.df.shape[0]:
            raise StopIteration
        holdout = slice(self.idx, self.idx + self.holdout_len)
        df_holdout = self.df.iloc[holdout]
        ser_holdout = self.ser.iloc[holdout]
        df = self.df.drop(index=self.df.index[holdout])
        ser = self.ser.drop(index=self.ser.index[holdout])
        self.idx += 1
        return df, df_holdout, ser, ser_holdout


@dataclass
class CrossValidationResult:
    lambdas: np.ndarray
    errors_mean: np.ndarray
    errors_se: np.ndarray

    def best_lambda(self, min_1se: bool = True) -> float:
        if min_1se:
            return (
                self.lambdas[
                    self.errors_mean
                    <= self.errors_mean.min()
                    + self.errors_se[self.errors_mean.argmin()]
                ]
                .max()
                .item()
            )
        return self.lambdas[self.errors_mean.argmin()].item()

    def plot(self) -> None:
        plt.errorbar(
            x=self.lambdas,
            y=self.errors_mean,
            yerr=self.errors_se,
            ecolor="black",
            capsize=2,
        )
        plt.xlabel("Lambda")
        plt.ylabel("Mean error")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Cross validation result")
        plt.grid()
        plt.show()


class PlaceboTest:
    """Class that carries out placebo tests by running a synthetic control
    study using each possible control unit as the treated unit and the
    remaining control units as controls.
    """

    def __init__(self) -> None:
        self.paths: Optional[pd.DataFrame] = None
        self.treated_path: Optional[pd.Series] = None
        self.gaps: Optional[pd.DataFrame] = None
        self.treated_gap: Optional[pd.Series] = None
        self.time_optimize_ssr: Optional[IsinArg_t] = None

    def fit(
        self,
        dataprep: Dataprep,
        scm: BaseSynth,
        scm_options: dict = {},
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        paths, gaps = list(), list()
        n_tests = len(dataprep.controls_identifier)
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = []
            for treated, controls in self.placebo_iter(dataprep.controls_identifier):
                # Make a shallow copy to reassign treatment/control labels.
                _dataprep = copy.copy(dataprep)
                _dataprep.treatment_identifier = treated
                _dataprep.controls_identifier = controls
                tasks.append(
                    executor.submit(
                        self._single_placebo,
                        dataprep=_dataprep,
                        scm=scm,
                        scm_options=scm_options,
                    )
                )
            for idx, future in enumerate(futures.as_completed(tasks), start=1):
                path, gap = future.result()
                if verbose:
                    print(f"({idx}/{n_tests}) Completed placebo test for {path.name}.")
                paths.append(path)
                gaps.append(gap)

        self.paths = pd.concat(paths, axis=1)
        self.gaps = pd.concat(gaps, axis=1)
        self.time_optimize_ssr = dataprep.time_optimize_ssr

        print("Calculating treated unit gaps.")
        self.treated_path, self.treated_gap = self._single_placebo(
            dataprep=dataprep, scm=scm, scm_options=scm_options
        )
        print("Done.")

    @staticmethod
    def placebo_iter(controls: List[str]) -> Tuple[str, List[str]]:
        for control in controls:
            yield (control, [c for c in controls if c != control])

    @staticmethod
    def _single_placebo(
        dataprep: Dataprep, scm: BaseSynth, scm_options: dict = {}
    ) -> Tuple[pd.Series, pd.Series]:
        scm_options.setdefault("optim_initial", "equal")
        scm.fit(dataprep=dataprep, **scm_options)
        # Build outcome matrices
        Z0, Z1 = dataprep.make_outcome_mats(
            time_period=dataprep.foo[dataprep.time_variable]
        )
        synthetic = scm._synthetic(Z0=Z0)
        # The gaps are computed as (treated - synthetic)
        gaps = scm._gaps(Z0=Z0, Z1=Z1)
        return (
            synthetic.rename(dataprep.treatment_identifier),
            gaps.rename(dataprep.treatment_identifier),
        )

    def gaps_plot(
        self,
        time_period: Optional[Union[pd.Index, slice, List]] = None,
        grid: bool = True,
        treatment_time: Optional[Union[int, str, pd.Timestamp]] = None,
        mspe_threshold: Optional[float] = None,
        exclude_units: Optional[List[str]] = None,
    ) -> None:
        """
        Plots the gap (difference between treated and synthetic) over time.
        Both placebo and treated gap are subset using the same time_period.

        Parameters
        ----------
        time_period : Index, slice, or list, optional
            The time periods to include. If None, uses self.time_optimize_ssr.
        grid : bool, optional
            Whether to show grid lines.
        treatment_time : int, str, or pd.Timestamp, optional
            The time at which treatment occurs (used for a vertical line and filtering).
        mspe_threshold : float, optional
            If set, excludes placebo units with poor pre-treatment fit.
        exclude_units : list of str, optional
            A list of control unit names to exclude.
        """
        if self.gaps is None or self.treated_gap is None:
            raise ValueError("No gaps available; run a placebo test first.")

        # Use the provided time_period or default to the time subset used in optimization.
        if time_period is None:
            time_period = self.time_optimize_ssr

        # Subset both placebo and treated gaps by time.
        placebo_gaps = self.gaps.loc[time_period]
        treated_gap = self.treated_gap.loc[time_period]

        # Exclude specified units from the placebo gap.
        if exclude_units is not None:
            placebo_gaps = placebo_gaps.drop(columns=exclude_units, errors="ignore")

        # Apply MSPE threshold filtering to placebo gaps only.
        if mspe_threshold is not None:
            if treatment_time is None:
                raise ValueError("Need `treatment_time` to use `mspe_threshold`.")
            pre_mspe = self.gaps.loc[:treatment_time].pow(2).sum(axis=0)
            pre_mspe_treated = self.treated_gap.loc[:treatment_time].pow(2).sum()
            valid_units = pre_mspe[pre_mspe < mspe_threshold * pre_mspe_treated].index
            placebo_gaps = placebo_gaps[valid_units]

        plt.figure(figsize=(10, 6))
        # Plot all placebo gaps.
        for col in placebo_gaps.columns:
            plt.plot(placebo_gaps.index, placebo_gaps[col], color="black", alpha=0.1)
        # Plot the treated gap as a thick black line.
        plt.plot(treated_gap.index, treated_gap, color="black", alpha=1.0, linewidth=2)
        if treatment_time is not None:
            plt.axvline(x=treatment_time, linestyle="dashed")
        if grid:
            plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Gap (Treated - Synthetic)")
        plt.title("Gap Plot: Treated vs. Placebos")
        plt.show()

    def pvalue(self, treatment_time: Union[int, str, pd.Timestamp]) -> float:
        """
        Calculate the p-value following Abadie et al.'s version of Fisher’s exact test.
        """
        if self.gaps is None or self.treated_gap is None:
            raise ValueError("Run a placebo test first.")

        all_gaps = pd.concat([self.gaps, self.treated_gap], axis=1)
        denom = all_gaps.loc[:treatment_time].pow(2).sum(axis=0)
        num = all_gaps.loc[treatment_time:].pow(2).sum(axis=0)
        t_total = len(all_gaps)
        t0 = len(all_gaps.loc[:treatment_time])
        rmspe = (num / (t_total - t0)) / (denom / t0)
        return (rmspe.drop(index=self.treated_gap.name) >= rmspe.loc[self.treated_gap.name]).sum() / len(rmspe)
