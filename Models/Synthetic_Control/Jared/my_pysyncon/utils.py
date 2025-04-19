from __future__ import annotations
from typing import Optional, Union
from concurrent import futures
import copy
from dataclasses import dataclass
import scipy.stats.mstats as mstats
from scipy.stats import boxcox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataprep import Dataprep, IsinArg_t
from .base import BaseSynth


class HoldoutSplitter:
    """Iterator that prepares the time series for cross-validation by
    progressively removing blocks of length `holdout_len`.
    """

    def __init__(self, df: pd.DataFrame, ser: pd.Series, holdout_len: int = 1):
        """Iterator that prepares the time series for cross-validation by
        progressively removing blocks of length `holdout_len`.

        Parameters
        ----------
        df : pandas.DataFrame, shape (r, c)
            Dataframe that will be split for the cross-validation.
        ser : pandas.Series, shape (r, 1)
            Series that will split for the cross-validation.
        holdout_len : int, optional
            Number of days to remove in each iteration, by default 1.

        Raises
        ------
        ValueError
            if df and ser do not have the same number of rows.
        ValueError
            if `holdout_len` is not >= 1.
        ValueError
            if `holdout_len` is larger than the number of rows of df.
        """
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

    def __next__(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (self.idx + self.holdout_len) > self.df.shape[0]:
            raise StopIteration
        holdout = slice(self.idx, self.idx + self.holdout_len)

        df_holdout = self.df.iloc[holdout,]  # fmt: skip
        ser_holdout = self.ser.iloc[holdout]

        df = self.df.drop(index=self.df.index[holdout])
        ser = self.ser.drop(index=self.ser.index[holdout])

        self.idx += 1
        return df, df_holdout, ser, ser_holdout


@dataclass
class CrossValidationResult:
    """Convenience class for holding the results of the cross-validation
    procedure from the AugSynth.
    """

    lambdas: np.ndarray
    errors_mean: np.ndarray
    errors_se: np.ndarray

    def best_lambda(self, min_1se: bool = True) -> float:
        """Return the best lambda.

        Parameters
        ----------
        min_1se : bool, optional
            return the largest lambda within 1 standard error of the minimum
            , by default True

        Returns
        -------
        float
        """
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
        """Plots the mean errors against the lambda values with the standard
        errors as error bars.
        """
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
    remaining control units as controls. See :cite:`germany2015` for more details.
    """

    def __init__(self) -> None:
        self.paths: Optional[pd.DataFrame] = None
        self.treated_path: Optional[pd.DataFrame] = None
        self.gaps: Optional[pd.DataFrame] = None
        self.treated_gap: Optional[pd.DataFrame] = None
        self.time_optimize_ssr: Optional[IsinArg_t] = None

    def fit(
        self,
        dataprep: Dataprep,
        scm: BaseSynth,
        scm_options: dict = {},
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ):
        """Run the placebo tests. This method is multi-process and by default
        will use all available processors. Use the `max_workers` option to change
        this behaviour.

        Parameters
        ----------
        dataprep : Dataprep
            :class:`Dataprep` object containing data to model, by default None.
        scm : Synth | AugSynth
            Synthetic control study to use
        scm_options : dict, optional
            Options to provide to the fit method of the synthetic control
            study, valid options are any valid option that the `scm_type`
            takes, by default {}
        max_workers : Optional[int], optional
            Maximum number of processes to use, if not provided then will use
            all available, by default None
        verbose : bool, optional
            Whether or not to output progress, by default True
        """
        paths, gaps = list(), list()
        n_tests = len(dataprep.controls_identifier)
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            to_do = list()
            for treated, controls in self.placebo_iter(dataprep.controls_identifier):
                _dataprep = copy.copy(dataprep)
                _dataprep.treatment_identifier = treated
                _dataprep.controls_identifier = controls
                to_do.append(
                    executor.submit(
                        self._single_placebo,
                        dataprep=_dataprep,
                        scm=scm,
                        scm_options=scm_options,
                    )
                )
            for idx, future in enumerate(futures.as_completed(to_do), 1):
                path, gap = future.result()
                if verbose:
                    print(f"({idx}/{n_tests}) Completed placebo test for {path.name}.")
                paths.append(path)
                gaps.append(gap)

        self.paths = pd.concat(paths, axis=1)
        self.gaps = pd.concat(gaps, axis=1)
        self.time_optimize_ssr = dataprep.time_optimize_ssr

        print(f"Calculating treated unit gaps.")
        self.treated_path, self.treated_gap = self._single_placebo(
            dataprep=dataprep, scm=scm, scm_options=scm_options
        )
        print("Done.")

    @staticmethod
    def placebo_iter(controls: list[str]) -> tuple[str, list[str]]:
        """Generates combinations of (treated unit, control units) for the
        placebo tests.

        Parameters
        ----------
        controls : list[str]
            List of unit labels to use

        Yields
        ------
        tuple[str, list[str]]
            Tuple of (treated unit label, control unit labels)

        :meta private:
        """
        for control in controls:
            yield (control, [c for c in controls if c != control])

    @staticmethod
    def _single_placebo(
        dataprep: Dataprep, scm: BaseSynth, scm_options: dict = {}
    ) -> tuple[pd.Series, pd.Series]:
        """Run a single placebo test.

        Parameters
        ----------
        dataprep : Dataprep
            :class:`Dataprep` object containing data to model
        scm : Synth | AugSynth
            Type of synthetic control study to use
        scm_options : dict, optional
            Options to provide to the fit method of the synthetic control
            study, valid options are any valid option that `scm` takes, by
            default {}

        Returns
        -------
        tuple[pandas.Series, pandas.Series]
            A time-series of the path of the synthetic control and a
            time-series of the gap between the treated unit and the synthetic
            control.

        :meta private:
        """
        # Force initialization of weights at zero if not already specified
        scm_options.setdefault("optim_initial", "equal")

        scm.fit(dataprep=dataprep, **scm_options)

        Z0, Z1 = dataprep.make_outcome_mats(
            time_period=dataprep.foo[dataprep.time_variable]
        )
        synthetic = scm._synthetic(Z0=Z0)
        gaps = scm._gaps(Z0=Z0, Z1=Z1)
        return synthetic.rename(dataprep.treatment_identifier), gaps.rename(
            dataprep.treatment_identifier
        )
    
    
    def gaps_plot(
        self,
        time_period: Optional[list] = None,
        grid: bool = True,
        treatment_time: Optional[Union[float, str, pd.Timestamp]] = None,
        mspe_threshold: Optional[float] = None,
        exclude_units: Optional[list] = None,
    ) -> None:
        """
        Plots the gap (treated minus synthetic) over time for both placebo units
        and the treated unit.
        
        Parameters
        ----------
        time_period : list, optional
            The list (or index) of time points to include in the plot. If None, uses
            self.time_optimize_ssr.
        grid : bool, optional
            Whether to show grid lines.
        treatment_time : float, str, or pd.Timestamp, optional
            The time at which treatment occurs; used both for drawing a vertical line
            and to determine the pre-treatment period.
        mspe_threshold : float, optional
            If provided, excludes donor units with poor pre-treatment fit.
        exclude_units : list, optional
            A list of donor unit names to exclude.
        """
        import matplotlib.pyplot as plt

        if self.gaps is None or self.treated_gap is None:
            raise ValueError("No gaps available; run a placebo test first.")

        # Use provided time_period or fallback to default.
        if time_period is None:
            time_period = self.time_optimize_ssr

        # Drop any excluded donor units.
        gaps = self.gaps.copy()
        if exclude_units is not None:
            gaps = gaps.drop(columns=exclude_units, errors="ignore")
        
        # Optionally filter placebo gaps by MSPE threshold (applied pre-treatment).
        if mspe_threshold is not None:
            if treatment_time is None:
                raise ValueError("Need `treatment_time` to use `mspe_threshold`.")
            pre_mspe = gaps.loc[:treatment_time].pow(2).sum(axis=0)
            pre_mspe_treated = self.treated_gap.loc[:treatment_time].pow(2).sum()
            valid_units = pre_mspe[pre_mspe < mspe_threshold * pre_mspe_treated].index
            gaps = gaps[valid_units]

        # Subset the time period for both donor gaps and the treated gap.
        placebo_gaps = gaps.loc[gaps.index.isin(time_period)].copy()
        treated_gap = self.treated_gap.loc[self.treated_gap.index.isin(time_period)].copy()

        # Convert the index to datetime for consistency (if not already).
        placebo_gaps.index = pd.to_datetime(placebo_gaps.index, errors="coerce")
        treated_gap.index = pd.to_datetime(treated_gap.index, errors="coerce")

            # Convert the provided treatment_time to datetime.
        try:
            treatment_time_dt = pd.to_datetime(treatment_time)
        except Exception as e:
            raise ValueError(f"Could not convert treatment_time to datetime: {e}")
        # Plot the placebo donor gaps (light lines) and the treated gap (thick line).
        plt.figure(figsize=(10, 6))
        for col in placebo_gaps.columns:
            plt.plot(placebo_gaps.index, placebo_gaps[col], color="black", alpha=0.1)
        plt.plot(treated_gap.index, treated_gap, color="black", alpha=1.0, linewidth=2)

        # Draw vertical line to indicate treatment time.
        if treatment_time is not None:
            plt.axvline(x=treatment_time_dt, linestyle="dashed")
        if grid:
            plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Gap (Treated - Synthetic)")
        plt.title("Gap Plot: Treated vs. Placebos")
        plt.show()


    def pvalue(self, treatment_time: int) -> float:
        """Calculate p-value of Abadie et al's version of Fisher's
        exact hypothesis test for no effect of treatment null, see also
        section 2.2. of :cite:`fp2018`.

        Parameters
        ----------
        treatment_time : int
            The time period that the treatment time occurred

        Returns
        -------
        float
            p-value for null hypothesis of no effect of treatment

        Raises
        ------
        ValueError
            if no placebo test has been run yet
        """
        if self.gaps is None or self.treated_gap is None:
            raise ValueError("Run a placebo test first.")

        all_ = pd.concat([self.gaps, self.treated_gap], axis=1)

        denom = all_.loc[:treatment_time].pow(2).sum(axis=0)
        num = all_.loc[treatment_time:].pow(2).sum(axis=0)

        t, _ = self.gaps.shape
        t0, _ = self.gaps.loc[:treatment_time].shape

        rmspe = (num / (t - t0)) / (denom / t0)
        return sum(
            rmspe.drop(index=self.treated_gap.name) >= rmspe.loc[self.treated_gap.name]
        ) / len(rmspe)


def date_to_str(date):
    return date.dt.strftime('%Y-%m-%d')


def winsorize_series_preserve_nans(s, limits):
    non_nan_mask = ~s.isna()
    s_winsor = s.copy()
    s_winsor[non_nan_mask] = mstats.winsorize(s[non_nan_mask], limits=limits)
    return s_winsor


def winsorize_pre_int(s, pre_mask, limits=(0.01, 0.01)):
    s = s.copy()
    non_nan_mask = s.notna()
    
    # Get pre-treatment non-NaN values
    pre_mask_non_nan = pre_mask & non_nan_mask
    pre_values = s[pre_mask_non_nan]
    
    if pre_values.empty:
        return s  # nothing to winsorize
    
    # Winsorize the pre-treatment values using scipy
    winsorized_pre = mstats.winsorize(pre_values, limits=limits)
    
    # Extract thresholds used (min and max values after winsorization)
    lower_bound = winsorized_pre.min()
    upper_bound = winsorized_pre.max()
    
    # Apply clipping to all non-NaN values using those thresholds
    s.loc[non_nan_mask] = s.loc[non_nan_mask].clip(lower=lower_bound, upper=upper_bound)
    
    return s


def boxcox_pre_int_v0(s, pre_mask, offset_eps=1e-6):
    """
    Applies a Box-Cox transformation to a pandas Series using only pre-treatment values
    to estimate the lambda parameter. Applies the same transformation to the full Series.
    
    Parameters:
        s (pd.Series): The full Series to transform (can include NaNs).
        pre_mask (pd.Series[bool]): Boolean mask identifying pre-treatment values.
        offset_eps (float): Small constant to shift data if nonpositive values exist.

    Returns:
        transformed (pd.Series): Box-Cox transformed series (NaNs preserved).
        lambda_bc (float): The fitted Box-Cox lambda from pre-treatment data.
        offset (float): The shift applied to make all values positive.
    """
    s = s.copy()
    non_nan_mask = s.notna()

    # Get pre-treatment non-NaN values
    pre_vals = s[pre_mask & non_nan_mask]

    if pre_vals.empty:
        raise ValueError("Pre-treatment data contains no valid values.")

    # Determine if a shift is needed to make values strictly positive
    min_val = pre_vals.min()
    offset = -min_val + offset_eps if min_val <= 0 else 0.0

    # Estimate Box-Cox lambda using shifted pre-treatment values
    pre_vals_shifted = pre_vals + offset
    transformed_pre, lambda_bc = boxcox(pre_vals_shifted)

    # Apply transformation to full series (only to non-NaN values)
    s_shifted = s + offset
    transformed_full = pd.Series(index=s.index, dtype=float)

    if lambda_bc == 0:
        transformed_full[non_nan_mask] = np.log(s_shifted[non_nan_mask])
    else:
        transformed_full[non_nan_mask] = ((s_shifted[non_nan_mask] ** lambda_bc) - 1) / lambda_bc

    return transformed_full, lambda_bc, offset


def boxcox_pre_int(df, features, pre_mask, post_mask, offset_eps=1e-6):
    """
    Applies Box-Cox transformations to selected features in a DataFrame using
    pre-treatment data to estimate lambda, then applies the transformation to
    the full series (pre and post), storing results in new columns.

    Parameters:
        df (pd.DataFrame): The dataframe with raw feature columns.
        features (list): List of feature names to transform.
        pre_mask (pd.Series[bool]): Boolean mask for pre-treatment period.
        post_mask (pd.Series[bool]): Boolean mask for post-treatment period.
        offset_eps (float): Small value to shift data if nonpositive values exist.

    Returns:
        pd.DataFrame: Modified dataframe with new _boxcox columns added.
    """
    df = df.copy()
    
    def apply_boxcox(x, lam, offset):
        x_shifted = x + offset
        if (x_shifted <= 0).any():
            raise ValueError("There are still non-positive values after shifting!")
        if lam == 0:
            return np.log(x_shifted)
        else:
            return (np.power(x_shifted, lam) - 1) / lam

    for feat in features:
        print(f"\nFeature: {feat}")
        trans_feat = feat + '_boxcox'
        df[trans_feat] = np.nan  # Initialize column

        # Pre-treatment transformation
        pre_values = df.loc[pre_mask, feat].dropna()
        min_val = pre_values.min()
        print("Minimum value (non-NA) before shift:", min_val)

        offset = -min_val + offset_eps if min_val <= 0 else 0.0
        pre_shifted = pre_values + offset

        if (pre_shifted <= 0).any():
            raise ValueError(f"Pre-treatment values for '{feat}' are not strictly positive after shift.")

        transformed_pre, lambda_bc = boxcox(pre_shifted)
        print("Optimal lambda for Box-Cox transformation:", lambda_bc)

        df.loc[pre_mask & df[feat].notna(), trans_feat] = transformed_pre

        # Post-treatment transformation
        post_values = df.loc[post_mask, feat].dropna()
        transformed_post = apply_boxcox(post_values, lambda_bc, offset)
        df.loc[post_mask & df[feat].notna(), trans_feat] = transformed_post

    return df