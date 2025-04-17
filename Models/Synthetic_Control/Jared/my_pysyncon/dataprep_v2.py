from __future__ import annotations
from typing import Any, Iterable, Union, Optional, Literal, Sequence, Mapping, Tuple

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from pandas._typing import Axes

# 1. Update allowed operators to include "std-box-cox"
AGG_OP = ("mean", "std", "median", "sum", "count", "max", "min", "var", "std-box-cox")
# Also update the type literal accordingly.
PredictorsOp_t = Literal["mean", "std", "median", "sum", "count", "max", "min", "var", "std-box-cox"]

IsinArg_t = Union[Iterable, pd.Series, dict]
SpecialPredictor_t = Tuple[
    Any, Union[pd.Series, pd.DataFrame, Sequence, Mapping], PredictorsOp_t
]


class Dataprep_v2:
    def __init__(
        self,
        foo: pd.DataFrame,
        predictors: Axes,
        predictors_op: PredictorsOp_t,
        dependent: Any,
        unit_variable: Any,
        time_variable: Any,
        treatment_identifier: Union[Any, list, tuple],
        controls_identifier: Union[list, tuple],
        time_predictors_prior: IsinArg_t,
        time_optimize_ssr: IsinArg_t,
        special_predictors: Optional[Iterable[SpecialPredictor_t]] = None,
    ) -> None:
        if not isinstance(foo, pd.DataFrame):
            raise TypeError("foo must be pandas.DataFrame.")
        self.foo = foo

        for predictor in predictors:
            if predictor not in foo.columns:
                raise ValueError(f"predictor {predictor} not in foo columns.")
        self.predictors = predictors

        if predictors_op not in AGG_OP:
            agg_op_str = ", ".join([f'"{o}"' for o in AGG_OP])
            raise ValueError(f"predictors_op must be one of {agg_op_str}.")
        self.predictors_op = predictors_op

        if dependent not in foo.columns:
            raise ValueError(f"dependent {dependent} not in foo columns.")
        self.dependent = dependent

        if unit_variable not in foo.columns:
            raise ValueError(f"unit_variable {unit_variable} not in foo columns.")
        self.unit_variable = unit_variable

        if time_variable not in foo.columns:
            raise ValueError(f"time_variable {time_variable} not in foo columns.")
        self.time_variable = time_variable

        if foo[[unit_variable, time_variable]].duplicated().any():
            raise ValueError("Multiple rows found in `foo` for same [unit, time] pairs.")

        if isinstance(treatment_identifier, (list, tuple)):
            for treated in treatment_identifier:
                if treated not in foo[unit_variable].values:
                    raise ValueError(f'treatment_identifier {treated} not found in foo["{unit_variable}"].')
        else:
            if treatment_identifier not in foo[unit_variable].values:
                raise ValueError(f'treatment_identifier {treatment_identifier} not found in foo["{unit_variable}"].')
        if isinstance(treatment_identifier, (list, tuple)) and len(treatment_identifier) == 1:
            self.treatment_identifier = treatment_identifier[0]
        else:
            self.treatment_identifier = treatment_identifier

        if not isinstance(controls_identifier, (list, tuple)):
            raise TypeError("controls_identifier should be an list or tuple")
        for control in controls_identifier:
            if isinstance(self.treatment_identifier, (list, tuple)):
                if control in treatment_identifier:
                    raise ValueError(f"{control} in both treatment_identifier and controls_identifier.")
            else:
                if control == treatment_identifier:
                    raise ValueError("treatment_identifier in controls_identifier.")
            if control not in foo[unit_variable].values:
                raise ValueError(f'controls_identifier {control} not found in foo["{unit_variable}"].')
        self.controls_identifier = controls_identifier

        if self.foo[self.foo[self.time_variable].isin(time_predictors_prior)].empty:
            raise ValueError(f"foo has no rows in the time range `time_predictors_prior`.")
        self.time_predictors_prior = time_predictors_prior

        if self.foo[self.foo[self.time_variable].isin(time_optimize_ssr)].empty:
            raise ValueError(f"foo has no rows in the time range `time_optimize_ssr`.")
        self.time_optimize_ssr = time_optimize_ssr

        if special_predictors:
            for el in special_predictors:
                if not isinstance(el, tuple) or len(el) != 3:
                    raise ValueError("Elements of special_predictors should be tuples of length 3.")
                predictor, time_range, op = el
                if predictor not in foo.columns:
                    raise ValueError(f"{predictor} in special_predictors not in foo columns.")
                if self.foo[self.foo[self.time_variable].isin(time_range)].empty:
                    raise ValueError(f"foo has no rows in the time range {time_range} for `special_predictor` {el}.")
                if op not in AGG_OP:
                    agg_op_str = ", ".join([f'"{o}"' for o in AGG_OP])
                    raise ValueError(f"{op} in special_predictors must be one of {agg_op_str}.")
        self.special_predictors = special_predictors

        # Initialize a dictionary to store Box–Cox lambda values per special predictor.
        self.boxcox_lambdas: dict[str, float] = {}

    def make_covariate_mats(self) -> tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
        X_nonspecial = (
            self.foo[self.foo[self.time_variable].isin(self.time_predictors_prior)]
            .groupby(self.unit_variable)[self.predictors]
            .agg(self.predictors_op)
            .T
        )
        X1_nonspecial = X_nonspecial[self.treatment_identifier]
        X0_nonspecial = X_nonspecial[list(self.controls_identifier)]

        if self.special_predictors is None:
            return X0_nonspecial, X1_nonspecial

        # Process special predictors for control units.
        X0_special = list()
        for control in self.controls_identifier:
            this_control = list()
            for predictor, time_period, op in self.special_predictors:
                mask = (self.foo[self.unit_variable] == control) & (self.foo[self.time_variable].isin(time_period))
                if op == "std-box-cox":
                    agg_value = self.foo[mask][predictor].std()
                    key = f"special.{len(this_control)+1}.{predictor}"
                    if agg_value <= 0 or pd.isna(agg_value):
                        transformed = np.nan  # Optionally, warn or handle negative/zero values.
                    else:
                        transformed, fitted_lambda = boxcox([agg_value])
                        self.boxcox_lambdas[key] = fitted_lambda
                        transformed = transformed[0]
                    this_control.append(transformed)
                else:
                    this_control.append(self.foo[mask][predictor].agg(op))
            X0_special.append(this_control)

        X0_special_columns = []
        for idx, (predictor, _, _) in enumerate(self.special_predictors, 1):
            X0_special_columns.append(f"special.{idx}.{predictor}")

        X0_special = pd.DataFrame(
            X0_special, columns=X0_special_columns, index=self.controls_identifier
        ).T
        X0 = pd.concat([X0_nonspecial, X0_special], axis=0)

        # Process special predictors for the treated unit.
        X1_special = []
        if isinstance(self.treatment_identifier, (list, tuple)):
            for treated in self.treatment_identifier:
                this_treated = []
                for predictor, time_period, op in self.special_predictors:
                    mask = (self.foo[self.unit_variable] == treated) & (self.foo[self.time_variable].isin(time_period))
                    if op == "std-box-cox":
                        agg_value = self.foo[mask][predictor].std()
                        key = f"special.{len(this_treated)+1}.{predictor}"
                        if agg_value <= 0 or pd.isna(agg_value):
                            transformed = np.nan
                        else:
                            transformed, fitted_lambda = boxcox([agg_value])
                            self.boxcox_lambdas[key] = fitted_lambda
                            transformed = transformed[0]
                        this_treated.append(transformed)
                    else:
                        this_treated.append(self.foo[mask][predictor].agg(op))
                X1_special.append(this_treated)
            X1_special = pd.DataFrame(
                X1_special, columns=X0_special_columns, index=self.treatment_identifier
            ).T
        else:
            for i, (predictor, time_period, op) in enumerate(self.special_predictors, 1):
                mask = (self.foo[self.unit_variable] == self.treatment_identifier) & (self.foo[self.time_variable].isin(time_period))
                if op == "std-box-cox":
                    agg_value = self.foo[mask][predictor].std()
                    key = f"special.{i}.{predictor}"
                    if agg_value <= 0 or pd.isna(agg_value):
                        transformed = np.nan
                    else:
                        transformed, fitted_lambda = boxcox([agg_value])
                        self.boxcox_lambdas[key] = fitted_lambda
                        transformed = transformed[0]
                    X1_special.append(transformed)
                else:
                    X1_special.append(self.foo[mask][predictor].agg(op))
            X1_special = pd.Series(X1_special, index=X0_special_columns).rename(self.treatment_identifier)
        X1 = pd.concat([X1_nonspecial, X1_special], axis=0)
        return X0, X1

    def make_outcome_mats(self, time_period: Optional[IsinArg_t] = None) -> tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
        time_period = time_period if time_period is not None else self.time_optimize_ssr
        Z = self.foo[self.foo[self.time_variable].isin(time_period)].pivot(
            index=self.time_variable, columns=self.unit_variable, values=self.dependent
        )
        Z0, Z1 = Z[list(self.controls_identifier)], Z[self.treatment_identifier]
        return Z0, Z1

    def __str__(self) -> str:
        str_rep = (
            "Dataprep\n"
            f"Treated unit: {self.treatment_identifier}\n"
            f"Dependent variable: {self.dependent}\n"
            f"Control units: {', '.join([str(c) for c in self.controls_identifier])}\n"
            f"Time range in data: {min(self.foo[self.time_variable])}"
            f" - {max(self.foo[self.time_variable])}\n"
            f"Time range for loss minimization: {self.time_optimize_ssr}\n"
            f"Time range for predictors: {self.time_predictors_prior}\n"
            f"Predictors: {', '.join([str(p) for p in self.predictors])}\n"
        )
        if self.special_predictors:
            str_special_pred = ""
            for predictor, time_range, op in self.special_predictors:
                rep = f"    `{predictor}` over `{time_range}` using `{op}`\n"
                str_special_pred += rep
            str_rep += f"Special predictors:\n{str_special_pred}"
        return str_rep
