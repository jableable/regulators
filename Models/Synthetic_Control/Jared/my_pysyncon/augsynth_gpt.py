from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dataprep import Dataprep
from .base import BaseSynth, VanillaOptimMixin
from .utils import HoldoutSplitter, CrossValidationResult



class AugSynthGPT(BaseSynth, VanillaOptimMixin):
    """Augmented Synthetic Control method, modified to behave like Synth.
    
    In this version:
      - The covariates used for matching are taken from `dataprep.make_covariate_mats()`
        and are used in the weight optimization and ridge adjustment.
      - The outcomes from `dataprep.make_outcome_mats()` are kept separate
        (e.g. for MSPE calculations) but are not automatically added to the matching step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lambda_: Optional[float] = None
        self.cv_result: Optional[CrossValidationResult] = None
        # These will store the outcome matrices for evaluation/diagnostics.
        self.Z0 = None  
        self.Z1 = None

    def _normalize(
        self, X0: pd.DataFrame, X1: pd.Series, Z0: pd.DataFrame, Z1: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Normalize the data before the weight calculation."""

        # Demean the covariates (predictors)
        X0_demean = X0.subtract(X0.mean(axis=1), axis=0)
        X1_demean = X1.subtract(X0.mean(axis=1), axis=0)

        # Demean the outcome
        Z0_demean = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_demean = Z1.subtract(Z0.mean(axis=1), axis=0)

        # Normalize the outcome to match the scale of the predictors
        Z0_std = Z0_demean.std(axis=1)
        X0_std = X0_demean.to_numpy().std(ddof=1).item()

        Z0_normal = Z0_demean.divide(Z0_std, axis=0) * X0_std
        Z1_normal = Z1_demean.divide(Z0_std, axis=0) * X0_std

        return X0_demean, X1_demean, Z0_normal, Z1_normal
    
    def fit2(self, dataprep: Dataprep, lambda_: Optional[float] = None) -> None:
        if (
            isinstance(dataprep.treatment_identifier, (list, tuple)) and
            len(dataprep.treatment_identifier) > 1
        ):
            raise ValueError("AugSynth requires exactly one treated unit.")

        self.dataprep = dataprep

        # Get predictor (covariate) matrices and outcomes
        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        self.Z0, self.Z1 = Z0, Z1  # Save for MSPE or plotting

        # Normalize predictors and outcomes
        X0_demean, X1_demean, Z0_normal, Z1_normal = self._normalize(X0, X1, Z0, Z1)

        # Stack for ridge adjustment
        X0_stacked = pd.concat([X0_demean, Z0_normal], axis=0)
        X1_stacked = pd.concat([X1_demean, Z1_normal], axis=0)

        X0_arr = X0_demean.to_numpy()
        X1_arr = X1_demean.to_numpy()
        Z0_arr = Z0.to_numpy()
        Z1_arr = Z1.to_numpy()

        n_cov = X0_arr.shape[0]

        # 🔍 Learn V using outer optimization (just like Synth)
        def outer_loss(log_v_diag):
            v_diag = np.exp(log_v_diag)
            V = np.diag(v_diag / np.sum(v_diag))
            W, _ = self.w_optimize(V_mat=V, X0=X0_arr, X1=X1_arr)
            loss = (Z1_arr - Z0_arr @ W).T @ (Z1_arr - Z0_arr @ W) / len(Z0_arr)
            return loss.item()

        # Initial guess: log(1) = 0 for all entries
        res = minimize(outer_loss, x0=np.zeros(n_cov), method='Nelder-Mead', options={"maxiter": 1000})
        v_diag_opt = np.exp(res.x)
        V_opt = np.diag(v_diag_opt / np.sum(v_diag_opt))
        self.V = np.diag(V_opt)  # Save diagonal of V

        # Final inner optimization using learned V
        W, _ = self.w_optimize(V_mat=V_opt, X0=X0_arr, X1=X1_arr)

        # Ridge adjustment
        if lambda_ is None:
            lambdas = self.generate_lambdas(X0)
            self.cv_result = self.cross_validate(X0_arr, X1_arr, lambdas)
            self.lambda_ = self.cv_result.best_lambda()
        else:
            self.lambda_ = lambda_

        W_ridge = self.solve_ridge(X1_stacked.to_numpy(), X0_stacked.to_numpy(), W, self.lambda_)
        self.W = W + W_ridge

    def fit(self, dataprep: Dataprep, lambda_: Optional[float] = None) -> None:
        """
        Fit the augmented synthetic control model.
        
        In this rewrite the roles of predictors and outcomes are like Synth:
          - X0, X1 are the covariate (predictor) matrices.
          - Z0, Z1 are the outcome time-series.
        """
        # Check that only one treated unit is given.
        if (isinstance(dataprep.treatment_identifier, (list, tuple)) and
            len(dataprep.treatment_identifier) > 1):
            raise ValueError("AugSynth requires exactly one treated unit.")

        self.dataprep = dataprep

        # Get predictors (covariate matrices) and outcomes.
        X0, X1 = dataprep.make_covariate_mats()   # predictors (controls, treated)
        Z0, Z1 = dataprep.make_outcome_mats()       # outcomes for evaluation

        # Save outcomes for later use (e.g. to compute MSPE).
        self.Z0 = Z0
        self.Z1 = Z1

        # -------- Normalize Predictors (mimicking Synth) --------
        # The idea is to scale the predictors row-wise (i.e. across covariates).
        X = pd.concat([X0, X1], axis=1)
        # Divide each row by its standard deviation.
        X_scaled = X.divide(X.std(axis=1), axis=0)
        # Separate the control predictors and the treated predictor.
        X0_scaled = X_scaled.drop(columns=X1.name)
        X1_scaled = X_scaled[X1.name]

        # Convert to NumPy arrays for the matching procedure.
        X0_arr = X0_scaled.to_numpy()
        X1_arr = X1_scaled.to_numpy()

        # -------- Ridge Parameter Selection (if lambda not provided) --------
        # Use predictors for lambda generation and cross-validation.
        if lambda_ is None:
            lambdas = self.generate_lambdas(X0)  # based on covariate matrix X0
            self.cv_result = self.cross_validate(X0_arr, X1_arr, lambdas)
            self.lambda_ = self.cv_result.best_lambda()
        else:
            self.lambda_ = lambda_

        # -------- Weight Optimization ----------
        # Create the V matrix based on the number of predictors (rows of X0_arr).
        n_r, _ = X0_arr.shape
        V_mat = np.diag(np.full(n_r, 1 / n_r))
        self.V = np.diag(V_mat)

        # Compute the initial weights using the predictors only.
        W, loss_W = self.w_optimize(V_mat=V_mat, X0=X0_arr, X1=X1_arr)

        # -------- Ridge Adjustment ----------
        # Solve for the ridge adjustment using the predictor matrices.
        # (In the original code outcomes were stacked into the predictors.
        #  Here, we only use the predictors.)
        W_ridge = self.solve_ridge(X1_arr, X0_arr, W, self.lambda_)
        self.W = W + W_ridge

    def solve_ridge(
        self, A: np.ndarray, B: np.ndarray, W: np.ndarray, lambda_: float
    ) -> np.ndarray:
        """Calculate the ridge adjustment to the weights.
        This remains the same as in the original AugSynth.
        """
        M = A - B @ W
        N = np.linalg.inv(B @ B.T + lambda_ * np.identity(B.shape[0]))
        return M @ N @ B

    def cross_validate(
        self, X0: np.ndarray, X1: np.ndarray, lambdas: np.ndarray, holdout_len: int = 1
    ) -> CrossValidationResult:
        """Perform cross-validation to choose the ridge parameter.
        
        Parameters:
          X0, X1: Predictor matrices (from covariate data)
          lambdas: Array of candidate lambda values.
          holdout_len: Length of hold-out portion for CV.
        
        Returns:
          CrossValidationResult with means and standard errors.
        """
        # Create identity matrix based on number of predictors available for training.
        V = np.identity(X0.shape[0] - holdout_len)
        res = []
        for X0_t, X0_v, X1_t, X1_v in HoldoutSplitter(X0, X1, holdout_len=holdout_len):
            W, _ = self.w_optimize(V_mat=V, X0=X0_t, X1=X1_t)
            this_res = []
            for l in lambdas:
                ridge_weights = self.solve_ridge(A=X1_t, B=X0_t, W=W, lambda_=l)
                W_aug = W + ridge_weights
                # Compute the squared error on the hold-out set.
                # Assuming X1_v and X0_v are in NumPy array format;
                # adjust if they are pandas DataFrames/Series.
                err = np.sum((X1_v - X0_v @ W_aug) ** 2)
                this_res.append(err)
            res.append(this_res)
        means = np.array(res).mean(axis=0)
        ses = np.array(res).std(axis=0) / np.sqrt(len(lambdas))
        return CrossValidationResult(lambdas, means, ses)

    def summary(
        self,
        round: int = 3,
        X0: Optional[pd.DataFrame] = None,
        X1: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Summary of predictor balance and V weights for AugSynth."""

        if self.V is None:
            raise ValueError("No V matrix available; fit model first.")

        # If not explicitly passed, use the data stored in self.dataprep
        if self.dataprep is not None:
            X0, X1 = self.dataprep.make_covariate_mats()
        elif X0 is None or X1 is None:
            raise ValueError("No dataprep or covariate matrices provided.")

        if not isinstance(X0, pd.DataFrame) or not isinstance(X1, pd.Series):
            raise TypeError("X0 must be a DataFrame and X1 must be a Series.")

        # Compute synthetic values using weights
        synth_vals = X0 @ self.W

        # Compute equal-weighted sample mean
        sample_mean = X0.mean(axis=1)

        summary_df = pd.DataFrame({
            "V": self.V,
            "Treated": X1,
            "Synthetic": synth_vals,
            "Sample Mean": sample_mean,
        })

        return summary_df.round(round)