from __future__ import annotations
from typing import Optional, Literal
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dataprep import Dataprep
from .base import BaseSynth, VanillaOptimMixin
from .utils import HoldoutSplitter, CrossValidationResult
from .inference import ConformalInference

OptimizerMethod_t = Literal[
    "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "trust-constr"
]

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
    
    def fit(
        self,
        dataprep: Dataprep,
        lambda_: Optional[float] = None,
        optim_method: OptimizerMethod_t = "Nelder-Mead",
        optim_initial: Literal["equal", "ols"] = "equal",
        optim_options: dict = {"maxiter": 1000},
    ) -> None:
        """
        Fit the AugSynthGPT model with learned V.
        
        Parameters:
          dataprep: Dataprep object containing study data.
          lambda_: Ridge penalty parameter; if None, cross-validation is used.
          optim_method: Optimization method to pass to scipy.optimize.minimize.
          optim_initial: Either "equal" (starting with equal weights) or "ols".
          optim_options: Options for the optimizer.
        
        Returns:
          None.
        """
        # Require exactly one treated unit.
        if (isinstance(dataprep.treatment_identifier, (list, tuple)) and
            len(dataprep.treatment_identifier) > 1):
            raise ValueError("AugSynthGPT requires exactly one treated unit.")
            
        self.dataprep = dataprep

        # Retrieve covariate (predictor) matrices and outcome matrices.
        # Following Synth, we let:
        #    X0, X1 = predictors (controls and treated, respectively)
        #    Z0, Z1 = outcomes (controls and treated, for evaluation).
        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        self.Z0, self.Z1 = Z0, Z1

        X0_demean, X1_demean, Z0_normal, Z1_normal = self._normalize(X0, X1, Z0, Z1)

        # Stack for ridge adjustment
        X0_stacked = pd.concat([X0_demean, Z0_normal], axis=0)
        X1_stacked = pd.concat([X1_demean, Z1_normal], axis=0)

        # Convert for inner optimization
        X0_arr = X0_demean.to_numpy()
        X1_arr = X1_demean.to_numpy()
        Z0_arr = Z0.to_numpy()
        Z1_arr = Z1.to_numpy()

        n_cov = X0_arr.shape[0]

        # Outer optimization to learn V (as before)
        def outer_loss(log_v_diag):
            v_diag = np.exp(log_v_diag)
            V = np.diag(v_diag / np.sum(v_diag))
            W, _ = self.w_optimize(V_mat=V, X0=X0_arr, X1=X1_arr)
            loss = (Z1_arr - Z0_arr @ W).T @ (Z1_arr - Z0_arr @ W) / len(Z0_arr)
            return loss.item()

        if optim_initial == "equal":
            x0 = np.log(np.full(n_cov, 1 / n_cov))
        elif optim_initial == "ols":
            x0 = np.log(np.full(n_cov, 1 / n_cov))
        else:
            raise ValueError("Unknown option for optim_initial.")

        res = minimize(outer_loss, x0=x0, method=optim_method, options=optim_options)
        v_diag_opt = np.exp(res.x)
        V_opt = np.diag(v_diag_opt / np.sum(v_diag_opt))
        self.V = np.diag(V_opt)

        # Final inner optimization
        W, loss_W = self.w_optimize(V_mat=V_opt, X0=X0_arr, X1=X1_arr)

        # Ridge parameter selection: use the original (pandas) versions
        if lambda_ is None:
            lambdas = self.generate_lambdas(X0)
            # Pass the pandas objects (X0_demean, X1_demean) to cross_validate
            self.cv_result = self.cross_validate(X0_demean, X1_demean, lambdas)
            self.lambda_ = self.cv_result.best_lambda()
        else:
            self.lambda_ = lambda_

        W_ridge = self.solve_ridge(X1_stacked.to_numpy(), X0_stacked.to_numpy(), W, self.lambda_)
        self.W = W + W_ridge

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

        # Learn V using outer optimization (just like Synth)
        def outer_loss(log_v_diag):
            v_diag = np.exp(log_v_diag)
            V = np.diag(v_diag / np.sum(v_diag))
            W, _ = self.w_optimize(V_mat=V, X0=X0_arr, X1=X1_arr)
            loss = (Z1_arr - Z0_arr @ W).T @ (Z1_arr - Z0_arr @ W) / len(Z0_arr)
            return loss.item()

        # Initial guess
        v0 = np.full(n_cov, 1 / n_cov)
        x0 = np.log(v0)
        res = minimize(outer_loss, x0=x0, method='Nelder-Mead', options={"maxiter": 1000})
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


    def solve_ridge(
        self, A: np.ndarray, B: np.ndarray, W: np.ndarray, lambda_: float
    ) -> np.ndarray:
        """Calculate the ridge adjustment to the weights.
        This remains the same as in the original AugSynth.
        """
        M = A - B @ W
        N = np.linalg.inv(B @ B.T + lambda_ * np.identity(B.shape[0]))
        return M @ N @ B


    def cross_validate(self, X0: pd.DataFrame, X1: pd.Series, lambdas: np.ndarray, holdout_len: int = 5) -> CrossValidationResult:
        # Create identity matrix for the training portion.
        T = X0.shape[0]
        T_alt = X0.shape
        print('T=',T,'T_alt=',T_alt)
        V = np.identity(T - holdout_len)
        res = []
        for X0_t, X0_v, X1_t, X1_v in HoldoutSplitter(X0, X1, holdout_len=holdout_len):
            # Reset indices so that the number of rows is contiguous.
            X0_t = X0_t.reset_index(drop=True)
            X1_t = X1_t.reset_index(drop=True)
            # Now these splits have shape (T - holdout_len, n_controls) or (T - holdout_len,)
            W, _ = self.w_optimize(V_mat=V, X0=X0_t, X1=X1_t)
            this_res = []
            for l in lambdas:
                ridge_weights = self.solve_ridge(A=X1_t.to_numpy(), B=X0_t.to_numpy(), W=W, lambda_=l)
                W_aug = W + ridge_weights
                err = np.sum((X1_v.to_numpy() - X0_v.to_numpy() @ W_aug) ** 2)
                this_res.append(err)
            res.append(this_res)
        means = np.array(res).mean(axis=0)
        ses = np.array(res).std(axis=0) / np.sqrt(len(lambdas))
        return CrossValidationResult(lambdas, means, ses)


    def generate_lambdas(
        self, X: pd.DataFrame, lambda_min_ratio: float = 1e-8, n_lambda: int = 20
    ) -> np.ndarray:
        """Generate a suitable set of lambdas to run the cross-validation
        procedure on.

        :meta private:
        """
        _, sing, _ = np.linalg.svd(X.T)
        lambda_max = sing[0].item() ** 2.0
        scaler = lambda_min_ratio ** (1 / n_lambda)
        return lambda_max * (scaler ** np.array(range(n_lambda)))
    

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
    
    def summary_with_variance(self, round: int = 3, X0: Optional[pd.DataFrame] = None, X1: Optional[pd.Series] = None) -> pd.DataFrame:
        """Extended summary that includes the variance and skewness of each covariate."""
        if self.V is None:
            raise ValueError("No V matrix available; fit model first.")

        # Use stored dataprep if X0 and X1 are not provided.
        if self.dataprep is not None:
            X0, X1 = self.dataprep.make_covariate_mats()
        elif X0 is None or X1 is None:
            raise ValueError("No dataprep or covariate matrices provided.")

        if not isinstance(X0, pd.DataFrame) or not isinstance(X1, pd.Series):
            raise TypeError("X0 must be a DataFrame and X1 must be a Series.")

        # Compute synthetic values using weights
        synth_vals = X0 @ self.W

        # Compute equal-weighted sample mean of the covariates
        sample_mean = X0.mean(axis=1)

        # Calculate the variance and skewness of each predictor (row-wise across units)
        predictor_variance = X0.var(axis=1)
        predictor_skewness = X0.skew(axis=1)

        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            "V": self.V,  # V weights per predictor
            "Treated": X1,
            "Synthetic": synth_vals,
            "Sample Mean": sample_mean,
            "Variance": predictor_variance,
            "Skewness": predictor_skewness
        })

        return summary_df.round(round)


