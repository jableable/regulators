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

        X0_demean = X0.subtract(X0.mean(axis=1), axis=0)
        X1_demean = X1.subtract(X0.mean(axis=1), axis=0)

        Z0_demean = Z0.subtract(Z0.mean(axis=1), axis=0)
        Z1_demean = Z1.subtract(Z0.mean(axis=1), axis=0)

        Z0_std = Z0_demean.std(axis=1)
        X0_std = X0_demean.to_numpy().std(ddof=1).item()

        Z0_normal = Z0_demean.divide(Z0_std, axis=0) * X0_std
        Z1_normal = Z1_demean.divide(Z0_std, axis=0) * X0_std
        return X0_demean, X1_demean, Z0_normal, Z1_normal
    
    def project_to_simplex(self, W_aug: np.ndarray) -> np.ndarray:
        """
        Project W_aug back onto the simplex:
        - All elements >= 0
        - Sum of elements == 1
        """
        n = W_aug.shape[0]

        def objective(w):
            return np.sum((w - W_aug) ** 2)  # minimize Euclidean distance to W_aug

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0) for _ in range(n)]
        result = minimize(objective, W_aug, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError("Projection to simplex failed:", result.message)

        return result.x

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

        ## WE SWAP THE ROLES OF X0/X1 WITH Z0/Z1 TO ALIGN WITH CONVENTIONS
        X0, X1 = dataprep.make_covariate_mats()
        Z0, Z1 = dataprep.make_outcome_mats()
        self.Z0, self.Z1 = Z0, Z1

        Z0_demean, Z1_demean, X0_normal, X1_normal = self._normalize(Z0, Z1, X0, X1)
        #X0_demean, X1_demean, Z0_normal, Z1_normal = self._normalize(X0, X1, Z0, Z1)

        # Stack for ridge adjustment
        X0_stacked = pd.concat([Z0_demean, X0_normal], axis=0)  # NOT SURE IF THE ORDER IS CORRECT
        X1_stacked = pd.concat([Z1_demean, X1_normal], axis=0)  
        #X0_stacked = pd.concat([X0_normal, Z0_demean], axis=0)  # NOT SURE IF THE ORDER IS CORRECT
        #X1_stacked = pd.concat([X1_normal, Z1_demean], axis=0)  

        # Convert for inner optimization
        X0_arr = X0_normal.to_numpy()
        X1_arr = X1_normal.to_numpy()
        Z0_arr = Z0_demean.to_numpy()   # NOT SURE ABOUT WHETHER Z0/Z1 SHOULD BE NORMALIZED
        Z1_arr = Z1_demean.to_numpy()   # THESE ARE USED IN OUTER LOSS OPTIMIZATION BELOW
                                        # I THINK THEY SHOULD BUT COULDN'T HURT TO TEST

          
        alpha = .1  # Total regularization strength
        rho = 0.75    # Balance: 1.0 = pure L1, 0.0 = pure L2

        def outer_loss(log_v_diag):
            # Exponentiate and normalize v_diag
            v_diag = np.exp(log_v_diag)
            v_norm = v_diag / np.sum(v_diag)
            V = np.diag(v_norm)

            # Compute synthetic control weights with current V
            W, _ = self.w_optimize(V_mat=V, X0=X0_arr, X1=X1_arr)

            # Reconstruction loss (MSE)
            residual = Z1_arr - Z0_arr @ W
            loss = (residual.T @ residual) / len(Z0_arr)

            # Elastic net penalty on normalized V weights
            l1_penalty = np.sum(np.abs(v_norm))
            l2_penalty = np.sum(v_norm ** 2)
            penalty = alpha * (rho * l1_penalty + (1 - rho) * l2_penalty)

            return loss.item() + penalty

        n_cov = X0.shape[0]
        x0 = np.log(np.full(n_cov, 1 / n_cov))

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
            self.cv_result = self.cross_validate(Z0_demean, Z1_demean, lambdas)
            self.lambda_ = self.cv_result.best_lambda()
        else:
            self.lambda_ = lambda_
        W_ridge = self.solve_ridge(X1_stacked.to_numpy(), X0_stacked.to_numpy(), W, self.lambda_)
        beta = 1
        self.W = W + beta * W_ridge


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
            "V": self.V[:X0.shape[0]],  # V weights per predictor
            "Treated": X1,
            "Synthetic": synth_vals,
            "Sample Mean": sample_mean,
            "Variance": predictor_variance,
            "Skewness": predictor_skewness
        })

        return summary_df.round(round)
    

    def post_treatment_synthetic_ratio(self, treatment_date: str) -> float:
        """
        Computes the ratio of the post-treatment cumulative synthetic outcome
        to the post-treatment cumulative treated outcome.

        Parameters:
            treatment_date (str): The intervention date in 'YYYY-MM-DD' format.

        Returns:
            float: Ratio of synthetic post-treatment sum to treated post-treatment sum.
        """
        if self.Z0 is None or self.Z1 is None or self.W is None:
            raise ValueError("Model must be fit before computing post-treatment synthetic ratio.")

        synth_series = self.Z0.T @ self.W  # shape: (T,) Series-like
        treated_series = self.Z1  # shape: (T,) Series

        # Ensure datetime index
        synth_series.index = pd.to_datetime(synth_series.index)
        treated_series.index = pd.to_datetime(treated_series.index)

        post_mask = synth_series.index >= pd.to_datetime(treatment_date)
        synth_sum = synth_series.loc[post_mask].sum()
        treated_sum = treated_series.loc[post_mask].sum()

        if treated_sum == 0:
            raise ZeroDivisionError("Sum of post-treatment treated outcomes is zero.")

        return synth_sum / treated_sum



