import os

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch


class SHAPModelWrapper(torch.nn.Module):
    """
    Wraps the forecasting model so that SHAP always receives
    a 2D output tensor of shape (batch_size, output_dim).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)

        # SHAP expects (batch_size, outputs)
        if output.ndim == 1:
            output = output.unsqueeze(1)

        return output


class SHAPExplainer:
    """
    Computes SHAP explanations for the trained LSTM model.
    """

    def __init__(self, model, device):
        self.device = device

        self.model = SHAPModelWrapper(
            model.to(device)
        )

        self.model.eval()

    def explain(
        self,
        background,
        samples,
    ):
        """
        Compute SHAP values.

        Parameters
        ----------
        background : np.ndarray
            Background dataset.

        samples : np.ndarray
            Samples to explain.

        Returns
        -------
        np.ndarray
            SHAP values.
        """

        background = torch.tensor(
            background,
            dtype=torch.float32,
            device=self.device,
        )

        samples = torch.tensor(
            samples,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        test_output = self.model(samples[:1])

        print(
            "Model output shape:",
            tuple(test_output.shape),
        )

        explainer = shap.GradientExplainer(
            self.model,
            background,
        )

        shap_values = explainer.shap_values(
            samples,
        )

        if isinstance(shap_values, list):
            print(
                "SHAP values shape:",
                shap_values[0].shape,
            )
        else:
            print(
                "SHAP values shape:",
                shap_values.shape,
            )

        return shap_values

    def save_summary_plot(
        self,
        shap_values,
        samples,
        feature_names,
        save_dir="models/explainability",
    ):
        """
        Save SHAP summary plot.
        """

        os.makedirs(save_dir, exist_ok=True)

        # Handle list output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        samples = np.asarray(samples)

        # Average sequence dimension for samples
        if samples.ndim == 3:
            samples = samples.mean(axis=1)

        # ----------------------------------------------------
        # Reduce SHAP values to (samples, features)
        # ----------------------------------------------------
        if shap_values.ndim == 4:
            # (samples, sequence, features, outputs)
            shap_values = shap_values.squeeze(-1)      # -> (samples, sequence, features)
            shap_values = shap_values.mean(axis=1)     # -> (samples, features)

        elif shap_values.ndim == 3:
            # (samples, sequence, features)
            shap_values = shap_values.mean(axis=1)

        elif shap_values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_values.shape}"
            )

        print("Samples shape:", samples.shape)
        print("Final SHAP shape:", shap_values.shape)

        plt.figure(figsize=(10, 6))

        shap.summary_plot(
            shap_values,
            samples,
            feature_names=feature_names,
            show=False,
        )

        plt.tight_layout()

        output_file = os.path.join(
            save_dir,
            "shap_summary.png",
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved SHAP summary plot to: {output_file}")