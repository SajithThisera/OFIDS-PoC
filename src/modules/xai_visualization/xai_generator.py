import os
import shap
import matplotlib.pyplot as plt
import numpy as np


def generate_shap_explanations(model, test_features, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_features)

    # Check if shap_values has more than one class
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # If there are multiple classes, plot for the second class (index 1)
        shap.summary_plot(shap_values[1], test_features, plot_type="bar", show=False)
    else:
        # If only one class or a single set of shap values, plot what is available
        shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(shap_values_to_plot, test_features, plot_type="bar", show=False)

    # Save the plot
    plt.savefig(output_dir)
    print("SHAP summary plot saved to Outputs/SHAP_summary_plot.png")
    plt.close()
