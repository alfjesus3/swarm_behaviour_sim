import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gc
from typing import Callable
from collections import defaultdict, Counter
from itertools import combinations

from swarm_hydra.entry_point import *

from swarm_hydra.my_experi.multiclass_class_experi_preprocessing import *

# A logger for this file
log = logging.getLogger(__name__)


###############################################################################
# Self-Organizing-Maps utils
###############################################################################


def visualize_training_errors(
    q_errors: list,
    t_errors: list,
    d_errors: list,
    num_iteration: int,
    file_name: str,
) -> None:
    """"""
    plt.figure(figsize=(8, 12), dpi=300)

    plt.subplot(3, 1, 1)
    plt.plot(np.arange(num_iteration), q_errors)
    plt.ylabel("quantization error")

    plt.subplot(3, 1, 2)
    plt.plot(np.arange(num_iteration), t_errors)
    plt.ylabel("topographic error")

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(num_iteration), d_errors)
    plt.ylabel("divergence measure")
    plt.xlabel("iteration index")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        file_name,
        format="png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
    )
    plt.close("all")  # Explicitly close all figures
    gc.collect()  # Force garbage collection


def visualize_GSOM_growth(growth_history, title="Network Growth", save_path=None):
    """Visualize how the network size grew during training"""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(
        growth_history["iterations"],
        growth_history["node_count"],
        "b-",
        linewidth=2,
    )
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel("Number of Nodes")
    plt.grid(True)

    plt.tight_layout()
    fig.savefig(
        save_path,
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0,
    )
    plt.close("all")
    gc.collect()  # Force garbage collection


def visualize_samples_per_neuron(
    som_labels_map: dict,
    label_names: dict,
    colors: dict,
    file_name: str,
    gt_labels: dict = None,
) -> None:
    """Simplified SOM visualization focused on color proportions without text labels.

    Args:
        som_labels_map: Dictionary mapping coordinates to predicted label counts
        label_names: Dict of label names
        colors: Dictionary mapping labels to colors
        file_name: Output file name
        gt_labels: Dictionary mapping coordinates to ground truth labels (optional)
    """
    # Extract all coordinates from the som_labels_map
    all_coords = list(som_labels_map.keys())

    if not all_coords:
        raise ValueError("No nodes found in som_labels_map")

    # Calculate grid dimensions based on coordinates
    # Important: the X and y are switched
    max_x = max(coord[1] for coord in all_coords)
    min_x = min(coord[1] for coord in all_coords)
    max_y = max(coord[0] for coord in all_coords)
    min_y = min(coord[0] for coord in all_coords)

    # Create label color mapping
    label_color_map = colors.copy()

    # Create figure with reduced DPI for smaller file size
    plt.figure(
        figsize=(
            max(8, (max_y - min_y + 2) * 0.4),
            max(8, (max_x - min_x + 2) * 0.4),
        ),
        dpi=300,
    )
    ax = plt.gca()

    # Draw grid lines
    for i in range(min_x - 1, max_x + 2):
        plt.axhline(y=i, color="lightgray", linestyle="-", alpha=0.5, linewidth=0.5)
    for j in range(min_y - 1, max_y + 2):
        plt.axvline(x=j, color="lightgray", linestyle="-", alpha=0.5, linewidth=0.5)

    # Draw grid labels - adjusted for bottom-left origin
    for i in range(min_x, max_x + 1):
        plt.text(min_y - 0.5, i, str(i), ha="right", va="center", fontsize=7)
    for j in range(min_y, max_y + 1):
        plt.text(j, min_x - 0.5, str(j), ha="center", va="bottom", fontsize=7)

    # Set pie chart radius based on grid density
    radius = 0.35

    # Plot each node as a pie chart at its coordinates
    for position in all_coords:
        i, j = position

        # Get label counts for this node and consolidate identical labels
        label_counts = {}
        for label in som_labels_map[position]:
            count = som_labels_map[position][label]
            if count > 0:
                if label in label_counts:
                    label_counts[label] += count
                else:
                    label_counts[label] = count

        # If no labels are present, draw empty node marker
        if not label_counts:
            plt.scatter(i, j, color="lightgray", s=50, marker="o", alpha=0.3)
            continue

        # Prepare data for pie chart - labels are now consolidated
        used_labels = list(label_counts.keys())
        used_fracs = list(label_counts.values())
        used_colors = [label_color_map.get(lbl, "gray") for lbl in used_labels]

        # Create pie chart with individual wedge styling based on ground truth
        if gt_labels is not None:
            # Create pie chart first with default properties
            wedges, _ = ax.pie(
                used_fracs,
                colors=used_colors,
                center=(i, j),
                radius=radius,
                wedgeprops=dict(edgecolor="w", linewidth=0.5),
            )

            gt_label = str(gt_labels[position].astype(np.int32).item())
            # Apply individual styling to each wedge
            for wedge, label in zip(wedges, used_labels):
                if label == gt_label:
                    # Correct prediction - keep normal white border
                    wedge.set_edgecolor("w")
                    wedge.set_linewidth(0.5)
                else:
                    # Incorrect prediction - thick light grey border
                    wedge.set_edgecolor("#C0C0C0")  # Using hex color for better control
                    wedge.set_linewidth(3)
        else:
            # No ground truth available - use normal white border for all wedges
            ax.pie(
                used_fracs,
                colors=used_colors,
                center=(i, j),
                radius=radius,
                wedgeprops=dict(edgecolor="w", linewidth=0.5),
            )

    # Create simplified legend with just the main color categories
    legend_handles = [
        matplotlib.patches.Patch(color=colors[k], label=label_names[k])
        for k in label_names.keys()
    ]

    # Add legend entries for border meanings if ground truth is provided
    if gt_labels is not None:
        legend_handles.append(
            matplotlib.patches.Patch(
                facecolor="white",
                edgecolor="#C0C0C0",
                linewidth=2,
                label="Incorrect Label",
            )
        )

    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(legend_handles),
        fontsize=11,
        frameon=False,
    )

    # Set axis limits with some padding
    plt.xlim(min_y - 1, max_y + 1)
    plt.ylim(min_x - 1, max_x + 1)

    # Hide regular axis
    ax.set_axis_off()

    # Add title
    title = "SOM Node Distribution"
    if gt_labels is not None:
        title += " (with Ground Truth Comparison)"
    plt.title(title, fontsize=12, pad=20)

    # Adjust layout and save with reduced DPI
    plt.tight_layout()
    plt.savefig(
        file_name,
        format="png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
    )
    plt.close("all")
    gc.collect()  # Force garbage collection


def visualize_rect_u_matrix(
    som_u_matrix: np.ndarray, file_name: str, x_range: tuple, y_range: tuple
) -> None:
    """
    Adapted from https://github.com/JustGlowing/minisom/blob/master/examples/BasicUsage.ipynb
    """
    fig = plt.figure(figsize=(9, 9))

    # Get matrix dimensions
    height, width = som_u_matrix.shape

    # Create the plot
    plt.pcolor(
        som_u_matrix.T, cmap=matplotlib.cm.Blues
    )  # plotting the distance map as background
    plt.colorbar(label="distance from neurons in the neighbourhood")

    # Set axis limits and labels
    plt.xlim(0, width)
    plt.ylim(0, height)

    # Calculate tick positions based on the desired range
    x_ticks = plt.xticks()[0]  # Get current tick positions
    y_ticks = plt.yticks()[0]  # Get current tick positions

    # Map tick positions to the desired range
    x_tick_labels = [
        f"{x_range[0] + (x_range[1] - x_range[0]) * tick / width:.1f}"
        for tick in x_ticks
        if 0 <= tick <= width
    ]
    y_tick_labels = [
        f"{y_range[0] + (y_range[1] - y_range[0]) * tick / height:.1f}"
        for tick in y_ticks
        if 0 <= tick <= height
    ]

    # Filter ticks to only include those within bounds
    valid_x_ticks = [tick for tick in x_ticks if 0 <= tick <= width]
    valid_y_ticks = [tick for tick in y_ticks if 0 <= tick <= height]

    plt.xticks(valid_x_ticks, x_tick_labels)
    plt.yticks(valid_y_ticks, y_tick_labels)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rectangular U-Matrix")

    plt.tight_layout()
    fig.savefig(
        file_name,
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0,
    )
    plt.close("all")
    gc.collect()  # Force garbage collection


def visualize_hexag_u_matrix(
    som_n_coords: tuple,
    som_u_matrix: np.ndarray,
    som_weights: np.ndarray,
    file_name: str,
    x_range: tuple,
    y_range: tuple,
) -> None:
    """
    Adapted from https://github.com/JustGlowing/minisom/blob/master/examples/HexagonalTopology.ipynb
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    xx, yy = som_n_coords
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.set_aspect("equal")

    hex_radius = 0.95 / np.sqrt(3)
    hex_height = np.sqrt(3) / 2

    # Get SOM dimensions
    som_width = som_weights.shape[0]
    som_height = som_weights.shape[1]

    # iteratively add hexagons
    all_x, all_y = [], []
    for i in range(som_weights.shape[0]):
        for j in range(som_weights.shape[1]):
            x = xx[(i, j)]
            y = yy[(i, j)] * hex_height
            all_x.append(x)
            all_y.append(y)

            hex_patch = matplotlib.patches.RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                facecolor=matplotlib.cm.Blues(som_u_matrix[i, j]),
                alpha=0.4,
                edgecolor="gray",
            )
            ax.add_patch(hex_patch)

    # Set x and y limits with one hexagon border padding
    plot_x_min, plot_x_max = min(all_x), max(all_x)
    plot_y_min, plot_y_max = min(all_y), max(all_y)

    ax.set_xlim(plot_x_min - 1, plot_x_max + 1)
    ax.set_ylim(plot_y_min - hex_height, plot_y_max + hex_height)

    # Create custom tick positions and labels
    x_tick_positions = np.arange(som_width) - 0.5
    y_tick_positions = np.arange(som_height) * hex_height

    # Map tick positions to the desired range
    x_tick_labels = [
        f"{x_range[0] + (x_range[1] - x_range[0]) * i / (som_width - 1):.1f}"
        for i in range(som_width)
    ]
    y_tick_labels = [
        f"{y_range[0] + (y_range[1] - y_range[0]) * j / (som_height - 1):.1f}"
        for j in range(som_height)
    ]

    plt.xticks(x_tick_positions, x_tick_labels)
    plt.yticks(y_tick_positions, y_tick_labels)

    # Add axis labels
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = matplotlib.colorbar.ColorbarBase(
        ax_cb, cmap=matplotlib.cm.Blues, orientation="vertical", alpha=0.4
    )
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel(
        "distance from neurons in the neighbourhood", rotation=270, fontsize=16
    )
    fig.add_axes(ax_cb)

    plt.title("Hexagonal U-Matrix")

    plt.tight_layout()
    fig.savefig(
        file_name,
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0,
    )
    plt.close("all")
    gc.collect()  # Force garbage collection


def visualize_feature_influence(
    som_weights,
    feature_names: list,
    file_name: str,
    max_features_per_page: int = 16,
) -> None:
    """Visualize the influence of each feature on the MiniSom map with pagination for memory efficiency"""
    x_dim, y_dim, total_num_features = som_weights.shape

    # Process features in batches to reduce memory usage
    num_batches = int(np.ceil(len(feature_names) / max_features_per_page))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_features_per_page
        end_idx = min(start_idx + max_features_per_page, len(feature_names))
        batch_features = feature_names[start_idx:end_idx]

        # Determine subplot grid size for this batch
        n_cols = int(np.ceil(np.sqrt(len(batch_features))))
        n_rows = int(np.ceil(len(batch_features) / n_cols))

        # Create figure for this batch
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = (
            np.array(axes).flatten()
            if isinstance(axes, np.ndarray)
            else np.array([axes]).flatten()
        )

        for i, feature in enumerate(batch_features):
            feature_idx = start_idx + i
            ax = axes[i]

            if feature_idx < total_num_features:
                # Extract the feature plane
                feature_plane = som_weights[:, :, feature_idx]

                # Plot the feature values as a heatmap
                im = ax.imshow(feature_plane, cmap="viridis", origin="lower")
                ax.set_title(f"Feature: {feature}")
                ax.set_xticks(range(x_dim))
                ax.set_yticks(range(y_dim))
                ax.set_xticklabels(range(x_dim))
                ax.set_yticklabels(range(y_dim))
                ax.grid(True, linestyle="--", alpha=0.5)

                # Add colorbar
                plt.colorbar(im, ax=ax, label=f"{feature} value")

        # Remove unused subplots
        for j in range(len(batch_features), len(axes)):
            fig.delaxes(axes[j])

        batch_suffix = f"_batch{batch_idx}" if num_batches > 1 else ""
        plt.tight_layout()
        fig.savefig(
            file_name.replace(".png", f"{batch_suffix}.png"),
            format="png",
            bbox_inches="tight",
            dpi=400,
            pad_inches=0,
        )
        plt.close("all")
        gc.collect()  # Force garbage collection


def visualize_time_series_feature_influence(
    som_weights,
    feature_names: list,
    file_name: str,
    x_range: tuple,
    y_range: tuple,
    max_features_per_page: int = 6,
) -> None:
    """Visualize time series feature influence with pagination for memory efficiency"""
    x_dim, y_dim, total_num_features = som_weights.shape

    # Process features in batches to reduce memory usage
    num_batches = int(np.ceil(len(feature_names) / max_features_per_page))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_features_per_page
        end_idx = min(start_idx + max_features_per_page, len(feature_names))
        batch_features = feature_names[start_idx:end_idx]

        # Determine subplot grid
        n_cols = min(3, len(batch_features))
        n_rows = int(np.ceil(len(batch_features) / n_cols))

        # Create figure for this batch
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = (
            np.array(axes).flatten()
            if isinstance(axes, np.ndarray)
            else np.array([axes]).flatten()
        )

        for i, feature in enumerate(batch_features):
            feature_idx = start_idx + i
            ax = axes[i]

            ax.set_title(f"Feature: {feature}")

            # Set axis limits with padding
            ax.set_xlim(0 - 2, x_dim + 2)
            ax.set_ylim(0 - 2, y_dim + 2)

            # Don't invert y-axis to start from bottom left
            # ax.invert_yaxis()  # Removed this line

            # Create custom tick positions and labels
            # X-axis ticks
            x_tick_positions = np.linspace(0, x_dim, min(6, x_dim + 1))
            x_tick_labels = [
                f"{x_range[0] + (x_range[1] - x_range[0]) * pos / x_dim:.1f}"
                for pos in x_tick_positions
            ]
            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels, fontsize=8)

            # Y-axis ticks (note: we need to reverse the mapping since we're not inverting)
            y_tick_positions = np.linspace(0, y_dim, min(6, y_dim + 1))
            y_tick_labels = [
                f"{y_range[0] + (y_range[1] - y_range[0]) * pos / y_dim:.1f}"
                for pos in y_tick_positions
            ]
            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels(y_tick_labels, fontsize=8)

            # Add axis labels
            ax.set_xlabel("X", fontsize=10)
            ax.set_ylabel("Y", fontsize=10)

            # Calculate the number of dimensions per feature
            num_dims_feature = total_num_features // len(feature_names)

            # Process one row of neurons at a time to reduce memory usage
            for x in range(x_dim):
                for y in range(y_dim):
                    w_vector = som_weights[
                        x,
                        y,
                        feature_idx
                        * num_dims_feature : (feature_idx + 1)
                        * num_dims_feature,
                    ]

                    # Normalize vector for plotting
                    w_min, w_max = np.min(w_vector), np.max(w_vector)
                    w_range = w_max - w_min
                    if w_range > 1e-9:  # Avoid division by near-zero
                        w_vector = (w_vector - w_min) / w_range
                    else:
                        w_vector = np.zeros_like(w_vector)

                    # Create evenly spaced x-points
                    x_points = np.linspace(0, 1, len(w_vector))

                    # Shift to grid position
                    x_shifted = np.array(x_points) + x
                    # Since we're not inverting y-axis, we need to adjust the y calculation
                    # to maintain the same visual pattern but with correct orientation
                    y_shifted = y + np.array(w_vector)

                    ax.plot(x_shifted, y_shifted, color="red", linewidth=0.1)

            ax.grid(True, linestyle="--", alpha=0.3)

        # Remove unused subplots
        for j in range(len(batch_features), len(axes)):
            fig.delaxes(axes[j])

        batch_suffix = f"_batch{batch_idx}" if num_batches > 1 else ""
        fig.savefig(
            file_name.replace(".png", f"{batch_suffix}.png"),
            format="png",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close("all")
        gc.collect()  # Force garbage collection after each batch


def visualize_prediction_trajectories(
    som_wn_fn: Callable,
    som_n_coords: tuple,
    te_data: np.ndarray,
    te_target: np.ndarray,
    num_te_samples_same_run: int,
    file_name: str,
) -> None:
    """Visualize the trajectory of test data samples through the SOM."""
    min_x = np.min(som_n_coords[0])
    max_x = np.max(som_n_coords[0])
    min_y = np.min(som_n_coords[1])
    max_y = np.max(som_n_coords[1])
    num_rows = (max_x - min_x + 1).astype(np.int32)
    num_cols = (max_y - min_y + 1).astype(np.int32)

    # Offset to shift negative coords to start at 0
    x_offset, y_offset = -min_x, -min_y

    for run_idx in range(int(te_target.shape[0] // num_te_samples_same_run)):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_aspect("equal")

        # Draw grid points
        for i in range(num_rows):
            for j in range(num_cols):
                ax.plot(i + 0.5, j + 0.5, "o", color="lightgray")

        # Grid lines
        for i in range(num_rows + 1):
            plt.axvline(x=i, color="lightgray", linestyle="-", alpha=0.5, linewidth=0.5)
        for j in range(num_cols + 1):
            plt.axhline(y=j, color="lightgray", linestyle="-", alpha=0.5, linewidth=0.5)

        # Label grid with actual (x, y) values
        for i in range(num_rows):
            plt.text(
                i + 0.5, -0.5, str(i - x_offset), ha="center", va="top", fontsize=8
            )
        for j in range(num_cols):
            plt.text(
                -0.5, j + 0.5, str(j - y_offset), ha="right", va="center", fontsize=8
            )

        # Set limits with padding
        ax.set_xlim(-1, num_rows + 1)
        ax.set_ylim(-1, num_cols + 1)

        # Use row/column positions for ticks
        plt.xticks(
            [i + 0.5 for i in range(num_rows)], [i - x_offset for i in range(num_rows)]
        )
        plt.yticks(
            [j + 0.5 for j in range(num_cols)], [j - y_offset for j in range(num_cols)]
        )

        prev_grid = None

        for r_s_idx in range(num_te_samples_same_run):
            te_sample_idx = run_idx * num_te_samples_same_run + r_s_idx
            bmu = som_wn_fn(te_data[te_sample_idx])

            # Convert BMU to grid coordinates
            grid_i, grid_j = (
                bmu[0] + x_offset,
                bmu[1] + y_offset,
            )

            # Plot current point (i+0.5, j+0.5) for x,y with the new coordinate system
            if r_s_idx == 0:
                ax.plot(grid_i + 0.5, grid_j + 0.5, "o", color="blue")  # Start point
            elif r_s_idx == (num_te_samples_same_run - 1):
                ax.plot(grid_i + 0.5, grid_j + 0.5, "o", color="red")  # End point
            else:
                ax.plot(
                    grid_i + 0.5, grid_j + 0.5, "o", color="black"
                )  # Intermediate point

            # Draw line from previous point to current
            if prev_grid is not None:
                ax.annotate(
                    "",
                    xy=(grid_i + 0.5, grid_j + 0.5),
                    xytext=(prev_grid[0] + 0.5, prev_grid[1] + 0.5),  # previous i, j
                    arrowprops=dict(arrowstyle="->", color="black", lw=1),
                )

            prev_grid = (grid_i, grid_j)

        # Add title
        plt.title(
            f"Sample Trajectory: {te_target[run_idx * num_te_samples_same_run]}",
            fontsize=12,
        )

        # Hide regular axis
        ax.set_axis_off()

        fig.savefig(
            file_name.format(te_target[run_idx * num_te_samples_same_run]),
            format="png",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close("all")
        gc.collect()  # Force garbage collection after each batch


######
# Post-processing feature selection
######


def get_vector_subset(
    in_vec: np.ndarray, all_features: list, num_per_feature: int, subset_features: list
) -> np.ndarray:
    """
    Extract a subset of features from flattened vector(s) - fully vectorized.

    Args:
        in_vec: numpy array of shape (N*K,) or (J, N*K) containing feature values
               ordered as [F_0,0, F_0,1, ..., F_0,K-1, F_1,0, ..., F_N-1,K-1]
        all_features: List of all feature names in the order they appear in in_vec
        num_per_feature: Number of values per feature (K)
        subset_features: List of feature names to extract (subset of all_features)

    Returns:
        np.ndarray: Array containing only the values for the requested features
                   Shape: (len(subset_features)*K,) for 1D input
                   Shape: (J, len(subset_features)*K) for 2D input
    """
    # Convert feature names to indices (done once, not in loop)
    all_features_dict = {name: idx for idx, name in enumerate(all_features)}
    feature_indices = np.array([all_features_dict[name] for name in subset_features])

    # Generate all column indices we need to extract
    # For each feature, we need indices [start, start+1, ..., start+K-1]
    base_indices = (
        feature_indices[:, np.newaxis] * num_per_feature
    )  # Shape: (n_features, 1)
    offsets = np.arange(num_per_feature)[np.newaxis, :]  # Shape: (1, K)
    all_indices = (base_indices + offsets).flatten()  # Shape: (n_features * K,)

    # Extract columns using advanced indexing - works for both 1D and 2D
    if in_vec.ndim == 1:
        return in_vec[all_indices]
    else:
        return in_vec[:, all_indices]


def calculate_multiclass_alpha_t(
    bmu_positions, class_labels, distance_metric="euclidean"
):
    """
    Calculate αT (clustering quality index) for multiple classes on a SOM.

    This extends the two-class αT calculation to work with any number of classes,
    maintaining the same logic of comparing inter-class vs intra-class distances.

    Parameters:
    -----------
    bmu_positions : array-like, shape (n_samples, n_dimensions)
        Best Matching Unit (BMU) positions on the SOM for each input vector
    class_labels : array-like, shape (n_samples,)
        Class labels for each input vector
    distance_metric : str, default='euclidean'
        Distance metric to use ('euclidean', 'manhattan', etc.)

    Returns:
    --------
    alpha_t : float
        Clustering quality index. Higher values indicate better clustering.

    Notes:
    ------
    For multiple classes, we calculate:
    - d_intra: Average of all intra-class distances (within same class)
    - d_inter: Average of all inter-class distances (between different classes)
    - αT = d_inter / d_intra
    """

    bmu_positions = np.array(bmu_positions)
    class_labels = np.array(class_labels)

    # Get unique classes
    unique_classes = np.unique(class_labels)
    n_classes = len(unique_classes)

    if n_classes < 2:
        raise ValueError("Need at least 2 classes to calculate αT")

    def calculate_distance(pos1, pos2, metric="euclidean"):
        """Calculate distance between two positions"""
        if metric == "euclidean":
            # NOTE: this is how it's done in `Minisom._topographic_error_rectangular()` too
            return np.sqrt(np.sum((pos1 - pos2) ** 2))
        # elif metric == "manhattan":
        #     return np.sum(np.abs(pos1 - pos2))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    # Group BMU positions by class
    class_bmus = defaultdict(list)
    for i, label in enumerate(class_labels):
        class_bmus[label].append(bmu_positions[i])

    # Convert to numpy arrays
    for class_label in class_bmus:
        class_bmus[class_label] = np.array(class_bmus[class_label])

    # Calculate intra-class distances (d_A, d_B, etc. for each class)
    intra_distances = []
    intra_counts = []

    for class_label in unique_classes:
        class_positions = class_bmus[class_label]
        n_samples = len(class_positions)

        if n_samples > 1:
            # Calculate all pairwise distances within this class
            class_distances = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # Only include pairs that map to different BMUs
                    if not np.array_equal(class_positions[i], class_positions[j]):
                        dist = calculate_distance(
                            class_positions[i], class_positions[j], distance_metric
                        )
                        class_distances.append(dist)

            if class_distances:
                intra_distances.extend(class_distances)
                intra_counts.append(len(class_distances))

    # Calculate inter-class distances (between different classes)
    inter_distances = []

    for class1, class2 in combinations(unique_classes, 2):
        class1_positions = class_bmus[class1]
        class2_positions = class_bmus[class2]

        # Calculate distances between all pairs from different classes
        for pos1 in class1_positions:
            for pos2 in class2_positions:
                # Only include pairs that map to different BMUs
                if not np.array_equal(pos1, pos2):
                    dist = calculate_distance(pos1, pos2, distance_metric)
                    inter_distances.append(dist)

    # Calculate average distances
    if not intra_distances:
        raise ValueError(
            "No valid intra-class distances found. All samples from same classes map to identical BMUs."
        )

    if not inter_distances:
        raise ValueError("No inter-class distances found.")

    d_intra = np.mean(intra_distances)
    d_inter = np.mean(inter_distances)

    # Calculate αT
    if d_intra == 0:
        raise ValueError("Intra-class distance is zero. Cannot calculate αT.")

    alpha_t = d_inter / d_intra

    return alpha_t


def calculate_multiclass_alpha_n(bmu_positions, class_labels, som_shape):
    """
    Calculate αN (neighborhood purity index) for multiple classes on a SOM.

    This extends the two-class αN calculation to work with any number of classes,
    measuring the local neighborhood purity around each BMU.

    Parameters:
    -----------
    bmu_positions : array-like, shape (n_samples, 2)
        Best Matching Unit (BMU) positions on the SOM for each input vector.
        Should be integer coordinates (row, col) on the SOM grid.
    class_labels : array-like, shape (n_samples,)
        Class labels for each input vector
    som_shape : tuple (height, width)
        Shape of the SOM grid

    Returns:
    --------
    alpha_n : float
        Neighborhood purity index. Higher values indicate better clustering.
        Range is [0, 1] where 1 means perfect clustering.

    Notes:
    ------
    For each BMU:
    - If it contains only one class: calculate r_class in unit radius neighborhood
    - If it contains multiple classes (mixed BMU): calculate r_class with zero radius
    - r_class = N_class / (total vectors in neighborhood)
    - αN = sum of all r_class values / total number of input vectors
    """

    bmu_positions = np.array(bmu_positions, dtype=int)
    class_labels = np.array(class_labels)

    if len(bmu_positions) != len(class_labels):
        raise ValueError("BMU positions and class labels must have same length")

    # Get unique classes and total counts
    unique_classes = np.unique(class_labels)
    total_vectors = len(class_labels)

    # Group vectors by their BMU positions
    bmu_to_vectors = defaultdict(list)
    for i, (bmu_pos, class_label) in enumerate(zip(bmu_positions, class_labels)):
        bmu_key = tuple(bmu_pos)
        bmu_to_vectors[bmu_key].append(class_label)

    # Get unique BMU positions
    unique_bmus = list(bmu_to_vectors.keys())

    def get_neighbors_in_radius(center_pos, radius, som_shape):
        """Get all BMU positions within given radius from center position"""
        neighbors = []
        center_r, center_c = center_pos

        for r in range(
            max(0, center_r - radius), min(som_shape[0], center_r + radius + 1)
        ):
            for c in range(
                max(0, center_c - radius), min(som_shape[1], center_c + radius + 1)
            ):
                # Calculate Manhattan distance (can be changed to Euclidean if needed)
                if abs(r - center_r) + abs(c - center_c) <= radius:
                    neighbors.append((r, c))

        return neighbors

    def get_class_counts_in_neighborhood(center_pos, radius, som_shape, bmu_to_vectors):
        """Get count of each class in neighborhood around center position"""
        neighbors = get_neighbors_in_radius(center_pos, radius, som_shape)
        # log.info("The neighbours of {} are {}".format(center_pos, neighbors))
        class_counts = Counter()

        for neighbor_pos in neighbors:
            if neighbor_pos in bmu_to_vectors:
                neighbor_classes = bmu_to_vectors[neighbor_pos]
                for class_label in neighbor_classes:
                    class_counts[class_label] += 1

        return class_counts

    def is_mixed_bmu(bmu_classes):
        """Check if BMU contains vectors from multiple classes"""
        return len(set(bmu_classes)) > 1

    # Calculate r_class values for each BMU
    total_r_sum = 0.0

    for bmu_pos in unique_bmus:
        bmu_classes = bmu_to_vectors[bmu_pos]

        if is_mixed_bmu(bmu_classes):
            # Mixed BMU: use zero radius (only this BMU)
            class_counts = Counter(bmu_classes)
            total_in_neighborhood = sum(class_counts.values())

            # Add r_class for each class present in this mixed BMU
            for class_label in class_counts:
                r_class = class_counts[class_label] / total_in_neighborhood
                total_r_sum += r_class

        else:
            # Pure BMU: use unit radius (or larger if needed)
            dominant_class = bmu_classes[0]  # All vectors are same class
            radius = 1

            # Get class counts in neighborhood, expanding radius if no neighbors found
            while True:
                class_counts = get_class_counts_in_neighborhood(
                    bmu_pos, radius, som_shape, bmu_to_vectors
                )
                total_in_neighborhood = sum(class_counts.values())

                # If we found vectors in the neighborhood or radius is getting too large, break
                if total_in_neighborhood > len(bmu_classes) or radius > max(som_shape):
                    break
                radius += 1

            # Calculate r_class for the dominant class
            if total_in_neighborhood > 0:
                r_class = class_counts[dominant_class] / total_in_neighborhood
                total_r_sum += r_class

    # Calculate αN
    alpha_n = total_r_sum / total_vectors

    return alpha_n


def visualize_feature_selection_scores(
    alpha_t: list,
    alpha_n: list,
    x_labels: list,
    file_name: str,
) -> None:
    """
    Create a bar plot showing feature selection scores.

    Args:
        alpha_t: List of alpha_t scores for each feature
        alpha_n: List of alpha_n scores for each feature
        x_labels: List of feature names/labels for x-axis
        file_name: Output filename for the plot
    """
    # Convert to numpy arrays for easier manipulation
    alpha_t = np.array(alpha_t)
    alpha_n = np.array(alpha_n)
    alpha_combined = alpha_t + alpha_n

    # Set up the figure with extra space at the top for legend
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Create x positions for bars
    x_pos = np.arange(len(x_labels))
    bar_width = 0.25

    # Create the three sets of bars
    bars1 = ax.bar(
        x_pos - bar_width, alpha_t, bar_width, label="α_t", alpha=0.8, color="#2E86AB"
    )
    bars2 = ax.bar(x_pos, alpha_n, bar_width, label="α_n", alpha=0.8, color="#A23B72")
    bars3 = ax.bar(
        x_pos + bar_width,
        alpha_combined,
        bar_width,
        label="α_t + α_n",
        alpha=0.8,
        color="#F18F01",
    )

    # Customize the plot
    ax.set_xlabel("Features", fontsize=12)
    ax.set_ylabel("Score Values", fontsize=12)
    ax.set_title(
        "Feature Selection Scores Comparison", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Position legend in center-right, below title, outside plot area
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.75, 0.98),  # Position: center-right below title
        ncol=3,  # Horizontal layout
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )

    # Add value labels on top of bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(alpha_combined),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Add value labels on the bars
    add_value_labels(bars1, alpha_t)
    add_value_labels(bars2, alpha_n)
    add_value_labels(bars3, alpha_combined)

    # Adjust layout to accommodate legend positioning
    plt.subplots_adjust(top=0.9)  # Make room for legend below title
    plt.tight_layout()

    # Save with extra space at top for legend
    plt.savefig(
        file_name,
        format="png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
    )
    plt.close("all")  # Explicitly close all figures
    gc.collect()  # Force garbage collection


###############################################################################


def from_cate_to_bin(cate: int, num_classes: int) -> np.ndarray:
    """"""
    bin_vec_class = np.zeros((num_classes))
    bin_vec_class[cate - 1] = 1.0
    return bin_vec_class


def from_bin_to_cate(bin_vec_class: np.ndarray) -> int:
    """"""
    return np.argmax(bin_vec_class) + 1


def subsample_local_boids_state(behaviour_sims: dict, n_boids: int):
    """"""
    behaviour_shape = list(behaviour_sims.items())[0][1].shape
    if behaviour_shape[-2] > n_boids:
        subsampled_behaviour_sims = {}
        for k, v in behaviour_sims.items():
            subsampled_behaviour_feats = []
            for i in range(behaviour_shape[0]):
                # Different seed per i
                rng = np.random.default_rng(seed=i)
                idxs = rng.choice(behaviour_shape[-2], size=n_boids, replace=False)

                # Filter the last dimension
                subsampled_behaviour_feats.append(v[i, :, idxs].transpose(1, 0, 2))

            # Stack back into a 4D array
            subsampled_behaviour_sims[k] = np.stack(subsampled_behaviour_feats, axis=0)

        return subsampled_behaviour_sims
    else:
        return behaviour_sims


def data_processing_six_behaviours(
    range_generate_and_optimize,
    swarm_behaviour_feats_combo,
    curr_cfg,
    num_iters_scalar: int,
    num_sliding_w: int,
    smoothing: bool,
    supervised_data: bool,
) -> tuple:
    (
        rey_swarm_metrics_40b,
        rey_swarm_metrics_30b,
        rey_swarm_metrics_40u,
        vic_swarm_metrics_40b,
        vic_swarm_metrics_30b,
        vic_swarm_metrics_40u,
        aggreg_swarm_metrics_40b,
        aggreg_swarm_metrics_30b,
        aggreg_swarm_metrics_40u,
        disper_swarm_metrics_40b,
        disper_swarm_metrics_30b,
        disper_swarm_metrics_40u,
        balli_swarm_metrics_40b,
        balli_swarm_metrics_30b,
        balli_swarm_metrics_40u,
        brown_swarm_metrics_40b,
        brown_swarm_metrics_30b,
        brown_swarm_metrics_40u,
        rey_all_f_idxs,
    ) = process_all_behaviours_variants(
        swarm_behaviour_feats_combo,
        range_generate_and_optimize,
        get_npz_paths(swarm_behaviour_feats_combo),
        curr_cfg=curr_cfg,
    )

    log.info(f"{range_generate_and_optimize.stop - range_generate_and_optimize.start}")

    ######
    # Train and test with global features over N seconds
    ######
    tr_perc, sub_perc, num_behaviours, num_variants = 0.8, 0.06 * 1, 6, 3
    num_behaviours_variants = num_behaviours * num_variants
    tr_f_rand_idxs = np.random.choice(
        len(rey_all_f_idxs),
        size=int(len(rey_all_f_idxs) * tr_perc),
        replace=False,
    )
    te_f_rand_idxs = np.setdiff1d(rey_all_f_idxs, tr_f_rand_idxs)
    log.info(
        "tr_f_rand_idxs {} te_f_rand_idxs {}".format(tr_f_rand_idxs, te_f_rand_idxs)
    )
    colors = {
        "1": "#1f77b4",  # blue
        "2": "#9467bd",  # purple
        "3": "#ff7f0e",  # orange
        "4": "#2ca02c",  # green
        "5": "#d62728",  # red
        "6": "#8c564b",  # brown
    }

    smallest_n_boids = 30  # TODO remove hardcoding
    tr_data = np.concatenate(
        (
            process_swarm_metrics_independently(
                subsample_local_boids_state(rey_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(rey_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(rey_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(vic_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(vic_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(vic_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(aggreg_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(aggreg_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(aggreg_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(disper_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(disper_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(disper_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(balli_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(balli_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(balli_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(brown_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(brown_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(brown_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                tr_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    tr_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
        ),
        axis=0,
    )
    tr_labels = (
        [f"R_{rand_idx}_40b" for rand_idx in tr_f_rand_idxs]
        + [f"R_{rand_idx}_30b" for rand_idx in tr_f_rand_idxs]
        + [f"R_{rand_idx}_40u" for rand_idx in tr_f_rand_idxs]
        + [f"V_{rand_idx}_40b" for rand_idx in tr_f_rand_idxs]
        + [f"V_{rand_idx}_30b" for rand_idx in tr_f_rand_idxs]
        + [f"V_{rand_idx}_40u" for rand_idx in tr_f_rand_idxs]
        + [f"A_{rand_idx}_40b" for rand_idx in tr_f_rand_idxs]
        + [f"A_{rand_idx}_30b" for rand_idx in tr_f_rand_idxs]
        + [f"A_{rand_idx}_40u" for rand_idx in tr_f_rand_idxs]
        + [f"D_{rand_idx}_40b" for rand_idx in tr_f_rand_idxs]
        + [f"D_{rand_idx}_30b" for rand_idx in tr_f_rand_idxs]
        + [f"D_{rand_idx}_40u" for rand_idx in tr_f_rand_idxs]
        + [f"BA_{rand_idx}_40b" for rand_idx in tr_f_rand_idxs]
        + [f"BA_{rand_idx}_30b" for rand_idx in tr_f_rand_idxs]
        + [f"BA_{rand_idx}_40u" for rand_idx in tr_f_rand_idxs]
        + [f"BR_{rand_idx}_40b" for rand_idx in tr_f_rand_idxs]
        + [f"BR_{rand_idx}_30b" for rand_idx in tr_f_rand_idxs]
        + [f"BR_{rand_idx}_40u" for rand_idx in tr_f_rand_idxs]
    )
    tr_target = np.repeat(
        tr_labels,
        (tr_data.shape[0] // (num_behaviours_variants * tr_f_rand_idxs.shape[0])),
    )
    tr_target_num = np.repeat(
        [i + 1 for i in range(num_behaviours)],
        (tr_data.shape[0] // (num_behaviours)),
    )
    tr_total_samples = tr_data.shape[0]
    tr_total_sub_samples = int(tr_total_samples * sub_perc)
    tr_idxs_sub = np.zeros(tr_total_sub_samples, dtype=int)
    for i in range(num_behaviours_variants):
        # Handle remainder distribution
        samples_per_behaviour = tr_total_samples // num_behaviours_variants
        start_original = i * samples_per_behaviour
        end_original = (
            (i + 1) * samples_per_behaviour
            if i < num_behaviours_variants - 1
            else tr_total_samples
        )
        sub_samples_per_behaviour = tr_total_sub_samples // num_behaviours_variants
        start_sub = i * sub_samples_per_behaviour
        end_sub = (
            (i + 1) * sub_samples_per_behaviour
            if i < num_behaviours_variants - 1
            else tr_total_sub_samples
        )
        tr_idxs_sub[start_sub:end_sub] = np.random.randint(
            start_original,
            end_original,
            end_sub - start_sub,
        )
    if supervised_data:
        tr_data = np.concatenate(
            (
                tr_data,
                np.array(
                    [
                        from_cate_to_bin(tr_data_lbl, num_behaviours)
                        for tr_data_lbl in tr_target_num.reshape(-1, 1)
                    ]
                ),
            ),
            axis=-1,
        )  # Adding target label idxs to data

    te_data = np.concatenate(
        (
            process_swarm_metrics_independently(
                subsample_local_boids_state(rey_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(rey_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(rey_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(vic_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(vic_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(vic_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(aggreg_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(aggreg_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(aggreg_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(disper_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(disper_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(disper_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(balli_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(balli_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(balli_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(brown_swarm_metrics_40b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(brown_swarm_metrics_30b, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
            process_swarm_metrics_independently(
                subsample_local_boids_state(brown_swarm_metrics_40u, smallest_n_boids),
                swarm_behaviour_feats_combo.names,
                te_f_rand_idxs,
                num_sliding_w,
                num_overlap=0,
                applying_smoothing=smoothing,
                smoothing_level=0.1,
            ).reshape(
                int(
                    te_f_rand_idxs.shape[0]
                    * (
                        range_generate_and_optimize.stop
                        - range_generate_and_optimize.start
                    )
                    / num_sliding_w
                ),
                -1,
            ),
        ),
        axis=0,
    )
    te_labels = (
        [f"R_{rand_idx}_40b" for rand_idx in te_f_rand_idxs]
        + [f"R_{rand_idx}_30b" for rand_idx in te_f_rand_idxs]
        + [f"R_{rand_idx}_40u" for rand_idx in te_f_rand_idxs]
        + [f"V_{rand_idx}_40b" for rand_idx in te_f_rand_idxs]
        + [f"V_{rand_idx}_30b" for rand_idx in te_f_rand_idxs]
        + [f"V_{rand_idx}_40u" for rand_idx in te_f_rand_idxs]
        + [f"A_{rand_idx}_40b" for rand_idx in te_f_rand_idxs]
        + [f"A_{rand_idx}_30b" for rand_idx in te_f_rand_idxs]
        + [f"A_{rand_idx}_40u" for rand_idx in te_f_rand_idxs]
        + [f"D_{rand_idx}_40b" for rand_idx in te_f_rand_idxs]
        + [f"D_{rand_idx}_30b" for rand_idx in te_f_rand_idxs]
        + [f"D_{rand_idx}_40u" for rand_idx in te_f_rand_idxs]
        + [f"BA_{rand_idx}_40b" for rand_idx in te_f_rand_idxs]
        + [f"BA_{rand_idx}_30b" for rand_idx in te_f_rand_idxs]
        + [f"BA_{rand_idx}_40u" for rand_idx in te_f_rand_idxs]
        + [f"BR_{rand_idx}_40b" for rand_idx in te_f_rand_idxs]
        + [f"BR_{rand_idx}_30b" for rand_idx in te_f_rand_idxs]
        + [f"BR_{rand_idx}_40u" for rand_idx in te_f_rand_idxs]
    )
    te_target = np.repeat(
        te_labels,
        (te_data.shape[0] // (num_behaviours_variants * te_f_rand_idxs.shape[0])),
    )
    te_target_num = np.repeat(
        [i + 1 for i in range(num_behaviours)],
        (te_data.shape[0] // (num_behaviours)),
    )
    te_total_samples = te_data.shape[0]
    te_total_sub_samples = int(te_total_samples * sub_perc)
    te_idxs_sub = np.zeros(te_total_sub_samples, dtype=int)
    for i in range(num_behaviours_variants):
        # Handle remainder distribution
        samples_per_behaviour = te_total_samples // num_behaviours_variants
        start_original = i * samples_per_behaviour
        end_original = (
            (i + 1) * samples_per_behaviour
            if i < num_behaviours_variants - 1
            else te_total_samples
        )
        sub_samples_per_behaviour = te_total_sub_samples // num_behaviours_variants
        start_sub = i * sub_samples_per_behaviour
        end_sub = (
            (i + 1) * sub_samples_per_behaviour
            if i < num_behaviours_variants - 1
            else te_total_sub_samples
        )
        te_idxs_sub[start_sub:end_sub] = np.random.randint(
            start_original,
            end_original,
            end_sub - start_sub,
        )
    if supervised_data:
        te_data = np.concatenate(
            (
                te_data,
                np.array(
                    [
                        from_cate_to_bin(te_data_lbl, num_behaviours)
                        for te_data_lbl in te_target_num.reshape(-1, 1)
                    ]
                ),
            ),
            axis=-1,
        )  # Adding target label idxs to data

    num_iters = int(tr_data.shape[0] * num_iters_scalar)

    # Randomizing training samples
    new_tr_indices = np.random.permutation(tr_data.shape[0])
    tr_data = tr_data[new_tr_indices]
    tr_target = tr_target[new_tr_indices]
    tr_target_num = tr_target_num[new_tr_indices]
    sorted_indices = np.argsort(new_tr_indices)
    tr_idxs_sub = sorted_indices[
        np.searchsorted(new_tr_indices[sorted_indices], tr_idxs_sub)
    ]

    return (
        tr_data,
        tr_target,
        tr_target_num,
        tr_idxs_sub,
        te_data,
        te_target,
        te_target_num,
        te_idxs_sub,
        {
            "1": "Reynolds",
            "2": "Vicsek",
            "3": "Aggregation",
            "4": "Dispersion",
            "5": "Ballistic Motion",
            "6": "Brownian Motion",
        },
        num_iters,
        num_behaviours,
        colors,
        te_f_rand_idxs.shape[0],
    )
