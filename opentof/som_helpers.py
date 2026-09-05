# Copyright 2026 Aerodyne Research, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# som_helpers.py
import numpy as np
import pandas as pd
from minisom import MiniSom
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
import os

from opentof.peak_fitting import (
    multi_overlap_peak_fit,
    peak_function_selector,
    unpack_fit_result, calculate_detection_threshold
)

from opentof.utils import ensure_dir, get_default_plot_dir

class RowMinMaxScaler:
    """
    Row-wise Min-Max Scaler.
    Scales each row of a 2D array to [0, 1].
    Also stores row-wise min and max for eventual inverse transformation.
    """

    def __init__(self):
        self.row_min_ = None
        self.row_max_ = None

    def fit(self, arr: np.ndarray):
        """Compute row-wise min and max for scaling."""
        self.row_min_ = arr.min(axis=1, keepdims=True)
        self.row_max_ = arr.max(axis=1, keepdims=True)
        return self

    def transform(self, arr: np.ndarray):
        """Scale rows of arr using stored min and max, return both results."""
        if self.row_min_ is None or self.row_max_ is None:
            raise ValueError("RowMinMaxScaler must be fitted before calling transform.")
        
        denominator = np.where(self.row_max_ == self.row_min_, 1, self.row_max_ - self.row_min_)
        scaled = (arr - self.row_min_) / denominator

        # Build scalers array (one entry per row, shape = (n_rows,))
        scalers = np.stack([self.row_min_.ravel(), self.row_max_.ravel()], axis=1)

        return scaled, scalers

    def fit_transform(self, arr: np.ndarray):
        """Convenience method: fit and transform in one step."""
        self.fit(arr)
        return self.transform(arr)

    @staticmethod
    def inverse_transform(arr_scaled: np.ndarray, scalers: np.ndarray):
        """
        Reconstruct original values for each row using provided scalers.
        
        Parameters
        ----------
        arr_scaled : np.ndarray
            Scaled array, shape (n_rows, n_cols).
        scalers : np.ndarray
            Row-wise (min, max) pairs, shape (n_rows, 2).
        """
        row_min = scalers[:, [0]]
        row_max = scalers[:, [1]]
        denominator = np.where(row_max == row_min, 1, row_max - row_min)
        return arr_scaled * denominator + row_min

    @staticmethod
    def inverse_node_weight(node_weight_scaled: np.ndarray, node_scalers: np.ndarray, aggregate: str = "mean"):
        """
        Reconstruct a SOM node's prototype vector in original units.
        
        Parameters
        ----------
        node_weight_scaled : np.ndarray
            Scaled SOM prototype vector, shape (n_features,).
        node_scalers : np.ndarray
            (min, max) scalers for all samples mapped to this node,
            shape (n_samples_in_node, 2).
        aggregate : str
            How to aggregate across samples. Options:
              - "mean" : average the inversions (default)
              - "median" : median across inversions
              - "none" : return all reconstructed samples, shape (n_samples_in_node, n_features).
        
        Returns
        -------
        np.ndarray
            Reconstructed prototype in original space.
        """
        # Tile the node weight so we can inverse-transform for each sample
        tiled = np.tile(node_weight_scaled, (len(node_scalers), 1))
        reconstructions = RowMinMaxScaler.inverse_transform(tiled, node_scalers)

        if aggregate == "mean":
            return reconstructions.mean(axis=0)
        elif aggregate == "median":
            return np.median(reconstructions, axis=0)
        elif aggregate == "none":
            return reconstructions
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")
        
def build_nm_som(nm,
                 input_samples,
                 # SOM parameters
                 som_size=(5,5), 
                 sigma=1, 
                 learning_rate=0.2,
                 nbh_function='gaussian',
                 decay_function='linear_decay_to_zero',
                 sigma_decay_function='asymptotic_decay', #'linear_decay_to_one'  # 'linear_decay_to_one'
                 init='random',  # 'random' # 'pca'
                 train='random', # 'random', 'batch'
                 max_iter=None,
                 topology='hexagonal',  # 'hexagonal' # 'rectangular'
                 activation_distance='euclidean',
                 verbose=False,
                 random_seed=42
                 ):
    """
    Builds a self-organizing map based on 'input_samples' formatted to be 
    compatible with MiniSom (one sample = one row).
    """
    print("-----")

    if max_iter is None:
        max_iter = len(input_samples)
    
    # Initialize the SOM
    som = MiniSom(x=som_size[0], y=som_size[1], input_len=input_samples.shape[1],
                sigma=sigma, learning_rate=learning_rate,
                neighborhood_function=nbh_function,
                topology=topology,
                decay_function=decay_function,
                sigma_decay_function=sigma_decay_function,
                activation_distance=activation_distance,
                random_seed=random_seed)
    
    if init == 'pca':
        print("Initializing SOM with weights calculated from principle components...")
        som.pca_weights_init(input_samples)
    else:
        print("Initializing SOM with random weights...")
        som.random_weights_init(input_samples)

    # Begin the Learning Curve visualization code
    print(f"Begin [nm: {nm}] SOM Training...")
    start = time.time()

    if train == 'batch':
        print("Training SOM with sequentially ordered samples...")
        som.train_batch(input_samples, max_iter, verbose=verbose)
    else:
        print("Training SOM with randomly ordered samples...")
        som.train_random(input_samples, max_iter, verbose=verbose)
    
    # Print a message to the user that the som generation process has completed
    print(f"End [nm: {nm}] SOM Training!")

    # Stop the LC plotting timer
    end = time.time()
    length = end - start
    print(f"[nm: {nm}] SOM training took: {np.round(length, 3)} seconds!")

    print("-----")
    return som

def get_neighbors(x, y, umat, topology="rectangular"):
    """
    Get neighbor coordinates for either rectangular or hexagonal topology.
    Includes the center (x, y).
    """
    xdim, ydim = umat.shape
    neighbors = [(x, y)]  # always include self

    if topology == "rectangular":
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    elif topology == "hexagonal":
        # Odd-r offset hex coordinates (MiniSom convention)
        if y % 2 == 0:  # even row
            deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1)]
        else:  # odd row
            deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1)]
    else:
        raise ValueError("topology must be 'rectangular' or 'hexagonal'")

    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < xdim and 0 <= ny < ydim:
            neighbors.append((nx, ny))

    return neighbors


def find_internal_node_iter(x, y, umat, topology="rectangular"):
    """
    Iterative version of Hamel & Brown's algorithm.
    Traverses the U-Matrix until it reaches a local minimum.
    Returns final grid indices (cx, cy).
    """
    while True:
        neighbors = get_neighbors(x, y, umat, topology=topology)
        minx, miny = min(neighbors, key=lambda t: umat[t[0], t[1]])

        if (minx == x) and (miny == y):
            return minx, miny  # reached local minimum
        else:
            x, y = minx, miny


def annotate_internal_nodes(all_node_data, umat, xx_scaled, yy_scaled, topology="rectangular"):
    """
    Annotates all_node_data_scaled with internal node positions
    in both grid and Euclidean coordinates.
    """
    for (x, y), node_data in all_node_data.items():
        cx, cy = find_internal_node_iter(x, y, umat, topology=topology)

        # add to node dictionary
        node_data["internal_node_grid"] = (cx, cy)
        node_data["internal_node_xx"] = xx_scaled[cx, cy]
        node_data["internal_node_yy"] = yy_scaled[cx, cy]

    return all_node_data

def assign_cluster_labels(all_node_data):
    """
    Assigns cluster labels based on shared internal_node_grid.
    Adds a 'cluster_label' entry to each node in all_node_data_scaled.
    """
    # Map each unique internal_node_grid to a cluster ID
    internal_node_to_label = {}
    cluster_counter = 0

    for node_key, node_data in all_node_data.items():
        internal_node = node_data["internal_node_grid"]

        if internal_node not in internal_node_to_label:
            internal_node_to_label[internal_node] = cluster_counter
            cluster_counter += 1

        node_data["cluster_label"] = internal_node_to_label[internal_node]

    return all_node_data

def organize_som_data(som, input_samples, variable_names, variable_scalers, variable_length, topology='hexagonal'):
    """
    Helper function to organize a few SOM parameters into a Python dictionary
    for easy access.

    Will dynamically create the dictionary for any sized SOM.
    Node parameters within the dictionary can be accessed via (i,j) node indicies.

    Parameters
    ----------
    som : MiniSom
        A pre-trained MiniSom
    input_samples : np.array
        Pre-Scaled SOM input samples (Vectorized representation of the input variables)
    variable_names : list ['intensity', 'neg2d_intensity', 'mass_axis', 'tof_axis']
        A list of variable names contained within the vectorized input sample array.
    variable_scalers : list [int_scalers, neg2d_scalers, ma_scalers, tof_scalers]
        A list of RowMinMaxScalar scalars constructed for each input variable/input sample
    variable_length : int
        The length of one variable within the vectorized representation of the input samples
    topology: str (optional)
        String of the SOM topology, topologies within MiniSom are supported
        so far this is only 'retangular' and 'hexagonal'

    Returns
    -------
    nm_node_data : nested dict
        Keys are (i, j) node indices for the primary dict. 
                nested keys:
                    'activation_response' : The activation response value for that node
                    'sum_u_matrix' : The sum distance matrix value for that node using total distance from a node to its neighbors 
                    'log_sum_u_matrix' : log scaled sum_u_matrix
                    'mean_u_matrix' : The mean distance matrix value for that node using average distance from a node to its neighbors
                    'xx' : x-axis coordinate of the node defined in euclidean space  
                    'yy' : y-axis coordinate of the node defined in euclidean space  
                    'node_samples' : (filtered) MS index aligned sample index values
                        these can be used to figure out which samples in the input data were matched to which node
                    'quanitzation_error' : quanitzation error of the SOM
                    'topographic_error' : topographic error of the SOM
                    'distortion_measure' : distortion measure of the SOM

    """
    # Get weights from the SOM
    weights = som.get_weights()  # shape=(grid_x, grid_y, features)

    # Build mapping of sample indices -> node coordinates
    samples_in_nodes = {(i, j): [] for i in range(weights.shape[0]) for j in range(weights.shape[1])}
    for idx, sample in enumerate(input_samples):
        winning_node = som.winner(sample)
        samples_in_nodes[winning_node].append(idx)

    # SOM diagnostic maps
    ar = som.activation_response(input_samples)
    sum_u_matrix = som.distance_map(scaling='sum')
    log_scaled_sum = np.log1p(sum_u_matrix)
    mean_u_matrix = som.distance_map(scaling='mean')
    xx, yy = som.get_euclidean_coordinates()

    som_err_metrics = {}
    try:
        # Quantization Error
        # Average distance between each sample and its BMU
        som_err_metrics['quantization_error'] = som.quantization_error(input_samples)
        
        # Topographic Error
        # Measure of topology preservation (should be near 0 for good maps)
        som_err_metrics['topographic_error'] = som.topographic_error(input_samples)
        
        # Distortion Measure
        # Measures the smoothness of the map weights relative to the data distribution
        som_err_metrics['distortion_measure'] = som.distortion_measure(input_samples)
        
    except Exception as e:
        # Handle cases where min_som methods might fail (e.g., small map size)
        print(f"Warning: Failed to calculate all SOM metrics. Error: {e}")
        som_err_metrics = {
            'quantization_error': np.nan, 
            'topographic_error': np.nan, 
            'distortion_measure': np.nan
        }

    # Dictionary to store node-wise data
    nm_node_data = {}

    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            this_node_data = weights[x, y, :]  # flattened prototype vector
            this_node_data_dict = {}

            # === Split SOM prototype vector into variable sections ===
            for i, var_name in enumerate(variable_names):
                start_scaled = i * variable_length
                end_scaled = (i + 1) * variable_length

                variable_data = this_node_data[start_scaled:end_scaled]
                this_node_data_dict[var_name + "_scaled"] = variable_data

                # Get scalers for samples mapped to this node
                node_samples = samples_in_nodes[(x, y)]
                if len(node_samples) > 0:
                    if var_name == "intensity":
                        node_scalers = variable_scalers[0][node_samples]
                    elif var_name == "neg2d_intensity":
                        node_scalers = variable_scalers[1][node_samples]
                    elif var_name == "mass_axes":
                        node_scalers = variable_scalers[2][node_samples]
                    elif var_name == "tof_axes":
                        node_scalers = variable_scalers[3][node_samples]

                    # Reconstruct SOM prototype in original space
                    variable_data_inverse = RowMinMaxScaler.inverse_node_weight(
                        variable_data, node_scalers, aggregate="median"
                    )
                    this_node_data_dict[var_name + "_inverse"] = variable_data_inverse
                else:
                    this_node_data_dict[var_name + "_inverse"] = None

            # === Add SOM diagnostic info ===
            this_node_data_dict['activation_response'] = ar[x, y]
            this_node_data_dict['sum_u_matrix'] = sum_u_matrix[x, y]
            this_node_data_dict['log_sum_u_matrix'] = log_scaled_sum[x, y]
            this_node_data_dict['mean_u_matrix'] = mean_u_matrix[x, y]
            this_node_data_dict['xx'] = xx[x, y]
            this_node_data_dict['yy'] = yy[x, y]

            # Add sample indices mapped to this node
            this_node_data_dict['node_samples'] = np.array(samples_in_nodes[(x, y)])

            nm_node_data[(x, y)] = this_node_data_dict
    
    nm_node_data = annotate_internal_nodes(nm_node_data, sum_u_matrix, xx, yy, topology=topology)
    nm_node_data = assign_cluster_labels(nm_node_data)

    return nm_node_data, som_err_metrics

def process_som_weight_peaks(nm_node_data, nm,
                     peak_width_function, peak_type='pseudo_voigt', 
                     custom_shape=None,
                     output_dir=None,
                     plot_subdir=None,
                     spacing_scale=0.85,
                     plot_flag=True, 
                     noise_level=None,
                     noise_std_mult=10.0,
                     verbose=False,
                     ):
    """
    Fits peaks contained within the weight vector of each node and optionally
    creates a plot of SOM weights

    Parameters
    ----------
    all_node_data : dict
        Dictionary containing SOM node data output from organize_som_data 
    nm : int 
        Nominal Mass identifier of the given SOM).
    peak_width : callable (pre-defined peak width)
        Function that returns peak width given nm.
    peak_type : str
        Type of peak function ('pseudo_voigt', 'custom', etc.)
    custom_shape : custom peak shape: custom_shape
        custom_shape output from ot.peakfitting.peak_shape function (if defined)
    output_dir : str
        Relative directory to save plots.
    spacing_scale : float
        Scaling factor for subplot spacing.
    plot_flag : bool
        If True, generate/save the SOM weight plot. If False, skip plotting but still fit peaks.
    verbose : bool
        If True, print extra debug commands to the console

    Returns
    -------
    all_som_peaks : dict
        Keys are (i, j) node indices, values are DataFrames of fitted peaks.
    """
    if output_dir is None:
        output_dir = get_default_plot_dir()
    ensure_dir(output_dir)

    if plot_flag:
        base_fig_size = 24
        fig = plt.figure(figsize=(base_fig_size, base_fig_size), dpi=600)

        # --- Extract SOM node positions ---
        xx_vals = np.array([v['xx'] for v in nm_node_data.values()])
        yy_vals = np.array([v['yy'] for v in nm_node_data.values()])

        # Side length assuming n by n SOM
        n_side = int(len(xx_vals) ** 0.5)

        # Set text size to tested values given the some size
        if n_side == 2:
            vertical_spacing_scale = 0.66
            textsize = 25
        elif n_side == 3:
            vertical_spacing_scale = 0.78
            textsize = 20
        elif n_side == 4:
            vertical_spacing_scale = 0.85
            textsize = 15
        elif n_side == 5:
            vertical_spacing_scale = 0.93
            textsize = 12
        elif n_side == 6:
            vertical_spacing_scale = 0.97
            textsize = 10
        elif n_side == 7:
            vertical_spacing_scale = 1
            textsize = 8
        elif n_side == 8:
            vertical_spacing_scale = 1
            textsize = 6
        elif n_side == 9:
            vertical_spacing_scale = 1
            textsize = 4
        else:
            vertical_spacing_scale = 1
            textsize = 4


        # Normalize to [0,1] figure coordinates
        x_norm = (xx_vals - xx_vals.min()) / (xx_vals.max() - xx_vals.min())
        y_norm = (yy_vals - yy_vals.min()) / (yy_vals.max() - yy_vals.min())
        y_norm *= vertical_spacing_scale

        # --- Compute adaptive subplot size ---
        coords = np.column_stack([x_norm, y_norm])
        dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)

        subplot_size = spacing_scale * min_dist

        # --- Colormap normalization ---
        sum_u_values = [v['sum_u_matrix'] for v in nm_node_data.values()]
        u_norm = Normalize(vmin=np.min(sum_u_values), vmax=np.max(sum_u_values))
        u_cmap = plt.cm.viridis

        cluster_labels = [v['cluster_label'] for v in nm_node_data.values()]
        c_norm = Normalize(vmin=min(cluster_labels), vmax=max(cluster_labels))
        c_cmap = plt.cm.nipy_spectral

    # --- Iterate over SOM nodes ---
    for idx, ((i, j), node) in enumerate(nm_node_data.items()):
        # Select peak function
        peak_func = peak_function_selector(peak_type, custom_shape=custom_shape)
        node_peaks = []

        node_mass_axis = node['mass_axes_inverse']
        node_tof_axis = node['tof_axes_inverse']
        node_intensity_axis = node['intensity_inverse']
        node_neg2d_intensity_axis = node['neg2d_intensity_inverse']

        if verbose:
            print(f"node_mass_axis is: {node_mass_axis}")
            print(f"node_tof_axis is: {node_tof_axis}")
            print(f"node_intensity_axis is: {node_intensity_axis}")
            print(f"node_neg2d_intensity_axis is: {node_neg2d_intensity_axis}")

        # --- Fit peaks (always done, even if plot_flag=False) ---
        min_separation = peak_width_function(nm)

        node_fr = multi_overlap_peak_fit(
            node_mass_axis,
            node_intensity_axis,
            node_tof_axis,
            peak_type=peak_type,
            custom_shape=custom_shape,
            min_separation_fwhm=min_separation,
            noise_level=noise_level,
            noise_std_mult=noise_std_mult,
            smooth_factor=-1,
            verbose=verbose,
        )

        unpack_fit_result(node_fr, node_peaks, f"({i},{j})", nm)
        node_all_peaks = pd.DataFrame(node_peaks)

        nm_node_data[(i, j)]['weight_peaks'] = node_all_peaks

        # --- Plotting (only if enabled) ---
        if plot_flag:
            # Node position in normalized coordinates
            x = x_norm[idx]
            y = y_norm[idx]

            # Create subplot anchored to (x,y)
            ax = fig.add_axes([
                x - subplot_size/2,
                y - subplot_size/2,
                subplot_size,
                subplot_size
            ])

            # Background color from U-matrix
            umatrix_val_at_node = node['sum_u_matrix']
            background_color = list(u_cmap(u_norm(umatrix_val_at_node)))
            background_color[-1] = 0.4
            activation_val = int(node['activation_response'])

            # Raw intensities
            ax.plot(node_mass_axis, node_neg2d_intensity_axis,
                    label='Node Neg2D Intensity', linestyle="dashdot", color='dimgray', zorder=1)
            ax.plot(node_mass_axis, node_intensity_axis,
                    label='Node Intensity', color='black', zorder=1)

            # Plot fitted peaks
            reconstructed = np.zeros_like(node_mass_axis)
            for _, peak in node_all_peaks.iterrows():
                if peak_type == "pseudo_voigt":
                    peak_curve = peak_func(
                        node_mass_axis,
                        peak["amplitude"],
                        peak["peak_center_mass"],
                        peak["fwhm_mass"],
                        0.5
                    )
                else:
                    peak_curve = peak_func(
                        node_mass_axis,
                        peak["amplitude"],
                        peak["peak_center_mass"],
                        peak["fwhm_mass"]
                    )
                ax.plot(node_mass_axis, peak_curve,
                        label=f'{np.round(peak["peak_center_mass"], 5)}', zorder=3)
                ax.axvline(peak["peak_center_mass"], color='gray',
                           linestyle="--", linewidth=0.5)
                reconstructed += peak_curve

            # Background coloring
            ax.set_facecolor(background_color)

            ax.tick_params(labelsize=textsize)

            # Activation response annotation
            ax.text(
                0.8, 0.9, f"AR={activation_val}",
                transform=ax.transAxes,
                fontsize=textsize, color="black",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="none", alpha=0)
            )

            # Transparent legend
            leg = ax.legend(fontsize=textsize)
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_alpha(0)

            # Cluster rectangle
            cluster_color = c_cmap(c_norm(node['cluster_label']))
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor=cluster_color,
                facecolor='none',
                linewidth=10,
                zorder=4
            )
            ax.add_patch(rect)

    # --- Save figure if plotting ---
    if plot_flag:
        plot_name = f"{plot_subdir}_{nm}_SOM_weights.png" if plot_subdir else f"{nm}_SOM_weights.png"
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.clf()
        plt.close()

    return nm_node_data