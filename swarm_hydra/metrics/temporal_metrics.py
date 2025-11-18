# import hydra
# import numpy as np
# import logging
# from functools import partial

# import fastdtw
# import scipy

# from swarm_hydra.metrics.spatial_metrics import distance_fn

# # A logger for this file
# log = logging.getLogger(__name__)


'''
def compute_dtw_swarm_configurations(demo_sequence, imit_sequence) -> tuple[np.array, np.array]:
    """
        Leveraging an opensource library to compute the fast data
        time warping metric.
    """
    dist, path = fastdtw.fastdtw(
        demo_sequence,
        imit_sequence,
        dist=partial(distance_fn, walls_b=False, box_size=500.)  # scipy.spatial.distance.euclidean  # lambda a, b: sum((a - b) ** 2) ** 0.5
    )
    return np.array(dist), np.array(path)


# Old code retrieved from (compute_metrics.py)
# FDTW
for idx in range(rey_pos.shape[1]):
    # Same agent based on relative position over time
    sorted_boids_pos1_over_time = rey_pos[:, idx, :]
    sorted_boids_pos2_over_time = vic_pos[:, idx, :]
    fdtw_distances1, fdtw_path1 = compute_dtw_swarm_configurations(
        sorted_boids_pos1_over_time,
        sorted_boids_pos2_over_time
    )
    print(f"The FDTW between agents w/ idx={idx} in  {rey_subf} and {vic_subf} has the mean euclidean distance of {np.mean(fdtw_distances1)}.")
    ## Adapted visualization from
    #       https://medium.com/trusted-data-science-haleon/fastdtw-in-action-optimizing-manufacturing-operations-c07f3cc5023c
    #       and https://ros-developer.com/2017/04/26/dynamic-time-warping-with-python/
    #       and https://www.d3view.com/compare-curves-with-dynamic-time-warping/
    # Create a grid of subplots
    fig = plt.figure(figsize=(16, 8))
    gs = mpl.gridspec.GridSpec(3, 3, width_ratios=[1, 1, 2], height_ratios=[2, 1, 1])
    # ---- Y Series Y Coordinate vs Time (left) ----
    ax1 = plt.subplot(gs[0])
    ax1.plot(sorted_boids_pos2_over_time[:, 1], np.arange(sorted_boids_pos2_over_time.shape[0]), color='blue')
    ax1.set_title('Y Series (Y Coordinate) vs Time')
    ax1.set_ylabel('Time')
    plt.grid()
    # ---- Y Series X Coordinate vs Time (right) ----
    ax0 = plt.subplot(gs[1])
    ax0.plot(sorted_boids_pos2_over_time[:, 0], np.arange(sorted_boids_pos2_over_time.shape[0]), color='black')
    ax0.set_title('Y Series (X Coordinate) vs Time')
    plt.grid()
    # ---- FastDTW Warping Path ----
    ax2 = plt.subplot(gs[2])
    index_a, _ =zip(*fdtw_path1)
    ax2.scatter(sorted_boids_pos1_over_time[:, 0], sorted_boids_pos1_over_time[:, 1], color='r', label=f"{rey_subf}")
    ax2.scatter(sorted_boids_pos2_over_time[:, 0], sorted_boids_pos2_over_time[:, 1], color='b', label=f"{vic_subf}")
    for i1 in index_a:
        x1=sorted_boids_pos1_over_time[i1][0]
        y1=sorted_boids_pos1_over_time[i1][1]
        x2=sorted_boids_pos2_over_time[i1][0]
        y2=sorted_boids_pos2_over_time[i1][1]
        ax2.plot([x1, x2], [y1, y2], color='#30EA03', linewidth=0.5)
    ax2.set_title('FastDTW Warping Path')
    ax2.legend()
    plt.grid()
    # ---- X Series X Coordinate vs Time ----
    ax3 = plt.subplot(gs[5])
    ax3.plot(np.arange(sorted_boids_pos1_over_time.shape[0]), sorted_boids_pos1_over_time[:, 0], color='magenta')
    ax3.set_title('X Series (X Coordinate) vs Time')
    ax3.set_xlabel('Time')
    plt.grid()
    # ---- X Series Y Coordinate vs Time ----
    ax4 = plt.subplot(gs[8])
    ax4.plot(np.arange(sorted_boids_pos1_over_time.shape[0]), sorted_boids_pos1_over_time[:, 1], color='cyan')
    ax4.set_title('X Series (Y Coordinate) vs Time')
    ax4.set_xlabel('Time')
    plt.grid()
    # Remove the unused middle subplot
    plt.delaxes(plt.subplot(gs[4]))
    plt.tight_layout()
    plt.show()
'''
# TODO connect these results to the pd.Dataframe comparison ...

# TODO
#   From above ...
#       Dynamic Time Warping (DTW) (interesting comparison to euclidean comparision matching \url{https://commons.wikimedia.org/wiki/File:Euclidean_vs_DTW.jpg} and \url{https://en.wikipedia.org/wiki/Dynamic_time_warping})
#       Fast DTW impl. paper \url{https://cs.fit.edu/~pkc/papers/tdm04.pdf}
#       Does it make sense to compare two simulations in velocities (directional vectors too) ?!
#       TODO How to apply this metric to swarm configurations over time? Is it only feasible with a compressed representation of the state (e.g. Classical and NN mean swarm embedding) ?!}
#   Consider impl Temporal part of k-tree pose metric proposal by R. Amit el al. ...
#   ...
#   Consider time series features generation using the tsfresh library https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
