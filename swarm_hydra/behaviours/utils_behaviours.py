import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import seaborn as sns
import logging
import os
import imageio.v2 as imageio  # Use imageio.v2 for compatibility
from PIL import Image
import einops
import h5py
import numpy as np
from collections import deque

from swarm_hydra.metrics.spatial_metrics import distance_fn


# A logger for this file
log = logging.getLogger(__name__)


sns.set_style(style="white")
dark_color = [56 / 256] * 3
light_color = [213 / 256] * 3
axis_color = "white"


def handling_box_limits(total_force: np.ndarray, pos: np.ndarray, params) -> np.ndarray:
    magnitude = params["max_dist"] * 100
    margin = params["box_size"] // 100
    min_border, max_border = params["box_init"], params["box_init"] + params["box_size"]

    total_force[:, 0] += (
        magnitude / (pos[:, 0] - min_border) * (pos[:, 0] < (min_border + margin))
    )
    total_force[:, 0] -= (
        magnitude / (max_border - pos[:, 0]) * (pos[:, 0] > (max_border - margin))
    )

    if total_force.shape[1] == 2:
        total_force[:, 1] += (
            magnitude / (pos[:, 1] - min_border) * (pos[:, 1] < (min_border + margin))
        )
        total_force[:, 1] -= (
            magnitude / (max_border - pos[:, 1]) * (pos[:, 1] > (max_border - margin))
        )

    # angle = 45
    # theta = np.deg2rad(np.random.randint(-angle, angle))
    # theta_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # rot_mask = (\
    #     (pos[:, 0] < (min_border + margin)) | (pos[:, 0] > (max_border - margin)) |
    #     (pos[:, 1] < (min_border + margin)) | (pos[:, 1] > (max_border - margin))
    # )
    # total_force[rot_mask] = np.squeeze(total_force[rot_mask] @ theta_mat)

    # if total_force.shape[1] == 3:
    #     total_force[:, 2] += (magnitude / (pos[:, 2] - min_border) * (pos[:, 2] < (min_border + margin)))
    #     total_force[:, 2] -= (magnitude / (max_border - pos[:, 2]) * (pos[:, 2] > (max_border - margin)))

    return total_force


def export_to_hdf5(data_dict, out_file_name, max_s) -> bool:
    """Converts the input dict to a HDF5."""
    try:
        if not os.path.exists(out_file_name):
            with h5py.File(out_file_name, "w") as f:
                for key, value in data_dict.items():
                    data = np.array(value)

                    # Tiling data to avoid resizing later ...
                    if len(data.shape) == 0:
                        data = data.reshape(1, -1)
                        data = np.tile(data, (max_s, data.shape[0]))
                    elif len(data.shape) == 1:
                        data = np.tile(data, (max_s, 1))
                    elif len(data.shape) == 2:
                        data = np.tile(data, (max_s, 1, 1))
                    elif len(data.shape) == 3:
                        data = np.tile(data, (max_s, 1, 1, 1))
                    else:
                        log.info("{WARNING] Undefined if for len(data.shape)")
                        return False

                    f.create_dataset(
                        key, data=data, maxshape=data.shape, compression="gzip"
                    )

            log.debug(f"Created HDF5 file '{out_file_name}' with initial data.")

        with h5py.File(out_file_name, "a") as f:
            curr_idx = data_dict["iter"]
            for key, value in data_dict.items():
                data = np.array(value)

                if key not in f:
                    log.info("Key is not available in the HDF5 file.")
                    return False

                dset = f[key]
                dset[curr_idx] = data

        log.debug(f"Appended data to '{out_file_name}'.")

    except Exception as exception:
        log.info(f"{exception}")
        return False

    return True


def render_boids_gif(
    box_size, boid_states, output_filename, agent_trace_steps, fps, radius, v_const
):
    """
    Renders a sequence of boid states with optional trails, visibility circles, and heading arrows, then saves it as a GIF.

    Args:
        box_size (float): The size of the simulation box.
        boid_states (list): A list of boid state objects, where each state has:
            - .X: Nx2 array of positions.
            - .X_dot: Nx2 array of velocity vectors.
        output_filename (str): Path to save the GIF file.
        agent_trace_steps (int): Number of steps to trace in grey.
        fps (int): Frames per second for the video.
        radius (float): Radius of the dashed circles around each boid.
        v_const: Constant velocity magnitude for the boids.
    """
    if agent_trace_steps < 1:
        raise ValueError("agent_trace_steps must be at least 1.")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    # ax.set_title(
    #     "Agents Collective Behavior with Trails, Perception Circles, and Headings"
    # )
    ax.grid(True)

    boid_trails = deque(maxlen=agent_trace_steps)  # Store past positions
    trail_scatters = [
        ax.scatter([], [], s=9, color="grey", alpha=0.5)
        for _ in range(agent_trace_steps)
    ]
    boid_scatter = ax.scatter([], [], s=10, color="blue", label="Boids")

    '''
    # Dashed circles for perception range
    circles = [
        plt.Circle(
            (0, 0), radius, color="lightgrey", linestyle="dashed", fill=False, alpha=0.5
        )
       
        for _ in range(len(boid_states[0].X))
    ]
    for circle in circles:
        ax.add_patch(circle)
    '''

    # Arrows for heading direction
    arrow_patches = [
        ax.arrow(0, 0, 0, 0, color="green", head_width=0.5, head_length=0.5, alpha=0.8)
        for _ in range(len(boid_states[0].X))
    ]

    def update(frame_idx):
        boids = boid_states[frame_idx]
        positions = boids.X
        velocities = boids.X_dot

        boid_trails.append(positions.copy())

        # Update trails in fading grey
        for step, (scatter, past_positions) in enumerate(
            zip(trail_scatters, reversed(boid_trails))
        ):
            scatter.set_offsets(past_positions)
            scatter.set_alpha((step + 1) / agent_trace_steps)

        # Update boid positions in blue
        boid_scatter.set_offsets(positions)

        '''
        # Update dashed perception circles **only on even frames**
        if frame_idx % 10 == 0:
            for circle, pos in zip(circles, positions):
                circle.set_center(pos)
        else:
            for circle in circles:
                circle.set_center((-1000, -1000))  # Move off-screen to "hide"
        '''

        # Update heading arrows
        for arrow, pos, vel in zip(arrow_patches, positions, velocities):
            dx, dy = vel
            arrow.set_data(x=pos[0], y=pos[1], dx=dx, dy=dy)

        return trail_scatters + [boid_scatter] + arrow_patches  # + circles

    ani = animation.FuncAnimation(
        fig, update, frames=len(boid_states), interval=1000 / fps, blit=True
    )

    # Save as GIF using Pillow writer
    ani.save(output_filename, writer="pillow", fps=fps)

    log.info(f"GIF saved as: {output_filename}")
    plt.close(fig)  # Free memory


def plot_boids_sakana(state, params):
    """
    Adapted the `render_state()` in
        https://github.com/SakanaAI/asal/blob/main/substrates/boids.py
    """

    def get_transformation_mats(x, v):
        (x, y), (u, v) = x, v
        global2local = np.array(
            [[u, v, -u * x - v * y], [-v, u, v * x - u * y], [0, 0, 1]]
        )

        local2global = np.array([[u, -v, x], [v, u, y], [0, 0, 1]])

        return global2local, local2global

    x, v = state.X, state.X_dot  # n_boids, 2
    x = (x - 0.0) / (params["img_size"] - 0.0)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)  # Normalize velocity

    # Compute transformation matrices in homogenous coordinates
    transformation_matrices = np.array(
        [get_transformation_mats(xi, vi) for xi, vi in zip(x, v)]
    )
    global2local = transformation_matrices[:, 0, :, :]  # (n_boids, 3, 3)
    local2global = transformation_matrices[:, 1, :, :]  # (n_boids, 3, 3)

    # Define local triangle shape(centered at actual CoM)
    raw_triangle = (
        np.array([[0, 1.0], [0, -1.0], [3, 0.0]]) * params["bird_render_size"]
    )
    # Compute centroid (center of mass)
    centroid = np.mean(raw_triangle, axis=0)
    # Recenter the triangle so that centroid is at (0,0)
    local_triangle_coords = raw_triangle - centroid
    # local_triangle_coords = np.array([[0, 1.], [0, -1.], [3, 0.]]) * params['bird_render_size']
    local_triangle_coords = np.concatenate(
        [local_triangle_coords, np.ones((3, 1))], axis=-1
    )  # (3, 3)
    local_triangle_coords = local_triangle_coords[:, :, None]  # (3, 3, 1)

    # Apply transformations to get global triangle coordinates
    global_triangle_coords = np.einsum(
        "nij,ajk->naik", local2global, local_triangle_coords
    )
    global_triangle_coords = global_triangle_coords[
        :, :, :2, 0
    ]  # Extract 2D coordinates

    # Note: until this step `np.mean(global_triangle_coords, axis=1) - x` is minimal

    # Not necessary since the periodic boundary conditions where handled correctly
    #   when the data was generated and this code only leads to blurry images ...
    # # Ensure triangles stay within the [0,1] bounds using periodic boundary conditions
    # global_triangle_coords = np.mod(global_triangle_coords, 1.0)
    # # Sanity check for boid positions
    # assert np.all(global_triangle_coords >= 0) and np.all(global_triangle_coords <= 1), \
    #         "Boids are going out of bounds!"

    # Initialize white image
    img = np.ones((params["img_size"], params["img_size"], 3))

    # Generate meshgrid for image space
    x_space = np.linspace(0, params["space_size"], params["img_size"])
    y_space = np.linspace(0, params["space_size"], params["img_size"])
    x_grid, y_grid = np.meshgrid(x_space, y_space, indexing="ij")

    def render_triangle(img, triangle, color=[0.0, 0.0, 0.0]):
        """Renders a triangle onto the image using barycentric coordinates"""
        v0 = triangle[2] - triangle[0]
        v1 = triangle[1] - triangle[0]
        v2 = np.stack([x_grid, y_grid], axis=-1) - triangle[0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        # Prevent division by zero
        denom = d00 * d11 - d01 * d01
        denom = np.where(denom == 0, 1e-6, denom)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        # Compute mask using sigmoid function
        mask = (
            (1 / (1 + np.exp(-params["bird_render_sharpness"] * u)))
            * (1 / (1 + np.exp(-params["bird_render_sharpness"] * v)))
            * (1 / (1 + np.exp(-params["bird_render_sharpness"] * w)))
        )

        img = mask[..., None] * np.array(color) + (1 - mask[..., None]) * img
        return img

    # Render all triangles
    for triangle in global_triangle_coords:
        img = render_triangle(img, triangle)

    # Render the first boid in red if enabled
    if len(params["red_boids"]) > 0:
        for i_d in params["red_boids"]:
            if 0 < i_d < len(global_triangle_coords):
                img = render_triangle(
                    img, global_triangle_coords[i_d], color=[1.0, 0.0, 0.0]
                )

    #
    # TODO understand why this image rotation is needed ...?!
    #
    import scipy

    # Rotate the final image by -90 degrees
    img = scipy.ndimage.rotate(
        img, angle=90, reshape=True, mode="nearest"
    )  # `mode='nearest'` to avoid interpolation artifacts

    return img


def render_boids_sakana(states, params, fm=None, time_sampling="final"):
    """
    Adapted from https://github.com/SakanaAI/asal/blob/main/asal.ipynb
        and https://github.com/SakanaAI/asal/blob/main/rollout.py

    Notes:
    - when time_sampling is 'final', the function returns data for the T timestep.
    - when time_sampling is 'video', the function returns data for the [0, ..., T-1] timesteps.
    - when time_sampling is (K, False), the function returns data for the [0, T//K, T//K * 2, ..., T - T//K] timesteps.
    - when time_sampling is (K, True), the function returns data for the [T//K, T//K * 2, ..., T] timesteps.
    """
    embed_img_fn = (lambda img: None) if fm is None else fm.embed_img

    def get_image(state_final, return_state=False):
        img = plot_boids_sakana(state_final, params=params)  # Render image
        z = embed_img_fn(img)  # Embed if function is provided
        return dict(rgb=img, z=z, state=(state_final if return_state else None))

    if time_sampling == "final":  # return only the final state
        # Get the rendered image
        image_data = get_image(states[-1])
        img = image_data["rgb"]

        img = np.pad(
            img, ((2, 2), (2, 2), (0, 0)), constant_values=0.5
        )  # Padding with gray (0.5)

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        # ax.grid(True)
        plt.xticks([]), plt.yticks([])
        plt.title("Agents Collective Behavior with Headings", fontsize=20)
        plt.savefig(params["out_name"], bbox_inches="tight")
        log.info(f"Image saved as: {params['out_name']}")
    elif time_sampling == "video":  # return the entire rollout
        imgs = []
        for i in range(len(states)):
            # Get the rendered image
            image_data = get_image(states[i])
            img = image_data["rgb"]
            img = np.pad(
                img, ((2, 2), (2, 2), (0, 0)), constant_values=0.5
            )  # Padding with gray (0.5)
            imgs.append(img)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks([]), ax.set_yticks([])  # Remove axis labels
        # ax.grid(True)
        # Display first frame
        img_display = ax.imshow(imgs[0])

        # Function to update frame
        def update(frame):
            img_display.set_array(imgs[frame])
            return [img_display]

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(states), blit=True)
        # Save as video using FFMPEG
        ani.save(params["out_name"], writer="ffmpeg", fps=params["fps"])
        plt.title("Agents Collective Behavior with Headings", fontsize=20)
        log.info(f"Video saved at {params['out_name']}")
    elif isinstance(time_sampling, int) or isinstance(
        time_sampling, tuple
    ):  # return the rollout at K sampled intervals
        K, chunk_ends = (
            time_sampling
            if isinstance(time_sampling, tuple)
            else (time_sampling, False)
        )
        chunk_steps = len(states) // K

        if chunk_ends:
            idx_sample = np.arange(chunk_steps - 1, len(states), chunk_steps)
        else:
            idx_sample = np.arange(0, len(states), chunk_steps)

        states_to_render = [states[idx] for idx in idx_sample]
        imgs = []
        for state in states_to_render:
            # Get the rendered image
            img = get_image(state)
            imgs.append(img["rgb"])

        # Arrange imgs in rows and columns
        img = np.pad(
            imgs, ((0, 0), (2, 2), (2, 2), (0, 0)), constant_values=0.5
        )  # Padding with gray (0.5)
        img = einops.rearrange(img, "T H W D -> H (T W) D")
        img = np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=0.5)

        # Plot the image
        plt.figure(figsize=(20, 4))
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])  # Remove axis labels
        # plt.title("Agents' Collective Behavior", fontsize=8)
        plt.tight_layout()
        plt.savefig(params["out_name"], bbox_inches="tight")
        log.info(f"Image with state idxs {idx_sample} saved as: {params['out_name']}")
    else:
        raise ValueError(f"time_sampling {time_sampling} not recognized")


def render_network_gif(box_size, boid_states, output_f, fps, radius, walls_b):
    """
    Renders a sequence of boid states as a dynamic network graph and saves the animation as a GIF.

    Args:
        box_size (float): The size of the simulation box.
        boid_states (list): A list of boid state objects, where each state has:
            - .X: Nx2 array of positions.
        output_f (str): Path to save the GIF file.
        fps (int): Frames per second for the animation.
        radius (float): Connection radius for the network graph.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Dynamic Network of Boid Agents")
    ax.grid(True)

    # Scatter plot for boid positions
    boid_scatter = ax.scatter([], [], s=10, color="blue", label="Boids")

    # Initialize LineCollection for network edges
    edge_collection = LineCollection([], color="grey", alpha=0.5, linewidth=0.8)
    ax.add_collection(edge_collection)

    def update(frame_idx):
        boids = boid_states[frame_idx]
        positions = boids.X

        # Update boid positions
        boid_scatter.set_offsets(positions)

        # Compute distance matrix & adjacency (connection) list
        dists = distance_fn(boids.X[:, np.newaxis], boids.X, walls_b, box_size)
        edges = np.unique(
            np.sort(np.argwhere(dists * ((dists < radius) & (dists != 0.0))), axis=1),
            axis=0,
        ).tolist()

        # Convert edges into line segments
        edge_segments = [[positions[i], positions[j]] for i, j in edges]

        # Update the LineCollection with new edges
        edge_collection.set_segments(edge_segments)

        return boid_scatter, edge_collection

    ani = animation.FuncAnimation(
        fig, update, frames=len(boid_states), interval=1000 / fps, blit=True
    )

    # Save as GIF using Pillow writer
    ani.save(output_f, writer="pillow", fps=fps)

    log.info(f"GIF saved as: {output_f}")
    plt.close(fig)  # Free memory


def visualizing_boids_force(
    positions: np.ndarray, forces_tp: tuple, output_f: str, title: str
) -> None:

    _, ax = plt.subplots(figsize=(7, 5))
    for fx, fy, co, sc, lb in forces_tp:
        ax.quiver(
            positions[:, 0], positions[:, 1], fx, fy, color=co, scale=sc, label=lb
        )
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        color="green",
        marker="o",
        s=0.1,
        label="Agents",
    )
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()
    plt.tight_layout()
    plt.savefig(output_f, bbox_inches="tight")


def stitch_images(img_path1: str, img_path2: str, new_img_pth: str) -> None:
    """"""
    # Load images as NumPy arrays
    img1 = imageio.imread(img_path1)
    img2 = imageio.imread(img_path2)
    # Stack images vertically (axis=0 for row-wise stacking)
    stitched_img = np.vstack((img1, img2))
    # Save or display the result
    Image.fromarray(stitched_img).save(new_img_pth)
