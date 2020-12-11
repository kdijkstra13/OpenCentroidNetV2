import torch
import torch.nn
import torch.autograd
import torch.nn.modules.loss
import torch.nn.functional as functional
import numpy as np
from typing import Tuple, Dict, List

from skimage.feature import peak_local_max


def variance_filter(tensor: torch.tensor, structuring_element: torch.tensor) -> torch.tensor:
    element_sum = torch.sum(structuring_element)
    tensor = functional.pad(tensor, [1, 1, 1, 1])  # t, b, l, r
    mu = functional.conv2d(tensor, structuring_element) / element_sum
    mu2 = functional.conv2d(tensor ** 2, structuring_element) / element_sum
    mu_mu = mu ** 2
    sigma = mu2 - mu_mu
    return sigma


def binary_edge_detector(instance_ids: np.ndarray) -> np.array:
    assert len(instance_ids.shape) == 2
    filter_a = np.array([[[0., 0., 0.],
                          [0., 1., 1.],
                          [0., 1., 1.]]])
    filter_b = np.array([[[1., 1., 0.],
                          [1., 1., 0.],
                          [0., 0., 0.]]])
    filter_a = torch.from_numpy(filter_a[np.newaxis])
    filter_b = torch.from_numpy(filter_b[np.newaxis])
    img = torch.from_numpy(instance_ids[np.newaxis, np.newaxis])
    a = variance_filter(img, filter_a)
    b = variance_filter(img, filter_b)
    c = a + b
    c[c > 0] = 1
    c *= img
    return c.squeeze().cpu().numpy()


def vector_distance_point(y, x, image_coords: np.ndarray = None, shape=None) -> Tuple[np.ndarray, np.ndarray]:
    assert (image_coords is not None or shape is not None)
    if shape is not None:
        assert len(shape) == 2
    image_coords = image_coords if image_coords is not None else np.transpose(np.indices(shape), (1, 2, 0))
    vec = np.array([y, x]) - image_coords
    dist = vec ** 2
    dist = np.sum(dist, axis=2)
    dist = np.sqrt(dist)
    return dist, vec


def calc_border_and_centroid_coords(instance_ids: np.ndarray, border_ids: np.ndarray) -> \
        Dict[int, Tuple[np.ndarray, np.ndarray]]:
    # Get unique labels
    uniq = np.unique(instance_ids)
    uniq = uniq[uniq != 0]
    # Get object coordinates
    object_coords = {int(label): np.transpose(np.argwhere(instance_ids == label)) for label in uniq}
    # Get object centroids
    centroid_coords = {int(label): [np.mean(y), np.mean(x)] for label, (y, x) in object_coords.items()}
    # Get border coordinates
    coords = {label: [centroid_coords[label], np.transpose(np.argwhere(border_ids == label))]
              for label in centroid_coords.keys()}
    # Return centroid and border coords
    return coords


def calc_vector_map(coords: Dict[int, Tuple[np.ndarray, np.ndarray]], shape: Tuple[int, int]) -> \
        Tuple[np.ndarray, np.ndarray]:
    assert len(shape) == 2
    image_coords_planar = np.indices(shape)
    image_coords = np.transpose(image_coords_planar, (1, 2, 0))
    if len(coords) == 0:
        vec_ctr = np.zeros((2, *shape))
        return vec_ctr, vec_ctr

    # Start of centroids
    # Make dist cube [h, w, len(ctr)]
    dist_cube = np.empty([len(coords), *shape])
    # Make vec cube [h, w, len(ctr), yx]
    vec_cube = np.empty([len(coords), *shape, 2])
    # Calculate all distance vectors to all centroids
    for i, (instance_id, ((y, x), (_, _))) in enumerate(coords.items()):
        dist_cube[i], vec_cube[i] = vector_distance_point(y, x, image_coords)
    # Get the smallest centroid distance index [h, w]
    dist_ctr_labels = np.argmin(dist_cube, axis=0)
    # Get the smallest distance vectors [h, w, yx]
    vec_ctr = vec_cube[dist_ctr_labels, image_coords_planar[0], image_coords_planar[1]]

    # Start of borders
    vec_brd = np.zeros((*shape, 2), dtype=np.float)
    for i, (instance_id, (_, yx)) in enumerate(coords.items()):
        yx = np.transpose(yx)  # convert to (#, yx)
        # Get all coordinates of pixels closest to centroid i.
        image_coords = np.transpose(np.stack(np.where(dist_ctr_labels == i)))
        # Get distance to border pixels of object i for all image coordinates
        dists = [np.sqrt(np.sum((yx - coord) ** 2, axis=1)) for coord in image_coords]
        # Get index of minimum distance border
        min_vec = np.stack([yx[np.argmin(dist)] for dist in dists])
        # Make coordinates relative
        min_vec -= image_coords
        # Write relative coordinates to result image
        vec_brd[image_coords[:, 0], image_coords[:, 1]] = min_vec

    return np.transpose(vec_ctr, (2, 0, 1)), np.transpose(vec_brd, (2, 0, 1))


def calc_centroid_and_border_vectors(instance_ids: np.ndarray, max_dist=0) -> Tuple[np.ndarray, np.ndarray]:
    instance_ids = instance_ids.astype(np.float)
    border_ids = binary_edge_detector(instance_ids=instance_ids)
    coords = calc_border_and_centroid_coords(instance_ids=instance_ids, border_ids=border_ids)
    centroid_vectors, border_vectors = calc_vector_map(coords=coords, shape=instance_ids.shape)
    if max_dist != 0:
        active = np.sqrt(np.sum(border_vectors ** 2, axis=0)) > max_dist
        centroid_vectors[:, active] = 0
        border_vectors[:, active] = 0
    return centroid_vectors, border_vectors


def calc_vote_image(centroid_vectors: np.array) -> np.ndarray:
    # Convert to relative to absolute voting vectors
    channels, height, width = centroid_vectors.shape
    indices = np.indices((height, width), dtype=centroid_vectors.dtype)
    vectors = np.round(centroid_vectors + indices).astype(np.int)

    # Prevent votes outside of the image
    logic = np.logical_and(
        np.logical_and(vectors[0] >= 0, vectors[1] >= 0),
        np.logical_and(vectors[0] < height, vectors[1] < width))
    coords = vectors[:, logic]

    # Cast votes
    votes = np.zeros([height, width])
    np.add.at(votes, (coords[0], coords[1]), 1)
    return np.expand_dims(votes, axis=0)


def calc_centroids(centroid_votes: np.array, centroid_threshold, nm_window) -> Tuple[np.ndarray, np.ndarray]:
    vote_image = centroid_votes[0]
    vote_mask = peak_local_max(image=vote_image, min_distance=nm_window // 2,
                               threshold_abs=centroid_threshold, indices=False)
    vote_image *= vote_mask
    coords = np.argwhere(vote_image > centroid_threshold)
    votes = vote_image[coords[:, 0], coords[:, 1]]
    return coords, votes


def calc_borders(centroid_list: np.ndarray, centroid_vectors: np.ndarray, border_vectors: np.ndarray,
                 border_threshold: int) -> Tuple[List[np.ndarray], np.ndarray]:
    channels, height, width = centroid_vectors.shape
    indices = np.indices((height, width), dtype=centroid_vectors.dtype)

    # Convert to absolute vectors
    image_ctr = (centroid_vectors + indices).astype(np.int)
    image_brd = (border_vectors + indices).astype(np.int)

    # Calculate border vectors
    all_border_coords = []
    border_votes = np.zeros([height, width])
    for centroid_coord in centroid_list:
        y, x = centroid_coord

        # Get all locations of contributing centroid_vectors
        contrib = np.logical_and(image_ctr[0] == y, image_ctr[1] == x)

        # Get border vector contribution
        border_coords = image_brd[:, contrib]

        # Filter border vectors outside of the image
        logic = np.logical_and(np.logical_and(border_coords[0] >= 0, border_coords[1] >= 0),
                               np.logical_and(border_coords[0] < height, border_coords[1] < width))
        border_coords = border_coords[:, logic]

        # Filter based on the votes
        if border_threshold > 0:
            vote_image = np.zeros((height, width))
            np.add.at(vote_image, (border_coords[0], border_coords[1]), 1)
            border_coords = np.argwhere(vote_image >= border_threshold)
            border_coords = np.transpose(border_coords)

            # Update the border votes
            border_votes = np.maximum(vote_image, border_votes)
        else:
            border_votes[border_coords[0], border_coords[1]] = 1

        all_border_coords.append(border_coords)

    return all_border_coords, border_votes
