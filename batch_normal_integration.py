# simpler but batched version of https://github.com/xucao-42/bilateral_normal_integration/blob/main/bilateral_normal_integration_cupy.py 
import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg
import time
import pytorch_pfn_extras as ppe

# this is required if used in tandem with pytorch
ppe.cuda.use_torch_mempool_in_cupy()


def b_move_left(mask): return cp.pad(mask,((0, 0), (0,0),(0,1)),'constant',constant_values=0)[..., :,1:]  # Shift the input mask array to the left by 1, filling the right edge with zeros.
def b_move_right(mask): return cp.pad(mask,((0, 0),(0,0),(1,0)),'constant',constant_values=0)[..., :,:-1]  # Shift the input mask array to the right by 1, filling the left edge with zeros.
def b_move_top(mask): return cp.pad(mask,((0, 0),(0,1),(0,0)),'constant',constant_values=0)[..., 1:,:]  # Shift the input mask array up by 1, filling the bottom edge with zeros.
def b_move_bottom(mask): return cp.pad(mask,((0, 0),(1,0),(0,0)),'constant',constant_values=0)[..., :-1,:]  # Shift the input mask array down by 1, filling the top edge with zeros.
def b_move_top_left(mask): return cp.pad(mask,((0, 0),(0,1),(0,1)),'constant',constant_values=0)[..., 1:,1:]  # Shift the input mask array up and to the left by 1, filling the bottom and right edges with zeros.
def b_move_top_right(mask): return cp.pad(mask,((0, 0),(0,1),(1,0)),'constant',constant_values=0)[..., 1:,:-1]  # Shift the input mask array up and to the right by 1, filling the bottom and left edges with zeros.
def b_move_bottom_left(mask): return cp.pad(mask,((0, 0),(1,0),(0,1)),'constant',constant_values=0)[..., :-1,1:]  # Shift the input mask array down and to the left by 1, filling the top and right edges with zeros.
def b_move_bottom_right(mask): return cp.pad(mask,((0, 0),(1,0),(1,0)),'constant',constant_values=0)[..., :-1,:-1]  # Shift the input mask array down and to the right by 1, filling the top and left edges with zeros.



def generate_dx_dy_batch(masks, nz_horizontal, nz_vertical, step_size=1):
    # pixel coordinates
    # ^ vertical positive
    # |
    # |
    # |
    # o ---> horizontal positive

    num_pixels_per_mask = cp.sum(masks, axis=(1, 2))
    num_pixels = cp.sum(num_pixels_per_mask)


    # Generate an integer index array with the same shape as the mask.
    pixel_idx = cp.zeros_like(masks, dtype=int)
    # Assign a unique integer index to each True value in the mask.
    pixel_idx[masks] = cp.arange(num_pixels)

    # Create boolean masks representing the presence of neighboring pixels in each direction.
    has_left_mask = cp.logical_and(b_move_right(masks), masks)
    has_right_mask = cp.logical_and(b_move_left(masks), masks)
    has_bottom_mask = cp.logical_and(b_move_top(masks), masks)
    has_top_mask = cp.logical_and(b_move_bottom(masks), masks)

    # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
    # repeat nz_horizontal and nz_vertical to match the shape of masks

    nz_left = nz_horizontal[has_left_mask[masks]]
    nz_right = nz_horizontal[has_right_mask[masks]]
    nz_top = nz_vertical[has_top_mask[masks]]
    nz_bottom = nz_vertical[has_bottom_mask[masks]]


    # Create sparse matrices representing the partial derivatives for each direction.
    # top/bottom/left/right = vertical positive/vertical negative/horizontal negative/horizontal positive
    # The matrices are constructed using the extracted normal components and pixel indices.
    data = cp.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[b_move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[masks].astype(int) * 2)])

    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixels, num_pixels))

    data = cp.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[b_move_right(has_right_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[masks].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixels, num_pixels))

    data = cp.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[b_move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[masks].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixels, num_pixels))

    data = cp.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = cp.stack((pixel_idx[b_move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[masks].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixels, num_pixels))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg



def convert_K_tobini_coordarates(K):
    K_new = K.copy()
    K_new[1, 2] = K[0, 2]
    K_new[0, 2] = K[1, 2]
    K_new[1, 1] = K[0, 0]
    K_new[0, 0] = K[1, 1]
    return K_new


def normal_integration_batch_cupy(normal_map,
                                  normal_mask,
                                  K=None,
                                  step_size=1,
                                  cg_max_iter=5000,
                                  cg_tol=1e-3):
    """
    This function performs the bilateral normal integration algorithm, as described in the paper.
    It takes as input the normal map, normal mask, and several optional parameters to control the integration process.

    :param normal_map: A normal map, which is an image where each pixel's color encodes the corresponding 3D surface normal.
    :param normal_mask: A binary mask that indicates the region of interest in the normal_map to be integrated.
    :param k: A parameter that controls the stiffness of the surface.
              The smaller the k value, the smoother the surface appears (fewer discontinuities).
              If set as 0, a smooth surface is obtained (No discontinuities), and the iteration should end at step 2 since the surface will not change with iterations.

    :param depth_map: (Optional) An initial depth map to guide the integration process.
    :param depth_mask: (Optional) A binary mask that indicates the valid depths in the depth_map.

    :param lambda1 (Optional): A regularization parameter that controls the influence of the depth_map on the final result.
                               Required when depth map is input.
                               The larger the lambda1 is, the result more close to the initial depth map (fine details from the normal map are less reflected)

    :param K: (Optional) A 3x3 camera intrinsic matrix, used for perspective camera models. If not provided, the algorithm assumes an orthographic camera model.
    :param step_size: (Optional) The pixel size in the world coordinates. Default value is 1.
                                 Used only in the orthographic camera mdoel.
                                 Default value should be fine, unless you know the true value of the pixel size in the world coordinates.
                                 Do not adjust it in perspective camera model.

    :param max_iter: (Optional) The maximum number of iterations for the optimization process. Default value is 150.
                                If set as 1, a smooth surface is obtained (No discontinuities).
                                Default value should be fine.
    :param tol:  (Optional) The tolerance for the relative change in energy to determine the convergence of the optimization process. Default value is 1e-4.
                            The larger, the iteration stops faster, but the discontinuity preservation quality might be worse. (fewer discontinuities)
                            Default value should be fine.

    :param cg_max_iter: (Optional) The maximum number of iterations for the Conjugate Gradient solver. Default value is 5000.
                                   Default value should be fine.
    :param cg_tol: (Optional) The tolerance for the Conjugate Gradient solver. Default value is 1e-3.
                              Default value should be fine.

    :return: depth_map: The resulting depth map after the bilateral normal integration process.
             surface: A pyvista PolyData mesh representing the 3D surface reconstructed from the depth map.
             wu_map: A 2D image that represents the horizontal smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             wv_map: A 2D image that represents the vertical smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             energy_list: A list of energy values during the optimization process.
    """
    # To avoid confusion, we list the coordinate systems in this code as follows
    #
    # pixel coordinates         camera coordinates     normal coordinates (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                       (o is the optical center;
    #                        xy-plane is parallel to the image plane;
    #                        +z is the viewing direction.)
    #
    # The input normal map should be defined in the normal coordinates.
    # The camera matrix K should be defined in the camera coordinates.
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]
    # I forgot why I chose the awkward coordinate system after getting used to opencv convention :(
    # but I won't touch the working code.
    verbose = False
    dtype = cp.float64

    # TODO check if it works w multiple cuda devices
    torch_device = normal_map.device
    
    # print(normal_map.__cuda_array_interface__)
    normal_map = cp.from_dlpack(to_dlpack(normal_map))
    normal_mask = cp.from_dlpack(to_dlpack(normal_mask)).astype(bool)

    

    wu_top = b_move_top(normal_mask)[normal_mask]
    wu_bottom = b_move_bottom(normal_mask)[normal_mask]
    wv_left = b_move_left(normal_mask)[normal_mask]
    wv_right = b_move_right(normal_mask)[normal_mask]


    if K is not None:  # perspective
        K = convert_K_tobini_coordarates(K)

    num_masks = normal_mask.shape[0]
    num_normals = cp.sum(normal_mask).item()
    projection = "orthographic" if K is None else "perspective"
    if verbose:
        print(f"Running bilateral normal integration with in the {projection} case. \n"
            f"The number of normal vectors is {num_normals}.")
    # transfer the normal map from the normal coordinates to the camera coordinates
    normal_map = normal_map[cp.newaxis, ...].repeat(num_masks, axis=0)
    normal_map = normal_map[normal_mask, :]


    nx = normal_map[:, 1]
    ny = - normal_map[:, 0]
    nz = - normal_map[:, 2]
    del normal_map

    if K is not None:  # perspective
        N, H, W = normal_mask.shape

        yy, xx = cp.meshgrid(cp.arange(W), cp.arange(H))
        xx = cp.flip(xx, axis=0)
        
        xx = xx[cp.newaxis, ...].repeat(num_masks, axis=0)
        yy = yy[cp.newaxis, ...].repeat(num_masks, axis=0)

        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        uu = uu.astype(dtype)
        vv = vv.astype(dtype)


        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:  # orthographic
        nz_u = nz.copy()
        nz_v = nz.copy()

    pixel_idx = cp.zeros_like(normal_mask, dtype=int)
    pixel_idx[normal_mask] = cp.arange(num_normals)
    pixel_idx_flat = cp.arange(num_normals)
    pixel_idx_flat_indptr = cp.arange(num_normals + 1)


    has_left_mask = cp.logical_and(b_move_right(normal_mask), normal_mask)
    has_left_mask_left = b_move_left(has_left_mask)
    has_right_mask = cp.logical_and(b_move_left(normal_mask), normal_mask)
    has_right_mask_right = b_move_right(has_right_mask)
    has_bottom_mask = cp.logical_and(b_move_top(normal_mask), normal_mask)
    has_bottom_mask_bottom = b_move_bottom(has_bottom_mask)
    has_top_mask = cp.logical_and(b_move_bottom(normal_mask), normal_mask)
    has_top_mask_top = b_move_top(has_top_mask)

    has_left_mask_flat = has_left_mask[normal_mask]
    has_right_mask_flat = has_right_mask[normal_mask]
    has_bottom_mask_flat = has_bottom_mask[normal_mask]
    has_top_mask_flat = has_top_mask[normal_mask]

    has_left_mask_left_flat = has_left_mask_left[normal_mask]
    has_right_mask_right_flat = has_right_mask_right[normal_mask]
    has_bottom_mask_bottom_flat = has_bottom_mask_bottom[normal_mask]
    has_top_mask_top_flat = has_top_mask_top[normal_mask]

    nz_left_square = nz_v[has_left_mask_flat] ** 2
    nz_right_square = nz_v[has_right_mask_flat] ** 2
    nz_top_square = nz_u[has_top_mask_flat] ** 2
    nz_bottom_square = nz_u[has_bottom_mask_flat] ** 2

    pixel_idx_left_center = pixel_idx[has_left_mask]
    pixel_idx_right_right = pixel_idx[has_right_mask_right]
    pixel_idx_top_center = pixel_idx[has_top_mask]
    pixel_idx_bottom_bottom = pixel_idx[has_bottom_mask_bottom]

    data_term_top = wu_top[has_top_mask_flat] * nz_top_square
    data_term_bottom = wu_bottom[has_bottom_mask_flat] * nz_bottom_square
    data_term_left = wv_right[has_left_mask_flat] * nz_left_square
    data_term_right = wv_left[has_right_mask_flat] * nz_right_square


    pixel_idx_left_left_indptr_new = cp.cumsum(has_left_mask_left_flat)
    pixel_idx_right_center_indptr_new = cp.cumsum(has_right_mask_flat)
    pixel_idx_top_top_indptr_new = cp.cumsum(has_top_mask_top_flat)
    pixel_idx_bottom_center_indptr_new = cp.cumsum(has_bottom_mask_flat)


    pixel_idx_left_left_indptr_new = cp.concatenate([cp.array([0]), pixel_idx_left_left_indptr_new])
    pixel_idx_right_center_indptr_new = cp.concatenate([cp.array([0]), pixel_idx_right_center_indptr_new ])
    pixel_idx_top_top_indptr_new = cp.concatenate([cp.array([0]),  pixel_idx_top_top_indptr_new])
    pixel_idx_bottom_center_indptr_new = cp.concatenate([cp.array([0]), pixel_idx_bottom_center_indptr_new])

    A3, A4, A1, A2 = generate_dx_dy_batch(normal_mask, 
                                          nz_horizontal=nz_v, nz_vertical=nz_u,
                                          step_size=step_size)


    b_vec = A1.T @ (wu_top * (-nx)) \
            + A2.T @ (wu_bottom * (-nx)) \
            + A3.T @ (wv_right * (-ny)) \
            + A4.T @ ((wv_left) * (-ny))


    tic = time.time()

    diagonal_data_term = cp.zeros(num_normals)
    diagonal_data_term[has_left_mask_flat] += data_term_left
    diagonal_data_term[has_left_mask_left_flat] += data_term_left
    diagonal_data_term[has_right_mask_flat] += data_term_right
    diagonal_data_term[has_right_mask_right_flat] += data_term_right
    diagonal_data_term[has_top_mask_flat] += data_term_top
    diagonal_data_term[has_top_mask_top_flat] += data_term_top
    diagonal_data_term[has_bottom_mask_flat] += data_term_bottom
    diagonal_data_term[has_bottom_mask_bottom_flat] += data_term_bottom
    
    A_mat_d = csr_matrix((diagonal_data_term, pixel_idx_flat, pixel_idx_flat_indptr),
                            shape=(num_normals, num_normals), dtype=dtype)

    A_mat_left_odu = csr_matrix((-data_term_left, pixel_idx_left_center, pixel_idx_left_left_indptr_new),
                                shape=(num_normals, num_normals), dtype=dtype)
    A_mat_right_odu = csr_matrix((-data_term_right, pixel_idx_right_right, pixel_idx_right_center_indptr_new),
                                    shape=(num_normals, num_normals), dtype=dtype)
    A_mat_top_odu = csr_matrix((-data_term_top, pixel_idx_top_center, pixel_idx_top_top_indptr_new),
                                shape=(num_normals, num_normals), dtype=dtype)
    A_mat_bottom_odu = csr_matrix((-data_term_bottom, pixel_idx_bottom_bottom, pixel_idx_bottom_center_indptr_new),
                                    shape=(num_normals, num_normals), dtype=dtype)

    A_mat_odu = A_mat_top_odu + A_mat_bottom_odu + A_mat_right_odu + A_mat_left_odu
    A_mat = A_mat_d + A_mat_odu + A_mat_odu.T  # diagnol + upper triangle + lower triangle matrix
    ################################################################################################################

    D = csr_matrix((1 / cp.clip(diagonal_data_term, 1e-5, None), pixel_idx_flat, pixel_idx_flat_indptr),
                    shape=(num_normals, num_normals), dtype=dtype)  # Jacobi preconditioner.

    toc = time.time()
    if verbose:
        print(f"Preconditioning time: {toc - tic:.3f} sec")

    z = cp.zeros(num_normals, dtype=dtype)

    z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)

    del A_mat, b_vec

    toc = time.time()

    if verbose:
        print(f"Total time: {toc - tic:.3f} sec")
    if K is not None:
        z = cp.exp(z)
    z = torch.as_tensor(z, device=torch_device)
    # make a copy of that to make sure it doesn't get destroyed
    z = z.float().clone() 
    #cp.get_default_memory_pool().free_all_blocks()
    return z