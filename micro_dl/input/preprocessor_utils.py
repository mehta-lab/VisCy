import glob
import os
import numpy as np

from micro_dl.utils.image_utils import resize_image, resize_mask


def crop_image(input_image, tile_size, step_size, isotropic=False):
    """Creates 3D blocks from the image from given crop and overlap size.

    :param np.array input_image: input image in 3d
    :param list/tuple/np array tile_size: size of the blocks to be cropped
     from the image
    :param list/tuple/np array step_size: size of the window shift. In case of
     no overlap, the step size is tile_size. If overlap, step_size < tile_size
    :return: a list with tuples of cropped image id of the format
     xxmin-xmz_yymin-ymax_zzmin-zmax and cropped image
    """

    assert len(tile_size) == len(step_size)
    assert np.all(tile_size) > 0
    if mask_image:
        assert isinstance(mask_image, bool)

    size_x = input_image.shape[0]
    size_y = input_image.shape[1]

    n_dim = len(input_image.shape)
    if n_dim == 3:
        size_z = input_image.shape[2]

    if isotropic:
        isotropic_shape = [tile_size[0], ] * len(tile_size)
        isotropic_cond = list(tile_size) == isotropic_shape
    else:
        isotropic_cond = isotropic

    cropped_image_list = []
    for x in range(0, size_x - tile_size[0] + 1, step_size[0]):
        for y in range(0, size_y - tile_size[1] + 1, step_size[1]):
            img_id = 'x{}-{}_y{}-{}'.format(x, x + tile_size[0],
                                            y, y + tile_size[1])
            if n_dim == 3:
                for z in range(0, size_z - tile_size[2] + 1, step_size[2]):
                    img_id = '{}_z{}-{}'.format(img_id, z, z + tile_size[2])
                    cropped_img = input_image[x: x + tile_size[0],
                                              y: y + tile_size[1],
                                              z: z + tile_size[2]]

                    if isotropic_cond:
                        cropped_img = resize_image(cropped_img,
                                                   isotropic_shape)
                        # tiled_img = np.rollaxis(tiled_img, 2, 0)
                    cropped_image_list.append((img_id, cropped_img))
            else:
                cropped_img = input_image[x: x + tile_size[0],
                                          y: y + tile_size[1]]
                cropped_image_list.append((img_id, cropped_img))
    return cropped_image_list


def get_crop_indices(mask_ip_dir, min_fraction, mask_op_dir, tile_size,
                     step_size, isotropic=False):
    """Get crop indices and save mask for tiles with roi_vf >= min_fraction

    Crops and saves mask to mask_output_dir

    :param str mask_ip_dir: directory containing individual sample masks
    :param float min_fraction: minimum volume fraction of the ROI to retain
    :param str mask_op_dir: directory to save the cropped masks
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :return: dict with fname as keys and list of indices as values
    """

    if isotropic:
        isotropic_shape = [tile_size[0], ] * len(tile_size)
        isotropic_cond = list(tile_size) == isotropic_shape
    else:
        isotropic_cond = isotropic
    masks_in_dir = glob.glob(os.path.join(mask_ip_dir, '*.npy'))
    index_dict = {}

    for mask_idx, mask_fname in enumerate(masks_in_dir):
        _, fname = os.path.split(mask_fname)
        mask = np.load(mask_fname)
        size_x = mask.shape[0]
        size_y = mask.shape[1]
        n_dim = len(mask.shape)
        if n_dim == 3:
            size_z = mask.shape[2]

        sample_num = int(fname.split('_')[1][1:])
        cur_index_list = []
        for x in range(0, size_x - tile_size[0] + 1, step_size[0]):
            for y in range(0, size_y - tile_size[1] + 1, step_size[1]):
                img_id = 'n{}_x{}-{}_y{}-{}'.format(
                    sample_num, x, x + tile_size[0], y, y + tile_size[1]
                )
                if n_dim == 3:
                    for z in range(0, size_z - tile_size[2] + 1, step_size[2]):
                        cropped_mask = mask[x: x + tile_size[0],
                                            y: y + tile_size[1],
                                            z: z + tile_size[2]]
                        roi_vf = np.sum(cropped_mask) / np.prod(cropped_mask.shape)
                        if roi_vf >= min_fraction:
                            cur_index = [x, x + tile_size[0], y,
                                         y + tile_size[1], z, z + tile_size[2]]
                            cur_index_list.append(cur_index)
                            if isotropic_cond is not None:
                                cropped_mask = resize_mask(cropped_mask,
                                                           isotropic_shape)
                            img_id = '{}_z{}-{}.npy'.format(
                                img_id, z, z + tile_size[2]
                            )
                            cropped_mask_fname = os.path.join(mask_op_dir,
                                                              img_id)
                            np.save(cropped_mask_fname, cropped_mask,
                                    allow_pickle=True, fix_imports=True)
                else:
                    cropped_mask = mask[x: x + tile_size[0], y: y + tile_size[1]]
                    roi_vf = np.sum(cropped_mask) / np.prod(cropped_mask.shape)
                    if roi_vf >= min_fraction:
                        cur_index = [x, x + tile_size[0], y, y + tile_size[1]]
                        cur_index_list.append(cur_index)
                        img_id = '{}.npy'.format(img_id)
                        cropped_mask_fname = os.path.join(mask_op_dir, img_id)
                        np.save(cropped_mask_fname, cropped_mask,
                                allow_pickle=True, fix_imports=True)
        index_dict[fname] = cur_index_list
    return index_dict


def crop_at_indices(input_image, crop_indices, isotropic=False):
    """Crop image into tiles at given indices

    :param np.array input_image: input image in 3d
    :param list crop_indices: list of indices for cropping
    :param bool isotropic: if 3D, make the grid/shape isotropic
    :return: a list with tuples of cropped image id of the format
     xxmin-xmz_yymin-ymax_zzmin-zmax and cropped image
    """

    n_dim = len(input_image.shape)
    cropped_img_list = []
    for cur_idx in crop_indices:
        img_id = 'x{}-{}_y{}-{}'.format(cur_idx[0], cur_idx[1],
                                        cur_idx[2], cur_idx[3])
        if n_dim == 3:
            img_id = '{}_z{}-{}'.format(img_id, cur_idx[4], cur_idx[5])
            cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                      cur_idx[2]: cur_idx[3],
                                      cur_idx[4]: cur_idx[5]]
            if isotropic:
                img_shape = cropped_img.shape
                isotropic_shape = [img_shape[0], ] * len(img_shape)
                cropped_img = resize_image(cropped_img, isotropic_shape)
        else:
            cropped_img = input_image[cur_idx[0]: cur_idx[1],
                                      cur_idx[2]: cur_idx[3]]
        cropped_img_list.append((img_id, cropped_img))
    return cropped_img_list