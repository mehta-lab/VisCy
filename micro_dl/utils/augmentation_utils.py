import gunpowder as gp
import numpy as np

import micro_dl.input.gunpowder_nodes as custom_nodes


class AugmentationNodeBuilder:
    """
    Optimally builds augmentations nodes given some augmentation config to
    maintain sequential compatibility and computational efficiency.
    """

    def __init__(
        self,
        augmentation_config,
        spatial_dims=2,
        noise_key=None,
        defect_key=None,
        intensities_key=None,
        blur_key=None,
        shear_key=None,
    ):
        self.config = augmentation_config
        self.spatial_dims = spatial_dims

        # augmentations with default parameters
        self.elastic_aug_params = {
            "control_point_spacing": (1, 1, 1),
            "jitter_sigma": (0, 0, 0),
            "spatial_dims": spatial_dims,
            "use_fast_points_transform": False,
            "subsample": 1,
        }
        self.simple_aug_params = {}
        self.shear_aug_params = {"shear_array_key": shear_key}
        self.intensity_aug_params = {"intensity_array_key": intensities_key}
        self.noise_aug_params = {"noise_array_key": noise_key}
        self.blur_aug_params = {"blur_array_key": blur_key}
        self.defect_aug_params = {"intensities_defect_key": defect_key}

        self.node_list = []
        self.initalized_builders = {}

    def build_nodes(self):
        """
        Build list of augmentation nodes in compatible sequential order going downstream.
        Augmentations that have spatial nonuniformity are prioritized upstream, while
        those that are uniform over a FoV are generally performed later.

        Strict ordering levels (going downstream):
        1 -> simple (mirror & transpose)
        2 -> elastic (zoom, rotation)
        3 -> blur, shear
        4 -> intensity, noise, defect (contrast & artifacts)
        """
        # collect params
        for name in self.config:
            self.init_aug_node_params(name, self.config[name])

        # build nodes in hard-coded compatible ordering
        node_ordering = [
            {self.build_simple_augment_node},
            {self.build_elastic_augment_node},
            {self.build_blur_augment_node, self.build_shear_augment_node},
            {
                self.build_intensity_augment_node,
                self.build_noise_augment_node,
                self.build_defect_augment_node,
            },
        ]

        for subset in node_ordering:
            for aug_node_builder in subset:
                if aug_node_builder in self.initalized_builders:
                    self.node_list.append(aug_node_builder())

    def get_nodes(self):
        """
        Getter for nodes.

        :return list node_list: list of initalized augmentation nodes
        """
        assert (
            len(self.node_list) > 0
        ), "Augmentation nodes not initiated or unspecified. "
        "Try .build_nodes() or check your config"

        return self.node_list

    def init_aug_node_params(self, aug_name, parameters):
        """
        Acts as a general initatialization method, which takes a augmentation name and
        parameters and initializes and returns a gunpowder node corresponding to that
        augmentation

        :param str aug_name: name of augmentation
        :param dict parameters: dict of parameter names and values for augmentation

        :return gp.BatchFilter aug_node: single gunpowder node for augmentation
        """

        # collect augmentation parameters
        if aug_name in {"transpose", "mirror"}:
            self.simple_aug_params.update(parameters)
            self.initalized_builders[self.build_simple_augment_node] = True
        elif aug_name in {"rotate", "zoom"}:
            self.elastic_aug_params.update(parameters)
            self.initalized_builders[self.build_elastic_augment_node] = True
        elif aug_name == "shear":
            self.shear_aug_params.update(parameters)
            self.initalized_builders[self.build_shear_augment_node] = True
        elif aug_name == "intensity_jitter":
            self.intensity_aug_params.update(parameters)
            self.initalized_builders[self.build_intensity_augment_node] = True
        elif aug_name == "noise":
            self.noise_aug_params.update(parameters)
            self.initalized_builders[self.build_noise_augment_node] = True
        elif aug_name == "blur":
            self.blur_aug_params.update(parameters)
            self.initalized_builders[self.build_blur_augment_node] = True
        elif aug_name == "contrast_shift":  # TODO explore
            pass
        elif aug_name == "defect":  # TODO explore
            pass

    def build_elastic_augment_node(self):
        """
        Passes parameters to elastic augmentation node and returns initialized node

        :return gp.BatchFilter: elastic augmentation node
        """

        rotation_interval = (0, 0)
        if "rotation_interval" in self.elastic_aug_params:
            rotation_interval = tuple(self.elastic_aug_params["rotation_interval"])

        scale_interval = (0, 0)
        if "scale_interval" in self.elastic_aug_params:
            scale_interval = tuple(self.elastic_aug_params["scale_interval"])

        elastic_aug = gp.ElasticAugment(
            rotation_interval=rotation_interval,
            scale_interval=scale_interval,
            control_point_spacing=self.elastic_aug_params["control_point_spacing"],
            jitter_sigma=self.elastic_aug_params["jitter_sigma"],
            spatial_dims=self.elastic_aug_params["spatial_dims"],
            use_fast_points_transform=self.elastic_aug_params[
                "use_fast_points_transform"
            ],
            subsample=self.elastic_aug_params["subsample"],
        )
        return elastic_aug

    def build_simple_augment_node(self):
        """
        Passes parameters to simple augmentation node and returns initialized node

        :return gp.BatchFilter: simple augmentation node
        """

        transpose_only = None
        transpose_probs = None
        if "transpose_only" in self.simple_aug_params:
            transpose_only = tuple(self.simple_aug_params["transpose_only"])
        else:
            transpose_probs = (0,) * self.spatial_dims + 3  # additional dim for b,t,c

        mirror_only = None
        mirror_probs = None
        if "mirror_only" in self.simple_aug_params:
            mirror_only = self.simple_aug_params["mirror_only"]
        else:
            mirror_probs = (0,) * self.spatial_dims + 3

        simple_aug = gp.SimpleAugment(
            transpose_only=transpose_only,
            mirror_only=mirror_only,
            transpose_probs=transpose_probs,
            mirror_probs=mirror_probs,
        )
        return simple_aug

    def build_shear_augment_node(self):
        """
        passes parameters to shear augmentation node and returns initialized node

        :return gp.BatchFilter: shear augmentation node
        """

        angle_range = (-15, 15)
        if "angle_range" in self.shear_aug_params:
            angle_range = self.shear_aug_params["angle_range"]

        prob = 0.2
        if "prob" in self.shear_aug_params:
            prob = self.shear_aug_params["prob"]

        shear_middle_chans = None
        if "shear_middle_slice_channels" in self.shear_aug_params:
            shear_middle_chans = self.shear_aug_params["shear_middle_slice_channels"]

        blur_aug = custom_nodes.ShearAugment(
            array=self.shear_aug_params["shear_array_key"],
            angle_range=tuple(angle_range),
            prob=prob,
            shear_middle_slice_channels=shear_middle_chans,
        )

        return blur_aug

    def build_blur_augment_node(self):
        """
        Passes parameters to blur augmentation node and returns initialized node

        :return gp.BatchFilter: blur augmentation node
        """
        mode = "gaussian"
        if "mode" in self.blur_aug_params:
            mode = self.blur_aug_params["mode"]

        width_range = (1, 7)
        if "width_range" in self.blur_aug_params:
            width_range = self.blur_aug_params["width_range"]

        sigma = 0.1
        if "sigma" in self.blur_aug_params:
            sigma = self.blur_aug_params["sigma"]

        prob = 0.2
        if "prob" in self.blur_aug_params:
            prob = self.blur_aug_params["prob"]

        blur_channels = None
        if "blur_channels" in self.blur_aug_params:
            blur_channels = self.blur_aug_params["blur_channels"]

        blur_aug = custom_nodes.BlurAugment(
            array=self.blur_aug_params["blur_array_key"],
            mode=mode,
            width_range=tuple(width_range),
            sigma=sigma,
            prob=prob,
            blur_channels=blur_channels,
        )

        return blur_aug

    def build_intensity_augment_node(self):
        """
        Passes parameters to intensity augmentation node and returns initialized node.

        """
        jitter_channels = None
        if "jitter_channels" in self.intensity_aug_params:
            if isinstance(self.intensity_aug_params["jitter_channels"], list):
                jitter_channels = tuple(self.intensity_aug_params["jitter_channels"])
            else:
                jitter_channels = tuple(self.intensity_aug_params["jitter_channels"])

        scale_range = [0.7, 1.3]
        if "scale_range" in self.intensity_aug_params:
            scale_range = tuple(self.intensity_aug_params["scale_range"])

        shift_range = [-0.15, 0.15]
        if "shift_range" in self.intensity_aug_params:
            shift_range = tuple(self.intensity_aug_params["shift_range"])

        norm_before_shift = True
        if "norm_before_shift" in self.intensity_aug_params:
            norm_before_shift = self.intensity_aug_params["norm_before_shift"]

        jitter_demeaned = True
        if "jitter_demeaned" in self.blur_aug_params:
            jitter_demeaned = self.blur_aug_params["jitter_demeaned"]

        prob = 1
        if "prob" in self.blur_aug_params:
            prob = self.blur_aug_params["prob"]

        intensity_aug = custom_nodes.IntensityAugment(
            array=self.intensity_aug_params["intensity_array_key"],
            jitter_channels=jitter_channels,
            scale_range=scale_range,
            shift_range=shift_range,
            norm_before_shift=norm_before_shift,
            jitter_demeaned=jitter_demeaned,
            prob=prob,
        )
        return intensity_aug

    def build_noise_augment_node(self):
        """
        passes parameters to noise augmentation node and returns initialized node
        """
        mode = "gaussian"
        if "mode" in self.noise_aug_params:
            mode = self.noise_aug_params["mode"]

        noise_channels = None
        if "noise_channels" in self.noise_aug_params:
            noise_channels = self.noise_aug_params["noise_channels"]

        seed = None
        if "seed" in self.noise_aug_params:
            seed = self.noise_aug_params["seed"]

        clip = False
        if "clip" in self.noise_aug_params:
            clip = self.noise_aug_params["clip"]

        prob = False
        if "prob" in self.noise_aug_params:
            prob = self.noise_aug_params["prob"]

        var = 0.01
        if "variance" in self.noise_aug_params:
            var = self.noise_aug_params["variance"]

        noise_augment = custom_nodes.NoiseAugment(
            array=self.noise_aug_params["noise_array_key"],
            mode=mode,
            noise_channels=noise_channels,
            seed=seed,
            clip=clip,
            prob=prob,
            var=var,
        )

        return noise_augment

    def build_defect_augment_node(self):
        """
        passes parameters to defect augmentation node and returns initialized node

        :return gp.BatchFilter: defect augmentation node
        """
        raise NotImplementedError("Defect augment not yet implemented")
        # TODO implement
