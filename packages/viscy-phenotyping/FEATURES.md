# viscy-phenotyping: Feature Reference

All features are computed per cell per timepoint. Per-channel features are prefixed
with the channel name (e.g. `raw_mCherry_EX561_EM600-37_intensity_cv`). Nuclear shape
features have no channel prefix and are computed once per cell.

---

## Nuclear Shape Features (Problem 6)

**Source:** `features_shape.py` — `shape_features(mask)`
**Input:** Binary nuclear mask (2-D, no intensity image required)
**Prefix:** none

These features describe the geometry of the nucleus boundary. They are independent of
fluorescence signal and are the same regardless of which channel is being processed.

| Feature | Description |
|---|---|
| `circularity` | 4π × area / perimeter². Equals 1.0 for a perfect circle. Low values indicate elongated or irregular shapes. |
| `convexity` | Convex-hull perimeter / object perimeter. Equals 1.0 for a fully convex shape. Values < 1 indicate lobes or indentations in the boundary. |
| `radial_std_norm` | Standard deviation of boundary radii (distance from centroid to each contour point), normalised by the mean radius. High values indicate an irregular or multi-lobed boundary. |
| `fsd_1` … `fsd_6` | Fourier Shape Descriptor amplitudes. The nuclear boundary is decomposed into Fourier harmonics; each `fsd_k` is the amplitude of the k-th harmonic normalised by the first harmonic. Low-order descriptors (fsd_1, fsd_2) capture global elongation and ellipticity; higher-order descriptors capture finer lobes and protrusions. |

---

## Radial Distribution Features (Problem 1)

**Source:** `features_radial.py` — `radial_distribution_features(image, nuclear_mask)`
**Input:** Full fluorescence patch; nuclear mask used only to locate the centroid
**Prefix:** `{channel}_`

These features describe how fluorescence intensity is distributed radially outward from
the nuclear centre across the entire patch.

| Feature | Description |
|---|---|
| `radial_frac_bin0` … `radial_frac_bin7` | Fraction of total patch intensity contained in each of 8 concentric rings, centred on the nuclear centroid. Bin 0 is the innermost ring (around the nucleus); bin 7 is the outermost (cytoplasm/background). |
| `radial_frac_cv` | Coefficient of variation (CV) across the 8 radial bins. High = signal concentrated in a few rings; low = evenly distributed. |
| `radial_slope` | Slope of a linear fit to mean bin intensity vs radial distance from the nuclear centroid, negated and normalised by mean intensity. Positive = signal is brighter at the centre of the patch (decreases outward); negative = signal is brighter at the periphery / boundary (increases outward). |
| `com_offset_norm` | Distance between the intensity centre-of-mass of the full patch and the geometric nuclear centroid, normalised by the equivalent circle radius of the nucleus. High = signal is shifted to one side of the nucleus. |
| `angular_cv` | CV of mean intensity across 8 angular sectors centred on the nucleus. High = signal is angularly asymmetric (concentrated in one direction). |

---

## Concentric Ring Uniformity Features (Problem 3)

**Source:** `features_radial.py` — `concentric_uniformity_features(image, nuclear_mask)`
**Input:** Full fluorescence patch; nuclear mask used only to locate the centroid
**Prefix:** `{channel}_`

These features characterise the uniformity and periodicity of the radial intensity
profile — useful for detecting ER-like concentric ring patterns.

| Feature | Description |
|---|---|
| `radial_profile_cv` | CV of the mean intensity across 16 concentric radial bins. Low = flat, uniform profile across the patch; high = one or more bright rings. |
| `radial_dominant_freq` | Index of the dominant frequency component in the FFT of the radial profile (1 = one bright ring, 2 = two rings, etc.). |
| `radial_spectral_cv` | CV of FFT amplitudes. Low = one clearly dominant frequency (regular ring spacing); high = irregular multi-frequency profile. |
| `radial_autocorr_lag1` | Lag-1 autocorrelation of the radial profile. High = slowly varying / smooth radial structure; near zero = no spatial correlation. |
| `peak_spacing_cv` | CV of distances between consecutive peaks in the radial profile. Low = evenly spaced rings; NaN if fewer than 2 peaks are found. |

---

## Texture / Homogeneity Features (Problem 2)

**Source:** `features_texture.py` — `texture_features(image)`
**Input:** Full fluorescence patch (no mask)
**Prefix:** `{channel}_`

These features characterise the spatial heterogeneity and co-occurrence structure of
pixel intensities across the entire patch.

### Intensity statistics

| Feature | Description |
|---|---|
| `intensity_mean` | Mean intensity of all pixels in the patch. Reflects the overall brightness of the fluorescence signal. |
| `intensity_median` | Median intensity of all pixels in the patch. More robust than the mean to bright outliers (e.g. single saturated spots). |
| `intensity_cv` | Coefficient of variation (std / mean) of all pixel intensities. Low = uniform signal; high = highly variable. |
| `intensity_entropy` | Shannon entropy of the 64-bin intensity histogram. High = broad, spread-out intensity distribution; low = narrow/concentrated. |

### GLCM Haralick features

The Grey-Level Co-occurrence Matrix (GLCM) is computed at 2 distances (1 px, 3 px)
and 4 angles (0°, 45°, 90°, 135°). Each property is summarised as a mean and standard
deviation across all distance–angle combinations.

| Feature | Description |
|---|---|
| `glcm_contrast_mean/std` | Measures local intensity variation. High = large differences between neighbouring pixels (sharp edges, heterogeneous texture). |
| `glcm_dissimilarity_mean/std` | Similar to contrast but grows linearly with difference rather than quadratically. |
| `glcm_homogeneity_mean/std` | Measures closeness of element distribution to the GLCM diagonal. High = similar neighbouring pixels (smooth, homogeneous signal). |
| `glcm_energy_mean/std` | Sum of squared GLCM elements (also called Angular Second Moment). High = repeating or uniform texture. |
| `glcm_correlation_mean/std` | Correlation between neighbouring pixel grey levels. High = linear relationships between neighbouring pixels. |
| `glcm_ASM_mean/std` | Angular Second Moment — same as energy, provides a redundant but commonly reported measure of texture uniformity. |

### Local Binary Pattern features

LBP encodes the local neighbourhood structure of each pixel as a binary code.

| Feature | Description |
|---|---|
| `lbp_entropy` | Shannon entropy of the LBP histogram. High = diverse local texture patterns; low = repetitive or uniform texture. |
| `lbp_energy` | Sum of squared LBP histogram values. High = one or few dominant texture patterns. |

---

## Signal Packing Density Features (Problem 4)

**Source:** `features_density.py` — `density_features(image)`
**Input:** Full fluorescence patch (no mask)
**Prefix:** `{channel}_`

These features characterise how densely bright structures (spots, puncta, organelles)
are packed across the patch.

### Binary thresholding features

An Otsu threshold is applied to segment bright structures from background.

| Feature | Description |
|---|---|
| `binary_area_fraction` | Fraction of all patch pixels above the Otsu threshold. High = densely bright patch. |
| `spot_count` | Number of connected components in the thresholded binary image. |
| `spot_mean_area` | Mean area (pixels) of detected components. |
| `spot_max_area` | Area of the largest detected component. |
| `spot_density` | Spot count per patch pixel (spot_count / total pixels). |

### Granularity spectrum

Morphological opening removes structures smaller than the structuring element. The
fraction of signal removed at each scale quantifies the size distribution of bright
structures.

| Feature | Description |
|---|---|
| `granularity_1` … `granularity_8` | Fraction of total image intensity removed by morphological opening with a disk of radius r (r = 1..8 pixels). High at small r = fine-grained puncta or dense small spots. High at large r = coarse or large bright regions. The peak of the granularity spectrum indicates the dominant size scale of bright structures. |

---

## Edge Density and Strand Continuity Features (Problem 5)

**Source:** `features_structure.py` — `structure_features(image)`
**Input:** Full fluorescence patch (no mask)
**Prefix:** `{channel}_`

These features characterise filamentous or strand-like structures (e.g. ER tubules,
cytoskeletal fibres) using edge detection and skeletonisation.

### Edge features

| Feature | Description |
|---|---|
| `edge_density` | Fraction of patch pixels classified as edges by Canny edge detection. High = many edges / complex boundary structure. |

### Connected component features

| Feature | Description |
|---|---|
| `n_connected_components` | Number of connected components in the Otsu-thresholded binary image. High = many disconnected structures. |
| `cc_mean_area` | Mean area of connected components. |
| `cc_max_area` | Area of the largest connected component. |
| `signal_euler_number` | Euler number of the binary signal (number of objects minus number of holes). Negative values indicate structures with holes (ring-like morphology). |

### Skeleton features

The binary image is skeletonised (reduced to single-pixel-wide centrelines).

| Feature | Description |
|---|---|
| `skeleton_length` | Total number of skeleton pixels. High = long or numerous filamentous structures. |
| `skeleton_branch_points` | Number of junction pixels (connected to > 2 neighbours). High = complex, networked topology. |
| `skeleton_endpoints` | Number of terminal pixels (connected to exactly 1 neighbour). High = many broken strand ends. |
| `skeleton_mean_segment_length` | Skeleton length divided by (branch_points + endpoints/2 + 1). A proxy for strand continuity — high = few breaks between junctions, long uninterrupted strands. |

---

## Gradient and Sharpness Features (Problem 7)

**Source:** `features_gradient.py` — `gradient_features(image, nuclear_mask)`
**Input:** Full fluorescence patch for gradient statistics; nuclear mask used only for `nucleus_to_cytoplasm_ratio`
**Prefix:** `{channel}_`

These features characterise the sharpness and boundary definition of signals across
the patch, and contrast between the nuclear and cytoplasmic regions.

| Feature | Description |
|---|---|
| `gradient_mean` | Mean Sobel gradient magnitude across all patch pixels. High = many strong edges / sharp signal transitions. |
| `gradient_std` | Standard deviation of Sobel gradient magnitude. High = spatially variable edge strength (some very sharp, others diffuse). |
| `gradient_p95` | 95th-percentile of Sobel gradient magnitude. Reflects the sharpest edges in the patch without being dominated by outliers. |
| `laplacian_variance` | Variance of the discrete Laplacian across the patch. Commonly used as a sharpness/focus metric — high = sharp, well-defined boundaries. |
| `gradient_entropy` | Shannon entropy of the gradient magnitude histogram. High = diverse range of edge strengths; low = uniformly weak or uniformly strong edges. |
| `nucleus_mean_intensity` | Mean intensity of pixels inside the nuclear mask. Directly reflects nuclear signal brightness independent of background — the most sensitive feature for detecting changes in nuclear fluorescence intensity over time. |
| `cytoplasm_mean_intensity` | Mean intensity of all pixels outside the nuclear mask. Reflects background / cytoplasmic signal level. |
| `nucleus_to_cytoplasm_ratio` | Mean intensity inside the nuclear mask divided by the mean intensity of all pixels outside the nuclear mask. High = bright nuclear signal against a dark background; values < 1 = cytoplasmic signal brighter than nuclear. |

---

## Nuclear Morphology Features (standalone)

**Source:** `features.py` — `extract_nuclear_morphology(label_image, label_ids)`
**Input:** Full-FOV integer label image (2-D or 3-D); used independently of the patch pipeline
**Prefix:** none (returned as a DataFrame, one row per nucleus)

These are classical region-property measurements on the segmented nuclear mask, useful
for population-level morphological profiling.

### 2-D properties

| Feature | Description |
|---|---|
| `area` | Number of pixels in the nucleus. |
| `eccentricity` | Eccentricity of the best-fit ellipse (0 = circle, 1 = line). |
| `equivalent_diameter_area` | Diameter of a circle with the same area as the nucleus. |
| `extent` | Ratio of nucleus area to its bounding-box area. Low = irregular or non-compact shape. |
| `euler_number` | Number of objects minus number of holes. |
| `major_axis_length` | Length of the major axis of the best-fit ellipse. |
| `minor_axis_length` | Length of the minor axis of the best-fit ellipse. |
| `orientation` | Angle of the major axis relative to the horizontal (radians). |
| `perimeter` | Perimeter length of the nucleus boundary. |
| `solidity` | Nucleus area / convex-hull area. Low = concave or irregular shape. |
| `aspect_ratio` | major_axis_length / minor_axis_length. High = elongated nucleus. |

### Additional 3-D properties

| Feature | Description |
|---|---|
| `inertia_eigval_0/1/2` | Eigenvalues of the inertia tensor, describing the principal axes of mass distribution of the 3-D nucleus volume. |

---

## Feature naming convention

All per-channel features in the CSV output follow the pattern:

```
{channel_name}_{feature_name}
```

Spaces in channel names are replaced with underscores. For example, the `intensity_cv`
feature for channel `raw mCherry EX561 EM600-37` is stored as:

```
raw_mCherry_EX561_EM600-37_intensity_cv
```

Nuclear shape features (`circularity`, `convexity`, `radial_std_norm`, `fsd_1`…`fsd_6`)
have no channel prefix.

---

## CellProfiler Measurements (`cp-measure`)

These features are computed using the
[cp-measure](https://github.com/afermg/cp_measure) library, which provides
faithful Python implementations of CellProfiler's measurement modules. They
complement the custom features above with established, widely-used morphological
and intensity descriptors.

### Naming convention

| Feature group | Prefix in CSV |
|---|---|
| MeasureObjectSizeShape | `cp_{feature}` |
| MeasureObjectIntensity | `{channel}_cp_{feature}` |
| MeasureTexture | `{channel}_cp_{feature}` |
| MeasureGranularity | `{channel}_cp_{feature}` |

---

## cp MeasureObjectSizeShape

**Source:** `features_cp_measure.py` — `cp_sizeshape_features(mask)`
**Input:** Binary nuclear mask (no intensity image)
**Prefix:** `cp_`

| Feature | Description |
|---|---|
| `cp_Area` | Number of pixels in the nucleus. |
| `cp_BoundingBoxArea` | Area of the nucleus bounding box. |
| `cp_ConvexArea` | Area of the convex hull of the nucleus. |
| `cp_EquivalentDiameter` | Diameter of a circle with the same area as the nucleus. |
| `cp_Perimeter` | Perimeter length of the nucleus boundary. |
| `cp_PerimeterCrofton` | Perimeter estimated using the Crofton formula (more accurate for digital images). |
| `cp_MajorAxisLength` | Length of the major axis of the best-fit ellipse. |
| `cp_MinorAxisLength` | Length of the minor axis of the best-fit ellipse. |
| `cp_Eccentricity` | Eccentricity of the best-fit ellipse (0 = circle, 1 = line). |
| `cp_Orientation` | Angle of the major axis relative to the horizontal (degrees). |
| `cp_FormFactor` | 4π × Area / Perimeter². Equals 1.0 for a perfect circle. |
| `cp_Extent` | Nucleus area / bounding-box area. Low = non-compact shape. |
| `cp_Solidity` | Nucleus area / convex-hull area. Low = concave or irregular shape. |
| `cp_Compactness` | Mean squared distance from centroid to boundary, normalised by area. |
| `cp_EulerNumber` | Number of objects minus number of holes. |
| `cp_MaximumRadius` | Maximum distance from centroid to boundary. |
| `cp_MeanRadius` | Mean distance from centroid to boundary. |
| `cp_MedianRadius` | Median distance from centroid to boundary. |
| `cp_FilledArea` | Area after filling holes in the nucleus mask. |
| `cp_MinFeretDiameter` | Minimum caliper diameter (shortest span across the nucleus). |
| `cp_MaxFeretDiameter` | Maximum caliper diameter (longest span across the nucleus). |
| `cp_HuMoment_0` … `cp_HuMoment_6` | Seven Hu invariant moments — rotation-, scale-, and translation-invariant shape descriptors. |
| `cp_Zernike_n_m` | Zernike polynomial magnitudes up to degree 9. Orthogonal shape descriptors on the unit disk. |
| `cp_SpatialMoment_p_q` | Raw spatial moments of the binary mask. |
| `cp_CentralMoment_p_q` | Translation-invariant central moments. |
| `cp_NormalizedMoment_p_q` | Scale-invariant normalised central moments. |
| `cp_InertiaTensor_i_j` | Elements of the 2×2 inertia tensor. |
| `cp_InertiaTensorEigenvalues_0/1` | Principal moments of inertia (eigenvalues of the inertia tensor). |
| `cp_Center_X/Y` | Centroid coordinates (pixels, patch-relative). |
| `cp_BoundingBoxMinimum/Maximum_X/Y` | Bounding-box corner coordinates. |

---

## cp MeasureObjectIntensity

**Source:** `features_cp_measure.py` — `cp_intensity_features(image, mask)`
**Input:** Single-channel fluorescence patch; nuclear mask
**Prefix:** `{channel}_cp_`

| Feature | Description |
|---|---|
| `Intensity_IntegratedIntensity` | Sum of all pixel intensities inside the nucleus. |
| `Intensity_MeanIntensity` | Mean intensity inside the nucleus. |
| `Intensity_StdIntensity` | Standard deviation of intensity inside the nucleus. |
| `Intensity_MinIntensity` | Minimum pixel intensity inside the nucleus. |
| `Intensity_MaxIntensity` | Maximum pixel intensity inside the nucleus. |
| `Intensity_MassDisplacement` | Distance between intensity centre-of-mass and geometric centroid, normalised by object radius. |
| `Intensity_LowerQuartileIntensity` | 25th-percentile intensity inside the nucleus. |
| `Intensity_MedianIntensity` | Median intensity inside the nucleus. |
| `Intensity_MADIntensity` | Median absolute deviation of intensities inside the nucleus. |
| `Intensity_UpperQuartileIntensity` | 75th-percentile intensity inside the nucleus. |
| `Intensity_IntegratedIntensityEdge` | Sum of pixel intensities on the nucleus boundary edge. |
| `Intensity_MeanIntensityEdge` | Mean intensity on the nucleus boundary edge. |
| `Intensity_StdIntensityEdge` | Standard deviation of intensity on the nucleus boundary edge. |
| `Intensity_MinIntensityEdge` | Minimum intensity on the nucleus boundary edge. |
| `Intensity_MaxIntensityEdge` | Maximum intensity on the nucleus boundary edge. |
| `Location_CenterMassIntensity_X/Y` | X/Y coordinates of the intensity-weighted centroid. |
| `Location_MaxIntensity_X/Y` | X/Y coordinates of the brightest pixel inside the nucleus. |

---

## cp MeasureTexture

**Source:** `features_cp_measure.py` — `cp_texture_features(image, mask)`
**Input:** Single-channel fluorescence patch; nuclear mask
**Prefix:** `{channel}_cp_`

Haralick features computed from the Grey-Level Co-occurrence Matrix (GLCM) at
scale 3 px and 4 directions (0°, 45°, 90°, 135°), quantised to 256 grey levels.
Feature names follow the pattern `{Property}_{scale}_{direction}_{levels}`.

| Feature | Description |
|---|---|
| `AngularSecondMoment_3_{dir}_256` | Uniformity of the GLCM (Angular Second Moment). High = repetitive or homogeneous texture. |
| `Contrast_3_{dir}_256` | Local intensity variation between neighbouring pixels. High = high-contrast, heterogeneous texture. |
| `Correlation_3_{dir}_256` | Linear correlation between neighbouring pixel grey levels. |
| `Variance_3_{dir}_256` | Variance of grey-level intensities in the GLCM. |
| `InverseDifferenceMoment_3_{dir}_256` | Homogeneity. High = similar neighbouring pixels (smooth texture). |
| `SumAverage_3_{dir}_256` | Mean of the sum of grey-level pairs. |
| `SumVariance_3_{dir}_256` | Variance of the sum of grey-level pairs. |
| `SumEntropy_3_{dir}_256` | Entropy of the sum distribution. |
| `Entropy_3_{dir}_256` | Shannon entropy of the full GLCM. High = complex, non-repetitive texture. |
| `DifferenceVariance_3_{dir}_256` | Variance of the difference between grey-level pairs. |
| `DifferenceEntropy_3_{dir}_256` | Entropy of the difference distribution. |
| `InfoMeas1_3_{dir}_256` | Information measure of correlation 1 (HXY1). |
| `InfoMeas2_3_{dir}_256` | Information measure of correlation 2 (HXY2). |

Directions: `00` = 0°, `01` = 45°, `02` = 90°, `03` = 135°.

---

## cp MeasureGranularity

**Source:** `features_cp_measure.py` — `cp_granularity_features(image, mask)`
**Input:** Single-channel fluorescence patch; nuclear mask
**Prefix:** `{channel}_cp_`

The granularity spectrum quantifies the size distribution of bright structures
by applying morphological opening at increasing scales and measuring the fraction
of signal removed at each scale.

| Feature | Description |
|---|---|
| `Granularity_1` … `Granularity_16` | Fraction of total image intensity removed by morphological opening at scale r = 1..16 pixels. High at small r = fine-grained puncta or dense small spots. High at large r = coarse or large bright regions. The peak of the spectrum indicates the dominant size scale of bright structures in the patch. |
