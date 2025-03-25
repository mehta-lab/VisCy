# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from viscy.representation.embedding_writer import read_embedding_dataset
from scipy.spatial.distance import pdist, squareform

# Set global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def analyze_pc_loadings(pca, feature_names=None, top_n=5):
    """Analyze which features contribute most to each PC."""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(pca.components_[0].shape[0])]

    pc_loadings = []
    for i, pc in enumerate(pca.components_):
        # Get the absolute loadings
        abs_loadings = np.abs(pc)
        # Get indices of top contributing features
        top_indices = np.argsort(abs_loadings)[-top_n:][::-1]

        # Store the results
        pc_dict = {
            "PC": i + 1,
            "Variance_Explained": pca.explained_variance_ratio_[i],
            "Top_Features": [feature_names[idx] for idx in top_indices],
            "Top_Loadings": [pc[idx] for idx in top_indices],
        }
        pc_loadings.append(pc_dict)

    return pd.DataFrame(pc_loadings)


def analyze_track_clustering(
    pca_result,
    track_ids,
    time_points,
    labels,
    phenotype_of_interest,
    seed_timepoint,
    time_window,
):
    """Analyze how tracks cluster in PC space within the time window."""
    # Get points within time window
    time_mask = (time_points >= seed_timepoint - time_window) & (
        time_points <= seed_timepoint + time_window
    )
    window_points = pca_result[time_mask]
    window_tracks = track_ids[time_mask]
    window_labels = labels[time_mask]

    # Calculate mean position for each track
    track_means = {}
    phenotype_tracks = []

    for track_id in np.unique(window_tracks):
        track_mask = (window_tracks == track_id) & (
            window_labels == phenotype_of_interest
        )
        if np.any(track_mask):
            track_means[track_id] = np.mean(window_points[track_mask], axis=0)
            phenotype_tracks.append(track_id)

    if len(phenotype_tracks) < 2:
        return None

    # Calculate pairwise distances between track means
    track_positions = np.array([track_means[tid] for tid in phenotype_tracks])
    distances = pdist(track_positions)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Calculate spread within each track
    track_spreads = {}
    for track_id in phenotype_tracks:
        track_mask = (window_tracks == track_id) & (
            window_labels == phenotype_of_interest
        )
        if np.sum(track_mask) > 1:
            track_points = window_points[track_mask]
            spread = np.mean(pdist(track_points))
            track_spreads[track_id] = spread

    mean_spread = np.mean(list(track_spreads.values())) if track_spreads else 0

    return {
        "n_tracks": len(phenotype_tracks),
        "mean_inter_track_distance": mean_distance,
        "std_inter_track_distance": std_distance,
        "mean_intra_track_spread": mean_spread,
        "clustering_ratio": mean_distance / mean_spread if mean_spread > 0 else np.inf,
    }


def analyze_pc_distributions(
    pca_result,
    labels,
    phenotype_of_interest,
    time_points=None,
    seed_timepoint=None,
    time_window=None,
):
    """Analyze the distributions of each PC for phenotype vs background."""
    n_components = pca_result.shape[1]
    results = []

    for i in range(n_components):
        # Get phenotype and background points
        if (
            time_points is not None
            and seed_timepoint is not None
            and time_window is not None
        ):
            time_mask = (time_points >= seed_timepoint - time_window) & (
                time_points <= seed_timepoint + time_window
            )
            pc_values_phenotype = pca_result[
                time_mask & (labels == phenotype_of_interest), i
            ]
            pc_values_background = pca_result[
                time_mask & (labels != phenotype_of_interest), i
            ]
        else:
            pc_values_phenotype = pca_result[labels == phenotype_of_interest, i]
            pc_values_background = pca_result[labels != phenotype_of_interest, i]

        # Calculate basic statistics
        stats = {
            "PC": i + 1,
            "phenotype_mean": np.mean(pc_values_phenotype),
            "background_mean": np.mean(pc_values_background),
            "phenotype_std": np.std(pc_values_phenotype),
            "background_std": np.std(pc_values_background),
            "separation": abs(
                np.mean(pc_values_phenotype) - np.mean(pc_values_background)
            )
            / (np.std(pc_values_phenotype) + np.std(pc_values_background)),
        }

        # Check for multimodality using a simple peak detection
        hist, bins = np.histogram(pc_values_phenotype, bins="auto")
        peaks = len(
            [
                i
                for i in range(1, len(hist) - 1)
                if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]
            ]
        )
        stats["n_peaks"] = peaks

        results.append(stats)

    return pd.DataFrame(results)


def analyze_gmm_clustering(
    pca_result,
    track_ids,
    time_points,
    tracks_of_interest,
    n_components_range=range(2, 7),
    seed_timepoint=None,
    time_window=None,
):
    """Analyze clusters using Gaussian Mixture Models."""
    # Get points from tracks of interest
    track_mask = np.isin(track_ids, tracks_of_interest)
    points = pca_result[track_mask]
    track_ids_subset = track_ids[track_mask]
    times = time_points[track_mask]

    # Apply time window if specified
    if seed_timepoint is not None and time_window is not None:
        time_mask = (times >= seed_timepoint - time_window) & (
            times <= seed_timepoint + time_window
        )
        points = points[time_mask]
        track_ids_subset = track_ids_subset[time_mask]
        times = times[time_mask]

    # Try different numbers of components
    bic_scores = []
    silhouette_scores = []
    models = []

    for n_components in n_components_range:
        gmm = GaussianMixture(
            n_components=n_components, random_state=RANDOM_SEED, n_init=10
        )
        gmm.fit(points)
        labels = gmm.predict(points)

        bic_scores.append(gmm.bic(points))
        silhouette_scores.append(silhouette_score(points, labels))
        models.append(gmm)

    # Plot model selection metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # BIC plot
    ax1.plot(list(n_components_range), bic_scores, "bo-")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("BIC Score")
    ax1.set_title("Model Selection: BIC")

    # Silhouette plot
    ax2.plot(list(n_components_range), silhouette_scores, "ro-")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Model Selection: Silhouette")

    plt.tight_layout()
    plt.show()

    # Select best model based on BIC
    best_idx = np.argmin(bic_scores)
    best_n_components = n_components_range[best_idx]
    best_model = models[best_idx]

    # Get cluster assignments
    labels = best_model.predict(points)
    probs = best_model.predict_proba(points)

    # Plot clustering results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot colored by cluster
    scatter = ax1.scatter(
        points[:, 0], points[:, 1], c=labels, cmap="tab10", alpha=0.6, s=50
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title(f"GMM Clustering (n={best_n_components})")
    plt.colorbar(scatter, ax=ax1, label="Cluster")

    # Plot cluster assignment probabilities
    max_probs = np.max(probs, axis=1)
    scatter = ax2.scatter(
        points[:, 0], points[:, 1], c=max_probs, cmap="viridis", alpha=0.6, s=50
    )
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Cluster Assignment Probability")
    plt.colorbar(scatter, ax=ax2, label="Probability")

    plt.tight_layout()
    plt.show()

    # Analyze cluster composition
    cluster_stats = []
    for i in range(best_n_components):
        cluster_mask = labels == i
        cluster_tracks = np.unique(track_ids_subset[cluster_mask])
        cluster_stats.append(
            {
                "cluster": i,
                "n_points": np.sum(cluster_mask),
                "n_tracks": len(cluster_tracks),
                "tracks": cluster_tracks,
                "mean_prob": np.mean(probs[cluster_mask, i]),
                "std_prob": np.std(probs[cluster_mask, i]),
            }
        )

    # Print cluster statistics
    print(f"\nBest number of clusters (BIC): {best_n_components}")
    print("\nCluster Statistics:")
    for stats in cluster_stats:
        print(f"\nCluster {stats['cluster']}:")
        print(f"  Points: {stats['n_points']}")
        print(f"  Tracks: {stats['n_tracks']}")
        print(f"  Mean probability: {stats['mean_prob']:.3f} ± {stats['std_prob']:.3f}")
        print(f"  Tracks in cluster: {stats['tracks']}")

    return {
        "best_model": best_model,
        "best_n_components": best_n_components,
        "labels": labels,
        "probabilities": probs,
        "bic_scores": bic_scores,
        "silhouette_scores": silhouette_scores,
        "cluster_stats": cluster_stats,
    }


def analyze_cluster_characteristics(
    gmm_results,
    pca_result,
    track_ids,
    time_points,
    tracks_of_interest,
    pc_analysis=None,
    seed_timepoint=None,
    time_window=None,
):
    """Analyze characteristics of GMM clusters including temporal patterns and PC contributions."""
    # Get points from tracks of interest first
    track_mask = np.isin(track_ids, tracks_of_interest)
    points = pca_result[track_mask]
    track_ids_subset = track_ids[track_mask]
    times = time_points[track_mask]

    # Apply time window if specified
    if seed_timepoint is not None and time_window is not None:
        time_mask = (times >= seed_timepoint - time_window) & (
            times <= seed_timepoint + time_window
        )
        points = points[time_mask]
        track_ids_subset = track_ids_subset[time_mask]
        times = times[time_mask]

    # Get cluster assignments for the filtered points
    labels = gmm_results["labels"]
    probs = gmm_results["probabilities"]
    n_clusters = gmm_results["best_n_components"]

    # Analyze temporal patterns in each cluster
    print("\nTemporal patterns in clusters:")
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_times = times[cluster_mask]
        if len(cluster_times) > 0:
            print(f"\nCluster {i}:")
            print(
                f"  Time range: {np.min(cluster_times):.1f} to {np.max(cluster_times):.1f}"
            )
            print(
                f"  Mean time: {np.mean(cluster_times):.1f} ± {np.std(cluster_times):.1f}"
            )

    # Analyze PC contributions to cluster separation
    print("\nPC contributions to cluster separation:")
    for pc_idx in range(min(4, points.shape[1])):  # Analyze first 4 PCs
        pc_values = points[:, pc_idx]
        cluster_means = [np.mean(pc_values[labels == i]) for i in range(n_clusters)]
        cluster_stds = [np.std(pc_values[labels == i]) for i in range(n_clusters)]

        # Calculate separation score (ratio of between-cluster to within-cluster variance)
        between_var = np.var(cluster_means)
        within_var = np.mean(cluster_stds)
        separation_score = between_var / within_var if within_var > 0 else float("inf")

        print(f"\nPC{pc_idx + 1}:")
        print(f"  Separation score: {separation_score:.3f}")
        if pc_analysis is not None:
            pc_info = pc_analysis[pc_analysis["PC"] == pc_idx + 1].iloc[0]
            print(
                f"  Top contributing features: {', '.join(pc_info['Top_Features'][:3])}"
            )

        # Print cluster-specific stats
        for i in range(n_clusters):
            cluster_mask = labels == i
            print(f"  Cluster {i}: {cluster_means[i]:.3f} ± {cluster_stds[i]:.3f}")

    # Analyze track transitions between clusters
    print("\nTrack transitions between clusters:")
    for track_id in tracks_of_interest:
        track_mask = track_ids_subset == track_id
        track_labels = labels[track_mask]
        track_times = times[track_mask]

        if len(track_labels) > 1:
            # Sort by time
            sort_idx = np.argsort(track_times)
            track_labels = track_labels[sort_idx]
            track_times = track_times[sort_idx]

            # Find transitions
            transitions = np.where(track_labels[1:] != track_labels[:-1])[0]
            if len(transitions) > 0:
                print(f"\nTrack {track_id}:")
                for trans_idx in transitions:
                    from_cluster = track_labels[trans_idx]
                    to_cluster = track_labels[trans_idx + 1]
                    trans_time = track_times[trans_idx + 1]
                    print(f"  {trans_time:.1f}: {from_cluster} -> {to_cluster}")

    return {
        "temporal_patterns": {
            i: {
                "mean_time": np.mean(times[labels == i]),
                "std_time": np.std(times[labels == i]),
            }
            for i in range(n_clusters)
        },
        "pc_contributions": {
            f"PC{pc_idx + 1}": {
                "separation_score": (
                    np.var(
                        [
                            np.mean(points[labels == i, pc_idx])
                            for i in range(n_clusters)
                        ]
                    )
                    / np.mean(
                        [np.std(points[labels == i, pc_idx]) for i in range(n_clusters)]
                    )
                    if np.mean(
                        [np.std(points[labels == i, pc_idx]) for i in range(n_clusters)]
                    )
                    > 0
                    else float("inf")
                )
            }
            for pc_idx in range(min(4, points.shape[1]))
        },
    }


def analyze_embeddings_with_pca(
    embedding_path,
    annotation_path=None,
    phenotype_of_interest=None,
    n_random_tracks=10,
    n_components=8,
    seed_timepoint=None,
    time_window=10,
    fov_patterns=None,
):
    """Analyze embeddings using PCA, either for specific phenotypes or random tracks.

    Args:
        embedding_path: Path to embedding zarr file
        annotation_path: Optional path to annotation CSV file. If None, uses random tracks
        phenotype_of_interest: Which phenotype to analyze (only used if annotation_path is provided)
        n_random_tracks: Number of random tracks to select (only used if annotation_path is None)
        n_components: Number of PCA components
        seed_timepoint: Center of time window. If None, uses all timepoints
        time_window: Size of time window (+/-). Only used if seed_timepoint is not None
        fov_patterns: List of patterns to filter FOVs (e.g. ['/C/2/*', '/B/3/*']).
                     Optional even when using annotation_path - can be used to restrict
                     analysis to specific FOVs while still using phenotype information.
    """
    if annotation_path is None:
        print(f"\nUsing random tracks (global seed: {RANDOM_SEED})")

    if seed_timepoint is None:
        print("\nUsing all timepoints")
    else:
        print(f"\nUsing time window: {seed_timepoint}±{time_window}")

    # Load embeddings
    embedding_dataset = read_embedding_dataset(embedding_path)
    features = embedding_dataset["features"]
    track_ids = embedding_dataset["track_id"].values
    fovs = embedding_dataset["fov_name"].values
    time_points = embedding_dataset["t"].values

    # Filter FOVs if patterns are provided
    if fov_patterns is not None:
        print(f"\nFiltering FOVs with patterns: {fov_patterns}")
        fov_mask = np.zeros_like(fovs, dtype=bool)
        for pattern in fov_patterns:
            fov_mask |= np.char.find(fovs.astype(str), pattern) >= 0

        # Update all arrays with the FOV mask
        features = features[fov_mask]
        track_ids = track_ids[fov_mask]
        fovs = fovs[fov_mask]
        time_points = time_points[fov_mask]

        print(f"Found {len(np.unique(fovs))} FOVs matching patterns")

    # Get tracks of interest
    if annotation_path is not None:
        # Load annotations and get phenotype tracks
        annotations_df = pd.read_csv(annotation_path)
        annotation_map = {
            (str(row["FOV"]), int(row["Track_id"])): row["Observed phenotype"]
            for _, row in annotations_df.iterrows()
        }
        labels = np.array(
            [
                annotation_map.get((str(fov), int(track_id)), -1)
                for fov, track_id in zip(fovs, track_ids)
            ]
        )
        selection_mask = labels == phenotype_of_interest
        tracks_of_interest = np.unique(track_ids[selection_mask])
        other_mask = ~selection_mask
        mode = f"phenotype {phenotype_of_interest}"
    else:
        # Select random tracks from different FOVs when possible
        # Create a mapping of FOV to tracks
        fov_track_map = {}
        for fov, track_id in zip(fovs, track_ids):
            if fov not in fov_track_map:
                fov_track_map[fov] = []
            if track_id not in fov_track_map[fov]:  # Avoid duplicates
                fov_track_map[fov].append(track_id)

        # Get list of all FOVs
        available_fovs = list(fov_track_map.keys())
        tracks_of_interest = []

        # First, try to get one track from each FOV
        np.random.shuffle(available_fovs)  # Randomize FOV order
        for fov in available_fovs:
            if len(tracks_of_interest) < n_random_tracks:
                # Randomly select a track from this FOV
                track = np.random.choice(fov_track_map[fov])
                tracks_of_interest.append(track)
            else:
                break

        # If we still need more tracks, randomly select from remaining tracks
        if len(tracks_of_interest) < n_random_tracks:
            # Get all remaining tracks that aren't already selected
            remaining_tracks = [
                track
                for track in np.unique(track_ids)
                if track not in tracks_of_interest
            ]
            # Select additional tracks
            additional_tracks = np.random.choice(
                remaining_tracks,
                size=min(
                    n_random_tracks - len(tracks_of_interest), len(remaining_tracks)
                ),
                replace=False,
            )
            tracks_of_interest.extend(additional_tracks)

        tracks_of_interest = np.array(tracks_of_interest)
        selection_mask = np.isin(track_ids, tracks_of_interest)
        other_mask = ~selection_mask
        labels = np.where(selection_mask, 1, 0)
        mode = "random tracks"

        # Print selected tracks with their FOVs
        print("\nSelected tracks:")
        for track in tracks_of_interest:
            track_fovs = np.unique(fovs[track_ids == track])
            print(f"Track {track}: FOV {track_fovs[0]}")

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.values)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_features)

    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Create track-specific colors
    track_colors = plt.cm.tab10(np.linspace(0, 1, len(tracks_of_interest)))
    track_color_map = dict(zip(tracks_of_interest, track_colors))

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scree plot
    ax1.plot(range(1, n_components + 1), explained_variance_ratio, "bo-")
    ax1.plot(range(1, n_components + 1), cumulative_variance_ratio, "ro-")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Scree Plot")
    ax1.legend(["Individual", "Cumulative"])

    # First two components plot
    # Plot other tracks/cells in gray
    ax2.scatter(
        pca_result[other_mask, 0],
        pca_result[other_mask, 1],
        alpha=0.1,
        color="gray",
        label="Other cells",
        s=10,
    )

    # Plot tracks of interest with decreasing opacity
    for track_id in tracks_of_interest:
        track_mask = track_ids == track_id
        track_points = pca_result[track_mask]
        track_times = time_points[track_mask]

        # Sort points by time
        sort_idx = np.argsort(track_times)
        track_points = track_points[sort_idx]
        track_times = track_times[sort_idx]

        # Apply time window if specified
        if seed_timepoint is not None:
            time_mask = (track_times >= seed_timepoint - time_window) & (
                track_times <= seed_timepoint + time_window
            )
        else:
            time_mask = np.ones_like(track_times, dtype=bool)  # Use all points

        if np.any(time_mask):  # Only plot if there are points in the window
            window_points = track_points[time_mask]
            window_times = track_times[time_mask]

            # Normalize times within window for opacity
            norm_times = (window_times - window_times.min()) / (
                window_times.max() - window_times.min() + 1e-10
            )
            alphas = 0.2 + 0.8 * norm_times  # Scale to [0.2, 1.0]

            # Plot points with opacity based on normalized time
            for idx in range(len(window_points)):
                ax2.scatter(
                    window_points[idx, 0],
                    window_points[idx, 1],
                    color=track_color_map[track_id],
                    alpha=alphas[idx],
                    s=50,
                    label=(
                        f"Track {track_id}" if idx == len(window_points) - 1 else None
                    ),
                )

    ax2.set_xlabel("First Principal Component")
    ax2.set_ylabel("Second Principal Component")
    title = f"First Two Principal Components - {mode}"
    if seed_timepoint is not None:
        title += f"\nTime window: {seed_timepoint}±{time_window}"
    ax2.set_title(title)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    # Pairwise component plots
    fig, axes = plt.subplots(n_components, n_components, figsize=(20, 20))

    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                # Plot other points first
                axes[i, j].scatter(
                    pca_result[other_mask, j],
                    pca_result[other_mask, i],
                    alpha=0.1,
                    color="gray",
                    s=5,
                )

                # Plot each track with decreasing opacity
                for track_id in tracks_of_interest:
                    track_mask = track_ids == track_id
                    track_points_j = pca_result[track_mask, j]
                    track_points_i = pca_result[track_mask, i]
                    track_times = time_points[track_mask]

                    # Sort points by time
                    sort_idx = np.argsort(track_times)
                    track_points_j = track_points_j[sort_idx]
                    track_points_i = track_points_i[sort_idx]
                    track_times = track_times[sort_idx]

                    # Select points within the time window
                    time_mask = (track_times >= seed_timepoint - time_window) & (
                        track_times <= seed_timepoint + time_window
                    )
                    if np.any(time_mask):  # Only plot if there are points in the window
                        window_points_j = track_points_j[time_mask]
                        window_points_i = track_points_i[time_mask]
                        window_times = track_times[time_mask]

                        # Normalize times within window for opacity
                        norm_times = (window_times - window_times.min()) / (
                            window_times.max() - window_times.min() + 1e-10
                        )
                        alphas = 0.2 + 0.8 * norm_times  # Scale to [0.2, 1.0]

                        # Plot points with opacity based on normalized time
                        for idx in range(len(window_points_j)):
                            axes[i, j].scatter(
                                window_points_j[idx],
                                window_points_i[idx],
                                color=track_color_map[track_id],
                                alpha=alphas[idx],
                                s=30,
                            )

                axes[i, j].set_xlabel(f"PC{j+1}")
                axes[i, j].set_ylabel(f"PC{i+1}")
            else:
                # On diagonal, show distribution
                sns.histplot(
                    pca_result[other_mask, i], ax=axes[i, i], color="gray", alpha=0.3
                )
                for track_id in tracks_of_interest:
                    track_mask = track_ids == track_id
                    # For histograms, use all points in the time window
                    time_mask = (
                        time_points[track_mask] >= seed_timepoint - time_window
                    ) & (time_points[track_mask] <= seed_timepoint + time_window)
                    if np.any(time_mask):
                        sns.histplot(
                            pca_result[track_mask][time_mask, i],
                            ax=axes[i, i],
                            color=track_color_map[track_id],
                            alpha=0.5,
                        )
                axes[i, i].set_xlabel(f"PC{i+1}")

    plt.tight_layout()
    plt.show()

    # Print variance explained
    print("\nExplained variance ratio by component:")
    for i, var in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {var:.3f} ({cumulative_variance_ratio[i]:.3f} cumulative)")

    # Add analysis of PC loadings
    pc_analysis = analyze_pc_loadings(pca)
    print("\nPC Loading Analysis:")
    print(pc_analysis.to_string(index=False))

    # Add analysis of track clustering
    cluster_analysis = analyze_track_clustering(
        pca_result,
        track_ids,
        time_points,
        labels,
        1 if annotation_path is None else phenotype_of_interest,
        seed_timepoint,
        time_window,
    )

    if cluster_analysis:
        print("\nTrack Clustering Analysis:")
        print(f"Number of tracks in window: {cluster_analysis['n_tracks']}")
        print(
            f"Mean distance between tracks: {cluster_analysis['mean_inter_track_distance']:.3f}"
        )
        print(
            f"Mean spread within tracks: {cluster_analysis['mean_intra_track_spread']:.3f}"
        )
        print(
            f"Clustering ratio (inter/intra): {cluster_analysis['clustering_ratio']:.3f}"
        )
        print("(Lower clustering ratio suggests tighter clustering)")

    # Add distribution analysis
    dist_analysis = analyze_pc_distributions(
        pca_result,
        labels,
        1 if annotation_path is None else phenotype_of_interest,
        time_points if seed_timepoint is not None else None,
        seed_timepoint,
        time_window,
    )
    print("\nPC Distribution Analysis:")
    print(
        "(Separation score > 1 suggests good separation between selected tracks and background)"
    )
    print(dist_analysis.to_string(index=False))

    # Return PCA results and additional data needed for clustering
    return (
        pca,
        pca_result,
        explained_variance_ratio,
        labels,
        tracks_of_interest,
        pc_analysis,
        cluster_analysis,
        dist_analysis,
        track_ids,
        time_points,
    )


# %%
if __name__ == "__main__":
    embedding_path = "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/3-phenotyping/predictions/timeAware_2chan__ntxent_192patch_70ckpt_rev7_GT.zarr"
    annotation_path = "/home/eduardo.hirata/repos/viscy/applications/pseudotime_analysis/phenotype_observations.csv"

    # Using phenotype annotations with specific FOVs
    print("\nAnalyzing phenotype 1 in specific FOVs:")
    (
        pca,
        pca_result,
        variance_ratio,
        labels,
        tracks,
        pc_analysis,
        cluster_analysis,
        dist_analysis,
        track_ids,
        time_points,
    ) = analyze_embeddings_with_pca(
        embedding_path,
        annotation_path=annotation_path,
        phenotype_of_interest=1,
        seed_timepoint=55,
        time_window=10,
        fov_patterns=["/C/2/", "/B/3/", "/B/2/"],
    )

    # Run GMM clustering analysis separately
    print("\nPerforming GMM clustering analysis...")
    gmm_results = analyze_gmm_clustering(
        pca_result,
        track_ids,
        time_points,
        tracks,
        seed_timepoint=55,
        time_window=10,
    )

    # Analyze cluster characteristics
    print("\nAnalyzing cluster characteristics...")
    cluster_characteristics = analyze_cluster_characteristics(
        gmm_results,
        pca_result,
        track_ids,
        time_points,
        tracks,
        pc_analysis=pc_analysis,
        seed_timepoint=55,
        time_window=10,
    )

    # Using random tracks from specific FOVs
    print("\nAnalyzing random tracks from specific FOVs:")
    (
        pca,
        pca_result,
        variance_ratio,
        labels,
        tracks,
        pc_analysis,
        cluster_analysis,
        dist_analysis,
        track_ids,
        time_points,
    ) = analyze_embeddings_with_pca(
        embedding_path,
        annotation_path=None,
        n_random_tracks=10,
        seed_timepoint=55,
        time_window=30,
        fov_patterns=["/C/2/", "/B/3/", "/B/2/"],
    )
    # %%
    # Run GMM clustering analysis for random tracks
    print("\nPerforming GMM clustering analysis for random tracks...")
    gmm_results = analyze_gmm_clustering(
        pca_result,
        track_ids,
        time_points,
        tracks,
        seed_timepoint=55,
        time_window=30,
    )

    # Analyze cluster characteristics for random tracks
    print("\nAnalyzing cluster characteristics for random tracks...")
    cluster_characteristics = analyze_cluster_characteristics(
        gmm_results,
        pca_result,
        track_ids,
        time_points,
        tracks,
        pc_analysis=pc_analysis,
        seed_timepoint=55,
        time_window=30,
    )

# %%
