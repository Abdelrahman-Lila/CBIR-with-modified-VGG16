def plot_feature_space(
    features, query_feat, top_matches, imgNames, title="Feature Space Visualization"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    pca = PCA(n_components=3)
    all_features = np.vstack([features, query_feat.reshape(1, -1)])
    features_3d = pca.fit_transform(all_features)

    db_features_3d = features_3d[:-1]
    query_feature_3d = features_3d[-1]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        db_features_3d[:, 0],
        db_features_3d[:, 1],
        db_features_3d[:, 2],
        c="blue",
        alpha=0.5,
        label="Database features",
    )
    ax.scatter(
        query_feature_3d[0],
        query_feature_3d[1],
        query_feature_3d[2],
        c="red",
        s=100,
        label="Query image",
    )
    matches_3d = db_features_3d[top_matches]
    ax.scatter(
        matches_3d[:, 0],
        matches_3d[:, 1],
        matches_3d[:, 2],
        c="green",
        s=100,
        label="Top matches",
    )
    for i, match_idx in enumerate(top_matches):
        img_name = imgNames[match_idx]
        if isinstance(img_name, (bytes, bytearray)):
            img_name = img_name.decode("utf-8")
        ax.text(
            matches_3d[i, 0], matches_3d[i, 1], matches_3d[i, 2], f"{i + 1}. {img_name}"
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend()
    # optional: fig.savefig('feature_space.png')
    plt.close(fig)
    return fig
