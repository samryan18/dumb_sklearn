# dumb scikit-learn
<img align="left" width="225" src="assets/logo.png">

Work in progress. Based on the popular [scikit-learn](https://github.com/scikit-learn/scikit-learn) python package...just a little dumber.

See: [Dumb Starbucks](https://www.youtube.com/watch?v=h0TRpGP8yH4).

Refreshing on the basics.

### Installation
```bash
$ git clone git@github.com:samryan18/dumb_sklearn.git
$ pip install dumb_sklearn
```

# Models
## 1. Neural Networks
[Implemented](https://github.com/samryan18/matlab-nn) in matlab. Sorry.

## 2. Principal Component Analysis (`dumb_sklearn.PCA`)
The API for this model is very similar to the one in `sklearn`.

### Basic usage:
```python
from dumb_sklearn import PCA

X = get_some_data()
pca = PCA(n_components=n_components, standardize=True)
X_transform = pca.fit_transform(X)
X_recon = pca.inverse_transform(X_transform)

```

### Example 1: Eigenfaces
<details>
<summary>In</summary>


```python
from sklearn.datasets import fetch_lfw_people

from dumb_sklearn import PCA
from dumb_sklearn.utils import plot_gallery

faces = fetch_lfw_people(min_faces_per_person=60)

_, h, w = faces.images.shape
X, y = faces.data, faces.target
target_names = faces.target_names
n_components = 200

pca = PCA(n_components=n_components, standardize=True)
X_transform = pca.fit_transform(X)
eigenfaces = pca.components.reshape((n_components, h, w))

print("ORIGINAL FACES:")
real_titles = [f"original {i}" for i in range(eigenfaces.shape[0])]
real_faces = faces.images
plot_gallery(real_faces, real_titles, h, w, n_row=2, n_col=7)

print("EIGENFACES:")
eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w, n_row=2, n_col=7)

print("AVERAGE FACE:")
plot_gallery(pca._mean.reshape(1, h, w), ["mean"], h, w, n_row=1, n_col=1)

print("STANDARD DEV FACE:")
plot_gallery(pca._std.reshape(1, h, w), ["stddev"], h, w, n_row=1, n_col=1)

print("RECON FACES:")
recon = pca.inverse_transform(X_transform)
recon = recon.reshape((faces.images.shape[0], h, w))
recon_titles = [f"recon {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(recon, recon_titles, h, w, n_row=2, n_col=7)

```

</details>

<details>
<summary>Out</summary>

##### ORIGINAL FACES
![ORIGINAL FACES](/assets/pca_original_faces.png)

##### EIGENFACES
![EIGENFACES](/assets/pca_eigenfaces.png)

##### AVERAGE FACE
![AVERAGE FACE](/assets/pca_mean_face.png)

##### STANDARD DEV FACE
![STANDARD DEV FACE](/assets/pca_std_face.png)

##### RECON FACES
![RECON FACES](/assets/pca_recon_faces.png)

</details>
