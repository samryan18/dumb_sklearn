# dumb scikit-learn

Based on the popular [scikit-learn](https://github.com/scikit-learn/scikit-learn) python package...just a little dumber.

Wanted to learn some basics. Inspired by [Dumb Starbucks](https://www.youtube.com/watch?v=h0TRpGP8yH4) from Nathan for You :)

### Installation
```bash
$ git clone git@github.com:samryan18/dumb_sklearn.git
$ pip install dumb_sklearn
```

# Models
## 1. Neural Networks
[Implemented](https://github.com/samryan18/matlab-nn) in matlab. Sorry.

## 2. Principal Component Analysis (`dumb_sklearn.PCA`)

```python
from dumb_sklearn import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people


def plot_gallery(images, titles, h, w, n_row=5, n_col=5):
    """ function adapted from sklearn documentation """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


faces = fetch_lfw_people(min_faces_per_person=60)

_, h, w = faces.images.shape
X = faces.data
y = faces.target
target_names = faces.target_names
n_components = 200
n_row = 2
n_col = 7

pca = PCA(n_components=n_components, standardize=True)
X_transform = pca.fit_transform(X)
eigenfaces = pca.components.reshape((n_components, h, w))

print("ORIGINAL FACES:")
real_titles = [f"original {i}" for i in range(eigenfaces.shape[0])]
real_faces = faces.images
plot_gallery(real_faces, real_titles, h, w, n_row=n_row, n_col=n_col)

print("EIGENFACES:")
eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w, n_row=n_row, n_col=n_col)

print("AVERAGE FACE:")
plot_gallery(pca._mean.reshape(1, h, w), ["mean"], h, w, n_row=1, n_col=1)

print("STANDARD DEV FACE:")
plot_gallery(pca._std.reshape(1, h, w), ["stddev"], h, w, n_row=1, n_col=1)

print("RECON FACES:")
recon = pca.inverse_transform(X_transform)
recon = recon.reshape((faces.images.shape[0], h, w))
recon_titles = [f"recon {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(recon, recon_titles, h, w, n_row=n_row, n_col=n_col)
```

### Output
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
