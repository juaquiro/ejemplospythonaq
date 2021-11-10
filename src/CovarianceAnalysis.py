# ejemplo de uso de la covarianza de una imagen para calcular la mejor elipse de ajuste y su orientacion
# ejemplo sacado de https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
# Computing the Axes or Orientation of a Blob

import numpy as np
import matplotlib.pyplot as plt
import imageio

# -------------------------------------------------
# 1. Using Covariance matrix and eigens
# -------------------------------------------------

# Read the image in grey-scale. We want the indexes of the white pixels to find the axes of the blob.
MAXGV: int=255 # MaxGV for normalization
img = imageio.imread('oval.png') / MAXGV #image is [0,1]
y, x = np.nonzero(img)

# Subtract mean from each dimension. We now have our 2xm matrix.
x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])

# 3 & 4) Covariance matrix and its eigenvectors and eigenvalues
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)

# 5) Sort eigenvalues in decreasing order (we only have 2 values)
sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]

# 6) Plot the principal components. The larger eigenvector is plotted in red and drawn twice as long as the smaller eigenvector in blue.
scale = 20
plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
         [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
plt.plot([x_v2 * -scale, x_v2 * scale],
         [y_v2 * -scale, y_v2 * scale], color='blue')
plt.plot(x, y, 'k.')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
# if not block=False execution is halted until we close the fig. see  https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
plt.show(block=False)

# 7) Bonus! We vertically-align the blob based on the major axis via a linear transformation.
# An anti-clockwise rotating transformation matrix has the general form: [cosθ -sinθ; sinθ cosθ].
theta = np.arctan((x_v1) / (y_v1))
rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
transformed_mat = rotation_mat @ coords
# plot the transformed blob, alternativamente podriamos poner x_transformed y_transformed=transformed_mat
x_transformed = transformed_mat[0]  # first row
y_transformed = transformed_mat[1]  # second row
plt.plot(x_transformed, y_transformed, 'g.')
plt.title('Orientation from covariance. Close figure to continue')
plt.show()

# -------------------------------------------------
# 2. Using Raw image moments
# -------------------------------------------------

# We can obtain the same axes and orientation of a blob with raw image moments and central moments.
#  Special thanks to this stack overflow answer. http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python

# moments
# here we are assuming that data is normalized [0,1]
def raw_moment(data, i_order, j_order):
  nrows, ncols = data.shape
  y_indices, x_indicies = np.mgrid[:nrows, :ncols]
  return (data * x_indicies**i_order * y_indices**j_order).sum()

# Now we can derive the second order central moments and construct its covariance matrix:
# here we are assuming that data is normalized [0,1]
def moments_cov(data):
  data_sum = data.sum()
  m10 = raw_moment(data, 1, 0)
  m01 = raw_moment(data, 0, 1)
  x_centroid = m10 / data_sum
  y_centroid = m01 / data_sum
  u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
  u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
  u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
  cov = np.array([[u20, u11], [u11, u02]])
  return cov

#if the image is not normalized [0,1]
cov = moments_cov(img)
evals, evecs = np.linalg.eig(cov)

# Given the covariance matrix, finding the axes and re-aligning the blob continues with steps 4 to 7 from the first method. As expected, the transformed plot looks the same:

# 5) Sort eigenvalues in decreasing order (we only have 2 values)
sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]

# 6) Plot the principal components. The larger eigenvector is plotted in red and drawn twice as long as the smaller eigenvector in blue.
scale = 20
plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
         [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
plt.plot([x_v2 * -scale, x_v2 * scale],
         [y_v2 * -scale, y_v2 * scale], color='blue')
plt.plot(x, y, 'k.')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
# if not block=False execution is halted until we close the fig. see  https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
plt.show(block=False)

# 7) Bonus! We vertically-align the blob based on the major axis via a linear transformation.
# An anti-clockwise rotating transformation matrix has the general form: [cosθ -sinθ; sinθ cosθ].
theta = np.arctan((x_v1) / (y_v1))
rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
transformed_mat = rotation_mat @ coords
# plot the transformed blob, alternativamente podriamos poner x_transformed y_transformed=transformed_mat
x_transformed = transformed_mat[0]  # first row
y_transformed = transformed_mat[1]  # second row
plt.plot(x_transformed, y_transformed, 'g.')
plt.title('Orientation from moments. Close figure to continue')
plt.show()
