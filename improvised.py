import numpy as np
from scipy import ndimage
from lbpface.datasets import load_orl
from lbpface.features import LBPDescriptor
from lbpface.distances import pairwise_distances
from lbpface.classify import recognition_rate

fs = load_orl("data/ORL")
desc = LBPDescriptor.orl()                          # LBP(16,2), 30x37 windows
is_gallery = (np.arange(len(fs.images)) % 10) < 5   # images 1-5 of each subject
gal, prb = fs.images[is_gallery], fs.images[~is_gallery]
gl, pl = fs.subjects[is_gallery], fs.subjects[~is_gallery]

# Remove large-scale lighting variations.
#  try to find the optimal, Preserving more local details
def normalize_illumination(img):
    img = np.asarray(img, dtype=float)
    # Remove broad illumination
    val = img - ndimage.gaussian_filter(img, sigma=9) + 256
    # print("min/max", val.min(), val.max())
    return val

def rotate(img, deg):
    return ndimage.rotate(img.astype(np.float32), deg, reshape=False, order=1, mode="nearest")

rotation_views = (-20, 0, 20)
gallery_views = [normalize_illumination(rotate(img, angle))
                 for angle in rotation_views for img in gal]
G = desc.transform(gallery_views)
gallery_labels = np.tile(gl, len(rotation_views))

# transform img under normalized illumination
def score(probe_images):
    P = desc.transform([normalize_illumination(img) for img in probe_images])
    # compute recognition rate over the gallery and probe
    return recognition_rate(pairwise_distances(P, G, "chi2"), gallery_labels, pl)

def illum_ramp(img, s):          # darker on the left, brighter on the right
    return np.clip(img * (1 + s * np.linspace(-1, 1, img.shape[1])), 0, 255)

print("baseline           ", score(prb))
print("uniform brightness ", score([0.5 * i + 20 for i in prb]))
for s in [0.2, 0.4, 0.6, 0.8]: print("lighting ramp", s, score([illum_ramp(i, s) for i in prb]))
for d in [5, 10, 20]:          print("rotate", d, "deg", score([rotate(i, d) for i in prb]))
