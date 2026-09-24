# torch must be imported before faiss: on macOS both wheels bundle their own libomp, and
# loading faiss's first makes the first torch forward pass segfault.
import torch  # noqa: F401

from .clustering import make_clusters
