# GeoConShapes

GeoConShapes can generate datasets containing images of various shapes in a specific geometric relationship.

The images containing a single shape are labeled ’alone’, while images with multiple 
shapes can be labeled as ’far’, ’close’ or ’overlap’, depending on the shapes’
relative positions to each other.

The script creates two dataset.
1. The SGS (Simple Geometric Shapes) dataset can contain circles, triangles and rectangles of different sizes and colors. The script creates images with a single shape or with two shapes. The two shapes are both from the same type (e.g. two rectangles, two circles)

2. The ACV (Artificial Crack and Void) dataset can contain images of artificially created versions of a crack and a void. Cracks and voids are possible structural defect types (related to the visual inspection of certain structures).

The code is based on the below (commonly used) python libraries:
OpenCV, Numpy, Random, Pathlib, Matplotlib

The implementation to generate an artificial void (random convex polygon of 42 nodes) and an artificial crack (a random polyline of 10 segments) is  based on the discussion in:
https://stackoverflow.com/questions/6758083/how-to-generate-a-random-convex-polygon
