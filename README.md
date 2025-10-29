# GeoConShapes

**GeoConShapes** is a synthetic dataset of 2D images designed to support research on spatial relation learning. It includes labeled examples of object pairs arranged according to basic spatial concepts such as `alone`, `close`, `far`, and `overlap`.

The dataset is organized into three main categories:
- **SGS** – Simple Geometric Shapes (e.g., rectangles, triangles, ellipses)
- **ACV** – Artificial Cracks and Voids (domain-specific anomalies)
- **CSI** – Complex Semantic Instances (segmented real-world objects like cats, birds, plants)

Each image is 299×299 pixels and contains one or two objects. Labels are derived from the spatial arrangement of the objects.

---

## 🔧 Installation

This project does not require a dedicated environment. You can run it in your existing Python setup.

Make sure you have the following packages installed:

```bash
pip install numpy opencv-python matplotlib shapely
```

---

## ▶️ Usage

To generate the dataset, run:

```bash
python generate_geocon_shapes.py
```

You can modify parameters in the script to control:
- object types and sizes
- spatial relation types
- number of samples
- output folder structure

The script uses fixed random seeds to ensure reproducibility.

---

## 📁 Dataset Structure

Each image is stored in a folder named according to the spatial relation and object types, e.g.:

```
SGS/close_rectangle_rectangle/
ACV/overlap_void_crack/
CSI_CB/far_cat_bird/
```

Each folder contains 100 images named `0.png` to `99.png`.


## 🙏 Acknowledgements

The implementation to generate an artificial void (random convex polygon of 42 nodes) and an artificial crack (a random polyline of 10 segments) is based on the discussion in:  
https://stackoverflow.com/questions/6758083/how-to-generate-a-random-convex-polygon
