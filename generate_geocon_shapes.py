import cv2
import numpy as np
import random
from pathlib import Path

import shapely
from shapely import LineString, Polygon

from random_void_crack import generateConvex, generateCrack, wh_vertices

from PIL import Image

random.seed(42)

# Definiton of image dimension
IMAGE_DIMENSIONS_WITH_CHANNELS = (299, 299, 3)
IMAGE_DIMENSIONS = IMAGE_DIMENSIONS_WITH_CHANNELS[:2]

# Definiton of samples to generate per class
NUM_SAMPLES = 100
max_new_shape_trials = 500

sizeX_min, sizeX_max = 80, 100
sizeY_min, sizeY_max = 80, 100

#AR_SGS_SHAPE1 = 1
AR_SGS_SHAPE2 = 1
#AR_ACV_VOID = 1
AR_ACV_CRACK = 0.1

# Definiton of output directory
outpath_base = Path('.') / 'datasets' / f'datasets_ipc_{NUM_SAMPLES}_seed_42_complex_2'
images_dir = 'images'

# Define the list of possible colors for the shapes (red, green or blue)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

#Define additional polygon shapes
trapezoid = Polygon([(-0.5, -0.5), (-1.0, 0.5), (1, 0.5), (0.5, -0.5)])
parallelogram = Polygon([(-1.5, -0.5), (0.5, -0.5), (1.5, 0.5), (-0.5, 0.5)])

# Define shape and geometric relation concept lists
simple_shapes = {1:"rectangle", 2:"triangle", 3:"ellipse"}
crack_void_shapes = {4: "void", 5: "crack"}
complex_shapes = {6: "cat", 7: "bird", 8: "sheep", 9: "plant", 10: "horse", 11: "dog"}
polygon_shapes = {12: "trapezoid", 13: "parallelogram"}

shapes = simple_shapes | crack_void_shapes | complex_shapes | polygon_shapes

concepts = ["alone", "far", "close", "overlap"]

class GeoConShape():
    def __init__(self, type, bounding_box, vertices):
        self.type = type
        self.bounding_box = bounding_box
        self.vertices = vertices

def draw_classes(dataset_type = 'SGS', outpath_draw = outpath_base / images_dir):
    if dataset_type =='SGS':
        classes = list_combinations_same_type(simple_shapes | polygon_shapes)
    elif dataset_type =='ACV':
        classes = list_combinations_crackvoid()
    elif dataset_type in ('CSI_CB', 'CSI_CB_SIL'):
        classes = list_combinations_csi_cat_bird()
    elif dataset_type in ('CSI_SP', 'CSI_SP_SIL'):
        classes = list_combinations_csi_sheep_plant()
    elif dataset_type in ('CSI_HD', 'CSI_HD_SIL'):
        classes = list_combinations_csi_horse_dog()
    else:
        raise Exception('Not supported dataset type')

    #concept classes
    for concept in concepts:
        #combine geometric concept classes with shape combinations
        for idx_class, my_class in enumerate(classes):
            #alone should be combined only with single shape classes
            if concept == "alone":
                if len(my_class) > 1:
                    continue
            # two or more shapes should not be combined with alone
            if concept != "alone":
                if len(my_class) == 1:
                    continue

            for idx_sample in range(NUM_SAMPLES):
                img_gen_success = False
                while not img_gen_success:
                    if '_SIL' in dataset_type:
                        ds = drawShapes(concept, only_silhouette=True)
                    else:
                        ds = drawShapes(concept, only_silhouette=False)

                    outpath_subfolder = f'{concept}_'

                    for i,idx_shape in enumerate(my_class):
                        shape_name = shapes[idx_shape]
                        ds.draw_new_shape(idx_shape)
                        outpath_subfolder = outpath_subfolder + f"{shape_name}"
                        if i < len(my_class)-1:
                            outpath_subfolder = outpath_subfolder + '_'
                    img_gen_success = ds.img_gen_success

                outpath = outpath_draw / dataset_type / outpath_subfolder
                outpath.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(outpath / f"{idx_sample}.png"), ds.image)

# Helper function to scale a randomly generated shape (crack or void) to the image coordinates
def to_pixel_coords(relative_coords, shape_pos, shape_new_dim, shape_proto_dim):
    pixel_coords_list = []
    for coords in relative_coords:
        pixel_coords = tuple(pos +  round( new_dim/2 + ( ((rel_coord/2)) / (proto_dim/2)) * new_dim) for pos, rel_coord, new_dim, proto_dim in zip(shape_pos, coords, shape_new_dim, shape_proto_dim))
        pixel_coords_list.append(pixel_coords)
    return pixel_coords_list

class drawShapes():
    def __init__(self, concept, only_silhouette = False):
        self.drawn_shapes = []
        self.drawn_shape_colors = []
        self.concept = concept
        self.only_silhouette = only_silhouette
        self.img_gen_success = False
        self.imageX,self.imageY = 299,299
        self.image = np.ones(IMAGE_DIMENSIONS_WITH_CHANNELS, dtype=np.uint8) * 255
        self.nth_shape = 0

    def is_overlap_rectangles(self, rect1, rect2):
    # A rectangle is defined by its top-left and bottom-right coordinates
        return (rect1[0] < rect2[2] and rect1[2] > rect2[0] and rect1[1] < rect2[3] and rect1[3] > rect2[1]) or \
               (rect2[0] < rect1[2] and rect2[2] > rect1[0] and rect2[1] < rect1[3] and rect2[3] > rect1[1])

    def circle_distance(self, circle1, circle2):
        x1, y1, r1, _ = circle1
        x2, y2, r2, _ = circle2
        d_centers = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        if d_centers >= r1 + r2:
            d = d_centers - (r1 + r2)
        else:
            d = -1
        return d

    def is_overlap_poly(self, v1, v2):
        poly1 = shapely.make_valid(Polygon(v1))
        poly2 = shapely.make_valid(Polygon(v2))
        return poly1.intersects(poly2)

    def poly_distance(self, v1, v2):
        poly1 = shapely.make_valid(Polygon(v1))
        poly2 = shapely.make_valid(Polygon(v2))
        return poly1.distance(poly2)

    def is_overlap_poly_segment(self, v1, v2):
        poly1 = shapely.make_valid(Polygon(v1))
        ls2 = shapely.make_valid(Polygon(v2))
        return poly1.intersects(ls2)

    def poly_segment_distance(self, v1, v2):
        poly1 = Polygon(v1)
        ls2 = LineString(v2)
        return poly1.distance(ls2)

    def bb_to_cirle(self, bb):
        x,y,sizeX,sizeY = bb
        size = min(sizeX, sizeY)
        center = (x + size // 2, y + size // 2)
        radius = size // 2
        circle = (center[0], center[1], radius, size)
        return circle

    def bounding_box(self, vertices):
        vx, vy = zip(*vertices)
        return [(min(vx), min(vy)), (max(vx), max(vy))]
    def random_shape_placement(self, idx_shape = None):
        sizeX = random.randint(sizeX_min, sizeX_max)
        sizeY = random.randint(sizeX_min, sizeX_max)
        match shapes[idx_shape]:
            case "triangle" | "rectangle" | "ellipse" | "trapezoid" | "parallelogram":
                if self.nth_shape == 0:
                    pass
                else:
                    sizeY = int(sizeX * AR_SGS_SHAPE2)
            case "cat" | "bird" | "sheep" | "plant" | "horse" | "dog":
                if self.nth_shape == 0:
                    pass
                else:
                    sizeY = int(sizeX * AR_SGS_SHAPE2)
            case "circle":
                pass
            case "void":
                pass
            case "crack":
                if self.nth_shape == 0:
                    sizeY = int(sizeX * AR_ACV_CRACK)
                else:
                    sizeY = int(sizeX * AR_SGS_SHAPE2)

        x = random.randint(0, self.imageX - sizeX)
        y = random.randint(0, self.imageX - sizeY)
        return (x,y,sizeX,sizeY)
        
    def draw_new_shape(self, idx_shape):
        shape_placement, new_poly = self.place_new_shape(idx_shape)
        self.draw_shape(idx_shape, new_poly, shape_placement)
        self.nth_shape = self.nth_shape + 1
        
    def place_new_shape(self, shape = None):
        trials=0
        max_trials = max_new_shape_trials
        while True:
            trials = trials + 1

            shape_placement = self.random_shape_placement(shape)
            new_poly = self.create_shape(shape, shape_placement)
            proto_shape = GeoConShape(shape, shape_placement, new_poly)

            if all([self.concept_state(proto_shape, p) for p in self.drawn_shapes]) or trials >= max_trials:
                self.drawn_shapes.append(proto_shape)
                break

        if trials >= max_trials:
            self.img_gen_success = False
        else:
            self.img_gen_success = True

        return shape_placement, new_poly
        
    def create_shape(self, idx_shape, shape_placement):
        match shapes[idx_shape]:
            case "triangle":
                v = self.create_triangle(shape_placement)
            case "rectangle":
                v = self.create_rectangle(shape_placement)
            case "circle":
                v = self.create_circle(shape_placement)
            case "ellipse":
                v = self.create_ellipse(shape_placement)
            case "void":
                v = self.create_random_void(shape_placement)
            case "crack":
                v = self.create_random_crack(shape_placement)
            case "cat" | "bird" | "sheep" | "plant" | "horse" | "dog":
                v = self.create_random_image_box_poly(shape_placement, shapes[idx_shape])
            case "trapezoid" | "parallelogram":
                v = self.create_random_poly(shape_placement, shapes[idx_shape])
        return v

    def draw_shape(self, idx_shape, shape_vertices, shape_placement):
        shape_name = shapes[idx_shape]
        
        # Choose a random color
        color = random.choice(list(set(colors) - set(self.drawn_shape_colors)))
        self.drawn_shape_colors.append(color)
        
        if shape_name == "crack":
            self.draw_polyline(shape_vertices, color)
        elif shape_name == "circle":
            self.draw_circle(shape_placement, color)
        elif shape_name in complex_shapes.values():
            if self.only_silhouette:
                self.draw_polygon(shape_vertices, color)
            else:
                self.draw_in_image_box_poly(shape_vertices)
        else:
            self.draw_polygon(shape_vertices, color)

    def concept_state(self, shape1, shape2):
        if shapes[shape1.type] == 'circle':
            d = self.circle_distance(self.bb_to_cirle(shape1.bounding_box), self.bb_to_cirle(shape2.bounding_box))
        elif shapes[shape1.type] == 'void':
            d = self.poly_segment_distance(shape1.vertices, shape2.vertices)
            is_overlap = self.is_overlap_poly_segment(shape1.vertices, shape2.vertices)
            if is_overlap:
                d = -1
        else:
            d = self.poly_distance(shape1.vertices,shape2.vertices)
            is_overlap = self.is_overlap_poly(shape1.vertices,shape2.vertices)
            if is_overlap:
                d = -1

        overlap = (d == -1)
        close = (5 < d < 25)
        touch = (d >= 0) and (d <= 1) #touch = (d == 0)
        far = (d > 75)

        if self.concept == 'alone':
            return True
        elif self.concept == 'far':
            return far
        elif self.concept == 'close':
            return close
        elif self.concept == 'overlap':
            return overlap
        elif self.concept == 'touch':
            return touch
        else:
            print('check concept list!')
            return True

    def create_rectangle(self, shape_placement):
        (x, y, sizeX, sizeY) = shape_placement
        v_x = [x, x + sizeX, x + sizeX, x]
        v_y = [y, y        , y + sizeY, y + sizeY]
        vertices = list(zip(v_x, v_y))
        return vertices

    # Define a function to draw a triangle
    def create_triangle(self, shape_placement):
        (x,y,sizeX,sizeY) = shape_placement

        # Draw a triangle on the image using the cv2.fillPoly function
        # The triangle is defined by three points: the top vertex and the two bottom vertices
        top_vertex = (x + sizeX // 2, y)
        bottom_left_vertex = (x, y + sizeY)
        bottom_right_vertex = (x + sizeX, y + sizeY)

        v_x = [top_vertex[0], bottom_left_vertex[0], bottom_right_vertex[0]]
        v_y = [top_vertex[1], bottom_left_vertex[1], bottom_right_vertex[1]]
        vertices = list(zip(v_x, v_y))
        return vertices

    def create_ellipse(self, shape_placement):
        from matplotlib.patches import Ellipse
        x, y, sizeX, sizeY = shape_placement
        center = (x + sizeX // 2, y + sizeY // 2)
        width, height, angle = sizeX, sizeY, 0.0
        ellipse = Ellipse(center, width, height)
        vertices = ellipse.get_verts()  # get the vertices from the ellipse object
        return vertices
        
    def create_circle(self, shape_placement):
        (x,y,sizeX,sizeY) = shape_placement
        size = min(sizeX,sizeY)
        #cx, cy, r, d = self.bb_to_cirle(shape_placement)
        #v_x = [x, x + size, x + size, x]
        #v_y = [y, y        , y + size, y + size]
        #vertices = list(zip(v_x, v_y))
        return shape_placement

    def create_random_void(self, shape_placement):

        (x, y, sizeX, sizeY) = shape_placement
        vertices, wh = generateConvex(42)

        contour_list = to_pixel_coords(vertices, (x, y),  (sizeX, sizeY), wh)
        contour_vertices = np.array(contour_list)
        contour_array = np.array(contour_list).reshape((-1,1,2))

        v_x = [x, x + sizeX, x + sizeX, x]
        v_y = [y, y, y + sizeY, y + sizeY]
        vdc = list(zip(v_x, v_y))

        # Add the coordinates to the list of existing shapes
        return contour_vertices

    def create_random_crack(self, shape_placement):

        (x, y, sizeX, sizeY) = shape_placement
        vertices, wh = generateCrack(12)

        contour_list = to_pixel_coords(vertices, (x, y),  (sizeX, sizeY), wh)
        contour_vertices = np.array(contour_list)

        return contour_vertices

    def create_random_poly(self, shape_placement, poly_type):

        (x, y, sizeX, sizeY) = shape_placement
        if poly_type == "trapezoid":
            poly = trapezoid
        elif poly_type == "parallelogram":
            poly = parallelogram

        vertices = list(poly.exterior.coords)
        wh = wh_vertices(vertices)

        contour_list = to_pixel_coords(vertices, (x, y),  (sizeX, sizeY), wh)
        contour_vertices = np.array(contour_list)

        return contour_vertices

    def create_random_image_box(self, shape_placement):
        (x, y, sizeX, sizeY) = shape_placement
        v_x = [x, x + sizeX, x + sizeX, x]
        v_y = [y, y        , y + sizeY, y + sizeY]
        vertices = list(zip(v_x, v_y))
        return vertices

    def create_random_image_box_poly(self, shape_placement, image_class):
        (x, y, sizeX, sizeY) = shape_placement

        img_dir = f'segmented_objects/{image_class}/'
        image_path = random.choice(list(Path(img_dir).glob('*.png')))
        contour_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        contours, hierarchy = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        ext_contour = contours[0].squeeze(axis=1)
        wh = wh_vertices(ext_contour)

        #the vertices of void originally used 0,0 as center of the shape, so we need to shift here too
        contour_vertices = ext_contour - tuple(ti/2 for ti in wh)

        contour_list = to_pixel_coords(contour_vertices, (x, y),  (sizeX, sizeY), wh)
        contour_vertices = np.array(contour_list)

        v_x = [x, x + sizeX, x + sizeX, x]
        v_y = [y, y, y + sizeY, y + sizeY]
        vdc = list(zip(v_x, v_y))

        # Add the coordinates to the list of existing shapes
        self.image_path = image_path
        return contour_vertices

    def draw_polygon(self, v_poly, color):
        # Draw a polygon on the image using the cv2.rectangle function
        #https://stackoverflow.com/questions/67837617/cant-parse-pt-sequence-item-with-index-0-has-a-wrong-type-pointpolygon-er
        #https://www.geeksforgeeks.org/python-convert-list-of-tuples-to-list-of-list/
        contour_array = np.array(v_poly).reshape((-1,1,2))
        v_poly_list = np.array(contour_array).tolist() #.astype(np.uint8)
        v_poly_list_int = np.array([list(map(np.int32, lst)) for lst in v_poly_list])
        cv2.fillPoly(self.image, [v_poly_list_int], color)

    def draw_polyline(self, v_poly, color):
        contour_array = np.array(v_poly).reshape((-1,1,2))
        v_poly = contour_array
        # Draw a polygon on the image using the cv2.rectangle function
        cv2.polylines(self.image, [v_poly], False, color, 2)

    def draw_circle(self, shape_placement, color):
        (x,y,sizeX,sizeY) = shape_placement
        # Draw a circle on the image using the cv2.circle function
        # The circle is defined by its center and radius
        size = min(sizeX, sizeY)
        center = (x + size // 2, y + size // 2)
        radius = size // 2
        cv2.circle(self.image, center, radius, color, -1)

    def draw_in_image_box(self, shape_placement, image_class):
        #https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
        (bb0, bb1) = self.bounding_box(shape_placement)

        img_dir = f'segmented_objects/{image_class}/'
        image = random.choice(list(Path(img_dir).glob('*.png')))

        overlay_image = cv2.imread(str(image),-1)
        x_offset= bb0[0]
        y_offset= bb0[1]
        sizeX = bb1[0] - bb0[0]
        sizeY = bb1[1] - bb0[1]

        y1, y2 = y_offset, y_offset + sizeY
        x1, x2 = x_offset, x_offset + sizeX

        overlay_image_resize = cv2.resize(overlay_image, [sizeX, sizeY])

        alpha_s = overlay_image_resize[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            self.image[y1:y2, x1:x2, c] = (alpha_s * overlay_image_resize[:, :, c] + alpha_l * self.image[y1:y2, x1:x2, c])

    def draw_in_image_box_poly(self, shape_placement):
        # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
        (bb0, bb1) = self.bounding_box(shape_placement)

        #img_dir = f'segmented_objects/{image_class}/'
        #image = random.choice(list(Path(self.image_path).glob('*.png')))

        overlay_image = cv2.imread(str(self.image_path), -1)
        x_offset = bb0[0]
        y_offset = bb0[1]
        sizeX = bb1[0] - bb0[0]
        sizeY = bb1[1] - bb0[1]

        y1, y2 = y_offset, y_offset + sizeY
        x1, x2 = x_offset, x_offset + sizeX

        overlay_image_resize = cv2.resize(overlay_image, [sizeX, sizeY])

        alpha_s = overlay_image_resize[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            self.image[y1:y2, x1:x2, c] = (alpha_s * overlay_image_resize[:, :, c] + alpha_l * self.image[y1:y2, x1:x2, c])


# List combinations of shapes and relationship concepts for SGS
def list_combinations_same_type(shapes):
    result = []
    for shape in shapes:
        result.append([shape])
        result.append([shape,shape])
    return result

# List relationship concepts for cracks and void for ACV
def list_combinations_crackvoid():
    result = []
    result.append([4])
    result.append([5])
    result.append([4,5])
    return result

def list_combinations_csi_cat_bird():
    result = []
    result.append([6])
    result.append([7])
    result.append([6,7])
    return result

def list_combinations_csi_sheep_plant():
    result = []
    result.append([8])
    result.append([9])
    result.append([8,9])
    return result

def list_combinations_csi_horse_dog():
    result = []
    result.append([10])
    result.append([11])
    result.append([10,11])
    return result


for AR_SGS_SHAPE2 in np.arange(0.1,1.05,0.1):
#for AR_SGS_SHAPE2 in np.arange(1.0, 1.05, 0.1):
    #https://www.random.org/integers/?num=5&min=10000&max=99999&col=5&base=10&format=html&rnd=new
    outpath = outpath_base / f'AR_{AR_SGS_SHAPE2:.1f}'.replace(".", "d")
    random.seed(47967)
    draw_classes('SGS', outpath / images_dir)
    random.seed(13490)
    draw_classes('ACV', outpath / images_dir)
    random.seed(74460)
    draw_classes('CSI_CB', outpath / images_dir)
    random.seed(74460)
    draw_classes('CSI_CB_SIL', outpath / images_dir)
    random.seed(76457)
    draw_classes('CSI_SP', outpath / images_dir)
    random.seed(76457)
    draw_classes('CSI_SP_SIL', outpath / images_dir)
    random.seed(69512)
    draw_classes('CSI_HD', outpath / images_dir)
    random.seed(69512)
    draw_classes('CSI_HD_SIL', outpath / images_dir)


