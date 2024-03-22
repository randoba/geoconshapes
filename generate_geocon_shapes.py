import cv2
import numpy as np
import random
from pathlib import Path
from random_void_crack import generateConvex, generateCrack

# Definiton of image dimension
IMAGE_DIMENSIONS_WITH_CHANNELS = (299, 299, 3)
IMAGE_DIMENSIONS = IMAGE_DIMENSIONS_WITH_CHANNELS[:2]

# Definiton of samples to generate per class
NUM_SAMPLES = 50

# Definiton of output directory
outpath_base = Path('.') / 'datasets'

# Define the list of possible colors for the shapes (red, green or blue)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

# Define shape and geometric relation concept lists
shapes = {1:"rectangle", 2:"triangle", 3:"circle", 4: "void", 5: "crack"}
simple_shapes = {1:"rectangle", 2:"triangle", 3:"circle"}
concepts = {"alone", "far", "close", "overlap"}

def draw_classes(dataset_type = 'SGS'):
    if dataset_type=='SGS':
        classes = list_combinations_same_type(simple_shapes)
    elif dataset_type=='ACV':
        classes = list_combinations_crackvoid()
    else:
        raise Exception('Not supported dataset type')

    #concept classes
    for concept in concepts:
        #here i combine geometric concept classes with shape combinations
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
                    image = np.ones(IMAGE_DIMENSIONS_WITH_CHANNELS, dtype=np.uint8) * 255
                    ds = drawShapes(concept)
                    outpath_subfolder = f'{concept}_'

                    for i,idx_shape in enumerate(my_class):
                        shape_name = shapes[idx_shape]
                        (x,y) = ds.draw_shape(image, idx_shape)
                        outpath_subfolder = outpath_subfolder + f"{shape_name}"
                        if i < len(my_class)-1:
                            outpath_subfolder = outpath_subfolder + '_'
                    img_gen_success = ds.img_gen_success

                outpath = outpath_base / dataset_type / outpath_subfolder
                outpath.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(outpath / f"{idx_sample}.png"), image)

# Helper function to scale a randomly generated shape (crack or void) to the image coordinates
def to_pixel_coords(relative_coords, shape_pos, shape_new_dim, shape_proto_dim):
    pixel_coords_list = []
    for coords in relative_coords:
        pixel_coords = tuple(pos +  round( new_dim/2 + ( ((rel_coord/2)) / (proto_dim/2)) * new_dim) for pos, rel_coord, new_dim, proto_dim in zip(shape_pos, coords, shape_new_dim, shape_proto_dim))
        pixel_coords_list.append(pixel_coords)
    return pixel_coords_list

class drawShapes():
    def __init__(self, concept):
        self.drawn_shapes=[]
        self.concept = concept
        self.img_gen_success = False

    # Method to place a new shape. In a particular geometric relation concept
    def place_new_shape(self, proto_shape,imageX,imageY):
        x = proto_shape[0]
        y = proto_shape[1]
        sizeX = abs(proto_shape[2]-proto_shape[0])
        sizeY = abs(proto_shape[3]-proto_shape[1])
        trials=0
        img_gen_success = True
        max_trials = 100
        while not all([self.concept_state(proto_shape,r) for r in self.drawn_shapes]) and trials<max_trials:
            x = random.randint(0,imageX-sizeX)
            y = random.randint(0,imageY-sizeY)
            trials = trials+1
        if trials >= max_trials:
            self.img_gen_success = False
        else:
            self.img_gen_success = True
        return (x,y)

    # Method to test whether a given geometric relation concept is fulfilled
    def concept_state(self, shape1, shape2):
        d = self.calc_distance(shape1, shape2)
        overlap = d==-1
        close = 0 < d < 25
        far = d > 75
        if self.concept == 'alone':
            return True
        elif self.concept == 'far':
            return far
        elif self.concept == 'close':
            return close
        elif self.concept == 'overlap':
            return overlap
        else:
            raise Exception('Not supported concept type')

    # Method to draw a square
    def draw_rectangle(self, image):

        # Choose a random size
        sizeX = random.randint(50,100)
        sizeY = random.randint(50,100)

        # Choose a random color
        color = random.choice(colors)
        # Choose a random position on the image
        x = random.randint(0, image.shape[1] - sizeX)
        y = random.randint(0, image.shape[0] - sizeY)

        imageX = image.shape[1]
        imageY = image.shape[0]

        # Place a new shape and check if it is according to the current concept
        proto_shape = (x,y,x+sizeX,y+sizeY)
        (x, y) = self.place_new_shape(proto_shape,imageX,imageY)

        # Draw a rectangle on the image
        cv2.rectangle(image,(x,y),(x+sizeX,y+sizeY),color,-1)

        # Add the shape coordinates to the list of existing shapes
        drawn_shape = (x, y, x + sizeX, y + sizeY)
        self.drawn_shapes.append(drawn_shape)

        return (x,y)

    # Method to draw a triangle
    def draw_triangle(self, image):
        # Choose a random color
        color = random.choice(colors)

        # Choose a random size
        size = random.randint(50,100)

        # Choose a random position on the image
        x = random.randint(0, image.shape[1] - size)
        y = random.randint(0, image.shape[0] - size)

        imageX = image.shape[1]
        imageY = image.shape[0]

        # Place a new shape and check if it is according to the current concept
        proto_shape = (x,y,x+size,y+size)
        (x, y) = self.place_new_shape(proto_shape,imageX,imageY)

        # Draw a triangle on the image
        top_vertex = (x + size // 2, y)
        bottom_left_vertex = (x, y + size)
        bottom_right_vertex = (x + size, y + size)
        triangle = np.array([top_vertex,bottom_left_vertex,bottom_right_vertex])
        cv2.fillPoly(image,[triangle],color)

        # Add the shape coordinates to the list of existing shapes
        drawn_shape = (x, y, x + size, y + size)
        self.drawn_shapes.append(drawn_shape)

        return (x, y)

    # Method to draw a circle
    def draw_circle(self, image):
        # Choose a random color
        color = random.choice(colors)

        # Choose a random size
        size = random.randint(50,100)

        # Choose a random position on the image
        x = random.randint(0,image.shape[1]-size)
        y = random.randint(0,image.shape[0]-size)

        imageX = image.shape[1]
        imageY = image.shape[0]

        # Place a new shape and check if it is according to the current concept
        proto_shape = (x,y,x+size,y+size)
        (x, y) = self.place_new_shape(proto_shape,imageX,imageY)

        # Draw a circle on the image
        center = (x + size // 2, y + size // 2)
        radius = size // 2
        cv2.circle(image, center, radius, color, -1)

        # Add the shape coordinates to the list of existing shapes
        drawn_shape = (x, y, x + size, y + size)
        self.drawn_shapes.append(drawn_shape)
        return (x, y)

    # Method to draw a void
    def draw_random_void(self, image):
        # Choose a random size
        sizeX = random.randint(50,100)
        sizeY = random.randint(50,100)

        # Choose a random color
        color = random.choice(colors)
        # Choose a random position on the image
        x = random.randint(0, image.shape[1] - sizeX)
        y = random.randint(0, image.shape[0] - sizeY)

        imageX = image.shape[1]
        imageY = image.shape[0]

        proto_shape = (x,y,x + sizeX,y + sizeY)
        (x, y) = self.place_new_shape(proto_shape,imageX,imageY)

        vertices, wh = generateConvex(42)

        contour_list = to_pixel_coords(vertices, (x, y),  (sizeX, sizeY), wh )
        contour_array = np.array(contour_list).reshape((-1,1,2))
        cv2.drawContours(image, [contour_array], -1, color, -1)

        # Add the shape coordinates to the list of existing shapes
        drawn_shape = (x,y,x + sizeX,y + sizeY)
        self.drawn_shapes.append(drawn_shape)

        return (x,y)

    # Method to draw a crack
    def draw_random_crack(self, image):
        # Choose a random size
        sizeX = random.randint(75,150)
        sizeY = random.randint(10,15)

        # Choose a random color
        color = random.choice(colors)
        # Choose a random position on the image
        x = random.randint(0, image.shape[1] - sizeX)
        y = random.randint(0, image.shape[0] - sizeY)

        imageX = image.shape[1]
        imageY = image.shape[0]

        # Place a new shape and check if it is according to the current concept
        proto_shape = (x, y, x + sizeX, y + sizeY)
        (x, y) = self.place_new_shape(proto_shape,imageX,imageY)

        vertices, wh = generateCrack(12)

        contour_list = to_pixel_coords(vertices, (x, y),  (sizeX, sizeY), wh )
        contour_array = np.array(contour_list).reshape((-1,1,2))
        cv2.polylines(image, [contour_array], False, color, 2)

        # Add the shape coordinates to the list of existing shapes
        drawn_shape = (x, y, x + sizeX, y + sizeY)
        self.drawn_shapes.append(drawn_shape)

        return (x,y)

    # Define a function to check if two rectangles overlap
    def is_overlap(self, rect1, rect2):
        # A rectangle is defined by its top-left and bottom-right coordinates
        return (rect1[0] < rect2[2] and rect1[2] > rect2[0] and rect1[1] < rect2[3] and rect1[3] > rect2[1]) or \
            (rect2[0] < rect1[2] and rect2[2] > rect1[0] and rect2[1] < rect1[3] and rect2[3] > rect1[1])

    # Calculates the distance of two rectangles
    def calc_distance(self, rect1, rect2):
        if not self.is_overlap(rect1, rect2):
            x1, y1, w1, h1 = rect1[0], rect1[1], rect1[2] - rect1[0], rect1[3] - rect1[1]
            x2, y2, w2, h2 = rect2[0], rect2[1], rect2[2] - rect2[0], rect2[3] - rect2[1]
            dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
            dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
            d = int((dx ** 2 + dy ** 2) ** 0.5)
        else:
            d = -1
        return d

    def draw_shape(self, image, shape):
        match shapes[shape]:
            case "rectangle":
                (x, y) = self.draw_rectangle(image)
            case "triangle":
                (x, y) = self.draw_triangle(image)
            case "circle":
                (x, y) = self.draw_circle(image)
            case "void":
                (x, y) = self.draw_random_void(image)
            case "crack":
                (x, y) = self.draw_random_crack(image)
        return (x,y)


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
    result.append([4,5])
    return result

draw_classes('SGS')
draw_classes('ACV')

