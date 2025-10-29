import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

rng = np.random.default_rng(42)

def wh_vertices(vertices):
    xx, yy = zip(*vertices)
    min_x = min(xx);
    min_y = min(yy);
    max_x = max(xx);
    max_y = max(yy)

    w = max_x-min_x
    h = max_y-min_y
    return (w,h)

def generateConvex(n):
    # https://stackoverflow.com/questions/6758083/how-to-generate-a-random-convex-polygon
    # initialise random coordinates
    XY_rand = np.sort(rng.random((n, 2)), axis=0)

    # divide the interior points into two chains
    rand_bool = rng.choice([True, False], n - 2)
    pos, neg = XY_rand[1:-1][rand_bool], XY_rand[1:-1][~rand_bool]

    pos = np.vstack((XY_rand[0], pos, XY_rand[-1]))
    neg = np.vstack((XY_rand[0], neg, XY_rand[-1]))
    vertices = np.vstack((pos[1:] - pos[:-1], neg[:-1] - neg[1:]))

    # randomly combine x and y and sort by polar angle
    rng.shuffle(vertices[:, 1])
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

    # arrange points end to end to form a polygon
    vertices = np.cumsum(vertices, axis=0)

    # center around the origin
    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] -= (x_max + np.min(vertices[:, 0])) / 2
    vertices[:, 1] -= (y_max + np.min(vertices[:, 1])) / 2

    return vertices, wh_vertices(vertices)

def generateCrack(n):
    X_rand = np.sort(rng.random(n)) / 10
    Y_rand = rng.random(n)
    XY_rand2 = np.array([(a, b) for a, b in zip(X_rand, Y_rand)])
    vertices = XY_rand2

    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] -= (x_max + np.min(vertices[:, 0])) / 2
    vertices[:, 1] -= (y_max + np.min(vertices[:, 1])) / 2

    return vertices, wh_vertices(vertices)

if __name__ == '__main__':
    n = 42
    vertices, wh = generateConvex(n)
    p = Polygon(vertices)
    fig, ax = plt.subplots()

    ax.add_patch(p)
    plt.title(f'{n}-sided convex polygon')
    plt.axis('equal')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.show()

    n = 12
    vertices, wh = generateCrack(n)
    x,y = list(zip(*vertices))
    x2 = [v * 10 for v in x]
    y2 = [v / 10 for v in y]
    plt.plot(x2,y2)
    plt.title(f'{n-2}-segment polyline')
    plt.axis('equal')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.show()