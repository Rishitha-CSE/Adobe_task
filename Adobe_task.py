#1. Reading and Visualizing Polylines
#Reading CSV Files:
import numpy as np

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

#Visualizing Shapes:
import matplotlib.pyplot as plt
import numpy as np

def plot(paths_XYs):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()


# Regularization of Curves
#Line Fitting:
from sklearn.linear_model import LinearRegression

def fit_line(points):
    model = LinearRegression()
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model.fit(X, y)
    return model.coef_, model.intercept_
#Circle Fitting:
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import numpy as np

def fit_circle(points):
    def objective(params):
        x0, y0, r = params
        return np.sum((np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r) ** 2)

    result = minimize(objective, x0=[0, 0, 1], bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)])
    x0, y0, r = result.x
    return (x0, y0, r)
#Ellipse Fitting:
import numpy as np
from scipy.optimize import least_squares

def fit_ellipse(points):
    x = points[:, 0]
    y = points[:, 1]

    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T

    S = np.dot(D.T, D)
    C = np.zeros((6, 6))
    C[0, 0] = S[0, 0]
    C[0, 1] = S[0, 1]
    C[0, 2] = S[0, 2]
    C[1, 0] = S[1, 0]
    C[1, 1] = S[1, 1]
    C[1, 2] = S[1, 2]
    C[2, 0] = S[2, 0]
    C[2, 1] = S[2, 1]
    C[2, 2] = S[2, 2]
    C[3, 3] = S[3, 3]
    C[3, 4] = S[3, 4]
    C[4, 3] = S[4, 3]
    C[4, 4] = S[4, 4]
    C[5, 5] = S[5, 5]

    def cost_function(params):
        A, B, C, D, E, F = params
        ellipse = A * x**2 + B * x * y + C * y**2 + D * x + E * y + F
        return ellipse

    result = least_squares(cost_function, x0=[1, 0, 1, 0, 0, -1])
    A, B, C, D, E, F = result.x

    return A, B, C, D, E, F

def plot_ellipse(A, B, C, D, E, F):
    import matplotlib.pyplot as plt

    t = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    ellipse_points = np.vstack([x, y]).T

    ellipse = A * ellipse_points[:, 0]**2 + B * ellipse_points[:, 0] * ellipse_points[:, 1] + C * ellipse_points[:, 1]**2 + D * ellipse_points[:, 0] + E * ellipse_points[:, 1] + F

    plt.figure()
    plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r--')
    plt.title("Fitted Ellipse")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

#3. Symmetry Detection
#Reflection Symmetry Detection:
    import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def compute_distances(points):
   
    distances = []
    for (i, p1), (j, p2) in combinations(enumerate(points), 2):
        dist = np.linalg.norm(p1 - p2)
        distances.append((dist, i, j))
    distances = np.array(distances)
    return distances

def find_symmetry_axes(points):
   
    distances = compute_distances(points)
    axes = []
    for dist1, i1, j1 in distances:
        for dist2, i2, j2 in distances:
            if i1 == i2 or j1 == j2:  # Skip if points are the same
                continue
            mid_point1 = (points[i1] + points[j1]) / 2
            mid_point2 = (points[i2] + points[j2]) / 2
            if np.all(mid_point1 == mid_point2):
                slope = (points[j1, 0] - points[i1, 0]) / (points[j1, 1] - points[i1, 1])
                intercept = mid_point1[1] - slope * mid_point1[0]
                axes.append((-slope, 1, -intercept))
    return axes

def reflect_points(points, axis):
   
    A, B, C = axis
    reflected_points = []
    for point in points:
        x, y = point
        denom = A**2 + B**2
        x_ref = (B * (B * x - A * y) - A * C) / denom
        y_ref = (A * (-B * x + A * y) - B * C) / denom
        reflected_points.append([x_ref, y_ref])
    return np.array(reflected_points)

def detect_symmetry(points):
   
    axes = find_symmetry_axes(points)
    symmetry_axes = []
    for axis in axes:
        reflected_points = reflect_points(points, axis)
        if np.allclose(np.sort(reflected_points, axis=0), np.sort(points, axis=0)):
            symmetry_axes.append(axis)
    return symmetry_axes

def plot_symmetry(points, symmetry_axes):
    
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points')
    
    for A, B, C in symmetry_axes:
        x_vals = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100)
        y_vals = -(A * x_vals + C) / B
        plt.plot(x_vals, y_vals, label='Symmetry Axis')
    
    plt.title('Points and Detected Symmetry Axes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Sample points
    points = np.array([
        [1, 2], [2, 1], [2, 3], [3, 2], [3, 3], [4, 2]
    ])

    symmetry_axes = detect_symmetry(points)
    print("Detected Symmetry Axes:")
    for axis in symmetry_axes:
        A, B, C = axis
        print(f"Axis: {A:.2f}x + {B:.2f}y + {C:.2f} = 0")

    plot_symmetry(points, symmetry_axes)

#4. Completing Incomplete Curves
#Curve Completion Using Interpolation:
    from scipy.interpolate import interp1d

def complete_curve(curve_segments):
    # Example for simple linear interpolation to complete curves
    all_points = np.concatenate(curve_segments)
    sorted_points = all_points[np.argsort(all_points[:, 0])]
    interpolator = interp1d(sorted_points[:, 0], sorted_points[:, 1], kind='linear', fill_value='extrapolate')
    x_new = np.linspace(sorted_points[:, 0].min(), sorted_points[:, 0].max(), num=100)
    y_new = interpolator(x_new)
    return np.column_stack((x_new, y_new))

#5. Exporting to SVG and Rasterization
#Exporting Polylines to SVG:
import svgwrite
import cairosvg

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact*W, output_height=fact*H, background_color='white')




