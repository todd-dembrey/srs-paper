import os
from collections import defaultdict

import dicom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage import measure


def load_by_type(path, file_type, contains=""):
    return [
        dicom.read_file(os.path.join(path, s))
        for s in os.listdir(path)
        if s.startswith(file_type) and (contains in s)
    ]


def load_planning(path):
    plans = load_by_type(path, "RS.", contains="GTV")
    return plans


def load_slices(path):
    slices = load_by_type(path, "CT.")
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def load_scan(path):
    return {
        "slices": load_slices(path),
        "planning": load_planning(path),
    }


def get_data(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    return image.astype(np.int16)


def get_pixels_hu(scans):
    image = get_data(scans)
    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    scan = scans[0]
    intercept = scan.RescaleIntercept
    slope = scan.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def get_plans(plans, number_slices):
    results = defaultdict(lambda: defaultdict(list))
    for plan in plans:
        users = {user.ROINumber: user.ROIName for user in plan.StructureSetROISequence}
        print(users)
        for user in plan.ROIContourSequence:
            # If there are more outlines than slices it is probably the outline
            # of the extent
            try:
                if len(user.ContourSequence) < number_slices:
                    for sequence in user.ContourSequence:
                        data = np.asarray(sequence.ContourData).reshape(-1, 3)
                        # Round helps with floating point errors
                        results[user.ReferencedROINumber][round(data[0, 2], 2)].append(
                            data
                        )
            except AttributeError:
                pass
    return results


def get_dimensions(scans):
    scan = scans[0]
    origin = np.array(scan.ImagePositionPatient, dtype=float)
    pixels = np.array([scan.Rows, scan.Columns, len(scans)])
    spacing = np.array([*scan.PixelSpacing, scan.SliceThickness])
    end = origin + pixels * spacing
    return np.array([*origin, *end])


data_path = "./data/zzsrs_outlining2/"
patient = load_scan(data_path)
dimensions = get_dimensions(patient["slices"])
plans = get_plans(patient["planning"], len(patient["slices"]))
imgs = get_pixels_hu(patient["slices"])

# Hounsfield histogram

# plt.hist(imgs.flatten(), bins=50, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()


def sample_stack(stack, rows=6, cols=6, start_with=0):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    show_every = len(stack) // (rows * cols)
    for i in range(rows * cols):
        ind = start_with + i * show_every
        idx = int(i / rows), int(i % rows)
        ax[idx].set_title("slice %d" % ind)
        im = ax[idx].imshow(stack[ind], cmap="gray")
        ax[idx].axis("off")
    plt.colorbar(im, ax=ax.ravel().tolist())
    plt.show()


# sample_stack(imgs)


def make_mesh(image, threshold=-500, step_size=1):

    p = image.transpose(2, 1, 0)

    verts, faces, norm, val = measure.marching_cubes(
        p, threshold, step_size=step_size, allow_degenerate=True
    )
    return verts, faces


def plt_3d(verts, faces):
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()


# v, f = make_mesh(imgs)
# plt_3d(v, f)


class IndexTracker(object):
    def __init__(self, ax, patient, slices, plans, limit_to=list()):
        self.ax = ax
        self.limit_to = limit_to
        ax.set_title("use scroll wheel to navigate images")

        self.slices = slices
        self.plans = plans
        number_users = len(plans.keys())
        cmap = plt.cm.get_cmap("jet", number_users)
        self.colors = {user: cmap(i) for i, user in enumerate(plans.keys())}
        self.elevation = np.arange(
            dimensions[2], dimensions[5], step=patient["slices"][0].SliceThickness
        )
        self.ind = 0
        self.extent = dimensions[[3, 0, 4, 1]]
        self.im = ax.imshow(
            self.slices[self.ind], extent=self.extent, origin="upper", cmap="gray"
        )
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % len(self.slices)
        else:
            self.ind = (self.ind - 1) % len(self.slices)
        self.update()

    def update(self):
        self.ax.patches.clear()
        height = self.elevation[self.ind]
        self.im.set_data(np.fliplr(self.slices[self.ind]))
        for user, heights in self.plans.items():
            if user in self.limit_to:
                color = self.colors[user]
                for outline in heights.get(height, list()):
                    self.ax.fill(outline[:, 0], outline[:, 1], color=color, fill=False)
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()


fig, ax = plt.subplots(1, 1)

tracker = IndexTracker(ax, patient, imgs, plans, [4, 7])

fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
plt.show()
