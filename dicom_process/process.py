import os
from collections import defaultdict

import dicom
import numpy as np
import progressbar
from shapely.geometry import Point as ShPoint
from shapely.geometry.polygon import Polygon
from shapely.prepared import prep
from shapely.ops import unary_union


def progress_bar(max_value):
    return progressbar.ProgressBar(
        max_value=max_value,
        widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.Counter(format='%(value)d/%(max_value)d'), ': ', progressbar.ETA(), ') ',
        ]
    )


class Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __str__(self):
        return str(self.x)[0:2] + str(self.y)[0:2] + str(self.z)[0:2]


class DicomData:
    RESOLUTION = 0.5

    def __init__(self, path, users=list(), data=None, dimensions=None):
        self.path = path
        if data:
            self.data = {
                user: value
                for user, value in data.items()
                if not users or user in users
            }
        else:
            self.data = self.load_data(users)

        if dimensions:
            self._dimensions = dimensions
        else:
            self._dimensions = None

    def get_outlines_from_file(self):
        dicom_file = dicom.read_file(self.path)
        identifiers = {
            roi_sequence_set.ROINumber: roi_sequence_set.ROIName
            for roi_sequence_set in dicom_file.StructureSetROISequence
        }
        for contour_sequences in dicom_file.ROIContourSequence:
            try:
                for contour in contour_sequences.ContourSequence:
                    yield identifiers[contour.ROINumber], contour.ContourData
            except:
                pass

    def load_data(self, users):
        results = defaultdict(lambda: defaultdict(list))
        # Identifier -> Elevation -> [outlines]
        for identifier, contour in self.get_outlines_from_file():
            if not users or identifier in users:
                # reshape to an array of (x, y, z)
                array = np.asarray(contour).reshape(-1, 3)
                results[identifier][array[0, 2]].append(array)
        return results

    def limit_to_identifiers(self, identifiers):
        return self.__class__(self.path, identifiers, self.data)

    @property
    def dimensions(self):
        if self._dimensions:
            return self._dimensions

        mins = np.ndarray((0, 3))
        maxs = np.ndarray((0, 3))
        for user in self.data.values():
            for height in user.values():
                for outline in height:
                    mins = np.vstack((mins, outline.min(0)))
                    maxs = np.vstack((maxs, outline.max(0)))
        minimums = [*np.floor(mins.min(0)[0:2]),  mins.min(0)[2]]
        maximums = [*np.ceil(maxs.max(0)[0:2]),  maxs.max(0)[2]]
        self._dimensions = Point(*minimums), Point(*maximums)
        return self._dimensions

    @property
    def z_resolution(self):
        for user in self.data.values():
            heights = list(user.keys())
            diffs = np.diff(list(heights))
            return diffs[0]

    @property
    def mesh_matrix(self):
        min_dims, max_dims = self.dimensions
        xs = np.arange(min_dims.x, max_dims.x, step=self.RESOLUTION)
        ys = np.arange(min_dims.y, max_dims.y, step=self.RESOLUTION)
        xv, yv, = np.meshgrid(xs, ys)
        zs = np.arange(min_dims.z, max_dims.z, step=self.z_resolution)
        return xv, yv, zs

    @property
    def voxel_volume(self):
        return self.RESOLUTION ** 2 * self.z_resolution

    @property
    def users(self):
        return list(self.data.keys())

    def resize_to(self, target):
        return DicomData(self.path, data=self.data, dimensions=target.dimensions)


class DicomProcess:
    def __init__(self, path, clear, users, suffix='', av_50_data=None):
        self.suffix = suffix
        self.path = path
        _, file_with_ext = os.path.split(path)
        self.name, _ = os.path.splitext(file_with_ext)

        self.all_data = DicomData(path)

        self.data = self.all_data.limit_to_identifiers(users)

        self.num_users = len(self.data.users)

        self.count_of_points = self.calc_results(self.data, clear)

        self.av_100, self.av_100_matrix = self.calculate_av(100, self.count_of_points)

        if av_50_data is None:
            self.av_50_data = self.data
            av_50_results = self.count_of_points
        else:
            resized_data = av_50_data.data.resize_to(self.data)
            self.av_50_data = resized_data
            av_50_results = self.calc_results(resized_data, clear)

        self.av_50, self.av_50_matrix = self.calculate_av(50, av_50_results)

        # Encompassing Volume
        self.ev, self.ev_matrix = self.calculate_av(0, self.results)

        self.cci, self.dci = self.calc_cci_and_dci(self.results)

    def calc_results(self, data, clear=False):
        dims = '-'.join(str(dim) for dim in self.data.dimensions)
        file_name = f"-{self.data.RESOLUTION}{self.suffix}{'-'.join(data.users)}-{dims}.npy"
        cached_file_name = os.path.splitext(self.path)[0] + file_name

        if not os.path.exists(cached_file_name) or clear:
            count = self.count_points_in_mesh(data)
            np.save(cached_file_name, count)
        else:
            count = np.load(cached_file_name)

        return count

    def process_outlines(self, data):
        polygons = {}
        for user, heights in data.data.items():
            for height, outlines in heights.items():
                for outline in outlines:
                    poly = Polygon(outline[:, 0:2])
                    polygons.setdefault(height, dict()).setdefault(user, list()).append(poly)
        return polygons

    def extent_of_outlines(self, outlines):
        boundary = dict()
        for height, polys in outlines.items():
            boundary[height] = unary_union([poly for user in polys.values() for poly in user])
        return boundary

    def count_points_in_mesh(self, data):
        *mesh, elevations = self.data.mesh_matrix
        counts = np.zeros(
            [*mesh[0].shape, len(elevations), self.num_users],
            dtype=np.int8,
        )
        points = {
            x: {y: ShPoint(x, y) for y in mesh[1][:, 0]}
            for x in mesh[0][0, :]
        }
        outlines = self.process_outlines(data)
        boundary = self.extent_of_outlines(outlines)

        progress = progress_bar(len(elevations)+1)

        for i, elevation in progress(enumerate(elevations)):
            try:
                outlines = outlines[elevation].items()
            except KeyError:
                pass
            else:
                for user, outlines in outlines:

                    prepared_boundary = prep(boundary[elevation])
                    prepared_outlines = [prep(poly) for poly in outlines]

                    @np.vectorize
                    def check_point_in_outline(x, y):
                        # Perform check on each element of matrix
                        point = points[x][y]
                        # Check if outside the boundary
                        if not prepared_boundary.contains(point):
                            return 0
                        # Count number of outlines it appears in
                        return sum(
                            outline.contains(point)
                            for outline in prepared_outlines
                        )
                    try:
                        user_index = self.data.users.index(user)
                    except ValueError:
                        # Its a resub on av50
                        user_index = self.av_50_data.users.index(user)

                    counts[:, :, i, user_index] = check_point_in_outline(*mesh)
        return counts

    def calculate_av(self, value, results):
        limit = value / 100 * results.shape[-1]
        if value == 100:
            # correct for boundary
            limit -= 1
        matrix = results.sum(3) > limit
        return np.sum(matrix) * self.data.voxel_volume, matrix

    def calc_cci_and_dci(self, results):
        all_cci = list()
        all_dci = list()
        sum_av_50 = np.sum(self.av_50_matrix)
        for elevation, user in enumerate(self.data.users):
            data = results[:, :, :, elevation]

            # Either the AV50 or the user outlined the area
            union = (self.av_50_matrix + data) > 0
            # Both the user and the AV50 match a point
            intersection = np.multiply(self.av_50_matrix, data)

            cci = np.sum(intersection) / np.sum(union)
            dci = (sum_av_50 - np.sum(intersection)) / sum_av_50

            all_cci.append(cci)
            all_dci.append(dci)

        return np.array(all_cci), np.array(all_dci)

    @property
    def volumes(self):
        return np.apply_over_axes(
            np.sum, self.results, [0, 1, 2]
        ).flatten() * self.data.voxel_volume

    @property
    def volumes_cm3(self):
        return [volume / 1000 for volume in self.volumes]

    @property
    def radii(self):
        return (self.volumes / np.pi) ** (1/3)
