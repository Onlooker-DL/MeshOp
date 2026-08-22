from __future__ import annotations

import time
from dataclasses import dataclass

import gmsh
import numpy as np


@dataclass
class MeshResult:
    nodes: np.ndarray
    triangles: np.ndarray
    generation_time_sec: float
    background_field_time_sec: float
    gmsh_generate_time_sec: float
    extraction_time_sec: float


class GmshSizeFieldMesher:
    """Generate a periodic Burgers space-time mesh from a scalar size image.

    ``normalized_xt`` constructs an isotropic mesh in
    ``xi=(x+1)/2, tau=t`` on the unit square, then maps nodes back to the
    physical ``(x,t)`` domain. This treats space and time symmetrically only
    after nondimensionalization and preserves the physical 2:1 macro-cell
    aspect ratio used by the original 32x32 MeshOp mesh.
    """

    def __init__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        *,
        size_scale: float = 1.0,
        minimum_size: float = 5.0e-4,
        maximum_size: float = 8.0e-2,
        algorithm: int = 6,
        verbosity: int = 0,
        coordinate_mode: str = "physical_xt",
    ) -> None:
        self.x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.t = np.asarray(t, dtype=np.float64).reshape(-1)
        if self.x.size < 2 or self.t.size < 2:
            raise ValueError("The size image needs at least a 2x2 grid")
        if not np.isclose(self.x[0], -1.0) or self.x[-1] >= 1.0:
            raise ValueError("Expected periodic Burgers x coordinates in [-1,1)")
        self.coordinate_mode = str(coordinate_mode).strip().lower()
        if self.coordinate_mode == "physical_xt":
            self.mesh_x = self.x.copy()
            self.mesh_xmin = -1.0
            self.mesh_xmax = 1.0
            self.mesh_x_period = 2.0
            self.log_size_offset = 0.0
        elif self.coordinate_mode == "normalized_xt":
            self.mesh_x = 0.5 * (self.x + 1.0)
            self.mesh_xmin = 0.0
            self.mesh_xmax = 1.0
            self.mesh_x_period = 1.0
            # The saved prediction uses h0=sqrt(2)/32 in physical (x,t)
            # area units. On the normalized unit square h0_hat=1/32.
            self.log_size_offset = -0.5 * np.log(2.0)
        else:
            raise ValueError(
                "coordinate_mode must be 'physical_xt' or 'normalized_xt'"
            )
        self.mesh_x_extended = np.concatenate((self.mesh_x, [self.mesh_xmax]))
        self.size_scale = float(size_scale)
        self.minimum_size = float(minimum_size)
        self.maximum_size = float(maximum_size)
        self.algorithm = int(algorithm)
        self.verbosity = int(verbosity)

    def mesh_log_h(self, log_h: np.ndarray) -> np.ndarray:
        """Return log mesh size in the coordinate system used by Gmsh."""
        values = np.asarray(log_h, dtype=np.float64)
        expected = (self.t.size, self.x.size)
        if values.shape != expected:
            raise ValueError(f"log_h shape {values.shape} does not match {expected}")
        return values + self.log_size_offset

    def _postview_triangles(self, size: np.ndarray) -> tuple[int, list[float]]:
        # size is [nt,nx], with x=1 supplied by periodic copying.
        extended = np.concatenate((size, size[:, :1]), axis=1)
        records: list[float] = []
        triangle_count = 0
        for jt in range(self.t.size - 1):
            t0, t1 = self.t[jt], self.t[jt + 1]
            for ix in range(self.mesh_x_extended.size - 1):
                x0, x1 = self.mesh_x_extended[ix], self.mesh_x_extended[ix + 1]
                h00 = extended[jt, ix]
                h10 = extended[jt, ix + 1]
                h01 = extended[jt + 1, ix]
                h11 = extended[jt + 1, ix + 1]
                records.extend((
                    x0, x1, x1,
                    t0, t0, t1,
                    0.0, 0.0, 0.0,
                    h00, h10, h11,
                ))
                records.extend((
                    x0, x1, x0,
                    t0, t1, t1,
                    0.0, 0.0, 0.0,
                    h00, h11, h01,
                ))
                triangle_count += 2
        return triangle_count, records

    def generate(self, log_h: np.ndarray) -> MeshResult:
        log_h_mesh = self.mesh_log_h(log_h)
        size = np.exp(log_h_mesh) * self.size_scale
        size = np.clip(size, self.minimum_size, self.maximum_size)
        if not np.all(np.isfinite(size)):
            raise ValueError("Mesh-size prediction contains NaN or Inf")

        start = time.perf_counter()
        generation_start = start
        extraction_start = start
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", self.verbosity)
            gmsh.option.setNumber("Mesh.Algorithm", self.algorithm)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.minimum_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.maximum_size)
            gmsh.model.add("burgers_space_time")

            p1 = gmsh.model.geo.addPoint(self.mesh_xmin, 0.0, 0.0)
            p2 = gmsh.model.geo.addPoint(self.mesh_xmax, 0.0, 0.0)
            p3 = gmsh.model.geo.addPoint(self.mesh_xmax, 1.0, 0.0)
            p4 = gmsh.model.geo.addPoint(self.mesh_xmin, 1.0, 0.0)
            bottom = gmsh.model.geo.addLine(p1, p2)
            right = gmsh.model.geo.addLine(p2, p3)
            top = gmsh.model.geo.addLine(p3, p4)
            left = gmsh.model.geo.addLine(p4, p1)
            loop = gmsh.model.geo.addCurveLoop((bottom, right, top, left))
            gmsh.model.geo.addPlaneSurface((loop,))
            gmsh.model.geo.synchronize()

            # Master left boundary translated by one mesh-domain period is
            # the slave right boundary.
            transform = [
                1, 0, 0, self.mesh_x_period,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ]
            gmsh.model.mesh.setPeriodic(1, [right], [left], transform)

            view = gmsh.view.add("predicted_mesh_size")
            count, data = self._postview_triangles(size)
            gmsh.view.addListData(view, "ST", count, data)
            field = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
            gmsh.model.mesh.field.setAsBackgroundMesh(field)
            generation_start = time.perf_counter()
            gmsh.model.mesh.generate(2)
            extraction_start = time.perf_counter()

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
            tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
            element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
            triangle_blocks: list[np.ndarray] = []
            for element_type, connectivity in zip(element_types, element_nodes):
                _, dimension, _, number_of_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                    int(element_type)
                )
                if dimension == 2 and number_of_nodes == 3:
                    tags = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
                    mapped = np.fromiter(
                        (tag_to_index[int(tag)] for tag in tags.ravel()),
                        dtype=np.int64,
                        count=tags.size,
                    ).reshape(-1, 3)
                    triangle_blocks.append(mapped)
            if not triangle_blocks:
                raise RuntimeError("Gmsh did not return first-order triangles")
            triangles = np.concatenate(triangle_blocks, axis=0)
        finally:
            gmsh.finalize()

        if self.coordinate_mode == "normalized_xt":
            # FEM and error evaluation always receive physical coordinates.
            nodes[:, 0] = 2.0 * nodes[:, 0] - 1.0

        p0 = nodes[triangles[:, 0]]
        p1n = nodes[triangles[:, 1]]
        p2n = nodes[triangles[:, 2]]
        signed = (p1n[:, 0] - p0[:, 0]) * (p2n[:, 1] - p0[:, 1]) - (
            p2n[:, 0] - p0[:, 0]
        ) * (p1n[:, 1] - p0[:, 1])
        if np.any(np.abs(signed) <= 1.0e-15):
            raise RuntimeError("Gmsh returned a degenerate triangle")
        negative = signed < 0
        triangles[negative, 1], triangles[negative, 2] = (
            triangles[negative, 2].copy(), triangles[negative, 1].copy()
        )
        finish = time.perf_counter()
        return MeshResult(
            nodes=nodes,
            triangles=triangles,
            generation_time_sec=finish - start,
            background_field_time_sec=generation_start - start,
            gmsh_generate_time_sec=extraction_start - generation_start,
            extraction_time_sec=finish - extraction_start,
        )
