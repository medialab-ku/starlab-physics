
import taichi as ti
import meshtaichi_patcher as patcher
import igl
import os
import numpy as np
import random

@ti.data_oriented
class MeshTaichiWrapper:
    def __init__(self,
                 model_path,
                 offsets,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 is_static=False):

        self.is_static = is_static
        self.mesh = patcher.load_mesh(model_path, relations=["FV", "EV", "FE"])
        self.mesh.verts.place({'fixed': ti.f32,
                               'm_inv': ti.f32,
                               'x0': ti.math.vec3,
                               'x': ti.math.vec3,
                               'y': ti.math.vec3,
                               'v': ti.math.vec3,
                               'dx': ti.math.vec3,
                               'dv': ti.math.vec3,
                               'nc': ti.f32})
        self.mesh.edges.place({'l0': ti.f32})
        self.mesh.faces.place({'aabb_min': ti.math.vec3,
                               'aabb_max': ti.math.vec3,
                               'morton_code': ti.uint32})

        self.offsets = offsets
        self.mesh.verts.fixed.fill(1.0)
        self.mesh.verts.m_inv.fill(0.0)
        self.mesh.verts.v.fill(0.0)
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)

        self.faces = self.mesh.faces
        self.verts = self.mesh.verts

        print(model_path, "# faces:", len(self.mesh.faces))

        # self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.edges) * 2)

        self.face_edge_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.faces) * 3)
        self.init_face_edge_indices()

        self.colors = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.mesh.verts))
        self.init_color()
        # self.colors.fill(0.1)
        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.edges = self.mesh.edges

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.initFaceIndices()
        self.fid_np = self.face_indices.to_numpy()
        self.fid_np = np.reshape(self.fid_np, (len(self.mesh.faces), 3))
        self.initEdgeIndices()
        self.eid_np = self.edge_indices.to_numpy()
        self.eid_np = np.reshape(self.eid_np, (len(self.mesh.edges), 2))
        self.mesh.verts.x0.copy_from(self.mesh.verts.x)

        self.bending_indices = ti.field(dtype=ti.i32)
        self.bending_constraint_count=0
        self.bending_l0 = ti.field(dtype=ti.f32)
        self.initBendingIndices()

        self.render_bending_vert = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.mesh.verts),))
        self.init_render_bending_vert()


    @ti.kernel
    def init_face_edge_indices(self):

        for f in self.mesh.faces:
            for d in range(3):
                self.face_edge_indices[3 * f.id + d] = f.edges[d].id



    @ti.kernel
    def init_render_bending_vert(self):
        for v in self.mesh.verts:
            self.render_bending_vert[v.id] = ti.Vector([v.x[0], v.x[1], v.x[2]])

    def initBendingIndices(self):
        # https://carmencincotti.com/2022-09-05/the-most-performant-bending-constraint-of-xpbd/
        self.bending_constraint_count, neighbor_set = self.findTriNeighbors()
        bending_indices_np = self.getBendingPair(self.bending_constraint_count, neighbor_set)

        ti.root.dense(ti.i, bending_indices_np.shape[0] * 2).place(self.bending_indices)
        ti.root.dense(ti.i, bending_indices_np.shape[0]).place(self.bending_l0)

        for i in range(bending_indices_np.shape[0]):
            self.bending_indices[2 * i] = bending_indices_np[i][0]
            self.bending_indices[2 * i + 1] = bending_indices_np[i][1]

        self.init_bengding_l0()

    @ti.kernel
    def init_bengding_l0(self):
        for bi in ti.ndrange(self.bending_constraint_count):
            v0,v1 = self.bending_indices[2*bi],self.bending_indices[2*bi+1]
            self.bending_l0[bi] = (self.mesh.verts.x[v0] - self.mesh.verts.x[v1]).norm()

    @ti.kernel
    def bendinIndi(self):
        for v in self.mesh.verts:
            self.render_bending_vert[v.id] = ti.Vector([v.x[0], v.x[1], v.x[2]])

    def findTriNeighbors(self):
        # print("initBend")
        # print(self.fid_np)
        # print(self.fid_np.shape)
        num_f = np.rint(self.fid_np.shape[0]).astype(int)
        edgeTable = np.zeros((num_f * 3, self.fid_np.shape[1]), dtype=int)
        # print(edgeTable.shape)

        for f in range(self.fid_np.shape[0]):
            eT = np.zeros((3, 3))
            v1, v2, v3 = np.sort(self.fid_np[f])
            e1, e2, e3 = 3 * f, 3 * f + 1, 3 * f + 2
            eT[0, :] = [v1, v2, e1]
            eT[1, :] = [v1, v3, e2]
            eT[2, :] = [v2, v3, e3]

            edgeTable[3 * f:3 * f + 3, :] = eT

        ind = np.lexsort((edgeTable[:, 1], edgeTable[:, 0]))
        edgeTable = edgeTable[ind]
        # print(edgeTable)

        neighbors = np.zeros((num_f * 3), dtype=int)
        neighbors.fill(-1)

        ii = 0
        bending_constraint_count = 0
        while (ii < 3 * num_f - 1):
            e0 = edgeTable[ii, :]
            e1 = edgeTable[ii + 1, :]

            if (e0[0] == e1[0] and e0[1] == e1[1]):
                neighbors[e0[2]] = e1[2]
                neighbors[e1[2]] = e0[2]

                bending_constraint_count = bending_constraint_count + 1
                ii = ii + 2
            else:
                ii = ii + 1

        # print(bending_constraint_count, "asdf!!")
        return bending_constraint_count, neighbors

    def getBendingPair(self, bend_count, neighbors):

        num_f = np.rint(self.fid_np.shape[0]).astype(int)
        pairs = np.zeros((bend_count, 2), dtype=int)

        count = 0
        # print(neighbors)

        for f in range(num_f):
            for i in range(3):
                eid = 3 * f + i
                neighbor_edge = neighbors[eid]

                if neighbor_edge >= 0:
                    # print(eid,neighbor_edge,neighbors[neighbor_edge])
                    neighbors[neighbor_edge] = -1
                    # find not shared vertex in common edge of adjacent triangles
                    v = np.sort(self.fid_np[f])

                    neighbor_fid = int(np.floor(neighbor_edge / 3.0 + 1e-4) + 1e-4)
                    neighbor_eid_local = neighbor_fid % 3

                    w = np.sort(self.fid_np[neighbor_fid])

                    pairs[count, 0] = v[~np.isin(v, w)]
                    pairs[count, 1] = w[~np.isin(w, v)]

                    count = count + 1

        # print("검산!!!",count == bend_count)
        return pairs



###########################
    def reset(self):
        self.mesh.verts.x.copy_from(self.mesh.verts.x0)
        self.mesh.verts.v.fill(0.)
        self.mesh.verts.fixed.fill(0.0)

    @ti.kernel
    def initFaceIndices(self):
        for f in self.mesh.faces:
            self.face_indices[f.id * 3 + 0] = f.verts[0].id
            self.face_indices[f.id * 3 + 1] = f.verts[1].id
            self.face_indices[f.id * 3 + 2] = f.verts[2].id


    @ti.kernel
    def initEdgeIndices(self):
        for e in self.mesh.edges:
            l0 = (e.verts[0].x - e.verts[1].x).norm()
            e.l0 = l0

            e.verts[0].m_inv += 0.5 * l0
            e.verts[1].m_inv += 0.5 * l0

            self.edge_indices[e.id * 2 + 0] = e.verts[0].id
            self.edge_indices[e.id * 2 + 1] = e.verts[1].id

        for v in self.mesh.verts:
            v.m_inv = 1.0 / v.m_inv


    # @ti.kernel
    # def Hierholzer(self):
    #
    #     edges = []
    #     first = 0
    #     stack = []
    #     stack.append(first)
    #     ret = []
    #     while len(stack) > 0:
    #         v = stack.top()
    #
    #     if !edges[v].size():
    #         ret.append(v)
    #         stack.pop()
    #     else:
    #         i = edges[v][edges[v].size() - 1]
    #         edges[v].pop_back()
    #         std::vector < int >::iterator it = std::find(edges[i].begin(), edges[i].end(), v);
    #         edges[i].erase(it);
    #         stack.push(i)
    #
    #     return ret

    def init_color(self):
        # print(self.offsets)
        for i in range(len(self.offsets)):

            r = random.randrange(0, 255) / 256
            g = random.randrange(0, 255) / 256
            b = random.randrange(0, 255) / 256
            size = 0
            if i < len(self.offsets) - 1:
                size = self.offsets[i + 1] - self.offsets[i]

            else:
                size = len(self.verts) - self.offsets[i]
            #
            self.init_colors(self.offsets[i], size, color=ti.math.vec3(r, g, b))

    @ti.kernel
    def init_colors(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):

        for i in range(size):
            self.colors[i + offset] = color

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for v in self.verts:
            center += v.x

        center /= self.num_verts
        for v in self.mesh.verts:
            v.x -= center

    @ti.kernel
    def applyTransform(self):

        # self.setCenterToOrigin()

        for v in self.mesh.verts:
            v.x *= self.scale

        for v in self.mesh.verts:
            v_4d = ti.Vector([v.x[0], v.x[1], v.x[2], 1])
            rot_rad_x = ti.math.radians(self.rot[0])
            rot_rad_y = ti.math.radians(self.rot[1])
            rot_rad_z = ti.math.radians(self.rot[2])
            rv = ti.math.rotation3d(rot_rad_x, rot_rad_y, rot_rad_z) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.mesh.verts:
            v.x += self.trans

    @ti.kernel
    def computeAABB(self, padding: ti.f32) -> (ti.math.vec3, ti.math.vec3):
        aabb_min = ti.math.vec3(1e5)
        aabb_max = ti.math.vec3(-1e5)

        for v in self.mesh.verts:
            temp = v.x
            ti.atomic_max(aabb_max, temp)
            ti.atomic_min(aabb_min, temp)

        aabb_min -= padding * ti.math.vec3(1.0)
        aabb_max += padding * ti.math.vec3(1.0)

        return aabb_min, aabb_max


    @ti.kernel
    def computeAABB_faces(self, padding: ti.f32):

        aabb_min = ti.math.vec3(1e5)
        aabb_max = ti.math.vec3(-1e5)
        ones = ti.math.vec3(1.0)
        for f in self.mesh.faces:
            f.aabb_min = ti.math.min(f.verts[0].x, f.verts[1].x, f.verts[2].x)
            f.aabb_max = ti.math.max(f.verts[0].x, f.verts[1].x, f.verts[2].x)

            f.aabb_min -= padding * ones
            f.aabb_max += padding * ones
        #     ti.atomic_max(aabb_max, f.aabb_max)
        #     ti.atomic_min(aabb_min, f.aabb_min)
        #
        # return aabb_min, aabb_max



    def export(self, scene_name, frame, is_static = False):
        if is_static:
            directory = os.path.join("../results/", scene_name, "StaticMesh_ID_")
        else:
            directory = os.path.join("../results/", scene_name, "Mesh_ID_")

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create folder" + directory)

        x_np = self.mesh.verts.x.to_numpy()
        file_name = "Mesh_obj_" + str(frame) + ".obj"
        file_path = os.path.join(directory, file_name)
        print("exporting ", file_path.__str__())
        igl.write_triangle_mesh(file_path, x_np, self.fid_np, force_ascii=True)
        print("done")
