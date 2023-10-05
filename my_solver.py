import taichi as ti
import meshtaichi_patcher as Patcher

@ti.dataclass
class contact_particle:
    vid: ti.u8
    w  : ti.math.vec3

@ti.dataclass
class contact_triangle:
    x: ti.math.vec3
    vids: ti.math.ivec3
    w: ti.math.vec3

@ti.dataclass
class contact_edge:
    x: ti.math.vec3
    w: ti.math.vec2

@ti.dataclass
class edge:
    vid: ti.math.uvec2
    l0: ti.float32

@ti.dataclass
class node:
    x   : ti.math.vec3
    v   : ti.math.vec3
    f   : ti.math.vec3
    y   : ti.math.vec3
    x_k : ti.math.vec3
    m   : ti.float32
    grad: ti.math.vec3
    hii : ti.math.mat3

@ti.dataclass
class static_node:
    x: ti.math.vec3

@ti.data_oriented
class Solver:
    def __init__(self,
                 my_mesh,
                 static_mesh,
                 bottom,
                 k=1e4,
                 dt=1e-3,
                 max_iter=1000):
        self.my_mesh = my_mesh
        self.static_mesh = static_mesh
        self.k = k
        self.dt = dt
        self.dtSq = dt ** 2
        self.max_iter = max_iter
        self.gravity = -1.0
        self.bottom = bottom
        self.idenity3 = ti.math.mat3([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

        self.radius = 0.01
        self.contact_stiffness = 1e5
        self.edges = edge.field(shape=len(self.my_mesh.mesh.edges))
        self.init_edges()
        self.nodes = node.field(shape=(len(self.my_mesh.mesh.verts)))
        self.num_nodes = len(self.my_mesh.mesh.verts)
        self.num_static_verts = len(self.static_mesh.mesh.verts)
        self.num_static_edges = len(self.static_mesh.mesh.edges)
        self.num_static_faces = len(self.static_mesh.mesh.faces)

        self.num_verts = len(self.my_mesh.mesh.verts)
        self.num_edges = len(self.my_mesh.mesh.edges)
        self.num_faces = len(self.my_mesh.mesh.faces)

        self.static_nodes = static_node.field(shape=(self.num_static_verts + self.num_static_edges + self.num_static_faces))
        self.contact_triangles = contact_triangle.field(shape=len(self.my_mesh.mesh.edges))
        self.num_static_nodes = self.num_static_verts + self.num_static_edges + self.num_static_faces
        self.init_nodes()
        self.grid_n = 128
        self.grid_particles_list = ti.field(ti.i32)
        self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        self.partical_array = self.grid_block.dynamic(ti.l, len(self.my_mesh.mesh.verts))
        self.partical_array.place(self.grid_particles_list)
        self.grid_particles_count = ti.field(ti.i32)
        ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)
        print(f"verts #: {len(self.my_mesh.mesh.verts)}, elements #: {len(self.my_mesh.mesh.edges)}")
        # self.setRadius()
        print(f"radius: {self.radius}")


        # self.initContactParticleData()



    @ti.kernel
    def init_edges(self):
        for e in self.my_mesh.mesh.edges:
            self.edges[e.id].vid[0] = e.verts[0].id
            self.edges[e.id].vid[1] = e.verts[1].id
            self.edges[e.id].l0 = e.l0

    @ti.kernel
    def init_nodes(self):
        for v in self.my_mesh.mesh.verts:
            self.nodes[v.id].x = v.x
            self.nodes[v.id].m = v.m
            self.nodes[v.id].v = v.v

        for v in self.static_mesh.mesh.verts:
            self.static_nodes[v.id].x = v.x

        for e in self.static_mesh.mesh.edges:
            self.static_nodes[e.id + self.num_static_verts].x = 0.5 * (e.verts[0].x + e.verts[1].x)

        for f in self.static_mesh.mesh.faces:
            self.static_nodes[f.id + self.num_static_verts + self.num_static_faces].x = 0.333 * (f.verts[0].x + f.verts[1].x + f.verts[2].x)


    # def setRadius(self):
    #     min = 100
    #
    #     for e in self.my_mesh.mesh.edges:
    #         if(min > e.l0):
    #             min = e.l0
    #
    #     self.radius = 0.4 * min

    # @ti.func
    # def resolve_contact(self, i, j):
    #     # test = i + j
    #     i0, i1, i2 = self.contact_triangles[i].vid[0], self.contact_triangles[i].vid[1], self.contact_triangles[i].vid[2]
    #     wi0, wi1, wi2 = self.contact_triangles[i].w[0], self.contact_triangles[i].w[1], self.contact_triangles[i].w[2]
    #     j0, j1, j2 = self.contact_triangles[j].vid[0], self.contact_triangles[j].vid[1], self.contact_triangles[j].vid[2]
    #     wj0, wj1, wj2 = self.contact_triangles[j].w[0], self.contact_triangles[j].w[1], self.contact_triangles[j].w[2]
    #
    #     rel_pos = self.contact_triangles[j].x - self.contact_triangles[i].x
    #     dist = rel_pos.norm()
    #     delta = dist - 2 * self.radius  # delta = d - 2 * r
    #     coeff = self.contact_stiffness * self.dtSq
    #     if delta < 0:  # in contact
    #         normal = rel_pos / dist
    #         f1 = normal * delta * coeff
    #         self.nodes[i0].grad -= wi0 * f1
    #         self.nodes[i1].grad -= wi1 * f1
    #         self.nodes[i2].grad -= wi2 * f1
    #
    #         self.nodes[j0].grad += wj0 * f1
    #         self.nodes[j1].grad += wj1 * f1
    #         self.nodes[j2].grad += wj2 * f1
    #
    #         coef_hess = self.idenity3 * coeff
    #
    #         self.nodes[i0].hii += wi0 * coef_hess
    #         self.nodes[i1].hii += wi1 * coef_hess
    #         self.nodes[i2].hii += wi2 * coef_hess
    #
    #         self.nodes[j0].hii += wj0 * coef_hess
    #         self.nodes[j1].hii += wj1 * coef_hess
    #         self.nodes[j2].hii += wj2 * coef_hess


    @ti.kernel
    def init_contact_triangles(self):

        for v in self.my_mesh.mesh.verts:
            self.contact_triangles[v.id].x = v.x
            self.contact_triangles[v.id].vids[0] = v.id
            self.contact_triangles[v.id].vids[1] = -1
            self.contact_triangles[v.id].vids[2] = -1

        for e in self.my_mesh.mesh.edges:
            self.contact_triangles[e.id + self.num_verts].x = 0.5 * (e.verts[0].x + e.verts[1].x)
            self.contact_triangles[e.id + self.num_verts].vids[0] = e.verts[0].id
            self.contact_triangles[e.id + self.num_verts].vids[1] = e.verts[1].id
            self.contact_triangles[e.id + self.num_verts].vids[2] = -1

        for f in self.my_mesh.mesh.faces:
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].x = 0.333 * (f.verts[0].x + f.verts[1].x + f.verts[2].x)
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].vids[0] = f.verts[0].id
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].vids[1] = f.verts[1].id
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].vids[2] = f.verts[2].id

    @ti.func
    def resolve_contact(self, i, j):
        # test = i + j
        rel_pos = self.nodes[j].x_k - self.nodes[i].x_k
        dist = rel_pos.norm()
        delta = dist - 2 * self.radius  # delta = d - 2 * r
        coeff = self.contact_stiffness * self.dtSq
        if delta < 0:  # in contact
            normal = rel_pos / dist
            f1 = normal * delta * coeff
            self.nodes[i].grad -= f1
            self.nodes[j].grad += f1
            self.nodes[i].hii += self.idenity3 * coeff
            self.nodes[j].hii += self.idenity3 * coeff

    @ti.func
    def resolve_contact_static(self, i, j):
        # test = i + j
        rel_pos = self.nodes[i].x_k - self.static_nodes[j].x
        dist = rel_pos.norm()
        delta = dist - 2 * self.radius  # delta = d - 2 * r
        coeff = self.contact_stiffness * self.dtSq
        if delta < 0:  # in contact
            normal = rel_pos / dist
            f1 = normal * delta * coeff
            self.nodes[i].grad += f1
            self.nodes[i].hii += self.idenity3 * coeff

    # @ti.kernel
    # def initContactParticleData(self):
    #     for c in self.contact_particle:
    #         self.contact_particle[c].vid = c

    @ti.kernel
    def computeNextState(self):
        for n in self.nodes:
            self.nodes[n].v = (self.nodes[n].x_k - self.nodes[n].x) / self.dt
            self.nodes[n].x = self.nodes[n].x_k

        # for v in self.my_mesh.mesh.verts:
        #     v.v = (v.x_k - v.x) / self.dt
        #     v.x = v.x_k

    @ti.kernel
    def computeGradientAndElementWiseHessian(self):

        # momentum gradient M * (x - y) and hessian M
        for n in self.nodes:
            self.nodes[n].grad = self.nodes[n].m * (self.nodes[n].x_k - self.nodes[n].y) - self.nodes[n].f * self.dtSq
            self.nodes[n].hii = self.nodes[n].m * self.idenity3

        # elastic energy gradient \nabla E (x)
        for e in self.edges:
            v0, v1 = self.edges[e].vid[0], self.edges[e].vid[1]
            l = (self.nodes[v0].x_k - self.nodes[v1].x_k).norm()
            normal = (self.nodes[v0].x_k - self.nodes[v1].x_k).normalized(1e-12)
            coeff = self.dtSq * self.k
            grad_e = coeff * (l - self.edges[e].l0) * normal
            self.nodes[v0].grad += grad_e
            self.nodes[v1].grad -= grad_e
            self.nodes[v0].hii += coeff * self.idenity3
            self.nodes[v1].hii += coeff * self.idenity3

        # handling bottom contact
        for n in self.nodes:
            if (self.nodes[n].x_k[1] < 0):
                depth = self.nodes[n].x_k[1] - self.bottom
                up = ti.math.vec3(0, 1, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii  += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[1] > 1):
                depth = 1 - self.nodes[n].x_k[1]
                up = ti.math.vec3(0, -1, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii  += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[0] < 0):
                depth = self.nodes[n].x_k[0] - self.bottom
                up = ti.math.vec3(1, 0, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[0] > 1):
                depth = 1 - self.nodes[n].x_k[0]
                up = ti.math.vec3(-1, 0, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[2] < 0):
                depth = self.nodes[n].x_k[2] - self.bottom
                up = ti.math.vec3(0, 0, 1)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[2] > 1):
                depth = 1 - self.nodes[n].x_k[2]
                up = ti.math.vec3(0, 0, -1)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

        # handling sphere contact
        # for v in self.my_mesh.mesh.verts:
        #     center = ti.math.vec3(0, -0.3, 0)
        #     radius = 0.3
        #     dist = (v.x_k - center).norm()
        #     normal = (v.x_k - center).normalized(1e-6)
        #     if(dist < radius):
        #         coeff = self.dtSq * self.contact_stiffness
        #         v.grad += coeff * (dist - radius) * normal
        #         v.hii += coeff * self.idenity3

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.resolve_contact(i, j)

        for i in range(self.num_nodes):
            for j in range(self.num_static_nodes):
                self.resolve_contact_static(i, j)
        # self.grid_particles_count.fill(0)
        # for n in self.nodes:
        #     grid_idx = ti.floor(self.nodes[n].x_k * self.grid_n, int)
        #     ti.append(self.grid_particles_list.parent(), grid_idx, int(n))
        #     ti.atomic_add(self.grid_particles_count[grid_idx], 1)
        #
        # for n in self.nodes:
        #     grid_idx = ti.floor(self.nodes[n].x_k * self.grid_n, int)
        #     x_begin = max(grid_idx[0] - 1, 0)
        #     x_end = min(grid_idx[0] + 2, self.grid_n)
        #
        #     y_begin = max(grid_idx[1] - 1, 0)
        #     y_end = min(grid_idx[1] + 2, self.grid_n)
        #
        #     z_begin = max(grid_idx[2] - 1, 0)
        #     # only need one side
        #     z_end = min(grid_idx[2] + 1, self.grid_n)
        #
        #     # todo still serialize
        #     for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
        #
        #         # on split plane
        #         if neigh_k == grid_idx[2] and (neigh_i + neigh_j) > (grid_idx[0] + grid_idx[1]) and neigh_i <= grid_idx[0]:
        #             continue
        #         # same grid
        #         iscur = neigh_i == grid_idx[0] and neigh_j == grid_idx[1] and neigh_k == grid_idx[2]
        #         for l in range(self.grid_particles_count[neigh_i, neigh_j, neigh_k]):
        #             j = self.grid_particles_list[neigh_i, neigh_j, neigh_k, l]
        #
        #             if iscur and n >= j:
        #                 continue
        #             self.resolve_contact(n, j)

        for n in self.nodes:
            self.nodes[n].x_k -= self.nodes[n].hii.inverse() @ self.nodes[n].grad


    # @ti.kernel
    # def computeExternalForce(self):
    #     for v in self.my_mesh.mesh.verts:

    @ti.kernel
    def computeY(self):
        # for v in self.my_mesh.mesh.verts:
        #     v.y = v.x + v.v * self.dt + (v.f / v.m) * self.dtSq

        for n in self.nodes:
            self.nodes[n].y = self.nodes[n].x + self.nodes[n].v * self.dt + (self.nodes[n].f / self.nodes[n].m) * self.dtSq

    def update(self):

        # self.computeExternalForce()
        # self.my_mesh.mesh.verts.f.fill([0.0, self.gravity, 0.0])
        self.nodes.f.fill([0.0, self.gravity, 0.0])
        self.computeY()
        self.nodes.x_k.copy_from(self.nodes.y)
        for i in range(self.max_iter):
            self.computeGradientAndElementWiseHessian()

        self.computeNextState()


