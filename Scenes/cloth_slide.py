import random
from framework.mesh import Mesh
import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=3,  kernel_profiler=True)

meshes_dynamic = []
mesh_dynamic_1 = Mesh("../models/OBJ/square_big.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh_dynamic_1 = Mesh("../models/OBJ/plane.obj", scale=2.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))


meshes_dynamic.append(mesh_dynamic_1)

tet_meshes_dynamic = []

meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/square_big.obj", scale=0.3, trans=ti.math.vec3(0.0, -1, 0.0), rot=ti.math.vec3(0.0, 20.0, 0.0), is_static=True)
# mesh_static_1 = Mesh("../models/OBJ/plane.obj", scale=2.5, trans=ti.math.vec3(0.0, -1, 0.0), rot=ti.math.vec3(0.0, 20.0, 0.0), is_static=True)
meshes_static.append(mesh_static_1)

particles = []

colors_tet_dynamic = []

for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)