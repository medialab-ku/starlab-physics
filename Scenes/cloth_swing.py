import random
from framework.meshtaichiwrapper import MeshTaichiWrapper
import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=3,  kernel_profiler=True)

meshes_dynamic = []
mesh_dynamic_1 = MeshTaichiWrapper("../models/OBJ/plane10K.obj", scale=1.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(90.0, 0.0, 0.0))


meshes_dynamic.append(mesh_dynamic_1)

tet_meshes_dynamic = []

meshes_static = []

particles = []

colors_tet_dynamic = []

for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)