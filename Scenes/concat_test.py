from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
from framework.meshio.concat import concat_mesh
import taichi as ti

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=6, kernel_profiler=enable_profiler)

model_dir = "../models/OBJ/"
model_names = []
trans_list = []
scale_list = []

concat_model_name = "concat.obj"

model_names.append("poncho_8K.obj")
trans_list.append([0.0, 5.0, 0.0])
scale_list.append(3.4)
#
model_names.append("poncho_8K.obj")
trans_list.append([0.0, 5.5, 0.0])
scale_list.append(3.0)

offsets = concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)

#dynamic mesh
mesh_dy = MeshTaichiWrapper("../models/concat.obj", offsets=offsets, scale=1.0, trans=ti.math.vec3(0, 0.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))

#static mesh
mesh_st = MeshTaichiWrapper("../models/OBJ/APoseSMPL.obj",  offsets=[0], scale=12.0, trans=ti.math.vec3(0.0, 0.0, 0.01), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
