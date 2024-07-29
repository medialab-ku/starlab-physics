from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
from framework.meshio.concat import concat_mesh
import taichi as ti
from pathlib import Path

ti.init(arch=ti.cuda, device_memory_GB=6)

model_path = Path(__file__).resolve().parent.parent / "models"
model_dir = str(model_path / "OBJ")
model_names = []
trans_list = []
scale_list = []

concat_model_name = "concat.obj"

model_names.append("poncho_8K.obj")
trans_list.append([0.0, 5.0, 0.0])
scale_list.append(4.4)
#
# model_names.append("poncho_8K.obj")
# trans_list.append([0.0, 5.5, 0.0])
# scale_list.append(5.0)

offsets = concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)

#dynamic mesh
mesh_dy = MeshTaichiWrapper(str(model_path / "concat.obj"), offsets=offsets, scale=1.0, trans=ti.math.vec3(0, 1.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))

#static mesh
mesh_st = MeshTaichiWrapper(str(model_path / "OBJ/APoseSMPL.obj"),  offsets=[0], scale=13.0, trans=ti.math.vec3(0.0, 0.0, 0.01), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)