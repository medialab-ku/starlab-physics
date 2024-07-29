from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
import taichi as ti
from pathlib import Path

ti.init(arch=ti.cuda, device_memory_GB=6)

model_path = Path(__file__).resolve().parent.parent / "models"

#dynamic mesh
mesh_dy = MeshTaichiWrapper(str(model_path / "OBJ/tet.obj"), offsets=[0], scale=2.0, trans=ti.math.vec3(0, 3.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))

#static mesh
mesh_st = MeshTaichiWrapper(str(model_path / "OBJ/tet.obj"),  offsets=[0], scale=2.0, trans=ti.math.vec3(0.0, 0.0, 0.), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)