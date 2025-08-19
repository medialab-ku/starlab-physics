import trimesh
import pyvista

# mesh = trimesh.load('./data/models/bunny.stl')
# mesh.export('./data/models/bunny.obj')

mesh = pyvista.read('./data/models/bunny.stl')
mesh.save('./data/models/bunny.obj')