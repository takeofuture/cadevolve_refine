import open3d as o3d

print("Open3D OK")

mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()

vis = o3d.visualization.Visualizer()
vis.create_window(width=512, height=512, visible=False)
vis.add_geometry(mesh)

ctr = vis.get_view_control()
params = ctr.convert_to_pinhole_camera_parameters()
print("Camera OK", params is not None)

vis.poll_events()
vis.update_renderer()

vis.destroy_window()
print("Render OK")
