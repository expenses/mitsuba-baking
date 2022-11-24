import os
import sys
import mitsuba as mi
import drjit as dr

# dr.set_log_level(dr.LogLevel.Info)


def array_from_values(values):
    array = dr.zeros(type(values[0]), len(values))

    for i, value in enumerate(values):
        dr.scatter(array, value, i)

    return array


mi.set_variant("llvm_ad_rgb")

scene = mi.load_file(sys.argv[1])

# Configuration
NUM_CAMERAS = 100
FACE_SIZE = 16
center = mi.ScalarPoint3f(0, 0, 0)
scale = mi.ScalarPoint3f(2, 2, 2)

image_res = [NUM_CAMERAS * FACE_SIZE * 6, NUM_CAMERAS * FACE_SIZE]

film = mi.load_dict(
    {
        "type": "hdrfilm",
        "width": image_res[0],
        "height": image_res[1],
        "rfilter": {
            "type": "box",
        },
        "pixel_format": "rgb",
        "component_format": "float32",
    }
)
sampler = mi.load_dict({"type": "independent", "sample_count": 1})
sampler.seed(0xDEADCAFE, image_res[0] * image_res[1])
integrator = mi.load_dict({"type": "path"})

lower_left = center - scale / 2
increment = scale / NUM_CAMERAS

# Create a meshgrid over all pixel coordinates
coord = mi.Vector2u(
    dr.meshgrid(
        dr.arange(dr.llvm.UInt, image_res[0]),
        dr.arange(dr.llvm.UInt, image_res[1]),
    )
)
coord_f = dr.float_array_t(coord)(coord)

# Get coordinates within the face
fract = coord % FACE_SIZE

# get coordinates outside the face, of views
index = coord // FACE_SIZE

# Get which face is being rendered for each probe
camera_face = index.x % 6
# And which probe is being rendered on x
index.x //= 6

index = dr.float_array_t(index)(index)

# Generate a uv coord that goes directly through the center of each pixel
uv = (dr.float_array_t(fract)(fract) + 0.5) / FACE_SIZE
# Flip on x
uv.x = 1.0 - uv.x

face_dir = array_from_values(
    [
        dr.llvm.Array3f(+1, 0, 0),  # +x
        dr.llvm.Array3f(-1, 0, 0),  # -x
        dr.llvm.Array3f(0, +1, 0),  # +y
        dr.llvm.Array3f(0, -1, 0),  # -y
        dr.llvm.Array3f(0, 0, +1),  # +z
        dr.llvm.Array3f(0, 0, -1),  # -z
    ]
)

up = array_from_values(
    [
        dr.llvm.Array3f(0, 1, 0),
        dr.llvm.Array3f(0, 1, 0),
        dr.llvm.Array3f(0, 0, -1),
        dr.llvm.Array3f(0, 0, +1),
        dr.llvm.Array3f(0, 1, 0),
        dr.llvm.Array3f(0, 1, 0),
    ]
)

face_dir = dr.gather(dtype=type(face_dir), source=face_dir, index=camera_face)
up = dr.gather(dtype=type(up), source=up, index=camera_face)

camera_to_sample = mi.perspective_projection(
    [FACE_SIZE, FACE_SIZE], [FACE_SIZE, FACE_SIZE], [0, 0], 90, 0.0001, 1000
)

sample_to_camera = camera_to_sample.inverse()

ray_dir = mi.Transform4f.look_at([0, 0, 0], face_dir, up) @ dr.normalize(
    sample_to_camera @ mi.Point3f(uv.x, uv.y, 0)
)

for z in range(NUM_CAMERAS):
    filename = f"output/{z}.exr"
    if os.path.exists(filename):
        print(f"Skipping {filename}")
        continue

    # Reset the film to stop samples for accumulating
    film.prepare([])

    print(f"rendering {filename}")
    index_z = dr.full(mi.Float, z, len(index.x))
    coord = mi.Vector3f(index.x, index.y, index_z)
    origin = lower_left + increment * (coord + 0.5)

    ray = mi.Ray3f(o=origin, d=ray_dir)

    (spec, mask, aov) = integrator.sample(scene, sampler, ray)
    image_block = film.create_block()
    image_block.put((coord_f.x, coord_f.y), [spec.x, spec.y, spec.z, 1.0])
    # For debugging
    # image_block.put((coord_f.x, coord_f.y), [origin.x, origin.y, origin.z, 1.0])
    film.put_block(image_block)

    image = film.develop()

    mi.util.write_bitmap(filename, image)
