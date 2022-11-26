import os
import sys
import mitsuba as mi
import drjit as dr

# dr.set_log_level(dr.LogLevel.Info)

mi.set_variant("llvm_ad_rgb")

scene = mi.load_file(sys.argv[1])

# Configuration
NUM_CAMERAS = 25
NUM_SAMPLES = 1024
NUM_VEC3_COEFFICIENTS = 9
center = mi.ScalarPoint3f(0, 0, 0)
scale = mi.ScalarPoint3f(2, 2, 2)

image_res = [NUM_CAMERAS, NUM_CAMERAS]

film = mi.load_dict(
    {
        "type": "hdrfilm",
        "width": image_res[0] * NUM_VEC3_COEFFICIENTS,
        "height": image_res[1],
        "rfilter": {
            "type": "box",
        },
        "pixel_format": "rgb",
        "component_format": "float32",
    }
)
sampler = mi.load_dict({"type": "independent", "sample_count": 1})
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

z = int(sys.argv[2])
sampler.seed(z, dr.prod(image_res))

filename = f"output/{z}.exr"
if os.path.exists(filename):
    print(f"Skipping {filename}")
    sys.exit(0)

coord_z = dr.full(mi.Float, z, len(coord_f.x))
coord = mi.Vector3f(coord_f.x, coord_f.y, coord_z)
origin = lower_left + increment * (coord + 0.5)

interaction = mi.Interaction3f(t=0, time=0, p=origin, wavelengths=mi.Color0f())

i = mi.UInt32(0)
sh_0 = mi.Vector3f(0, 0, 0)
sh_1 = mi.Vector3f(0, 0, 0)
sh_2 = mi.Vector3f(0, 0, 0)
sh_3 = mi.Vector3f(0, 0, 0)
sh_4 = mi.Vector3f(0, 0, 0)
sh_5 = mi.Vector3f(0, 0, 0)
sh_6 = mi.Vector3f(0, 0, 0)
sh_7 = mi.Vector3f(0, 0, 0)
sh_8 = mi.Vector3f(0, 0, 0)

# Forsyth's weights
weight1 = 0.25#4.0 / 17.0
weight2 = 0.5#8.0 / 17.0
weight3 = 0#5.0 / 68.0
weight4 = 0#15.0 / 17.0
weight5 = 0#15.0 / 68.0

loop = mi.Loop(
    name="",
    state=lambda: (
        i,
        sh_0,
        sh_1,
        sh_2,
        sh_3,
        sh_4,
        sh_5,
        sh_6,
        sh_7,
        sh_8,
        sampler,
    ),
)

while loop(i < NUM_SAMPLES):
    i += 1
    (ds, spec) = scene.sample_emitter_direction(interaction, sampler.next_2d())

    # Check if we hit an emitter. If the pdf is 0 then we didn't and need to run the pathfinder
    pathfinder_active = dr.eq(ds.pdf, 0.0)

    # Optionally change the sample direction to a point on a uniform sphere
    #ds.d = dr.select(pathfinder_active, mi.warp.square_to_uniform_sphere(sampler.next_2d()), ds.d)

    (path_spec, mask, aov) = integrator.sample(
        scene, sampler, interaction.spawn_ray(ds.d), active=pathfinder_active
    )

    spec[pathfinder_active] += path_spec

    sh_0 += spec

    sh_1 += spec * ds.d.y
    sh_2 += spec * ds.d.z
    sh_3 += spec * ds.d.x

    sh_4 += spec * ds.d.y * ds.d.x
    sh_5 += spec * ds.d.y * ds.d.z
    sh_6 += spec * (3.0 * ds.d.z * ds.d.z - 1.0)
    sh_7 += spec * ds.d.z * ds.d.x
    sh_8 += spec * (ds.d.x * ds.d.x - ds.d.y * ds.d.y)

sh = [sh_0, sh_1, sh_2, sh_3, sh_4, sh_5, sh_6, sh_7, sh_8]
weights = [
    weight1,
    weight2,
    weight2,
    weight2,
    weight3,
    weight3,
    weight4,
    weight3,
    weight5,
]

sh = [sh * weight / NUM_SAMPLES for sh, weight in zip(sh, weights)]

film.prepare([])
image_block = film.create_block()
for i, sh in enumerate(sh):
    image_block.put((coord_f.x + i * NUM_CAMERAS, coord_f.y), [sh.x, sh.y, sh.z, 1.0])
film.put_block(image_block)

image = film.develop()

mi.util.write_bitmap(filename, image)
