import os
import sys
import mitsuba as mi
import drjit as dr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample-count", type=int, default=1024)
parser.add_argument("-n", "--probe-count", type=int, nargs=3, default=[100, 100, 100])
parser.add_argument("scene_path")
parser.add_argument("z_level", type=int)
parser.add_argument("--scale", type=float, nargs=3, default=[100, 100, 100])
parser.add_argument("--center", type=float, nargs=3, default=[0, 0, 0])

args = parser.parse_args()
print(args)

# dr.set_log_level(dr.LogLevel.Info)

mi.set_variant("llvm_ad_rgb")


def add_value_to_spherical_harmonics(harmonics, value, normal, mask):
    harmonics[0][mask] += value * 0.25
    harmonics[1][mask] += value * 0.5 * normal.x
    harmonics[2][mask] += value * 0.5 * normal.y
    harmonics[3][mask] += value * 0.5 * normal.z


scene = mi.load_file(args.scene_path)

# Configuration
NUM_VEC3_COEFFICIENTS = 4
probe_count = mi.ScalarPoint3u(args.probe_count)

film = mi.load_dict(
    {
        "type": "hdrfilm",
        "width": probe_count.x * NUM_VEC3_COEFFICIENTS,
        "height": probe_count.y,
        "rfilter": {
            "type": "box",
        },
        "pixel_format": "rgb",
        "component_format": "float32",
    }
)
sampler = mi.load_dict({"type": "independent", "sample_count": args.sample_count})
integrator = mi.load_dict({"type": "path"})

scale = mi.ScalarPoint3f(args.scale)
lower_left = mi.ScalarPoint3f(args.center) - scale / 2
increment = scale / probe_count

# Create a meshgrid over the 2d slice of probes
coord = mi.Vector2u(
    dr.meshgrid(
        dr.arange(dr.llvm.UInt, probe_count.x),
        dr.arange(dr.llvm.UInt, probe_count.y),
    )
)
coord_f = dr.float_array_t(coord)(coord)

sampler.seed(args.z_level, probe_count.x * probe_count.y)

filename = f"output/{args.z_level}.exr"
if os.path.exists(filename):
    print(f"Skipping {filename}")
    sys.exit(0)

coord_z = dr.full(mi.Float, args.z_level, len(coord_f.x))
coord = mi.Vector3f(coord_f.x, coord_f.y, coord_z)
origin = lower_left + increment * (coord + 0.5)

interaction = mi.Interaction3f(t=0, time=0, p=origin, wavelengths=mi.Color0f())

i = mi.UInt32(0)
sh_0 = mi.Vector3f(0, 0, 0)
sh_1 = mi.Vector3f(0, 0, 0)
sh_2 = mi.Vector3f(0, 0, 0)
sh_3 = mi.Vector3f(0, 0, 0)

loop = mi.Loop(
    name="",
    state=lambda: (
        i,
        sh_0,
        sh_1,
        sh_2,
        sh_3,
        sampler,
    ),
)

spherical_harmonics = [sh_0, sh_1, sh_2, sh_3]

while loop(i < args.sample_count):
    # Importance sample emitters

    (ds, spec) = scene.sample_emitter_direction(interaction, sampler.next_2d())
    hit_emitter = dr.neq(ds.pdf, 0.0)

    add_value_to_spherical_harmonics(spherical_harmonics, spec, ds.d, hit_emitter)
    i[hit_emitter] += 1

    more_samples_needed = dr.neq(i, args.sample_count)

    # Sample a random point in the scene

    sample_dir = mi.warp.square_to_uniform_sphere(sampler.next_2d())

    ray = interaction.spawn_ray(sample_dir)

    # Run an initial intersection test
    intersection_test = scene.ray_intersect(ray, mi.RayFlags(0), more_samples_needed)
    intersection_emitter_pointers = mi.UInt.reinterpret_array_(
        intersection_test.emitter(scene)
    )

    # If we didn't hit an emitter, run the pathtracer
    pathtracer_active = dr.eq(intersection_emitter_pointers, 0) & more_samples_needed

    (spec, _, _) = integrator.sample(scene, sampler, ray, active=pathtracer_active)

    add_value_to_spherical_harmonics(
        spherical_harmonics, spec, sample_dir, pathtracer_active
    )
    i[pathtracer_active] += 1


spherical_harmonics = [sh / args.sample_count for sh in spherical_harmonics]

film.prepare([])
image_block = film.create_block()
for i, sh in enumerate(spherical_harmonics):
    image_block.put((coord_f.x + i * probe_count.x, coord_f.y), [sh.x, sh.y, sh.z, 1.0])
film.put_block(image_block)

image = film.develop()

mi.util.write_bitmap(filename, image)
