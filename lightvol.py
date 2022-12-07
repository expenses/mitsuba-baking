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

filename = f"output/{args.z_level}_{args.sample_count}.exr"
if os.path.exists(filename):
    print(f"Skipping {filename}")
    sys.exit(0)

dr.set_log_level(dr.LogLevel.Info)

mi.set_variant("llvm_ad_rgb")


def generate_sphere_harmonics(value, normal):
    return [value, value * normal.x, value * normal.y, value * normal.z]


scene = mi.load_file(args.scene_path)
probe_count = mi.ScalarPoint3u(args.probe_count)
scale = mi.ScalarPoint3f(args.scale)
lower_left = mi.ScalarPoint3f(args.center) - scale / 2
increment = scale / probe_count

film = mi.load_dict(
    {
        "type": "hdrfilm",
        "width": probe_count.x * 4,
        "height": probe_count.y * probe_count.z,
        "rfilter": {
            "type": "box",
        },
        "pixel_format": "rgb",
        "component_format": "float32",
    }
)
sampler = mi.load_dict({"type": "independent", "sample_count": args.sample_count})
integrator = mi.load_dict({"type": "path"})

# Create a meshgrid over the 2d slice of probes
coord = mi.Vector2u(
    dr.meshgrid(
        # Repeat the x coordinate for the number of samples
        dr.repeat(dr.arange(dr.llvm.UInt, probe_count.x), args.sample_count),
        dr.arange(dr.llvm.UInt, probe_count.y * probe_count.z),
    )
)
coord_z = coord.y // probe_count.y
coord.y = coord.y % probe_count.y
coord = mi.Vector3u(coord.x, coord.y, coord_z)
coord = dr.float_array_t(coord)(coord)

sampler.seed(
    args.z_level, probe_count.x * probe_count.y * probe_count.z * args.sample_count
)

origin = lower_left + increment * (coord + 0.5)

interaction = mi.Interaction3f(t=0, time=0, p=origin, wavelengths=mi.Color0f())

film.prepare([])
image_block = film.create_block()

sample_dir = mi.warp.square_to_uniform_sphere(sampler.next_2d())

ray = interaction.spawn_ray(sample_dir)

(spec, _, _) = integrator.sample(scene, sampler, ray)

spherical_harmonics = generate_sphere_harmonics(spec, sample_dir)

for i, sh in enumerate(spherical_harmonics):
    image_block.put(
        (coord.x + i * probe_count.x, coord.y + coord.z * probe_count.y),
        [sh.x, sh.y, sh.z, 1.0],
    )
film.put_block(image_block)

image = film.develop()

mi.util.write_bitmap(filename, image)
