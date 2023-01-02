import sys
import mitsuba as mi
import drjit as dr
import argparse

def generate_sphere_harmonics(value, normal):
    return [value, value * normal.x, value * normal.y, value * normal.z]

def create_film(width, height):
    film = mi.load_dict({
        "type": "hdrfilm",
        "width": width,
        "height": height,
        "rfilter": {
            "type": "box",
        },
        "pixel_format": "rgb",
        "component_format": "float32",
    })
    film.prepare([])
    return film, film.create_block()

def normalize_value(value, average):
    value /= average
    value = dr.select(dr.isnan(value), 0, value)
    return value# * 0.5 + 0.5

parser = argparse.ArgumentParser()
parser.add_argument("scene_path")
parser.add_argument("positions_path")
parser.add_argument("normals_path")
parser.add_argument("seed", type=int)
parser.add_argument("--sample-count", type=int, default=1024)
parser.add_argument("--supersampling", type=int, default=4)

args = parser.parse_args()

origin_sample_count = args.sample_count // (args.supersampling * args.supersampling)

print(args, origin_sample_count)

dr.set_log_level(dr.LogLevel.Info)
mi.set_variant("llvm_ad_rgb")

scene = mi.load_file(args.scene_path)

positions = mi.TensorXf(mi.Bitmap(args.positions_path))
normals = mi.TensorXf(mi.Bitmap(args.normals_path))

# Note: these are flipped for some reason.
height = positions.shape[0]
width = positions.shape[1]
num_channels = positions.shape[2]

integrator = mi.load_dict({"type": "path"})

sampler = mi.load_dict({"type": "independent", "sample_count": origin_sample_count})

sampler.seed(args.seed, width * height * origin_sample_count)

index = dr.repeat(
    dr.arange(dr.llvm.UInt, width * height), origin_sample_count
)

pixel_offset = index * num_channels
pixel_coord = mi.Vector3u(pixel_offset, pixel_offset + 1, pixel_offset + 2)

position = dr.gather(mi.Vector3f, positions.array, pixel_coord)
normal = dr.gather(mi.Vector3f, normals.array, pixel_coord)

mask = dr.neq(normal.x, 0) | dr.neq(normal.y, 0) | dr.neq(normal.z, 0)

coord = mi.Vector2u(index % width, index // width)
coord = dr.float_array_t(coord)(coord)

sample_dir = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

interaction = mi.Interaction3f(
    t=0, time=0, p=position, n=normal, wavelengths=mi.Color0f()
)

sample_dir = mi.Frame3f(normal).to_world(sample_dir)

ray = interaction.spawn_ray(sample_dir)

(spec, _, _) = integrator.sample(scene, sampler, ray, active=mask)

spherical_harmonics = generate_sphere_harmonics(spec, sample_dir)

film, image_block = create_film(width * 4 // args.supersampling, height // args.supersampling)

for i, sh in enumerate(spherical_harmonics):
    image_block.put(
        (coord + mi.ScalarVector2f(width * i, 0)) / args.supersampling,
        [sh.x, sh.y, sh.z, 1.0],
        active=mask,
    )

film.put_block(image_block)

image = film.develop()

mi.util.write_bitmap(f"{args.seed}_{args.sample_count}.exr", image)
