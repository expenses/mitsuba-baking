import sys
import mitsuba as mi
import drjit as dr

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

scene = sys.argv[1]
positions = sys.argv[2]
normals = sys.argv[3]

dr.set_log_level(dr.LogLevel.Info)
mi.set_variant("llvm_ad_rgb")

scene = mi.load_file(scene)

positions = mi.TensorXf(mi.Bitmap(positions))
normals = mi.TensorXf(mi.Bitmap(normals))

integrator = mi.load_dict({"type": "path"})

sample_count = 1024
seed = 0x5EED

sampler = mi.load_dict({"type": "independent", "sample_count": sample_count})
sampler.seed(seed, positions.shape[0] * positions.shape[1] * sample_count)

index = dr.repeat(
    dr.arange(dr.llvm.UInt, positions.shape[0] * positions.shape[1]), sample_count
)

pixel_coord = mi.Vector3u(index * 4, index * 4 + 1, index * 4 + 2)

position = dr.gather(mi.Vector3f, positions.array, pixel_coord)
normal = dr.gather(mi.Vector3f, normals.array, pixel_coord)

mask = dr.neq(normal.x, 0) | dr.neq(normal.y, 0) | dr.neq(normal.z, 0)

coord = mi.Vector2u(index % positions.shape[0], index // positions.shape[0])
coord = dr.float_array_t(coord)(coord)

sample_dir = mi.warp.square_to_uniform_hemisphere(sampler.next_2d())

interaction = mi.Interaction3f(
    t=0, time=0, p=position, n=normal, wavelengths=mi.Color0f()
)

sample_dir = mi.Frame3f(normal).to_world(sample_dir)

ray = interaction.spawn_ray(sample_dir)

(spec, _, _) = integrator.sample(scene, sampler, ray, active=mask)

spherical_harmonics = generate_sphere_harmonics(spec, sample_dir)

sh0_film, sh0_image_block = create_film(positions.shape[0], positions.shape[1])
sh1x_film, sh1x_image_block = create_film(positions.shape[0], positions.shape[1])
sh1y_film, sh1y_image_block = create_film(positions.shape[0], positions.shape[1])
sh1z_film, sh1z_image_block = create_film(positions.shape[0], positions.shape[1])


sh0_image_block.put(
    coord,
    [spherical_harmonics[0].x, spherical_harmonics[0].y, spherical_harmonics[0].z, 1.0],
)
sh0_film.put_block(sh0_image_block)

sh1x_image_block.put(
    coord,
    [spherical_harmonics[1].x, spherical_harmonics[1].y, spherical_harmonics[1].z, 1.0],
)
sh1x_film.put_block(sh1x_image_block)

sh1y_image_block.put(
    coord,
    [spherical_harmonics[2].x, spherical_harmonics[2].y, spherical_harmonics[2].z, 1.0],
)
sh1y_film.put_block(sh1y_image_block)

sh1z_image_block.put(
    coord,
    [spherical_harmonics[3].x, spherical_harmonics[3].y, spherical_harmonics[3].z, 1.0],
)
sh1z_film.put_block(sh1z_image_block)

sh0 = sh0_film.develop()
sh1x = sh1x_film.develop()
sh1y = sh1y_film.develop()
sh1z = sh1z_film.develop()

sh1x = normalize_value(sh1x, sh0)
sh1y = normalize_value(sh1y, sh0)
sh1z = normalize_value(sh1z, sh0)

mi.util.write_bitmap(f"{seed}_{sample_count}.exr", sh0)
mi.util.write_bitmap("sh1x.exr", sh1x)
mi.util.write_bitmap("sh1y.exr", sh1y)
mi.util.write_bitmap("sh1z.exr", sh1z)
