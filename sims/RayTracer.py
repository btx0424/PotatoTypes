import taichi as ti
import taichi_glsl as ts

ti.init(arch=ti.opengl)

res = (600, 480)
samples_per_pix = 25
max_steps = 25

img             = ti.Vector.field(3, ti.f32, shape=res)

viewport_height = 2.0
viewport_width = viewport_height * res[0]/res[1]
focal_length = 1.0

cam_origin      = ti.Vector.field(3, ti.f32, ())
cam_vup         = ti.Vector.field(3, ti.f32, ())
cam_direction   = ti.Vector.field(3, ti.f32, ())
cam_horizontal  = ti.Vector.field(3, ti.f32, ())
cam_vertical    = ti.Vector.field(3, ti.f32, ())
cam_lower_left_corner = ti.Vector.field(3, ti.f32, ())

ray_origin      = ti.Vector.field(3, ti.f32, shape=res)
ray_direction   = ti.Vector.field(3, ti.f32, shape=res)


hit_t       = ti.field(ti.f32)
hit_p       = ti.Vector.field(3, ti.f32)
hit_normal  = ti.Vector.field(3, ti.f32)
hit_material= ti.field(ti.i32)
hit_front   = ti.field(ti.i32)

ti.root.dense(ti.ij, res).place(hit_normal)
ti.root.dense(ti.ij, res).place(hit_p, hit_t)
ti.root.dense(ti.ij, res).place(hit_material,  hit_front)

sphere_center = ti.Vector.field(3, ti.f32, 16)
sphere_radius = ti.field(ti.f32, 16)
sphere_n = ti.field(ti.i32, ())
sphere_material = ti.field(ti.i32, 16)

maxn_material = 5
n_material = ti.field(ti.i32, ())
m_color   = ti.Vector.field(3, ti.f32, maxn_material)
m_fuzz    = ti.field(ti.f32, maxn_material)
m_ior = ti.field(ti.f32, maxn_material)
m_type = ti.field(ti.i32, maxn_material)

DIFFUSE = 0
METAL = 1
GLASS = 2

def add_material(type, **kwargs):
    n = n_material[None]
    m_type[n] = type
    if type==DIFFUSE:
        m_color[n] = kwargs['color']
    elif type==METAL:
        m_fuzz[n] = kwargs['fuzz']
        m_color[n] = kwargs['color']
    elif type==GLASS:
        m_ior[n] = kwargs['ior']
        m_color[n] = kwargs['color']
    n_material[None] += 1
    return n

def add_sphere(center, radius, material=None):
    n = sphere_n[None]
    sphere_center[n] = center
    sphere_radius[n] = radius
    sphere_material[n] = material
    sphere_n[None] += 1
    return n

@ti.func
def hit_sphere(s, i, t_min, t_max):
    oc = ray_origin[i] - sphere_center[s]
    a = ray_direction[i].norm_sqr()
    half_b = oc.dot(ray_direction[i])
    c = oc.dot(oc) - sphere_radius[s]*sphere_radius[s]

    d = half_b*half_b - a*c
    
    result = False
    if d>0:
        t = (-half_b - ti.sqrt(d)) / a
        if t<0: t = (-half_b + ti.sqrt(d)) / a
        if t_min < t < t_max:
            result = True
            hit_t[i] = t
            hit_p[i] = at(i, t)
            hit_normal[i] = (hit_p[i] - sphere_center[s]) / sphere_radius[s] 
            hit_front[i] =  ray_direction[i].dot(hit_normal[i]) < 0
            if not hit_front[i]: hit_normal[i] *= -1.
            hit_material[i] = sphere_material[s]
    return result

@ti.func
def at(ray, t) -> ti.Vector:
    return ray_origin[ray] + t * ray_direction[ray]

@ti.func
def random(min, max):
    return ti.random()*(max-min)+min

@ti.func
def refract(I, N, ior_ratio):
    cos_theta = ti.min(-I.dot(N), 1.)
    sin_theta = ti.sqrt(1. - cos_theta*cos_theta)
    result = ts.reflect(I, N)
    if ior_ratio * sin_theta < 1.0:
        r_perp = ior_ratio * (I + cos_theta*N)
        r_para = -ti.sqrt(ti.abs(1 - r_perp.norm_sqr())) * N
        result = r_perp + r_para
    return result

@ti.func
def scatter(i, atten):
    result = ts.vec3(0., 0., 0.)
    m = hit_material[i]
    random_unit = ti.Vector([random(-1., 1.), random(-1., 1.), random(-1., 1.)]).normalized()
    
    if m_type[m] == DIFFUSE: 
        diffuse = hit_normal[i] + random_unit
        ray_direction[i] = diffuse
        result = atten * m_color[m] * 0.5
    elif m_type[m] == GLASS:
        ior = 1/m_ior[m] if hit_front[i] else m_ior[m]
        refracted = refract(ray_direction[i], hit_normal[i], ior)
        ray_direction[i] = refracted
        result = m_color[m] * 0.5
    elif m_type[m] == METAL:
        reflected = ts.reflect(ray_direction[i], hit_normal[i]) + random_unit * m_fuzz[m]
        ray_direction[i] = reflected
        result = atten * m_color[m] * 0.5

    ray_origin[i] = hit_p[i]
    return result 

@ti.func
def ray_color(i):
    color = ti.Vector([0., 0., 0.])
    attenuation = ti.Vector([1., 1., 1.])
    for _ in range(1):
        for __ in range(max_steps):
            closest_so_far = 100000.
            hit_anything = False
            for s in range(sphere_n[None]):
                if hit_sphere(s, i, 0.001, closest_so_far):
                    hit_anything = True
                    closest_so_far = hit_t[i]
            
            if hit_anything:
                attenuation = scatter(i, attenuation)
            else:
                unit_direction = ray_direction[i].normalized()
                t = 0.5*(unit_direction[1] + 1.0)
                color = attenuation*((1.0-t)*ti.Vector([1.0, 1.0, 1.0]) + t*ti.Vector([0.5, 0.7, 1.0]))
                break
    return color
@ti.func
def cam_get_direction(i):
    u = (i[0]+ti.random()) / res[0]
    v = (i[1]+ti.random()) / res[1]      

    return  cam_lower_left_corner[None] \
            + u * cam_horizontal[None] \
            + v * cam_vertical[None] \
            - cam_origin[None]

@ti.func
def sample(i):
    img[i] *= 0
    for _ in range(samples_per_pix):
        ray_origin[i] = cam_origin[None]
        ray_direction[i] = cam_get_direction(i)
        img[i] += ray_color(i)
    img[i] = ti.sqrt(img[i] / samples_per_pix)

@ti.kernel
def render():
    for i in ti.grouped(ray_origin):
        sample(i)

@ti.kernel
def _config_cam():
    u = cam_direction[None].cross(cam_vup[None]) 
    v = u.cross(cam_direction[None])

    cam_horizontal[None] = viewport_width * u
    cam_vertical[None] = viewport_height * v
    cam_lower_left_corner[None] = cam_origin[None] \
        - cam_horizontal[None] / 2 \
        - cam_vertical[None] / 2 \
        + cam_direction[None]

def set_camera(pos=None, direction=None, vup=None):
    if pos: cam_origin[None] = pos
    if direction: cam_direction[None] = direction
    if vup: cam_vup[None] = vup
    _config_cam()
    
if __name__ == "__main__":
    
    gui = ti.GUI(res=res, fast_gui=True)
    
    sphere_n[None] = 0
    m1 = add_material(DIFFUSE, color=[0.5, 0.5, 0.9])
    m2 = add_material(METAL, color=[0.5, 0.5, 0.5], fuzz=0.1)
    
    m3 = add_material(GLASS, color=[1., 1., 1.], ior=1.5)
    m4 = add_material(DIFFUSE, color=[0.8, 0.2, 0.2])
    m5 = add_material(GLASS, color=[1., 1., 1.], ior=1.5)
    m6 = add_material(METAL, color=[0.9, 0.9, 0.1], fuzz=0.)

    ground = add_material(DIFFUSE, color=[0.6, 0.6, 0.7])

    add_sphere(ti.Vector([0, -200.5, -1]), 200, ground)

    add_sphere(ti.Vector([0, 0, -2]), 0.5, m1)
    add_sphere(ti.Vector([-1, -0.5+0.15, -1]), 0.15, m2)
    add_sphere(ti.Vector([-1.2, -0.5+0.4, -2.2]), 0.4, m3)
    add_sphere(ti.Vector([1, -0.5+0.3, -1]), -0.3, m5)
    add_sphere(ti.Vector([-0.6, -0.5+0.2, -1]), 0.2, m4)
    add_sphere(ti.Vector([0.5, -0.5+0.2, -1.5]), 0.2, m6)

    set_camera(
        ti.Vector([0., 0., 0.]),
        ti.Vector([0., 0., -1.]),
        ti.Vector([0., 1., 0.])
    )
    t = 0
    r = 2.35
    # video_manager = ti.VideoManager(output_dir='./', framerate=24, automatic_build=False)
    while gui.running:
        t += 0.02
        pos = ti.Vector([-r*ti.sin(t), 0, -2-r*ti.cos(t)])
        direction = ti.Vector([r*ti.sin(t), 0, r*ti.cos(t)]).normalized()
        set_camera(pos=pos, direction=direction)
        render()
        gui.set_image(img)
        gui.show()
        # video_manager.write_frame(img)
    # video_manager.make_video(mp4=False, gif=True)