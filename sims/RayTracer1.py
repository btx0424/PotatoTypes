import taichi as ti
ti.init(arch=ti.cpu)

res = (800, 640)

pixels = ti.Vector.field(3, ti.f32, shape=(*res, 1))

focal_length = 1.
viewport_height = 2.

def vec3(x, y, z):
    return ti.Vector([x, y, z])

def color(r, g, b):
    return ti.Vector([r, g, b])

img_size    = vec3(*res, 1.)
origin      = vec3(0., 0., 0.)

viewport    = vec3(viewport_height, viewport_height*res[1]/res[0], 0.)
lower_left = origin - viewport/2 - ti.Vector([0., 0., focal_length])

@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    @ti.func
    def at(self, t):
        return self.origin + t*self.direction

@ti.data_oriented
class Sphere:
    def __init__(self, center, radius) -> None:
        self.center = ti.Vector(center)
        self.radius = radius

    @ti.func
    def hit(self, ray):
        oc = ray.origin - self.center
        a = ray.direction.norm_sqr()
        b = 2 * oc.dot(ray.direction)
        c = oc.norm_sqr() - self.radius*self.radius
        d = b*b - 4*a*c
        t = -1.
        if d > 0:
            t = (-b - ti.sqrt(d)) / (2.*a)
        return t

    @ti.func
    def normal_at(self, pos):
        return (pos - self.center).normalized()

s0 = Sphere(center=(0, -100, 0.), radius=100.)
s1 = Sphere(center=(0., 0, -1.), radius=.5)

objects = [s0, s1]

@ti.func
def skybox(direction):
    t = 0.5 * (direction.normalized().y + 1.)
    return (1.-t)*color(1., 1., 1.) + t*color(0.5, 0.7, 1.0)

@ti.func
def ray_color(ray):
    result = skybox(ray.direction)
    s.get(0)
    return result

@ti.kernel
def render():
    for I in ti.grouped(pixels):
        ray = Ray(origin, lower_left + I/img_size*viewport - origin)
        pixels[I] = ray_color(ray)

render()
gui = ti.GUI("Scnene", res=res)
while gui.running:
    img = pixels.to_numpy().squeeze()
    gui.set_image(img)
    gui.show()