

# particle
import taichi as ti


n_particles = 1
x = ti.Vector.field(2, ti.f32, shape=n_particles)
u = ti.Vector.field(2, ti.f32, shape=n_particles)

# grid


def P2G():
    pass

def G2P():
    pass

def step():
    P2G()

    G2P()
