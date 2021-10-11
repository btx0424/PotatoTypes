

# particle
import argparse
import taichi as ti
ti.init(arch=ti.gpu)

gui_res = (512, 512)
max_n_particles = 16384
d = 2
world_size = 128
dx = 1 / world_size
p_vol = dx ** d
E = 400

dt = 0.01

# particle
x = ti.Vector.field(d, ti.f32, shape=max_n_particles)
u = ti.Vector.field(d, ti.f32, shape=max_n_particles)
C = ti.Vector.field(d, ti.f32, shape=max_n_particles)

# grid
grid_u = ti.Vector.field(d, ti.f32, shape=world_size)
grid_m = ti.field(ti.f32, shape=world_size)

def clamp_pos(x):
    return x

def quad_bspline(r):
    result = 0.
    r_norm = r.norm()
    if r_norm < .5:
        result = 3/4 - r.norm_sqr()
    elif r_norm < 1.2:
        result = 0.5 * (1.5 - r_norm)**2
    return result

def P2G_APIC():
    grid_u.fill(0)
    grid_m.fill(0)

    for I in x:
        base = (x[I]-0.5).cast(int)
        fx = x[I] - base.cast(float)

        affine = C[I]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float)-fx)
                weight = 1
                grid_u[base + offset] += weight * (u[I] + affine @ dpos)
                grid_m[base + offset] += weight 

    # 
    for I in grid_u:
        if grid_m[I] > 0:
            grid_u[I] /= grid_m[I]

def G2P():
    for I in x:
        base = (x[I]-0.5).cast(int)

        new_u = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Vector.zero(ti.f32, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = 1
                new_u += weight * grid_u[base + offset]
        
        x[I] = x[I] + new_u*dt
        u[I] = new_u
        C[I] = new_C

def step():
    P2G_APIC()

    G2P()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gif', type=bool, default=False)
    
    gui = ti.GUI("ML-MPM", res=gui_res)
    while gui.running:
        step()
        gui.show()
