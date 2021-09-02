import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

dt = 0.01

res = (512, 512)
u       = ti.Vector.field(2, ti.f32, shape=res)
u_new   = ti.Vector.field(2, ti.f32, shape=res)
u_new_aux = ti.Vector.field(2, ti.f32, shape=res)
p       = ti.field(ti.f32, shape=res)

@ti.func
def linear_interp(a, b, frac):
    return a + (b-a) * frac

@ti.func
def bilinear_interp(f, p):
    m, n = res
    return ti.Vector([p[1]/m, p[0]/n])

@ti.func
def velocity(x) -> ti.Vector:
    
    return None

@ti.func
def backtrace(I, dt):
    x0 = I
    v1 = velocity(x0)
    x1 = x0 - 0.50*dt*v1
    v2 = velocity(x1)
    x2 = x1 - 0.75*dt*v2
    v3 = velocity(x2)
    return x0 - dt*((2/9)*v1 + (1/3)*v2 + (4/9)*v3)

@ti.func
def semi_lagrangian(q, q_new, dt):
    for I in ti.grouped(q_new):
        q_new[I] = bilinear_interp(q, backtrace(I, dt))

@ti.kernel
def advect():
    semi_lagrangian(u, u_new, dt)
    semi_lagrangian(u_new, u_new_aux, -dt)

    for I in ti.grouped(u):
        u_new[I] = u_new[I] + 0.5 * (u[I] - u_new_aux[I])
        
        u[I] = u_new[I]
    

@ti.kernel
def project():
    pass

def step():
    advect()
    project()

if __name__=="__main__":
    gui = ti.GUI("Semi-Lagrangian", res=res)
    while gui.running:
        step()
        u_field = np.linalg.norm(u.to_numpy(), axis=2)
        gui.set_image(u_field)
        gui.show()