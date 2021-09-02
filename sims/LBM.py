import taichi as ti 
import numpy as np
import matplotlib.cm as cm
ti.init(arch=ti.gpu)

Re = 1000
nx, ny = 600, 200
ly = ny-1
cx, cy, r = nx//4, ny//2, ny//9
uLB     = 0.04                  # Velocity in lattice units.
nulb    = uLB*r/Re;             # Viscoscity in lattice units.
omega = 1 / (3*nulb+0.5);    # Relaxation parameter.

# macroscopic variables
fin = ti.Vector.field(9, ti.f32, shape=(nx, ny))
fout = ti.Vector.field(9, ti.f32, shape=(nx, ny))
rho = ti.field(ti.f32, shape=(nx, ny))
u = ti.Vector.field(2, ti.f32, shape=(nx, ny))
u0 = ti.Vector.field(2, ti.f32, shape=ny)

# 
v = ti.Vector.field(2, ti.i32, shape=9)
t = ti.Vector([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36])
E = ti.Vector.field(9, ti.f32, shape=(nx, ny))

#
obstacle = ti.field(ti.i32, shape=(nx, ny))

@ti.kernel
def macroscopic():
    for i, j in rho:
        rho[i, j] = fin[i, j].sum()
        u[i, j] *= 0.0
        for k in ti.static(range(9)):
            u[i, j] += v[k] * fin[i, j][k]
        u[i, j] /= rho[i, j]

@ti.func
def equilibrium():
    for i, j in E: 
        if not obstacle[i, j]:
            usqr = 3/2 * u[i, j].dot(u[i, j])
            for k in ti.static(range(9)):
                vu = 3 * v[k].dot(u[i, j]) # dot?
                E[i, j][k] = rho[i, j] * t[k] * (1 + vu + 0.5*vu**2 - usqr)

@ti.kernel
def collision():
    equilibrium()
    for j in range(ny):
        fin[0, j][0] = E[0, j][0] + fin[0, j][8] - E[0, j][8]
        fin[0, j][1] = E[0, j][1] + fin[0, j][7] - E[0, j][7]
        fin[0, j][2] = E[0, j][2] + fin[0, j][6] - E[0, j][6]

    for i, j in fout:
        fout[i, j] = fin[i, j] - omega * (fin[i, j] - E[i, j])

@ti.kernel
def streaming():
    for i, j in fin:
        if not obstacle[i, j]:
            for k in ti.static(range(9)):
                from_ = ti.Vector([i, j]) - v[k]
                if 0 <= from_.x < nx and 0 <= from_.y < ny:
                    if not obstacle[from_]:
                        fin[i, j][k] = fout[from_][k]
                    else:
                        fin[i, j][k] = fout[i, j][8-k]

@ti.kernel
def apply_bc():
    for j in range(ny):
        fin[-1, j][6] = 2*fin[-2, j][6] - fin[-3, j][6]
        fin[-1, j][7] = 2*fin[-2, j][7] - fin[-3, j][7]
        fin[-1, j][8] = 2*fin[-2, j][8] - fin[-3, j][8]

    for j in range(ny):
        u[0, j] = u0[j]
        summ = 0.0
        for k in ti.static((3, 4, 5)):
            summ += fin[0, j][k]
            summ += fin[0, j][k+3]*2
        rho[0, j] = 1/(1-u[0, j].x) * summ   
      
@ti.kernel
def init_fin():
    for i, j in rho: 
        if obstacle[i, j]:
            rho[i, j] = 1e5
        else:
            rho[i, j] = 1.0
    equilibrium()
    for i, j in fin: fin[i, j] = E[i, j]

def init():
    for i, direction in enumerate([
        [1, 1], [1, 0], [1, -1], [0,  1], [0,  0], [0, -1], [-1,  1], [-1,  0], [-1, -1]
    ]):v[i] = ti.Vector(direction)
        
    def inivel(x, y, d):
        return (1-d) * uLB * (1 + 1e-4*np.sin(y/ly*2*np.pi))
    init_u = np.fromfunction(inivel, (nx, ny, 2), dtype=np.float32)

    u.from_numpy(init_u)
    u0.from_numpy(init_u[0])

    def obstacle_fun(x, y):
        return (x-cx)**2+(y-cy)**2<r**2
    cylinder = np.fromfunction(obstacle_fun, (nx,ny)).astype(np.int32)
    obstacle.from_numpy(cylinder)
    
    init_fin()

if __name__ == "__main__":
    video = True
    init()
    gui = ti.GUI('LBM', res=(nx, ny))
    if video:
        video_manager = ti.VideoManager(output_dir='./', automatic_build=False)
    while gui.running:
        for i in range(40):
            collision()
            streaming()
            apply_bc()
            macroscopic()

        u_field = u.to_numpy()
        u_magnitude = np.linalg.norm(u_field, axis=2)
        img = cm.viridis(u_magnitude*10)
        gui.set_image(img)
        gui.show()
        if video:
            video_manager.write_frame(img)
    if video:
        video_manager.make_video(mp4=False, gif=True)