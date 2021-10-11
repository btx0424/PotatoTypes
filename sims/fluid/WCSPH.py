import taichi as ti 
import numpy as np
ti.init(arch=ti.gpu)

max_num_particles = 2500
max_num_neighbors = 64
res = np.array([600, 600], dtype=int)
screen_to_world_ratio = 35

boundary  = np.array([
    res[0] / screen_to_world_ratio,
    res[1] / screen_to_world_ratio,
])

eps = 5e-4

particle_radius_in_world = 0.25
h = particle_radius_in_world * 1.3
cell_size = h * 2
grid_size = np.ceil(boundary / cell_size).astype(int)

bg_color = 0x112f41
particle_color = 0x068587

gamma = 7
dim = 2
sigma = 10/(7*np.pi*h**dim)

rho0 = 1000
c_0 = 80
B = rho0 * c_0**2 / gamma

alpha = 0.24
viscosity = 2*alpha*h*c_0

g = ti.Vector([0, -9.8])
m = particle_radius_in_world**dim * rho0

dt = 3e-4 # 0.1 * h / c_0

print(sigma, B, m, dt)
@ti.data_oriented
class WCSPH:
    def __init__(self):
        self.static = ti.field(int)
        self.x = ti.Vector.field(2, ti.float32) # position
        self.m = ti.field(ti.float32) # mass
        self.v = ti.Vector.field(2, ti.float32) # velocity
        self.rho = ti.field(ti.float32) # density
        self.p = ti.field(ti.float32) # pressure
        self.num_particles = ti.field(int, ())
        ti.root.dense(ti.i, max_num_particles).place(self.static, self.x, self.m, self.rho)
        ti.root.dense(ti.i, max_num_particles).place(self.p, self.v)
        
        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        grid_snode = ti.root.dense(ti.ij, grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.k, 100).place(self.grid2particles)
        
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        nb_node = ti.root.dense(ti.i, max_num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, 100).place(self.particle_neighbors)

    @ti.func
    def cubic_spline(self, r):
        res = ti.cast(0.0, ti.f32)
        q = r / h
        if q <= 1.0:
            res = 1 - 1.5 * q**2 + 0.75 * q**3
        elif q <= 2.0:
            res = 0.25 * (2 - q)**3
        return sigma * res

    @ti.func
    def cubic_spline_derivative(self, r):
        res = ti.cast(0.0, ti.f32)
        q = r / h
        if q <= 1.0:
            res = 1/h * (-3 * q + 2.25 * q**2)
        elif q <= 2.0:
            res = -0.75 / h * (2 - q)**2
        return sigma * res

    @ti.func
    def get_cell(self, x):
        return (x / cell_size).cast(int)
    
    @ti.pyfunc
    def add_particle(self, x, v, m, static):
        i = self.num_particles[None]
        self.static[i] = static
        self.x[i] = x
        self.v[i] = v
        self.m[i] = m
        self.rho[i] = rho0
        self.num_particles[None] += 1

    @ti.kernel
    def step(self):
        # update grid
        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0
        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        for p_i in range(self.num_particles):
            cell = self.get_cell(self.x[p_i])
            offs = ti.atomic_add(self.grid_num_particles[cell], 1)
            self.grid2particles[cell, offs] = p_i 

        # neighborhood search
        for p_i in range(self.num_particles):
            cell = self.get_cell(self.x[p_i])
            nb_i = 0
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cell_to_check = cell + offset
                if 0<=cell_to_check[0]<grid_size[0] and 0<=cell_to_check[1]<grid_size[1]:
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]
                        if p_i!=p_j and (self.x[p_i] - self.x[p_j]).norm()<h:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i

        # compute density
        for p_i in range(self.num_particles):
            DrhoDt = 0.
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                p_ij = self.x[p_i] - self.x[p_j]
                r = ti.max(p_ij.norm(), eps)
                v_ij = self.v[p_i] - self.v[p_j]
                DrhoDt += self.m[p_j] * self.cubic_spline_derivative(r) * v_ij.dot(p_ij/r)
            self.rho[p_i] += dt * DrhoDt

            # self.rho[p_i] = self.m[p_i] * self.cubic_spline(0)
            # for j in range(self.particle_num_neighbors[p_i]):
            #     p_j = self.particle_neighbors[p_i, j]
            #     p_ij = self.x[p_i] - self.x[p_j]
            #     r = ti.max(p_ij.norm(), 1e-5)
            #     self.rho[p_i] += self.m[p_j] * self.cubic_spline(r)

            self.p[p_i] = B * ((self.rho[p_i]/rho0)**gamma - 1)
            
        # compute grad_p
        for p_i in range(self.num_particles):
            if not self.static[p_i]:
                DvDt = ti.Vector([0., 0.], dt=ti.f32)
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    p_ij = self.x[p_i] - self.x[p_j]
                    v_ij = self.v[p_i] - self.v[p_j]
                    r = ti.max(p_ij.norm(), eps)
                    gW_ij = self.cubic_spline_derivative(r) * p_ij/r
                    DvDt += -self.m[p_j] \
                        * (self.p[p_i]/self.rho[p_i]**2 + self.p[p_j]/self.rho[p_j]**2) \
                        * gW_ij

                    if v_ij.dot(p_ij) < 0.0:
                        DvDt += self.m[p_j] \
                            * viscosity/(self.rho[p_i] + self.rho[p_j]) * v_ij.dot(p_ij)/(r**2 + eps) \
                            * gW_ij
                
                DvDt += g
                self.v[p_i] += dt * DvDt
                self.x[p_i] = self.x[p_i] + dt * self.v[p_i]

                for i in ti.static(range(2)):
                    if self.x[p_i][i] < particle_radius_in_world:
                        self.x[p_i][i] = particle_radius_in_world + eps * ti.random()
                        self.v[p_i][i] *=-0.3
                    elif self.x[p_i][i] > boundary[i]-particle_radius_in_world:
                        self.x[p_i][i] = boundary[i]- particle_radius_in_world - eps * ti.random()
                        self.v[p_i][i] *=-0.3

    def solve(self, output=False):
        gui = ti.GUI('WCSPH', res=res)
        cnt = 0
        
        while gui.running:
            cnt += 1
            for i in range(32):
                self.step()
            pos = self.x.to_numpy()[:self.num_particles[None]]
            for j in range(dim):
                pos[:, j] *= screen_to_world_ratio / res[j]
            
            density = self.rho.to_numpy()[:self.num_particles[None]]
            if cnt==10:
                print(f"density: max:{density.max()}, avg:{density.mean()}")
                cnt = 0

            gui.circles(pos, radius=3, color=particle_color)
            if output:
                gui.show(f"results/sph/frames/{gui.frame:04d}.png")
            else:
                gui.show()

if __name__ == '__main__':
    sph = WCSPH()

    n = 48
    for y in np.linspace(3, 10, n):
        for x in np.linspace(3, 10, n):
            sph.add_particle(ti.Vector([x, y]), ti.Vector([0, 0]), m, False)
    
    print(sph.num_particles)
    sph.solve(False)
