# Taichi Stuff

Simulations and renderer in [Taichi](https://taichi.graphics/). Good to know about and maybe useful someday.

* Lattice Boltzmann Method: Flow Over a Cylinder
  * ![lbm.gif](https://github.com/sustc11810424/PotatoTypes/blob/main/sims/gifs/lbm.gif?raw=true)
* Weakly Compressed Smoothed Particle Hydrodynamics
  * From [Eurographics Tutorial 2019 | Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids](https://interactivecomputergraphics.github.io/SPH-Tutorial/)
  * ![sph.gif](https://github.com/sustc11810424/PotatoTypes/blob/main/sims/gifs/sph.gif?raw=true)
  * The handling of pressure is currently erroneous and relies largely on artificial viscosity to be stable.
* Real-Time Ray Tracer
  * From [Ray Tracing in One Weekend Series](https://raytracing.github.io/).
  * ![scene.gif](https://github.com/sustc11810424/PotatoTypes/blob/main/sims/gifs/scene.gif?raw=true)
  * Since Taichi currently doesn't support recursive function call, some materials are not correctly implemented.



