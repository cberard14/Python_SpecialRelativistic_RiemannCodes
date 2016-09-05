This folder contains a special relativistic 1.5 dimensional Riemann solver
made after following Pons, Marti & Muller 2000 (see paper here: http://arxiv.org/abs/astro-ph/0005038)

This is an old code that isn't very well documented; the Riemann class in my C++ folder
(holding the entire hydrodynamics code) is an improvement on this code, but I have
included it here as an easy-to-run stand-alone code which demonstrates the basics of the 
Riemann solver.

The analytic solution plots the analytic solution to an initial condition problem
(Sod shock tube initial condition problem, to be precise, where this entails a
grid with a left side and a right side each with their own values for pressure, 
density, and tangential and perpendicular velocities (respective to the grid) which
interact at time t=0 and continue until the finishing time specified (tf))

The Riemann solver needs to be called once in order to get the boundary values. 
This analytic solution can be plotted against the numerical solution (output of the C++
code in my other repository!)
