using Gmsh: Gmsh, gmsh
Gmsh.initialize()

gmsh.model.add("ex_mesh_magneto")

# parameters
const L=0.1;      # beam length
const W=L/20;       # beam width
const T=L/100;     # beam thickness

# const nl=40; # X element size
# const nw=10; # Y element size
# const nt=4; # Z element size


const nl=40; # X element size
const nw=3; # Y element size
const nt=2; # Z element size

#const nl=5; # X element size
#const nw=2; # Y element size
#const nt=2; # Z element size



const lc = 1.0; # characteristic length for meshing
 
geo = gmsh.model.geo
#  fixed section at (0,0,0)
geo.addPoint(0, 0, -T, lc, 1)
geo.addPoint(L, 0, -T, lc, 2)
geo.addPoint(0, W, -T, lc, 3)
geo.addPoint(L, W, -T, lc, 4)
geo.addPoint(0, 0, 0, lc, 5)
geo.addPoint(L, 0, 0, lc, 6)
geo.addPoint(0, W, 0, lc, 7)
geo.addPoint(L, W, 0, lc, 8)


geo.addLine(1, 2, 1)
geo.addLine(3, 4, 2)
geo.addLine(1, 3, 3)
geo.addLine(2, 4, 4)
geo.addLine(5, 6, 5)
geo.addLine(7, 8, 6)
geo.addLine(5, 7, 7)
geo.addLine(6, 8, 8)
geo.addLine(1,5,9)
geo.addLine(3,7,10)
geo.addLine(2,6,11)
geo.addLine(4,8,12)


geo.addCurveLoop([1,4,-2,-3], 1)
geo.addCurveLoop([5,8,-6,-7], 2)
geo.addCurveLoop([1, 11, -5, -9], 3)
geo.addCurveLoop([2, 12, -6, -10], 4)
geo.addCurveLoop([3, 10, -7, -9], 5)
geo.addCurveLoop([4, 12, -8, -11], 6)

for i in [1,2,5,6]
      geo.mesh.setTransfiniteCurve(i, nl )  
end
for i in [3,4,7,8]
      geo.mesh.setTransfiniteCurve(i, nw )  
end
for i in [9,10,11,12]
      geo.mesh.setTransfiniteCurve(i, nt )  
end
for i in 1:6
    geo.addPlaneSurface([i], i)
    geo.mesh.setTransfiniteSurface(i)
    geo.mesh.setRecombine(2,i)
end  

geo.addSurfaceLoop([1,2,3,4,5,6], 1)
geo.addVolume([1], 1)
geo.mesh.setTransfiniteVolume(1)

gmsh.model.geo.extrude([(2, 2)], 0, 0, T , [nt-1], [1],true )


for i in [20,19,28,24]
      geo.mesh.setTransfiniteCurve(i, nt )  
end 
for i in [15,17]
      geo.mesh.setTransfiniteCurve(i, nw )  
end 
for i in [14,16]
      geo.mesh.setTransfiniteCurve(i, nl )  
end 
for i in [25,21,33,29,34]
      geo.mesh.setTransfiniteSurface(i )  
      geo.mesh.setRecombine(2,i)
end 
 
geo.mesh.setTransfiniteVolume(2)
gmsh.model.geo.synchronize()

# Generate mesh
gmsh.model.mesh.generate(3) 
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(0, [1,3,5,7,9,18], 1,"all_fixed")  
gmsh.model.addPhysicalGroup(1, [3,7,9,10,17,19,28], 1, "all_fixed")  
gmsh.model.addPhysicalGroup(2, [5, 33], 1, "all_fixed")  




gmsh.model.addPhysicalGroup(3, [1,2], 1, "Volume")  

gmsh.model.geo.synchronize()

output_file = joinpath(dirname(@__FILE__), "ex_mesh_magneto.msh")
gmsh.write(output_file)

 # Launch the GUI to see the results:
if !("-nopopup" in ARGS)
        gmsh.fltk.run()
end

Gmsh.finalize()