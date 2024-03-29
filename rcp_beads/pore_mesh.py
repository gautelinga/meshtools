import argparse
import os
import meshio
import numpy as np
from meshtools.io import numpy_to_dolfin
import dolfin as df

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and check tube mesh using gmsh.")
    parser.add_argument("-Lx", type=float, default=10.0, help="Length")
    parser.add_argument("-Ly", type=float, default=10.0, help="Length")
    parser.add_argument("-Lz", type=float, default=10.0, help="Length")
    parser.add_argument("-R", type=float, default=0.5, help="Radius")
    parser.add_argument("-reps", type=float, default=0.05, help="Regularised radius")
    parser.add_argument("-N", type=int, default=10, help="Number of spheres")
    parser.add_argument("--num_tries", type=int, default=1000, help="Number of tries")
    parser.add_argument("-res", type=float, default=0.03, help="Resolution")
    parser.add_argument("--show", action="store_true", help="Show (XDMF)")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input file (optional)")
    parser.add_argument("-o", "--output", type=str, default="porous", help="Output file (optional)")
    parser.add_argument("--shift_x", type=float, default=0., help="Shift x")
    parser.add_argument("--shift_y", type=float, default=0., help="Shift y")
    parser.add_argument("--shift_z", type=float, default=0., help="Shift z")
    parser.add_argument("-nt", type=int, default=4, help="Number of threads for gmsh")
    return parser.parse_args()

gmsh_code_header = """
Lx={L[0]};
Ly={L[1]};
Lz={L[2]};
x0 = 0; y0 = 0; z0 = 0;
x = 0.; y = 0; z = 0; 
R={R};
eps = 1e-3;
res = {res};
"""

gmsh_code_body_1 = """
SetFactory("OpenCASCADE");
Box(10) = {x0, y0, z0, x0+Lx, y0+Ly, z0+Lz};

"""

gmsh_code_body_2 = """
Sphere(11) = {x, y, z, R};
Sphere(12) = {x+Lx, y, z, R};
Sphere(13) = {x, y+Ly, z, R};
Sphere(14) = {x, y, z+L, R};
Sphere(15) = {x+Lx, y+Ly, z, R};
Sphere(16) = {x, y+Ly, z+Lz, R};
Sphere(17) = {x+Lx, y, z+Lz, R};
Sphere(18) = {x+Lx, y+Ly, z+Lz, R};
Sphere(19) = {x+0.5*Lx, y+0.5*Ly, z+0.5*Lz, R};

vin() = BooleanDifference { Volume{10}; Delete; }{ Volume{11:19}; Delete; };
"""

gmsh_code_body_3 = """
// Ask OpenCASCADE to compute more accurate bounding boxes of entities using the
// STL mesh:
Geometry.OCCBoundsUseStl = 1;

// We now set a uniform mesh size constraint
MeshSize { PointsOf{ Volume{vin()}; }} = res;

//Mesh.MeshSizeExtendFromBoundary = 0;
//Mesh.MeshSizeFromPoints = 0;
//Mesh.MeshSizeFromCurvature = 0;

Mesh.Algorithm = 6; //6
//Mesh.Algorithm = 1;
//Mesh.Algorithm3D = 7;

"""
gmsh_code_body_4 = """
// First we get all surfaces on the left/bottom/front:
Sxmin() = Surface In BoundingBox{x0-eps, y0-eps, z0-eps, x0+Lx+eps, y0+Ly+eps, z0+Lz+eps};

For i In {0:#Sxmin()-1}
  // Then we get the bounding box of each left/bottom/front surface
  bb() = BoundingBox Surface { Sxmin(i) };

  // We translate the bounding box to the right and look for surfaces inside it:
  Sxmax() = Surface In BoundingBox { bb(0)-eps+Lx, bb(1)-eps, bb(2)-eps,
                                     bb(3)+eps+Lx, bb(4)+eps, bb(5)+eps };
  Symax() = Surface In BoundingBox { bb(0)-eps, bb(1)-eps+Ly, bb(2)-eps,
                                     bb(3)+eps, bb(4)+eps+Ly, bb(5)+eps };
  Szmax() = Surface In BoundingBox { bb(0)-eps, bb(1)-eps, bb(2)-eps+Lz,
                                     bb(3)+eps, bb(4)+eps, bb(5)+eps+Lz };
  // For all the matches, we compare the corresponding bounding boxes...
  For j In {0:#Sxmax()-1}
    bbX() = BoundingBox Surface { Sxmax(j) };
    bbX(0) -= Lx;
    bbX(3) -= Lx;
    // ...and if they match, we apply the periodicity constraint
    If(Fabs(bbX(0)-bb(0)) < eps && Fabs(bbX(1)-bb(1)) < eps &&
       Fabs(bbX(2)-bb(2)) < eps && Fabs(bbX(3)-bb(3)) < eps &&
       Fabs(bbX(4)-bb(4)) < eps && Fabs(bbX(5)-bb(5)) < eps)
      Periodic Surface {Sxmax(j)} = {Sxmin(i)} Translate {Lx,0,0};
    EndIf
  EndFor

  For j In {0:#Symax()-1}
    bbY() = BoundingBox Surface { Symax(j) };
    bbY(1) -= Ly;
    bbY(4) -= Ly;
    If(Fabs(bbY(0)-bb(0)) < eps && Fabs(bbY(1)-bb(1)) < eps &&
       Fabs(bbY(2)-bb(2)) < eps && Fabs(bbY(3)-bb(3)) < eps &&
       Fabs(bbY(4)-bb(4)) < eps && Fabs(bbY(5)-bb(5)) < eps)
      Periodic Surface {Symax(j)} = {Sxmin(i)} Translate {0,Ly,0};
    EndIf
  EndFor

  For j In {0:#Szmax()-1}
    bbZ() = BoundingBox Surface { Szmax(j) };
    bbZ(2) -= Lz;
    bbZ(5) -= Lz;
    If(Fabs(bbZ(0)-bb(0)) < eps && Fabs(bbZ(1)-bb(1)) < eps &&
       Fabs(bbZ(2)-bb(2)) < eps && Fabs(bbZ(3)-bb(3)) < eps &&
       Fabs(bbZ(4)-bb(4)) < eps && Fabs(bbZ(5)-bb(5)) < eps)
      Periodic Surface {Szmax(j)} = {Sxmin(i)} Translate {0,0,Lz};
    EndIf
  EndFor
EndFor
"""

def generate_gmsh_code_body(x_ext, x_cnt, R, r, res):
    code = gmsh_code_body_1
    for i in range(len(x_ext)):
        xloc = x_ext[i, :]
        code += f"Sphere({11+i}) = {{{xloc[0]}, {xloc[1]}, {xloc[2]}, {R}}};\n"

    for i in range(len(x_cnt)):
        xloc = x_cnt[i, :]
        code += f"Sphere({11+len(x_ext)+i}) = {{{xloc[0]}, {xloc[1]}, {xloc[2]}, {r}}};\n"
    
    if True:
        code += f"vin() = BooleanDifference {{ Volume{{10}}; Delete; }}{{ Volume{{11:{11+len(x_ext)+len(x_cnt)-1}}}; Delete; }};\n"
    elif True:
        code += f"vin() = BooleanIntersection {{ Volume{{11:{11+len(x_ext)+len(x_cnt)-1}}}; Delete; }}{{ Volume{{10}}; Delete; }};\n"
    else:
        code += f"v() = BooleanFragments {{ Volume{{10}}; Delete; }}{{ Volume{{11:{11+len(x_ext)+len(x_cnt)-1}}}; Delete; }};\n"
        code += f"""
    Geometry.OCCBoundsUseStl = 1;
    vin() = Volume In BoundingBox {{x0-eps, y0-eps, z0-eps, x0+Lx+eps, y0+Ly+eps, z0+Lz+eps}};
v() -= vin();
Recursive Delete{{ Volume{{v()}}; }}
"""

    code += gmsh_code_body_3
    code += gmsh_code_body_4

    if False:
        for i in range(len(x_ext)):
            xloc = x_ext[i, :]
            code += f"Point({10000+i}) = {{{xloc[0]},{xloc[1]},{xloc[2]},0.04}};\n"

        for i in range(len(x_cnt)):
            xloc = x_cnt[i, :]
            code += f"Point({10000+len(x_ext)+i}) = {{{xloc[0]},{xloc[1]},{xloc[2]},0.04}};\n"

        code += f"""Field[1] = Distance;
Field[1].PointsList = {{10000:{10000+len(x_ext)-1}}};
Field[2] = Distance;
Field[2].PointsList = {{{10000+len(x_ext)}:{10000+len(x_ext)+len(x_cnt)-1}}};

Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = {res/4};
Field[3].SizeMax = {res};
Field[3].DistMin = {r};
Field[3].DistMax = {R};

Field[4] = Threshold;
Field[4].InField = 1;
Field[4].SizeMin = {res/2};
Field[4].SizeMax = {res};
Field[4].DistMin = {R};
Field[4].DistMax = {1.5*R};

Field[5] = Min;
Field[5].FieldsList = {{3,4}};

Background Field = 5;
    """

    return code

def place_spheres(L, R, N, num_tries):
    pos = []
    tries = 0
    while len(pos) < N and tries < num_tries-1:
        tries = 0
        found = False
        x = np.array(pos)
        for tries in range(num_tries):
            xloc = np.random.rand(3).flatten() * L
            if len(x) == 0:
                found = True
            else:
                dx = x - np.outer(np.ones(len(x)), xloc)
                dist2 = np.zeros(len(dx))
                for i in range(3):
                    dist2[:] += np.minimum(abs(dx[:, i]), L-abs(dx[:, i]))**2
                if np.all(dist2 > 4*R*R):
                    found = True
            if found:
                break
        if found:
          pos.append(xloc)
    return np.array(pos)

def extend_spheres(x, L, R):
    pos_ext = []
    xshift = np.array([L[0], 0, 0])
    yshift = np.array([0, L[1], 0])
    zshift = np.array([0, 0, L[2]])
    shift = [xshift, yshift, zshift]
    for xloc in x:
        is_x_l = xloc < 1*R
        is_x_h = xloc > L - 1*R
        is_x = np.vstack([is_x_l, is_x_h])

        # 8 + 12 + 6 cases?
        if is_x_l.all():
            pos_ext.append(xloc + sum(shift))
    
        for k1 in range(2):
            for k2 in range(2):
                for k3 in range(2):
                    if is_x[k1, 0] and is_x[k2, 1] and is_x[k3, 2]:
                        pos_ext.append(xloc + (-1)**k1*shift[0] + (-1)**k2*shift[1] + (-1)**k3*shift[2])

        for d1 in range(3):
            for d2 in range(3):
                if d2 != d1:
                    for k1 in range(2):
                        for k2 in range(2):
                            if is_x[k1, d1] and is_x[k2, d2]:
                                pos_ext.append(xloc + (-1)**k1 * shift[d1] + (-1)**k2 * shift[d2])
        
        for d in range(3):
            for k in range(2):
                if is_x[k, d]:
                    pos_ext.append(xloc + (-1)**k * shift[d])

    x_ext = np.array(pos_ext)
    return np.vstack([x, x_ext])

def xyz_shift(x, dx, L):
    assert(len(dx) == len(L))
    xnew = np.zeros_like(x)
    for d in range(len(L)):
        dx[d] = np.remainder(dx[d], L[d])
        xnew[:, d] = x[:, d] + dx[d]  # np.outer(np.ones(len(x)), dx[d])
        xnew[:, d] = np.remainder(xnew[:, d], L[d])
    return xnew


if __name__ == "__main__":
    args = parse_args()

    tol = 1e-1
    fname = args.output

    L = np.array([args.Lx, args.Ly, args.Lz])
    ddx = np.array([args.shift_x, args.shift_y, args.shift_z])

    np.random.seed(0)

    if args.input is None:
        x = place_spheres(L, args.R, args.N, args.num_tries)
        #print("num_obst:", len(x))
    else:
        x = np.loadtxt(args.input)
        #import matplotlib.pyplot as plt
        #plt.hist(x[:, 0])
        #plt.show()
    x = xyz_shift(x, ddx, L)

    x_ext = x

    x_ext = extend_spheres(x, L, args.R)

    dmat = -np.ones((len(x_ext), len(x_ext)))
    for i in range(len(x_ext)):
        for j in range(i):
            dx = x_ext[i, :] - x_ext[j, :]
            #dx = np.minimum(dx, L - dx)
            dmat[i, j] = np.linalg.norm(dx)
    print("min dist:", dmat[dmat > 0].min())

    x_cnt = []
    print(args.reps)
    if args.reps > 0:
        contacts = np.argwhere(np.logical_and(dmat > 0, dmat < 2*args.R+args.res/20))
        for i, j in contacts:
            xmid = 0.5*(x_ext[i, :] + x_ext[j, :])
            dx = np.linalg.norm(x_ext[i, :] - x_ext[j, :])
            x_cnt.append(xmid)
    x_cnt = np.array(x_cnt)
    if len(x_cnt) > 1:
        print("cnt before:", len(x_cnt))
        for d in range(3):
            x_cnt = x_cnt[np.logical_and(x_cnt[:, d] >= 0, x_cnt[:, d] < L[d]), :]
        print("cnt after:", len(x_cnt))
        x_cnt = extend_spheres(x_cnt, L, args.reps)
        print("cnt after that again:", len(x_cnt))

    tmpname = "tmp_porous"
    with open(f"{tmpname}.geo", "w") as ofile:
        code = gmsh_code_header.format(L=L, R=args.R, res=args.res)
        code += generate_gmsh_code_body(x_ext, x_cnt, args.R, args.reps, args.res)
        ofile.write(code)

    #os.system(f"gmsh {tmpname}.geo -3 -nt {args.nt}")
    os.system(f"gmsh {tmpname}.geo -2 -nt {args.nt}")

    mesh = meshio.read(f"{tmpname}.msh")
    meshio.write(f"{tmpname}.xdmf", mesh)

    nodes = mesh.points
    cells = [c for c in mesh.cells if c.type == "tetra"]
    #assert(len(cells)==1)
    elems = []
    for cellset in cells:
        elems.append(cellset.data)
    elems = np.vstack(elems)

    #os.remove(f"{tmpname}.geo")
    #os.remove(f"{tmpname}.msh")

    mesh = numpy_to_dolfin(nodes, elems)
    with df.HDF5File(mesh.mpi_comm(), f"{fname}.h5", "w") as h5f:
        h5f.write(mesh, "mesh")
    #meshio.write()

    obst = np.zeros((len(x_ext)+len(x_cnt), 4))
    obst[:len(x_ext), :3] = x_ext
    obst[:len(x_ext), 3] = args.R
    if len(x_cnt):
        obst[len(x_ext):, :3] = x_cnt
        obst[len(x_ext):, 3] = args.reps

    np.savetxt(f"{fname}.obst", obst)

    if args.show:
        with df.XDMFFile(mesh.mpi_comm(), f"{fname}_show.xdmf") as xdmff:
            xdmff.write(mesh)
