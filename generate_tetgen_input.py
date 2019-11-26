import h5py
import numpy as np
# import os
# import dolfin as df
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate tetgen input.")
    parser.add_argument("-i", type=str, default="rough_surf_mesh.h5",
                        help="Input file.")
    parser.add_argument("-o", type=str, default="rough",
                        help="Output file.")
    args = parser.parse_args()

    with h5py.File(args.i, "r") as h5f:
        node = np.array(h5f["surface/node"])
        face = np.array(h5f["surface/face"])

    # edge_lengths = np.array(
    #     np.linalg.norm(node[face[:, 0], :]-node[face[:, 1], :],
    #                    axis=1).tolist() +
    #     np.linalg.norm(node[face[:, 1], :]-node[face[:, 2], :],
    #                    axis=1).tolist() +
    #     np.linalg.norm(node[face[:, 2], :]-node[face[:, 0], :],
    #                    axis=1).tolist())

    # dx = edge_lengths.mean()

    # mi = MeshInfo()
    # mi.set_points([tuple(el) for el in node.tolist()])
    # mi.set_facets(face.tolist())
    # mesh = build(mi, verbose=True, max_volume=dx**3)
    # print "%d elements" % len(mesh.elements)
    # mesh.save_elements("rough_mesh")
    # mesh.save_nodes("rough_mesh")
    # mesh.write_vtk("rough_mesh.vtk")
    # print np.array(mesh.elements)
    # print np.array(mesh.points)

    # exit("the rest is not done yet")
    with open(args.o + ".node", 'w') as outfile:
        # Information header
        outfile.write("#\n"
                      "# TetGen input file\n"
                      "#\n"
                      "# rough.node\n"
                      "#\n"
                      "# Rough duct or channel in .smesh format.\n"
                      "#\n"
                      "# Created by Gaute Linga, NBI, KU\n"
                      "#\n\n")

        num_nodes = np.size(node, 0)
        np.savetxt(outfile, np.array([(num_nodes, 3, 0, 0)]), fmt='%d')

        node_out = np.zeros((num_nodes, 4))
        node_out[:, 0] = np.arange(1, num_nodes+1)
        node_out[:, 1:4] = node
        np.savetxt(outfile,
                   node_out,
                   fmt='%d\t%1.10f\t%1.10f\t%1.10f')
    with open(args.o + ".smesh", 'w') as outfile:
        # Write header
        outfile.write("#\n"
                      "# TetGen input file\n"
                      "#\n"
                      "# rough.smesh\n"
                      "#\n"
                      "# Rough duct or channel in .smesh format.\n"
                      "#\n"
                      "# Created by Gaute Linga, NBI, KU\n"
                      "#\n\n"
                      "# part 1, node list\n"
                      "#   '0' indicates that the node "
                      "list is stored in 'rough.node'\n")
        np.savetxt(outfile, np.array([(0, 3, 0, 0)]), fmt='%d')
        outfile.write("\n"
                      "# part 2, facet list\n")

        num_faces = np.size(face, 0)
        np.savetxt(outfile, np.array([(num_faces, 0)]), fmt='%d')
        outfile.write("\n")

        face_out = np.zeros((num_faces, 4))
        face_out[:, 0] = 3
        face_out[:, 1:4] = face + 1
        np.savetxt(outfile, face_out, fmt='%d', delimiter='\t')
        outfile.write("\n"
                      "# part 3, hole list\n"
                      "0\n"
                      "\n"
                      "# part 4, region list\n"
                      "0")

        # os.system("tetgen -pq1.1a0.005Yg rough.smesh")
        # os.system("dolfin-convert rough.1.mesh rough.1.xml")

        # mesh = df.Mesh("rough.1.xml")
        # fout = df.File("rough.1.pvd")
        # fout << mesh

if __name__ == "__main__":
    main()
