if __name__ == "__main__":
    import networkx as nx
    from pyassimp import load, postprocess, release
    import timeit
    import random
    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy

    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    __device__ float edge_gradient(float x0, float y0, float z0, float x1, float y1, float z1){
        float xdelta = x1 - x0;
        float ydelta = y1 - y0;
        float zdelta = z1 - z0;
        float flat_distance = sqrtf(xdelta * xdelta + zdelta * zdelta);
        float slope = fabsf(ydelta) / flat_distance;
        return slope;
    }

    __device__ float triangle_gradient(float* triangle){
        if(triangle[1] == triangle[4] == triangle[7]){
            return 0.0;
        }

        float g1 = edge_gradient(triangle[0], triangle[1], triangle[2], triangle[3], triangle[4], triangle[5]);
        float g2 = edge_gradient(triangle[3], triangle[4], triangle[5], triangle[6], triangle[7], triangle[8]);
        float g3 = edge_gradient(triangle[6], triangle[7], triangle[8], triangle[0], triangle[1], triangle[2]);

        return (((g1 > g2) && (g1 > g3)) ? g1 : (g2 > g3) ? g2 : g3);
    }

    __global__ void calculate_gradients_GPU(unsigned long int *triangles, float *vertices, unsigned int N)
    {
      long int globalId = blockIdx.x *blockDim.x + threadIdx.x;
      if(globalId < N){
        unsigned int p1Id = triangles[globalId];
        unsigned int p2Id = triangles[globalId+1];
        unsigned int p3Id = triangles[globalId+2];
        float triangle[9] = {vertices[p1Id*3], vertices[p1Id*3+1], vertices[p1Id*3+2],
                            vertices[p2Id*3], vertices[p2Id*3+1], vertices[p2Id*3+2],
                            vertices[p3Id*3], vertices[p3Id*3+1], vertices[p3Id*3+2]};
        triangle_gradient(triangle);
       }
    }
    """)

    calculate_gradients_GPU = mod.get_function("calculate_gradients_GPU")

    #graph
    G = nx.Graph()

    #colors available for coloring
    colors = range(65535)
    #for number in random.sample(xrange(0xFFFFFF), 65535):
    #    colors.add(number)

    #Hash that holds the color of each node
    color_for_node = {}
    #Hash that holds the nodes of each color
    nodes_for_color = {}
    #A hash that holds, for each vertex, the triangles that use it
    triangles_of_vertices = {}

    def neighbor_has_color(node, color):
       for neighbor in G.neighbors(node):
           color_of_neighbor = color_for_node.get(neighbor, None)
           if color_of_neighbor == color:
              return True
       return False

    def get_color(node):
        for color in colors:
           if not neighbor_has_color(node, color):
              return color

    def add_edges_to_neighbors(triangle):
        for vertex in triangle:
            if triangles_of_vertices.has_key(vertex):
                for neighbor in triangles_of_vertices[vertex]:
                    if neighbor != triangle:
                        G.add_edge(triangle,neighbor)

    def graph_coloring():
        for node in G.nodes():
            color = get_color(node)
            color_for_node[node] = color

    def edge_gradient(pt0,pt1):
        xdelta = pt1[0] - pt0[0]
        ydelta = pt1[1] - pt0[1]
        zdelta = pt1[2] - pt0[2]
        flat_distance = (xdelta * xdelta + zdelta * zdelta) ** 0.5 #square root of squares
        slope = abs(ydelta) / flat_distance # 1=45 degree
        return slope

    gradient_memo = {}
    def edge_gradient_memo(pt0,pt1):
        edge_key = frozenset((pt0,pt1))
        if edge_key in gradient_memo:
            return gradient_memo[edge_key]
        xdelta = pt1[0] - pt0[0]
        ydelta = pt1[1] - pt0[1]
        zdelta = pt1[2] - pt0[2]
        flat_distance = (xdelta * xdelta + zdelta * zdelta) ** 0.5 #square root of squares
        slope = abs(ydelta) / flat_distance # 1=45 degree
        gradient_memo[edge_key] = slope
        return slope

    #simplified version that does not account for perfecly vertical triangles
    def triangle_gradient(triangle, vertices, dynamic_programming=False):
        triangle_vertices = [tuple(vertices[triangle[0]]), tuple(vertices[triangle[1]]), tuple(vertices[triangle[2]])]
        if(triangle_vertices[0][1] == triangle_vertices[1][1] and triangle_vertices[1][1] == triangle_vertices[2][1]):
            return 0
        if not dynamic_programming:
            gradient = max(edge_gradient(triangle_vertices[0],triangle_vertices[1]),edge_gradient(triangle_vertices[1],triangle_vertices[2]),edge_gradient(triangle_vertices[2],triangle_vertices[0]))
        else:
            gradient = max(edge_gradient_memo(triangle_vertices[0],triangle_vertices[1]),edge_gradient(triangle_vertices[1],triangle_vertices[2]),edge_gradient(triangle_vertices[2],triangle_vertices[0]))
        return gradient

    def main():
        scene = load('pieta100.obj',processing = postprocess.aiProcess_JoinIdenticalVertices)
        mesh = scene.meshes[0]

        #building the graph
        #TODO: this is eating some points :(
        t0 = timeit.default_timer()
        for face in mesh.faces:
            face_tuple = tuple(face)
            G.add_node(face_tuple)
            for value in face_tuple:
                triangles_of_vertices.setdefault(value,[]).append(face_tuple)
            add_edges_to_neighbors(face_tuple)
        t1 = timeit.default_timer()

        print "Time spent building the graph: ", t1 - t0
        print "Nodes: ", G.number_of_nodes(),"Edges: ", G.number_of_edges()

        #paint the nodes
        t0 = timeit.default_timer()
        graph_coloring()
        t1 = timeit.default_timer()

        print "Time spent coloring the graph: ", t1 - t0

        #populate color:triangles hash
        for node in color_for_node.keys():
            nodes_for_color.setdefault(color_for_node[node],[]).append(node)

        print "Available colors: ", len(colors)
        print "Used colors: ", len(nodes_for_color)

        #naive / standard
        t0 = timeit.default_timer()
        for node in G.nodes():
            triangle_gradient(node, mesh.vertices)
        t1 = timeit.default_timer()

        print "Time spent calculating gradients naively: ", t1 - t0

        #naive / DP
        gradient_memo.clear()
        t0 = timeit.default_timer()
        for node in G.nodes():
            triangle_gradient(node, mesh.vertices, True)
        t1 = timeit.default_timer()

        print "Time spent calculating gradients naively (with memo): ", t1 - t0

        #per color / standard
        t0 = timeit.default_timer()
        for color in nodes_for_color:
            for triangle in nodes_for_color[color]:
                triangle_gradient(triangle, mesh.vertices)
        t1 = timeit.default_timer()

        print "Time spent calculating gradients by color: ", t1 - t0

        #per color / DP
        gradient_memo.clear()
        t0 = timeit.default_timer()
        for color in nodes_for_color:
            for triangle in nodes_for_color[color]:
                triangle_gradient(triangle, mesh.vertices, True)
        t1 = timeit.default_timer()

        print "Time spent calculating gradients by color (with memo): ", t1 - t0

        vertices_numpy = numpy.array(mesh.vertices,dtype = numpy.float32)
        vertices_gpu = drv.mem_alloc(vertices_numpy.nbytes)
        drv.memcpy_htod(vertices_gpu, vertices_numpy)

        t0 = timeit.default_timer()
        for color in nodes_for_color:
            gpu_triangles = numpy.array(nodes_for_color[color],dtype=int).flatten()
            calculate_gradients_GPU(drv.In(gpu_triangles),
                                    vertices_gpu,
                                    numpy.int32(len(gpu_triangles) / 3),
                                    block=(1024,1,1),
                                    grid=(-(-len(G.nodes()) / 1024),1))
        t1 = timeit.default_timer()

        print "Time spent calculating gradients by color (with gpu): ", t1 - t0

        release(scene)

    main()