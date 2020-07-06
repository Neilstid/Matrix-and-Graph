import sys
import subprocess
import os
import platform
import random


class Matrix:
    """
    NAME:
        Matrix

    Description:
        Implementation of matrix in python, with some calcul possible on it easily

    Attributes:
       matrix_value : list[list[int]]
        
    Version:
        1.0.0

    Creator:
        Neil FARMER
    """

    # Constructor_____________

    def __init__(self, path=""):
        """
        The constructor for Matrix class.

        Parameters:
           path (str): The path of the file where the matrix is store
        """
        self.matrix_value = []
        if path != "":
            self.new_matrix(path)

    # convert list -> Matrix_____________

    def list_2dimension_convert(self, lst):
        """
        The function to convert a 2-dimension list into a Matrix

        Parameters:
           lst (list): The list to convert into Matrix
        """
        self.matrix_value = []
        for sub_list in lst:
            self.matrix_value.append(sub_list)

    def lists_to_matrix(self, *args):
        """
        The function to convert n list into a Matrix

        Parameters:
           args (list): The list to convert into Matrix
        """
        self.matrix_value = []
        for lst in args:
            self.matrix_value.append(lst)

    def new_matrix(self, path):
        """
        The function to convert a matrix store n a file into a Matrix

        Parameters:
           path (str): The path of the file where the matrix is store
        """
        file = open(path, "r")

        for line in file:
            new_line = list(map(int, str(line.replace("\n", "")).split(" ")))
            self.matrix_value.append(new_line)

    # operation_____________
    def __str__(self):
        """
        The function to convert n list into a Matrix

        Returns:
           str : The representation of the Matrix
        """
        to_return = ""
        for line in self.matrix_value:
            for values in line:
                to_return += str(values) + " "
            to_return += "\n"

        return to_return

    def copy(self, other_matrix):
        """
        The function to copy a Matrix into an other

        Parameters:
           other_matrix (Matrix) : The Matrix to copy
        """
        if type(other_matrix) == Matrix:
            other_matrix = other_matrix.matrix_value

        self.matrix_value = []
        for line in other_matrix:
            self.matrix_value.append(line)

    def __copy__(self, other_matrix):
        """
        The function to copy a Matrix into an other

        Parameters:
           other_matrix (Matrix) : The Matrix to copy
        """
        if type(other_matrix) == Matrix:
            other_matrix = other_matrix.matrix_value

        self.matrix_value = []
        for line in other_matrix:
            self.matrix_value.append(line)

    def __len__(self):
        """
        The function to get the size of the Matrix

        Returns:
           list : The size horizontally and vertically
        """
        # Number of row, Number of column
        return len(self.matrix_value), len(self.matrix_value[0])

    def __add__(self, other):
        """
        The function to add two Matrix

        Parameters:
            other (Matrix) : The matrix to add

        Returns:
           Matrix : Sum of the two matrix
        """
        size_self = self.__len__()
        size_other = other.__len__()
        if size_self[0] == size_other[0] and size_self[1] == size_other[1]:
            to_return = []
            size = self.__len__()
            for horrizontal in range(0, size[0]):
                new_line = []
                for vertical in range(0, size[1]):
                    new_line.append(self.matrix_value[horrizontal][vertical] + \
                                    other.matrix_value[horrizontal][vertical])
                to_return.append(new_line)
            return to_return
        else:
            return -1

    def __sub__(self, other):
        """
        The function to subtract two Matrix

        Parameters:
            other (Matrix) : The matrix to subtract

        Returns:
           Matrix : The subtraction of the two matrix
        """
        size_self = self.__len__()
        size_other = other.__len__()
        if size_self[0] == size_other[0] and size_self[1] == size_other[1]:
            to_return = []
            size = self.__len__()
            for horizontal in range(0, size[0]):
                new_line = []
                for vertical in range(0, size[1]):
                    new_line.append(self.matrix_value[horizontal][vertical] - \
                                    other.matrix_value[horizontal][vertical])
                to_return.append(new_line)
            return to_return
        else:
            return -1

    def __mul__(self, other):
        """
        The function to multiply two Matrix

        Parameters:
            other (Matrix) : The matrix to multiply

        Returns:
           Matrix : The multiplication of the two matrix
        """
        size_self = self.__len__()
        size_other = other.__len__()
        if size_self[1] != size_other[0]:
            return -1

        to_return = []
        for line in range(0, size_self[0]):
            new_line = []
            for values in range(0, size_other[1]):
                sum_values = 0
                for to_sum in range(0, size_self[1]):
                    sum_values += int(self.matrix_value[line][to_sum] * other.matrix_value[to_sum][values])
                new_line.append(sum_values)
            to_return.append(new_line)

        return to_return

    def __pow__(self, other):
        """
        The function to subtract two Matrix

        Parameters:
            other (int) : The power

        Returns:
           Matrix : The matrix at the power other
        """
        size_self = self.__len__()
        if size_self[0] != size_self[1]:
            return -1

        if type(other) == int and other >= 1:
            actual_pow = 1
            memory = Matrix()
            memory.copy(self)

            while actual_pow < other:
                memory.copy(memory * self)
                actual_pow += 1

            return memory

    def __iadd__(self, other):
        """
        The function to add two Matrix

        Parameters:
            other (Matrix) : The matrix to add

        Returns:
           Matrix : Sum of the two matrix
        """
        self.copy(self.__add__(other))
        return self

    def __isub__(self, other):
        """
        The function to subtract two Matrix

        Parameters:
            other (Matrix) : The matrix to subtract

        Returns:
           Matrix : The subtraction of the two matrix
        """
        self.copy(self.__sub__(other))
        return self

    def __imul__(self, other):
        """
        The function to multiply two Matrix

        Parameters:
            other (Matrix) : The matrix to multiply

        Returns:
           Matrix : The multiplication of the two matrix
        """
        self.copy(self.__mul__(other))
        return self

    def __eq__(self, other):
        """
        The function to compare two Matrix

        Parameters:
            other (Matrix) : The matrix to compare

        Returns:
           bool : True if the two matrix are the same, else False
        """
        size_self = self.__len__()
        size_other = other.__len__()
        if size_self[0] != size_other[0] or size_self[1] != size_other[1]:
            return False

        for line in range(0, size_self[0]):
            for values in range(0, size_self[1]):
                if self.matrix_value[line][values] != other.matrix_value[line][values]:
                    return False

        return True

    # Calcul for matrix_____________
    def transpose(self):
        """
        The function to transpose n list into a Matrix

        Returns:
           Matrix : The transpose of the Matrix
        """
        temp = Matrix()
        temp.copy(self)
        size = self.__len__()
        to_return = Matrix()
        for horizontal in range(0, size[0]):
            new_line = []
            for vertical in range(0, size[1]):
                new_line.append(temp.matrix_value[vertical][horizontal])
            to_return.matrix_value.append(new_line)
        return to_return

    def triangular_form(self):
        """
        The function to put the matrix in a triangular form

        Returns:
           Matrix : the Matrix in the triangular form
        """
        pivot_row = 0
        pivot_col = 0
        size = self.__len__()

        copy_of_matrix = []
        for line in self.matrix_value:
            new_line = []
            for values in line:
                new_line.append(values)
            copy_of_matrix.append(new_line)

        while pivot_row < size[0] and pivot_col < size[1]:
            line_max = 0
            val_max = 0
            for new_pivot in range(pivot_row, size[0]):
                if abs(copy_of_matrix[new_pivot][pivot_col]) > val_max:
                    line_max = new_pivot

                if copy_of_matrix[line_max][pivot_col] == 0:
                    pivot_col += 1
                else:
                    swap = copy_of_matrix[pivot_row]
                    copy_of_matrix[pivot_row] = copy_of_matrix[new_pivot]
                    copy_of_matrix[new_pivot] = swap
                    for rows in range(pivot_row + 1, size[0]):
                        coefficient = copy_of_matrix[rows][pivot_col] / copy_of_matrix[pivot_row][pivot_col]
                        copy_of_matrix[rows][pivot_col] = 0
                        for col in range(pivot_col + 1, size[1]):
                            copy_of_matrix[rows][col] = copy_of_matrix[rows][col] - \
                                                        copy_of_matrix[pivot_row][col] * coefficient

                    pivot_row += 1
                    pivot_col += 1
            to_return = Matrix()
            to_return.list_2dimension_convert(copy_of_matrix)
            return to_return

    def gauss_elimination(self, other):
        """
        The function to calcul the gauss elimination

        Parameters:
            other (Matrix) : the sub matrix

        Returns:
           Matrix : the result sub matrix
        """
        pivot_row = 0
        pivot_col = 0
        size = self.__len__()
        size_result = other.__len__()

        copy_of_matrix = []
        for line in self.matrix_value:
            new_line = []
            for values in line:
                new_line.append(values)
            copy_of_matrix.append(new_line)

        copy_of_result = []
        if type(other) == Matrix:
            other = other.matrix_value
        for line in other:
            new_line = []
            for values in line:
                new_line.append(values)
            copy_of_result.append(new_line)

        while pivot_row < size[0] and pivot_col < size[1]:
            line_max = 0
            val_max = 0
            for new_pivot in range(pivot_row, size[0]):
                if abs(copy_of_matrix[new_pivot][pivot_col]) > val_max:
                    line_max = new_pivot

                if copy_of_matrix[line_max][pivot_col] == 0:
                    pivot_col += 1
                else:
                    swap = copy_of_matrix[pivot_row]
                    copy_of_matrix[pivot_row] = copy_of_matrix[new_pivot]
                    copy_of_matrix[new_pivot] = swap

                    swap_result = copy_of_result[pivot_row]
                    copy_of_result[pivot_row] = copy_of_result[new_pivot]
                    copy_of_result[new_pivot] = swap_result

                    for rows in range(pivot_row + 1, size[0]):
                        coef = copy_of_matrix[rows][pivot_col] / copy_of_matrix[pivot_row][pivot_col]
                        copy_of_matrix[rows][pivot_col] = 0
                        for col in range(pivot_col + 1, size[1]):
                            copy_of_matrix[rows][col] = copy_of_matrix[rows][col] - copy_of_matrix[pivot_row][
                                col] * coef
                        for col_result in range(0, size_result[1]):
                            copy_of_result[rows][col_result] = copy_of_result[rows][col_result] - \
                                                               copy_of_result[pivot_row][col_result] * coef

                    pivot_row += 1
                    pivot_col += 1

        pivot_row = size[0] - 1
        pivot_col = size[1] - 1

        while pivot_row > 0 and pivot_col > 0:
            line_max = 0
            val_max = 0
            for new_pivot in range(pivot_row, -1, -1):
                if abs(copy_of_matrix[new_pivot][pivot_col]) > val_max:
                    line_max = new_pivot

                if copy_of_matrix[line_max][pivot_col] == 0:
                    pivot_col -= 1
                else:
                    swap = copy_of_matrix[pivot_row]
                    copy_of_matrix[pivot_row] = copy_of_matrix[new_pivot]
                    copy_of_matrix[new_pivot] = swap

                    swap = copy_of_result[pivot_row]
                    copy_of_result[pivot_row] = copy_of_result[new_pivot]
                    copy_of_result[new_pivot] = swap

                    for rows in range(pivot_row - 1, -1, -1):
                        coef = copy_of_matrix[rows][pivot_col] / copy_of_matrix[pivot_row][pivot_col]
                        copy_of_matrix[rows][pivot_col] = 0
                        for col in range(pivot_col - 1, -1, -1):
                            copy_of_matrix[rows][col] = copy_of_matrix[rows][col] - copy_of_matrix[pivot_row][
                                col] * coef
                        for col_result in range(0, size_result[1]):
                            copy_of_result[rows][col_result] = copy_of_result[rows][col_result] - \
                                                               copy_of_result[pivot_row][col_result] * coef
                    pivot_row -= 1
                    pivot_col -= 1

        for row in range(0, size[0]):
            for column in range(0, size_result[1]):
                copy_of_result[row][column] = copy_of_result[row][column] / copy_of_matrix[row][row]
            copy_of_matrix[row][row] = 1

        to_return = Matrix()
        to_return.list_2dimension_convert(copy_of_result)
        return to_return

    def determinant(self):
        """
        The function to get the determinant

        Returns:
           int : determinant
        """
        size = self.__len__()
        if size[0] != size[1]:
            return "Error"

        determinant = 1
        nb_change_line = 0

        copy_of_matrix = []
        for line in self.matrix_value:
            new_line = []
            for values in line:
                new_line.append(values)
            copy_of_matrix.append(new_line)

        for line in range(0, size[0] - 1):
            if copy_of_matrix[line][line] == 0:
                diagonal = 0
                for after_line in range(line + 1, size[0]):
                    diagonal += copy_of_matrix[after_line][after_line]
                if diagonal == 0:
                    return 0

                first_non_nul = 0
                for after_line in range(line + 1, size[0]):
                    if first_non_nul == 0 and copy_of_matrix[after_line][after_line] != 0:
                        first_non_nul = after_line

                for values in copy_of_matrix[first_non_nul]:
                    values = -1 * values

                memory = copy_of_matrix[line]
                copy_of_matrix[line] = copy_of_matrix[first_non_nul]
                copy_of_matrix[first_non_nul] = memory
                nb_change_line += 1
            else:
                for after_line in range(line + 1, size[0]):
                    coefficient = copy_of_matrix[after_line][line] / copy_of_matrix[line][line]
                    for values in range(0, size[0]):
                        copy_of_matrix[after_line][values] += -1 * coefficient * copy_of_matrix[line][values]

        for coefficient in range(0, size[0]):
            determinant = copy_of_matrix[coefficient][coefficient] * determinant

        return determinant * pow(-1, nb_change_line)

    def append(self, lst, tpe):
        """
        The function to add to the end of the matrix

        Parameters:
            lst (list) : list to add in the matrix
            tpe (str) : the type where the list should be add, "c" for the column or "r" for row

        Returns:
           Matrix : Matrix with the list add
        """
        size = self.__len__()
        if str(tpe) == "c":
            if len(lst) != size[0]:
                return -1
            for line in range(0, size[0]):
                self.matrix_value[line].append(lst[line])
            return self
        elif str(tpe) == "r":
            if len(lst) != size[1]:
                return -1
            self.matrix_value.append(lst)
            return self
        else:
            return -1

    def remove(self, index, tpe):
        """
        The function to remove a line or column of the matrix

        Parameters:
            index (int) : index of the row or column that will be delete
            tpe (str) : the type where the list should be add, "c" for the column or "r" for row

        Returns:
           Matrix : Matrix with the row or column removed
        """
        size = self.__len__()
        if str(tpe) == "c":
            for line in range(0, size[0]):
                self.matrix_value[line].pop(index)
            return self
        elif str(tpe) == "r":
            self.matrix_value.pop(index)
            return self
        else:
            return -1

    def invert_matrix(self):
        """
        The function to get the invert of the matrix

        Returns:
           Matrix : the invert Matrix
        """
        if self.determinant() == 0:
            return "Non invertible matrix"
        size = self.__len__()
        diagonnal_matrix = []
        for line in range(0, size[0]):
            new_line = []
            for col in range(0, size[1]):
                if line == col:
                    new_line.append(1)
                else:
                    new_line.append(0)
            diagonnal_matrix.append(new_line)
        d_matrix = Matrix()
        d_matrix.list_2dimension_convert(diagonnal_matrix)
        return self.gauss_elimination(d_matrix)


class Graph:
    """
    NAME:
        Graph

    Description:
        Implementation of Graph in python, with some algorithm

    Attributes:
       graph_matrix : Matrix
       path_file_graph : SvgWriterGraph

    Version:
        1.0.0

    Creator:
        Neil FARMER
    """
    def __init__(self, matrix, path="matrix_svg.html"):
        """
        The function to print a Graph

        Parameters:
            matrix (Matrix) : matrix to represent the graph
            path (str) : path to the file where the graph in svg will be stored
        """
        self.graph_matrix = matrix
        self.path_file_graph = SvgWriterGraph(path, "")

    def __str__(self):
        """
        The function to print a Graph

        Returns:
           str : The representation of the Matrix
        """
        self.print_to_file()
        if platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', self.path_file_graph.file))
        elif platform.system() == 'Windows':  # Windows
            os.startfile(self.path_file_graph.file)
        else:  # linux variants
            subprocess.call(('xdg-open', self.path_file_graph.file))

        return self.graph_matrix.__str__()

    def __copy__(self, other_graph):
        """
        The function to copy a Graph into an other

        Parameters:
            other_graph (Graph) : Graph to copy
        """
        other_graph.graph_matrix = self.graph_matrix

    def print_to_file(self, x_position=[], y_position=[], name=[], color=[], size=[]):
        """
        The function to update the graph svg in file

        Parameters:
            x_position (list[int]) : x position of all node
            Y_position (list[int]) : y position of all node
            name (list[str]) : name of all node
            color (list[str]) : color of all node
            size (list[int]) : size of all node
        """
        self.path_file_graph.clear("")
        size_matrix = self.graph_matrix.__len__()
        if len(x_position) == 0:
            x_position = [x * 60 * ((-1) ** x) + 500 + random.randrange(-30, 30) for x in range(0, size_matrix[0])]
            print(-1 ** 2)
        if len(y_position) == 0:
            y_position = [x * 90 + 30 + random.randrange(-40, 40) for x in range(0, size_matrix[0])]
        if len(name) == 0:
            name = [chr(x + 65) for x in range(0, size_matrix[0])]
        if len(color) == 0:
            color = ["black" for x in range(0, size_matrix[0])]
        if len(size) == 0:
            size = [5 for x in range(0, size_matrix[0])]

        for vertex in range(0, size_matrix[0]):
            self.path_file_graph.print_vertex(x_position[vertex], y_position[vertex], name[vertex], color[vertex],
                                              size[vertex])

        for row in range(0, size_matrix[0]):
            for value in range(0, size_matrix[0]):
                if row == value or self.graph_matrix.matrix_value[row][value] == 0:
                    continue

                if color[row] == color[value]:
                    edge_color = color[row]
                else:
                    edge_color = "black"

                if self.graph_matrix.matrix_value[row][value] == self.graph_matrix.matrix_value[value][row] \
                        and row > value:
                    self.path_file_graph.print_non_oriented_edge(x_position[row], y_position[row],
                                                                 x_position[value], y_position[value], edge_color, 5,
                                                                 self.graph_matrix.matrix_value[row][value])
                elif self.graph_matrix.matrix_value[row][value] != self.graph_matrix.matrix_value[value][row]:
                    self.path_file_graph.print_oriented_ege(x_position[row], y_position[row],
                                                            x_position[value], y_position[value], edge_color, 5,
                                                            self.graph_matrix.matrix_value[row][value])

    def add_vertex(self, link_departure, link_arrival):
        """
        The function to add a vertex in the graph

        Parameters:
            link_departure (list[int]) : list of egde that start from this vertex
            link_arrival (list[int]) : list of egde that end on this vertex
        """
        self.graph_matrix.append(link_arrival[:-1], "c")
        self.graph_matrix.append(link_departure, "r")

    def add_edge(self, vertex_departure, vertex_arrival, weigh):
        """
        The function to add an edge in the graph

        Parameters:
            vertex_departure (int) : index of the vertex where the edge start
            vertex_arrival (int) : index of the vertex where the edge end
            weigh (int) : weigh of the edge
        """
        self.graph_matrix.matrix_value[vertex_departure][vertex_arrival] = weigh

    def num_of_path(self, vertex_departure, vertex_arrival, size_of_path):
        """
        The function to get the nulber of path between 2 vertex of a given size

        Parameters:
            vertex_departure (int) : index of the vertex where the path start
            vertex_arrival (int) : index of the vertex where the path end
            size_of_path (int) : number of edge in the path

        Returns:
            int : The number of path
        """
        path_init = []
        for line in self.graph_matrix.matrix_value:
            new_line = []
            for element in line:
                if int(element) != 0:
                    new_line.append(1)
                else:
                    new_line.append(0)
            path_init.append(new_line)
        matrix_init = Matrix()
        matrix_init.list_2dimension_convert(path_init)

        matrix_result = Matrix()
        matrix_result = matrix_init ** size_of_path
        return matrix_result.matrix_value[vertex_departure][vertex_arrival]

    def dijkstra(self, source):
        """
        The function to get the shortest path to all other edge from an edge source using dijkstra algorithm
        For more information : https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

        Parameters:
            source (int) : index of the vertex where dijkstra's algorithm start

        Returns:
            list[int] : distance to all edge from the source
            list[int] : the predecessor of this edge in the algorithm (you can then find a path from source to all other
                        edge
        """
        unvisited = []
        size = self.graph_matrix.__len__()
        distance = []
        previous = []

        for vertex in range(0, size[0]):
            distance.append(sys.maxsize)
            previous.append(-1)
            unvisited.append(vertex)
        distance[source] = 0

        while len(unvisited) != 0:
            min_distance = sys.maxsize
            for vertex in unvisited:
                if distance[vertex] < min_distance:
                    vertex_selected = vertex

            unvisited.remove(vertex_selected)

            for neighbors in range(0, size[0]):
                if self.graph_matrix.matrix_value[vertex_selected][neighbors] != 0:
                    alternative = distance[vertex_selected] + self.graph_matrix.matrix_value[vertex_selected][neighbors]
                    if alternative < distance[neighbors]:
                        distance[neighbors] = alternative
                        previous[neighbors] = vertex_selected

        return distance, previous

    def dijkstra_target(self, source, target):
        """
        The function to get the shortest path from an edge source to a specific edge using dijkstra algorithm
        For more information : https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

        Parameters:
            source (int) : index of the vertex where dijkstra's algorithm start
            target (int) : index of the vertex where dijkstra's algorithm end

        Returns:
            int : distance from source to the target
            list[int] : the path from source to target
        """
        unvisited = []
        size = self.graph_matrix.__len__()
        distance = []
        previous = []

        for vertex in range(0, size[0]):
            distance.append(sys.maxsize)
            previous.append(-1)
            unvisited.append(vertex)
        distance[source] = 0

        while len(unvisited) != 0:
            min_distance = sys.maxsize
            for vertex in unvisited:
                if distance[vertex] < min_distance:
                    vertex_selected = vertex

            unvisited.remove(vertex_selected)
            if vertex_selected == target:
                break

            for neighbors in range(0, size[0]):
                if self.graph_matrix.matrix_value[vertex_selected][neighbors] != 0:
                    alternative = distance[vertex_selected] + self.graph_matrix.matrix_value[vertex_selected][neighbors]
                    if alternative < distance[neighbors]:
                        distance[neighbors] = alternative
                        previous[neighbors] = vertex_selected

        path_to_target = []
        path_vertex = target
        if previous[path_vertex] != -1 or path_vertex == source:
            while path_vertex != -1:
                path_to_target.insert(0, path_vertex)
                path_vertex = previous[path_vertex]

        return distance[target], path_to_target

    def bfs(self, root):
        """
        The function to get a bfs(Breadth-first search) from the graph.
        For more information : https://en.wikipedia.org/wiki/Breadth-first_search

        Parameters:
            root (int) : the edge will be the root of the tree

        Returns:
            list[int] : the path from source to target
        """
        tree = []

        size = self.graph_matrix.__len__()
        visited = []
        for vertex in range(0, size[0]):
            visited.append(False)

        queue = [root]
        visited[root] = True

        while queue:
            last_leaf = queue.pop(0)
            tree.append(last_leaf)

            for neighbors in range(0, size[0]):
                if self.graph_matrix.matrix_value[last_leaf][neighbors] != 0:
                    if not visited[neighbors]:
                        queue.append(neighbors)
                        visited[neighbors] = True

        return tree

    def dfs(self, root):
        """
        The function to get a dfs(Depth-first search) from the graph.
        For more information : https://en.wikipedia.org/wiki/Depth-first_search

        Parameters:
            root (int) : the edge will be the root of the tree

        Returns:
            list[int] : the path from source to target
        """
        tree = []

        size = self.graph_matrix.__len__()
        visited = []
        for vertex in range(0, size[0]):
            visited.append(False)

        queue = [root]
        visited[root] = True

        while queue:
            last_leaf = queue.pop(0)
            try:
                queue.remove(last_leaf)
                break
            except ValueError:
                pass
            tree.append(last_leaf)

            for neighbors in range(0, size[0]):
                if self.graph_matrix.matrix_value[last_leaf][neighbors] != 0:
                    if not visited[neighbors]:
                        queue.insert(0, neighbors)
                        visited[neighbors] = True
            queue = list(dict.fromkeys(queue))

        return tree

    def degree_of_vertex(self):
        """
        The function to get the degree of each vertex

        Returns:
            list[int] : degree of the vertex
        """
        degree = []
        size = self.graph_matrix.__len__()
        for vertex in range(0, size[0]):
            degree.append(0)

        for row in range(0, size[0]):
            for col in range(0, size[1]):
                if self.graph_matrix.matrix_value[row][col] != 0:
                    degree[row] += 1
                    degree[col] += 1

        return degree

    def welsh_powell(self):
        """
        Function to color the graph with welsh_powell's algorithm
        For more information : https://en.wikipedia.org/wiki/Graph_coloring

        Returns:
            list[int] : the color of the vertex
        """
        size = self.graph_matrix.__len__()
        degree = self.degree_of_vertex()
        vertex = [v for v in range(0, size[0])]

        ordered_vertex = [v for _, v in sorted(zip(degree, vertex), key=None, reverse=True)]
        color_of_vertex = [-1 for color_v in range(0, size[0])]
        color = 0

        for graph_vertex in ordered_vertex:
            list_vertex_color = []
            if color_of_vertex[graph_vertex] == -1:
                color += 1
                color_of_vertex[graph_vertex] = color
                list_vertex_color.append(graph_vertex)
            else:
                continue

            for other_vertex in ordered_vertex:
                if color_of_vertex[other_vertex] != -1:
                    continue

                color_this = True
                for value in list_vertex_color:
                    if self.graph_matrix.matrix_value[value][other_vertex] != 0 or \
                            self.graph_matrix.matrix_value[other_vertex][value] != 0:
                        color_this = False
                if color_this:
                    color_of_vertex[other_vertex] = color
                    list_vertex_color.append(other_vertex)

        return color_of_vertex

    def prim(self):
        """
        The function to find mst(minimum spanning tree) using prim's algorithm.
        For more information : https://en.wikipedia.org/wiki/Prim%27s_algorithm

        Returns:
            list[int] : vertex in minimum spanning tree
        """
        mst = []
        size = self.graph_matrix.__len__()
        in_mst = [False] * size[0]

        in_mst[0] = True

        edge_in_mst = 0
        minimum_cost = 0

        while edge_in_mst < size[0] - 1:
            min_weigh_edge = sys.maxsize
            memory_vertex_first = -1
            memory_vertex_second = -1

            for first_vertex in range(0, size[0]):
                for second_vertex in range(0, size[0]):
                    if self.graph_matrix.matrix_value[first_vertex][second_vertex] == 0:
                        continue

                    if min_weigh_edge > self.graph_matrix.matrix_value[first_vertex][second_vertex]:
                        if in_mst[first_vertex] and not in_mst[second_vertex]:
                            min_weigh_edge = self.graph_matrix.matrix_value[first_vertex][second_vertex]
                            memory_vertex_first = first_vertex
                            memory_vertex_second = second_vertex
                        elif not in_mst[first_vertex] and in_mst[second_vertex]:
                            min_weigh_edge = self.graph_matrix.matrix_value[second_vertex][first_vertex]
                            memory_vertex_first = second_vertex
                            memory_vertex_second = first_vertex

            if memory_vertex_first != -1 and memory_vertex_second != -1:
                new_edge = [memory_vertex_first, memory_vertex_second]
                mst.append(new_edge)
                edge_in_mst += 1
                minimum_cost += min_weigh_edge
                in_mst[memory_vertex_second] = in_mst[memory_vertex_first] = True

        return mst

    def topological_sort_kahn(self):
        """
        The function to find make topological sort using Kahn's algorithm.
        For more information : https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

        Returns:
            list[int] : vertex in topological sort
        """
        degree = [0] * self.graph_matrix.__len__()[0]

        for row in range(0, self.graph_matrix.__len__()[0]):
            for value in range(0, self.graph_matrix.__len__()[0]):
                if self.graph_matrix.matrix_value[row][value] != 0:
                    degree[value] += 1

        queue = []
        for row in range(0, self.graph_matrix.__len__()[0]):
            if degree[row] == 0:
                queue.append(row)

        count = 0
        top = []

        while queue:

            val = queue.pop(0)
            top.append(val)

            for neighbor in range(0, self.graph_matrix.__len__()[0]):
                if self.graph_matrix.matrix_value[val][neighbor] != 0:
                    degree[neighbor] -= 1
                    if degree[neighbor] <= 0:
                        queue.append(neighbor)

            count += 1

        if count != self.graph_matrix.__len__()[0]:
            return -1
        else:
            return top

    def yen_ksp(self, source, sink, k):
        """
        The function to find https://en.wikipedia.org/wiki/Yen%27s_algorithm using Yen's algorithm
        For more information : https://en.wikipedia.org/wiki/Yen%27s_algorithm

        Parameter:
            source (int) : start of path
            sink (int) : end of path
            k (int) : number of path

        Returns:
            list[list[int]] : list of all path found
        """
        _, dijkstra = self.dijkstra_target(source, sink)
        dijkstra_path = [dijkstra]
        potential_short_path = []
        copy_of_graph = Graph(self.graph_matrix)

        for iteration in range(0, k):
            for element in range(0, len(dijkstra_path[iteration]) - 2):
                spur_node = dijkstra_path[iteration][element]
                root_path = dijkstra_path[iteration][0:element]

                for path in dijkstra_path:
                    if root_path == path[0:element]:
                        copy_of_graph.graph_matrix.matrix_value[path[element]][(element + 1) % len(path)] = 0

                for root_path_node in root_path:
                    if root_path_node == spur_node:
                        continue
                    copy_of_graph.graph_matrix.matrix_value[root_path_node] = \
                        [0 for x in range(0, copy_of_graph.graph_matrix.__len__()[0])]

                _, dijkstra = copy_of_graph.dijkstra_target(spur_node, sink)
                spur_path = dijkstra
                total_path = root_path + spur_path

                if total_path not in potential_short_path:
                    potential_short_path.append(total_path)

                copy_of_graph = Graph(self.graph_matrix)

            if len(potential_short_path) == 0:
                break

            min_cost = sys.maxsize
            min_path = []
            for path in potential_short_path:
                cost = 0
                for val in range(0, len(path)):
                    cost += self.graph_matrix.matrix_value[path[val]][path[(val + 1) % len(path)]]
                    if min_cost > cost:
                        min_cost = cost
                        min_path = path
            dijkstra_path.append(min_path)
            potential_short_path.remove(min_path)

        return dijkstra_path


class Tsp(Graph):
    """
    NAME:
        Tsp

    Description:
        Sub class of Graph to solve travelling salesman problem

    Attributes:
       graph_matrix : Matrix
       path_file_graph : SvgWriterGraph
       cycle : list[int]

    Version:
        1.0.0

    Creator:
        Neil FARMER
    """
    def __init__(self, matrix, hamiltonian_cycle=[], path="matrix_svg.html"):
        self.graph_matrix = matrix
        self.path_file_graph = SvgWriterGraph(path, "")
        self.cycle = hamiltonian_cycle

    def print_to_file(self, x_position=[], y_position=[], name=[], color=[], size=[]):
        """
        The function to update the graph svg in file

        Parameters:
            x_position (list[int]) : x position of all node
            Y_position (list[int]) : y position of all node
            name (list[str]) : name of all node
            color (list[str]) : color of all node
            size (list[int]) : size of all node
        """
        self.path_file_graph.clear("")
        size_matrix = self.graph_matrix.__len__()
        if len(x_position) == 0:
            # x_position = [random.randrange(0, 1000) for x in range(0, size_matrix[0])]
            x_position = [x * 60 * ((-1) ** x) + 500 + random.randrange(-10, 10) for x in range(0, size_matrix[0])]
            print(-1 ** 2)
        if len(y_position) == 0:
            # y_position = [random.randrange(0, 1000) for x in range(0, size_matrix[0])]
            y_position = [x * 90 + 30 + random.randrange(-10, 10) for x in range(0, size_matrix[0])]
        if len(name) == 0:
            name = [chr(x + 65) for x in range(0, size_matrix[0])]
        if len(color) == 0:
            color = ["black" for x in range(0, size_matrix[0])]
        if len(size) == 0:
            size = [5 for x in range(0, size_matrix[0])]

        edge_color = [["black" for x in range(0, size_matrix[0])] for y in range(0, size_matrix[0])]
        for col in range(0, len(self.cycle)):
            edge_color[self.cycle[col]][self.cycle[(col + 1) % len(self.cycle)]] = "red"
            if self.graph_matrix.matrix_value[self.cycle[col]][self.cycle[(col + 1) % len(self.cycle)]] == \
                    self.graph_matrix.matrix_value[self.cycle[(col + 1) % len(self.cycle)]][self.cycle[col]] \
                    and self.cycle[col] > self.cycle[(col + 1) % len(self.cycle)]:
                edge_color[self.cycle[(col + 1) % len(self.cycle)]][self.cycle[col]] = "red"

        for vertex in range(0, size_matrix[0]):
            self.path_file_graph.print_vertex(x_position[vertex], y_position[vertex], name[vertex], color[vertex],
                                              size[vertex])

        for row in range(0, size_matrix[0]):
            for value in range(0, size_matrix[0]):
                if row == value or self.graph_matrix.matrix_value[row][value] == 0:
                    continue

                if self.graph_matrix.matrix_value[row][value] == self.graph_matrix.matrix_value[value][row] \
                        and row < value:
                    self.path_file_graph.print_non_oriented_edge(x_position[row], y_position[row],
                                                                 x_position[value], y_position[value],
                                                                 edge_color[row][value], 5,
                                                                 self.graph_matrix.matrix_value[row][value])
                elif self.graph_matrix.matrix_value[row][value] != self.graph_matrix.matrix_value[value][row]:
                    self.path_file_graph.print_oriented_ege(x_position[row], y_position[row],
                                                            x_position[value], y_position[value],
                                                            edge_color[row][value], 5,
                                                            self.graph_matrix.matrix_value[row][value])

    def tsp_cost_evaluation(self, path):
        """
        The function to evaluate the cost of a path

        Parameters:
            path (list[int]) : path to evaluate

        Returns:
            int : cost of the path
        """
        cost = 0
        for element in range(0, len(path)):
            cost += self.graph_matrix.matrix_value[path[element]][path[(element + 1) % len(path)]]

        return cost

    def tsp_nearest_neighbor(self):
        """
        The function to find a path with the nearest neighbor algorithm

        Returns:
            list[int] : path find by the algorithm
        """
        path = []
        size = self.graph_matrix.__len__()
        for row in range(0, size[0]):
            min_value = sys.maxsize
            memory_node = -1
            for values in range(0, size[0]):
                if values == row or values in path:
                    continue

                if self.graph_matrix.matrix_value[row][values] < min_value and \
                        self.graph_matrix.matrix_value[row][values] != 0:
                    min_value = self.graph_matrix.matrix_value[row][values]
                    memory_node = values

            path.append(memory_node)

        final_path = []
        previous = 0
        for path_between in range(0, len(path)):
            final_path.append(path[previous])
            previous = path[previous]

        self.cycle = final_path
        return final_path

    def tsp_tabu_search(self, iteration, length, initial_path=[]):
        """
        The function to do tabu search for TSP

        Parameters:
            iteration (int) : number of iteration to do
            length (int) : length of tabu list
            initial_path  (list[int]) : path from where the algorithm will start

        Returns:
            list[int] : path find by the algorithm
        """
        tabu_list = [-1 for x in range(0, length)]
        if len(initial_path) != 0:
            path = initial_path
        else:
            if len(self.cycle) != 0:
                path = self.cycle
            else:
                path = self.tsp_nearest_neighbor()

        size = self.graph_matrix.__len__()

        best_solution = path
        best_evaluation = self.tsp_cost_evaluation(initial_path)
        for i in range(0, iteration):

            min_value = sys.maxsize
            solution = []
            for element in range(0, size[0]):
                for swaps in range(element + 1, size[0]):
                    copy_of_path = [x for x in path]

                    temp = copy_of_path[swaps]
                    copy_of_path[swaps] = copy_of_path[element]
                    copy_of_path[element] = temp

                    if copy_of_path not in tabu_list and self.tsp_cost_evaluation(copy_of_path) < min_value:
                        solution = copy_of_path
                        min_value = self.tsp_cost_evaluation(copy_of_path)

            tabu_list.pop(0)
            tabu_list.append(solution)
            path = solution

            if min_value < best_evaluation:
                best_solution = path
                best_evaluation = min_value

        return best_solution, best_evaluation


class SvgWriter:
    """
    NAME:
        Tsp

    Description:
        Class to make svg

    Attributes:
       file : str
       id : int

    Version:
        1.0.0

    Creator:
        Neil FARMER
    """
    id = 0

    def __init__(self, file, name="Default name", width=1000, height=1000):
        self.file = str(file)
        if file[-5:] != ".html":
            self.file = self.file.replace(".", "_")
            self.file += ".html"

        svg_file = open(self.file, "w")

        svg_file.write("<!DOCTYPE html> <html> <body> <h1>" + str(name) + "</ h1> \
        <svg width = \"" + str(width) + "\" height = \"" + str(height) + "\"> \
        <defs><marker id=\"arrow\" markerWidth=\"5\" markerHeight=\"5\" refX=\"4\" refY=\"1.5\" orient=\"auto\" \
        markerUnits=\"strokeWidth\"><path d=\"M0,0 L0,3 L5,1 z\" fill=\"auto\" /></marker></defs> \
        Sorry, your browser does not support inline SVG. </svg> </body> </html>")

    def __str__(self):
        return self.file

    def clear(self, name="Default name", width=1000, height=1000):
        """
        The function to clear the file

        Parameters:
            name (str) : name of the svg
            width (int) : width of the svg
            height  (int) : height of the svg
        """
        svg_file = open(self.file, "w")

        svg_file.write("<!DOCTYPE html> <html> <body> <h1>" + str(name) + "</ h1> \
        <svg width = \"" + str(width) + "\" height = \"" + str(height) + "\"> \
        <defs><marker id=\"arrow\" markerWidth=\"5\" markerHeight=\"5\" refX=\"4\" refY=\"1.5\" orient=\"auto\" \
        markerUnits=\"strokeWidth\"><path d=\"M0,0 L0,3 L5,1 z\" fill=\"auto\" /></marker></defs> \
        Sorry, your browser does not support inline SVG. </svg> </body> </html>")

    def add_element(self, line):
        """
        The function to add an element

        Parameters:
            line (str) : line to add in the svg
        """
        svg_file_read = open(self.file, "r")
        svg_file_read_text = svg_file_read.readlines()
        modify_before = str(svg_file_read_text).find("</svg>")
        svg_file_read.close()

        new_svg = str(svg_file_read_text)[:modify_before] + line + str(svg_file_read_text)[modify_before:]
        svg_file_write = open(self.file, "w")
        svg_file_write.write(new_svg[2:-2])
        svg_file_write.close()


class SvgWriterGraph(SvgWriter):
    """
    NAME:
        Tsp

    Description:
        sub class of SvgWritter to make graph in svg

    Attributes:
       file : str
       id : int

    Version:
        1.0.0

    Creator:
        Neil FARMER
    """

    def print_vertex(self, x_position, y_position, name="", color="Black", size=5):
        self.add_element("<circle cx=\"" + str(x_position) + "\" cy=\"" + str(y_position)
                         + "\" r=\"" + str(size) + "\" fill=\"" + str(color) + "\" />")
        if str(name) != "":
            self.add_element("<text x=\"" + str(x_position - size)
                             + "\" y=\"" + str(y_position - size * 2) + "\" fill = \""
                             + str(color) + "\" + font-size=\"" + str(20) + "px\">" + str(name) + "</text>")

    def print_oriented_ege(self, x_position_first, y_position_first, x_position_second, y_position_second,
                           colour="Black", size=5, weigh=""):
        id_edge = "MyId" + str(self.id)
        self.id = self.id + 1
        self.add_element("<path id = \"" + id_edge + "\" marker-end=\"url(#arrow)\" stroke-width=\"" + str(size) + "\" fill=\"none\" \
        stroke=\"" + str(colour) + "\" d=\"M " + str(x_position_first) + " " + str(y_position_first) + "q" +
                         str(int((x_position_second - x_position_first) * 2)) + " "
                         + str(int((y_position_second - y_position_first) / 2)) + " "
                         + str(x_position_second - x_position_first) + " " +
                         str(y_position_second - y_position_first) + "\"/>")
        self.add_element("<text><textPath href=\"#" + str(id_edge) +
                         "\" startOffset=\"50%\" text-anchor=\"middle\" style=\"font-size: 24px;\">" + str(weigh) +
                         "</textPath></text>")

    def print_non_oriented_edge(self, x_position_first, y_position_first, x_position_second, y_position_second,
                                colour="Black", size=5, weigh=""):
        self.add_element("<line x1=\"" + str(x_position_first) + "\" y1=\"" + str(y_position_first)
                         + "\" x2=\"" + str(x_position_second) + "\" y2=\"" + str(y_position_second)
                         + "\" style=\"stroke:" + str(colour) + ";stroke-width:" + str(size) + "\" />")
        if str(weigh) != "":
            val_x = [x for x in range(sorted([x_position_first, x_position_second])[0],
                                      sorted([x_position_first, x_position_second])[1])]
            val_y = [y for y in range(sorted([y_position_first, y_position_second])[0],
                                      sorted([y_position_first, y_position_second])[1])]
            self.add_element("<text x=\"" + str(val_x[int(len(val_x) / 2)] + size)
                             + "\" y=\"" + str(val_y[int(len(val_y) / 2)] + size) + "\" fill = \""
                             + str(colour) + "\" + font-size=\"" + str(20) + "px\">" + str(weigh) + "</text>")