import numpy as np

class UserInput:
    def init(self):
        self.C = None  # A vector of coefficients of objective function
        self.A = None  # A matrix of coefficients of constraint function
        self.b = None  # A vector of right-hand side numbers
        self.a = None  # The approximation accuracy
        self.size = []
        self.max_problem = True
        self.x = None

    def collect_data(self):
        print("This program is designed to solve Linear Programming Problems (LPP) for both maximization "
              "and minimization objectives (in standart form).")
        self.input_type_of_problem()
        self.input_C()
        self.input_size()
        self.input_A()
        self.input_b()
        self.input_a()

    def input_type_of_problem(self):
        print("Please specify whether you wish to solve a maximization or minimization problem "
              "(Enter 'max' for maximization or 'min' for minimization):")
        type_of_problem = input()
        if type_of_problem == "min":
            self.max_problem = False

    def input_size(self):
        self.size = list(map(int, input("Enter the size of matrix A (example: 4 5): ").split()))

    def input_C(self):
        self.C = list(map(float, input("Enter vector C (example: 2 3 4 0 0): ").split()))
        if not self.max_problem:
            self.C = [-c for c in self.C]

    def input_A(self):
        self.A = []
        print("Enter matrix A:\nexample: 4 5 6 1 0\n         5 1 2 0 1")
        for i in range(self.size[0]):
            line = list(map(float, input().split()))
            self.A.append(line)

    def input_b(self):
        self.b = list(map(float, input("Enter vector b (example: 11 34 20): ").split()))

    def input_a(self):
        self.a = float(input("Enter approximation accuracy (example: 0.01): "))

    def additional_input_for_interior_point(self):
        print("For the Interior-Point method you need to find and provide an initial solution "
              "where all values >= 0.")
        self.x = list(map(float, input("Enter the initial solution (example: 1 2 3 4 5): ").split()))

class SimplexMethod:
    def __init__(self, size, C, A, b, a):
        self.size = size  # size of A
        self.C = np.array(C).astype(np.float64)  # A vector of coefficients of objective function
        self.A = np.array(A).astype(np.float64)  # A matrix of coefficients of constraint function
        self.b = np.array(b).astype(np.float64)  # A vector of right-hand side numbers
        self.a = np.array(a).astype(np.float64)  # The approximation accuracy
        self.B = np.array([[0 for j in range(self.size[0])] for i in range(self.size[0])]).astype(np.float64)  # basis
        self.B_indexes = [0 for i in range(self.size[0])]  # indexes of basis
        self.X_b = np.zeros((1, self.size[0]))      # optimal solution vector
        self.z = 0  # optimal value
        self.C_b = np.zeros((self.size[0]))  # values of non-basic variables in C
        self.non_basis = [0] * (self.size[1] - self.size[0])    # A vector of non-basic variables (indexes)

    def revised_simplex_method(self):
        self.find_basic_variables()
        self.find_non_basic_variables()
        B_inversed = np.linalg.inv(self.B)

        while True:
            entering_vector_inx = self.compute_optimality(B_inversed)

            if entering_vector_inx is None:
                result = np.append([self.z], np.zeros(self.size[1]))
                j = 0
                for i in self.B_indexes:
                    result[i + 1] = self.X_b[j]
                    j += 1
                return result

            B_inv_P_j = B_inversed.dot(self.A[:, entering_vector_inx])

            # check for unbounded solutions
            if all(v <= 0 for v in B_inv_P_j):
                return None

            leaving_vector_index = self.determine_leaving_vector(B_inv_P_j)

            entering_var_index = self.non_basis.index(entering_vector_inx)
            temp = self.non_basis[entering_var_index]
            self.non_basis[entering_var_index] = self.B_indexes[leaving_vector_index]
            self.B_indexes[leaving_vector_index] = temp
            self.B[:, leaving_vector_index] = self.A[:, entering_vector_inx]
            B_inversed = np.linalg.inv(self.B)
            self.C_b[leaving_vector_index] = self.C[self.B_indexes[leaving_vector_index]]

    def find_basic_variables(self):
        for i in range(self.size[0]):
            basis = np.array([0 for i in range(self.size[0])])
            basis[i] = 1
            for j in range(self.size[1]):
                if np.array_equal(self.A[:, j], basis.transpose()):
                    self.B[i][i] = 1
                    self.B_indexes[i] = j

    def find_non_basic_variables(self):
        j = 0
        for i in range(self.size[0]):
            if not(i in self.B_indexes):
                self.non_basis[j] = i
                j += 1

    def compute_optimal_solution(self, B_inversed):
        b_transposed = self.b.transpose()
        self.X_b = (B_inversed.dot(b_transposed))
        self.z = self.C_b.dot(self.X_b.transpose())

    def compute_optimality(self, B_inversed):
        self.compute_optimal_solution(B_inversed)
        cur_size = self.size[1] - self.size[0]  # number of non-basic variables
        z = [0 for i in range(cur_size)]
        temp_product = self.C_b.dot(B_inversed)
        count = 0
        for i in range(cur_size):
            z[i] = temp_product.dot(self.A[:, self.non_basis[i]]) - self.C.transpose()[self.non_basis[i]]
            if z[i] >= 0:
                count += 1

        if count == cur_size:
            return None
        else:
            min_neg_el = np.min(z)
            j = z.index(min_neg_el)  # index of entering vector Pj
        return self.non_basis[j]

    def determine_leaving_vector(self, B_inv_P_j):
        min_ratio = float('inf')
        leaving_vector_index = None

        for i in range(self.size[0]):
            if B_inv_P_j[i] != 0:
                ratio = self.X_b[i] / B_inv_P_j[i]
                if min_ratio > ratio > 0:
                    min_ratio = ratio
                    leaving_vector_index = i
        return leaving_vector_index


class InteriorPointMethod:
    def __init__(self, size, C, A, b, a, alpha, x):
        self.size = size  # size of A
        self.C = np.array(C).astype(np.float64)  # A vector of coefficients of objective function
        self.A = np.array(A).astype(np.float64) # A matrix of coefficients of constraint function
        self.b = np.array(b).astype(np.float64)  # A vector of right-hand side numbers
        self.a = a # The approximation accuracy
        self.x = np.array(x).astype(np.float64) # initial solution vector
        self.alpha = alpha
        self.x_prev = None  # vector x from previous iteration

    def check_data(self):
        if any(v < 0 for v in self.x):
            return False
        Ax = np.dot(self.A, self.x).astype(np.float64)
        if not np.allclose(Ax, self.b, rtol=self.a, atol=self.a):
            return False
        return True

    def interior_point_method(self):
        if not self.check_data():   # check that problem is not infeasible
            return None
        while True:
            self.x_prev = self.x.astype(np.float64)
            D = np.diag(self.x.astype(np.float64))
            AD = np.dot(self.A, D).astype(np.float64)
            Dc = np.dot(D, self.C).astype(np.float64)

            I = np.eye(self.size[1], dtype=np.float64)

            F = np.dot(AD, np.transpose(AD)).astype(np.float64)
            F_inv = np.linalg.inv(F).astype(np.float64)
            G = np.dot(np.transpose(AD), F_inv).astype(np.float64)
            P = np.subtract(I, np.dot(G, AD)).astype(np.float64)

            c_p = np.dot(P, Dc).astype(np.float64)
            nu = np.max(np.absolute(c_p).astype(np.float64))

            y = np.add(np.ones(self.size[1], dtype=np.float64), (self.alpha / nu) * c_p).astype(np.float64)
            self.x = np.dot(D, y).astype(np.float64)

            if np.linalg.norm(np.subtract(self.x, self.x_prev), ord=2) < self.a:
                break

        result = [np.dot(self.C, self.x).astype(np.float64)]
        result.extend(self.x)
        return result
