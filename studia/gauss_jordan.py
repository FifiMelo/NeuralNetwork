def multiply_row(row, number):
    return [x * number for x in row]

def add_row(row1,row2):
    return [x + y for x,y in zip(row1,row2)]

def is_contradictory(row):
    for x in range(len(row) - 1):
        if row[x]:
            return False
    if row[-1]:
        return True
    return False

def is_empty_row(row):
    for cell in row:
        if cell:
            return False
    return True

def gauss_jordan_reduction(matrix):
    if not isinstance(matrix, list):
        print("ERROR: Matrix to reduce must by 2d array of numbers with equal row length.")
        return 0
    length = len(matrix[0])
    for row in matrix:
        if len(row) != length:
            print("ERROR: Matrix to reduce must by 2d array of numbers with equal row length.")
            return 0
        for cell in row:
            if not isinstance(cell, (int, float)):
                print("ERROR: Matrix to reduce must by 2d array of numbers with equal row length.")
                return 0

    for y in range(len(matrix)):
        found = False
        for x in range(y,len(matrix)):
            if matrix[x][y]:
                found = True
                matrix[x], matrix[y] = matrix[y], matrix[x]
                break
        if not found:
            continue
        matrix[y] = multiply_row(matrix[y],1/matrix[y][y])        
        for x in range(len(matrix)):
            if not x == y:
                matrix[x] = add_row(matrix[x], multiply_row(matrix[y], matrix[x][y] * -1))
    if is_contradictory(matrix[-1]):
        print("Error: Matrix is contradictory")
        return 0
    if is_empty_row(matrix[-1]):
        del matrix[-1]
        return gauss_jordan_reduction(matrix)
    return matrix


print(gauss_jordan_reduction([[1,2,4],[1,2,4]]))
        