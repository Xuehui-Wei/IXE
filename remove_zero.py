#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np 

def find_largest_center_zero_submatrix(arr):
    # this function will return the square matrix size and starting position, 
    rows, cols = arr.shape
    
    # Ensure the array is odd-sized in both dimensions
    if rows % 2 == 0 or cols % 2 == 0:
        raise ValueError("The array must have an odd number of rows and columns.")
    
    # Find the center of the matrix
    middle_row = rows // 2
    middle_col = cols // 2
    
    # Start with size 1 (1x1 submatrix) and expand outward
    max_possible_size = min(middle_row, middle_col)  # Maximum possible half-size from the center
    size = 1  # Start with a 1x1 sub-matrix

    def check_outer_borders_and_neighbors(submatrix, row_start, row_end, col_start, col_end):
        """ Check if any non-zero values exist in the outermost rows and columns
            and their immediate neighbors just outside the submatrix """
        top_row = submatrix[0, :]  # First row
        bottom_row = submatrix[-1, :]  # Last row
        left_col = submatrix[:, 0]  # First column
        right_col = submatrix[:, -1]  # Last column
        
        # Check outer borders
        if np.any(top_row != 0) or np.any(bottom_row != 0) or np.any(left_col != 0) or np.any(right_col != 0):
            return True
        
        # Check neighbors just outside the submatrix if they exist
        if row_start > 0 and np.any(arr[row_start-1, col_start:col_end] != 0):  # Row above
            return True
        if row_end < rows and np.any(arr[row_end, col_start:col_end] != 0):  # Row below
            return True
        if col_start > 0 and np.any(arr[row_start:row_end, col_start-1] != 0):  # Column to the left
            return True
        if col_end < cols and np.any(arr[row_start:row_end, col_end] != 0):  # Column to the right
            return True
        
        return False
    
    while True:
        half_size = size // 2
        
        # Determine the bounds for the current sub-matrix
        row_start = middle_row - half_size
        row_end = middle_row + half_size + 1
        col_start = middle_col - half_size
        col_end = middle_col + half_size + 1
        
        # Check if we're out of bounds
        if row_start < 0 or row_end > rows or col_start < 0 or col_end > cols:
            break  # We've gone beyond the array boundaries
        
        # Extract the current sub-matrix
        submatrix = arr[row_start:row_end, col_start:col_end]
        
        # Check if the outer borders and their immediate neighbors contain any non-zero values
        if check_outer_borders_and_neighbors(submatrix, row_start, row_end, col_start, col_end):
            break  # Stop expanding if non-zero values are found
        
        # Expand the square size by 2 (1x1 -> 3x3 -> 5x5, etc.)
        size += 2
    
    # Return the largest zero sub-matrix size (the previous one was valid)
    final_size = size - 2  # The last valid size was before the break
    if final_size <= 0:
        return None, None  # No valid zero sub-matrix found
    
    half_final_size = final_size // 2
    row_start = middle_row - half_final_size
    row_end = middle_row + half_final_size + 1
    col_start = middle_col - half_final_size
    col_end = middle_col + half_final_size + 1
    
    return final_size, (row_start, col_start)

def delete_zero_rows_and_columns(arr, sub_size = 5, pos =(175, 191), fix = False):#, submatrix_size, position):
    """
    Delete rows and columns that correspond to the zero-only sub-matrix.
    
    Parameters:
    arr (np.ndarray): The original 2D array.
    submatrix_size (int): Size of the zero-only sub-matrix.
    position (tuple): The (row_start, col_start) position of the zero-only sub-matrix.
    
    Returns:
    np.ndarray: The updated array after deleting the specified rows and columns.
    """
    # Get the return from find_largest_center_zero_submatrix
    if fix:
        submatrix_size, position= sub_size, pos
    else:
        submatrix_size, position = find_largest_center_zero_submatrix(arr)
    
    # Get the start positions
    row_start, col_start = position
    
    # Calculate the end positions
    row_end = row_start + submatrix_size
    col_end = col_start + submatrix_size
    
    # Indices to delete (rows and columns)
    rows_to_delete = list(range(row_start, row_end))
    cols_to_delete = list(range(col_start, col_end))
    
    # Delete the rows and columns from the array
    arr_reduced = np.delete(arr, rows_to_delete, axis=0)  # Delete rows
    arr_reduced = np.delete(arr_reduced, cols_to_delete, axis=1)  # Delete columns
    
    return arr_reduced


# In[ ]:




