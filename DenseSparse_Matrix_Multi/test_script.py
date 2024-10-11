import subprocess
import itertools
import os
from datetime import datetime

# ---------------------------
# Configuration Parameters
# ---------------------------

# Define the matrix sizes
matrix_sizes = [1000, 10000]

# Define the matrix types
matrix_types = ["dd", "ds", "ss"]  # "dd"=Dense-Dense, "ds"=Dense-Sparse, "ss"=Sparse-Sparse

# Define all combinations of optional parameters
optional_params = [
    "",                     # No optional parameters
    "-m 12",
    "-s",
    "-o",
    "-m 12 -s",
    "-m 12 -o",
    "-s -o",
    "-m 12 -s -o"
]

# Define the output file
output_file = "test_results.txt"

# ---------------------------
# Prepare the Output File
# ---------------------------

# Remove the output file if it already exists to avoid appending to old results
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Existing output file '{output_file}' removed.")

# Open the output file in write mode and add a header
with open(output_file, "w") as f:
    f.write("=================================\n")
    f.write("Matrix Multiplication Test Results\n")
    f.write(f"Run Date: {datetime.now()}\n")
    f.write("=================================\n\n")

# ---------------------------
# Iterate Through All Test Cases
# ---------------------------

for size, mat_type, opts in itertools.product(matrix_sizes, matrix_types, optional_params):
    # Construct the command
    command = [".\\matrix_mul.exe", "-n", str(size), "-t", mat_type]
    if opts:
        # Split the optional parameters and extend the command list
        command.extend(opts.split())
    
    # Display the command being executed
    cmd_str = ' '.join(command)
    print(f"Executing: {cmd_str}")

    # Execute the command and capture the output
    try:
        # Run the command and capture stdout and stderr
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # If an error occurs, capture it
        output = f"Error executing command: {e.stderr.strip()}"

    # Capture the current time
    start_time = datetime.now()

    # Log the results to the output file
    with open(output_file, "a") as f:
        f.write("=================================\n")
        f.write("Test Case:\n")
        f.write(f"Matrix Size (n): {size}\n")
        if mat_type == "dd":
            type_desc = "Dense-Dense"
        elif mat_type == "ds":
            type_desc = "Dense-Sparse"
        else:
            type_desc = "Sparse-Sparse"
        f.write(f"Matrix Types: A = {type_desc.split('-')[0]}, B = {type_desc.split('-')[-1]}\n")
        f.write(f"Optional Parameters: {opts if opts else 'None'}\n")
        f.write(f"Start Time: {start_time}\n")
        f.write("Output:\n")
        f.write(f"{output}\n")
        f.write("=================================\n\n")

print(f"All test cases have been executed. Results are saved in '{output_file}'.")
