import ast
import os
import sys

# Get a list of standard library module names
std_lib = list(sys.builtin_module_names)

# Initialize a set to store the names of the imports
imports = set()

# Recursively scan all Python files in the directory
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            try:
                with open(os.path.join(root, file)) as f:
                    # Parse the file and extract the imports
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            imports.add(node.module)
            except Exception as e:
                print(f'Error parsing {file}: {e}')
                continue

# Filter out standard library imports
imports = [name for name in imports if name not in std_lib]

# Print the imports
for name in sorted(imports):
    print(name)