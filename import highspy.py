import highspy
import os

# This usually points to the site-packages folder
path = os.path.dirname(highspy.__file__)
print(f"Look for the 'highs' binary in: {path}")