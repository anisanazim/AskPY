import os

test_path = "data_source/txt"
print("Checking test directory contents:", os.listdir(test_path))
print("Test file content preview:")
with open(os.path.join(test_path, "test.txt")) as f:
    print(f.read())
