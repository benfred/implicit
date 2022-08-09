import os
import shutil

# pylint: disable=consider-using-f-string

if __name__ == "__main__":
    # auditwheel will try to vendor cublas/cudart/curand etc - which makes the
    # package unacceptably large. instead of running auditwheel lets just rename
    # to be manylinux
    wheelhouse = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "wheelhouse"
    )
    for filename in os.listdir(wheelhouse):
        if filename.endswith(".whl") and "linux" in filename and "manylinux" not in filename:
            new_filename = filename.replace("linux", "manylinux2014")
            print("moving '%s' to '%s'" % (filename, new_filename))
            shutil.move(os.path.join(wheelhouse, filename), os.path.join(wheelhouse, new_filename))
