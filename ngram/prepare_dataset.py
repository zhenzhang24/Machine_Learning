import os
import sys

def process_files(root_dir, heldout_dir):
    total_files = os.listdir(root_dir)
    split = int(0.8 * len(total_files))
    i = 0
    for name in total_files[split:]:
        if os.path.isfile(os.path.join(root_dir, name)):
            sys.stderr.write("hold out file %d %s\n"%(i,
            os.path.join(root_dir, name)))
            old_path = os.path.join(root_dir, name)
            heldout_path = os.path.join(heldout_dir, name)
            os.rename(old_path, heldout_path)
        i = i + 1

process_files(sys.argv[1], sys.argv[2])
