import os

root_dir = r''

def count_files_in_dirs(root_dir):
    path, dirs, files = next(os.walk(root_dir))
    file_count = len(files)
    # print(dirs)
    for d in dirs:
        path, dirs, files = next(os.walk(os.path.join(root_dir, d)))
        # print(f'{i},{len(files)}')
        yield (d, len(files))

for root, dirs, files in os.walk(SOME_DIR):
    for file in files:
        pass
    for dr in dirs:
        pass

