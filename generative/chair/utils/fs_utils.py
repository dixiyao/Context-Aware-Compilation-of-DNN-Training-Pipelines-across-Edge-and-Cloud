import os, errno
import glob
import json

def read_file_into_list(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f]
    return data

def write_list_into_file_append(filename, lst):
    with open(filename, 'a') as f:
        for s in lst:
            f.write(s + '\n')

def write_list_into_file(filename, lst):
    with open(filename, 'w') as f:
        for s in lst:
            f.write(s + '\n')

def write_dict_to_file(filename, dct):
    with open(filename, 'w') as f:
        f.write(repr(dct))

def write_json_to_file(filename, dct):
    with open(filename, 'w') as outfile:
        json.dump(dct, outfile, sort_keys=True, indent=4, ensure_ascii=False)

def append_line_to_file(filename, line):
    with open(filename, 'a') as f:
        f.write(line + '\n')

def clear_file(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            print(e.args[1])
            raise

def create_folder(folderPath):    
    try:
        os.mkdir(folderPath)
        print("Directory " , folderPath,  " Created ")
    except FileExistsError:
        print("Directory " , folderPath,  " already exists")

def clear_folder(folderPath):
    files = glob.glob(folderPath)
    for f in files:
        if f is folderPath:
            continue
        try:
            os.remove(f)
        except OSError as e:
            if e.errno != errno.ENOENT:
                print(e.args[1])
                raise

def load_files_folder_into_list(folderPath):
    ret = []
    files = glob.glob(folderPath + '/*.json')
    for f in files:
        if f is folderPath:
            continue
        ret.append(f)
    return ret

# def argsort(seq):
#     # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
#     return sorted(range(len(seq)), key=seq.__getitem__)

def get_all_filenames(root_dir, wildcard):
    # return [s.rsplit('\\')[-1] for s in glob.glob(os.path.join(root_dir, wildcard))]
    return glob.glob(os.path.join(root_dir, wildcard), recursive=True)

# def get_all_filenames(root_dir, extension):
#     """get train or val images
#         return: image list: train or val images list
#     """
#     image_arr = glob.glob(os.path.join(root_dir, 'images/*.jpg'))
#     # print(image_arr)
#     image_nums_arr = [float(s.rsplit('\\')[-1][2:-4]) for s in image_arr]
#     # print(image_nums_arr)
#     sorted_image_arr = arrays.array(image_arr, argsort(image_nums_arr))
#     return sorted_image_arr

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))