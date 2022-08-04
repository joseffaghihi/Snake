import os
import shutil


def delete_file(path):
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (path, e))


def empty_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
