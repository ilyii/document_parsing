import os


def find_images(root):
    if os.path.isfile(root):
        return [root]
    elif os.path.isdir(root):
        return [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".jpg", ".jpeg", ".png"))]
    else:
        exit()
