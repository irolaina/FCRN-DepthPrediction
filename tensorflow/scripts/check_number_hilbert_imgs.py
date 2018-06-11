import glob
import os

# ============================================================================================================== #
#  Check if KittiDiscrete/KittiContinuous have the same number of images than it was supposed to have (RawData)  #
# ============================================================================================================== #
root_folder = "/home/nicolas/remote/olorin_root/media/olorin/Documentos/datasets/kitti/raw_data/*/"

# Finds input images and labels inside list of folders.
folders = glob.glob(root_folder + "*/")

for i, folder in enumerate(folders):
    os.chdir(folder)

    files_image = glob.glob(folder + 'image_00/data/*.png')
    files_hilbert = glob.glob(folder + 'proc_kitti_nick/disp1/*.png')

    detected = "!!!" if len(files_hilbert) != len(files_image) else ''

    print(i, folder, "\t%d/%d\t%s"  % (len(files_hilbert), len(files_image), detected))


# ======================================================================================= #
#  Check if KittiDiscrete/KittiContinuous have the same number of images than KittiDepth  #
# ======================================================================================= #
# Obs: KittiDepth has 10 images (5 @begin, 5 @end) less than KittiContinuous for each scene.

root_folder_depth = "/home/nicolas/remote/olorin_root/media/olorin/hd_nicolas/datasets/kitti/depth/depth_prediction/data/train/*/"
root_folder_hilbert = "/home/nicolas/remote/olorin_root/media/olorin/Documentos/datasets/kitti/raw_data/*/"

# Finds input images and labels inside list of folders.
folders_depth = glob.glob(root_folder_depth)
folders_hilbert = glob.glob(root_folder_hilbert+"*/")

for i, folder in enumerate(folders_depth):
    os.chdir(folder)

    files_depth = glob.glob(folder + 'proj_depth/groundtruth/image_02/*.png')

    scene_depth = folder.split("/")[-2]
    # scene_hilbert = folders_hilbert[0].split("/")[-2]

    for folder2 in folders_hilbert:
        if folder2.find(scene_depth) != -1:
            files_hilbert = glob.glob(folder2 + 'proc_kitti_nick/disp1/*.png')

            # print(folder2)
            # print(files_hilbert)
            # print(len(files_hilbert))

    detected = "!!!" if len(files_hilbert) - 10 != len(files_depth) else ''

    print(i, folder, "\t%d/%d\t%s"  % (len(files_hilbert), len(files_depth), detected))

print("Done.")