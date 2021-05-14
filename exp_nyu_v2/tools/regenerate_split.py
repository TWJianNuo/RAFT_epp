import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')
import os

def write_to_txt(split, entries):
    filepath = "exp_nyu_v2/splits/nyudepthv2_{}_files.txt".format(split)
    filepath = os.path.join(project_rootdir, filepath)
    with open(filepath, 'w') as file_handler:
        for idx, entry in enumerate(entries):
            if idx == len(entries) - 1:
                file_handler.write("{}".format(entry))
            else:
                file_handler.write("{}\n".format(entry))

split_root = os.path.join(project_rootdir, 'exp_nyu_v2/splits')
train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_train_files_with_gt.txt'), 'r')]
evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'nyudepthv2_test_files_with_gt.txt'), 'r')]

train_entries_organized = list()
for entry in train_entries:
    rgb_path, depth_path, _ = entry.split(' ')

    seq = rgb_path.split('/')[1]
    rgbname = rgb_path.split('/')[2]
    index = int(rgbname.split('.')[0].split('_')[1])
    train_entries_organized.append("{} {}".format(seq, str(index).zfill(5)))

write_to_txt('train', train_entries_organized)

test_entries_organized = list()
for entry in evaluation_entries:
    rgb_path, depth_path, _ = entry.split(' ')

    seq = rgb_path.split('/')[0]
    rgbname = rgb_path.split('/')[1]
    index = int(rgbname.split('.')[0].split('_')[1])
    test_entries_organized.append("{} {}".format(seq, str(index).zfill(5)))
write_to_txt('test', test_entries_organized)