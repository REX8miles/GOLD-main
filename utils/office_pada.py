import os
import random
import sys
source = sys.argv[1]
target = sys.argv[2]
p_path = os.path.join('data/', source, 'images')
dir_list = os.listdir(p_path)
class_list_shared = ["back_pack", "bike", "calculator", "headphones", "keyboard", "laptop_computer", "monitor", "mouse", "mug", "projector"]
unshared_list = list(set(dir_list) - set(class_list_shared))
print(class_list_shared)
unshared_list.sort()
source_list = class_list_shared + unshared_list
target_list = class_list_shared
print(target_list)
path_source = "txt/source_%s_pada.txt"%(source)
path_target = "txt/target_%s_pada.txt"%(target)
write_source = open(path_source,"w")
write_target = open(path_target,"w")
for k, direc in enumerate(source_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            class_name = direc
            file_name = os.path.join('data', source, 'images', direc, file)
            write_source.write('%s %s\n' % (file_name, source_list.index(class_name)))
p_path = os.path.join('data/', target,'images')
dir_list = os.listdir(p_path)
for k, direc in enumerate(target_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            file_name = os.path.join('data', target, 'images', direc, file)
            if direc in class_list_shared:
                class_name = direc
                write_target.write('%s %s\n' % (file_name, class_list_shared.index(class_name)))
            elif direc in target_list:
                file_name = os.path.join(p_path, direc, file)
                write_target.write('%s %s\n' % (file_name, len(source_list)))


