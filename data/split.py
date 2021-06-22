
import os
import glob

file_num = []
for dir in ['training', 'validation']:
    i = 0
    for _ in glob.iglob(dir + '/**/**', recursive=True):
      i += 1

    print("number of {} files: {}".format(dir, i))
    file_num.append(i)

assert(file_num[0] == file_num[1])

lines_train = [line.rstrip('\n') for line in open('train.txt')]
lines_val = [line.rstrip('\n') for line in open('val.txt')]

for i in lines_train:
  os.remove('training/image_2/'+i+'.png')
  os.remove('training/label_2/'+i+'.txt')
  os.remove('training/velodyne/'+i+'.bin')
  os.remove('training/calib/'+i+'.txt')

for i in lines_val:
  os.remove('validation/image_2/'+i+'.png')
  os.remove('validation/label_2/'+i+'.txt')
  os.remove('validation/velodyne/'+i+'.bin')
  os.remove('validation/calib/'+i+'.txt')

for dir in ['training', 'validation']:
    i = 0
    for _ in glob.iglob(dir + '/image_2/**/**', recursive=True):
      i += 1
    print("number of {} image files: {}".format(dir, i))

    i = 0
    for _ in glob.iglob(dir + '/label_2/**/**', recursive=True):
      i += 1
    print("number of {} lable files: {}".format(dir, i))
    
    i = 0
    for _ in glob.iglob(dir + '/velodyne/**/**', recursive=True):
      i += 1
    print("number of {} velodyne files: {}".format(dir, i))

    i = 0
    for _ in glob.iglob(dir + '/calib/**/**', recursive=True):
      i += 1
    print("number of {} calib files: {}".format(dir, i))