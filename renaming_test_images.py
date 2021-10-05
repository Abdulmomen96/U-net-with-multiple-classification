from glob import glob
images = glob(r'C:\Users\USER\PycharmProjects\U-net-with-multiple-classification\data\sperms\test\*.png')

def get_num(s):
  if 'txt' in s:
    return 1000
  return int(s.replace('.', '_').split('_')[1])

import os


images.sort(key=lambda s: get_num(s))

for n, file in enumerate(images):
  dir = os.path.split(file)[0]
  dir = dir + '\\'
  print(dir)
  file_name = os.path.splitext(file)[1]
  os.rename(file, dir + str(n + 1) + '.png')
  print(file, 'to   ', dir + str(n + 1) + '.png')