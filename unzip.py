'''解压zip文件到目的目录'''

import zipfile
import os

# zip_src: 需要解压的文件路径
# dst_dir: 解压后文件存放路径
def unzip_file(zip_src, dst_dir):
	r = zipfile.is_zipfile(zip_src)
	if r:
		fz = zipfile.ZipFile(zip_src, 'r')
		for file in fz.namelist():
			fz.extract(file, dst_dir)
	else:
		print('This is not a zip file !!!')
