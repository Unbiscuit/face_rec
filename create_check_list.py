def create_check_list():
    import os, random, shutil, glob
    for i in range(1, 41):
        try:
            shutil.move(f'used_images/s{i}/{random.choice(os.listdir("used_images/s{}".format(i)))}', f'Data_base/s{i}')
        except:
            pass
    for file in glob.glob('/home/uncookie/PycharmProjects/face_rec/Check_list/*'):
        os.remove(file)
    for k in range(1, 41):
        f = random.choice(os.listdir(f'Data_base/s{k}'))
        shutil.copyfile(f'Data_base/s{k}/{f}', f'newfile{k}.pgm')
        shutil.move(f'newfile{k}.pgm', 'Check_list')
        shutil.move(f'Data_base/s{k}/{f}', f'used_images/s{k}')





