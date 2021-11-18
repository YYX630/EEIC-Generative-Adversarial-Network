import os

for i in range(11):
    os.system('ls ./image/'+str(i)+'> ./image_name_list/'+str(i)+'.txt')
    with open('./image_name_list/'+str(i)+'.txt') as f:
        with open('./image_name_list/all_images.txt', mode='a') as g:
            while True:
                l = f.readline()
                if l:
                    new_line = str(i)+'/'+l
                    g.writelines(new_line)
                else: 
                    break
