import os
from time import time

def check_dir(save_directory):
    if not os.path.isdir(f'save/{save_directory}'):
        print('모델을 저장할 디렉토리를 생성합니다.')
        os.makedirs(f'save/{save_directory}')
        if os.path.isdir(f'save/{save_directory}'):
            print('생성완료')
    elif os.path.isdir(f'save/{save_directory}') and bool(os.listdir(f'save/{save_directory}')):
        print('='*50,f'파일이 덮여씌여지는 것을 방지하기 위해 새로운 디렉토리를 생성합니다. 필요없는 모델은 확인 후 삭제해주세요.', '='*50, sep='\n\n')
        save_dir_list = sorted([folder for folder in os.listdir('save')  if save_directory in os.listdir('save') ])
        if len(save_dir_list[-1].split('_')) == 1:
            save_directory = save_directory + '_1'
        else:
            dir_name, order = save_dir_list[-1].split('_')
            save_directory = dir_name + '_' + str(int(order) + 1)
        os.makedirs(f'save/{save_directory}')
        if os.path.isdir(f'save/{save_directory}'):
            print('생성완료')
    
    return save_directory