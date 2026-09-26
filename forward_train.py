import os
import subprocess
import time



def main():
    # Note: Start
    config_list = [
        'CRUW/CE/RC-ROSNet',
        ]
    python_path = '/home/jackychou/software/miniconda3/envs/D2FA-Net/bin/python3'
    code_root = '/home/jackychou/Zhou/code/D2FA-Net'
    data_root = '/home/jackychou/dataset'
    # End
    data_path = "CRUW_train_test"
    config_root = os.path.join(code_root, 'configs')
    log_root = os.path.join(code_root, 'logs')

    if os.path.exists(log_root) is False:
        os.mkdir(log_root)
    for config_path in config_list:
        if 'RC-ROSNet' in config_path:
            train_path = os.path.join(code_root, 'tools/train_RAD_w_IMG.py')
        else:
            train_path = os.path.join(code_root, 'tools/train.py')
        param = f'"{python_path}" "{train_path}" --config "{os.path.join(config_root, config_path + ".py")}" --data_dir "{os.path.join(data_root, data_path)}" --log_dir "{os.path.join(log_root, config_path)}" --code_dir "{code_root}"'
        subprocess.run(param, shell=True)
        time.sleep(60)

if __name__ == '__main__':
    main()