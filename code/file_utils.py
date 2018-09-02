import os

def get_home_dir(repo_name='stock_prediction'):
    cwd = os.getcwd()
    cwd_list = cwd.split('/')
    repo_position = [i for i, s in enumerate(cwd_list) if s == repo_name]
    if len(repo_position) > 1:
        print("error!  more than one intance of repo name in path")
        return None
    # had to add this hack to import from multiple github folders
    elif len(repo_position) == 0:
        home_dir = '/home/nate/github/' + repo_name + '/'
    else:
        home_dir = '/'.join(cwd_list[:repo_position[0] + 1]) + '/'

    return home_dir
