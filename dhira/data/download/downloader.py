from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

class Downloader:
    """
    Class to download data sets form the web.
    """

    @staticmethod
    def get(url: str, local_file_name: str):
        """
        
        :param url: URL of the file to be downloaded
        :param local_file_name: File path where to be locally to be stored
        :return: 
        """
        if not isfile(local_file_name):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc=local_file_name +'Dataset') as pbar:
                urlretrieve(
                    url,
                    local_file_name,
                    pbar.hook)
        else:
            logger.info('Resusing the file: {}'.format(local_file_name))
        return local_file_name

    @staticmethod
    def extract_tar(file_name: str,  path_to_extract: str):
        """
        Extracts the file contents
        :param file_name: Tar file path
        :return: 
        """
        extracted_folder_name = None
        if not isdir(file_name):
            with tarfile.open(file_name) as tar:
                extracted_folder_name = tar.getnames()[0]
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=path_to_extract)
                tar.close()

        return path_to_extract+'/'+extracted_folder_name #extracted_folder_name needed for cifiar
