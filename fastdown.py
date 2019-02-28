import ftplib
import re
import os
from concurrent.futures import ThreadPoolExecutor, Future

import requests


class AtomTask:

    def __init__(self, url, filename, chunk_size=None, offset=None):
        self.url = url
        self.filename = filename
        self.chunk_size = chunk_size
        self.offset = offset
        self.tries = 0

    def __str__(self):
        return self.filename


class BaseDownloader:

    def __init__(self, chunk_parallel=1, chunk_size=8192, retry=5, timeout=3):
        self.chunk_parallel = chunk_parallel
        self.chunk_size = chunk_size
        self.retry = retry
        self.timeout = timeout
        self.executor = None
        self.basedir = ''
        self.process_hook = None

    def set_executor(self, executor):
        self.executor = executor

    def set_basedir(self, basedir):
        self.basedir = basedir

    def set_process_hook(self, func):
        self.process_hook = func

    def __call__(self, task):
        pass

    def failed(self, task):
        print('Failed: {}'.format(task))
        task.tries += 1
        if task.tries < self.retry:
            return self.executor.submit(self, task)
        else:
            raise IOError('Maximum tries exceed! Task: {}'.format(task))


class RequestsDownloader(BaseDownloader):

    def __call__(self, task):
        if task.chunk_size is None:
            return self.single(task)
        else:
            raise NotImplementedError('Chunk download feature hasn\'t been implemented, please be patient!')

    def get_filename_from_header(self, header):
        disposition = header.get('content-disposition')
        if not disposition:
            raise ValueError('Cannot fetch filename!')
        filenames = re.findall('filename=(.+)', disposition)
        if len(filenames) == 0:
            raise ValueError('Cannot fetch filename!')
        return filenames[0]

    def single(self, task):
        try:
            res = requests.get(task.url, allow_redirects=True, timeout=self.timeout)
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            return self.failed(task)
        if task.filename is None:
            task.filename = self.get_filename_from_header(res.headers)
        if self.process_hook:
            self.process_hook(res.content)
        else:
            open(os.path.join(self.basedir, task.filename), 'wb').write(res.content)
        print('Succeed: {}'.format(task))


class FastDown:

    default_service = RequestsDownloader

    def __init__(self, file_parallel=1, chunk_parallel=1, chunk_size=None, retry=5, timeout=3):
        self.file_parallel = file_parallel
        self.chunk_parallel = chunk_parallel
        self.chunk_size = chunk_size
        if self.chunk_size is None and self.chunk_parallel > 1:
            self.chunk_size = 8192
        self.retry = retry
        self.timeout = timeout
        self.service = self.default_service
        self.basedir = ''

    def use_downloader(self, service):
        self.service = service

    def set_basedir(self, basedir):
        self.basedir = basedir

    def set_task(self, task):
        """Task should be arranged like list of (url, filename) tuples."""
        self.targets = []
        if isinstance(task, tuple):
            self.targets.extend(self.create_atom_task(*task))
        elif isinstance(task, str):
            self.targets.extend(self.create_atom_task(task, None))
        else:
            try:
                for t in task:
                    if isinstance(t, tuple):
                        url, filename = t
                    else:
                        url, filename = t, None
                    self.targets.extend(self.create_atom_task(url, filename))
            except TypeError:
                raise ValueError('task must be length-2 tuple or iterable of length-2 tuples.')

    def create_atom_task(self, url, filename):
        if self.chunk_parallel == 1:
            return [AtomTask(url, filename)]
        return [AtomTask(url, filename, self.chunk_size, self.chunk_size * i)
                for i in range(self.chunk_parallel)]

    def download(self):
        downloader = self.service(self.chunk_parallel, self.chunk_size, self.retry, self.timeout)
        downloader.set_basedir(self.basedir)
        with ThreadPoolExecutor(max_workers=self.file_parallel) as executor:
            downloader.set_executor(executor)
            futures = executor.map(downloader, self.targets)
            self.targets = []
            for future in futures:
                while isinstance(future, Future):
                    future = future.result()


class FTPDownloader(BaseDownloader):
    
    def __call__(self, task):
        if task.chunk_size is None:
            return self.single(task)
        else:
            raise NotImplementedError('Chunk download feature hasn\'t been implemented, please be patient!')

    def set_ftp_handler(self, ftp):
        self.ftp = ftp

    def single(self, task):
        try:
            self.ftp.retrbinary(
                'RETR {}'.format(task.url),
                open(os.path.join(self.basedir, task.filename), 'wb').write
            )
        except Exception as err:
            print('Download error', err)


class FTPFastDown:

    default_service = FTPDownloader

    def __init__(self, file_parallel=1, chunk_parallel=1, chunk_size=None, retry=5, timeout=3):
        self.file_parallel = file_parallel
        self.chunk_parallel = chunk_parallel
        self.chunk_size = chunk_size
        if self.chunk_size is None and self.chunk_parallel > 1:
            self.chunk_size = 8192
        self.retry = retry
        self.timeout = timeout
        self.service = self.default_service
        self.basedir = ''

    def use_downloader(self, service):
        self.service = service

    def set_ftp(self, ftp):
        self.ftp = ftp

    def set_basedir(self, basedir):
        self.basedir = basedir

    def set_task(self, task):
        """Task should be arranged like list of (url, filename) tuples."""
        self.targets = []
        if isinstance(task, tuple):
            self.targets.extend(self.create_atom_task(*task))
        elif isinstance(task, str):
            self.targets.extend(self.create_atom_task(task, None))
        else:
            try:
                for t in task:
                    if isinstance(t, tuple):
                        url, filename = t
                    else:
                        url, filename = t, None
                    self.targets.extend(self.create_atom_task(url, filename))
            except TypeError:
                raise ValueError('task must be length-2 tuple or iterable of length-2 tuples.')

    def create_atom_task(self, url, filename):
        if self.chunk_parallel == 1:
            return [AtomTask(url, filename)]
        return [AtomTask(url, filename, self.chunk_size, self.chunk_size * i)
                for i in range(self.chunk_parallel)]

    def download(self):
        downloader = self.service(self.chunk_parallel, self.chunk_size, self.retry, self.timeout)
        downloader.set_basedir(self.basedir)
        downloader.set_ftp_handler(self.ftp)
        with ThreadPoolExecutor(max_workers=self.file_parallel) as executor:
            downloader.set_executor(executor)
            futures = executor.map(downloader, self.targets)
            self.targets = []
            for future in futures:
                while isinstance(future, Future):
                    future = future.result()
