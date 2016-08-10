#Copyright (C) Nial Peters 2014
#
#This file is part of plumetrack.
#
#plumetrack is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#plumetrack is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with plumetrack.  If not, see <http://www.gnu.org/licenses/>.

import os.path
import Queue
import threading
import glob
import itertools

from watcher import create_dir_watcher, can_watch_directories

"""
The dir_iter module provides an iterator class for iterating through the files
in a directory structure. In its most simple form it can be used to list 
the files in a directory (much like os.listdir()). However, it can also be used
to list files in realtime as they are created (using the watcher module).

The following code shows how the dir_iter module can be used to print the names
of any new files created in a directory:

    from plumetrack import dir_iter
    
    for filename in dir_iter.ListDirIter("my_directory", realtime=True):
        print filename
    
"""

def find_files(path, recursive=False, pattern='*', skip_links=True, full_paths=False):
    """
    Returns a list of files in a directory with various filter options applied.
    
        * path - the directory to search for files
        * recursive - boolean controls whether to search sub-directories or not
        * pattern - Unix "glob" type pattern to match for filenames.
        * skip_links - boolean controls whether links are followed or not
        * full_paths - boolean controls whether the filenames returned are full
                       paths or relative paths.
    """
    if not os.path.isdir(path):
        raise ValueError, "\'%s\' is not a recognised folder" %path
    
    found_files = []
        
    for folder, dirs, files in os.walk(path, topdown=True, followlinks=not skip_links):
        
        if full_paths:
            files = [os.path.abspath(os.path.join(folder,f)) for f in files]
        else:
            files = [os.path.join(folder,f) for f in files]
            
        if pattern != '*':
            found_files += glob.fnmatch.filter(files, pattern)
        else:
            found_files += files
                
        if not recursive:
            break
    
    return found_files


class AsyncFileFinder:
    def __init__(self, path, recursive=False, pattern='*', skip_links=True, 
                 full_paths=False):
        """
        Asynchronous version of the find_files function. The AsyncFileFinder is 
        an iterator which starts returning filenames as soon as it has them 
        (rather than waiting until the full file search is completed).
        
        * path - the directory to search for files
        * recursive - boolean controls whether to search sub-directories or not
        * pattern - Unix "glob" type pattern to match for filenames.
        * skip_links - boolean controls whether links are followed or not
        * full_paths - boolean controls whether the filenames returned are full
                       paths or relative paths.
        
        """
        
        self._recursive = recursive
        self._pattern = pattern
        self._follow_links = not skip_links
        self._full_paths = full_paths
        self._path = path
        
        self._output_q = Queue.Queue()
        self._stay_alive = True
        
        self._worker_thread = threading.Thread(target=self._find_files)
        self._worker_thread.start()
    
    
    def _find_files(self):
        
        if self._recursive:
            topdown=False
        else:
            topdown=True
        
        for folder, dirs, files in os.walk(self._path, topdown=topdown, followlinks=self._follow_links):
            
            if self._full_paths:
                files = [os.path.abspath(os.path.join(folder,f)) for f in files]
            else:
                files = [os.path.join(folder,f) for f in files]
            
            if self._pattern != '*':
                files = glob.fnmatch.filter(files, self._pattern)
            
            for f in files:
                self._output_q.put(f)            
                    
            if not self._recursive or not self._stay_alive:
                break
        
        #signal that this is all the files
        self._output_q.put(None)

    
    def close(self):
        """
        Stops the search for files and causes any calls to next() to raise
        StopIteration
        """       
        self._stay_alive = False
        self._worker_thread.join()
    
    
    def __iter__(self):
        """
        Method required by iterator protocol. Allows iterator to be used in 
        for loops.
        """
        return self
    

    def __next__(self):
        #needed for Py3k compatibility
        return self.next()
        
        
    def next(self):
        """
        Method required by iterator protocol. Returns the next filename found
        or raises StopIteration if there are no files left to list (or if 
        close() has been called on the iterator.
        
        Note that if the iterator was created with the realtime option set to 
        True, then this method will block until a new file is detected (or until
        close() is called).
        """
        
        #we do the get() call in a loop so that signals can be recieved by the 
        #thread calling next() even if no items come into the filename queue     
        while True:
            try:
                s = self._output_q.get(block=True,timeout=1)
                break
            except Queue.Empty:
                continue
        
        if s is None or not self._stay_alive:
            
            raise StopIteration
        
        return s
    
    

class ListDirIter(object):
    def __init__(self, directory, realtime=False, skip_existing=False, 
                 recursive=False, sort_func=None, test_func=None, max_n=None,
                 pattern='*', full_paths=False, skip_links=True):
        """
        Iterator class which returns the filenames in a directory structure, with
        the option of monitoring for new files in realtime.
        
            * directory - the directory to list the files in
            * realtime - boolean controls whether to continuously monitor the
                         directory for new files. Use of this option requires 
                         you have either pyinotify or pywin32 installed on your
                         system.
            * skip_existing - boolean controls whether to return the files that
                              already exist in the directory (if set to False, 
                              then only files created after the iterator will be
                              returned).
            * recursive - boolean controls whether to list files in sub-directories
            * sort_func - a comparator function used to sort the list of existing
                          files before they are returned. Default is that they 
                          are sorted by name. Note that once all the existing 
                          filenames have been returned, filenames will be 
                          returned in the order that they are created regardless
                          of what sort_func is set to. If the sort function is 
                          set to None, then filenames will be returned 
                          asynchronously (i.e. the first filenames will be 
                          returned whilst the search is still running) which may 
                          improve performance.
            * test_func - optional test function which should take a filename as
                          its only argument and return True or False. Only filenames
                          which evaluate to True with the test function will
                          be returned from the iterator. Setting this to None 
                          results in all filenames being returned.
            * max_n     - Return at most max_n filenames (regardless of how many
                          files are in the directory). Default is None, return
                          all filenames.
            * pattern   - A UNIX 'glob' style pattern string. Only filenames 
                          matching this pattern will be returned. If full_paths
                          is specified then the pattern matching is done against
                          the absolute paths of the files found.
            * full_paths- boolean controls whether absolute (full) or relative 
                          paths to files are returned.
            * skip_links- boolean specifies whether to follow links or not. 
                          Default is to ignore links.
        """
        
        if not os.path.isdir(directory):
            raise ValueError("Cannot access "+directory+". No such directory.")
        
        if skip_existing and not realtime:
            raise RuntimeError("If skip_existing is set to True then realtime "
                               "must also be set to True (otherwise no files " 
                               "will be returned).")
        
        self.__dir = directory
        self.__sort_func = sort_func
        self.__test_func = test_func
        self.__pattern = pattern
        self.__full_paths = full_paths
        self.__skip_links = skip_links
        self._filename_q = Queue.Queue()
        self.__realtime_filename_q = Queue.Queue()
        self._stay_alive = True
        self.__existing_loader_thread = None
        self._realtime_loader_thread = None
        self.__async_file_iter = None
        self._finished_loading_existing_lock = threading.Lock()
        self._finished_loading_existing_lock.acquire()
        self.__existing_files_found = {}
        self.__recursive = recursive
        self.__max_n = max_n
        self.__return_count = 0
        
        #note that we start the realtime loader thread BEFORE the existing loader thread
        #this is to prevent files that are created during the find_files() call being 
        #skipped.
        if realtime:
            if not can_watch_directories():
                raise RuntimeError("No directory watching implementation available on this system")
            
            self._realtime_loader_thread = threading.Thread(target=self.__load_realtime)
            self._realtime_loader_thread.start()
              
        if not skip_existing:
            if sort_func is None:
                self.__existing_loader_thread = threading.Thread(target=self.__load_existing_async)
            else:
                self.__existing_loader_thread = threading.Thread(target=self.__load_existing)
            self.__existing_loader_thread.start()
        else:
            self._finished_loading_existing_lock.release()

           
    def __iter__(self):
        """
        Method required by iterator protocol. Allows iterator to be used in 
        for loops.
        """
        return self
    

    def __next__(self):
        #needed for Py3k compatibility
        return self.next()
        
        
    def next(self):
        """
        Method required by iterator protocol. Returns the next filename found
        or raises StopIteration if there are no files left to list (or if 
        close() has been called on the iterator.
        
        Note that if the iterator was created with the realtime option set to 
        True, then this method will block until a new file is detected (or until
        close() is called).
        """
        
        #we do the get() call in a loop so that signals can be recieved by the 
        #thread calling next() even if no items come into the filename queue
        self.__return_count += 1
        if self.__max_n is not None and self.__return_count > self.__max_n:
            self.close()
            raise StopIteration
        
        while True:
            try:
                s = self._filename_q.get(block=True,timeout=1)
                break
            except Queue.Empty:
                continue
        
        if s is None or not self._stay_alive:
            
            raise StopIteration
        
        return s

    
    def close(self):
        """
        This is only needed if the iterator was created with the realtime option
        set to True. Calling close() stops the iterator from waiting for new
        files and causes any pending calls to next() to raise StopIteration.
        """
        self._stay_alive = False
        
        if self.__async_file_iter is not None:
            self.__async_file_iter.close()
        
        #the loader threads may be blocking waiting to put something into the queue
        while True:
            try:
                self._filename_q.get(block=False)
            except Queue.Empty:
                break
            
        #the next() method may be blocking waiting to get something from the queue
        try:
            self._filename_q.put(None, block=False)
        except Queue.Full:
            pass
        
        #the realtime loader thread may be blocking waiting to get a new filename to load
        try:
            self.__realtime_filename_q.put(None)
        except Queue.Full:
            pass
                  
        if self.__existing_loader_thread is not None:
            self.__existing_loader_thread.join()
            
        if self._realtime_loader_thread is not None:
            self._realtime_loader_thread.join()
               
    
    def __load_existing_async(self):
        try:
            self.__async_file_iter = AsyncFileFinder(self.__dir, recursive=self.__recursive,
                                     full_paths=self.__full_paths, 
                                     skip_links=self.__skip_links,
                                     pattern=self.__pattern)
            
            for filename in self.__async_file_iter:
                if self.__test_func is not None and not self.__test_func(filename):
                    continue
                
                self.__existing_files_found[filename] = None
                self._filename_q.put(filename)
             
        finally:
            self._finished_loading_existing_lock.release()
            
        if self._realtime_loader_thread is None:        
            #put None into the queue so that the iteration finishes when the q is emptied
            self._filename_q.put(None)    
    
       
    def __load_existing(self):
        try: 
            existing_files = find_files(self.__dir, recursive=self.__recursive,
                                        full_paths=self.__full_paths, 
                                        skip_links=self.__skip_links,
                                        pattern=self.__pattern)
            
            #if a test function was specified, then only keep the filenames which satisfy it
            if self.__test_func is not None:
                existing_files = [x for x in itertools.ifilter(self.__test_func, existing_files)]
             
            #sort the filenames using the comparator function specified in the 
            #kwargs to the constructor of the DirFilesIter object
            existing_files.sort(cmp=self.__sort_func)
            
            #create a dict of all the files found so that we can check for 
            #duplicates when the realtime loader thread starts returning filenames
            for filename in existing_files:
                self.__existing_files_found[filename] = None
                self._filename_q.put(filename)
                       
        finally:
            self._finished_loading_existing_lock.release()
            
        if self._realtime_loader_thread is None:        
            #put None into the queue so that the iteration finishes when the q is emptied
            self._filename_q.put(None)
            
    
    
    def __load_realtime(self):
        
        put_into_q = lambda f,t,q: q.put(f)
        
    
        watcher = create_dir_watcher(self.__dir, self.__recursive,
                                     put_into_q, self.__realtime_filename_q)
        watcher.start()
        
        #wait for the existing files to be found
        self._finished_loading_existing_lock.acquire()
        self._finished_loading_existing_lock.release()
        
        #skip any files that were picked up by the existing loader
        while self._stay_alive:
            filename = self.__realtime_filename_q.get(block=True)
            
            if filename is None:
                break
            
            if self.__full_paths:
                filename = os.path.abspath(filename)
            
            if self.__pattern != '*' and not glob.fnmatch.fnmatch(filename, self.__pattern):
                    continue
            
            if self.__test_func is not None and not self.__test_func(filename):
                continue
            
            if not self.__existing_files_found.has_key(filename):
                break
        
        self.__existing_files_found = {} #done with this dict now - leave it to be GC'd
        
        while self._stay_alive:         
            if filename is None:
                break

            try:
                if self.__test_func is not None and not self.__test_func(filename):
                    continue
                
                if self.__pattern != '*' and not glob.fnmatch.fnmatch(filename, self.__pattern):
                    continue
                
                self._filename_q.put(filename, block=False)

            except Queue.Full:
                #if the output queue is full, then just skip the file - otherwise it might block
                #forever
                print "Warning! Filename output queue is full - skipping file \'"+filename+"\'"
                   
            filename = self.__realtime_filename_q.get(block=True)
        
        watcher.stop()
