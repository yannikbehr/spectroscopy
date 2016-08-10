#Copyright (C) Nial Peters 2015
#
#This file is part of gns_flyspec.
#
#gns_flyspec is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#gns_flyspec is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with gns_flyspec.  If not, see <http://www.gnu.org/licenses/>.
import dir_iter
import watcher
import numpy
import datetime
import threading
import time
import os
import calendar

try:
    class UpdateWatcher(watcher.LinuxDirectoryWatcher):
        """
        Because of complications about how files are written (e.g. sometimes they are
        written to a temporary file and then moved into their final location) to find
        when the wind file is updated, we monitor it's host directory for all close_write
        events and then check to see if they correspond to the file we are interested in.
        
        Unfortunately, there are often multiple write_close events for a single update - 
        so we reload the file multiple times. This isn't much of a problem, it is just
        inefficient.
        """
        def __init__(self, file_name, func, recursive, *args, **kwargs):
            
            watcher.LinuxDirectoryWatcher.__init__(self,  os.path.dirname(file_name), func, recursive, *args, **kwargs)
            self.__file_name = file_name
        
        def process_IN_CREATE(self, event):
            #we're not interested in creation events - only close_write events
            pass
        
        def process_IN_CLOSE_WRITE(self, event):
            #TODO - multiple close_write events are produced for each file update
            #really we should only process one of these.
            filename = os.path.join(event.path, event.name)
            if filename == self.__file_name:
                time.sleep(1)
                self._on_new_file(filename, None) #pass None as creation_time since we don't care about it
except AttributeError:
#then we are probably running on Windows
    class UpdateWatcher(watcher.WindowsDirectoryWatcher):
            """
            Because of complications about how files are written (e.g. sometimes they are
            written to a temporary file and then moved into their final location) to find
            when the wind file is updated, we monitor it's host directory for all close_write
            events and then check to see if they correspond to the file we are interested in.
            
            Unfortunately, there are often multiple write_close events for a single update - 
            so we reload the file multiple times. This isn't much of a problem, it is just
            inefficient.
            """
            def __init__(self, file_name, func, recursive, *args, **kwargs):
                
                watcher.WindowsDirectoryWatcher.__init__(self,  os.path.dirname(file_name), func, recursive, *args, **kwargs)
                self.__file_name = file_name
                
            def _do_watching(self):
                while self.stay_alive:
                    #try:
        
                    for event in self.monitor.read_events():
                        
                        file_path = os.path.join(self.dir_to_watch, event.name)
                    
                        if event.action == 1 or event.action == 3: #file creation event
                            t = datetime.datetime.utcnow()
                            if self.__created_files.has_key(file_path):
                                if self.__created_files[file_path] + datetime.timedelta(seconds=1) > t:
                                    continue
                            
                            self.__created_files[file_path] = calendar.timegm(t.timetuple()) + t.microsecond*1e-6
                            self.__new_files_q.put(file_path)

            
class WindData:
    def __init__(self, config, realtime, day_to_process=None):
        self._stay_alive = True
        self.__use_data_lock = threading.Lock()
        self.data_path = config["wind_data_folder"]
        self.realtime = realtime
        self.times = numpy.array([])
        self.directions = numpy.array([])
        self.speeds = numpy.array([])
        self._worker_thread = None
        self._file_update_watcher = None
        
        if self.realtime:
            self.__iterator = dir_iter.ListDirIter(self.data_path,recursive=True,
                                                   realtime=True, sort_func=cmp,
                                                   pattern="*.txt")
            
        
        elif day_to_process is not None:
            print "Looking for wind file: %s"%day_to_process.strftime("*%Y_%m_%d.txt")
            self.__iterator = dir_iter.ListDirIter(self.data_path,recursive=True,
                                                   realtime=False, sort_func=cmp,
                                                   pattern=day_to_process.strftime("*%Y_%m_%d.txt"))
        
        else:
            raise NotImplementedError("Non-realtime wind data loading is not implemented yet!")
    
        self._worker_thread = threading.Thread(target=self.__load_winddata)
        self._worker_thread.start()
        
        
    def close(self):
        self._stay_alive = False
        self.__iterator.close()
        if self._file_update_watcher is not None:
            self._file_update_watcher.stop()
        
        if self._worker_thread is not None:
            self._worker_thread.join()
    
    
    def __load_file(self, filename):
        with open(filename,"r") as ifp:
            times = []
            directions = []
            speeds = []
            
            for line in ifp:
                if line == "" or line.isspace():
                    continue
                words = line.split()
                
                times.append(datetime.datetime.strptime(words[0],"%Y-%m-%dT%H:%M:%SZ"))
                directions.append(float(words[1]))
                speeds.append(float(words[2]))
        
        return numpy.array(times), numpy.array(directions), numpy.array(speeds)
                
    
    def __load_updated_file(self, filename, dummy_param):
        """
        Loads the file and appends any new data in it to self.times, self.speeds etc.
        This is called automagically from the UpdateWatcher.
        """
        times, directions, speeds = self.__load_file(filename)
        with self.__use_data_lock:
            overlap_idx = numpy.where(times == self.times[-1])

            assert len(overlap_idx[0])>=1
            overlap_idx = overlap_idx[0][-1]
            #crop the data from the file to only contain new data
            times = times[overlap_idx + 1:]
            directions = directions[overlap_idx + 1:]
            speeds = speeds[overlap_idx + 1:]

            self.times = numpy.concatenate((self.times,times))
            self.directions = numpy.concatenate((self.directions,directions))
            self.speeds = numpy.concatenate((self.speeds, speeds))

    
    def __load_winddata(self):
        """
        New wind files are produced daily - but then the files are updated every
        ten minutes. This function therefore has to check both for new files and
        updated files.
        """
        for wind_file in self.__iterator:
            print "Loading wind file: %s"%wind_file
            #stop watching the old file for changes
            if self._file_update_watcher is not None:
                self._file_update_watcher.stop()
            
            new_times, new_directions, new_speeds = self.__load_file(wind_file)
            
            #crop the old data to be <= day in length
            with self.__use_data_lock: #make sure another thread is not using the data
                t_now = datetime.datetime.now()
                one_day = datetime.timedelta(days=1)
                mask_idxs = numpy.where(self.times < t_now - one_day)
                self.times = numpy.concatenate((self.times[mask_idxs],new_times))
                self.directions = numpy.concatenate((self.directions[mask_idxs],new_directions))
                self.speeds = numpy.concatenate((self.speeds[mask_idxs], new_speeds))
            
            #start watching the new file for changes
            self._file_update_watcher = UpdateWatcher(wind_file, self.__load_updated_file ,False)
            self._file_update_watcher.start()
    
    
    def __block_until_data_ready(self,t):
        """
        Blocks until data for the specified time exists
        """
        with self.__use_data_lock:
            if len(self.times)>0:
                assert t > self.times[0], "The scan data predates the available wind data"
        
        while (self._stay_alive):
            with self.__use_data_lock:
                if len(self.times)>0 and t <= self.times[-1]:
                    break
            time.sleep(1)
    
    
    def get_direction_and_speed(self, t):
        """
        Returns the wind speed and direction at time t (datetime object).
        
        Note that this method does not return until the wind data from the 
        requested time becomes available.
        """            
        self.__block_until_data_ready(t) #wait for data to arrive
        if not self._stay_alive:
            return None
        
        with self.__use_data_lock:
            closest_idx = numpy.argmin(numpy.abs(self.times-t))
            return self.directions[closest_idx], self.speeds[closest_idx]
            